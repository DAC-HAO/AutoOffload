import argparse
import time
import pytest
from functools import partial

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils._pytree import tree_map
from torch.testing import assert_close

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.amp import convert_to_apex_amp
from colossalai.fx.profiler import parameter_size
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import free_port
from colossalai.nn.parallel import zero_model_wrapper, zero_optim_wrapper

from basic_offload_module import AMPOptimizer
from mem_offload_optimize import memory_optimization
from model_utils import *
from util import OffloadManager

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def check_grad(model: torch.nn.Module, torch_model: torch.nn.Module):
    for (p0, p1) in zip(model.parameters(), torch_model.parameters()):
        assert_close(OffloadManager.param_fp16_to_fp32[p0], p1.grad.cpu(), rtol=1e-3, atol=5e-5)


def exam_fwd_bwd():
    parser = argparse.ArgumentParser(description="offload testing")
    parser.add_argument("-mn", type=str, default="simplenet",
                        help="model name")
    parser.add_argument("-ms", type=float, default=32, help="memory budget (MB)")
    parser.add_argument('-et', type=int, default=1, choices=[0, 1, 2], help='execution type')
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.3, 0)
    torch.cuda.empty_cache()

    start_time = time.time()

    # build model
    get_components_func = non_distributed_component_funcs.get_callable(args.mn)
    model_builder, data_gen = get_components_func()
    data_args = data_gen(device="cpu")
    wrap_fn = lambda x: x.to(dtype=torch.half) if isinstance(x, torch.Tensor) and torch.is_floating_point(x) else x
    data_args = tree_map(wrap_fn, data_args)
    model = model_builder()
    model.train()

    torch_model = model_builder().cuda()
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p.data)
        # print(torch_p.data)
    torch_model.train()

    with ColoInitContext(device='cpu'):
        gemini_model = model_builder()
    for gemini_p, p in zip(gemini_model.parameters(), model.parameters()):
        gemini_p.data.copy_(p.data)
    gemini_model.train()

    param_size = parameter_size(model) / 1024 ** 2 / 2
    init_time = time.time() - start_time
    print(f"init_param_size={param_size:.3f} MB | init_model_time={init_time:.3f} s")

    start_time = time.time()
    model = memory_optimization(model, data_args, 1024 * 1024 * args.ms, args.et)
    solver_time = time.time() - start_time
    print(f"linearize_solver_time={solver_time:.3f} s")

    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)
    optim = AMPOptimizer(optimizer, initial_scale=1)

    amp_config = dict(opt_level='O2', keep_batchnorm_fp32=False, loss_scale=1)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)

    hybrid_optimizer = HybridAdam(gemini_model.parameters(), lr=1e-3)
    gemini_config = dict(strict_ddp_mode=False,
                         device=torch.device('cpu'),
                         placement_policy='cpu',
                         pin_memory=True,
                         hidden_dim=8192,
                         search_range_mb=128)
    gemini_model = zero_model_wrapper(gemini_model, 3, gemini_config)
    optim_config = dict(reduce_bucket_size=12 * 1024 * 1024, overlap_communication=True, verbose=True)
    gemini_optim = zero_optim_wrapper(gemini_model, hybrid_optimizer, optim_config=optim_config)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    time_list = []

    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=2),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/auto_asyn_' + args.mn),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    # ) as prof:

    for step in range(10):
        data_args = data_gen(device="cuda")
        data_args = tree_map(wrap_fn, data_args)
        torch.cuda.synchronize()

        torch_optim.zero_grad()
        gemini_optim.zero_grad()
        optim.zero_grad()

        torch_loss = torch.mean(torch_model(**data_args))
        gemini_loss = torch.mean(gemini_model(**data_args))

        # start_time = time.time()
        loss = torch.mean(model(**data_args))
        # model.backward(loss)
        # torch.cuda.synchronize()
        # time_list.append(time.time() - start_time)

        for torch_p, p in zip(torch_model.parameters(), model.model.parameters()):
            print(torch_p.data.shape)
            try:
                assert torch.equal(torch_p.data.half(), p.data)
            except:
                print(torch_p.data.half(), p.data)
            # assert_close(torch_p.data.half(), p.data, rtol=1e-3, atol=5e-5)

        print(torch_loss, loss)
        # assert torch.equal(torch_loss, loss)
        print(gemini_loss.data, loss)
        assert torch.equal(gemini_loss, loss)

        # torch_optim.backward(torch_loss)
        gemini_optim.backward(gemini_loss)
        check_grad(model, gemini_model)

        # torch_optim.step()
        gemini_optim.step()
        optim.step()
        # prof.step()

    torch.cuda.synchronize()

    exec_time = sum(sorted(time_list)[:5]) / 5

    runtime_peak_mem_alc = torch.cuda.max_memory_allocated() / 1024 ** 2
    runtime_peak_mem_res = torch.cuda.max_memory_reserved() / 1024 ** 2
    print(
        f'|exec_time={exec_time:.3f} s | param_size={param_size:.3f} MB '
        f'| runtime_peak_mem_alc={runtime_peak_mem_alc:.3f} MB| runtime_peak_mem_res={runtime_peak_mem_res:.3f} MB|'
    )

    print(time_list)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_fwd_bwd()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1])
def test_offload(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_offload(1)
