from functools import partial
import time

import pytest
import torch
import torch.multiprocessing as mp

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.gemini.chunk import ChunkManager, search_chunk_configuration
from colossalai.gemini.gemini_mgr import GeminiManager
from colossalai.nn.parallel import zero_model_wrapper, zero_optim_wrapper
from colossalai.testing import parameterize
from colossalai.utils import free_port
from colossalai.utils.cuda import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.fx.profiler import parameter_size
from tests.test_tensor.common_utils import set_seed

from model_utils import *


@parameterize('init_device', [get_current_device()])
@parameterize('placement_policy', ['cpu'])
@parameterize('keep_gather', [False])
@parameterize('model_name', ['mlp'])
@parameterize('use_grad_checkpoint', [False])
def exam_gpt_fwd_bwd(placement_policy,
                     keep_gather,
                     model_name: str,
                     use_grad_checkpoint: bool = False,
                     init_device=get_current_device()):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()

    torch.cuda.set_per_process_memory_fraction(0.03, 0)
    torch.cuda.empty_cache()

    start_time = time.time()
    set_seed(42)
    with ColoInitContext(device=torch.device('cpu')):
        model = model_builder(use_grad_checkpoint)
    optimizer = HybridAdam(model.parameters(), lr=1e-3)

    param_size = parameter_size(model) / 1024 ** 2
    init_time = time.time() - start_time
    print(f"init_param_size={param_size:.3f} MB | init_model_time={init_time:.3f} s")
    data_args = data_gen(device=init_device)

    gemini_config = dict(strict_ddp_mode=False,
                         device=torch.device('cpu'),
                         placement_policy=placement_policy,
                         pin_memory=True,
                         hidden_dim=8192,
                         search_range_mb=128)
    model = zero_model_wrapper(model, 3, gemini_config)
    optim_config = dict(reduce_bucket_size=12 * 1024 * 1024, overlap_communication=True, verbose=True)

    optimizer = zero_optim_wrapper(model, optimizer, optim_config=optim_config)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    time_list = []

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/gemini_' + model_name),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for step in range(10):
            set_seed(42)
            data_args = data_gen(device=init_device)
            optimizer.zero_grad()
            torch.cuda.synchronize()
            start_time = time.time()
            loss = torch.mean(model(**data_args))
            optimizer.backward(loss)
            torch.cuda.synchronize()
            time_list.append(time.time() - start_time)
            optimizer.step()
            prof.step()

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
    exam_gpt_fwd_bwd()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1])
def test_gpt(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_gpt(1)
