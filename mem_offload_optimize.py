from typing import List, Dict
import torch
import torch.fx
from torch.fx import GraphModule

from torch.utils._pytree import tree_map

from colossalai.fx import ColoTracer, is_compatible_with_meta
from colossalai.fx.passes.meta_info_prop import MetaInfoProp

from strategies_constructor import OffloadStrategiesConstructor
from solver import AsynGreedySolver, SynGreedySolver
from runtime_syn import runtime_syn_offload_apply_pass
from runtime_asyn import runtime_asyn_offload_apply_pass
from basic_offload_module import BasicOffloadModule, AMPOptimizer
from util import compute_max_param_mem, compute_total_param_mem, compute_act_peak_mem, ExeType, OffloadManager


def memory_optimization(model: torch.nn.Module,
                        inps: Dict[str, torch.Tensor],
                        memory_budget: float = -1.0,
                        exe_type: int = 1):
    model = model.cpu().half()
    tracer = ColoTracer()
    assert is_compatible_with_meta()
    wrap_fn = lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x
    meta_args = tree_map(wrap_fn, inps)
    graph = tracer.trace(model, meta_args=meta_args)
    # graph.print_tabular()
    gm = GraphModule(model, graph, model.__class__.__name__)

    interp = MetaInfoProp(gm)
    interp.propagate(*meta_args.values())

    offload_strategies_constructor = OffloadStrategiesConstructor(graph)
    region_list = offload_strategies_constructor._linearize_graph()

    OffloadManager.region_list = region_list

    act_peak_mem = compute_act_peak_mem(region_list) / 1024 ** 2
    max_param_mem = compute_max_param_mem(region_list) / 1024 ** 2
    total_param_mem = compute_total_param_mem(region_list) / 1024 ** 2
    print(
        f"act_peak_mem={act_peak_mem:.3f} MB | max_param_mem={max_param_mem:.3f} MB | total_param_mem={total_param_mem:.3f}")

    if exe_type == ExeType.Syn2Syn.value:
        solver = SynGreedySolver(region_list, memory_budget)
        solver._call_solver_greedy()
        gm = runtime_syn_offload_apply_pass(gm, region_list)
    elif exe_type == ExeType.Asyn2Asyn.value:
        solver = AsynGreedySolver(region_list, memory_budget)
        solver._call_solver_greedy()
        gm = runtime_asyn_offload_apply_pass(gm, region_list)
    elif exe_type == ExeType.Asyn2Syn.value:
        solver = AsynGreedySolver(region_list, memory_budget)
        solver._call_solver_greedy()
        gm = runtime_syn_offload_apply_pass(gm, region_list)
    else:
        raise RuntimeError(f"exe_type only in [0, 1, 2], but get {exe_type}!")

    # print offload region
    print("****************** offload plan *******************")
    for region in region_list:
        if region.is_offload or (region.bwd_prefetch_region is not None):
            print(region.r_id, region.bwd_prefetch_region.r_id if region.bwd_prefetch_region is not None else None,
                  region.is_offload)

    gm.recompile()
    # print(gm.code)
    optimized_model = BasicOffloadModule(gm, exe_type == 0)
    return optimized_model
