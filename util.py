from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
import torch
from torch.fx import Graph, Node

from colossalai.fx.profiler import (calculate_fwd_out, calculate_fwd_tmp, calculate_fwd_in, is_compatible_with_meta,
                                    parameter_size)

from offload_strategy import OffloadStrategiesVector


@dataclass
class Region:
    r_id: int = 0
    is_offload: bool = False
    nodes: List[Node] = None
    param_indices: List[int] = None
    param_size: int = 0
    # out_node: Node = None
    region_to_prefetch = None
    is_syn: bool = False
    region_shared_param = None


class ModelParameters:
    param_idx = 0
    fp16_params = []
    fp32_master_params = []
    # param_offload_dict = {}


class GlobalCudaInfo:
    h2d_stream = torch.cuda.Stream()
    prefetch_event_map = {}


@dataclass
class NodeInfo:
    node_id: int = 0
    param_size: float = 0
    offload_param_flag: bool = False
    param_indices: List = None
    runtime_fwd_mem: float = 0
    runtime_bwd_mem: float = 0
    offload_strategies_vector: OffloadStrategiesVector = None
    # asyn
    node_to_prefetch: Node = None
    syn_upload_flag: bool = False
    prefetch_end_timestamp: float = 0


class ExeType(Enum):
    Syn2Syn = 0
    Asyn2Asyn = 1
    Asyn2Syn = 2


def compute_act_peak_mem(region_list: List[Region]) -> float:
    act_peak_mem = 0
    runtime_mem = 0

    # forward
    for region in region_list:
        for node in region.nodes:

            if node.name.__contains__("transpose") and node.meta['fwd_out'][0].dim() <= 2:
                continue

            runtime_mem = runtime_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            # if (runtime_mem - act_peak_mem) / 1024 ** 2 > 1 or node.name.__contains__("transpose"):
            #     print(f"n_name: {node.name} | fwd_mem_tmp={calculate_fwd_tmp(node) / 1024 ** 2:.3f} MB | "
            #           f"fwd_mem_out={calculate_fwd_out(node) / 1024 ** 2:.3f} MB | "
            #           f"bwd_mem_tmp={node.meta['bwd_mem_tmp'] / 1024 ** 2:.3f} MB | "
            #           f"bwd_mem_out={node.meta['bwd_mem_out'] / 1024 ** 2:.3f} MB | "
            #           f"fwd_out={node.meta['fwd_out']}")

            act_peak_mem = max(runtime_mem, act_peak_mem)
    # print("forward peak memory size:", act_peak_mem / 1024 ** 2, "MB")

    # backward
    grad_in_computed = {}
    for region in region_list.__reversed__():
        for node in region.nodes.__reversed__():
            runtime_mem -= calculate_fwd_out(node)
            runtime_mem = runtime_mem + node.meta['bwd_mem_tmp'] + node.meta['bwd_mem_out']

            act_peak_mem = max(runtime_mem, act_peak_mem)
            if runtime_mem > act_peak_mem:
                print(node.name, "backward runtime memory size:", runtime_mem / 1024 ** 2, "MB")

            runtime_mem = runtime_mem - node.meta['bwd_mem_tmp'] - calculate_fwd_tmp(node)

            # TODO 需要考虑有多个user node 的情况，当前只释放了一个bwd_out
            # release grad_in of current node
            for grad_in in node.meta["fwd_out"]:
                if isinstance(grad_in, torch.Tensor):
                    runtime_mem -= grad_in.numel() * grad_in.element_size()

            for in_node in list(node._input_nodes.keys()):
                # # release fwd_in (fwd_out) of current node (input nodes)
                # if calculate_fwd_out(in_node) > 0 and (not fwd_out_released[in_node]):
                #     runtime_mem -= calculate_fwd_out(in_node)
                #     fwd_out_released[in_node] = True

                # map multiple gradients of output to one tensor
                if grad_in_computed.get(in_node, False):
                    runtime_mem -= calculate_fwd_out(in_node)
                    grad_in_computed[in_node] = True
    return act_peak_mem


def compute_max_param_mem(region_list: List[Region]) -> float:
    return max(region.param_size for region in region_list)


def compute_total_param_mem(region_list: List[Region]) -> float:
    return sum(region.param_size for region in region_list if requires_upload_p_in_fwd(region))


def requires_upload_p_in_fwd(region: Region):
    return region.param_size > 0 and (
            region.region_shared_param is None or region.r_id < region.region_shared_param.r_id or (
            region.r_id > region.region_shared_param.r_id and region.region_shared_param.is_offload))


def requires_offload_g_in_bwd(region: Region):
    return region.region_shared_param is None or region.r_id < region.region_shared_param.r_id


def requires_release_p_in_bwd(region: Region):
    return region.region_shared_param is None or region.r_id < region.region_shared_param.r_id or (
            region.r_id > region.region_shared_param.r_id and region.region_shared_param.is_offload)


def is_first_shared_region(region: Region) -> bool:
    return region.region_shared_param is not None and region.r_id < region.region_shared_param.r_id


def is_last_shared_region(region: Region) -> bool:
    return region.region_shared_param is not None and region.r_id > region.region_shared_param.r_id
