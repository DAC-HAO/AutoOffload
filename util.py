from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
import torch
from torch.fx import Graph, Node

from colossalai.fx.profiler import (calculate_fwd_out, calculate_fwd_tmp, calculate_fwd_in, is_compatible_with_meta,
                                    parameter_size)


@dataclass
class Region:
    r_id: int = 0
    is_offload: bool = False
    nodes: List[Node] = None
    param_size: int = 0
    is_syn: bool = False
    region_shared_param = None
    fwd_prefetch_region = None
    bwd_prefetch_region = None
    fp16_params: List[torch.nn.Parameter] = None
    shared_rid: int = 0


class OffloadManager:
    h2d_stream = torch.cuda.Stream()
    d2h_stream = torch.cuda.Stream()
    fwd_prefetch_event_map = {}
    bwd_prefetch_event_map = {}
    region_list = []
    param_fp16_to_fp32 = {}
    param_fp16_to_grad = {}


@dataclass
class NodeInfo:
    node_id: int = 0
    param_size: float = 0
    runtime_fwd_mem: float = 0
    runtime_bwd_mem: float = 0


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

            runtime_mem = runtime_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)

            # if (runtime_mem - act_peak_mem) / 1024 ** 2 > 1 or node.name.__contains__("transpose"):
            #     print(f"n_name: {node.name} | fwd_mem_tmp={calculate_fwd_tmp(node) / 1024 ** 2:.3f} MB | "
            #           f"fwd_mem_out={calculate_fwd_out(node) / 1024 ** 2:.3f} MB | "
            #           f"bwd_mem_tmp={node.meta['bwd_mem_tmp'] / 1024 ** 2:.3f} MB | "
            #           f"bwd_mem_out={node.meta['bwd_mem_out'] / 1024 ** 2:.3f} MB | "
            #           f"fwd_out={node.meta['fwd_out']}")

            act_peak_mem = max(runtime_mem, act_peak_mem)
    print("forward peak memory size:", act_peak_mem / 1024 ** 2, "MB")

    # backward
    bwd_deps = {}
    for region in region_list.__reversed__():
        for node in region.nodes.__reversed__():
            runtime_mem -= calculate_fwd_out(node)
            runtime_mem = runtime_mem + node.meta['bwd_mem_tmp'] + node.meta['bwd_mem_out']

            if runtime_mem > act_peak_mem:
                print(node.name, "backward runtime memory size:", runtime_mem / 1024 ** 2, "MB\t param size",
                      node.node_info.param_size / 1024 ** 2, "MB")
            act_peak_mem = max(runtime_mem, act_peak_mem)

            runtime_mem = runtime_mem - node.meta['bwd_mem_tmp'] - calculate_fwd_tmp(node)

            # free bwd_mem_out
            bwd_deps[node] = len(node.all_input_nodes)
            for user_node in node.users:
                if user_node in bwd_deps:
                    bwd_deps[user_node] -= 1
                    if bwd_deps[user_node] <= 0:
                        runtime_mem -= user_node.meta['bwd_mem_out']

    return act_peak_mem


def compute_max_param_mem(region_list: List[Region]) -> float:
    return max(region.param_size for region in region_list)


def compute_total_param_mem(region_list: List[Region]) -> float:
    return sum(region.param_size for region in region_list if region.r_id <= region.shared_rid)


def requires_upload_p_in_fwd(region: Region):
    return region.param_size and (
            region.region_shared_param is None or region.r_id < region.region_shared_param.r_id or (
            region.r_id > region.region_shared_param.r_id and region.region_shared_param.is_offload))


def requires_offload_g_in_bwd(region: Region):
    return region.region_shared_param is None or region.r_id < region.region_shared_param.r_id


def requires_release_p_in_bwd(region: Region):
    return region.region_shared_param is None or region.r_id < region.region_shared_param.r_id or (
            region.r_id > region.region_shared_param.r_id and region.region_shared_param.is_offload)

