from dataclasses import dataclass
from typing import List, Dict
import torch
from torch.fx import Graph, Node

from offload_strategy import OffloadStrategiesVector


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
    has_param: bool = False
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
