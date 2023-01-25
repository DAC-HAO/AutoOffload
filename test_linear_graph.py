import argparse
from typing import List, Dict
import torch
import torch.fx
from torch.fx import GraphModule

from torch.utils._pytree import tree_map

from colossalai.fx import ColoTracer, is_compatible_with_meta
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from strategies_constructor import OffloadStrategiesConstructor
from model_utils import *


def partition_graph(model: torch.nn.Module, inps: Dict[str, torch.Tensor]):
    model.cpu()
    tracer = ColoTracer()
    assert is_compatible_with_meta()
    wrap_fn = lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x
    meta_args = tree_map(wrap_fn, inps)
    graph = tracer.trace(model, meta_args=meta_args)
    graph.print_tabular()
    gm = GraphModule(model, graph, model.__class__.__name__)

    interp = MetaInfoProp(gm)
    interp.propagate(*meta_args.values())

    offload_strategies_constructor = OffloadStrategiesConstructor(graph)
    region_list = offload_strategies_constructor._linearize_graph()

    for region in region_list:
        print("*****************************************************", region.r_id, region_list.index(region))
        for node in region.nodes:
            print(node.op, "\t", node.name, "\t", node.target, "\t", node.args)


parser = argparse.ArgumentParser(description="offload testing")
parser.add_argument("-m_name", type=str, default="simplenet",
                    help="model name")
parser.add_argument("-mem_size", type=float, default=32, help="memory budget (MB)")
parser.add_argument('-is_syn', action='store_true', help='If true, offload is performed synchronously.')
args = parser.parse_args()

# build model
get_components_func = non_distributed_component_funcs.get_callable(args.m_name)
model_builder, data_gen = get_components_func()
data_args = data_gen(device="cpu")
model = model_builder()

partition_graph(model, data_args)