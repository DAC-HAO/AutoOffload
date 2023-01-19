from typing import List

import torch
from torch.fx import Graph, Node

from offload_strategy import OffloadStrategy, OffloadStrategiesVector, SystemConfig

class StrategyGenerator:
    """
    StrategyGenerator is used to generate the offload strategies.
    """

    def __init__(self, node: Node, graph: Graph):
        self.node = node
        self.graph = graph
        self.nodes = list(self.graph.nodes)
        self.node_idx = self.nodes.index(node)

    def collate_strategies(self) -> List[OffloadStrategy]:

        # TODO currently have only one strategy

        strategies = []
        # strategies.append(OffloadStrategy(False))

        # param_size = self.compute_param_size(self.node)
        # comm_cost = param_size / SystemConfig.BANDWIDTH

        comm_cost = self.node.node_info.param_size / SystemConfig.BANDWIDTH
        strategies.append(OffloadStrategy(offload_flag=True, comm_cost=comm_cost))

        return strategies

    def compute_param_size(self, node: Node):
        assert node.op in ['call_function', 'call_module']
        param_size = 0
        if node.op == 'call_function':
            for inp_node in list(node._input_nodes.keys()):
                if inp_node.op == "get_attr":
                    attr_itr = self.graph.owning_module
                    atoms = inp_node.target.split(".")
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    if isinstance(attr_itr, torch.nn.parameter.Parameter):
                        param_size += attr_itr.data.numel() * attr_itr.data.element_size()
                    elif isinstance(attr_itr, torch.Tensor):
                        param_size += attr_itr.numel() * attr_itr.element_size()

        elif node.op == 'call_module':
            target = node.target
            submod = self.graph.owning_module.get_submodule(target)
            for p in submod.parameters():
                param_size += p.data.numel() * p.data.element_size()

        return param_size

    def update_reuse_interval(self, strategy: OffloadStrategy):
        reuse_interval = 0
        for following_node in self.nodes[self.node_idx:]:
            reuse_interval += following_node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER
            reuse_interval += following_node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER
            if hasattr(following_node, 'node_info') and following_node.node_info.offload_param_flag:
                # TODO will extrate it from node.strategy
                # reuse_interval += self.compute_param_size(following_node) / SystemConfig.BANDWIDTH
                reuse_interval += following_node.node_info.param_size / SystemConfig.BANDWIDTH

        strategy.reuse_interval = reuse_interval

    def update_strategies(self, strategies: List[OffloadStrategy]):
        for strategy in strategies:
            self.update_reuse_interval(strategy)

    def generate(self) -> List[OffloadStrategy]:
        """
        Generate all possible sharding strategies for this operation.
        """
        strategies = self.collate_strategies()

        # update the costs
        for strategy in strategies:
            self.update_reuse_interval(strategy)
        return strategies

