from typing import List, Any
import torch
from torch.fx import Graph, Node

from offload_strategy import OffloadStrategiesVector
from strategy_generator import StrategyGenerator
from util import ModelParameters, NodeInfo, Region

class OffloadStrategiesConstructor:
    """
    OffloadStrategiesConstructor is used to construct the offload plan for the model execution.

    Args:
        graph (Graph): a Graph object used for analysis and strategy generation.
        solver_option (SolverOption): a SolverOptions object which specifies the preferences for plan searching.
    """

    def __init__(self, graph: Graph, cnode: List[str] = None):
        self.graph = graph
        assert graph.owning_module is not None, 'The given graph is not associated with a owning_module'
        self.root_module = self.graph.owning_module
        self.nodes = list(graph.nodes)
        self.leaf_strategies = []
        self.strategy_map = {}
        self.no_strategy_nodes = []
        self.cnode = cnode
        self.param_ops = []


    def _linearize_graph(self) -> List[Region]:
        """Linearizing the graph

        Args:
            graph (Graph): The computing graph to be optimized.

        Returns:
            List[List[Node]]: List of list, each inside list of Node presents
            the actual 'node' in linearized manner.

        Remarks:
            Do merge the inplace ops and shape-consistency ops into the previous node.
        """

        # Common nodes are type of nodes that could be seen as attributes and remain
        # unchanged throughout the whole model, it will be used several times by
        # different blocks of model, so that it is hard for us to linearize the graph
        # when we encounter those kinds of nodes. We let users to annotate some of the
        # input as common node, such as attention mask, and the followings are some of
        # the ops that could actually be seen as common nodes. With our common node prop,
        # we could find some of the "real" common nodes (e.g. the real attention mask
        # used in BERT and GPT), the rule is simple, for node who's parents are all common
        # nodes or it's op belongs to the following operations, we view this node as a
        # newly born common node.
        # List of target name that could be seen as common node
        common_ops = ["getattr", "getitem", "size"]

        def _is_cop(target: Any) -> bool:
            """Check if an op could be seen as common node

            Args:
                target (Any): node target

            Returns:
                bool
            """

            if isinstance(target, str):
                return target in common_ops
            else:
                return target.__name__ in common_ops

        def _is_param_comp_op() -> bool:
            """Check if an op could be seen as compute node contained params

            Args:
                n (Node): node

            Returns:
                bool
            """

            if n.op == "call_module":
                target = n.target
                submod = self.root_module.get_submodule(target)
                if (
                        len(list(submod.named_parameters(recurse=False))) != 0
                        or len(list(submod.named_buffers(recurse=False))) != 0
                ):
                    return True

            elif n.op == "call_function":
                return any(map(lambda x: x.name in self.param_ops, n.all_input_nodes)) and any(
                    map(lambda x: x.name not in self.param_ops and not _is_cop(n.target),
                        n.all_input_nodes)) and not sum([v for _, v in param_op_deps.items()])

            return False

        def _is_sink() -> bool:
            """Check if we can free all dependencies

            Returns:
                bool
            """

            def _is_inplace(n: Node):
                """Get the inplace argument from ``torch.fx.Node``
                """
                inplace = False
                if n.op == "call_function":
                    inplace = n.kwargs.get("inplace", False)
                elif n.op == "call_module":
                    inplace = getattr(n.graph.owning_module.get_submodule(n.target), "inplace", False)
                return inplace

            return not sum([v for _, v in deps.items()]) and not any(map(_is_inplace, n.users))


        # make sure that item in cnode is valid
        if self.cnode:
            for name in self.cnode:
                try:
                    assert next(node for node in self.graph.nodes if node.name == name).op == "placeholder", \
                        f"Common node {name} is not an input of the model."
                except StopIteration:
                    raise ValueError(f"Common node name {name} not in graph.")

        else:
            self.cnode = []

        node_id = 0
        region_id = 0

        param_op_deps = {}

        deps = {}
        region_list = []
        region = Region(r_id=region_id, nodes=[], param_indices=[])

        for n in self.graph.nodes:
            if n.op != "placeholder" and n.op != "output":
                for n_par in n.all_input_nodes:
                    if n_par.op != "placeholder" and n_par.name not in self.cnode:
                        deps[n_par] -= 1
                    if n_par.op != "placeholder" and n_par.name in self.param_ops:
                        param_op_deps[n_par] -= 1

                region.nodes.append(n)
                self._set_node_and_region_info(node_id, n, region)

                # if the node could free all dependencies in graph
                # we could begin a new node
                if _is_sink() or _is_param_comp_op():
                    region_list.append(region)
                    region = Region(r_id=region_id, nodes=[], param_indices=[])
                    region_id += 1

                # propagate common node attr if possible
                if len(n.all_input_nodes) == len([node for node in n.all_input_nodes if node.name in self.cnode
                                                  ]) or _is_cop(n.target):
                    self.cnode.append(n.name)
                else:
                    deps[n] = len([user for user in n.users if user.op != "output"])

                # propagate common node attr if possible
                if len(n.all_input_nodes) == len([node for node in n.all_input_nodes if node.name in self.param_ops
                                                  ]) or n.op == "get_attr":
                    self.param_ops.append(n.name)
                    param_op_deps[n] = len([user for user in n.users if user.op != "output"])

        return region_list


    def _set_node_and_region_info(self, node_id: int, cur_n: Node, cur_reg: Region):

        node_info = NodeInfo(node_id)
        node_info.param_indices = []

        if cur_n.op == 'call_module':
            target = cur_n.target
            submod = self.root_module.get_submodule(target)
            for p in list(submod.parameters(recurse=False)):
                node_info.param_indices.append(ModelParameters.param_idx)
                node_info.param_size += p.data.numel() * p.data.element_size()
                ModelParameters.fp16_params.append(p)
                ModelParameters.fp32_master_params.append(p.detach().clone().float().pin_memory())
                ModelParameters.param_idx += 1

        elif cur_n.op == "get_attr":
            attr_itr = self.root_module
            atoms = cur_n.target.split(".")
            for atom in atoms:
                attr_itr = getattr(attr_itr, atom)

            if isinstance(attr_itr, torch.nn.Parameter):
                node_info.param_indices.append(ModelParameters.param_idx)
                node_info.param_size += attr_itr.data.numel() * attr_itr.data.element_size()
                ModelParameters.fp16_params.append(attr_itr)
                ModelParameters.fp32_master_params.append(attr_itr.detach().clone().float().pin_memory())
                ModelParameters.param_idx += 1

        # get_attr 的参数应该下沉

        cur_n.node_info = node_info
        cur_reg.param_size += node_info.param_size
        cur_reg.param_indices.extend(node_info.param_indices)