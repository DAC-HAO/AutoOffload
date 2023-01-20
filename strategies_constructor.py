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
                print(n.op, n.name, inplace)
                return inplace

            return (not sum([v for _, v in deps.items()]) or (not sum([v for _, v in deps_in_region.items()])) and deps_in_region.__len__()>0) and not any(map(_is_inplace, n.users))

        def _set_region_info(cur_reg: Region):
            pass

        def _set_param_info_for_node(cur_n: Node):
            if cur_n.op in ('placeholder', 'get_attr', 'call_method', 'output'):
                label = True

            elif cur_n.op == "call_module":
                target = cur_n.target
                submod = self.root_module.get_submodule(target)
                if (
                        len(list(submod.named_parameters(recurse=False))) == 0
                        and len(list(submod.named_buffers(recurse=False))) == 0
                ):
                    label = True

            elif cur_n.op == "call_function":
                label = True
                input_nodes = list(cur_n._input_nodes.keys())
                for inp_node in input_nodes:
                    if (inp_node.op == "get_attr"):
                        label = False
                        break

                if len(input_nodes) == 1:
                    unique_inp_node = input_nodes[0]
                    if (unique_inp_node.op == "get_attr"):
                        label = False



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

        deps = {}
        deps_in_region = {}
        region_list = []
        region = Region(has_param=False, nodes=[])

        for n in self.graph.nodes:
            has_param = False
            if n.op != "placeholder" and n.op != "output":
                for n_par in n.all_input_nodes:
                    if n_par.op != "placeholder" and n_par.name not in self.cnode:
                        deps[n_par] -= 1
                        if n_par in deps_in_region:
                            deps_in_region[n_par] -= 1
                region.nodes.append(n)

                # if the node could free all dependencies in graph
                # we could begin a new node
                if _is_sink():
                    region_list.append(region)
                    region = Region(has_param=False, nodes=[])
                    deps_in_region.clear()

                # propagate common node attr if possible
                if len(n.all_input_nodes) == len([node for node in n.all_input_nodes if node.name in self.cnode
                                                  ]) or _is_cop(n.target):
                    self.cnode.append(n.name)
                else:
                    deps[n] = len([user for user in n.users if user.op != "output"])
                    deps_in_region[n] = deps[n]
        return region_list


    def build_strategies_and_cost(self):
        """
        This method is to build the strategy vector for each node in the computation graph.
        """

        def _check_no_strategy_for_node(node: Node):
            label = False
            if node.op in ('placeholder', 'get_attr', 'call_method', 'output'):
                label = True

            elif node.op == "call_module":
                target = node.target
                submod = self.root_module.get_submodule(target)
                if (
                        len(list(submod.named_parameters(recurse=False))) == 0
                        and len(list(submod.named_buffers(recurse=False))) == 0
                ):
                    label = True

            elif node.op == "call_function":
                label = True
                input_nodes = list(node._input_nodes.keys())
                for inp_node in input_nodes:
                    if (inp_node.op == "get_attr") or (inp_node in no_offload_param_list):
                        label = False
                        break

                if len(input_nodes) == 1:
                    unique_inp_node = input_nodes[0]
                    if (unique_inp_node.op == "get_attr") or (unique_inp_node in no_offload_param_list):
                        label = False

            return label

        def _set_params_info_for_node(node: Node):
            assert node.op in ['call_function', 'call_module']

            assert hasattr(node, "node_info") and isinstance(node.node_info, NodeInfo)
            node_info = node.node_info
            node_info.has_param = True
            if node_info.param_indices is None:
                node_info.param_indices = []

            if node.op == 'call_module':
                target = node.target
                submod = self.root_module.get_submodule(target)
                for p in list(submod.parameters(recurse=False)):

                    node_info.param_indices.append(ModelParameters.param_idx)
                    node_info.param_size += p.data.numel() * p.data.element_size()
                    ModelParameters.fp16_params.append(p)
                    ModelParameters.fp32_master_params.append(p.detach().clone().float().pin_memory())
                    ModelParameters.param_idx += 1

            elif node.op == 'call_function':
                input_nodes = list(node._input_nodes.keys())
                for inp_node in input_nodes:
                    if inp_node.op == "get_attr":
                        attr_itr = self.root_module
                        atoms = inp_node.target.split(".")
                        for atom in atoms:
                            attr_itr = getattr(attr_itr, atom)

                        if isinstance(attr_itr, torch.nn.Parameter):
                            node_info.param_indices.append(ModelParameters.param_idx)
                            node_info.param_size += attr_itr.data.numel() * attr_itr.data.element_size()
                            ModelParameters.fp16_params.append(attr_itr)
                            ModelParameters.fp32_master_params.append(attr_itr.detach().clone().float())
                            ModelParameters.param_idx += 1

                if len(input_nodes) == 1:
                    unique_inp_node = input_nodes[0]
                    if unique_inp_node.op == "get_attr":
                        no_offload_param_list.append(node)
                    elif unique_inp_node in no_offload_param_list:
                        assert len(node_info.param_indices) == 0
                        assert node_info.param_size == 0
                        node_info.param_indices = unique_inp_node.node_info.param_indices.copy()
                        node_info.param_size += unique_inp_node.node_info.param_size
                        unique_inp_node.node_info.param_indices.clear()
                        unique_inp_node.node_info.param_size = 0
                        unique_inp_node.node_info.has_param = False
                        no_offload_param_list.remove(unique_inp_node)
                        no_offload_param_list.append(node)
                else:
                    for inp_node in input_nodes:
                        if inp_node in no_offload_param_list:
                            assert len(inp_node.node_info.param_indices) > 0
                            assert inp_node.node_info.param_size > 0
                            node_info.param_indices.extend(inp_node.node_info.param_indices)
                            node_info.param_size += inp_node.node_info.param_size
                            inp_node.node_info.param_indices.clear()
                            inp_node.node_info.param_size = 0
                            inp_node.node_info.has_param = False
                            no_offload_param_list.remove(inp_node)

        node_id = 0
        no_offload_param_list = []
        for node in self.nodes:
            node_id += 1
            setattr(node, "node_info", NodeInfo(node_id=node_id))
            strategies_vector = OffloadStrategiesVector(node)

            if _check_no_strategy_for_node(node):
                self.no_strategy_nodes.append(node)
                continue

            _set_params_info_for_node(node)
            generator = StrategyGenerator(node, self.graph)
            strategies_vector.extend(generator.generate())
            setattr(node, 'strategies_vector', strategies_vector)
            self.leaf_strategies.append(strategies_vector)
            self.strategy_map[node] = strategies_vector
