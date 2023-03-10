from typing import List, Any
import torch
from torch.fx import Graph, Node

from util import NodeInfo, Region


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
        self.only_param_ops = []
        self.param_to_region = {}

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

        def _is_act(data: Any) -> bool:
            """Check if an op could be seen as parameter computation start

            Args:
                data (Any): meta_data

            Returns:
                bool
            """

            label = False
            if isinstance(data, torch.Tensor):
                return True
            elif isinstance(data, (tuple, list)):
                for d in data:
                    label = label or _is_act(d)
            return label

        def _maybe_param_comp_start() -> bool:
            """Check if an op could be seen as parameter computation start

            Args:
                n (Node): node

            Returns:
                bool
            """

            label = False
            if n.op == "get_attr":
                label = True
            elif n.op == "call_module":
                target = n.target
                submod = self.root_module.get_submodule(target)
                if (
                        len(list(submod.named_parameters(recurse=False))) != 0
                        or len(list(submod.named_buffers(recurse=False))) != 0
                ):
                    label = True

            return label and not sum([v for _, v in param_op_deps.items()])

        def _is_param_comp_end() -> bool:
            """Check if an op could be seen as parameter computation end

            Args:
                n (Node): node

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

            label = False

            if n.op == "call_module":
                target = n.target
                submod = self.root_module.get_submodule(target)
                if (
                        len(list(submod.named_parameters(recurse=False))) != 0
                        or len(list(submod.named_buffers(recurse=False))) != 0
                ):
                    label = True

            elif n.op == "call_function":
                label = any(map(lambda x: x.name in self.only_param_ops, n.all_input_nodes)) and any(
                    map(lambda x: x.name not in self.only_param_ops and not _is_cop(n.target), n.all_input_nodes))

            return label and not sum([v for _, v in param_op_deps.items()]) and not any(map(_is_inplace, n.users))

        def _exception_node_handling():
            # TODO meta info prop bug
            if n.name.__contains__("transpose") and n.meta['fwd_out'][0].dim() <= 2:
                n.meta['fwd_out'] = []

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
        region = Region(r_id=region_id, nodes=[], fp16_params=[], shared_rid=region_id)

        act_n = None

        for n in self.graph.nodes:
            if n.op != "placeholder" and n.op != "output":
                for n_par in n.all_input_nodes:
                    if n_par.op != "placeholder" and n_par.name not in self.cnode:
                        deps[n_par] -= 1
                    if n_par.op != "placeholder" and n_par.name in self.only_param_ops:
                        param_op_deps[n_par] -= 1

                if act_n in region.nodes and _maybe_param_comp_start():
                    ns = []
                    border_n_idx = region.nodes.index(act_n)
                    if border_n_idx < len(region.nodes):
                        ns = region.nodes[border_n_idx + 1:]
                        region.nodes = region.nodes[:border_n_idx + 1]
                    region_list.append(region)
                    region_id += 1
                    region = Region(r_id=region_id, nodes=ns, fp16_params=[], shared_rid=region_id)

                _exception_node_handling()
                region.nodes.append(n)
                self._set_node_and_region_info(node_id, n, region)
                node_id += 1

                # if the node could free all dependencies in graph
                # we could begin a new region
                if _is_param_comp_end():
                    region_list.append(region)
                    region_id += 1
                    region = Region(r_id=region_id, nodes=[], fp16_params=[], shared_rid=region_id)

                # propagate common node attr if possible
                if len(n.all_input_nodes) == len([node for node in n.all_input_nodes if node.name in self.cnode
                                                  ]) or _is_cop(n.target):
                    self.cnode.append(n.name)
                else:
                    deps[n] = len([user for user in n.users if user.op != "output"])

                # propagate param node attr if possible
                if len(n.all_input_nodes) == len([node for node in n.all_input_nodes if node.name in self.only_param_ops
                                                  ]) or n.op == "get_attr":
                    self.only_param_ops.append(n.name)
                    param_op_deps[n] = len([user for user in n.users if user.op != "output"])

                # record last activation node
                if _is_act(n._meta_data):
                    act_n = n

        return region_list

    def _set_node_and_region_info(self, node_id: int, cur_n: Node, cur_reg: Region):

        node_info = NodeInfo(node_id)

        if cur_n.op == 'call_module':
            target = cur_n.target
            submod = self.root_module.get_submodule(target)
            for p in list(submod.parameters(recurse=False)):

                if p in self.param_to_region:
                    print(f"region {cur_reg.r_id} param existed! {p.data.numel() * p.data.element_size() / 1024 ** 2}")
                    cur_reg.shared_rid = self.param_to_region[p].r_id
                    self.param_to_region[p].shared_rid = cur_reg.r_id
                else:
                    self.param_to_region[p] = cur_reg

                node_info.param_size += p.data.numel() * p.data.element_size()
                cur_reg.fp16_params.append(p)

        elif cur_n.op == "get_attr":
            attr_itr = self.root_module
            atoms = cur_n.target.split(".")
            for atom in atoms:
                attr_itr = getattr(attr_itr, atom)

            if isinstance(attr_itr, torch.nn.Parameter):

                if attr_itr in self.param_to_region:
                    print(
                        f"region {cur_reg.r_id} param existed! {attr_itr.data.numel() * attr_itr.data.element_size() / 1024 ** 2}")
                    cur_reg.shared_rid = self.param_to_region[attr_itr].r_id
                    self.param_to_region[attr_itr].shared_rid = cur_reg.r_id
                else:
                    self.param_to_region[attr_itr] = cur_reg

                node_info.param_size += attr_itr.data.numel() * attr_itr.data.element_size()
                cur_reg.fp16_params.append(attr_itr)

        cur_n.node_info = node_info
        cur_reg.param_size += node_info.param_size

        # if cur_reg.region_shared_param is not None and cur_reg.r_id > cur_reg.region_shared_param.r_id:
        #     cur_reg.param_size = cur_reg.region_shared_param.param_size
