
import torch
from torch.fx import Graph, Node

from colossalai.amp.naive_amp import FP16Optimizer

from offload_strategy import OffloadStrategiesVector
from strategy_generator import StrategyGenerator
from options import SolverOption
from util import ModelParameters, NodeInfo

class OffloadStrategiesConstructor:
    """
    OffloadStrategiesConstructor is used to construct the offload plan for the model execution.

    Args:
        graph (Graph): a Graph object used for analysis and strategy generation.
        solver_option (SolverOption): a SolverOptions object which specifies the preferences for plan searching.
    """

    def __init__(self, graph: Graph, amp_optimizer: FP16Optimizer=None, solver_option: SolverOption=None):
        self.graph = graph
        assert graph.owning_module is not None, 'The given graph is not associated with a owning_module'
        self.root_module = self.graph.owning_module
        self.amp_optimizer = amp_optimizer
        self.nodes = list(graph.nodes)
        self.leaf_strategies = []
        self.strategy_map = {}
        self.solver_option = solver_option
        self.no_strategy_nodes = []

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

            def _get_fp16_param_index(param):
                for group_idx, param_group in enumerate(self.amp_optimizer._fp16_param_groups):
                    try:
                        param_idx = param_group.index(param)
                    except:
                        continue
                    return group_idx, param_idx

            # fp16_params = []
            # fp32_master_params = []

            assert hasattr(node, "node_info") and isinstance(node.node_info, NodeInfo)
            node_info = node.node_info
            node_info.has_param = True
            if node_info.param_indices is None:
                node_info.param_indices = []

            if node.op == 'call_module':
                target = node.target
                submod = self.root_module.get_submodule(target)
                for p in list(submod.parameters(recurse=False)):
                    # fp16_params.append(p)
                    # group_idx, param_idx = _get_fp16_param_index(p)
                    # fp32_master_params.append(self.amp_optimizer._fp32_master_param_groups[group_idx][param_idx])
                    # fp32_master_params.append(p.detach().clone().float())

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
                        # fp16_params.append(attr_itr)
                        # group_idx, param_idx = _get_fp16_param_index(attr_itr)
                        # fp32_master_params.append(self.amp_optimizer._fp32_master_param_groups[group_idx][param_idx])
                        # fp32_master_params.append(attr_itr.detach().clone().float())

                        # print(type(attr_itr), inp_node.name, attr_itr.data.shape)
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
