from typing import List
import copy
import torch
from torch.fx.graph import Graph
from torch.fx.node import Node
from colossalai.utils.cuda import get_current_device
from colossalai.fx.profiler import (calculate_fwd_out, calculate_fwd_tmp, calculate_fwd_in, is_compatible_with_meta, parameter_size)
from strategies_constructor import OffloadStrategiesConstructor
from offload_strategy import SystemConfig

class Solver:

    def __init__(self,
                 graph: Graph,
                 strategies_constructor: OffloadStrategiesConstructor,
                 memory_budget: float = -1.0):
        self.graph = graph
        self.strategies_constructor = strategies_constructor
        self.leaf_strategies = self.strategies_constructor.leaf_strategies
        self.nodes = [strategies_vector.node for strategies_vector in self.leaf_strategies]
        self.memory_budget = memory_budget if memory_budget > 0 \
            else torch.cuda.get_device_properties(get_current_device()).total_memory

    def _compute_mem_saving(self):
        peak_mem = 0
        total_mem_saving = 0

        # TODO 参数大小不应该初始化为整个模型，应该逐层的upload到gpu
        # param_size = parameter_size(self.graph.owning_module)

        # runtime_mem = param_size

        runtime_mem = 0

        for node in self.graph.nodes:
            runtime_mem = runtime_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            # prefetch parameter
            runtime_mem += node.node_info.param_size
            total_mem_saving += max(node.node_info.runtime_fwd_mem - runtime_mem, 0)
            node.node_info.runtime_fwd_mem = runtime_mem

            peak_mem = max(runtime_mem, peak_mem)
            if node.node_info.offload_param_flag:
                runtime_mem -= node.node_info.param_size

        grad_in_computed = {}
        for node in self.graph.nodes.__reversed__():
            runtime_mem -= calculate_fwd_out(node)
            runtime_mem = runtime_mem + node.meta['bwd_mem_tmp'] + node.meta['bwd_mem_out']
            if node.node_info.has_param:
                if node.node_info.offload_param_flag:
                    # upload
                    runtime_mem += node.node_info.param_size
                # add weighted node gradient
                runtime_mem += node.node_info.param_size
                total_mem_saving += max(node.node_info.runtime_bwd_mem - runtime_mem, 0)
                # 不应该在此时更新
                node.node_info.runtime_bwd_mem = runtime_mem

                peak_mem = max(runtime_mem, peak_mem)

                # release parameter and offload gradient
                runtime_mem -= 2 * node.node_info.param_size
            peak_mem = max(runtime_mem, peak_mem)
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

        return peak_mem, total_mem_saving

    def _call_solver_greedy_v1(self):
        peak_mem, total_mem_saving = self._compute_mem_saving()
        assert total_mem_saving == 0
        while peak_mem > self.memory_budget:
            offload_node = None
            max_profit = 0
            reduced_peak_mem = peak_mem
            for node in self.nodes:
                if (not node.node_info.offload_param_flag) and node.node_info.has_param:
                    node.node_info.offload_param_flag = True
                    tmp_peak_mem, tmp_total_mem_saving = self._compute_mem_saving()
                    profit = (peak_mem - tmp_peak_mem) / node.strategies_vector[0].comm_cost
                    if profit > max_profit:
                        offload_node = node
                        max_profit = profit
                        reduced_peak_mem = tmp_peak_mem
                    node.node_info.offload_param_flag = False
            offload_node.node_info.offload_param_flag = True
            peak_mem = reduced_peak_mem

    def _call_solver_l2l(self):
        for node in self.nodes:
            node.node_info.offload_param_flag = True


class AsynGreedySolver:

    def __init__(self,
                 graph: Graph,
                 # strategies_constructor: OffloadStrategiesConstructor,
                 memory_budget: float = -1.0):
        self.graph = graph
        # self.strategies_constructor = strategies_constructor
        # self.leaf_strategies = self.strategies_constructor.leaf_strategies
        # self.nodes = [strategies_vector.node for strategies_vector in self.leaf_strategies]
        self.nodes = list(self.graph.nodes)
        self.memory_budget = memory_budget if memory_budget > 0 \
            else torch.cuda.get_device_properties(get_current_device()).total_memory
        # used to record computation start and end time stamp of each node
        self.node_compute_stream: List[List[float, float]] = []
        # used to record prefetch operation start and end time stamp of each node
        self.param_prefetch_stream: List[List[float, float]] = []

        self._init_compute_stream()

        self.peak_mem = -1
        # record corresponding host node which prefetch the node to be offloaded
        self.node_to_node_map = {}
        # record the memory saving from the node to be offloaded
        self.node_to_mem_saving_map = {}

    def _init_compute_stream(self):
        compute_timestamp = 0
        for node in self.graph.nodes:
            if node.node_info.has_param:
                # upload parameter
                compute_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH
            self.node_compute_stream.append(
                [compute_timestamp, compute_timestamp + node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER])
            compute_timestamp += node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER

        for node in self.graph.nodes.__reversed__():
            self.node_compute_stream.append(
                [compute_timestamp, compute_timestamp + node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER])
            compute_timestamp += node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER
            if node.node_info.has_param:
                # offload gradient
                compute_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH


    def _call_solver_greedy(self):
        peak_mem_saving, total_mem_saving = self._compute_mem_saving()
        assert peak_mem_saving == 0 and total_mem_saving < 0
        print("node num", len(self.nodes))
        print("init peak memory", self.peak_mem/1024**2, "MB")
        # record corresponding host node which prefetch the node to be offloaded
        node_to_node_map = {}
        # record the memory saving from the node to be offloaded
        node_to_mem_saving_map = {}
        while self.peak_mem > self.memory_budget:
            node_to_offload = None
            max_offload_profit = (0,)

            # search which node should be offloaded
            for node in self.nodes:
                if node.node_info.has_param and (not node.node_info.offload_param_flag):
                    node_idx = self.nodes.index(node)
                    max_prefetch_profit = (0,)

                    # TODO 当前并未保证 prefetch 遵循 backward 的顺序执行
                    # search when to prefetch the node offloaded
                    for following_node in self.nodes[node_idx:]:
                        if following_node.op == "get_attr":
                            continue
                        if following_node.node_info.node_to_prefetch is not None:
                            continue
                        tmp_peak_mem_saving, tmp_total_mem_saving = self._compute_mem_saving(following_node, node)

                        if tmp_peak_mem_saving <= 0:
                            # print(node_idx, following_node, tmp_peak_mem_saving)
                            continue

                        extra_comm_cost = self._compute_extra_comm_cost(following_node, node)
                        # tmp_profit = self._compute_offload_profit(tmp_peak_mem_saving, extra_comm_cost)
                        tmp_profit = self._compute_offload_profit(tmp_total_mem_saving, extra_comm_cost)

                        if self._compare_profit(tmp_profit, max_prefetch_profit):
                            node_to_node_map[node] = following_node
                            node_to_mem_saving_map[node] = tmp_peak_mem_saving
                            max_prefetch_profit = tmp_profit
                            if tmp_profit[0] == float('inf'):
                                break

                    if self._compare_profit(max_prefetch_profit, max_offload_profit):
                        node_to_offload = node
                        max_offload_profit = max_prefetch_profit

            if node_to_node_map.get(node_to_offload, None) is not None:

                print('node_to_offload', node_to_offload, node_to_node_map[node_to_offload])
                if node_to_node_map[node_to_offload] == node_to_offload:
                    node_to_offload.node_info.syn_upload_flag = True
                else:
                    node_to_node_map[node_to_offload].node_info.node_to_prefetch = node_to_offload

                node_to_offload.node_info.offload_param_flag = True
                self.peak_mem -= node_to_mem_saving_map[node_to_offload]

                assert self.node_to_node_map.get(node_to_offload, None) is None
                assert self.node_to_mem_saving_map.get(node_to_offload, None) is None
                self.node_to_node_map[node_to_offload] = node_to_node_map[node_to_offload]
                self.node_to_mem_saving_map[node_to_offload] = node_to_mem_saving_map[node_to_offload]

            else:
                self._repair_strategy()

            self._update_rumtime_mem_for_node()
            self._update_exec_stream_and_node_info()

            node_to_node_map.clear()
            node_to_mem_saving_map.clear()


    def _repair_strategy(self):
        print("repair.........................")

        peak_mem_saving = 0

        while peak_mem_saving <= 0:

            max_profit = (0,)
            cancel_offload_node = None
            cancel_host_node = None

            for node_to_offload, host_node in self.node_to_node_map.items():
                if node_to_offload == host_node:
                    assert node_to_offload.node_info.offload_param_flag
                    assert node_to_offload.node_info.syn_upload_flag
                    continue

                assert node_to_offload.node_info.offload_param_flag
                assert host_node.node_info.node_to_prefetch == node_to_offload

                # cancel offload for the node
                node_to_offload.node_info.offload_param_flag = False
                host_node.node_info.node_to_prefetch = None

                tmp_peak_mem_saving, tmp_total_mem_saving = self._compute_mem_saving(node_to_offload, node_to_offload)
                # print("tmp_mem_saving", tmp_peak_mem_saving/1024**2, tmp_total_mem_saving/1024**2, node_to_offload)

                assert tmp_peak_mem_saving >= 0

                extra_comm_cost = self._compute_extra_comm_cost(node_to_offload, node_to_offload)
                tmp_profit = self._compute_offload_profit(tmp_total_mem_saving, extra_comm_cost)
                if self._compare_profit(tmp_profit, max_profit):
                    cancel_offload_node = node_to_offload
                    cancel_host_node = host_node
                    peak_mem_saving = tmp_peak_mem_saving
                    max_profit = tmp_profit

                # restore node info for the offload node
                node_to_offload.node_info.offload_param_flag = True
                host_node.node_info.node_to_prefetch = node_to_offload

            assert cancel_offload_node.node_info.syn_upload_flag == False
            cancel_offload_node.node_info.syn_upload_flag = True
            cancel_host_node.node_info.node_to_prefetch = None
            self.peak_mem -= peak_mem_saving
            self.node_to_node_map[cancel_offload_node] = cancel_offload_node
            self.node_to_mem_saving_map[cancel_offload_node] += peak_mem_saving

            # print("aft each repair node map.......................")
            # for node_to_offload, host_node in self.node_to_node_map.items():
            #     print(node_to_offload, host_node)
            # print("repair...peak_mem_saving", peak_mem_saving/1024**2, self.peak_mem / 1024 ** 2)
            # print("aft each repair offload plan.......................")
            # for node_tmp in self.graph.nodes:
            #     print(node_tmp, node_tmp.node_info.offload_param_flag, node_tmp.node_info.node_to_prefetch)

        # print("********************************************************")
        # print("aft final repair node map.......................")
        # for node_to_offload, host_node in self.node_to_node_map.items():
        #     print(node_to_offload, host_node)
        # print("aft final repair offload plan.......................")
        # for node_tmp in self.graph.nodes:
        #     print(node_tmp, node_tmp.node_info.offload_param_flag, node_tmp.node_info.node_to_prefetch)


    def _update_rumtime_mem_for_node(self):
        self._compute_mem_saving(update_flag=True)

    def _update_exec_stream_and_node_info(self):

        self.node_compute_stream.clear()
        self.param_prefetch_stream.clear()

        compute_timestamp = 0
        prefetch_timestamp = 0

        # forward
        for node in self.graph.nodes:
            if node.node_info.has_param:
                # upload parameter
                compute_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH

            self.node_compute_stream.append(
                [compute_timestamp, compute_timestamp + node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER])
            compute_timestamp += node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER

        # backward
        for node in self.graph.nodes.__reversed__():

            # prefetch parameter, which is parallel to node computation
            node_to_prefetch = node.node_info.node_to_prefetch
            if node.node_info.syn_upload_flag:
                # synchronous upload parameter
                assert node.node_info.offload_param_flag
                node_to_prefetch = node
            if node_to_prefetch is not None:
                prefetch_timestamp = max(prefetch_timestamp, compute_timestamp)
                self.param_prefetch_stream.append(
                    [prefetch_timestamp,
                     prefetch_timestamp + node_to_prefetch.node_info.param_size / SystemConfig.BANDWIDTH])
                prefetch_timestamp += node_to_prefetch.node_info.param_size / SystemConfig.BANDWIDTH
                node_to_prefetch.node_info.prefetch_end_timestamp = prefetch_timestamp

            if node.node_info.offload_param_flag:
                # wait parameter prefetch
                # TODO 最后一个节点是 output node，不会被offload
                assert node.node_info.prefetch_end_timestamp != 0
                compute_timestamp = max(node.node_info.prefetch_end_timestamp, compute_timestamp)

            self.node_compute_stream.append(
                [compute_timestamp, compute_timestamp + node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER])
            compute_timestamp += node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER

            if node.node_info.has_param:
                # offload gradient
                compute_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH


    def _compute_offload_profit(self, mem_saving: float, extra_cost: float):
        if extra_cost == 0:
            # If the prefetch operation can be completely overlapped,
            # then will provide memory saving information to downstream
            return (float('inf'), mem_saving)
        return (mem_saving/extra_cost, mem_saving)

    def _compare_profit(self, profit_a: tuple, profit_b: tuple):
        for val1, val2 in zip(profit_a, profit_b):
            if val1 != val2:
                return val1 > val2
        return False

    def _compute_mem_saving(self,
                            host_node_for_prefetch: Node=None,
                            node_to_offload: Node=None,
                            update_flag=False):
        cur_peak_mem = 0
        total_mem_saving = 0
        runtime_mem = 0

        # forward
        for node in self.graph.nodes:
            runtime_mem = runtime_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            # upload parameter
            runtime_mem += node.node_info.param_size
            total_mem_saving += max(node.node_info.runtime_fwd_mem - runtime_mem, 0)

            if update_flag:
                node.node_info.runtime_fwd_mem = runtime_mem

            cur_peak_mem = max(runtime_mem, cur_peak_mem)
            if cur_peak_mem > self.peak_mem and self.peak_mem > 0:
                print("cur peak mem too high in forward", node, host_node_for_prefetch, node_to_offload)
            if node.node_info.offload_param_flag or (node == node_to_offload):
                runtime_mem -= node.node_info.param_size

        # backward
        grad_in_computed = {}
        for node in self.graph.nodes.__reversed__():
            runtime_mem -= calculate_fwd_out(node)

            if cur_peak_mem > self.peak_mem and self.peak_mem > 0:
                print("cur peak mem too high in backward", node, host_node_for_prefetch, node_to_offload)
            # param prefetch
            node_to_prefetch = node.node_info.node_to_prefetch
            if node == host_node_for_prefetch:
                node_to_prefetch = node_to_offload
            if node_to_prefetch is not None:
                # TODO 如果 prefetch stream 被阻塞，内存是否有可能也被延迟分配
                runtime_mem += node_to_prefetch.node_info.param_size
            if node.node_info.syn_upload_flag:
                # synchronous upload parameter
                assert node.node_info.offload_param_flag
                node_to_prefetch = node
                runtime_mem += node_to_prefetch.node_info.param_size
            # if node_to_prefetch is not None:
            #     # TODO 如果 prefetch stream 被阻塞，内存是否有可能也被延迟分配
            #     runtime_mem += node_to_prefetch.node_info.param_size

            runtime_mem = runtime_mem + node.meta['bwd_mem_tmp'] + node.meta['bwd_mem_out']
            if node.node_info.has_param:
                # There is no need to add up the parameter size because it may be prefetched or not offloaded.

                # add the gradient of the parameter
                runtime_mem += node.node_info.param_size

                # The memory savings of a node may be negative due to parameter prefetch.
                total_mem_saving += (node.node_info.runtime_bwd_mem - runtime_mem)

                if update_flag:
                    node.node_info.runtime_bwd_mem = runtime_mem

                cur_peak_mem = max(runtime_mem, cur_peak_mem)

                # release parameter and offload gradient
                runtime_mem -= 2 * node.node_info.param_size
            cur_peak_mem = max(runtime_mem, cur_peak_mem)
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

        if (host_node_for_prefetch is None) and (node_to_offload is None):
            if update_flag:
                assert self.peak_mem == cur_peak_mem
            else:
                assert self.peak_mem < 0
                self.peak_mem = cur_peak_mem
        peak_mem_saving = self.peak_mem - cur_peak_mem
        return peak_mem_saving, total_mem_saving

    def _compute_extra_comm_cost(self, host_node_for_prefetch: Node, node_to_offload: Node):
        # 假设所有被 offload 的 node 的 prefetch 都在 backward 期间
        # 假设 node 的 prefetch 是遵循 backward 的执行顺序的
        # 假设不会在同一个 node 上挂两个 prefetch operation
        # forward stream 不会被影响

        node_prefetch_end_timestamp = {}

        compute_start_timestamp = self.node_compute_stream[len(self.nodes)][0]
        prefetch_start_timestamp = compute_start_timestamp
        for node in self.graph.nodes.__reversed__():

            # prefetch parameter, which is parallel to node computation
            if node.node_info.syn_upload_flag:
                # synchronous upload parameter
                assert node.node_info.offload_param_flag
                node_to_prefetch = node
                prefetch_start_timestamp = max(prefetch_start_timestamp, compute_start_timestamp)
                prefetch_start_timestamp += node_to_prefetch.node_info.param_size / SystemConfig.BANDWIDTH
                # node_to_prefetch.node_info.prefetch_end_timestamp = prefetch_start_timestamp
                node_prefetch_end_timestamp[node_to_prefetch] = prefetch_start_timestamp
            if node == host_node_for_prefetch:
                # assert node.node_info.node_to_prefetch is None
                node_to_prefetch = node_to_offload
                prefetch_start_timestamp = max(prefetch_start_timestamp, compute_start_timestamp)
                prefetch_start_timestamp += node_to_prefetch.node_info.param_size / SystemConfig.BANDWIDTH
                # node_to_prefetch.node_info.prefetch_end_timestamp = prefetch_start_timestamp
                node_prefetch_end_timestamp[node_to_prefetch] = prefetch_start_timestamp

            node_to_prefetch = node.node_info.node_to_prefetch
            if node_to_prefetch is not None:
                prefetch_start_timestamp = max(prefetch_start_timestamp, compute_start_timestamp)
                prefetch_start_timestamp += node_to_prefetch.node_info.param_size / SystemConfig.BANDWIDTH
                # node_to_prefetch.node_info.prefetch_end_timestamp = prefetch_start_timestamp
                node_prefetch_end_timestamp[node_to_prefetch] = prefetch_start_timestamp

            # if node.node_info.syn_upload_flag:
            #     # synchronous upload parameter
            #     assert node.node_info.offload_param_flag
            #     node_to_prefetch = node
            # if node_to_prefetch is not None:
            #     prefetch_start_timestamp = max(prefetch_start_timestamp, compute_start_timestamp)
            #     prefetch_start_timestamp += node_to_prefetch.node_info.param_size / SystemConfig.BANDWIDTH
            #     # node_to_prefetch.node_info.prefetch_end_timestamp = prefetch_start_timestamp
            #     node_prefetch_end_timestamp[node_to_prefetch] = prefetch_start_timestamp

            if node.node_info.offload_param_flag or (node == node_to_offload):
                # wait parameter prefetch
                # assert node.node_info.prefetch_end_timestamp != 0
                # compute_start_timestamp = max(node.node_info.prefetch_end_timestamp, compute_start_timestamp)
                assert node_prefetch_end_timestamp.get(node, 0) != 0
                compute_start_timestamp = max(node_prefetch_end_timestamp[node], compute_start_timestamp)

            compute_start_timestamp += node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER

            if node.node_info.has_param:
                # offload gradient
                compute_start_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH

        # restore node info
        # node_to_offload.node_info.prefetch_end_timestamp = 0

        node_prefetch_end_timestamp.clear()

        return max(compute_start_timestamp-self.node_compute_stream[-1][1], 0)


    def plot_execution_stream(self):
        # 画图
        x1 = self.node_compute_stream
        x2 = self.param_prefetch_stream
        pass