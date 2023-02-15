from typing import List
import copy
import torch
from torch.fx.graph import Graph
from torch.fx.node import Node
from colossalai.utils.cuda import get_current_device
from colossalai.fx.profiler import (calculate_fwd_out, calculate_fwd_tmp, calculate_fwd_in, is_compatible_with_meta,
                                    parameter_size)
from strategies_constructor import OffloadStrategiesConstructor
from offload_strategy import SystemConfig
from util import Region, NodeInfo


class SynGreedySolver:

    def __init__(self,
                 region_list: List[Region],
                 memory_budget: float = -1.0):

        self.region_list = region_list
        self.memory_budget = memory_budget if memory_budget > 0 \
            else torch.cuda.get_device_properties(get_current_device()).total_memory
        self.peak_mem = -1

    def _call_solver_greedy(self):
        peak_mem_saving, total_mem_saving = self._compute_mem_saving()
        assert peak_mem_saving == 0 and total_mem_saving == 0, \
            f"pms={peak_mem_saving / 1024 ** 2:.3f}MB, tms={total_mem_saving / 1024 ** 2:.3f}MB"
        while self.peak_mem > self.memory_budget:
            offload_region = None
            max_profit = 0
            for region in self.region_list[:-1]:
                if region.param_size > 0 and not region.is_offload:
                    region.is_offload = True
                    tmp_peak_mem_saving, tmp_total_mem_saving = self._compute_mem_saving()
                    comm_cost = region.param_size / SystemConfig.BANDWIDTH
                    if region.r_id < region.shared_rid:
                        comm_cost *= 2.0
                    profit = tmp_peak_mem_saving / comm_cost
                    if profit > max_profit:
                        offload_region = region
                        max_profit = profit
                        peak_mem_saving = tmp_peak_mem_saving
                    region.is_offload = False

            # assert offload_region is not None
            if offload_region is not None:
                print('region_to_offload', offload_region.r_id,
                      self.peak_mem / 1024 ** 2)
                offload_region.is_offload = True
                offload_region.is_syn = True
                self.peak_mem -= peak_mem_saving
            else:
                raise RuntimeError(
                    f"can't find the offload strategy met the memory budget {self.memory_budget / 1024 ** 2} MB, "
                    f"it needs {self.peak_mem / 1024 ** 2:.3f} MB at least!")

            self._update_rumtime_mem_for_node()

    def _call_solver_l2l(self):
        for region in self.region_list[:-1]:
            region.is_offload = True
            region.is_syn = True

    def _update_rumtime_mem_for_node(self):
        self._compute_mem_saving(update_flag=True)

    def _compute_mem_saving(self, update_flag=False):
        cur_peak_mem = 0
        total_mem_saving = 0
        runtime_mem = 0

        # forward
        for region in self.region_list:
            # upload parameter
            if region.r_id <= region.shared_rid or (
                    region.r_id > region.shared_rid and self.region_list[region.shared_rid].is_offload):
                runtime_mem += region.param_size

            for node in region.nodes:

                runtime_mem = runtime_mem + \
                    calculate_fwd_tmp(node) + calculate_fwd_out(node)

                total_mem_saving += max(node.node_info.runtime_fwd_mem -
                                        runtime_mem, 0)

                if update_flag:
                    node.node_info.runtime_fwd_mem = runtime_mem

                cur_peak_mem = max(runtime_mem, cur_peak_mem)
                if cur_peak_mem > self.peak_mem and self.peak_mem > 0:
                    print("cur peak mem too high in forward", node, region.r_id)

            if region.is_offload:
                runtime_mem -= region.param_size

        # backward
        bwd_deps = {}
        for region in self.region_list.__reversed__():

            # upload parameters
            if region.is_offload:
                runtime_mem += region.param_size

            # add the gradient of the parameter
            if region.r_id < region.shared_rid:
                runtime_mem += 2.0 * region.param_size
            else:
                runtime_mem += region.param_size

            for node in region.nodes.__reversed__():

                runtime_mem -= calculate_fwd_out(node)
                runtime_mem = runtime_mem + \
                    node.meta['bwd_mem_tmp'] + node.meta['bwd_mem_out']

                # The memory savings of a node may be negative due to parameter prefetch.
                total_mem_saving += max(node.node_info.runtime_bwd_mem -
                                        runtime_mem, 0)

                cur_peak_mem = max(runtime_mem, cur_peak_mem)

                if cur_peak_mem > self.peak_mem and self.peak_mem > 0:
                    print("cur peak mem too high in backward", node, region.r_id)
                    return 0, 0

                if update_flag:
                    node.node_info.runtime_bwd_mem = runtime_mem

                runtime_mem = runtime_mem - \
                    node.meta['bwd_mem_tmp'] - calculate_fwd_tmp(node)

                # free bwd_mem_out
                bwd_deps[node] = len(node.all_input_nodes)
                for user_node in node.users:
                    if user_node in bwd_deps:
                        bwd_deps[user_node] -= 1
                        if bwd_deps[user_node] <= 0:
                            runtime_mem -= user_node.meta['bwd_mem_out']

                if runtime_mem < 0:
                    raise RuntimeError(f"region id: {region.r_id}, node name: {node.name}, "
                                       f"runtime_mem: {runtime_mem / 1024 ** 2:.3f}MB ---"
                                       f"runtime memory computed less than 0, which is miscalculated!")

            # release parameter and offload gradient in region
            if region.r_id == region.shared_rid:
                runtime_mem -= 2.0 * region.param_size
            elif region.r_id < region.shared_rid:
                runtime_mem -= 3.0 * region.param_size
            elif self.region_list[region.shared_rid].is_offload:
                runtime_mem -= region.param_size

        if update_flag:
            assert self.peak_mem == cur_peak_mem
        if self.peak_mem < 0:
            self.peak_mem = cur_peak_mem
        peak_mem_saving = self.peak_mem - cur_peak_mem
        return peak_mem_saving, total_mem_saving


class AsynGreedySolver:

    def __init__(self,
                 region_list: List[Region],
                 memory_budget: float = -1.0):

        self.region_list = region_list

        self.memory_budget = memory_budget if memory_budget > 0 \
            else torch.cuda.get_device_properties(get_current_device()).total_memory
        # used to record computation start and end time stamp of each region
        self.region_compute_stream: List[List[float, float]] = []
        # used to record prefetch operation start and end time stamp of each region
        self.region_prefetch_stream: List[List[float, float]] = []
        # used to record offload operation start and end time stamp of each region
        self.region_offload_stream: List[List[float, float]] = []

        self.peak_mem = -1
        self.iter_end_time = -1
        # record corresponding host region which prefetch the region to be offloaded
        self.region_to_region_map = {}
        # record the memory saving from the region to be offloaded
        self.region_to_mem_saving_map = {}

    def _call_solver_greedy(self):
        _, peak_mem_saving, total_mem_saving = self._eval_one_choice(
            update_flag=True)
        assert peak_mem_saving < 0 and total_mem_saving <= 0, \
            f"pms={peak_mem_saving / 1024 ** 2:.3f}MB, tms={total_mem_saving / 1024 ** 2:.3f}MB"
        print("region num", len(self.region_list))
        print("init peak memory", self.peak_mem / 1024 ** 2, "MB")
        # record corresponding host region which prefetch the region to be offloaded
        region_to_region_map = {}
        # record the memory saving from the region to be offloaded
        region_to_mem_saving_map = {}
        while self.peak_mem > self.memory_budget:
            region_to_offload = None
            max_offload_profit = (0,)

            # search which region should be offloaded, last region is not offloaded
            for region in self.region_list[:-1]:
                assert region.r_id == self.region_list.index(region)
                if region.param_size and not region.is_offload:
                    max_prefetch_profit = (0,)

                    # TODO 当前并未保证 prefetch 遵循 backward 的顺序执行
                    # search when to prefetch the node offloaded
                    for host_region in self.region_list[region.r_id:]:
                        if host_region.bwd_prefetch_region is not None:
                            continue

                        profit, tmp_peak_mem_saving, tmp_total_mem_saving = self._try_to_offload(
                            host_region, region)
                        if tmp_peak_mem_saving == 0:
                            continue

                        if self._compare_profit(profit, max_prefetch_profit):
                            region_to_region_map[region.r_id] = host_region
                            region_to_mem_saving_map[region.r_id] = tmp_peak_mem_saving
                            max_prefetch_profit = profit
                            if profit[0] == float('inf'):
                                break

                    if self._compare_profit(max_prefetch_profit, max_offload_profit):
                        region_to_offload = region
                        max_offload_profit = max_prefetch_profit

            if region_to_offload is not None:
                assert region_to_region_map.get(
                    region_to_offload.r_id, None) is not None
                assert self.region_to_region_map.get(
                    region_to_offload.r_id, None) is None
                assert self.region_to_mem_saving_map.get(
                    region_to_offload.r_id, None) is None

                region_to_offload.is_offload = True

                print('region_to_offload', region_to_offload.r_id, region_to_region_map[region_to_offload.r_id].r_id,
                      self.peak_mem / 1024 ** 2)
                if region_to_region_map[region_to_offload.r_id] == region_to_offload:
                    region_to_offload.is_syn = True
                else:
                    region_to_region_map[region_to_offload.r_id].bwd_prefetch_region = region_to_offload
                    self.region_to_region_map[region_to_offload.r_id] = region_to_region_map[region_to_offload.r_id]

                self.peak_mem -= region_to_mem_saving_map[region_to_offload.r_id]
                self.region_to_mem_saving_map[region_to_offload.r_id] = region_to_mem_saving_map[region_to_offload.r_id]

            elif self.region_to_region_map.__len__() > 0:
                self._repair_strategy()
            else:
                raise RuntimeError(
                    f"can't find the offload strategy met the memory budget {self.memory_budget / 1024 ** 2} MB, "
                    f"it needs {self.peak_mem / 1024 ** 2:.3f} MB at least!")

            self._eval_one_choice(update_flag=True)

            region_to_region_map.clear()
            region_to_mem_saving_map.clear()


    def _try_to_offload(self, host_region: Region, offload_region: Region):

        orig_prefetch = host_region.bwd_prefetch_region
        orig_is_syn = offload_region.is_syn
        orig_is_offload = offload_region.is_offload

        if host_region == offload_region:
            offload_region.is_syn = True
        else:
            host_region.bwd_prefetch_region = offload_region
        offload_region.is_offload = True

        profit, peak_mem_saving, total_mem_saving = self._eval_one_choice()

        host_region.bwd_prefetch_region = orig_prefetch
        offload_region.is_syn = orig_is_syn
        offload_region.is_offload = orig_is_offload

        return profit, peak_mem_saving, total_mem_saving

    def _try_convert_to_syn_prefetch(self, host_region: Region, offload_region: Region):

        orig_prefetch = host_region.bwd_prefetch_region
        orig_is_syn = offload_region.is_syn
        assert orig_prefetch is not None and not orig_is_syn

        host_region.bwd_prefetch_region = None
        offload_region.is_syn = True

        profit, peak_mem_saving, total_mem_saving = self._eval_one_choice()

        host_region.bwd_prefetch_region = orig_prefetch
        offload_region.is_syn = orig_is_syn

        return profit, peak_mem_saving, total_mem_saving

    def _repair_strategy(self):
        print("repair.........................")

        # succeed = True

        peak_mem_saving = 0
        while peak_mem_saving <= 0:

            max_profit = (0,)
            undo_host_region = None
            undo_offload_region = None

            for offload_region_id, host_region in self.region_to_region_map.items():
                offload_region = self.region_list[offload_region_id]
                assert host_region.bwd_prefetch_region == offload_region
                assert offload_region.is_offload
                assert not offload_region.is_syn

                profit, tmp_peak_mem_saving, tmp_total_mem_saving = self._try_convert_to_syn_prefetch(host_region,
                                                                                                      offload_region)
                # if tmp_peak_mem_saving <= 0:
                #     print("not reduce peak memory:", offload_region_id, host_region.r_id)
                #     # continue

                if self._compare_profit(profit, max_profit):
                    undo_host_region = host_region
                    undo_offload_region = offload_region
                    peak_mem_saving = tmp_peak_mem_saving
                    max_profit = profit

            if undo_host_region is None and undo_offload_region is None:
                if self.region_to_region_map.__len__() == 0:
                    return
                undo_offload_region_id, undo_host_region = list(
                    self.region_to_region_map.items())[0]
                undo_offload_region = self.region_list[undo_offload_region_id]

            assert not undo_offload_region.is_syn
            undo_offload_region.is_syn = True
            undo_host_region.bwd_prefetch_region = None

            self.peak_mem -= peak_mem_saving
            self.region_to_region_map.pop(undo_offload_region.r_id)
            self.region_to_mem_saving_map[undo_offload_region.r_id] += peak_mem_saving

        # return succeed

    def _compute_offload_profit(self, mem_saving: float, extra_cost: float):
        if extra_cost == 0:
            # If the prefetch operation can be completely overlapped,
            # then will provide memory saving information to downstream
            return (float('inf'), mem_saving)
        return (mem_saving / extra_cost, mem_saving)

    def _compare_profit(self, profit_a: tuple, profit_b: tuple):
        for val1, val2 in zip(profit_a, profit_b):
            if val1 != val2:
                return val1 > val2
        return False

    def _eval_one_choice(self, update_flag=False):
        '''
        In forward pass, parameter prefetch is advanced by one region
        In backward pass, computation and gradient offload are urgent
        '''

        def eval_fwd_cost_per_region():

            start_prefetch = max(
                region_prefetch_stream[-1][1], region_compute_stream[-1][1])
            # upload parameters of the current region
            if region.r_id == 0:
                region_prefetch_stream.append(
                    [start_prefetch, start_prefetch + region.param_size / SystemConfig.BANDWIDTH])
                fwd_pref_ends[region.r_id] = region_prefetch_stream[-1][1]

            start_prefetch = max(
                region_prefetch_stream[-1][1], region_compute_stream[-1][1])
            # prefetch parameters of the next region
            fwd_prefetch_region = region.fwd_prefetch_region
            if fwd_prefetch_region and self._requires_fwd_pref(fwd_prefetch_region):
                region_prefetch_stream.append(
                    [start_prefetch, start_prefetch + fwd_prefetch_region.param_size / SystemConfig.BANDWIDTH])
                fwd_pref_ends[fwd_prefetch_region.r_id] = region_prefetch_stream[-1][1]

            # execution computation
            start_comp = max(
                region_compute_stream[-1][1], fwd_pref_ends.get(region.r_id, 0))
            region_compute_stream.append([start_comp, start_comp + sum([node.meta.get(
                'fwd_flop', 0)/SystemConfig.COMPUTE_POWER for node in region.nodes])])

        def eval_fwd_mem_per_region():

            runtime_mem = runtime_mem_list[-1]
            cur_peak_mem = peak_mem_list[-1]
            total_mem_saving = total_mem_saving_list[-1]

            # upload parameters of the current region
            if region.r_id == 0:
                runtime_mem += region.param_size

            # prefetch parameters of the next region
            fwd_prefetch_region = region.fwd_prefetch_region
            if fwd_prefetch_region and self._requires_fwd_pref(fwd_prefetch_region):
                runtime_mem += fwd_prefetch_region.param_size

            for node in region.nodes:
                runtime_mem = runtime_mem + \
                    calculate_fwd_tmp(node) + calculate_fwd_out(node)
                cur_peak_mem = max(runtime_mem, cur_peak_mem)

                if cur_peak_mem > self.peak_mem and self.peak_mem > 0:
                    print("cur peak mem too high in forward", node, region.r_id)

                total_mem_saving += node.node_info.runtime_fwd_mem - runtime_mem

                if update_flag:
                    node.node_info.runtime_fwd_mem = runtime_mem

            if region.is_offload:
                runtime_mem -= region.param_size

            runtime_mem_list.append(runtime_mem)
            peak_mem_list.append(cur_peak_mem)
            total_mem_saving_list.append(total_mem_saving)

        def eval_bwd_cost_per_region():
            # upload parameters of the current region
            if region.is_syn:
                assert region.is_offload
                start_prefetch = max(
                    region_prefetch_stream[-1][1], region_compute_stream[-1][1])
                region_prefetch_stream.append(
                    [start_prefetch, start_prefetch + region.param_size / SystemConfig.BANDWIDTH])
                bwd_pref_ends[region.r_id] = region_prefetch_stream[-1][1]

            # prefetch parameters of the region choiced, which is parallel to computation
            if region.bwd_prefetch_region is not None:
                start_prefetch = max(
                    region_prefetch_stream[-1][1], region_compute_stream[-1][1])
                region_prefetch_stream.append(
                    [start_prefetch, start_prefetch + region.bwd_prefetch_region.param_size / SystemConfig.BANDWIDTH])
                bwd_pref_ends[region.bwd_prefetch_region.r_id] = region_prefetch_stream[-1][1]

            # ensure parameters are available
            start_comp = max(
                region_compute_stream[-1][1], bwd_pref_ends.get(region.r_id, 0))
            region_compute_stream.append([start_comp, start_comp + sum([node.meta.get(
                'bwd_flop', 0)/SystemConfig.COMPUTE_POWER for node in region.nodes.__reversed__()])])

            # release gradient of the region which complete offloading
            while len(region_offload_stream_waited) and region_offload_stream_waited[0][1] < region_compute_stream[-1][1]:
                g_offload_period = region_offload_stream_waited.pop(0)
                region_offload_stream_freed.append(g_offload_period)
                trg_r_id = region.r_id if g_offload_period[1] >= start_comp else region.r_id - 1
                if trg_r_id not in reg_to_g_freed:
                    reg_to_g_freed[trg_r_id] = []
                try:
                    reg_to_g_freed[trg_r_id].append(
                        gid2rid[len(region_offload_stream_freed)-1])
                except:
                    print(gid2rid)
                    print(len(region_offload_stream_freed), len(region_offload_stream_waited))
                    raise RuntimeError("KeyError")


            # offload gradient
            if self._required_bwd_offl(region):
                start_offload = max(region_offload_stream_waited[-1][1], region_compute_stream[-1][1]) if len(
                    region_offload_stream_waited) else region_compute_stream[-1][1]
                region_offload_stream_waited.append(
                    [start_offload, start_offload + region.param_size/SystemConfig.BANDWIDTH])

        def eval_bwd_mem_per_region():

            runtime_mem = runtime_mem_list[-1]
            cur_peak_mem = peak_mem_list[-1]
            total_mem_saving = total_mem_saving_list[-1]

            # upload parameters of the current region
            if region.is_syn:
                runtime_mem += region.param_size
            # prefetch parameters of the region choiced
            if region.bwd_prefetch_region is not None:
                # TODO 如果 prefetch stream 被阻塞，内存是否有可能也被延迟分配
                runtime_mem += region.bwd_prefetch_region.param_size

            # add the gradient of the parameter
            if region.r_id < region.shared_rid:
                # gradient accumulation is required for shared parameters
                runtime_mem += 2.0 * region.param_size
            else:
                runtime_mem += region.param_size

            for node in region.nodes.__reversed__():

                runtime_mem -= calculate_fwd_out(node)
                runtime_mem = runtime_mem + \
                    node.meta['bwd_mem_tmp'] + node.meta['bwd_mem_out']
                cur_peak_mem = max(runtime_mem, cur_peak_mem)

                if cur_peak_mem > self.peak_mem and self.peak_mem > 0:
                    print("cur peak mem too high in backward", node, region.r_id)
                    return 0, 0

                # The memory savings of a node may be negative due to parameter prefetch.
                total_mem_saving += node.node_info.runtime_bwd_mem - runtime_mem

                if update_flag:
                    node.node_info.runtime_bwd_mem = runtime_mem

                runtime_mem = runtime_mem - \
                    node.meta['bwd_mem_tmp'] - calculate_fwd_tmp(node)

                # free bwd_mem_out
                bwd_node_deps[node] = len(node.all_input_nodes)
                for user_node in node.users:
                    if user_node in bwd_node_deps:
                        bwd_node_deps[user_node] -= 1
                        if bwd_node_deps[user_node] <= 0:
                            runtime_mem -= user_node.meta['bwd_mem_out']

            # release parameters of the region
            runtime_mem -= region.param_size
            # release gradients of the region which complete offloading
            if region.r_id in reg_to_g_freed:
                runtime_mem -= sum(
                    [self.region_list[rid].param_size for rid in reg_to_g_freed[region.r_id]])
                reg_to_g_freed.pop(region.r_id)

            runtime_mem_list.append(runtime_mem)
            peak_mem_list.append(cur_peak_mem)
            total_mem_saving_list.append(total_mem_saving)

        region_compute_stream = [[0, 0]]
        region_prefetch_stream = [[0, 0]]
        fwd_pref_ends = {}

        runtime_mem_list = [0]
        peak_mem_list = [0]
        total_mem_saving_list = [0]

        # evaluate forward
        for region in self.region_list:
            if region.param_size and region.r_id < len(self.region_list) - 1:
                for nr in self.region_list[region.r_id+1:]:
                    if nr.param_size:
                        region.fwd_prefetch_region = nr
                        break
            eval_fwd_cost_per_region()
            eval_fwd_mem_per_region()

        bwd_pref_ends = {}
        bwd_node_deps = {}
        reg_to_g_freed = {}

        region_offload_stream_freed = []
        region_offload_stream_waited = []

        gid2rid = {}
        r_id_with_g = 0

        # evaluate backward
        for region in self.region_list.__reversed__():
            # print(region.r_id, r_id_with_g, len(region_offload_stream_freed), len(region_offload_stream_waited))
            if self._required_bwd_offl(region):
                gid2rid[r_id_with_g] = region.r_id
                r_id_with_g += 1
            eval_bwd_cost_per_region()
            eval_bwd_mem_per_region()

        # release remaining grads
        while(len(region_offload_stream_waited)):
            # assert not len(reg_to_g_freed)
            region_offload_stream_freed.append(region_offload_stream_waited.pop(0))
            unreleased_grad_size = self.region_list[gid2rid[len(region_offload_stream_freed)-1]].param_size
            runtime_mem_list.append(runtime_mem_list[-1]-unreleased_grad_size)
            peak_mem_list.append(peak_mem_list[-1])
            total_mem_saving_list.append(total_mem_saving_list[-1])

        iter_end_time = max(region_offload_stream_freed[-1][1], region_compute_stream[-1][1])
        peak_mem_saving = self.peak_mem - peak_mem_list[-1]
        if self.peak_mem > 0:
            assert peak_mem_saving >= 0
        total_mem_saving = total_mem_saving_list[-1]
        extra_comm_cost = max(iter_end_time - self.iter_end_time, 0)
        profit = self._compute_offload_profit(peak_mem_saving, extra_comm_cost)
        # profit = self._compute_offload_profit(total_mem_saving, extra_comm_cost)

        if update_flag:
            # assert self.peak_mem == peak_mem_list[-1]
            self.peak_mem = peak_mem_list[-1]
            self.iter_end_time = iter_end_time
            self.region_compute_stream = region_compute_stream
            self.region_prefetch_stream = region_prefetch_stream
            self.region_offload_stream = region_offload_stream_freed

        return profit, peak_mem_saving, total_mem_saving


    def _requires_fwd_pref(self, region: Region):
        return region.r_id <= region.shared_rid or (region.r_id > region.shared_rid and self.region_list[region.shared_rid].is_offload)

    def _required_bwd_offl(self, region: Region):
        return region.param_size and (region.r_id <= region.shared_rid)