from typing import List
import torch
from torch.fx.node import Node
from colossalai.gemini.tensor_utils import alloc_storage, free_storage

from util import OffloadManager, Region, requires_upload_p_in_fwd, requires_release_p_in_bwd


def move_params_to_cuda(region_id: int):
    for param in OffloadManager.region_list[region_id].fp16_params:
        alloc_storage(param.data)
        fp32_data = torch.empty(
            param.data.shape, device='cuda', dtype=torch.float32)
        fp32_data.copy_(
            OffloadManager.param_fp16_to_fp32[param].data, non_blocking=True)
        param.data.copy_(fp32_data)
        # print(param.data)
        # free_storage(fp32_data)
        # del fp32_data


def move_grads_to_cpu(region_id: int):
    # torch.cuda.synchronize()
    for param in OffloadManager.region_list[region_id].fp16_params:
        assert param.grad is not None
        assert param not in OffloadManager.param_fp16_to_grad
        cpu_grad = torch.empty(param.data.shape, dtype=torch.half, pin_memory=True)
        cpu_grad.copy_(param.grad, non_blocking=True)
        OffloadManager.param_fp16_to_grad[param] = cpu_grad
        param.grad = None


def free_cuda_params(region_id: int):
    for param in OffloadManager.region_list[region_id].fp16_params:
        free_storage(param.data)
        torch.cuda.empty_cache()


class PreForwardUpload(torch.autograd.Function):
    """
    A customized upload operation.

    Args:
        input_: input tensor.
        fwd_info: information dict, which contains region indices that need to be uploaded during forward pass.
    """

    @staticmethod
    def forward(ctx, input_, fwd_info):
        move_params_to_cuda(fwd_info['pref_rid'])
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class PreFwdPostBwdOP(torch.autograd.Function):
    """
    A customized prefetch and offload operation.

    Args:
        input_: input tensor.
        fwd_info: information dict, which contains region indices that need to be prefetched, waited, or freed during forward pass.
        bwd_info: information dict, which contains region indices that need to be prefetched, waited, or freed during backward pass.
    """

    @staticmethod
    def forward(ctx, input_, fwd_info, bwd_info):
        ctx.bwd_info = bwd_info

        free_rid = fwd_info.get('free_rid', None)
        if free_rid:
            free_cuda_params(free_rid)

        wait_rid = fwd_info.get('wait_rid', None)
        if wait_rid:
            prefetch_event = OffloadManager.fwd_prefetch_event_map.get(
                wait_rid, None)
            if prefetch_event:
                prefetch_event.wait()
                # torch.cuda.current_stream().wait_event(prefetch_event)

        pref_rid = fwd_info.get('pref_rid', None)
        if pref_rid:
            with torch.cuda.stream(OffloadManager.h2d_stream):
                move_params_to_cuda(pref_rid)

            prefetch_event = torch.cuda.Event()
            prefetch_event.record(OffloadManager.h2d_stream)
            OffloadManager.fwd_prefetch_event_map[pref_rid] = prefetch_event

        return input_

    @staticmethod
    def backward(ctx, grad_output):
        free_rid = ctx.bwd_info.get('free_rid', None)

        if free_rid:
            free_cuda_params(free_rid)
            with torch.cuda.stream(OffloadManager.d2h_stream):
                move_grads_to_cpu(free_rid)

        wait_rid = ctx.bwd_info.get('wait_rid', None)
        if wait_rid:
            prefetch_event = OffloadManager.bwd_prefetch_event_map.get(
                wait_rid, None)
            if prefetch_event:
                prefetch_event.wait()
                # torch.cuda.current_stream().wait_event(prefetch_event)
            else:
                assert OffloadManager.region_list[wait_rid].is_syn
                move_params_to_cuda(wait_rid)

        pref_rid = ctx.bwd_info.get('pref_rid', None)
        if pref_rid:
            with torch.cuda.stream(OffloadManager.h2d_stream):
                move_params_to_cuda(pref_rid)

            prefetch_event = torch.cuda.Event()
            prefetch_event.record(OffloadManager.h2d_stream)
            OffloadManager.bwd_prefetch_event_map[pref_rid] = prefetch_event
        return grad_output, None, None


def convert_upload_to_action(tensor, fwd_info):
    '''
    Convert Upload operation into runtime action.

    Argument:
        tensor(torch.Tensor): input tensor.
        fwd_info(dict): information dict, which contains region indices that need to be prefetched, waited, or freed during forward pass.
    '''
    with torch._C.DisableTorchFunction():
        ret = PreForwardUpload.apply(tensor, fwd_info)
    return ret


def convert_fwd_prefetch_bwd_offload_to_action(tensor, fwd_info, bwd_info):
    '''
    Convert Prefetch and Offload operation into runtime action.

    Argument:
        tensor(torch.Tensor): input tensor.
        fwd_info(dict): information dict, which contains region indices that need to be prefetched, waited, or freed during forward pass.
        bwd_info(dict): information dict, which contains region indices that need to be prefetched, waited, or freed during backward pass.
    '''
    with torch._C.DisableTorchFunction():
        ret = PreFwdPostBwdOP.apply(tensor, fwd_info, bwd_info)
    return ret


def replace_node_users(orig_node: Node, inserted_node: Node, rep_user_nodes: List[Node] = None):
    user_list = list(orig_node.users.keys())
    if rep_user_nodes is not None:
        user_list = rep_user_nodes
    for user in user_list:
        if user == inserted_node:
            continue
        new_args = list(user.args)
        new_kwargs = dict(user.kwargs)
        # the origin node may be a positional argument or key word argument of user node
        if orig_node in new_args:
            # substitute the origin node with offload_apply_node
            new_args[new_args.index(orig_node)] = inserted_node
            user.args = tuple(new_args)
        elif str(orig_node) in new_kwargs:
            # substitute the origin node with offload_apply_node
            new_kwargs[str(orig_node)] = inserted_node
            user.kwargs = new_kwargs


def runtime_asyn_offload_apply_pass(gm: torch.fx.GraphModule, region_list: List[Region]):
    """
    This pass is used to add the asynchronous prefetch and offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph

    # upload parameters of the first region
    last_inp_node = tuple(mod_graph.nodes)[0]
    first_region_with_p = [
        region for region in region_list if region.param_size][0]
    fwd_info = {"pref_rid": first_region_with_p.r_id}
    if first_region_with_p.r_id < first_region_with_p.shared_rid:
        first_region_with_p.fp16_params = region_list[first_region_with_p.shared_rid].fp16_params
    with mod_graph.inserting_after(last_inp_node):
        upload_apply_node = mod_graph.create_node('call_function', convert_upload_to_action,
                                                  args=(last_inp_node, fwd_info))
    replace_node_users(last_inp_node, upload_apply_node)
    last_inp_node = upload_apply_node

    for r_idx, region in enumerate(region_list):
        # forward prefetch
        fwd_info = {}
        fwd_prefetch_region = region.fwd_prefetch_region
        if fwd_prefetch_region and requires_upload_p_in_fwd(fwd_prefetch_region):
            if fwd_prefetch_region.r_id < fwd_prefetch_region.shared_rid:
                fwd_prefetch_region.fp16_params = region_list[fwd_prefetch_region.shared_rid].fp16_params
            fwd_info['wait_rid'] = region.r_id
            fwd_info['pref_rid'] = fwd_prefetch_region.r_id

        # forward offload last region
        if r_idx > 0 and region_list[r_idx - 1].is_offload:
            fwd_info['free_rid'] = r_idx - 1

        # backward offload
        bwd_info = {}
        if region.param_size and (region.r_id <= region.shared_rid or (
                region.r_id > region.shared_rid and region_list[region.shared_rid].is_offload)):
            bwd_info['free_rid'] = region.r_id

        # backward prefetch
        if r_idx > 0 and region_list[r_idx - 1].is_offload:
            bwd_info['wait_rid'] = r_idx - 1
        if r_idx > 0 and region_list[r_idx - 1].bwd_prefetch_region:
            bwd_info['pref_rid'] = region_list[r_idx - 1].bwd_prefetch_region.r_id

        if fwd_info or bwd_info:
            with mod_graph.inserting_after(last_inp_node):
                new_node = mod_graph.create_node('call_function', convert_fwd_prefetch_bwd_offload_to_action,
                                                 args=(last_inp_node, fwd_info, bwd_info))
            replace_node_users(last_inp_node, new_node)

        last_inp_node = region.nodes[-1]

    # gm.graph.print_tabular()
    return gm