from dataclasses import dataclass
from typing import List
import torch
from torch.fx.node import Node
from colossalai.gemini.tensor_utils import alloc_storage, free_storage

from util import ModelParameters, GlobalCudaInfo, Region


class PreForwardUpload(torch.autograd.Function):
    """
    A customized upload operation which forward is parameter upload operation,
    backward is a parameter release operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices):
        # upload
        ctx.params_indices = params_indices
        for param_idx in params_indices:
            # print("PreForwardUpload", param_idx, ModelParameters.fp16_params[param_idx].data.shape)
            fp16_param = ModelParameters.fp16_params[param_idx]
            # print(input_.shape, input_.device, fp16_param.data.device)
            if fp16_param.data.device.type == "cpu":
                fp16_param.data = fp16_param.data.to("cuda")
            else:
                # print("PreForwardUpload", fp16_param.data.shape)
                alloc_storage(fp16_param.data)
                fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)
            # print(input_.shape, input_.device, fp16_param.data.device)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # release
        # print(ctx.params_indices, grad_output.shape, grad_output.device)
        for param_idx in ctx.params_indices:
            fp16_param = ModelParameters.fp16_params[param_idx]
            free_storage(fp16_param.data)
        return grad_output, None


class AftForwardOffloadSyn(torch.autograd.Function):
    """
    A customized offload operation which forward is parameter release operation,
    backward is a parameter upload operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices):
        # release
        ctx.params_indices = params_indices
        for param_idx in params_indices:
            free_storage(ModelParameters.fp16_params[param_idx].data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # upload
        for param_idx in ctx.params_indices:
            fp16_param = ModelParameters.fp16_params[param_idx]
            alloc_storage(fp16_param.data)
            fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)
        return grad_output, None


class PreBackwardPrefetch(torch.autograd.Function):
    """
    A customized prefetch operation which forward is parameter upload operation,
    backward is a parameter release operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices, node_id):
        ctx.params_indices = params_indices
        ctx.node_id = node_id
        # nothing to run
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # prefetch
        params_indices = ctx.params_indices
        with torch.cuda.stream(GlobalCudaInfo.h2d_stream):
            for param_idx in params_indices:
                fp16_param = ModelParameters.fp16_params[param_idx]
                alloc_storage(fp16_param.data)
                fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data, non_blocking=True)

        # insert event to record H2D stream
        prefetch_event = torch.cuda.Event()
        prefetch_event.record(GlobalCudaInfo.h2d_stream)
        GlobalCudaInfo.prefetch_event_map[ctx.node_id] = prefetch_event

        return grad_output, None, None

class AftForwardOffloadAsyn(torch.autograd.Function):
    """
    A customized offload operation which forward is parameter release operation,
    backward is a parameter upload operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices, syn_upload_flag, node_id):
        # offload
        ctx.params_indices = params_indices
        ctx.syn_upload_flag = syn_upload_flag
        ctx.node_id = node_id
        for param_idx in params_indices:
            print("AftForwardOffloadAsyn", param_idx, ModelParameters.fp16_params[param_idx].data.shape)
            free_storage(ModelParameters.fp16_params[param_idx].data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):

        # wait parameter prefetch
        prefetch_event = GlobalCudaInfo.prefetch_event_map.get(ctx.node_id, None)
        if prefetch_event is not None:
            assert isinstance(prefetch_event, torch.cuda.Event)
            prefetch_event.wait()
            # torch.cuda.current_stream().wait_event(prefetch_event)
        elif ctx.syn_upload_flag:
            for param_idx in ctx.params_indices:
                fp16_param = ModelParameters.fp16_params[param_idx]
                alloc_storage(fp16_param.data)
                fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)
        return grad_output, None, None, None


class PostForwardOperation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, offload_info, prefetch_info):
        # offload
        ctx.offload_info = offload_info
        ctx.prefetch_info = prefetch_info

        if offload_info is not None:
            for param_idx in offload_info['param_indices']:
                free_storage(ModelParameters.fp16_params[param_idx].data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):

        # wait current parameter prefetch
        if ctx.offload_info is not None:
            prefetch_event = GlobalCudaInfo.prefetch_event_map.get(ctx.offload_info['region_id'], None)
            if prefetch_event is not None:
                assert isinstance(prefetch_event, torch.cuda.Event)
                prefetch_event.wait()
                # torch.cuda.current_stream().wait_event(prefetch_event)
            elif ctx.offload_info['is_syn']:
                for param_idx in ctx.offload_info['param_indices']:
                    fp16_param = ModelParameters.fp16_params[param_idx]
                    alloc_storage(fp16_param.data)
                    fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)

        # prefetch following node parameter
        if ctx.prefetch_info is not None:
            # prefetch
            param_indices = ctx.prefetch_info['param_indices']
            with torch.cuda.stream(GlobalCudaInfo.h2d_stream):
                for param_idx in param_indices:
                    fp16_param = ModelParameters.fp16_params[param_idx]
                    alloc_storage(fp16_param.data)
                    fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data, non_blocking=True)

            # insert event to record H2D stream
            prefetch_event = torch.cuda.Event()
            prefetch_event.record(GlobalCudaInfo.h2d_stream)
            GlobalCudaInfo.prefetch_event_map[ctx.prefetch_info['region_id']] = prefetch_event

        return grad_output, None, None



def convert_upload_to_action(tensor, params_indices):
    '''
    Convert UploadSpec into runtime action, implement upload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return PreForwardUpload.apply(tensor, params_indices)

def convert_syn_offload_to_action(tensor, params_indices):
    '''
    Convert OffloadSpec into runtime action, implement offload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return AftForwardOffloadSyn.apply(tensor, params_indices)

def convert_prefetch_to_action(tensor, params_indices, node_id):
    '''
    Convert UploadSpec into runtime action, implement upload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return PreBackwardPrefetch.apply(tensor, params_indices, node_id)

def convert_asyn_offload_to_action(tensor, params_indices, syn_upload_flag=False, node_id=0):
    '''
    Convert OffloadSpec into runtime action, implement offload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return AftForwardOffloadAsyn.apply(tensor, params_indices, syn_upload_flag, node_id)


def convert_asyn_offload_prefetch_to_action(tensor, offload_info=None, prefetch_info=None):
    '''
    Convert OffloadSpec into runtime action, implement offload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return PostForwardOperation.apply(tensor, offload_info, prefetch_info)


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


def runtime_syn_offload_apply_pass(gm: torch.fx.GraphModule, region_list: List[Region]):
    """
    This pass is used to add the offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph
    for r_idx, region in enumerate(region_list):
        assert r_idx == region.r_id
        if region.param_size > 0:
            param_indices = region.param_indices
            assert isinstance(param_indices, list)

            if r_idx == 0:
                last_inp_node = tuple(mod_graph.nodes)[0]
            else:
                last_inp_node = region_list[r_idx-1].nodes[-1]

            # mod_graph.inserting_before(node) maybe invalid
            with mod_graph.inserting_after(last_inp_node):
                upload_apply_node = mod_graph.create_node('call_function', convert_upload_to_action,
                                                          args=(last_inp_node, param_indices))
            replace_node_users(last_inp_node, upload_apply_node)

            if region.is_offload:
                node = region.nodes[-1]
                with mod_graph.inserting_after(node):
                    offload_apply_node = mod_graph.create_node('call_function', convert_syn_offload_to_action,
                                                               args=(node, param_indices))
                replace_node_users(node, offload_apply_node)
    return gm


def runtime_asyn_offload_apply_pass(gm: torch.fx.GraphModule, region_list: List[Region]):
    """
    This pass is used to add the asynchronous offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph
    # nodes = tuple(mod_graph.nodes)
    no_insert_after_node_list = []

    for r_idx, region in enumerate(region_list):

        if region.param_size > 0:
            param_indices = region.param_indices
            assert isinstance(param_indices, list)

            def _extract_last_input_node(cur_node):
                for n in list(cur_node._input_nodes.keys()).__reversed__():
                    if (n.op == "get_attr") or (n in no_insert_after_node_list):
                        continue
                    return n
                cur_user_nodes = list(cur_node.users.keys())
                # print("last_inp_node", cur_node, list(node._input_nodes.keys()), len(cur_user_nodes))
                # mod_graph.print_tabular()
                assert len(cur_user_nodes) == 1
                no_insert_after_node_list.append(cur_node)
                return _extract_last_input_node(cur_user_nodes[0])

            if r_idx == 0:
                last_inp_node = tuple(mod_graph.nodes)[0]
            else:
                last_inp_node = region_list[r_idx-1].nodes[-1]

            with mod_graph.inserting_after(last_inp_node):
                upload_apply_node = mod_graph.create_node('call_function', convert_upload_to_action,
                                                          args=(last_inp_node, param_indices))
            replace_node_users(last_inp_node, upload_apply_node)

        offload_info = None
        prefetch_info = None
        if region.is_offload:
            offload_info = {}
            offload_info['param_indices'] = region.param_indices
            offload_info['region_id'] = region.r_id
            offload_info['is_syn'] = region.is_syn
        region_to_prefetch = region.region_to_prefetch
        if region_to_prefetch is not None:
            prefetch_info = {}
            prefetch_info['param_indices'] = region_to_prefetch.param_indices
            prefetch_info['region_id'] = region_to_prefetch.r_id
        if (offload_info is not None) or (prefetch_info is not None):
            node = region.nodes[-1]
            with mod_graph.inserting_after(node):
                new_node = mod_graph.create_node('call_function', convert_asyn_offload_prefetch_to_action,
                                                            args=(node, offload_info, prefetch_info))
            replace_node_users(node, new_node)
            if (node.op == "get_attr") or (node in no_insert_after_node_list):
                no_insert_after_node_list.append(new_node)

    # gm.graph.print_tabular()
    # print(len(ModelParameters.fp16_params), ModelParameters.param_idx)
    return gm