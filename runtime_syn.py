from dataclasses import dataclass
from typing import List
import torch
from torch.fx.node import Node
from colossalai.gemini.tensor_utils import alloc_storage, free_storage

from util import *


class PreForwardUpload(torch.autograd.Function):
    """
    A customized upload operation which forward is parameter upload operation,
    backward is a parameter release operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices, release_p_flag):
        # upload
        ctx.params_indices = params_indices
        ctx.release_p_flag = release_p_flag
        for param_idx in params_indices:
            # print("PreForwardUpload", param_idx, ModelParameters.fp16_params[param_idx].data.shape)
            fp16_param = ModelParameters.fp16_params[param_idx]
            # print(input_.shape, input_.device, fp16_param.data.device)
            if fp16_param.data.device.type == "cpu":
                fp16_param.data = fp16_param.data.to("cuda")
            else:
                # print("PreForwardUpload", fp16_param.data.shape)
                alloc_storage(fp16_param.data)
                fp16_param.data.copy_(
                    ModelParameters.fp32_master_params[param_idx].data)
            # print(input_.shape, input_.device, fp16_param.data.device)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # release
        # print(ctx.params_indices, grad_output.shape, grad_output.device)
        if ctx.release_p_flag:
            for param_idx in ctx.params_indices:
                fp16_param = ModelParameters.fp16_params[param_idx]
                free_storage(fp16_param.data)
        return grad_output, None, None



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
            fp16_param.data.copy_(
                ModelParameters.fp32_master_params[param_idx].data)
        return grad_output, None




def convert_upload_to_action(tensor, params_indices, release_p_flag):
    '''
    Convert UploadSpec into runtime action, implement upload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return PreForwardUpload.apply(tensor, params_indices, release_p_flag)


def convert_syn_offload_to_action(tensor, params_indices):
    '''
    Convert OffloadSpec into runtime action, implement offload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return AftForwardOffloadSyn.apply(tensor, params_indices)



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
        # upload parameter
        if requires_upload_p_in_fwd(region):
            if region.region_shared_param is not None:
                param_indices = region.region_shared_param.param_indices
            else:
                param_indices = region.param_indices
            assert isinstance(param_indices, list)

            if r_idx == 0:
                last_inp_node = tuple(mod_graph.nodes)[0]
            else:
                last_inp_node = region_list[r_idx - 1].nodes[-1]
            release_p_flag = requires_release_p_in_bwd(region)

            # mod_graph.inserting_before(node) maybe invalid
            with mod_graph.inserting_after(last_inp_node):
                upload_apply_node = mod_graph.create_node('call_function', convert_upload_to_action,
                                                          args=(last_inp_node, param_indices, release_p_flag))
            replace_node_users(last_inp_node, upload_apply_node)

            if region.is_offload:
                node = region.nodes[-1]
                with mod_graph.inserting_after(node):
                    offload_apply_node = mod_graph.create_node('call_function', convert_syn_offload_to_action,
                                                               args=(node, param_indices))
                replace_node_users(node, offload_apply_node)
    return gm