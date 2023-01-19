from typing import List
import torch
import torch.nn as nn
from torch.optim import Optimizer

from colossalai.tensor.param_op_hook import ColoParamOpHook
from colossalai.tensor.param_op_hook import ColoParamOpHookManager


from colossalai.logging import get_dist_logger
from colossalai.utils import clip_grad_norm_fp32, copy_tensor_parallel_attributes

from colossalai.amp.naive_amp import FP16Optimizer
from colossalai.amp.naive_amp.grad_scaler import BaseGradScaler

class AMPOptimizer(FP16Optimizer):

    def __init__(self,
                 optimizer: Optimizer,
                 grad_scaler: BaseGradScaler,
                 verbose: bool = False,
                 clip_grad_norm=0):

        # have a defaults for compatibility with pytorch optim
        self._optimizer = optimizer
        self._defaults = optimizer.defaults

        # fp16-related params
        assert isinstance(grad_scaler, BaseGradScaler)
        self._grad_scaler = grad_scaler
        self._found_overflow = torch.FloatTensor([0.0])
        self._dummy_overflow_buf = torch.IntTensor([0])

        # misc params
        self._clip_grad_max_norm = clip_grad_norm

        # we maintain three groups of parameters
        # so that the model can have a mixture
        # of fp16 and fp32 params
        # fp16_param_groups: the fp16 params of the model
        # fp32_master_param_groups: the fp32 params cast from the fp16 param of the model
        # fp32_param_groups: the fp32 params of the model
        # NOTE:
        # 1. fp16_param_groups and fp32_master_param_groups have one-to-one correspondence
        # 2. fp32_param_groups and fp16_param_groups are exclusive of each other
        self._fp16_param_groups = []
        self._fp32_master_param_groups = []
        self._fp32_param_groups = []

        # For all the groups in the original optimizer:
        for param_group in self._optimizer.param_groups:
            fp16_params = []
            fp32_master_params = []
            fp32_params = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    # float16 params:
                    if param.type() in ['torch.HalfTensor']:
                        fp16_params.append(param)

                        # Create a fp32 copy
                        fp32_param = param.detach().clone().float()

                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = fp32_param
                        fp32_master_params.append(fp32_param)

                        # Reset existing state dict key to the new main param.
                        if param in self._optimizer.state:
                            self._optimizer.state[fp32_param] = self._optimizer.state.pop(param)

                    # fp32 params.
                    elif param.type() == 'torch.FloatTensor':
                        fp32_params.append(param)
                    else:
                        raise TypeError('Expected parameter of type torch.cuda.FloatTensor '
                                        f'or torch.cuda.HalfTensor, but got {param.type()}')

            self._fp16_param_groups.append(fp16_params)
            self._fp32_master_param_groups.append(fp32_master_params)
            self._fp32_param_groups.append(fp32_params)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self._optimizer.load_state_dict(self._optimizer.state_dict())

        # log config
        self._logger = get_dist_logger()
        if verbose:
            self._logger.info(
                f"\n=========  FP16 Optimizer Config =========\n"
                f"Optimizer: {optimizer.__class__.__name__}\n"
                f"clip_grad_norm = {clip_grad_norm}\n"
                f"grad_scaler = {self._grad_scaler.__class__.__name__}"
                f"==========================================",
                ranks=[0])


class ParamUploadHook(ColoParamOpHook):

    def __init__(self) -> None:
        super().__init__()

    def pre_op(self, params):
        # move to cuda
        for p in params:
            p.data = p.data.to("cuda")

    def post_op(self, params):
        pass

    def pre_forward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_forward(self, params: List[torch.Tensor]) -> None:
        pass

    def pre_backward(self, params: List[torch.Tensor]) -> None:
        pass

    def post_backward(self, params: List[torch.Tensor]) -> None:
        pass


class GradOffloadHook():

    def __init__(self):
        self.grad_hook_list = []

    def grad_handle(self, grad):
        grad.data = grad.data.to("cpu")
        return grad

    def register_grad_hook(self, module: torch.nn.Module):
        for p in module.parameters():
            if p.requires_grad:
                self.grad_hook_list.append(p.register_hook(self.grad_handle))

    def remove_grad_hook(self):
        for hook in self.grad_hook_list:
            hook.remove()

class BasicOffloadModule:

    def __init__(self, model: nn.Module):
        self.model = model
        # self.param_upload_hook = ParamUploadHook()
        self.grad_offload_hook = GradOffloadHook()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _pre_forward(self):
        self.grad_offload_hook.register_grad_hook(self.model)

    def forward(self, *args, **kwargs):
        self.model.zero_grad(set_to_none=True)
        self._pre_forward()
        # with ColoParamOpHookManager.use_hooks(self.param_upload_hook):
        outputs = self.model(*args, **kwargs)
        return outputs

    def backward(self, loss):
        loss.backward()
        self._post_backward()

    def _post_backward(self):
        self.grad_offload_hook.remove_grad_hook()
