from typing import List
from enum import Enum
import torch
import torch.nn as nn
from torch.optim import Optimizer

from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.amp.naive_amp import FP16Optimizer
from colossalai.amp.naive_amp.grad_scaler import BaseGradScaler, DynamicGradScaler
from colossalai.nn.parallel.data_parallel import _cast_float
from colossalai.gemini.tensor_utils import alloc_storage, free_storage

from util import OffloadManager


class GradOffloadHook:

    def __init__(self, is_syn_offload=True):
        self.grad_hook_list = []
        self.is_syn_offload = is_syn_offload

    def grad_handle(self, grad):

        cpu_grad = torch.empty(grad.shape, dtype=torch.half, device='cpu', pin_memory=True)

        if self.is_syn_offload:
            cpu_grad.copy_(grad.data)
            grad.data = cpu_grad
        else:
            with torch.cuda.stream(OffloadManager.d2h_stream):
                cpu_grad.copy_(grad.data, non_blocking=True)
                grad.data = cpu_grad

        return grad

    def register_grad_hook(self, module: torch.nn.Module):
        for p in module.parameters():
            if p.requires_grad:
                self.grad_hook_list.append(p.register_hook(self.grad_handle))

    def remove_grad_hook(self):
        for hook in self.grad_hook_list:
            hook.remove()


class BasicOffloadModule:

    def __init__(self, model: nn.Module, is_syn):
        self.model = model
        # self.grad_offload_hook = GradOffloadHook(is_syn)

        for p in model.parameters():
            # p.data = p.data.to(torch.half)
            fp32_param = p.detach().clone().float().pin_memory()
            OffloadManager.param_fp16_to_fp32[p] = fp32_param
            p.data = p.data.to('cuda')
            free_storage(p.data)

        self._cast_buffers()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _pre_forward(self):
        # self.grad_offload_hook.register_grad_hook(self.model)
        OffloadManager.param_fp16_to_grad.clear()

    def forward(self, *args, **kwargs):
        args, kwargs = _cast_float(args, torch.half), _cast_float(kwargs, torch.half)
        self.model.zero_grad(set_to_none=True)
        self._pre_forward()
        outputs = self.model(*args, **kwargs)
        return outputs

    def backward(self, loss):
        loss.backward()
        self._post_backward()

    def _post_backward(self):
        # self.grad_offload_hook.remove_grad_hook()
        torch.cuda.synchronize()
        for p in self.model.parameters():
            if p not in OffloadManager.param_fp16_to_grad:
                free_storage(p.data)
                cpu_grad = torch.empty(p.data.shape, dtype=torch.half, pin_memory=True)
                cpu_grad.copy_(p.grad, non_blocking=True)
                OffloadManager.param_fp16_to_grad[p] = cpu_grad
                p.grad = None

        OffloadManager.fwd_prefetch_event_map.clear()
        OffloadManager.bwd_prefetch_event_map.clear()

    def _cast_buffers(self):
        for buffer in self.model.buffers():
            buffer.data = buffer.cuda()
            # if torch.is_floating_point(buffer):
            #     buffer.data = buffer.half()


class AMPOptimizer(FP16Optimizer):

    def __init__(self,
                 optimizer: Optimizer,
                 verbose: bool = False,
                 clip_grad_norm=0,
                 initial_scale: float = 2 ** 32,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2 ** 32):

        # have a defaults for compatibility with pytorch optim
        self._optimizer = optimizer
        self._defaults = optimizer.defaults

        # fp16-related params
        self._grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                              min_scale=min_scale,
                                              growth_factor=growth_factor,
                                              backoff_factor=backoff_factor,
                                              growth_interval=growth_interval,
                                              hysteresis=hysteresis,
                                              max_scale=max_scale)

        self._found_overflow = torch.FloatTensor([0.0])
        self._dummy_overflow_buf = torch.IntTensor([0])

        # misc params
        self._clip_grad_max_norm = clip_grad_norm

        self._dp_process_group = None
        self._mp_process_group = None

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
                    if param.type() in ['torch.HalfTensor', 'torch.cuda.HalfTensor']:
                        fp16_params.append(param)

                        # Create a fp32 copy
                        fp32_param = OffloadManager.param_fp16_to_fp32[param]

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
                        raise TypeError('Expected parameter of type torch.FloatTensor '
                                        f'or torch.HalfTensor, but got {param.type()}')

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

    def _unscale_grads(self):
        for group in self._get_fp32_param_groups_to_update():
            for p in group:
                if p.grad is not None:
                    p.grad.data.div_(self.loss_scale.to('cpu'))

    def _assign_grad_to_fp32_master_param(self):
        # This only needs to be done for the float16 group.
        for fp16_param_group, fp32_master_param_group in zip(self._fp16_param_groups, self._fp32_master_param_groups):
            for fp16_param, fp32_param in zip(fp16_param_group, fp32_master_param_group):
                if fp16_param in OffloadManager.param_fp16_to_grad:
                    assert fp16_param.grad is None
                    fp32_param.grad = OffloadManager.param_fp16_to_grad[fp16_param].float()
                    print(fp32_param.grad.shape)
                    # clear unneeded grad on fp16 param
                    # fp16_param.grad = None

    def _update_fp16_param_from_fp32_param(self):
        pass


class OptimState(Enum):
    SCALED = 0
    UNSCALED = 1


class AMPOptimizer_v2(ColossalaiOptimizer):

    def __init__(self,
                 optimizer: Optimizer,
                 initial_scale: float = 2 ** 32,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2 ** 32,
                 clipping_norm: float = 0.0,
                 norm_type: float = 2.0, ):

        # have a defaults for compatibility with pytorch optim
        self._optimizer = optimizer
        self._defaults = optimizer.defaults

        # fp16-related params
        self._grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                              min_scale=min_scale,
                                              growth_factor=growth_factor,
                                              backoff_factor=backoff_factor,
                                              growth_interval=growth_interval,
                                              hysteresis=hysteresis,
                                              max_scale=max_scale)

        self._found_overflow = torch.FloatTensor([0.0])
        self._dummy_overflow_buf = torch.IntTensor([0])

        # misc params
        self._clip_grad_max_norm = clip_grad_norm

        self._dp_process_group = None
        self._mp_process_group = None

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
                        fp32_param = param.detach().clone().float().pin_memory()
                        OffloadManager.param_fp16_to_fp32[param] = fp32_param

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
                        raise TypeError('Expected parameter of type torch.FloatTensor '
                                        f'or torch.HalfTensor, but got {param.type()}')

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

    def _unscale_grads(self):
        for group in self._get_fp32_param_groups_to_update():
            for p in group:
                if p.grad is not None:
                    p.grad.data.div_(self.loss_scale.to('cpu'))

    def _assign_grad_to_fp32_master_param(self):
        # This only needs to be done for the float16 group.
        for fp16_param_group, fp32_master_param_group in zip(self._fp16_param_groups, self._fp32_master_param_groups):
            for fp16_param, fp32_param in zip(fp16_param_group, fp32_master_param_group):
                if fp16_param.grad is not None:
                    # fp32_param.grad = fp16_param.grad.float()
                    fp32_param.grad = fp16_param.grad
                    # clear unneeded grad on fp16 param
                    fp16_param.grad = None

    def _update_fp16_param_from_fp32_param(self):
        # fp16_param_data = []
        # fp32_master_param_data = []
        # for fp16_group, fp32_group in zip(self._fp16_param_groups, self._fp32_master_param_groups):
        #     for fp16_param, fp32_param in zip(fp16_group, fp32_group):
        #         fp16_param_data.append(fp16_param.data)
        #         fp32_master_param_data.append(fp32_param.data)
        # _multi_tensor_copy_this_to_that(this=fp32_master_param_data,
        #                                 that=fp16_param_data,
        #                                 overflow_buf=self._dummy_overflow_buf)

        pass

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(self.module.overflow_counter)

        return self._found_overflow.item() > 0

    def _clear_global_norm(self) -> None:
        for c16 in self.chunk16_set:
            c16.l2_norm = None

    def _calc_global_norm(self) -> float:
        norm_sqr: float = 0.0
        group_to_norm = dict()
        for c16 in self.chunk16_set:
            assert c16.l2_norm is not None

            if c16.is_gathered:
                norm_sqr += c16.l2_norm
            else:
                # this chunk is sharded, use communication to collect total norm
                if c16.torch_pg not in group_to_norm:
                    group_to_norm[c16.torch_pg] = 0.0
                group_to_norm[c16.torch_pg] += c16.l2_norm

            c16.l2_norm = None  # clear l2 norm

        comm_buffer = torch.zeros(1, dtype=torch.float, device=get_current_device())
        for group, part_norm in group_to_norm.items():
            comm_buffer.fill_(part_norm)
            norm_sqr += comm_buffer.item()

        global_norm = math.sqrt(norm_sqr)
        return global_norm

    def _get_combined_scale(self):
        loss_scale = 1

        if self.optim_state == OptimState.SCALED:
            loss_scale = self.loss_scale
            self.optim_state = OptimState.UNSCALED

        combined_scale = loss_scale
        if self.clipping_flag:
            total_norm = self._calc_global_norm()
            clip = ((total_norm / loss_scale) + 1e-6) / self.max_norm
            if clip > 1:
                combined_scale = clip * loss_scale

        if combined_scale == 1:
            return -1
        else:
            return combined_scale

    @property
    def loss_scale(self):
        return self.grad_scaler.scale.item()

    def zero_grad(self, *args, **kwargs):
        self.module.overflow_counter = 0
        return self.optim.zero_grad(set_to_none=True)

    def step(self, *args, **kwargs):
        # Copy gradients from model params to main params.
        self._assign_grad_to_fp32_master_param()

        found_inf = self._check_overflow()
        if found_inf:
            self.optim_state = OptimState.UNSCALED  # no need to unscale grad
            self.grad_scaler.update(found_inf)  # update gradient scaler
            self._logger.info(f'Found overflow. Skip step')
            self._clear_global_norm()  # clear recorded norm
            self.zero_grad()  # reset all gradients
            self._update_fp16_params()
            return

        # get combined scale. combined scale = loss scale * clipping norm
        # so that gradient = gradient / combined scale
        combined_scale = self._get_combined_scale()
        self.grad_scaler.update(found_inf)

        ret = self.optim.step(div_scale=combined_scale, *args, **kwargs)
        self._register_states()
        self.zero_grad()
        self._update_fp16_params()
        return ret

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float, norm_type: float = 2.0):
        raise NotImplementedError

    def backward(self, loss: torch.Tensor):
        loss = self.loss_scale * loss
        self.optim_state = OptimState.SCALED
        self.module.backward(loss)