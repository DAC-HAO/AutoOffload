import random
import argparse
import time

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from colossalai.fx.profiler import parameter_size
from colossalai.utils import get_current_device

from mem_offload_optimize import memory_optimization
from model_utils import *

parser = argparse.ArgumentParser(description="offload testing")
parser.add_argument("-mn", type=str, default="simplenet",
                    help="model name")
parser.add_argument("-ms", type=float, default=32, help="memory budget (MB)")
parser.add_argument('-is_syn', action='store_true', help='If true, offload is performed synchronously.')
args = parser.parse_args()

# build model
get_components_func = non_distributed_component_funcs.get_callable(args.mn)
model_builder, data_gen = get_components_func()
data_args = data_gen(device="cpu")
model = model_builder()

param_size = parameter_size(model) / 1024 ** 2
print("init param size: ", param_size)
model = memory_optimization(model, data_args, 1024 * 1024 * args.ms, args.is_syn)
wrap_fn = lambda x: x.to("cuda") if isinstance(x, torch.Tensor) else x
data_args = tree_map(wrap_fn, data_args)

# print("model buffer.....")
for n, buff in model.model.named_buffers():
    # print(n, buff.data.shape)
    buff.data = buff.data.cuda()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start_time = time.time()

# prof = torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/offload'),
#         record_shapes=True,
#         with_stack=True)
# prof.start()
# prof.step()
loss = torch.sum(model(**data_args))
print(loss)
loss.backward()
# prof.step()
# prof.stop()

torch.cuda.synchronize()

exec_time = time.time() - start_time
runtime_peak_mem_alc = torch.cuda.max_memory_allocated() / 1024 ** 2
runtime_peak_mem_res = torch.cuda.max_memory_reserved() / 1024 ** 2
print(
    f'|exec_time={exec_time:.3f} s | param_size={param_size:.3f} MB '
    f'| runtime_peak_mem_alc={runtime_peak_mem_alc:.3f} MB| runtime_peak_mem_res={runtime_peak_mem_res:.3f} MB|'
)
