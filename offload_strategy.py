from dataclasses import dataclass
from torch.fx.node import Node
from colossalai.auto_parallel.tensor_shard.sharding_strategy import StrategiesVector


class SystemConfig:
    BANDWIDTH = 1.2e9
    COMPUTE_POWER = 1.9e12
