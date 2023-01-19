from dataclasses import dataclass
from enum import Enum



class SolverOption(Enum):
    """
    This enum class is to define the solver option.
    """
    SYNC = 0
    ASYNC = 1