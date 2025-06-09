# quantKit/financial/__init__.py

"""Financial‐math utilities: interest‐timevalue and return‐calculation functions.

"""

__all__ = [
    "simple_interest",
    "discrete_compound_interest",
    "continuous_compound_interest",
    "simple_returns",
    "log_returns",
    "multi_period_simple_returns",
    "multi_period_log_returns",
]

from .interest import (
    simple_interest,
    discrete_compound_interest,
    continuous_compound_interest,
)
from .returns import (
    simple_returns,
    log_returns,
    multi_period_simple_returns,
    multi_period_log_returns,
)
