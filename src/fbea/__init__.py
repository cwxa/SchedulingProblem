"""
FBEA (Feedback-Based Evolutionary Algorithm) reproduction package.

This package provides a faithful implementation of the components described in
the user's reproduction checklist, including fuzzy number operations, instance
loading, scheduling heuristics, evolutionary operators, and evaluation
utilities. Modules are exposed here for convenient access.
"""

from .random_manager import (
    set_global_seed,
    get_random_seed,
    get_rng,
    reset_global_seed,
)
from .evaluation_counter import (
    MAX_EVALUATIONS,
    reset_global_counter,
    increment_evaluation_counter,
    is_termination_reached,
    calculate_objectives_with_count,
    get_evaluation_count,
)

__all__ = [
    "set_global_seed",
    "get_random_seed",
    "get_rng",
    "reset_global_seed",
    "MAX_EVALUATIONS",
    "reset_global_counter",
    "increment_evaluation_counter",
    "is_termination_reached",
    "calculate_objectives_with_count",
    "get_evaluation_count",
]
