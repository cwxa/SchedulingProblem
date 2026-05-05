"""
Global random number generator management utilities.

All stochastic components must draw randomness from the RNG provided by this
module to ensure repeatability across the entire algorithm.
"""

from __future__ import annotations

import random
from typing import Optional

_global_seed: Optional[int] = None
_global_rng: random.Random = random.Random()


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed and reinitialise the shared RNG.

    Parameters
    ----------
    seed:
        Integer seed to initialise the global RNG.
    """
    global _global_seed, _global_rng
    _global_seed = int(seed)
    _global_rng = random.Random(_global_seed)


def reset_global_seed() -> None:
    """
    Reset the global random seed and RNG to a non-deterministic state.
    """
    global _global_seed, _global_rng
    _global_seed = None
    _global_rng = random.Random()


def get_random_seed() -> Optional[int]:
    """
    Return the currently configured random seed, if any.
    """
    return _global_seed


def get_rng() -> random.Random:
    """
    Return the shared RNG instance.
    """
    return _global_rng
