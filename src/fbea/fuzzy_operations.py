"""
Triangular fuzzy number utilities used by the FBEA implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from .constants import EPS


@dataclass(frozen=True, slots=True)
class TriangularFuzzyNumber:
    """
    Triangular fuzzy number (TFN) defined by (l, m, r) with l <= m <= r.
    """

    l: float
    m: float
    r: float
    _c1_cache: float = field(init=False, repr=False)
    _c3_cache: float = field(init=False, repr=False)

    @staticmethod
    def _new_unchecked(l: float, m: float, r: float) -> "TriangularFuzzyNumber":
        obj = object.__new__(TriangularFuzzyNumber)
        object.__setattr__(obj, "l", l)
        object.__setattr__(obj, "m", m)
        object.__setattr__(obj, "r", r)
        object.__setattr__(obj, "_c1_cache", (l + 2 * m + r) / 4.0)
        object.__setattr__(obj, "_c3_cache", r - l)
        return obj

    def __post_init__(self) -> None:
        if not (self.l <= self.m <= self.r):
            raise ValueError(
                f"Invalid triangular fuzzy number: requires l <= m <= r, got {self.l}, {self.m}, {self.r}"
            )
        object.__setattr__(self, "_c1_cache", (self.l + 2 * self.m + self.r) / 4.0)
        object.__setattr__(self, "_c3_cache", self.r - self.l)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"({self.l:.3f},{self.m:.3f},{self.r:.3f})"

    __repr__ = __str__

    def to_tuple(self) -> Tuple[float, float, float]:
        """
        Return the TFN as a tuple.
        """
        return self.l, self.m, self.r

    def add(self, other: "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":
        """
        Return the sum of two TFNs via component-wise addition.
        """
        # Addition preserves ordering for valid TFNs, so re-validation is redundant.
        return TriangularFuzzyNumber._new_unchecked(
            self.l + other.l, self.m + other.m, self.r + other.r
        )

    def __add__(self, other: "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":  # pragma: no cover - wrapper
        return self.add(other)

    def mul(self, scalar: float) -> "TriangularFuzzyNumber":
        """
        Multiply by a scalar, reversing endpoints if the scalar is negative.
        """
        if scalar >= 0:
            l, m, r = scalar * self.l, scalar * self.m, scalar * self.r
        else:
            l, m, r = scalar * self.r, scalar * self.m, scalar * self.l
        return TriangularFuzzyNumber._new_unchecked(l, m, r)

    def __mul__(self, scalar: float) -> "TriangularFuzzyNumber":  # pragma: no cover - wrapper
        return self.mul(scalar)

    def __rmul__(self, scalar: float) -> "TriangularFuzzyNumber":  # pragma: no cover - wrapper
        return self.mul(scalar)

    def max_with(self, other: "TriangularFuzzyNumber") -> "TriangularFuzzyNumber":
        """
        Component-wise maximum following Sakawa & Mori approximation.
        Fast-path: if one TFN dominates component-wise, return it directly.
        """
        if self.l >= other.l and self.m >= other.m and self.r >= other.r:
            return self
        if other.l >= self.l and other.m >= self.m and other.r >= self.r:
            return other
        return TriangularFuzzyNumber._new_unchecked(
            max(self.l, other.l), max(self.m, other.m), max(self.r, other.r)
        )

    def _c1(self) -> float:
        return self._c1_cache

    def _c2(self) -> float:
        return self.m

    def _c3(self) -> float:
        return self._c3_cache

    def gt(self, other: "TriangularFuzzyNumber") -> bool:
        """
        Strict greater-than based on c1 -> c2 -> c3 lexicographic comparison.
        """
        self_c1 = self._c1_cache
        other_c1 = other._c1_cache
        if self_c1 > other_c1 + EPS:
            return True
        if self_c1 < other_c1 - EPS:
            return False
        self_c2 = self.m
        other_c2 = other.m
        if self_c2 > other_c2 + EPS:
            return True
        if self_c2 < other_c2 - EPS:
            return False
        return self._c3_cache > other._c3_cache + EPS

    def lt(self, other: "TriangularFuzzyNumber") -> bool:
        return other.gt(self)

    def leq(self, other: "TriangularFuzzyNumber") -> bool:
        return not self.gt(other)

    def geq(self, other: "TriangularFuzzyNumber") -> bool:
        return not self.lt(other)
