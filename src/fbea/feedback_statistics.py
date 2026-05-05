"""
Feedback statistics and probability updates corresponding to Algorithm 4.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from .constants import EPS


@dataclass
class FeedbackStatistics:
    new_1: int = 0
    new_2: int = 0
    w_11: int = 0
    w_12: int = 0
    w_13: int = 0
    w_21: int = 0
    w_22: int = 0
    w_23: int = 0
    q_bar_1: float = 0.5
    q_bar_2: float = 0.5
    w_bar_1: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    w_bar_2: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)

    def reset_per_generation(self) -> None:
        """
        Reset per-generation feedback counters.

        Algorithm 4 sets new1/new2 and w11..w23 to zero at the start of each generation.
        """
        self.new_1 = 0
        self.new_2 = 0
        self.w_11 = 0
        self.w_12 = 0
        self.w_13 = 0
        self.w_21 = 0
        self.w_22 = 0
        self.w_23 = 0

    def refresh_probabilities(self) -> None:
        """
        Recompute Q-bar and W-bar from current generation counters.
        """
        total_crossover_usage = self.new_1 + self.new_2
        if total_crossover_usage == 0:
            self.q_bar_1 = 0.5
            self.q_bar_2 = 0.5
        else:
            # Paper equations (26)-(27): higher successful usage in this generation
            # increases next-generation operator probability.
            self.q_bar_1 = 0.25 + 0.5 * (self.new_1 / max(total_crossover_usage, EPS))
            self.q_bar_2 = 0.25 + 0.5 * (self.new_2 / max(total_crossover_usage, EPS))

        total_sched = self.w_11 + self.w_12 + self.w_13
        if total_sched == 0:
            self.w_bar_1 = (1 / 3, 1 / 3, 1 / 3)
        else:
            w1 = 1 / 6 + 0.5 * (self.w_11 / total_sched)
            w2 = 1 / 6 + 0.5 * (self.w_12 / total_sched)
            w3 = 1 / 6 + 0.5 * (self.w_13 / total_sched)
            self.w_bar_1 = (w1, w2, w3)

        total_mach = self.w_21 + self.w_22 + self.w_23
        if total_mach == 0:
            self.w_bar_2 = (1 / 3, 1 / 3, 1 / 3)
        else:
            w1 = 1 / 6 + 0.5 * (self.w_21 / total_mach)
            w2 = 1 / 6 + 0.5 * (self.w_22 / total_mach)
            w3 = 1 / 6 + 0.5 * (self.w_23 / total_mach)
            self.w_bar_2 = (w1, w2, w3)

    def update_crossover_stats(self, operator_info) -> None:
        if operator_info.used_gpx:
            self.new_1 += 1
        if operator_info.used_pox:
            self.new_2 += 1

    def update_mutation_stats(self, operator_info) -> None:
        if operator_info.used_insert_mutation:
            self.w_11 += 1
        if operator_info.used_swap_mutation:
            self.w_12 += 1
        if operator_info.used_inverse_mutation:
            self.w_13 += 1
        if operator_info.used_change1_mutation:
            self.w_21 += 1
        if operator_info.used_change2_mutation:
            self.w_22 += 1
        if operator_info.used_change3_mutation:
            self.w_23 += 1


def calculate_delta_i(population, other_population, archive) -> float:
    pop_size = max(population.size, EPS)
    if archive is None:
        return 0.5
    if population.population_id == 1:
        opt_ratio_i = archive.opt_count_p1 / pop_size
        opt_ratio_other = archive.opt_count_p2 / max(other_population.size, EPS)
    else:
        opt_ratio_i = archive.opt_count_p2 / pop_size
        opt_ratio_other = archive.opt_count_p1 / max(other_population.size, EPS)
    denom = opt_ratio_i + opt_ratio_other
    if denom <= EPS:
        return 0.5
    return opt_ratio_i / denom


def calculate_eta_i(population, other_population) -> float:
    rate_i = population.renew_i / max(population.size, EPS)
    rate_other = other_population.renew_i / max(other_population.size, EPS)
    denom = rate_i + rate_other
    if denom <= EPS:
        return 0.5
    return rate_i / denom


def calculate_u_values(population) -> Tuple[float, float, float]:
    if population.size == 0:
        return 0.0, 0.0, 0.0
    sum_f1 = sum(1 / max(sol.makespan._c1(), EPS) for sol in population.solutions)
    sum_f2 = sum(1 / max(sol.agreement._c1(), EPS) for sol in population.solutions)
    sum_f3 = sum(1 / max(sol.energy._c1(), EPS) for sol in population.solutions)
    size = population.size
    return sum_f1 / size, sum_f2 / size, sum_f3 / size


def update_population_u_values(population_a, population_b) -> None:
    population_a.u_values = calculate_u_values(population_a)
    population_b.u_values = calculate_u_values(population_b)


def calculate_new_population_sizes(population_a, population_b, delta_values, eta_values) -> Tuple[int, int]:
    total_size = population_a.size + population_b.size
    if total_size == 0:
        return 0, 0
    delta_1, delta_2 = delta_values
    eta_1, eta_2 = eta_values
    u11, u12, u13 = population_a.u_values
    u21, u22, u23 = population_b.u_values

    def safe_frac(num, denom):
        return num / max(denom, EPS)

    term_p1 = (
        safe_frac(u11, u11 + u21)
        + safe_frac(u12, u12 + u22)
        + safe_frac(u13, u13 + u23)
    ) / 3

    n1_real = total_size / 4 + (total_size / 8) * (
        delta_1 + eta_1 + term_p1 + population_a.size / max(total_size, EPS)
    )
    n1_new = math.ceil(n1_real / 2.0) * 2
    n1_new = max(n1_new, total_size // 4)
    n1_new = min(n1_new, total_size - 2)
    if n1_new < 2:
        n1_new = 2
    if n1_new % 2 != 0:
        if n1_new + 1 <= total_size - 2:
            n1_new += 1
        else:
            n1_new -= 1
    n2_new = total_size - n1_new
    if n2_new < 2:
        n2_new = 2
        n1_new = total_size - n2_new
    if n2_new % 2 != 0:
        n2_new += 1
        n1_new -= 1
    return n1_new, n2_new
