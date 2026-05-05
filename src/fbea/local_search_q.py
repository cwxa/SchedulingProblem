"""
Q-learning policy for local-search operator transition in FBEA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

MacroAction = Tuple[str, int | None, int | None]

# (mode, scheduling_operator_idx, machine_operator_idx)
# scheduling_operator_idx: 0=insert, 1=swap, 2=inverse
# machine_operator_idx:    0=change1, 1=change2, 2=change3
MACRO_ACTIONS: Tuple[MacroAction, ...] = (
    ("scheduling", 0, None),
    ("scheduling", 1, None),
    ("scheduling", 2, None),
    ("machine", None, 0),
    ("machine", None, 1),
    ("machine", None, 2),
    ("both", 0, 0),
    ("both", 0, 1),
    ("both", 0, 2),
    ("both", 1, 0),
    ("both", 1, 1),
    ("both", 1, 2),
    ("both", 2, 0),
    ("both", 2, 1),
    ("both", 2, 2),
)


@dataclass
class LocalSearchQConfig:
    # Paper-style defaults.
    alpha: float = 0.1
    gamma: float = 0.9
    greedy_start: float = 0.85
    greedy_end: float = 0.85
    greedy_power: float = 1.0
    reward_dominate: float = 2.0
    reward_nondominated: float = 1.0
    reward_dominated: float = 0.0

    # Backward-compatible aliases for earlier CLI/config usage.
    epsilon_start: float | None = None
    epsilon_end: float | None = None
    epsilon_power: float | None = None
    reward_clip: float | None = None
    success_ema_beta: float | None = None
    state_design: str = "two_step_eff"
    eff_w_makespan: float | None = None
    eff_w_energy: float | None = None
    eff_w_agreement: float | None = None
    eff_neutral_band: float | None = None


class LocalSearchQAgent:
    ACTION_COUNT = len(MACRO_ACTIONS)

    def __init__(self, config: LocalSearchQConfig | None = None) -> None:
        self.config = config or LocalSearchQConfig()
        self._normalize_config_aliases()
        self.q_table: List[List[float]] = [
            [0.0 for _ in range(self.ACTION_COUNT)] for _ in range(self.ACTION_COUNT)
        ]
        self.visits: List[List[int]] = [
            [0 for _ in range(self.ACTION_COUNT)] for _ in range(self.ACTION_COUNT)
        ]
        self.previous_action_idx: int | None = None

    def _normalize_config_aliases(self) -> None:
        # Old config interpreted epsilon as exploration probability;
        # greedy_factor = 1 - epsilon.
        if self.config.epsilon_start is not None:
            self.config.greedy_start = 1.0 - float(self.config.epsilon_start)
        if self.config.epsilon_end is not None:
            self.config.greedy_end = 1.0 - float(self.config.epsilon_end)
        if self.config.epsilon_power is not None:
            self.config.greedy_power = float(self.config.epsilon_power)
        self.config.greedy_start = min(max(float(self.config.greedy_start), 0.0), 1.0)
        self.config.greedy_end = min(max(float(self.config.greedy_end), 0.0), 1.0)
        self.config.greedy_power = max(float(self.config.greedy_power), 0.0)
        # Single-step Q was retired; keep runtime behaviour on the two-step entry.
        self.config.state_design = "two_step_eff"

    @classmethod
    def action_from_index(cls, action_idx: int) -> MacroAction:
        if action_idx < 0 or action_idx >= cls.ACTION_COUNT:
            return MACRO_ACTIONS[0]
        return MACRO_ACTIONS[action_idx]

    def current_state(self) -> int | None:
        return self.previous_action_idx

    def set_previous_action(self, action_idx: int) -> None:
        if 0 <= action_idx < self.ACTION_COUNT:
            self.previous_action_idx = action_idx

    def greedy_factor(self, progress: float) -> float:
        p = min(max(float(progress), 0.0), 1.0)
        span = self.config.greedy_start - self.config.greedy_end
        return self.config.greedy_end + span * ((1.0 - p) ** self.config.greedy_power)

    def _state_values(self, state_idx: int) -> List[float]:
        return self.q_table[state_idx]

    def select_action(self, state_idx: int | None, progress: float, rng) -> int:
        if state_idx is None:
            return int(rng.randrange(self.ACTION_COUNT))
        q_values = self._state_values(state_idx)
        if rng.random() < self.greedy_factor(progress):
            best = max(q_values)
            candidates = [idx for idx, value in enumerate(q_values) if abs(value - best) <= 1e-12]
            return int(rng.choice(candidates))
        return int(rng.randrange(self.ACTION_COUNT))

    def compute_reward(
        self,
        dominates_old: bool,
        dominated_by_old: bool,
        archive_added: bool,
    ) -> float:
        if dominates_old:
            return float(self.config.reward_dominate)
        if dominated_by_old:
            return float(self.config.reward_dominated)
        if archive_added:
            return float(self.config.reward_nondominated)
        return float(self.config.reward_dominated)

    def update(
        self,
        state_idx: int,
        action_idx: int,
        reward: float,
        next_state: int | None = None,
    ) -> None:
        if not (0 <= state_idx < self.ACTION_COUNT and 0 <= action_idx < self.ACTION_COUNT):
            return
        self.visits[state_idx][action_idx] += 1
        visits = self.visits[state_idx][action_idx]
        alpha = self.config.alpha / (1.0 + 0.01 * visits)
        target = reward
        if next_state is not None and 0 <= next_state < self.ACTION_COUNT:
            target += self.config.gamma * max(self.q_table[next_state])
        current = self.q_table[state_idx][action_idx]
        self.q_table[state_idx][action_idx] = current + alpha * (target - current)
