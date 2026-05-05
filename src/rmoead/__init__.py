"""
RMOEA/D: A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling.

该模块实现了基于强化学习的 MOEA/D 算法，包含：
- MIX3 初始化策略 (Algorithm 2)
- Q-PAS 参数自适应策略 (Algorithm 3)
- RVNS 变邻域搜索 (Algorithm 4)
- 精英归档管理 (Algorithm 5)
- CV 和 DV 收敛性与多样性指标

参考文献:
Li R, Gong W, Lu C. "A reinforcement learning based RMOEA/D for bi-objective fuzzy
flexible job shop scheduling." Expert Systems with Applications, 2022.
"""

from .rmoead_enhanced import run_rmoead

__all__ = ["run_rmoead"]
