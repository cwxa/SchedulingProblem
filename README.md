# SchedulingProblem

模糊柔性作业车间调度问题（FJSP）多目标进化算法研究项目。

## 📁 项目结构

```
SchedulingProblem/
├── src/
│   ├── fbea/              # 双种群反馈进化算法
│   │   ├── algorithm4_main_loop.py
│   │   ├── nsga2_algorithm.py
│   │   └── ...
│   ├── rmoead/            # RMOEA/D 算法（论文规范实现）
│   │   ├── rmoead_enhanced.py       # 主算法
│   │   └── ...
│   ├── COA/               # COA 算法
│   ├── run_batch_launcher.py # 批量算法运行器
│   └── run_batch_metrics.py  # 性能指标计算器
├── scripts/                # 实验运行脚本
│   ├── run_rmoead.py       # RMOEA/D 单实例运行
│   ├── run_rmoead_all_mk.py # RMOEA/D 批量运行 (mk01-mk10)
│   └── run_fbea_*.py       # FBEA 批量运行脚本
├── dataset/                # 基准数据集
│   ├── mk01-mk15/         # 标准 MK 数据集
│   └── special_*/         # 特殊数据集
├── experiments/
│   ├── prepared_instances/ # 预处理实例
│   └── results/            # 实验结果
└── tools/                  # 基准测试工具
    └── benchmark/
```

## 🔬 实现算法

### 1. FBEA (Bi-Population Evolutionary Algorithm with Feedback)

**双种群反馈进化算法**，用于求解三目标模糊柔性作业车间调度问题。

**目标函数**：
- **Makespan**: 最大完工时间（模糊数）
- **Energy**: 总机器能耗（模糊数）
- **Agreement**: 模糊数一致性指标

**核心特性**：
- 双种群协同进化
- 反馈机制指导算子选择
- Q-learning 增强的局部搜索
- 外部归档管理

**参考文献**：
> A Bi-Population Evolutionary Algorithm With Feedback for Energy-Efficient Fuzzy Flexible Job Shop Scheduling

### 2. RMOEA/D (Reinforcement Learning based MOEA/D)

**基于强化学习的 MOEA/D 算法**，针对模糊柔性作业车间调度问题。

**核心组件** (按论文 Algorithm 1 规范)：

| 组件 | 说明 |
|------|------|
| **MIX3 初始化** | 组合 GW (最小负载)、LS (最短加工时间)、Random 三种策略 |
| **Q-PAS** | Q-learning 自适应选择邻域参数 T ∈ {5, 10, 15, 20} |
| **RVNS** | 5种变邻域搜索策略 + 成功/失败记忆 |
| **Elite Archive** | 精英归档管理 |
| **CV/DV** | 收敛性与多样性指标 |

**目标函数**：
- **Makespan**: 最大完工时间（模糊数）
- **Energy**: 总机器能耗（模糊数）
- **Agreement**: 模糊一致度指标（由 `calculate_average_dissatisfaction_degree` 计算）

**实现细节**：
- Tchebycheff 分解同时优化 makespan、energy、agreement 三个标量化目标
- Q-PAS 使用 ε-greedy 策略从 {5, 10, 15, 20} 中选择邻域大小 T，Q-table 按标准 Q-learning 公式更新
- MOEA/D 邻域更新使用种群副本选父代，避免 offspring 在同代内污染父代
- 所有解评估通过 `evaluate_solutions_with_count` 计数，确保 budget 控制准确
- RVNS 包含 LS1（交换同作业操作）、LS2（移到最短加工时间机器）、LS3（从最大负载机器移出）、LS4（更换机器分配）、LS5（逆序同作业片段）五种局部搜索
- VNS 每代对全部种群成员执行

**参考文献**：
> A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling (Li, Gong, Lu)

## 🚀 快速开始

### 环境要求

```bash
pip install -r requirements.txt
```

### 运行 RMOEA/D (单实例)

```bash
python scripts/run_rmoead.py \
    --instance-json experiments/prepared_instances/mk01/a=1.5-2/mk01/seed_102.json \
    --algorithm-seed 42 \
    --max-evaluations 10000 \
    --population-size 100 \
    --mutation-rate 0.8 \
    --alpha 0.4 \
    --gamma 0.6 \
    --epsilon 0.8 \
    --memory-size 40
```

### 运行 RMOEA/D 批量实验

```bash
python scripts/run_rmoead_all_mk.py \
    --mk-filter mk01 mk02 mk03 \
    --budgets 10000 50000 150000 \
    --runs 20 \
    --skip-existing
```

参数说明：
- `--mk-filter`: 指定要运行的数据集，默认 mk01-mk10
- `--budgets`: 评估预算，默认 `[10000, 50000, 150000]`
- `--runs`: 每个配置运行次数，默认 20
- `--skip-existing`: 跳过已有结果文件，避免重复计算

## ⚙️ 参数设置

### RMOEA/D 论文推荐参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `population-size` | 100 | 种群大小 Np |
| `mutation-rate` | 0.8 | 变异率 R |
| `max-generations` | 200 | 最大代数 |
| `alpha` | 0.4 | Q-learning 学习率 |
| `gamma` | 0.6 | Q-learning 折扣因子 |
| `epsilon` | 0.8 | 贪婪因子 |
| `memory-size` | 40 | VNS 记忆大小 LP |
| `T` | {5,10,15,20} | 邻域参数候选值 |

### FBEA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `total-population-size` | 100 | 总种群大小 |
| `crossover-probability` | 0.7 | 交叉概率 |
| `mutation-probability` | 0.2 | 变异概率 |
| `feedback-mode` | "feedback" | 反馈模式 |

## 📊 数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| MK01-MK15 | 10-20 jobs, 6-15 machines | 标准基准 |
| special_5j3m | 5 jobs, 3 machines | 小规模测试 |

模糊处理时间通过参数 `a` 控制不确定度：
- `a=1.5-2`: 模糊幅度 [1.5×, 2.0×]
- `a=0.5-1`: 模糊幅度 [0.5×, 1.0×]

## 📈 输出说明

实验结果保存在 `experiments/results/` 目录：

```
experiments/results/
├── mk01/
│   ├── RMOEAD_PAPER_COMPLIANT/
│   │   ├── maxeval_10000/
│   │   │   └── mk01/
│   │   │       └── seed_102/
│   │   │           ├── seed_1000.json    # 单次运行结果
│   │   │           └── snapshots/        # 收敛过程快照 (gen 50/100/150/200)
│   │   ├── maxeval_50000/
│   │   └── maxeval_150000/
│   └── FBEA_IDLE25_INIT_DEDUP_GAP50_Q_LS_2STEP_MS_EGO/
│       └── maxeval_10000/
│           └── seed_102/
│               └── fbea_*.json
```

### 输出文件格式

```json
{
  "algorithm": "RMOEAD_ENHANCED",
  "dataset_id": "mk01",
  "instance_id": "seed_102",
  "algorithm_seed": 1000,
  "evaluations": 10002,
  "runtime_seconds": 3.24,
  "solutions": [
    {
      "makespan": [40.63, 44.0, 48.17],
      "energy": [1834.86, 1986.79, 2169.39],
      "agreement": [0.087, 0.106, 0.3],
      "dissatisfaction": [0.0, 0.106, 0.3]
    }
  ],
  "extra_metadata": {
    "population_size": 100,
    "mutation_rate": 0.8,
    "alpha": 0.4,
    "gamma": 0.6,
    "epsilon": 0.8,
    "memory_size": 40,
    "max_evaluations": 10000,
    "paper_compliant": true
  }
}
```

## 🔧 开发指南

### 添加新算法

1. 在 `src/` 下创建新模块目录
2. 实现算法主函数 `run_<algorithm_name>()`
3. 在 `scripts/` 下创建运行脚本
4. 更新本文档

### 代码规范

- 注释使用中文
- 打印日志使用英文
- 函数命名清晰易懂
- 添加适当的日志记录

## 📝 实验记录

### RMOEA/D 关键修复验证 (mk01, 10k budget)

修复后在 mk01 上的初步验证（4 seeds）：

| 指标 | 修复前 | 修复后 | FBEA 对比 |
|------|--------|--------|-----------|
| Best Makespan (scalar) | ~50.97 | **44.20 ~ 48.94** | 41.77 |
| Best Energy (scalar) | ~1932 | **1816 ~ 1912** | 1736.88 |
| 解数量 | ~7.8 | **5 ~ 16** | 66 |
| 运行时间 | ~3.8s | ~3.3s | ~1.6s |

主要改进：
- **Makespan 显著改善**：从 ~51 提升到 ~45，接近 FBEA 水平
- **三目标支持**：agreement 目标已纳入 Tchebycheff 分解和归档管理
- **Evaluation counter 准确**：实际 evaluations 与预算设定一致
- **局部搜索全面启用**：LS1~LS5 全部可用，VNS 每代全种群执行

### 已知限制

- RMOEA/D 解数量仍偏少（~10 vs FBEA ~66），主要受限于精英归档的截断策略
- 归档修剪目前使用简单截断，未使用拥挤度距离筛选

## 📚 相关文献

1. Li R, Gong W, Lu C. "A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling." Expert Systems with Applications, 2022.

2. FBEA 原始论文 (项目参考)

## 🤝 贡献者

项目维护者

## 📄 许可证

MIT License
