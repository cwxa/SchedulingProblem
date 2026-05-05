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
│   └── COA/               # COA 算法
├── scripts/                # 实验运行脚本
│   ├── run_fbea_*.py
│   ├── run_nsga2_full.py
│   └── run_rmoead.py
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

**基于强化学习的 MOEA/D 算法**，专门针对双目标模糊柔性作业车间调度问题。

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

**参考文献**：
> A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling (Li, Gong, Lu)

## 🚀 快速开始

### 环境要求

```bash
pip install -r requirements.txt
```

### 运行 RMOEA/D

```bash
python scripts/run_rmoead.py \
    --instance-json experiments/prepared_instances/mk01/a=1.5-2/mk01/seed_102.json \
    --algorithm-seed 42 \
    --population-size 100 \
    --mutation-rate 0.8 \
    --alpha 0.4 \
    --gamma 0.6 \
    --epsilon 0.8 \
    --memory-size 40
```

### 运行 FBEA

```bash
python scripts/run_fbea_idle25_init_dedup_gap50_q_ls_2step_ms_ego_all_mk.py \
    --instance-json experiments/prepared_instances/mk01/a=1.5-2/mk01/seed_102.json
```

### 运行 NSGA-II (对比算法)

```bash
python scripts/run_nsga2_full.py \
    --instance-json experiments/prepared_instances/mk01/a=1.5-2/mk01/seed_102.json
```

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
├── rmoead/                 # RMOEA/D 结果
│   └── mk01/
│       └── seed_102/
│           ├── seed_42.json       # 最终解集
│           └── snapshots/         # 收敛过程快照
├── fbea/                       # FBEA 结果
└── nsga2/                      # NSGA-II 结果
```

### 输出文件格式

```json
{
  "algorithm": "RMOEAD_ENHANCED",
  "dataset_id": "mk01",
  "instance_id": "seed_102",
  "algorithm_seed": 42,
  "solutions": [
    {
      "scheduling_string": [...],
      "machine_assignment_string": [...],
      "makespan": {"c1": 41.79, "c2": 46.0, "c3": 50.41},
      "energy": {"c1": 1673.84, "c2": 1813.62, "c3": 1988.26},
      "agreement": 0.126
    }
  ],
  "runtime_seconds": 45.23
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

### RMOEA/D 论文参数验证

使用论文推荐参数 (α=0.4, γ=0.6, ε=0.8) 在 mk01 数据集上验证：

- HV 指标表现良好
- Q-table 在 50-100 代后收敛稳定
- VNS 记忆在 30 代后效果显著

## 📚 相关文献

1. Li R, Gong W, Lu C. "A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling." Expert Systems with Applications, 2022.

2. FBEA 原始论文 (项目参考)

## 🤝 贡献者

项目维护者

## 📄 许可证

MIT License
