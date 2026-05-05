
1. Rule: 写代码或回答问题前，必须阅读《A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling》中的算法要求。
2. Rule: 涉及 PlatEMO 的讨论或开发前，先查阅《manual  (Chinese).md》中 PlatEMO 使用手册内容。

# 命名约定（批量运行输出）
- runs_root / 输出文件名统一带：模型名 + 比例/标签 + 实例名 + 时间戳。示例（20% 混入、模型 ppo_mk02_fuzzy3obj_run3、实例 seed_101、时间戳 20241215_0930）：  
  - runs_root: `experiments/results/batch_runs_ppo_mk02_fuzzy3obj_run3_20pct_seed_101_20241215_0930`  
  - output_csv: `experiments/results/batch_compare_ppo_mk02_fuzzy3obj_run3_20pct_seed_101_20241215_0930.json`  
  - summary_csv: `experiments/results/summary_ppo_mk02_fuzzy3obj_run3_20pct_seed_101_20241215_0930.json`  
  - coverage_csv: `experiments/results/coverage_ppo_mk02_fuzzy3obj_run3_20pct_seed_101_20241215_0930.json`
- 任何新建 runs_root 必须带时间戳或唯一区分标签，避免覆盖：例如 `..._20251216_1530` 或 `..._run2`，不要重复使用旧 runs_root。
- 甘特图导出路径也放在对应 runs_root 下，避免散乱。例如：  
  - base 甘特图输出：`<runs_root>/gantt_base_<instance>.png`  
  - ppo 甘特图输出：`<runs_root>/gantt_ppo_<instance>.png`
- 收敛监控输出一律放在当次 runs_root / seed 子目录中：  
  - 最终前沿：`.../<variant>/<dataset>/<instance>/seed_<id>.json`（原有终局文件保持不变）  
  - 快照：`.../<variant>/<dataset>/<instance>/seed_<id>_snapshots/gen_xxxx.json`  
  - 参考前沿 / 收敛报表 / 图像：`reference_front.json`、`convergence.csv`、`convergence.png` 等也放在对应 seed 子目录下，避免散落。
- 图像输出统一用 PNG（无论收敛曲线还是甘特图等），路径仍放在对应 runs_root/seed 子目录，命名需包含算法/seed 以免覆盖。

PowerShell 环境执行命令时不要用 `^` 作为换行连接，请直接一行写完整命令或用文本块包装，避免解析错误。
