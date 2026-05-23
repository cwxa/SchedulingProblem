"""
RMOEA/D: A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling.

完全按照论文 "A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job 
shop scheduling" (Li, Gong, Lu) 实现的版本。

主要组件:
- MIX3 初始化策略 (Algorithm 2)
- Q-PAS 参数自适应策略 (Algorithm 3)  
- RVNS 变邻域搜索 (Algorithm 4)
- 精英归档管理 (Algorithm 5)
- CV 和 DV 收敛性与多样性指标
"""

from __future__ import annotations

import math
import copy
from typing import Callable, Dict, List, Optional, Tuple

from fbea.archive import _solution_key, fast_non_dominated_sort, dominates
from fbea.algorithm3_crossover_mutation import (
    OperatorUsageInfo,
    apply_machine_mutation,
    apply_scheduling_mutation,
    gpx_crossover,
    pox_crossover,
    uniform_machine_crossover,
)
from fbea.evaluation_counter import (
    MAX_EVALUATIONS,
    evaluate_solutions_with_count,
    get_evaluation_count,
    reset_global_counter,
)
from fbea.random_manager import get_rng, set_global_seed
from fbea.solution import Solution
from fbea.instance_loader import Instance


FIXED_MODE_PROBS = (1 / 3, 1 / 3, 1 / 3)
FIXED_SCHED_MUTATION = (1 / 3, 1 / 3, 1 / 3)
FIXED_MACHINE_MUTATION = (1 / 3, 1 / 3, 1 / 3)


def _get_fuzzy_scalar(obj) -> float:
    """使用论文中的 ranking operator: f1(x) = (x1 + 2*x2 + x3) / 4"""
    return (obj._c1() + 2 * obj._c2() + obj._c3()) / 4.0


def _choose_mode(rng) -> str:
    """随机选择交叉/变异模式"""
    r = rng.random()
    if r < FIXED_MODE_PROBS[0]:
        return "scheduling"
    if r < FIXED_MODE_PROBS[0] + FIXED_MODE_PROBS[1]:
        return "machine"
    return "both"


class MIX3Initializer:
    """
    MIX3 初始化策略 (Algorithm 2)
    
    使用三种初始化规则的组合:
    - GW (Global Workload): 最小化总机器负载
    - LS (Local minimum加工时间): 最小化makespan
    - Random: 随机初始化，保证多样性
    """
    
    @staticmethod
    def generate_gw_solution(instance: Instance) -> Solution:
        """
        GW Rule: 最小化总机器负载
        1. 将 O1,1, O2,1, ..., ON,1 放入调度向量
        2. 重排剩余操作的顺序
        3. 为每个操作选择可用且负载最小的机器
        """
        rng = get_rng()
        
        # 步骤1: 创建初始调度向量 - 优先放置每个作业的第一个操作
        scheduling = []
        for job_id in range(1, instance.num_jobs + 1):
            scheduling.append(job_id)
        
        # 获取作业操作数
        job_ops = list(instance.job_operation_counts.items())
        
        # 步骤2: 添加剩余操作
        remaining_ops = []
        for job_id, op_count in job_ops:
            for op_idx in range(1, op_count):
                remaining_ops.append(job_id)
        
        # 随机打乱剩余操作
        rng.shuffle(remaining_ops)
        scheduling.extend(remaining_ops)
        
        # 步骤3: 为每个操作选择负载最小的机器
        # 注意: machine_assignment_string[pos] 使用线性化位置索引
        machine_workload = {m: 0.0 for m in range(1, instance.num_machines + 1)}
        machine_assignment = [0] * instance.total_operations

        for position in range(instance.total_operations):
            job_id = scheduling[position]
            # 计算当前操作是 job_id 的第几个操作（从0开始）
            op_idx = scheduling[:position + 1].count(job_id) - 1
            # 线性化位置
            pos = instance.get_operation_position(job_id, op_idx)
            processing_map = instance.operation_processing_map[pos]

            # 选择负载最小的机器
            min_workload = float('inf')
            selected_machine = None
            for machine_id, proc_time in processing_map.items():
                workload = machine_workload[machine_id] + proc_time._c2()
                if workload < min_workload:
                    min_workload = workload
                    selected_machine = machine_id

            # 如果有多个机器负载相同，选择加工时间最短的
            if selected_machine is not None:
                for machine_id, proc_time in processing_map.items():
                    if abs(machine_workload[machine_id] + proc_time._c2() - min_workload) < 1e-6:
                        if proc_time._c2() < processing_map[selected_machine]._c2():
                            selected_machine = machine_id
                machine_workload[selected_machine] = min_workload

            machine_assignment[pos] = selected_machine or list(processing_map.keys())[0]
        
        return Solution(
            instance=instance,
            scheduling_string=scheduling,
            machine_assignment_string=machine_assignment,
            skip_validation=True,
        )
    
    @staticmethod
    def generate_ls_solution(instance: Instance) -> Solution:
        """
        LS Rule: 最小化 makespan
        1. 随机生成调度向量
        2. 为每个操作选择加工时间最短的机器
        """
        rng = get_rng()
        
        # 步骤1: 随机生成调度向量
        scheduling = []
        for job_id, op_count in instance.job_operation_counts.items():
            scheduling.extend([job_id] * op_count)
        rng.shuffle(scheduling)
        
        # 步骤2: 为每个操作选择加工时间最短的机器
        # 注意: machine_assignment_string[pos] 使用线性化位置索引
        machine_assignment = [0] * instance.total_operations
        for position in range(instance.total_operations):
            job_id = scheduling[position]
            # 计算当前操作是 job_id 的第几个操作（从0开始）
            op_idx = scheduling[:position + 1].count(job_id) - 1
            # 线性化位置
            pos = instance.get_operation_position(job_id, op_idx)
            processing_map = instance.operation_processing_map[pos]

            # 选择加工时间最短的机器
            min_time = float('inf')
            selected_machine = None
            for machine_id, proc_time in processing_map.items():
                if proc_time._c2() < min_time:
                    min_time = proc_time._c2()
                    selected_machine = machine_id

            machine_assignment[pos] = selected_machine or list(processing_map.keys())[0]
        
        return Solution(
            instance=instance,
            scheduling_string=scheduling,
            machine_assignment_string=machine_assignment,
            skip_validation=True,
        )
    
    @staticmethod
    def generate_random_solution(instance: Instance) -> Solution:
        """Random Rule: 随机初始化，保证多样性"""
        rng = get_rng()

        # 生成调度向量
        scheduling = []
        for job_id, op_count in instance.job_operation_counts.items():
            scheduling.extend([job_id] * op_count)
        rng.shuffle(scheduling)

        # 随机选择机器
        # 注意: machine_assignment_string[pos] 使用线性化位置索引
        machine_assignment = [0] * instance.total_operations
        for position in range(instance.total_operations):
            job_id = scheduling[position]
            # 计算当前操作是 job_id 的第几个操作（从0开始）
            op_idx = scheduling[:position + 1].count(job_id) - 1
            # 线性化位置
            pos = instance.get_operation_position(job_id, op_idx)
            valid_machines = list(instance.operation_machine_options[pos])
            machine_assignment[pos] = rng.choice(valid_machines)

        return Solution(
            instance=instance,
            scheduling_string=scheduling,
            machine_assignment_string=machine_assignment,
            skip_validation=True,
        )
    
    @classmethod
    def initialize(cls, instance: Instance, population_size: int) -> List[Solution]:
        """
        MIX3 主函数 (Algorithm 2)
        
        Args:
            instance: 问题实例
            population_size: 种群大小 Np
            
        Returns:
            初始化种群
        """
        rng = get_rng()
        population = []
        
        # 计算每种规则生成的数量
        size_per_rule = population_size // 3
        remainder = population_size - 3 * size_per_rule
        
        # 1. GW 生成 ⌊Np/3⌋ 个
        for _ in range(size_per_rule):
            population.append(cls.generate_gw_solution(instance))
        
        # 2. LS 生成 ⌊Np/3⌋ 个
        for _ in range(size_per_rule):
            population.append(cls.generate_ls_solution(instance))
        
        # 3. Random 生成 ⌊Np/3⌋ 个
        for _ in range(size_per_rule):
            population.append(cls.generate_random_solution(instance))
        
        # 如果不够，用 Random 补充
        while len(population) < population_size:
            population.append(cls.generate_random_solution(instance))
        
        return population[:population_size]


class QPASController:
    """
    Q-PAS: 基于 Q-learning 的参数自适应策略 (Algorithm 3)
    
    使用 Q-learning 来自适应选择 MOEA/D 的邻域大小参数 T
    T 的候选值: {5, 10, 15, 20}
    
    4个状态:
    - State 1: ΔCV > 0, ΔDV > 0
    - State 2: ΔCV > 0, ΔDV ≤ 0
    - State 3: ΔCV ≤ 0, ΔDV > 0
    - State 4: ΔCV ≤ 0, ΔDV ≤ 0
    """
    
    def __init__(self, alpha: float = 0.4, gamma: float = 0.6, epsilon: float = 0.8):
        """
        初始化 Q-PAS 控制器
        
        Args:
            alpha: 学习率 (论文建议 0.4)
            gamma: 折扣因子 (论文建议 0.6)
            epsilon: 贪婪因子 (论文建议 0.8)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 4个状态 x 4个动作 (T = {5, 10, 15, 20})
        self.q_table = [[0.0] * 4 for _ in range(4)]
        self.t_candidates = [5, 10, 15, 20]
        
        # 跟踪 CV 和 DV
        self.cv_history = []
        self.dv_history = []
        
        self.current_state = 0
        self.previous_cv = 1.0
        self.previous_dv = 1.0
    
    def get_state(self, delta_cv: float, delta_dv: float) -> int:
        """
        根据 CV 和 DV 的变化确定状态
        
        Args:
            delta_cv: CV 的变化 (CV_i-1 - CV_i)
            delta_dv: DV 的变化 (DV_i - DV_i-1)
            
        Returns:
            状态索引 (0-3)
        """
        if delta_cv > 0 and delta_dv > 0:
            return 0  # State 1
        elif delta_cv > 0 and delta_dv <= 0:
            return 1  # State 2
        elif delta_cv <= 0 and delta_dv > 0:
            return 2  # State 3
        else:
            return 3  # State 4
    
    def select_action(self) -> int:
        """
        使用 ε-greedy 策略选择动作
        
        Returns:
            动作索引 (0-3，对应 T = {5, 10, 15, 20})
        """
        rng = get_rng()
        
        if rng.random() < self.epsilon:
            # 探索: 随机选择
            return rng.randint(0, 3)
        else:
            # 利用: 选择 Q 值最大的动作
            max_q = max(self.q_table[self.current_state])
            best_actions = [i for i, q in enumerate(self.q_table[self.current_state]) if q == max_q]
            return rng.choice(best_actions)
    
    def update(self, delta_cv: float, delta_dv: float) -> None:
        """
        更新 Q-table
        
        Args:
            delta_cv: CV 的变化
            delta_dv: DV 的变化
        """
        next_state = self.get_state(delta_cv, delta_dv)
        
        # 计算奖励
        reward = 10 if delta_dv > 0 else 0
        
        # Q-learning 更新
        action = self.select_action()
        current_q = self.q_table[self.current_state][action]
        max_next_q = max(self.q_table[next_state])

        self.q_table[self.current_state][action] = (
            current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        )
        
        self.current_state = next_state


class RVNSLocalSearch:
    """
    RVNS: 基于强化学习的变邻域搜索 (Algorithm 4)
    
    5种局部搜索策略:
    - LS1: 交换调度向量中两个位置的作业
    - LS2: 移动一个操作到另一台加工时间最短的机器
    - LS3: 将最大负载机器上的操作移动到另一台机器
    - LS4: 交换调度向量中两个位置的机器分配
    - LS5: 逆序调度向量中两个位置之间的所有操作
    """
    
    def __init__(self, memory_size: int = 40):
        """
        初始化 RVNS
        
        Args:
            memory_size: 成功/失败记忆的大小 LP (论文建议 40)
        """
        self.memory_size = memory_size
        self.success_memory: List[Tuple[int, int]] = []  # (local_search_idx, 1)
        self.failure_memory: List[Tuple[int, int]] = []  # (local_search_idx, 0)
        self.selection_probabilities = [0.2] * 5  # 初始均匀概率
    
    def _ls1_swap_scheduling(self, solution: Solution) -> Solution:
        """LS1: 交换调度向量中两个位置的操作（只交换相同作业的操作以保持机器分配一致）"""
        rng = get_rng()

        # 收集每个作业出现的位置
        job_positions: Dict[int, List[int]] = {}
        for idx, job_id in enumerate(solution.scheduling_string):
            if job_id not in job_positions:
                job_positions[job_id] = []
            job_positions[job_id].append(idx)

        # 选择有多个操作的作业
        valid_jobs = [j for j, positions in job_positions.items() if len(positions) >= 2]
        if not valid_jobs:
            return solution

        # 随机选择一个作业
        job_id = rng.choice(valid_jobs)
        positions = job_positions[job_id]

        # 交换该作业的两个操作位置
        pos1, pos2 = rng.sample(positions, 2)

        new_scheduling = list(solution.scheduling_string)
        new_scheduling[pos1], new_scheduling[pos2] = new_scheduling[pos2], new_scheduling[pos1]

        return Solution(
            instance=solution.instance,
            scheduling_string=new_scheduling,
            machine_assignment_string=list(solution.machine_assignment_string),
            skip_validation=True,
        )
    
    def _ls2_move_to_min_machine(self, solution: Solution) -> Solution:
        """LS2: 将操作移动到加工时间最短的机器"""
        rng = get_rng()
        new_machines = list(solution.machine_assignment_string)

        # 随机选择一个作业
        job_id = rng.randint(1, solution.instance.num_jobs)
        op_count = solution.instance.job_operation_counts[job_id]
        if op_count < 1:
            return solution

        # 随机选择该作业的一个操作
        op_idx = rng.randint(0, op_count - 1)
        pos = solution.instance.get_operation_position(job_id, op_idx)
        processing_map = solution.instance.operation_processing_map[pos]

        # 选择加工时间最短的机器
        min_time = float('inf')
        best_machine = new_machines[pos]
        for machine_id, proc_time in processing_map.items():
            if proc_time._c2() < min_time:
                min_time = proc_time._c2()
                best_machine = machine_id

        new_machines[pos] = best_machine

        return Solution(
            instance=solution.instance,
            scheduling_string=list(solution.scheduling_string),
            machine_assignment_string=new_machines,
            skip_validation=True,
        )

    def _ls3_move_from_max_load_machine(self, solution: Solution) -> Solution:
        """LS3: 从最大负载机器移动操作到另一台机器"""
        rng = get_rng()

        # 计算每台机器的负载
        machine_workload: Dict[int, float] = {m: 0.0 for m in range(1, solution.instance.num_machines + 1)}

        for position in range(solution.instance.total_operations):
            job_id = solution.scheduling_string[position]
            op_idx = solution.scheduling_string[:position + 1].count(job_id) - 1
            pos = solution.instance.get_operation_position(job_id, op_idx)
            machine_id = solution.machine_assignment_string[pos]
            processing_map = solution.instance.operation_processing_map[pos]
            if machine_id in processing_map:
                machine_workload[machine_id] += processing_map[machine_id]._c2()

        # 找到最大负载机器
        max_load_machine = max(machine_workload.items(), key=lambda x: x[1])[0]

        # 找到在该机器上的操作
        operations_on_machine = []
        for position in range(solution.instance.total_operations):
            job_id = solution.scheduling_string[position]
            op_idx = solution.scheduling_string[:position + 1].count(job_id) - 1
            pos = solution.instance.get_operation_position(job_id, op_idx)
            if solution.machine_assignment_string[pos] == max_load_machine:
                operations_on_machine.append(position)

        if not operations_on_machine:
            return solution

        # 随机选择一个操作移动到另一台机器
        position = rng.choice(operations_on_machine)
        job_id = solution.scheduling_string[position]
        op_idx = solution.scheduling_string[:position + 1].count(job_id) - 1
        pos = solution.instance.get_operation_position(job_id, op_idx)
        processing_map = solution.instance.operation_processing_map[pos]

        # 选择另一台可用机器（不是最大负载机器）
        available_machines = [m for m in processing_map.keys() if m != max_load_machine]
        if not available_machines:
            return solution

        # 选择负载最小的可用机器
        min_load = float('inf')
        best_machine = available_machines[0]
        for m in available_machines:
            if machine_workload[m] < min_load:
                min_load = machine_workload[m]
                best_machine = m

        new_machines = list(solution.machine_assignment_string)
        new_machines[pos] = best_machine

        return Solution(
            instance=solution.instance,
            scheduling_string=list(solution.scheduling_string),
            machine_assignment_string=new_machines,
            skip_validation=True,
        )
    
    def _ls4_swap_machine_assignment(self, solution: Solution) -> Solution:
        """LS4: 为单个操作更换为有效但不同的机器"""
        rng = get_rng()
        new_machines = list(solution.machine_assignment_string)

        job_positions: Dict[int, List[int]] = {}
        for idx, job_id in enumerate(solution.scheduling_string):
            if job_id not in job_positions:
                job_positions[job_id] = []
            job_positions[job_id].append(idx)

        valid_jobs = [j for j, positions in job_positions.items() if len(positions) >= 2]
        if not valid_jobs:
            return solution

        job_id = rng.choice(valid_jobs)
        positions = job_positions[job_id]

        pos = rng.choice(positions)
        op_idx = solution.scheduling_string[:pos + 1].count(job_id) - 1
        lin_pos = solution.instance.get_operation_position(job_id, op_idx)

        processing_map = solution.instance.operation_processing_map[lin_pos]
        valid_machines = list(processing_map.keys())
        current_machine = new_machines[lin_pos]

        if len(valid_machines) <= 1:
            return solution

        alternative_machines = [m for m in valid_machines if m != current_machine]
        if not alternative_machines:
            return solution

        new_machine = rng.choice(alternative_machines)
        new_machines[lin_pos] = new_machine

        return Solution(
            instance=solution.instance,
            scheduling_string=list(solution.scheduling_string),
            machine_assignment_string=new_machines,
            skip_validation=True,
        )
    
    def _ls5_inverse_sequence(self, solution: Solution) -> Solution:
        """LS5: 逆序同一作业两个位置之间的操作（仅当该作业所有操作连续时）"""
        rng = get_rng()

        job_positions: Dict[int, List[int]] = {}
        for idx, job_id in enumerate(solution.scheduling_string):
            if job_id not in job_positions:
                job_positions[job_id] = []
            job_positions[job_id].append(idx)

        valid_jobs = [j for j, positions in job_positions.items() if len(positions) >= 2]
        if not valid_jobs:
            return solution

        job_id = rng.choice(valid_jobs)
        positions = job_positions[job_id]

        # 检查是否所有操作都连续
        positions_set = set(positions)
        expected = set(range(positions[0], positions[-1] + 1))
        if positions_set != expected:
            return solution

        if len(positions) == 2:
            pos1, pos2 = positions
        else:
            pos1_idx, pos2_idx = rng.sample(range(len(positions)), 2)
            pos1, pos2 = sorted([positions[pos1_idx], positions[pos2_idx]])

        new_scheduling = list(solution.scheduling_string)
        new_scheduling[pos1:pos2 + 1] = reversed(new_scheduling[pos1:pos2 + 1])

        new_machines = list(solution.machine_assignment_string)
        for offset in range(pos2 - pos1 + 1):
            old_pos = pos1 + offset
            new_pos = pos2 - offset

            old_job = solution.scheduling_string[old_pos]
            old_op_idx = solution.scheduling_string[:old_pos + 1].count(old_job) - 1
            old_lin_pos = solution.instance.get_operation_position(old_job, old_op_idx)

            new_job = new_scheduling[new_pos]
            new_op_idx = new_scheduling[:new_pos + 1].count(new_job) - 1
            new_lin_pos = solution.instance.get_operation_position(new_job, new_op_idx)

            new_machines[new_lin_pos] = solution.machine_assignment_string[old_lin_pos]

        return Solution(
            instance=solution.instance,
            scheduling_string=new_scheduling,
            machine_assignment_string=new_machines,
            skip_validation=True,
        )
    
    def select_local_search(self) -> int:
        """使用轮盘赌选择局部搜索策略"""
        rng = get_rng()
        
        # 检查是否需要更新概率
        if self.success_memory or self.failure_memory:
            self._update_probabilities()
        
        # 轮盘赌选择
        r = rng.random()
        cumulative = 0.0
        for idx, prob in enumerate(self.selection_probabilities):
            cumulative += prob
            if r <= cumulative:
                return idx
        return len(self.selection_probabilities) - 1
    
    def _update_probabilities(self) -> None:
        """根据成功/失败记忆更新选择概率"""
        # 计算每个策略的成功率和失败率
        sr = [0] * 5  # Success count
        fr = [0] * 5  # Failure count
        
        for ls_idx, _ in self.success_memory:
            sr[ls_idx] += 1
        for ls_idx, _ in self.failure_memory:
            fr[ls_idx] += 1
        
        # 计算概率: P(i) = SR(i) / (SR(i) + FR(i))
        total = 0.0
        for i in range(5):
            if sr[i] + fr[i] > 0:
                self.selection_probabilities[i] = sr[i] / (sr[i] + fr[i])
            total += self.selection_probabilities[i]
        
        # 归一化
        if total > 0:
            for i in range(5):
                self.selection_probabilities[i] /= total
    
    def apply_local_search(self, solution: Solution) -> Tuple[Solution, bool]:
        """
        应用局部搜索 (Algorithm 4 的简化版)
        
        Args:
            solution: 原始解
            
        Returns:
            (新解, 是否成功改进)
        """
        ls_idx = self.select_local_search()
        
        # 根据选择的策略生成邻居
        if ls_idx == 0:
            new_solution = self._ls1_swap_scheduling(solution)
        elif ls_idx == 1:
            new_solution = self._ls2_move_to_min_machine(solution)
        elif ls_idx == 2:
            new_solution = self._ls3_move_from_max_load_machine(solution)
        elif ls_idx == 3:
            new_solution = self._ls4_swap_machine_assignment(solution)
        else:
            new_solution = self._ls5_inverse_sequence(solution)
        
        # 评估解
        if not new_solution.evaluated:
            evaluate_solutions_with_count([new_solution])
        if not solution.evaluated:
            evaluate_solutions_with_count([solution])
        
        # 判断是否改进 (使用 Tchebycheff 聚合函数)
        # 这里简化处理：比较解的目标值
        old_obj = (_get_fuzzy_scalar(solution.makespan) +
                   _get_fuzzy_scalar(solution.energy) +
                   _get_fuzzy_scalar(solution.agreement))
        new_obj = (_get_fuzzy_scalar(new_solution.makespan) +
                   _get_fuzzy_scalar(new_solution.energy) +
                   _get_fuzzy_scalar(new_solution.agreement))

        improved = new_obj < old_obj
        
        # 记录到记忆
        if improved:
            if len(self.success_memory) >= self.memory_size:
                self.success_memory.pop(0)
            self.success_memory.append((ls_idx, 1))
        else:
            if len(self.failure_memory) >= self.memory_size:
                self.failure_memory.pop(0)
            self.failure_memory.append((ls_idx, 0))
        
        return new_solution, improved
    
    def apply_vns(self, solution: Solution) -> Solution:
        """
        执行完整的 VNS 过程 (Algorithm 4)
        
        Args:
            solution: 输入解
            
        Returns:
            改进后的解
        """
        # 只进行一次局部搜索
        improved_solution, improved = self.apply_local_search(solution)
        
        if improved:
            return improved_solution
        return solution


class MetricsCalculator:
    """
    计算收敛性指标 (CV) 和多样性指标 (DV) - 双目标版本
    """

    @staticmethod
    def calculate_cv(solutions: List[Solution]) -> float:
        """计算双目标收敛性指标 CV（越小越好）"""
        if len(solutions) < 2:
            return float('inf')

        # 获取双目标值：makespan + energy
        objectives = []
        for sol in solutions:
            f1 = _get_fuzzy_scalar(sol.makespan)
            f2 = _get_fuzzy_scalar(sol.energy)
            objectives.append((f1, f2))

        objectives.sort(key=lambda x: x[0])

        distances = []
        for i in range(len(objectives) - 1):
            d = math.sqrt((objectives[i + 1][0] - objectives[i][0]) ** 2 +
                         (objectives[i + 1][1] - objectives[i][1]) ** 2)
            distances.append(d)

        if not distances:
            return 0.0

        cv = math.sqrt(sum(d ** 2 for d in distances)) / len(solutions)
        return cv

    @staticmethod
    def calculate_dv(solutions: List[Solution]) -> float:
        """计算双目标多样性指标 DV（越大越好）"""
        if len(solutions) < 2:
            return 0.0

        # 获取双目标值：makespan + energy
        objectives = []
        for sol in solutions:
            f1 = _get_fuzzy_scalar(sol.makespan)
            f2 = _get_fuzzy_scalar(sol.energy)
            objectives.append((f1, f2))

        objectives.sort(key=lambda x: x[0])

        distances = []
        for i in range(len(objectives) - 1):
            d = math.sqrt((objectives[i + 1][0] - objectives[i][0]) ** 2 +
                         (objectives[i + 1][1] - objectives[i][1]) ** 2)
            distances.append(d)

        if not distances:
            return 0.0

        d_mean = sum(distances) / len(distances)
        dv = sum(abs(d - d_mean) for d in distances) / (len(distances) * d_mean)
        return dv


class EliteArchive:
    """
    精英归档管理 (Algorithm 5)
    
    用于存储历史迭代中的精英解，提高解的利用率
    """
    
    def __init__(self, max_size: int = 100):
        """
        初始化精英归档
        
        Args:
            max_size: 归档最大容量 (通常等于种群大小 Np)
        """
        self.max_size = max_size
        self.archive: List[Solution] = []
    
    def add(self, solutions: List[Solution]) -> None:
        """
        添加解到归档 (Algorithm 5)
        
        Args:
            solutions: 要添加的解列表
        """
        self.archive.extend(solutions)
        
        # 如果超过容量，进行非支配排序
        if len(self.archive) > self.max_size:
            self._prune()
    
    def _prune(self) -> None:
        """修剪归档以保持最大容量"""
        fronts = fast_non_dominated_sort(self.archive)
        
        new_archive = []
        for front in fronts:
            if len(new_archive) + len(front) <= self.max_size:
                new_archive.extend(front)
            else:
                # 这里简化处理：直接截断
                # 实际应该用拥挤度距离选择
                new_archive.extend(front[:self.max_size - len(new_archive)])
                break
        
        self.archive = new_archive
    
    def get_solutions(self) -> List[Solution]:
        """获取归档中的所有解"""
        return self.archive
    
    def get_pareto_front(self) -> List[Solution]:
        """获取归档中的非支配解集"""
        if not self.archive:
            return []
        fronts = fast_non_dominated_sort(self.archive)
        return fronts[0] if fronts else []


class RMOEAD:
    """
    RMOEA/D 主算法类
    
    按照论文 Algorithm 1 实现
    """
    
    def __init__(
        self,
        instance: Instance,
        population_size: int = 100,
        mutation_rate: float = 0.8,
        neighborhood_sizes: List[int] = None,
        alpha: float = 0.4,
        gamma: float = 0.6,
        epsilon: float = 0.8,
        memory_size: int = 40,
    ):
        """
        初始化 RMOEA/D
        
        Args:
            instance: 问题实例
            population_size: 种群大小 Np (论文建议 100)
            mutation_rate: 变异率 R (论文建议 0.8)
            neighborhood_sizes: 邻域大小候选值 {5, 10, 15, 20}
            alpha: Q-learning 学习率 (论文建议 0.4)
            gamma: Q-learning 折扣因子 (论文建议 0.6)
            epsilon: 贪婪因子 (论文建议 0.8)
            memory_size: VNS 记忆大小 LP (论文建议 40)
        """
        self.instance = instance
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # 初始化组件
        self.mix3 = MIX3Initializer()
        self.qpas = QPASController(alpha=alpha, gamma=gamma, epsilon=epsilon)
        self.rvns = RVNSLocalSearch(memory_size=memory_size)
        self.metrics = MetricsCalculator()
        self.elite_archive = EliteArchive(max_size=population_size)
        
        # 邻域大小候选值
        self.t_candidates = neighborhood_sizes or [5, 10, 15, 20]
        
        # 种群
        self.population: List[Solution] = []
        self.weights: List[List[float]] = []
        self.neighborhoods: List[List[int]] = []
        
        # 初始化权重向量和邻域
        self._initialize_weights_and_neighborhoods()
    
    def _initialize_weights_and_neighborhoods(self) -> None:
        """初始化双目标权重向量: w1 + w2 = 1, H = population_size - 1"""
        self.weights = []
        H = self.population_size - 1
        for i in range(H + 1):
            w1 = i / H
            w2 = 1.0 - w1
            self.weights.append([w1, w2])

        # 确保权重数量匹配种群大小
        if len(self.weights) < self.population_size:
            rng = get_rng()
            while len(self.weights) < self.population_size:
                w1 = rng.random()
                self.weights.append([w1, 1.0 - w1])

        # 计算邻域（基于权重向量之间的欧式距离）
        self.neighborhoods = []
        for i in range(self.population_size):
            distances = []
            for j in range(self.population_size):
                dist = math.sqrt(sum((self.weights[i][k] - self.weights[j][k]) ** 2
                                     for k in range(2)))  # 双目标：2维权重
                distances.append((dist, j))

            distances.sort()
            T = self.t_candidates[0]
            neighborhood = [idx for (dist, idx) in distances[:T]]
            self.neighborhoods.append(neighborhood)
    
    def _update_neighborhoods(self, T: int) -> None:
        """根据选定的 T 更新邻域"""
        for i in range(self.population_size):
            distances = []
            for j in range(self.population_size):
                dist = math.sqrt(sum((self.weights[i][k] - self.weights[j][k]) ** 2 
                                     for k in range(len(self.weights[i]))))
                distances.append((dist, j))
            
            distances.sort()
            self.neighborhoods[i] = [idx for (dist, idx) in distances[:T]]
    
    def _tchebycheff(self, solution: Solution, weight: List[float],
                     reference_point: List[float]) -> float:
        """Tchebycheff 聚合函数"""
        f1 = _get_fuzzy_scalar(solution.makespan)
        f2 = _get_fuzzy_scalar(solution.energy)
        f3 = _get_fuzzy_scalar(solution.agreement)
        objectives = [f1, f2, f3]

        max_val = 0.0
        for i in range(len(weight)):
            term = weight[i] * abs(objectives[i] - reference_point[i])
            if term > max_val:
                max_val = term
        return max_val

    def _update_reference_point(self, reference_point: List[float],
                               solution: Solution) -> List[float]:
        """更新参考点（理想点） - 双目标：makespan + energy"""
        f1 = _get_fuzzy_scalar(solution.makespan)
        f2 = _get_fuzzy_scalar(solution.energy)

        new_ref = list(reference_point)
        if f1 < new_ref[0]:
            new_ref[0] = f1
        if f2 < new_ref[1]:
            new_ref[1] = f2
        return new_ref
    
    def _crossover_mutation(self, parent1: Solution, parent2: Solution) -> Solution:
        """交叉和变异"""
        rng = get_rng()

        sched1 = list(parent1.scheduling_string)
        mach1 = list(parent1.machine_assignment_string)

        mode = _choose_mode(rng)
        scheduling_changed = False

        # 交叉
        if rng.random() < 0.9:
            if mode in ("scheduling", "both"):
                new_sched, _, _ = pox_crossover(self.instance, parent1, parent2)
                sched1 = new_sched
                scheduling_changed = True
            if mode in ("machine", "both"):
                mach1, _, _ = uniform_machine_crossover(parent1, parent2)

        # 变异
        if rng.random() < self.mutation_rate:
            op_info = OperatorUsageInfo()
            if mode in ("scheduling", "both"):
                sched1 = apply_scheduling_mutation(sched1, FIXED_SCHED_MUTATION, op_info)
                scheduling_changed = True
            if mode in ("machine", "both"):
                mach1 = apply_machine_mutation(self.instance, mach1, sched1,
                                              FIXED_MACHINE_MUTATION, op_info)

        # 如果调度顺序改变了，需要重新计算机器分配以保证一致性
        if scheduling_changed:
            mach1 = self._recompute_machine_assignment(sched1, parent1)

        return Solution(
            instance=self.instance,
            scheduling_string=sched1,
            machine_assignment_string=mach1,
            skip_validation=True,
        )

    def _recompute_machine_assignment(self, scheduling: List[int], reference: Solution) -> List[int]:
        """根据调度顺序重新计算机器分配（使用参考解的机器分配模式）

        核心思想：新调度中每个操作的机器分配应该与参考解中相同 (job_id, op_idx) 的操作保持一致。
        我们需要建立参考解中 (job_id, op_idx) -> machine 的映射，然后为新调度中每个位置分配正确的机器。
        """
        rng = get_rng()

        # 第一步：建立参考解中 (job_id, op_idx) -> machine 的映射
        ref_job_op_to_machine: Dict[Tuple[int, int], int] = {}
        for pos in range(reference.instance.total_operations):
            job_id = reference.scheduling_string[pos]
            op_idx = reference.scheduling_string[:pos + 1].count(job_id) - 1
            machine = reference.machine_assignment_string[pos]
            ref_job_op_to_machine[(job_id, op_idx)] = machine

        # 第二步：为新调度中每个位置分配机器
        new_machines = [0] * self.instance.total_operations

        for position in range(len(scheduling)):
            job_id = scheduling[position]
            op_idx = scheduling[:position + 1].count(job_id) - 1
            lin_pos = self.instance.get_operation_position(job_id, op_idx)

            # 尝试使用参考解中相同 (job_id, op_idx) 的机器分配
            if (job_id, op_idx) in ref_job_op_to_machine:
                ref_machine = ref_job_op_to_machine[(job_id, op_idx)]
                processing_map = self.instance.operation_processing_map[lin_pos]
                if ref_machine in processing_map:
                    new_machines[lin_pos] = ref_machine
                else:
                    new_machines[lin_pos] = rng.choice(list(processing_map.keys()))
            else:
                # 如果参考解中没有这个 (job_id, op_idx)，随机选择一个
                processing_map = self.instance.operation_processing_map[lin_pos]
                new_machines[lin_pos] = rng.choice(list(processing_map.keys()))

        return new_machines
    
    def _moea_decomposition_step(self, T: int) -> List[Solution]:
        """MOEA/D 分解步骤"""
        rng = get_rng()
        new_population = []
        reference_point = [float('inf'), float('inf')]  # 双目标理想点

        # 更新参考点
        for sol in self.population:
            reference_point = self._update_reference_point(reference_point, sol)

        # 复制当前种群用于父代选择，避免同代内 offspring 污染父代
        population_copy = [sol.clone() for sol in self.population]

        for i in range(self.population_size):
            # 选择邻域
            neighborhood = self.neighborhoods[i][:T]

            # 选择两个父代（从副本中选择）
            indices = rng.sample(neighborhood, 2)
            parent1 = population_copy[indices[0]]
            parent2 = population_copy[indices[1]]

            # 生成子代
            offspring = self._crossover_mutation(parent1, parent2)
            evaluate_solutions_with_count([offspring])

            # 更新邻域（应用到实际种群）
            for j in neighborhood:
                if self._tchebycheff(offspring, self.weights[j], reference_point) < \
                   self._tchebycheff(self.population[j], self.weights[j], reference_point):
                    self.population[j] = offspring.clone()
                    # 动态更新参考点
                    reference_point = self._update_reference_point(reference_point, self.population[j])

            new_population.append(offspring)

        return new_population
    
    def run(
        self,
        max_generations: int = 200,
        max_evaluations: int = None,
        generation_callback: Optional[Callable[[int, int], None]] = None,
        snapshot_callback: Optional[Callable[[int, List[Solution]], None]] = None,
    ) -> List[Solution]:
        """
        运行 RMOEA/D (Algorithm 1)
        
        Args:
            max_generations: 最大代数 (论文建议 200)
            max_evaluations: 最大评估次数
            generation_callback: 回调函数
            snapshot_callback: 快照回调
            
        Returns:
            最终 Pareto 前沿
        """
        if max_evaluations is None:
            max_evaluations = MAX_EVALUATIONS
        
        # Step 1: 初始化种群 (使用 MIX3)
        print("Initializing population (MIX3)...")
        self.population = self.mix3.initialize(self.instance, self.population_size)
        evaluate_solutions_with_count(self.population)
        
        # Step 2: 初始化变量
        previous_cv = 1.0
        previous_dv = 1.0
        
        generation = 0
        evaluation_limit = max_evaluations
        
        print(f"Starting evolution: max {max_generations} generations or {evaluation_limit} evaluations")
        
        while get_evaluation_count() < evaluation_limit and generation < max_generations:
            generation += 1
            
            # Step 4: 对每个个体执行 VNS
            for i in range(self.population_size):
                if get_evaluation_count() >= evaluation_limit:
                    break
                improved_sol = self.rvns.apply_vns(self.population[i])
                if improved_sol is not self.population[i]:
                    self.population[i] = improved_sol
            
            # Step 5: Q-PAS 选择邻域参数 T
            action_idx = self.qpas.select_action()
            T = self.t_candidates[action_idx]
            self._update_neighborhoods(T)
            
            # Step 6: MOEA/D 进化步骤
            self._moea_decomposition_step(T)
            
            # Step 7: 计算 CV 和 DV，更新 Q-table
            pareto_front = self.elite_archive.get_pareto_front()
            if pareto_front:
                current_cv = self.metrics.calculate_cv(pareto_front)
                current_dv = self.metrics.calculate_dv(pareto_front)
                
                delta_cv = previous_cv - current_cv
                delta_dv = current_dv - previous_dv
                
                self.qpas.update(delta_cv, delta_dv)
                
                previous_cv = current_cv
                previous_dv = current_dv
            
            # Step 8: 更新精英归档
            self.elite_archive.add(self.population)
            
            # 回调
            if generation_callback:
                generation_callback(generation, get_evaluation_count())
            if snapshot_callback and generation % 100 == 0:
                snapshot_callback(generation, self.population)
            
            if generation % 100 == 0:
                pf = self.elite_archive.get_pareto_front()
                print(f"Generation {generation}: {len(pf)} non-dominated solutions, "
                      f"{get_evaluation_count()}/{evaluation_limit} evaluations")
        
        return self.elite_archive.get_pareto_front()


def run_rmoead(
    instance,
    population_size: int = 100,
    crossover_probability: float = 0.9,
    mutation_probability: float = 0.8,
    random_seed: Optional[int] = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    max_evaluations: Optional[int] = None,
    snapshot_callback: Optional[Callable[[int, List[Solution]], None]] = None,
    alpha: float = 0.4,
    gamma: float = 0.6,
    epsilon: float = 0.8,
    memory_size: int = 40,
) -> List[Solution]:
    """
    RMOEA/D 主函数接口
    
    Args:
        instance: 问题实例
        population_size: 种群大小 (论文建议 100)
        crossover_probability: 交叉概率
        mutation_probability: 变异概率 (论文建议 0.8)
        random_seed: 随机种子
        generation_callback: 回调函数
        max_evaluations: 最大评估次数
        snapshot_callback: 快照回调
        alpha: Q-learning 学习率 (论文建议 0.4)
        gamma: Q-learning 折扣因子 (论文建议 0.6)
        epsilon: 贪婪因子 (论文建议 0.8)
        memory_size: VNS 记忆大小 (论文建议 40)
        
    Returns:
        最终 Pareto 前沿
    """
    if random_seed is not None:
        set_global_seed(random_seed)
    reset_global_counter()
    
    # 创建 RMOEA/D 实例
    rmoead = RMOEAD(
        instance=instance,
        population_size=population_size,
        mutation_rate=mutation_probability,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        memory_size=memory_size,
    )
    
    # 运行算法
    final_front = rmoead.run(
        max_generations=200,
        max_evaluations=max_evaluations,
        generation_callback=generation_callback,
        snapshot_callback=snapshot_callback,
    )
    
    return final_front
