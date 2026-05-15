"""验证实验结果的有效性"""
import json

# 检查最新生成的结果文件
result_file = 'experiments/results/mk01/RMOEAD_PAPER_COMPLIANT/maxeval_10000/mk01/seed_102/seed_1002.json'

with open(result_file, 'r') as f:
    data = json.load(f)

# 检查第一个解的 Job 4 工序数量
solution = data['solutions'][0]
gantt_ops = solution['gantt_operations']

# 统计 Job 4 的工序
job4_ops = [op for op in gantt_ops if op['job_id'] == 4]
print(f'Job 4 的工序数量: {len(job4_ops)}')
print('\nJob 4 的工序详情:')
for op in sorted(job4_ops, key=lambda x: x['operation_idx']):
    print(f'  Op{op["operation_idx"]}: 机器{op["machine_id"]}, 时间[{op["start_time"][1]:.2f}-{op["completion_time"][1]:.2f}]')

# 验证所有工序的机器分配是否有效
print('\n验证机器分配有效性...')
instance_file = 'experiments/prepared_instances/mk01/a=1.5-2/mk01/seed_102.json'
with open(instance_file, 'r') as f:
    instance = json.load(f)

job4_def = instance['jobs'][3]  # Job 4 (索引为3)
print(f'\nJob 4 定义中的工序数量: {len(job4_def["operations"])}')

all_valid = True
for op_idx, op_def in enumerate(job4_def['operations']):
    valid_machines = [opt['machine_id'] for opt in op_def['options']]
    gantt_op = next((op for op in job4_ops if op['operation_idx'] == op_idx), None)
    if gantt_op:
        assigned_machine = gantt_op['machine_id']
        is_valid = assigned_machine in valid_machines
        status = '✓ 有效' if is_valid else '✗ 无效'
        print(f'  Op{op_idx}: 分配机器{assigned_machine}, 可选机器{valid_machines} {status}')
        if not is_valid:
            all_valid = False
    else:
        print(f'  Op{op_idx}: 未找到对应的甘特图数据')
        all_valid = False

print('\n' + '='*60)
if all_valid and len(job4_ops) == len(job4_def['operations']):
    print('✓ 验证通过！所有工序都已正确调度，机器分配有效！')
else:
    print('✗ 验证失败！存在未调度或无效的工序。')
print('='*60)
