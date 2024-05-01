from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Union
import os

# import numpy as np
#
# # 定义问题参数
# cost_matrix = np.array([
#     [10, 15, 20, 25, 30, 35, 40, 45],
#     [12, 18, 24, 30, 36, 42, 48, 54],
#     [8, 12, 16, 20, 24, 28, 32, 36],
#     [14, 21, 28, 35, 42, 49, 56, 63],
#     [16, 24, 32, 40, 48, 56, 64, 72]
# ])
#
# supply = np.array([80, 120, 90, 110, 100])
# demand = np.array([70, 50, 80, 60, 90, 110, 75, 85])
#
# # 初始化变量
# num_supply = len(supply)
# num_demand = len(demand)
# num_vars = num_supply * num_demand
# solution = np.zeros((num_supply, num_demand))
#
# # 迭代优化
# while True:
#     # 计算剩余供应和需求
#     remaining_supply = supply - np.sum(solution, axis=1)
#     remaining_demand = demand - np.sum(solution, axis=0)
#
#     # 找到最大的供应和需求差距
#     max_supply_idx = np.argmax(remaining_supply)
#     max_demand_idx = np.argmax(remaining_demand)
#
#     # 计算最小运输量
#     min_transport = min(remaining_supply[max_supply_idx], remaining_demand[max_demand_idx])
#
#     # 更新解
#     solution[max_supply_idx, max_demand_idx] += min_transport
#
#     # 更新剩余供应和需求
#     remaining_supply[max_supply_idx] -= min_transport
#     remaining_demand[max_demand_idx] -= min_transport
#
#     # 检查是否满足供应和需求
#     if np.all(remaining_supply == 0) and np.all(remaining_demand == 0):
#         break

import torch

# 定义问题参数
cost_matrix = torch.tensor([
    [10, 15, 20, 25, 30, 35, 40, 45],
    [12, 18, 24, 30, 36, 42, 48, 54],
    [8, 12, 16, 20, 24, 28, 32, 36],
    [14, 21, 28, 35, 42, 49, 56, 63],
    [16, 24, 32, 40, 48, 56, 64, 72]
], dtype=torch.float32)

supply = torch.tensor([50, 80, 80, 120, 100], dtype=torch.float32)  # 调整供应量
demand = torch.tensor([70, 50, 80, 60, 90, 110, 75, 85], dtype=torch.float32)

# 将问题参数移动到GPU
cost_matrix = cost_matrix.cuda()
supply = supply.cuda()
demand = demand.cuda()

# 初始化变量
num_supply = len(supply)
num_demand = len(demand)
solution = torch.zeros((num_supply, num_demand), device="cuda")

# 迭代优化
# iteration_limit = 100  # 限制迭代次数，避免无限循环

for iteration in range():
    # 计算剩余供应和需求
    remaining_supply = supply - torch.sum(solution, dim=1)
    remaining_demand = demand - torch.sum(solution, dim=0)

    # 找到未满足供应和需求的最大差距
    max_diff_supply = torch.max(remaining_supply)
    max_diff_demand = torch.max(remaining_demand)

    if max_diff_supply > 0 and max_diff_demand > 0:
        # 如果存在未满足供应和需求的最大差距，则选择其中一个进行运输
        max_supply_idx = torch.argmax(remaining_supply)
        max_demand_idx = torch.argmax(remaining_demand)

        # 计算最小运输量
        min_transport = torch.min(remaining_supply[max_supply_idx], remaining_demand[max_demand_idx])

        # 更新解
        solution[max_supply_idx, max_demand_idx] += min_transport

    else:
        # 如果不存在未满足供应和需求的最大差距，说明已经找到可行解
        break

# 检查是否找到可行解
if torch.any(remaining_supply > 0) or torch.any(remaining_demand > 0):
    print("无法找到可行解。")
else:
    # 计算总成本
    total_cost = torch.sum(solution * cost_matrix)

    # 输出结果
    print("最小成本：", total_cost.item())
    print("最优运输方案矩阵：")
    print(solution.cpu().numpy())
