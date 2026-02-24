"""
连接生命周期管理器 - 管理连接的年龄、保护期和候选池

核心功能：
1. 追踪连接年龄
2. 管理保护期机制
3. 维护连接候选池
4. 计算连接优先级
"""

import torch
import heapq
from typing import List, Tuple, Set


class ConnectionManager:
    """管理神经网络连接的生命周期"""

    def __init__(self, num_neurons: int, protection_period: int = 75, device: str = 'cpu'):
        """
        初始化连接管理器

        Args:
            num_neurons: 神经元数量
            protection_period: 新连接的保护期（步数）
            device: 设备（'cpu' 或 'cuda'）
        """
        self.num_neurons = num_neurons
        self.protection_period = protection_period
        self.device = device

        # 连接年龄追踪
        self.connection_age = torch.zeros(num_neurons, num_neurons, dtype=torch.int32, device=device)

        # 共同激活计数器
        self.coactivation_count = torch.zeros(num_neurons, num_neurons, dtype=torch.int16, device=device)

        # 共同激活EMA（避免保存长窗口占用内存）
        self.coactivation_ema = torch.zeros(num_neurons, num_neurons, dtype=torch.float32, device=device)
        self.coactivation_decay = 0.95

        # 连接候选池（优先队列）
        self.connection_candidates: List[Tuple[float, int, int]] = []
        self.max_candidates = 100

        # 统计信息
        self.total_steps = 0

    def update_connection_ages(self, adj_mask: torch.Tensor):
        """
        更新所有活跃连接的年龄

        Args:
            adj_mask: 邻接矩阵掩码
        """
        active_mask = adj_mask > 0
        self.connection_age[active_mask] += 1

    def reset_connection_age(self, source: int, target: int):
        """
        重置连接年龄（用于新建连接）

        Args:
            source: 源神经元
            target: 目标神经元
        """
        self.connection_age[source, target] = 0

    def get_connection_status(self, source: int, target: int) -> str:
        """
        获取连接状态

        Args:
            source: 源神经元
            target: 目标神经元

        Returns:
            'protected' 或 'mature'
        """
        age = self.connection_age[source, target].item()
        if age < self.protection_period:
            return 'protected'
        else:
            return 'mature'

    def can_be_pruned(self, source: int, target: int, credit_score: float, prune_threshold: float) -> bool:
        """
        判断连接是否可以被剪枝

        Args:
            source: 源神经元
            target: 目标神经元
            credit_score: 连接的信用分数
            prune_threshold: 剪枝阈值

        Returns:
            True表示可以剪枝，False表示不可以
        """
        # 检查保护期
        if self.connection_age[source, target].item() < self.protection_period:
            return False

        # 检查信用分数
        if credit_score < prune_threshold:
            return True

        return False

    def update_coactivation(self, activations: torch.Tensor):
        """
        更新共同激活计数

        Args:
            activations: 神经元激活值 (batch_size, num_neurons)
        """
        # 计算批次内的共同激活
        # 使用二值化激活（激活>0为1，否则为0）
        binary_act = (activations > 0).float()

        # 计算共同激活矩阵：batch内每对神经元同时激活的次数
        coactivation_batch = torch.matmul(binary_act.t(), binary_act) / activations.size(0)

        # EMA 平滑
        self.coactivation_ema = (
            self.coactivation_decay * self.coactivation_ema
            + (1.0 - self.coactivation_decay) * coactivation_batch
        )
        # 缩放到int16范围，便于快速计算
        self.coactivation_count = (self.coactivation_ema * 1000).to(torch.int16)

        self.total_steps += 1

    def calculate_connection_priority(self, source: int, target: int, relative_distance: float) -> float:
        """
        计算连接优先级分数

        Args:
            source: 源神经元
            target: 目标神经元
            relative_distance: 相对拓扑距离

        Returns:
            优先级分数（越高越优先）
        """
        # 相对距离因子（距离越近，优先级越高）
        if relative_distance == float('inf') or relative_distance < 2 or relative_distance > 3:
            return 0.0

        distance_factor = 1.0 / relative_distance

        # 共同激活因子
        coactivation_factor = self.coactivation_count[source, target].item() / 1000.0

        # 综合评分
        priority = distance_factor * coactivation_factor

        return priority

    def add_candidate(self, source: int, target: int, priority: float):
        """
        添加连接候选到优先队列

        Args:
            source: 源神经元
            target: 目标神经元
            priority: 优先级分数
        """
        # 使用负优先级，因为heapq是最小堆
        heapq.heappush(self.connection_candidates, (-priority, source, target))

        # 保持队列大小
        if len(self.connection_candidates) > self.max_candidates:
            heapq.heappop(self.connection_candidates)

    def get_top_candidates(self, n: int = 10) -> List[Tuple[int, int, float]]:
        """
        获取优先级最高的n个候选连接

        Args:
            n: 候选数量

        Returns:
            [(source, target, priority), ...]
        """
        # 获取前n个候选（不移除）
        top_n = heapq.nsmallest(n, self.connection_candidates)

        # 转换为正优先级
        result = [(source, target, -priority) for priority, source, target in top_n]

        return result

    def clear_candidates(self):
        """清空候选池"""
        self.connection_candidates.clear()

    def get_protected_connections(self, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        获取所有处于保护期的连接

        Args:
            adj_mask: 邻接矩阵掩码

        Returns:
            保护期连接的掩码
        """
        active_mask = adj_mask > 0
        protected_mask = (self.connection_age < self.protection_period) & active_mask
        return protected_mask

    def get_statistics(self) -> dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        return {
            'total_steps': self.total_steps,
            'avg_coactivation': self.coactivation_count.float().mean().item(),
            'max_coactivation': self.coactivation_count.max().item(),
            'candidate_pool_size': len(self.connection_candidates)
        }

    def to(self, device):
        """
        将连接管理器的张量移动到指定设备

        Args:
            device: 目标设备
        """
        self.device = device
        self.connection_age = self.connection_age.to(device)
        self.coactivation_count = self.coactivation_count.to(device)
        self.coactivation_ema = self.coactivation_ema.to(device)
        return self
