"""
优化版连接生命周期管理器 v1.0.3 - 性能优化的连接管理

核心优化：
1. 向量化共同激活计算，替代逐对循环
2. 内存优化：使用更紧凑的数据类型
3. EMA平滑：替代历史窗口存储
4. 批量操作：减少GPU-CPU同步
"""

import torch
import heapq
from typing import List, Tuple, Set
import numpy as np


class OptimizedConnectionManager:
    """优化版连接生命周期管理器"""

    def __init__(self, num_neurons: int, protection_period: int = 75, device: str = 'cpu'):
        """
        初始化优化版连接管理器

        Args:
            num_neurons: 神经元数量
            protection_period: 新连接的保护期（步数）
            device: 设备（'cpu' 或 'cuda'）
        """
        self.num_neurons = num_neurons
        self.protection_period = protection_period
        self.device = device

        # 🚀 性能优化1：使用更紧凑的数据类型
        self.connection_age = torch.zeros(num_neurons, num_neurons, dtype=torch.int16, device=device)
        
        # 🚀 性能优化2：共同激活使用EMA，无需历史窗口
        self.coactivation_ema = torch.zeros(num_neurons, num_neurons, dtype=torch.float16, device=device)
        self.coactivation_decay = 0.95  # EMA衰减因子
        
        # 🚀 性能优化3：预分配候选池（固定大小）
        self.max_candidates = 50  # 减少候选池大小
        self.candidate_priorities = torch.zeros(self.max_candidates, 3, device=device)  # [priority, source, target]
        self.candidate_count = 0

        # 统计信息
        self.total_steps = 0

    def update_connection_ages_vectorized(self, adj_mask: torch.Tensor):
        """
        🚀 向量化更新所有活跃连接的年龄

        Args:
            adj_mask: 邻接矩阵掩码
        """
        with torch.no_grad():
            # 向量化操作：一次性更新所有活跃连接的年龄
            active_mask = adj_mask > 0
            self.connection_age[active_mask] = torch.clamp(
                self.connection_age[active_mask] + 1, 
                max=32767  # int16最大值
            )

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

    def can_be_pruned_vectorized(self, adj_mask: torch.Tensor, credit_scores: torch.Tensor, prune_threshold: float) -> torch.Tensor:
        """
        🚀 向量化判断哪些连接可以被剪枝（改进版：渐进式保护期）

        Args:
            adj_mask: 邻接矩阵掩码
            credit_scores: 信用分数矩阵
            prune_threshold: 剪枝阈值

        Returns:
            可以剪枝的连接掩码
        """
        with torch.no_grad():
            # 向量化条件检查
            active_mask = adj_mask > 0

            # 🔥 改进1：渐进式保护期（而非硬阈值）
            # 保护期内的连接也可能被剪枝，但概率随年龄增加而增加
            age_ratio_raw = self.connection_age.float() / max(self.protection_period, 1)
            age_ratio = torch.clamp(age_ratio_raw, 0.0, 1.0)  # 限制在 [0, 1]

            # 🔥 改进2：动态剪枝阈值（年龄越小，阈值越严格）
            # 年龄=0: 阈值 = prune_threshold * 0.1（很难剪枝）
            # 年龄=protection_period: 阈值 = prune_threshold * 1.0（正常剪枝）
            # 年龄>protection_period: 阈值 = prune_threshold * 1.5（更容易剪枝）
            dynamic_threshold = prune_threshold * (0.1 + 0.9 * age_ratio + 0.5 * (age_ratio_raw > 1.0).float())

            # 信用分数低于动态阈值的连接可以被剪枝
            credit_mask = credit_scores < dynamic_threshold

            # 综合条件：活跃 + 低信用分数（动态阈值）
            can_prune_mask = active_mask & credit_mask

            return can_prune_mask

    def can_be_pruned(self, source: int, target: int, credit_score: float, prune_threshold: float) -> bool:
        """
        兼容性方法：判断单个连接是否可以被剪枝

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

    def update_coactivation_vectorized(self, activations: torch.Tensor):
        """
        🚀 向量化更新共同激活计数

        Args:
            activations: 神经元激活值 (batch_size, num_neurons)
        """
        with torch.no_grad():
            # 🚀 优化1：使用二值化激活（向量化）
            binary_act = (activations > 0).half()  # 使用半精度以节省内存
            
            # 🚀 优化2：向量化计算共同激活矩阵
            # coactivation_batch[i,j] = 同时激活的批次比例
            batch_size = activations.size(0)
            coactivation_batch = torch.matmul(binary_act.t(), binary_act) / batch_size
            
            # 🚀 优化3：EMA更新（无需存储历史）
            self.coactivation_ema = (
                self.coactivation_decay * self.coactivation_ema
                + (1.0 - self.coactivation_decay) * coactivation_batch.half()
            )

            self.total_steps += 1

    def update_coactivation(self, activations: torch.Tensor):
        """
        兼容性方法：更新共同激活计数
        """
        self.update_coactivation_vectorized(activations)

    def calculate_connection_priority_vectorized(self, sources: torch.Tensor, targets: torch.Tensor, 
                                               relative_distances: torch.Tensor) -> torch.Tensor:
        """
        🚀 向量化批量计算连接优先级分数

        Args:
            sources: 源神经元张量
            targets: 目标神经元张量
            relative_distances: 相对距离张量

        Returns:
            优先级分数张量
        """
        with torch.no_grad():
            # 🚀 向量化距离因子计算
            valid_dist_mask = (relative_distances >= 2) & (relative_distances <= 3)
            distance_factors = torch.where(
                valid_dist_mask,
                1.0 / relative_distances,
                torch.zeros_like(relative_distances)
            )
            
            # 🚀 向量化共同激活因子获取
            coactivation_factors = self.coactivation_ema[sources, targets].float()
            
            # 🚀 向量化综合评分
            priorities = distance_factors * coactivation_factors
            
            return priorities

    def calculate_connection_priority(self, source: int, target: int, relative_distance: float) -> float:
        """
        兼容性方法：计算单个连接优先级分数

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

        # 🚀 使用EMA值替代历史计数
        coactivation_factor = self.coactivation_ema[source, target].item()

        # 综合评分
        priority = distance_factor * coactivation_factor

        return priority

    def add_candidate_vectorized(self, sources: torch.Tensor, targets: torch.Tensor, priorities: torch.Tensor):
        """
        🚀 向量化批量添加连接候选

        Args:
            sources: 源神经元张量
            targets: 目标神经元张量
            priorities: 优先级张量
        """
        if len(sources) == 0:
            return
            
        with torch.no_grad():
            # 🚀 筛选有效候选
            valid_mask = priorities > 0
            if not valid_mask.any():
                return
                
            valid_sources = sources[valid_mask]
            valid_targets = targets[valid_mask]
            valid_priorities = priorities[valid_mask]
            
            # 🚀 按优先级排序并取top-k
            _, top_indices = torch.topk(valid_priorities, 
                                       min(len(valid_priorities), self.max_candidates - self.candidate_count))
            
            # 🚀 批量添加到候选池
            for i, idx in enumerate(top_indices):
                if self.candidate_count < self.max_candidates:
                    self.candidate_priorities[self.candidate_count] = torch.tensor([
                        valid_priorities[idx].item(),
                        valid_sources[idx].item(), 
                        valid_targets[idx].item()
                    ], device=self.device)
                    self.candidate_count += 1

    def add_candidate(self, source: int, target: int, priority: float):
        """
        兼容性方法：添加单个连接候选到优先队列
        """
        # 简化实现：直接添加到张量
        if self.candidate_count < self.max_candidates and priority > 0:
            self.candidate_priorities[self.candidate_count] = torch.tensor([
                priority, source, target
            ], device=self.device)
            self.candidate_count += 1

    def get_top_candidates_vectorized(self, n: int = 10) -> List[Tuple[int, int, float]]:
        """
        🚀 向量化获取优先级最高的n个候选连接

        Args:
            n: 候选数量

        Returns:
            [(source, target, priority), ...]
        """
        if self.candidate_count == 0:
            return []

        with torch.no_grad():
            # 🚀 向量化排序前min(n, self.candidate_count)个候选
            actual_n = min(n, self.candidate_count)
            candidates = self.candidate_priorities[:self.candidate_count]
            
            # 按优先级降序排序
            _, sorted_indices = torch.sort(candidates[:, 0], descending=True)
            top_candidates = candidates[sorted_indices[:actual_n]]
            
            # 转换为列表格式
            result = []
            for i in range(actual_n):
                priority = top_candidates[i, 0].item()
                source = int(top_candidates[i, 1].item())
                target = int(top_candidates[i, 2].item())
                result.append((source, target, priority))
                
            return result

    def get_top_candidates(self, n: int = 10) -> List[Tuple[int, int, float]]:
        """
        兼容性方法：获取优先级最高的n个候选连接
        """
        return self.get_top_candidates_vectorized(n)

    def clear_candidates(self):
        """清空候选池"""
        with torch.no_grad():
            self.candidate_priorities.zero_()
            self.candidate_count = 0

    def get_protected_connections_vectorized(self, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        🚀 向量化获取所有处于保护期的连接

        Args:
            adj_mask: 邻接矩阵掩码

        Returns:
            保护期连接的掩码
        """
        with torch.no_grad():
            active_mask = adj_mask > 0
            protected_mask = (self.connection_age < self.protection_period) & active_mask
            return protected_mask

    def get_protected_connections(self, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        兼容性方法：获取所有处于保护期的连接
        """
        return self.get_protected_connections_vectorized(adj_mask)

    def get_statistics(self) -> dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        with torch.no_grad():
            return {
                'total_steps': self.total_steps,
                'avg_coactivation': self.coactivation_ema.float().mean().item(),
                'max_coactivation': self.coactivation_ema.float().max().item(),
                'candidate_pool_size': self.candidate_count,
                'memory_usage_mb': (
                    self.connection_age.numel() * self.connection_age.element_size() +
                    self.coactivation_ema.numel() * self.coactivation_ema.element_size()
                ) / (1024 * 1024)
            }

    def to(self, device):
        """
        将连接管理器的张量移动到指定设备

        Args:
            device: 目标设备
        """
        self.device = device
        self.connection_age = self.connection_age.to(device)
        self.coactivation_ema = self.coactivation_ema.to(device)
        self.candidate_priorities = self.candidate_priorities.to(device)
        return self
