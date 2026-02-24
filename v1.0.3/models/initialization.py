"""
智能初始化策略模块 - Stage 2 优化

提供基于拓扑距离的智能初始化，确保初始连接符合拓扑约束
"""
import torch
import random
from typing import Set, Tuple, List


class TopologyAwareInitializer:
    """基于拓扑距离的智能初始化器"""

    def __init__(self, num_neurons: int, topology_manager):
        """
        初始化器

        Args:
            num_neurons: 神经元数量
            topology_manager: 拓扑管理器实例
        """
        self.num_neurons = num_neurons
        self.topology_manager = topology_manager

    def initialize_connections(
        self,
        target_sparsity: float = 0.6,
        strategy: str = 'topology_aware'
    ) -> torch.Tensor:
        """
        智能初始化连接

        Args:
            target_sparsity: 目标稀疏度（0-1之间）
            strategy: 初始化策略
                - 'topology_aware': 基于拓扑距离（推荐）
                - 'progressive': 渐进式（高稀疏度起步）
                - 'random': 随机（回退方案）

        Returns:
            初始化的邻接矩阵
        """
        # 创建上三角矩阵作为基础
        adj_mask = torch.triu(
            torch.ones(self.num_neurons, self.num_neurons),
            diagonal=1
        )

        if strategy == 'topology_aware':
            return self._topology_aware_init(adj_mask, target_sparsity)
        elif strategy == 'progressive':
            return self._progressive_init(adj_mask, target_sparsity)
        else:
            return self._random_init(adj_mask, target_sparsity)

    def _topology_aware_init(
        self,
        adj_mask: torch.Tensor,
        target_sparsity: float
    ) -> torch.Tensor:
        """
        基于拓扑距离的初始化

        只从符合拓扑约束的有效连接中采样，确保初始连接都在2-3跳范围内

        Args:
            adj_mask: 基础邻接矩阵
            target_sparsity: 目标稀疏度

        Returns:
            初始化后的邻接矩阵
        """
        # 获取所有有效连接（符合拓扑约束的连接）
        valid_connections = list(self.topology_manager.valid_connections)

        if not valid_connections:
            print("[WARNING] No valid connections found, falling back to random init")
            return self._random_init(adj_mask, target_sparsity)

        # 计算要保留的连接数
        num_to_keep = int(len(valid_connections) * (1 - target_sparsity))

        # 随机采样
        selected = random.sample(
            valid_connections,
            min(num_to_keep, len(valid_connections))
        )

        # 清空adj_mask
        adj_mask.fill_(0)

        # 设置选中的连接
        for source, target in selected:
            adj_mask[source, target] = 1.0

        print(f"[INFO] Topology-aware init: {len(selected)} connections "
              f"from {len(valid_connections)} valid candidates")
        print(f"[INFO] Target sparsity: {target_sparsity:.1%}, "
              f"Actual sparsity: {1 - len(selected)/len(valid_connections):.1%}")

        return adj_mask

    def _progressive_init(
        self,
        adj_mask: torch.Tensor,
        target_sparsity: float
    ) -> torch.Tensor:
        """
        渐进式初始化（高稀疏度起步）

        从高稀疏度开始，让可塑性机制自然生长连接

        Args:
            adj_mask: 基础邻接矩阵
            target_sparsity: 目标稀疏度

        Returns:
            初始化后的邻接矩阵
        """
        mask_prob = torch.rand(self.num_neurons, self.num_neurons)
        adj_mask = (mask_prob > target_sparsity).float() * adj_mask

        print(f"[INFO] Progressive init with sparsity: {target_sparsity:.1%}")
        return adj_mask

    def _random_init(
        self,
        adj_mask: torch.Tensor,
        target_sparsity: float
    ) -> torch.Tensor:
        """
        随机初始化（回退方案）

        完全随机选择连接，不考虑拓扑约束

        Args:
            adj_mask: 基础邻接矩阵
            target_sparsity: 目标稀疏度

        Returns:
            初始化后的邻接矩阵
        """
        mask_prob = torch.rand(self.num_neurons, self.num_neurons)
        adj_mask = (mask_prob > target_sparsity).float() * adj_mask

        print(f"[INFO] Random init with sparsity: {target_sparsity:.1%}")
        return adj_mask
