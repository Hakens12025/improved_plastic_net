"""
改进版神经可塑性网络 - 核心模型

核心创新：
1. 双向动态拓扑调整（剪枝 + 生成新连接）
2. 基于相对拓扑距离的最小作用量原理
3. 持续演化模式（保持5-10%变化率）
4. 连接生命周期管理（短保护期）
"""

import torch
import torch.nn as nn
from typing import Tuple, List
from collections import deque

from .topology_manager import TopologyManager
from .connection_manager import ConnectionManager


class ImprovedPlasticNet(nn.Module):
    """改进版神经可塑性网络"""

    def __init__(
        self,
        num_neurons: int,
        input_dim: int,
        output_dim: int,
        iterations: int = 5,
        initial_sparsity: float = 0.5
    ):
        """
        初始化网络

        Args:
            num_neurons: 内部神经元数量
            input_dim: 输入维度
            output_dim: 输出维度
            iterations: 内部迭代次数
            initial_sparsity: 初始稀疏度
        """
        super(ImprovedPlasticNet, self).__init__()

        self.num_neurons = num_neurons
        self.iterations = iterations
        self.use_sparse = True

        # 内部拓扑权重
        self.weights = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.01)

        # Buffer：不参与梯度计算
        self.register_buffer('credit_score', torch.zeros(num_neurons, num_neurons))
        self.register_buffer('adj_mask', torch.triu(torch.ones(num_neurons, num_neurons), diagonal=1))

        # 初始化稀疏掩码
        if initial_sparsity > 0:
            mask_prob = torch.rand(num_neurons, num_neurons)
            self.adj_mask = (mask_prob > initial_sparsity).float() * self.adj_mask

        # 输入输出投影层
        self.input_proj = nn.Linear(input_dim, num_neurons)
        self.output_proj = nn.Linear(num_neurons, output_dim)

        # 拓扑管理器
        self.topology_manager = TopologyManager(num_neurons, min_distance=2, max_distance=3)
        self.topology_manager.update_topology(self.adj_mask)

        # 连接管理器
        self.connection_manager = ConnectionManager(num_neurons, protection_period=75)

        # 超参数
        self.decay_rate = 0.96
        self.prune_threshold = 0.01
        self.growth_threshold = 0.3

        # 持续演化参数
        self.target_change_rate_min = 0.05
        self.target_change_rate_max = 0.10

        # 统计信息
        self.change_history = deque(maxlen=50)
        self.pruned_count = 0
        self.added_count = 0
        self.step_count = 0

        # CSR 稀疏结构缓存（存储 W^T 的 CSR，避免转置触发 CSC）
        self.register_buffer('_csr_crow_t', torch.empty(0, dtype=torch.int64))
        self.register_buffer('_csr_col_t', torch.empty(0, dtype=torch.int64))
        self.register_buffer('_csr_row_t', torch.empty(0, dtype=torch.int64))
        self._rebuild_sparse_structure()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, input_dim)

        Returns:
            输出张量 (batch_size, output_dim)
        """
        # 投影到内部空间
        h = torch.relu(self.input_proj(x))

        # 应用掩码的权重
        active_w = None
        if not self.use_sparse:
            active_w = self.weights * self.adj_mask

        # 内部演化循环
        for _ in range(self.iterations):
            h_prev = h.clone()
            if self.use_sparse:
                h = torch.relu(self._sparse_matmul(h))
            else:
                h = torch.relu(torch.matmul(h, active_w))

            # 训练时更新信用分数和共同激活
            if self.training:
                with torch.no_grad():
                    # 更新信用分数
                    correlation = torch.matmul(h_prev.t(), h) / x.size(0)
                    self.credit_score += correlation

        # 更新共同激活（每10步一次，减少开销）
        if self.training and self.step_count % 10 == 0:
            with torch.no_grad():
                self.connection_manager.update_coactivation(h)

        # 衰减信用分数
        if self.training:
            self.credit_score *= self.decay_rate
            self.step_count += 1

        # 输出投影
        return self.output_proj(h)

    def apply_neuroplasticity(self) -> Tuple[int, int]:
        """
        应用神经可塑性：剪枝 + 生成新连接

        Returns:
            (剪枝数量, 新增数量)
        """
        with torch.no_grad():
            # 更新连接年龄
            self.connection_manager.update_connection_ages(self.adj_mask)

            # 1. 剪枝阶段
            pruned = self._prune_connections()

            # 剪枝后更新拓扑，保证后续生长阶段使用最新结构
            self.topology_manager.update_topology(self.adj_mask)

            # 2. 生成新连接阶段
            added = self._grow_connections()

            # 3. 更新拓扑结构
            self.topology_manager.update_topology(self.adj_mask)

            # 4. 记录变化率
            total_connections = self.adj_mask.sum().item()
            if total_connections > 0:
                change_rate = (pruned + added) / total_connections
                self.change_history.append(change_rate)

            # 5. 动态调整阈值
            if len(self.change_history) >= 10:
                avg_change_rate = sum(list(self.change_history)[-10:]) / 10
                self._adjust_thresholds(avg_change_rate)

            self.pruned_count = pruned
            self.added_count = added

            # 拓扑变化后重建稀疏结构
            self._rebuild_sparse_structure()

            return pruned, added

    def _prune_connections(self) -> int:
        """
        剪枝连接

        Returns:
            剪枝数量
        """
        active_mask = self.adj_mask > 0
        mature_mask = self.connection_manager.connection_age >= self.connection_manager.protection_period
        credit_mask = self.credit_score < self.prune_threshold
        prune_mask = active_mask & mature_mask & credit_mask

        pruned_count = int(prune_mask.sum().item())
        if pruned_count == 0:
            return 0

        # 批量剪枝
        self.adj_mask[prune_mask] = 0.0
        self.weights.data[prune_mask] = 0.0
        self.connection_manager.connection_age[prune_mask] = 0

        return pruned_count

    def _grow_connections(self) -> int:
        """
        生成新连接（优化版）

        Returns:
            新增数量
        """
        added_count = 0
        max_new_connections = 5  # 减少每次添加的连接数：10 -> 5
        max_candidates = 30     # 大幅减少候选搜索数：100 -> 30

        # 找到活跃的神经元（最近有激活）
        active_neurons = self._get_active_neurons()
        
        # 限制活跃神经元数量，减少搜索空间
        if len(active_neurons) > 50:
            active_neurons = set(list(active_neurons)[:50])

        # 获取连接候选（减少候选数）
        candidates = self.topology_manager.get_connection_candidates(active_neurons, max_candidates=max_candidates)

        # 计算每个候选的优先级
        candidate_priorities = []
        for source, target in candidates:
            rel_dist = self.topology_manager.calculate_relative_distance(source, target)
            priority = self.connection_manager.calculate_connection_priority(source, target, rel_dist)

            if priority > self.growth_threshold:
                candidate_priorities.append((priority, source, target))

        # 按优先级排序（只对前20个排序，减少排序开销）
        if len(candidate_priorities) > 20:
            candidate_priorities = sorted(candidate_priorities, reverse=True, key=lambda x: x[0])[:20]
        else:
            candidate_priorities.sort(reverse=True, key=lambda x: x[0])

        # 添加前N个候选
        for priority, source, target in candidate_priorities[:max_new_connections]:
            # 添加连接
            self.adj_mask[source, target] = 1.0
            self.weights.data[source, target] = torch.randn(1).item() * 0.03  # 小初始权重
            self.connection_manager.reset_connection_age(source, target)
            added_count += 1

        return added_count

    def _get_active_neurons(self) -> set:
        """
        获取活跃的神经元（有输入或输出连接）

        Returns:
            活跃神经元的集合
        """
        edges = torch.nonzero(self.adj_mask > 0, as_tuple=False)
        if edges.numel() == 0:
            return set()
        active_idx = torch.unique(edges)
        return set(active_idx.tolist())

    def _adjust_thresholds(self, current_change_rate: float):
        """
        根据当前变化率动态调整阈值

        Args:
            current_change_rate: 当前变化率
        """
        if current_change_rate < self.target_change_rate_min:
            # 变化太少，放宽阈值
            self.prune_threshold *= 0.95
            self.growth_threshold *= 0.95
        elif current_change_rate > self.target_change_rate_max:
            # 变化太多，收紧阈值
            self.prune_threshold *= 1.05
            self.growth_threshold *= 1.05

        # 限制阈值范围
        self.prune_threshold = max(0.001, min(0.1, self.prune_threshold))
        self.growth_threshold = max(0.1, min(0.8, self.growth_threshold))

    def get_sparsity(self) -> float:
        """
        获取当前稀疏度

        Returns:
            稀疏度（0-1之间）
        """
        total = self.num_neurons * (self.num_neurons - 1) / 2
        active = self.adj_mask.sum().item()
        return 1.0 - (active / total)

    def get_statistics(self) -> dict:
        """
        获取网络统计信息

        Returns:
            统计信息字典
        """
        return {
            'sparsity': self.get_sparsity(),
            'total_connections': self.adj_mask.sum().item(),
            'pruned_last': self.pruned_count,
            'added_last': self.added_count,
            'change_rate': (self.pruned_count + self.added_count) / max(1, self.adj_mask.sum().item()),
            'prune_threshold': self.prune_threshold,
            'growth_threshold': self.growth_threshold,
            'protected_connections': self.connection_manager.get_protected_connections(self.adj_mask).sum().item(),
            **self.connection_manager.get_statistics()
        }

    def _rebuild_sparse_structure(self):
        """
        重建 CSR 稀疏结构缓存
        """
        mask = (self.adj_mask > 0)
        if mask.sum().item() == 0:
            self._csr_crow_t = torch.zeros(self.num_neurons + 1, dtype=torch.int64, device=self.adj_mask.device)
            self._csr_col_t = torch.empty(0, dtype=torch.int64, device=self.adj_mask.device)
            self._csr_row_t = torch.empty(0, dtype=torch.int64, device=self.adj_mask.device)
            return

        edges = torch.nonzero(mask, as_tuple=False)
        src = edges[:, 0]
        dst = edges[:, 1]

        # 构建 W^T 的 CSR：row = dst, col = src
        row_t = dst
        col_t = src

        # 按 (row, col) 排序以构建 CSR
        order = torch.argsort(row_t * self.num_neurons + col_t)
        row_t = row_t[order]
        col_t = col_t[order]

        row_counts = torch.bincount(row_t, minlength=self.num_neurons)
        self._csr_crow_t = torch.zeros(self.num_neurons + 1, dtype=torch.int64, device=self.adj_mask.device)
        self._csr_crow_t[1:] = torch.cumsum(row_counts, dim=0)
        self._csr_col_t = col_t
        self._csr_row_t = row_t

    def _sparse_matmul(self, h: torch.Tensor) -> torch.Tensor:
        """
        使用 CSR 稀疏矩阵进行 h @ W 计算
        """
        if self._csr_col_t.numel() == 0:
            return torch.zeros_like(h)

        values = self.weights[self._csr_col_t, self._csr_row_t]
        active_w_t = torch.sparse_csr_tensor(
            self._csr_crow_t,
            self._csr_col_t,
            values,
            size=(self.num_neurons, self.num_neurons)
        )
        return torch.sparse.mm(active_w_t, h.t()).t()

    def to(self, device):
        """
        将模型移动到指定设备

        Args:
            device: 目标设备
        """
        # 调用父类的to方法
        super().to(device)
        # 移动connection_manager的张量
        self.connection_manager.to(device)
        return self
