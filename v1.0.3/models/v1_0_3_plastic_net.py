"""
优化版神经可塑性网络 v1.0.3 - 核心模型性能优化

核心优化：
1. 向量化剪枝操作，批量处理替代循环
2. 优化forward传播，减少内存复制
3. 批量连接生成，向量化优先级计算
4. 内存优化：就地操作和智能缓存
"""

import torch
import torch.nn as nn
from typing import Tuple, List
from collections import deque

from .v1_0_3_topology_manager import OptimizedTopologyManager
from .v1_0_3_connection_manager import OptimizedConnectionManager
from .initialization import TopologyAwareInitializer


class OptimizedPlasticNet(nn.Module):
    """优化版神经可塑性网络 v1.0.3"""

    def __init__(
        self,
        num_neurons: int,
        input_dim: int,
        output_dim: int,
        iterations: int = 5,
        initial_sparsity: float = 0.5,
        protection_period: int = 60,
        inhibitory_ratio: float = 0.2,
        trace_decay: float = 0.9,
        trace_scale: float = 0.05,
        homeostasis_target: float = 0.1,
        homeostasis_lr: float = 0.01,
        homeostasis_interval: int = 20,
        structural_interval: int = 200,
        activity_min: float = 0.02,
        activity_max: float = 0.5
    ):
        """
        初始化优化版网络

        Args:
            num_neurons: 内部神经元数量
            input_dim: 输入维度
            output_dim: 输出维度
            iterations: 内部迭代次数
            initial_sparsity: 初始稀疏度
            protection_period: 连接保护期
        """
        super(OptimizedPlasticNet, self).__init__()

        self.num_neurons = num_neurons
        self.iterations = iterations

        # 内部拓扑权重
        self.weights = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.01)

        # Buffer：不参与梯度计算
        self.register_buffer('credit_score', torch.zeros(num_neurons, num_neurons, dtype=torch.float32))
        self.register_buffer('adj_mask', torch.triu(torch.ones(num_neurons, num_neurons), diagonal=1))

        # 🚀 优化1：使用更紧凑的数据类型
        self.credit_score = self.credit_score.half()  # 半精度节省内存

        # 🚀 Stage 2 优化：先初始化拓扑管理器
        self.topology_manager = OptimizedTopologyManager(num_neurons, min_distance=2, max_distance=3)
        self.topology_manager.update_topology(self.adj_mask)  # 用初始adj_mask更新

        # 🚀 Stage 2 优化：使用智能初始化
        if initial_sparsity > 0:
            initializer = TopologyAwareInitializer(num_neurons, self.topology_manager)
            self.adj_mask = initializer.initialize_connections(
                target_sparsity=initial_sparsity,
                strategy='topology_aware'
            )
            # 更新拓扑
            self.topology_manager.update_topology(self.adj_mask)

        # 输入输出投影层
        self.input_proj = nn.Linear(input_dim, num_neurons)
        self.output_proj = nn.Linear(num_neurons, output_dim)

        # 🚀 优化2：使用优化版连接管理器（使用传入的protection_period）
        self.connection_manager = OptimizedConnectionManager(num_neurons, protection_period=protection_period, device='cpu')

        # Neuronal type constraint (Dale's law)
        self.inhibitory_ratio = inhibitory_ratio
        inhibitory_count = int(num_neurons * inhibitory_ratio)
        inhibitory_mask = torch.zeros(num_neurons, dtype=torch.bool)
        if inhibitory_count > 0:
            inhibitory_mask[-inhibitory_count:] = True
        self.register_buffer('inhibitory_mask', inhibitory_mask)
        dale_sign = torch.ones(num_neurons, dtype=torch.float32)
        dale_sign[inhibitory_mask] = -1.0
        self.register_buffer('dale_sign', dale_sign)
        with torch.no_grad():
            self.weights.data = self.weights.data.abs()
            if inhibitory_count > 0:
                self.weights.data[inhibitory_mask, :] *= -1.0

        # Local plasticity traces
        self.trace_decay = trace_decay
        self.trace_scale = trace_scale
        self.register_buffer('pre_trace', torch.zeros(num_neurons, dtype=torch.float32))
        self.register_buffer('post_trace', torch.zeros(num_neurons, dtype=torch.float32))

        # Homeostatic regulation
        self.homeostasis_target = homeostasis_target
        self.homeostasis_lr = homeostasis_lr
        self.homeostasis_interval = homeostasis_interval
        self.firing_rate_decay = 0.99
        self.register_buffer('firing_rate_ema', torch.zeros(num_neurons, dtype=torch.float32))

        # Structural plasticity timescale and activity gating
        self.structural_interval = structural_interval
        self.activity_min = activity_min
        self.activity_max = activity_max
        self.last_structural_step = 0

        # 超参数 - Stage 1 优化：调整阈值
        self.decay_rate = 0.96
        self.prune_threshold = 0.015  # 从0.01提高到0.015，更容易剪枝
        self.growth_threshold = 0.2   # 从0.3降低到0.2，更容易生长

        # 持续演化参数
        self.target_change_rate_min = 0.05
        self.target_change_rate_max = 0.10

        # 统计信息
        self.change_history = deque(maxlen=50)
        self.pruned_count = 0
        self.added_count = 0
        self.step_count = 0

        # 🚀 优化3 & Stage 3：预分配临时张量（仅对小规模网络）
        if self.num_neurons < 2000:
            self._temp_buffer = torch.zeros(num_neurons, num_neurons, dtype=torch.float32)
        else:
            self._temp_buffer = None  # 大规模网络按需分配，节省内存

    def forward_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """
        🚀 优化版前向传播 - 减少内存复制和GPU-CPU同步

        Args:
            x: 输入张量 (batch_size, input_dim)

        Returns:
            输出张量 (batch_size, output_dim)
        """
        # 投影到内部空间
        h = torch.relu(self.input_proj(x))

        # 🚀 优化：使用掩码的权重（就地操作）
        active_w = self.weights.abs() * self.dale_sign.view(-1, 1)

        # Use masked effective weights
        active_w = active_w * self.adj_mask

        if self.training:
            pre_activity = h.mean(dim=0)
            for _ in range(self.iterations):
                h = torch.relu(torch.matmul(h, active_w))
            post_activity = h.mean(dim=0)

            with torch.no_grad():
                # Local traces
                self.pre_trace = self.pre_trace * self.trace_decay + (1.0 - self.trace_decay) * pre_activity
                self.post_trace = self.post_trace * self.trace_decay + (1.0 - self.trace_decay) * post_activity

                # Local plasticity update on existing connections
                delta = torch.outer(self.pre_trace, self.post_trace)
                self.credit_score += (delta * self.adj_mask).half() * self.trace_scale

                # Homeostatic firing-rate tracking
                self.firing_rate_ema = (
                    self.firing_rate_decay * self.firing_rate_ema
                    + (1.0 - self.firing_rate_decay) * post_activity
                )
        else:
            for _ in range(self.iterations):
                h = torch.relu(torch.matmul(h, active_w))

        if self.training and self.step_count % 20 == 0:
            with torch.no_grad():
                self.connection_manager.update_coactivation_vectorized(h)

        # 衰减信用分数
        if self.training:
            with torch.no_grad():
                self.credit_score *= self.decay_rate

                # Homeostatic synaptic scaling
                if self.step_count % self.homeostasis_interval == 0:
                    rate_error = self.firing_rate_ema - self.homeostasis_target
                    scale = torch.clamp(1.0 - self.homeostasis_lr * rate_error, 0.5, 1.5)
                    self.weights.data.mul_(scale.unsqueeze(0))

                self.step_count += 1

        # 输出投影
        return self.output_proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """兼容性方法：调用优化版forward"""
        return self.forward_optimized(x)

    def apply_neuroplasticity_optimized(self) -> Tuple[int, int]:
        """
        🚀 优化版神经可塑性应用 - 向量化操作（改进版：平衡剪枝和生长）

        Returns:
            (剪枝数量, 新增数量)
        """
        with torch.no_grad():
            # 1. 🚀 向量化更新连接年龄
            self.connection_manager.update_connection_ages_vectorized(self.adj_mask)

            avg_rate = float(self.firing_rate_ema.mean().item())
            should_update_structure = (self.step_count - self.last_structural_step) >= self.structural_interval
            if (not should_update_structure) or (avg_rate < self.activity_min or avg_rate > self.activity_max):
                self.pruned_count = 0
                self.added_count = 0
                return 0, 0
            self.last_structural_step = self.step_count

            # 2. 🔥 改进：先生长，再剪枝（确保网络有足够的探索能力）
            added = self._grow_connections_vectorized()

            # 3. 🔥 改进：根据生长数量动态调整剪枝数量（保持平衡）
            pruned = self._prune_connections_balanced(target_prune_ratio=0.7)

            # 4. Stage 4 优化：只在变化较大时更新拓扑（减少计算开销）
            # 提高阈值，减少更新频率，避免卡顿
            change_threshold = max(20, int(self.num_neurons * 0.02))  # 至少20个或2%
            if pruned + added >= change_threshold:
                self.topology_manager.update_topology(self.adj_mask)

            # 5. 记录变化率
            total_connections = self.adj_mask.sum().item()
            if total_connections > 0:
                change_rate = (pruned + added) / total_connections
                self.change_history.append(change_rate)

            # 6. 动态调整阈值
            if len(self.change_history) >= 10:
                avg_change_rate = sum(list(self.change_history)[-10:]) / 10
                self._adjust_thresholds(avg_change_rate)

            self.pruned_count = pruned
            self.added_count = added

            return pruned, added

    def apply_neuroplasticity(self) -> Tuple[int, int]:
        """兼容性方法：调用优化版实现"""
        return self.apply_neuroplasticity_optimized()

    def _prune_connections_vectorized(self) -> int:
        """
        🚀 向量化剪枝连接 - 批量处理替代循环

        Returns:
            剪枝数量
        """
        # 🚀 使用向量化剪枝判断
        can_prune_mask = self.connection_manager.can_be_pruned_vectorized(
            self.adj_mask, self.credit_score, self.prune_threshold
        )

        if not can_prune_mask.any():
            return 0

        # 🚀 获取要剪枝的连接索引
        prune_indices = torch.nonzero(can_prune_mask, as_tuple=False)
        
        if len(prune_indices) == 0:
            return 0

        # 🚀 向量化剪枝操作
        sources = prune_indices[:, 0]
        targets = prune_indices[:, 1]

        # 批量剪枝
        self.adj_mask[sources, targets] = 0.0
        self.weights.data[sources, targets] = 0.0
        self.connection_manager.connection_age[sources, targets] = 0

        # 更新拓扑管理器（批量更新）
        # 不在这里逐个调用 remove_connection，等待统一的 update_topology
        # for source, target in zip(sources.tolist(), targets.tolist()):
        #     self.topology_manager.remove_connection(source, target)

        return len(prune_indices)

    def _prune_connections_balanced(self, target_prune_ratio: float = 0.3) -> int:
        """
        🔥 平衡剪枝连接 - 确保剪枝和生长的比例合理

        Args:
            target_prune_ratio: 目标剪枝比例（相对于可剪枝连接数）

        Returns:
            剪枝数量
        """
        # 🚀 使用向量化剪枝判断
        can_prune_mask = self.connection_manager.can_be_pruned_vectorized(
            self.adj_mask, self.credit_score, self.prune_threshold
        )

        if not can_prune_mask.any():
            return 0

        # 🚀 获取要剪枝的连接索引
        prune_indices = torch.nonzero(can_prune_mask, as_tuple=False)

        if len(prune_indices) == 0:
            return 0

        # 🔥 改进：动态计算剪枝数量
        # 基于当前连接数的百分比，而非固定上限
        total_connections = self.adj_mask.sum().item()

        # 每次剪枝不超过总连接数的1%，且不超过可剪枝连接数的30%
        max_prune_by_total = max(10, int(total_connections * 0.01))
        max_prune_by_candidates = max(10, int(len(prune_indices) * target_prune_ratio))
        max_prune = min(max_prune_by_total, max_prune_by_candidates)

        if len(prune_indices) > max_prune:
            # 🔥 优先剪枝信用分数最低的连接
            sources = prune_indices[:, 0]
            targets = prune_indices[:, 1]
            scores = self.credit_score[sources, targets]

            # 选择信用分数最低的 max_prune 个连接
            _, bottom_indices = torch.topk(scores, max_prune, largest=False)
            prune_indices = prune_indices[bottom_indices]

        # 🚀 向量化剪枝操作
        sources = prune_indices[:, 0]
        targets = prune_indices[:, 1]

        # 批量剪枝
        self.adj_mask[sources, targets] = 0.0
        self.weights.data[sources, targets] = 0.0
        self.connection_manager.connection_age[sources, targets] = 0

        return len(prune_indices)

    def _grow_connections_vectorized(self) -> int:
        """
        🚀 向量化生成新连接 - 批量优先级计算（改进版：更积极的生长）

        Returns:
            新增数量
        """
        # 🔥 改进：增加生长配额，确保每次都有足够的生长机会
        max_new_connections = max(20, int(self.num_neurons * 0.02))  # 1500神经元->30
        max_candidates = max(100, int(self.num_neurons * 0.05))      # 1500神经元->75

        # 🚀 快速获取活跃神经元
        active_neurons = self._get_active_neurons_fast()

        if len(active_neurons) < 2:
            return 0

        # 🚀 从优化版拓扑管理器获取候选
        candidates = self.topology_manager.get_connection_candidates_fast(active_neurons, max_candidates=max_candidates)

        if not candidates:
            return 0

        # 🚀 向量化批量计算优先级
        sources = torch.tensor([c[0] for c in candidates], device=self.weights.device)
        targets = torch.tensor([c[1] for c in candidates], device=self.weights.device)

        # 批量计算相对距离
        relative_distances = torch.tensor([
            self.topology_manager.calculate_relative_distance(c[0], c[1])
            for c in candidates
        ], device=self.weights.device, dtype=torch.float32)

        # 🚀 批量计算优先级
        priorities = self.connection_manager.calculate_connection_priority_vectorized(
            sources, targets, relative_distances
        )

        # 🔥 改进：降低生长阈值，增加生长机会
        # 使用动态阈值：优先级 > growth_threshold * 0.5
        effective_threshold = self.growth_threshold * 0.5
        valid_mask = priorities > effective_threshold
        if not valid_mask.any():
            return 0

        valid_sources = sources[valid_mask]
        valid_targets = targets[valid_mask]
        valid_priorities = priorities[valid_mask]

        # 🚀 选择top-k候选
        top_k = min(len(valid_priorities), max_new_connections)
        if top_k == 0:
            return 0

        _, top_indices = torch.topk(valid_priorities, top_k)

        # 🚀 批量添加新连接
        added_count = 0
        final_sources = []
        final_targets = []

        for idx in top_indices:
            source = valid_sources[idx].item()
            target = valid_targets[idx].item()

            if self.adj_mask[source, target] == 0:  # 确保没有连接
                final_sources.append(source)
                final_targets.append(target)
                added_count += 1

        if added_count > 0:
            # 🚀 批量设置连接
            final_sources_tensor = torch.tensor(final_sources, device=self.weights.device)
            final_targets_tensor = torch.tensor(final_targets, device=self.weights.device)

            self.adj_mask[final_sources_tensor, final_targets_tensor] = 1.0
            new_weights = torch.randn(added_count, device=self.weights.device) * 0.03
            new_weights = new_weights.abs() * self.dale_sign[final_sources_tensor]
            self.weights.data[final_sources_tensor, final_targets_tensor] = new_weights

            # 批量重置年龄和更新拓扑
            for source, target in zip(final_sources, final_targets):
                self.connection_manager.reset_connection_age(source, target)
                # 不在这里调用 add_connection，等待统一的 update_topology
                # self.topology_manager.add_connection(source, target)

        return added_count

    def _get_active_neurons_fast(self) -> set:
        """
        🚀 快速获取活跃神经元 - 向量化操作

        Returns:
            活跃神经元的集合
        """
        # 🚀 向量化查找活跃连接
        active_mask = self.adj_mask > 0
        if not active_mask.any():
            return set()

        # 获取所有活跃连接的索引
        active_edges = torch.nonzero(active_mask, as_tuple=False)
        
        # 🚀 批量提取源和目标神经元
        sources = set(active_edges[:, 0].tolist())
        targets = set(active_edges[:, 1].tolist())
        
        return sources.union(targets)

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
            'total_connections': int(self.adj_mask.sum().item()),
            'pruned_last': self.pruned_count,
            'added_last': self.added_count,
            'change_rate': (self.pruned_count + self.added_count) / max(1, self.adj_mask.sum().item()),
            'prune_threshold': self.prune_threshold,
            'growth_threshold': self.growth_threshold,
            'protected_connections': int(self.connection_manager.get_protected_connections(self.adj_mask).sum().item()),
            **self.connection_manager.get_statistics(),
            **self.topology_manager.get_statistics()
        }

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
