"""
优化版拓扑管理器 v1.0.2 - 性能优化的拓扑距离计算

核心优化：
1. 预计算所有有效连接，O(1)连接检查
2. 缓存相对距离，避免重复BFS计算
3. 增量更新策略，只在拓扑变化时更新
4. 向量化操作，减少循环开销
"""

import torch
from collections import deque
from typing import Dict, Tuple, Set, List
import random


class OptimizedTopologyManager:
    """优化版拓扑管理器 - 大幅提升性能"""

    def __init__(self, num_neurons: int, min_distance: int = 2, max_distance: int = 3):
        """
        初始化优化版拓扑管理器

        Args:
            num_neurons: 神经元数量
            min_distance: 允许连接的最小相对距离
            max_distance: 允许连接的最大相对距离
        """
        self.num_neurons = num_neurons
        self.min_distance = min_distance
        self.max_distance = max_distance

        # 🚀 性能优化1：预计算所有可能的有效连接
        self.valid_connections = set()
        self.distance_cache = {}  # (source, target) -> distance
        
        # 邻接表：用于快速查询
        self.in_neighbors: Dict[int, Set[int]] = {i: set() for i in range(num_neurons)}
        self.out_neighbors: Dict[int, Set[int]] = {i: set() for i in range(num_neurons)}

        # 🚀 性能优化2：增量更新标记
        self._topology_hash = None
        self._is_dirty = True

        # 🚀 性能优化3：预计算基础距离矩阵（只计算一次）
        self._base_distances = None
        self._precompute_base_distances()

        # 初始化所有可能的有效连接
        self._precompute_valid_connections()

    def _precompute_base_distances(self):
        """预计算基础距离矩阵（不考虑连接状态）"""
        # 使用Floyd-Warshall算法的简化版本
        # 这里我们使用一个简化的距离计算，实际应用中可以进一步优化
        self._base_distances = {}
        
        # 预计算1-3跳的所有可能路径
        for source in range(self.num_neurons):
            for target in range(self.num_neurons):
                if source != target:
                    # 基础距离 = 直接距离或通过中间节点的距离
                    base_dist = abs(target - source)  # 简化：使用索引距离
                    self._base_distances[(source, target)] = min(base_dist, self.num_neighbors_distance(source, target))

    def num_neighbors_distance(self, source: int, target: int) -> int:
        """基于邻居数量的距离启发式"""
        # 这是一个简化的距离计算，可以进一步优化
        return min(abs(source - target), 3)  # 限制最大距离为3

    def _precompute_valid_connections(self):
        """🚀 预计算所有有效的连接对"""
        self.valid_connections.clear()
        
        for source in range(self.num_neurons):
            for target in range(self.num_neurons):
                if source != target:
                    # 计算基础距离
                    base_dist = self._base_distances.get((source, target), float('inf'))
                    
                    # 检查是否在允许的距离范围内
                    if self.min_distance <= base_dist <= self.max_distance:
                        self.valid_connections.add((source, target))
                        self.distance_cache[(source, target)] = base_dist

    def update_topology(self, adj_mask: torch.Tensor):
        """
        优化版拓扑更新 - 只在必要时重建

        Args:
            adj_mask: 邻接矩阵掩码 (num_neurons, num_neurons)
        """
        # 🚀 性能优化：检查拓扑是否真的发生了变化
        current_hash = hash(adj_mask.data_ptr())
        if current_hash == self._topology_hash and not self._is_dirty:
            return  # 拓扑未变化，跳过更新

        # 重建邻接表（向量化操作）
        for i in range(self.num_neurons):
            self.in_neighbors[i].clear()
            self.out_neighbors[i].clear()

        # 🚀 向量化获取所有连接（优化：减少GPU-CPU传输）
        edges = torch.nonzero(adj_mask > 0, as_tuple=False)
        if len(edges) > 0:
            # 一次性转换到CPU和numpy，避免多次传输
            edges_cpu = edges.cpu().numpy()

            # 批量更新邻接表
            for edge in edges_cpu:
                source, target = int(edge[0]), int(edge[1])
                self.out_neighbors[source].add(target)
                self.in_neighbors[target].add(source)

        self._topology_hash = current_hash
        self._is_dirty = False

    def get_neighbors(self, neuron_id: int, direction: str = 'in') -> Set[int]:
        """获取神经元的邻居"""
        if direction == 'in':
            return self.in_neighbors[neuron_id]
        else:
            return self.out_neighbors[neuron_id]

    def bfs_distance_optimized(self, source: int, target: int, max_depth: int = 3) -> int:
        """
        优化版BFS距离计算 - 使用缓存和剪枝

        Args:
            source: 起始神经元
            target: 目标神经元
            max_depth: 最大搜索深度

        Returns:
            最短距离，如果不可达返回float('inf')
        """
        if source == target:
            return 0

        # 🚀 检查缓存
        cache_key = (source, target)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        # 限制搜索范围以提高性能
        visited = set([source])
        queue = deque([(source, 0)])

        while queue:
            current, dist = queue.popleft()

            if dist >= max_depth:
                continue

            # 遍历当前节点的所有输出邻居
            for neighbor in self.out_neighbors[current]:
                if neighbor == target:
                    # 🚀 缓存结果
                    if dist + 1 <= self.max_distance:
                        self.distance_cache[cache_key] = dist + 1
                    return dist + 1

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf')

    def calculate_relative_distance_optimized(self, source: int, target: int) -> float:
        """
        优化版相对拓扑距离计算

        Args:
            source: 源神经元
            target: 目标神经元

        Returns:
            最小相对距离
        """
        # 🚀 优先使用预计算的缓存
        cache_key = (source, target)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        # 获取target的所有输入邻居
        target_neighbors = self.get_neighbors(target, direction='in')

        if not target_neighbors:
            # 如果target没有输入邻居，返回基础距离
            base_dist = self._base_distances.get(cache_key, float('inf'))
            if self.min_distance <= base_dist <= self.max_distance:
                self.distance_cache[cache_key] = base_dist
            return base_dist

        # 计算最小相对距离（限制搜索数量以提高性能）
        min_relative_dist = float('inf')
        neighbor_count = 0
        
        for neighbor in target_neighbors:
            dist = self.bfs_distance_optimized(source, neighbor, max_depth=self.max_distance)
            min_relative_dist = min(min_relative_dist, dist)
            
            neighbor_count += 1
            # 🚀 限制搜索的邻居数量以提高性能
            if neighbor_count >= 10:  # 最多检查10个邻居
                break

        # 只缓存在允许范围内的距离
        if self.min_distance <= min_relative_dist <= self.max_distance:
            self.distance_cache[cache_key] = min_relative_dist

        return min_relative_dist

    def is_connection_allowed_optimized(self, source: int, target: int) -> bool:
        """
        优化版连接允许检查 - O(1)时间复杂度

        Args:
            source: 源神经元
            target: 目标神经元

        Returns:
            True表示允许连接，False表示不允许
        """
        # 🚀 不允许自连接
        if source == target:
            return False

        # 🚀 O(1)检查预计算的有效连接集合
        return (source, target) in self.valid_connections

    def get_connection_candidates_fast(self, active_neurons: Set[int], max_candidates: int = 30) -> List[Tuple[int, int]]:
        """
        🚀 超快速候选连接获取

        Args:
            active_neurons: 活跃的神经元集合
            max_candidates: 最大候选数量

        Returns:
            候选连接列表 [(source, target), ...]
        """
        candidates = []

        # 🚀 从预计算的候选池中快速筛选
        for source, target in self.valid_connections:
            if source in active_neurons and target in active_neurons:
                # 🚀 快速检查是否已有连接
                if target not in self.out_neighbors[source]:
                    candidates.append((source, target))
                    
                    # 限制候选数量以提高性能
                    if len(candidates) >= max_candidates:
                        break

        # 🚀 随机打乱以增加多样性
        if len(candidates) > max_candidates:
            candidates = random.sample(candidates, max_candidates)
            
        return candidates

    def add_connection(self, source: int, target: int):
        """
        添加新连接并更新拓扑结构

        Args:
            source: 源神经元
            target: 目标神经元
        """
        self.out_neighbors[source].add(target)
        self.in_neighbors[target].add(source)
        
        # 🚀 标记拓扑为脏状态，下次更新时重新计算
        self._is_dirty = True

    def remove_connection(self, source: int, target: int):
        """
        删除连接并更新拓扑结构

        Args:
            source: 源神经元
            target: 目标神经元
        """
        self.out_neighbors[source].discard(target)
        self.in_neighbors[target].discard(source)
        
        # 🚀 清除相关的距离缓存（只清除受影响的项）
        keys_to_remove = []
        for key in self.distance_cache:
            if key[0] == source or key[1] == target:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.distance_cache[key]
            
        # 🚀 标记拓扑为脏状态
        self._is_dirty = True

    def get_connection_candidates(self, active_neurons: Set[int], max_candidates: int = 100) -> List[Tuple[int, int]]:
        """
        兼容性方法 - 调用优化版实现
        """
        return self.get_connection_candidates_fast(active_neurons, max_candidates)

    def calculate_relative_distance(self, source: int, target: int) -> float:
        """
        兼容性方法 - 调用优化版实现
        """
        return self.calculate_relative_distance_optimized(source, target)

    def is_connection_allowed(self, source: int, target: int) -> bool:
        """
        兼容性方法 - 调用优化版实现
        """
        return self.is_connection_allowed_optimized(source, target)

    def clear_cache(self):
        """清空距离缓存"""
        self.distance_cache.clear()
        self._is_dirty = True

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'valid_connections_count': len(self.valid_connections),
            'cached_distances_count': len(self.distance_cache),
            'cache_hit_ratio': len(self.distance_cache) / max(1, len(self.valid_connections))
        }