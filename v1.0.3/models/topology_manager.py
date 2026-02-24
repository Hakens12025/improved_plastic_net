"""
拓扑管理器 - 管理神经元之间的相对拓扑距离

核心功能：
1. 计算相对拓扑距离（以目标神经元的邻居为参考点）
2. 使用BFS搜索最短路径
3. 增量更新距离缓存
4. 判断连接是否允许建立
"""

import torch
from collections import deque
from typing import Dict, Tuple, Set, List


class TopologyManager:
    """管理神经网络的拓扑结构和距离计算"""

    def __init__(self, num_neurons: int, min_distance: int = 2, max_distance: int = 3):
        """
        初始化拓扑管理器

        Args:
            num_neurons: 神经元数量
            min_distance: 允许连接的最小相对距离
            max_distance: 允许连接的最大相对距离
        """
        self.num_neurons = num_neurons
        self.min_distance = min_distance
        self.max_distance = max_distance

        # 相对距离缓存：只存储在允许范围内的距离
        # 格式：(source, target) -> distance
        self.relative_distance: Dict[Tuple[int, int], int] = {}

        # 邻接表：存储每个神经元的输入和输出邻居
        self.in_neighbors: Dict[int, Set[int]] = {i: set() for i in range(num_neurons)}
        self.out_neighbors: Dict[int, Set[int]] = {i: set() for i in range(num_neurons)}

    def update_topology(self, adj_mask: torch.Tensor):
        """
        更新拓扑结构（从邻接矩阵重建邻接表）

        Args:
            adj_mask: 邻接矩阵掩码 (num_neurons, num_neurons)
        """
        # 清空现有邻接表
        self.relative_distance.clear()
        for i in range(self.num_neurons):
            self.in_neighbors[i].clear()
            self.out_neighbors[i].clear()

        # 重建邻接表
        edges = torch.nonzero(adj_mask > 0, as_tuple=False)
        for edge in edges:
            source, target = edge[0].item(), edge[1].item()
            self.out_neighbors[source].add(target)
            self.in_neighbors[target].add(source)

    def get_neighbors(self, neuron_id: int, direction: str = 'in') -> Set[int]:
        """
        获取神经元的邻居

        Args:
            neuron_id: 神经元ID
            direction: 'in' 表示输入邻居，'out' 表示输出邻居

        Returns:
            邻居集合
        """
        if direction == 'in':
            return self.in_neighbors[neuron_id]
        else:
            return self.out_neighbors[neuron_id]

    def bfs_distance(self, source: int, target: int, max_depth: int = 3) -> int:
        """
        使用BFS计算从source到target的最短路径距离

        Args:
            source: 起始神经元
            target: 目标神经元
            max_depth: 最大搜索深度

        Returns:
            最短距离，如果不可达返回float('inf')
        """
        if source == target:
            return 0

        visited = set([source])
        queue = deque([(source, 0)])

        while queue:
            current, dist = queue.popleft()

            if dist >= max_depth:
                continue

            # 遍历当前节点的所有输出邻居
            for neighbor in self.out_neighbors[current]:
                if neighbor == target:
                    return dist + 1

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf')

    def calculate_relative_distance(self, source: int, target: int) -> float:
        """
        计算相对拓扑距离：以target的邻居为参考点

        核心思想：
        - 如果source是target的某个输入邻居的2-3跳范围内，则允许连接
        - 这样可以形成跨层连接，类似残差连接
        - 模拟"相似神经元"之间的连接

        Args:
            source: 源神经元
            target: 目标神经元

        Returns:
            最小相对距离
        """
        # 检查缓存
        cache_key = (source, target)
        if cache_key in self.relative_distance:
            return self.relative_distance[cache_key]

        # 获取target的所有输入邻居
        target_neighbors = self.get_neighbors(target, direction='in')

        if not target_neighbors:
            # 如果target没有输入邻居，返回无穷大
            return float('inf')

        # 计算source到target的每个邻居的距离，取最小值
        min_relative_dist = float('inf')
        for neighbor in target_neighbors:
            dist = self.bfs_distance(source, neighbor, max_depth=self.max_distance)
            min_relative_dist = min(min_relative_dist, dist)

        # 只缓存在允许范围内的距离
        if self.min_distance <= min_relative_dist <= self.max_distance:
            self.relative_distance[cache_key] = min_relative_dist

        return min_relative_dist

    def is_connection_allowed(self, source: int, target: int) -> bool:
        """
        判断是否允许在source和target之间建立连接

        Args:
            source: 源神经元
            target: 目标神经元

        Returns:
            True表示允许连接，False表示不允许
        """
        # 不允许自连接
        if source == target:
            return False

        # 计算相对距离
        rel_dist = self.calculate_relative_distance(source, target)

        # 判断是否在允许范围内
        return self.min_distance <= rel_dist <= self.max_distance

    def add_connection(self, source: int, target: int):
        """
        添加新连接并更新拓扑结构

        Args:
            source: 源神经元
            target: 目标神经元
        """
        self.out_neighbors[source].add(target)
        self.in_neighbors[target].add(source)

        # 增量更新相对距离缓存
        self._update_distances_after_add(source, target)

    def remove_connection(self, source: int, target: int):
        """
        删除连接并更新拓扑结构

        Args:
            source: 源神经元
            target: 目标神经元
        """
        self.out_neighbors[source].discard(target)
        self.in_neighbors[target].discard(source)

        # 清除相关的距离缓存
        self._clear_distances_after_remove(source, target)

    def _update_distances_after_add(self, source: int, target: int):
        """
        添加连接后增量更新距离缓存

        Args:
            source: 源神经元
            target: 目标神经元
        """
        # 只更新target的邻居到source的距离
        for neighbor in self.in_neighbors[target]:
            if neighbor != source:
                cache_key = (neighbor, source)
                old_dist = self.relative_distance.get(cache_key, float('inf'))
                new_dist = self.bfs_distance(neighbor, source, max_depth=self.max_distance)

                if self.min_distance <= new_dist <= self.max_distance:
                    if new_dist < old_dist:
                        self.relative_distance[cache_key] = new_dist
                elif cache_key in self.relative_distance:
                    del self.relative_distance[cache_key]

    def _clear_distances_after_remove(self, source: int, target: int):
        """
        删除连接后清除相关的距离缓存

        Args:
            source: 源神经元
            target: 目标神经元
        """
        # 连接移除后最短路径可能整体变化，直接清空缓存更安全
        self.relative_distance.clear()

    def get_connection_candidates(self, active_neurons: Set[int], max_candidates: int = 100) -> List[Tuple[int, int]]:
        """
        获取潜在的连接候选

        Args:
            active_neurons: 活跃的神经元集合
            max_candidates: 最大候选数量

        Returns:
            候选连接列表 [(source, target), ...]
        """
        candidates = []

        # 只在活跃神经元之间寻找候选
        for target in active_neurons:
            for source in active_neurons:
                if self.is_connection_allowed(source, target):
                    # 检查是否已经存在连接
                    if target not in self.out_neighbors[source]:
                        candidates.append((source, target))

                        if len(candidates) >= max_candidates:
                            return candidates

        return candidates

    def clear_cache(self):
        """清空距离缓存"""
        self.relative_distance.clear()
