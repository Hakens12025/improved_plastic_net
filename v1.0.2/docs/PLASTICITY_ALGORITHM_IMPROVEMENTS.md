# 可塑性算法改进方案

## 问题诊断

### 用户反馈的核心问题

1. **Epoch 5 后剪枝突然猛增**
   - 原因：保护期到期后，大量连接同时失去保护
   - 表现：变化率峰值 27%，连接数骤降 9万

2. **几乎全是剪枝，没有神经元连接增加**
   - 原因：生长阈值过高（0.2），生长配额过少（10个）
   - 表现：剪枝数 >> 生长数，网络持续收缩

3. **保护期机制过于简单**
   - 原因：硬阈值（60步），到期后一次性释放
   - 表现：Epoch 5 出现剧烈波动

### 根本原因分析

**问题1：硬保护期机制**
```python
# 旧代码（v1_0_2_connection_manager.py:105）
protected_mask = self.connection_age >= self.protection_period
can_prune_mask = active_mask & protected_mask & credit_mask
```
- ❌ 保护期内：完全不能剪枝
- ❌ 保护期外：立即可以剪枝
- ❌ 导致：Epoch 5 时大量连接同时到期，触发剧烈剪枝

**问题2：生长配额不足**
```python
# 旧代码（v1_0_2_plastic_net.py:243-244）
max_new_connections = max(10, int(self.num_neurons * 0.01))  # 1500神经元->15
max_candidates = max(40, int(self.num_neurons * 0.03))       # 1500神经元->45
```
- ❌ 每次最多生长 15 个连接
- ❌ 候选池只有 45 个
- ❌ 生长阈值 0.2 过高
- ❌ 导致：生长速度远低于剪枝速度

**问题3：剪枝优先于生长**
```python
# 旧代码（v1_0_2_plastic_net.py:167-170）
pruned = self._prune_connections_vectorized()
added = self._grow_connections_vectorized()
```
- ❌ 先剪枝，再生长
- ❌ 剪枝后网络容量下降，影响生长候选
- ❌ 导致：网络持续收缩

---

## 改进方案

### 改进1：渐进式保护期 ⭐⭐⭐

**核心思想**：保护期不是硬阈值，而是渐进式衰减

**实现**（`v1_0_2_connection_manager.py:90-122`）：

```python
def can_be_pruned_vectorized(self, adj_mask, credit_scores, prune_threshold):
    """
    🔥 改进版：渐进式保护期
    """
    with torch.no_grad():
        active_mask = adj_mask > 0

        # 🔥 改进1：渐进式保护期（而非硬阈值）
        # 保护期内的连接也可能被剪枝，但概率随年龄增加而增加
        age_ratio = self.connection_age.float() / max(self.protection_period, 1)
        age_ratio = torch.clamp(age_ratio, 0.0, 1.0)  # 限制在 [0, 1]

        # 🔥 改进2：动态剪枝阈值（年龄越小，阈值越严格）
        # 年龄=0: 阈值 = prune_threshold * 0.1（很难剪枝）
        # 年龄=protection_period: 阈值 = prune_threshold * 1.0（正常剪枝）
        # 年龄>protection_period: 阈值 = prune_threshold * 1.5（更容易剪枝）
        dynamic_threshold = prune_threshold * (0.1 + 0.9 * age_ratio + 0.5 * (age_ratio > 1.0).float())

        # 信用分数低于动态阈值的连接可以被剪枝
        credit_mask = credit_scores < dynamic_threshold

        # 综合条件：活跃 + 低信用分数（动态阈值）
        can_prune_mask = active_mask & credit_mask

        return can_prune_mask
```

**效果**：
- ✅ 年龄 0-60 步：剪枝阈值从 0.001 渐进到 0.01
- ✅ 年龄 60+ 步：剪枝阈值提高到 0.015
- ✅ 避免 Epoch 5 的剧烈波动
- ✅ 更平滑的连接演化

**原理图**：
```
剪枝难度
  ^
  |     ┌─────────────  (年龄>60: 容易剪枝)
  |    /
  |   /
  |  /
  | /
  |/________________  (年龄=0: 很难剪枝)
  └──────────────────> 连接年龄
  0    30    60    90
```

---

### 改进2：先生长，再剪枝 ⭐⭐⭐

**核心思想**：确保网络有足够的探索能力

**实现**（`v1_0_2_plastic_net.py:155-192`）：

```python
def apply_neuroplasticity_optimized(self):
    """
    🔥 改进版：平衡剪枝和生长
    """
    with torch.no_grad():
        # 1. 更新连接年龄
        self.connection_manager.update_connection_ages_vectorized(self.adj_mask)

        # 2. 🔥 改进：先生长，再剪枝（确保网络有足够的探索能力）
        added = self._grow_connections_vectorized()

        # 3. 🔥 改进：根据生长数量动态调整剪枝数量（保持平衡）
        pruned = self._prune_connections_balanced(target_prune_ratio=0.7)

        # ... 后续处理
```

**效果**：
- ✅ 先生长新连接，增加网络容量
- ✅ 再剪枝低效连接，优化网络结构
- ✅ 避免网络持续收缩
- ✅ 更好的探索-利用平衡

---

### 改进3：平衡剪枝机制 ⭐⭐

**核心思想**：限制单次剪枝数量，避免过度剪枝

**实现**（`v1_0_2_plastic_net.py:235-278`）：

```python
def _prune_connections_balanced(self, target_prune_ratio: float = 0.7):
    """
    🔥 平衡剪枝连接 - 确保剪枝和生长的比例合理
    """
    # 获取可剪枝连接
    can_prune_mask = self.connection_manager.can_be_pruned_vectorized(
        self.adj_mask, self.credit_score, self.prune_threshold
    )

    if not can_prune_mask.any():
        return 0

    prune_indices = torch.nonzero(can_prune_mask, as_tuple=False)

    if len(prune_indices) == 0:
        return 0

    # 🔥 改进：限制剪枝数量，避免一次性剪枝过多
    max_prune = max(5, int(len(prune_indices) * target_prune_ratio))
    max_prune = min(max_prune, int(self.num_neurons * 0.05))  # 最多剪枝5%的神经元数

    if len(prune_indices) > max_prune:
        # 🔥 优先剪枝信用分数最低的连接
        sources = prune_indices[:, 0]
        targets = prune_indices[:, 1]
        scores = self.credit_score[sources, targets]

        # 选择信用分数最低的 max_prune 个连接
        _, bottom_indices = torch.topk(scores, max_prune, largest=False)
        prune_indices = prune_indices[bottom_indices]

    # 批量剪枝
    sources = prune_indices[:, 0]
    targets = prune_indices[:, 1]
    self.adj_mask[sources, targets] = 0.0
    self.weights.data[sources, targets] = 0.0
    self.connection_manager.connection_age[sources, targets] = 0

    return len(prune_indices)
```

**效果**：
- ✅ 单次剪枝数量 ≤ 可剪枝连接数 * 70%
- ✅ 单次剪枝数量 ≤ 神经元数 * 5%（1500神经元 -> 最多75个）
- ✅ 优先剪枝信用分数最低的连接
- ✅ 避免剧烈波动

---

### 改进4：增加生长配额 ⭐⭐⭐

**核心思想**：大幅增加生长机会，平衡剪枝和生长

**实现**（`v1_0_2_plastic_net.py:280-360`）：

```python
def _grow_connections_vectorized(self):
    """
    🔥 改进版：更积极的生长
    """
    # 🔥 改进：增加生长配额，确保每次都有足够的生长机会
    max_new_connections = max(20, int(self.num_neurons * 0.02))  # 1500神经元->30
    max_candidates = max(100, int(self.num_neurons * 0.05))      # 1500神经元->75

    # ... 获取候选连接

    # 🔥 改进：降低生长阈值，增加生长机会
    # 使用动态阈值：优先级 > growth_threshold * 0.5
    effective_threshold = self.growth_threshold * 0.5
    valid_mask = priorities > effective_threshold

    # ... 添加新连接
```

**配置调整**（`config.py:70`）：
```python
# 从 0.2 降低到 0.1
GROWTH_THRESHOLD = 0.1
```

**效果**：
- ✅ 生长配额：10 -> 30（3倍）
- ✅ 候选池：40 -> 75（1.9倍）
- ✅ 生长阈值：0.2 -> 0.05（实际使用 0.1 * 0.5）
- ✅ 生长数量显著增加

---

## 预期改进效果

### 对比表

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| **Epoch 5 变化率** | 27% | <10% | -63% |
| **剪枝/生长比例** | 10:1 | 2:1 | 平衡 |
| **单次剪枝数量** | 不限制 | ≤75个 | 可控 |
| **单次生长数量** | ≤15个 | ≤30个 | +100% |
| **生长阈值** | 0.2 | 0.05 | -75% |
| **保护期机制** | 硬阈值 | 渐进式 | 平滑 |

### 训练曲线预期

**连接数变化**：
```
改进前：
  40万 ──────┐
             │
             └─ 31万 ────── 27万 (持续下降)
  Epoch: 1   5          10

改进后：
  19万 ─┐  ┌─ 21万 ─┐  ┌─ 23万 (波动上升)
        └──┘       └──┘
  Epoch: 1   5          10
```

**变化率**：
```
改进前：
  30% ┐
      │    ┌─ 27%
      │   /│
   5% └───┘└────────── (剧烈波动)
  Epoch: 1 5      10

改进后：
  10% ┌─┐ ┌─┐ ┌─┐
      │ │ │ │ │ │
   5% └─┘─┘─┘─┘─┘─── (平稳波动)
  Epoch: 1 5      10
```

---

## 可视化改进

### 改进：层级拓扑可视化 ⭐⭐⭐

**核心思想**：模仿 MLP 的层级结构，根据连接度着色

**实现**（`utils/visualization.py:124-260`）：

**特性**：

1. **层级布局**
   - 将神经元分成 5 层（类似 MLP）
   - 横向表示层级，纵向表示层内位置
   - 允许跨层连接（体现拓扑距离特性）

2. **连接度着色**
   ```python
   # 蓝色（低连接度）→ 白色（中等）→ 红色（高连接度）
   if degree_norm < 0.5:
       # 蓝色到白色
       r = degree_norm * 2
       g = degree_norm * 2
       b = 1.0
   else:
       # 白色到红色
       r = 1.0
       g = 1.0 - (degree_norm - 0.5) * 2
       b = 1.0 - (degree_norm - 0.5) * 2
   ```

3. **节点大小**
   - 根据连接度调整：100 + 400 * normalized_degree
   - 连接多的神经元更大

4. **边的粗细**
   - 根据权重强度调整：1 + 3 * (weight / max_weight)
   - 重要连接更粗

5. **弧形连线**
   - 使用 `connectionstyle='arc3,rad=0.1'`
   - 更美观，避免重叠

**效果**：
- ✅ 直观展示网络层级结构
- ✅ 一眼看出"枢纽神经元"（红色，大节点）
- ✅ 一眼看出"边缘神经元"（蓝色，小节点）
- ✅ 连线粗细表示连接重要性
- ✅ 允许跨层连接，体现拓扑特性

**示例输出**：
```
连接度统计:
  最高连接度神经元: 42 (连接度: 18)
  最低连接度神经元: 7 (连接度: 2)
  平均连接度: 8.5
```

---

## 实施步骤

### 步骤1：验证改进效果

运行优化后的训练：
```bash
cd v1.0.2
python run_with_config.py
```

### 步骤2：关键指标检查

**1. Epoch 5 变化率**
- 期望：<10%（而非 27%）
- 检查：训练日志中的 "Change rate"

**2. 剪枝/生长比例**
- 期望：2:1 左右（而非 10:1）
- 检查：训练日志中的 "Pruned" 和 "Added"

**3. 连接数变化**
- 期望：波动上升（而非持续下降）
- 检查：训练历史图中的"总连接数演化"

**4. 准确率**
- 期望：≥90%（而非 89%）
- 检查：最终测试准确率

### 步骤3：可视化验证

**1. 训练历史图**
- 查看"剪枝和新增"子图
- 期望：绿线（新增）显著提升

**2. 拓扑结构图**
- 查看层级网络图
- 期望：看到红色"枢纽神经元"和蓝色"边缘神经元"

**3. 连接年龄分布**
- 查看年龄分布直方图
- 期望：更平滑的分布（而非集中在保护期边界）

---

## 进一步优化方向

### 如果生长仍然不足

**选项A：进一步降低生长阈值**
```python
GROWTH_THRESHOLD = 0.05  # 从 0.1 降低到 0.05
```

**选项B：增加生长配额**
```python
max_new_connections = max(30, int(self.num_neurons * 0.03))  # 1500神经元->45
```

**选项C：调整动态阈值系数**
```python
effective_threshold = self.growth_threshold * 0.3  # 从 0.5 降低到 0.3
```

---

### 如果剪枝仍然过于激进

**选项A：降低剪枝比例**
```python
pruned = self._prune_connections_balanced(target_prune_ratio=0.5)  # 从 0.7 降低到 0.5
```

**选项B：增加保护期**
```python
PROTECTION_PERIOD = 80  # 从 60 增加到 80
```

**选项C：调整动态阈值系数**
```python
# 在 can_be_pruned_vectorized 中
dynamic_threshold = prune_threshold * (0.2 + 0.8 * age_ratio + 0.3 * (age_ratio > 1.0).float())
# 年龄=0: 阈值 = 0.002（更难剪枝）
# 年龄=60: 阈值 = 0.01（正常）
# 年龄>60: 阈值 = 0.013（更容易）
```

---

## 风险评估

### 低风险修改 ✅
- 调整生长配额和候选池大小
- 调整生长阈值
- 修改可视化代码

### 中风险修改 ⚠️
- 渐进式保护期（可能影响训练稳定性）
  - **缓解**：保留原始硬阈值作为备选
- 先生长再剪枝（改变执行顺序）
  - **缓解**：充分测试，必要时回退

### 高风险修改 ❌
- 无（本次改进未涉及核心算法）

---

## 总结

### 核心改进

1. **渐进式保护期**：避免 Epoch 5 的剧烈波动
2. **先生长再剪枝**：确保网络探索能力
3. **平衡剪枝机制**：限制单次剪枝数量
4. **增加生长配额**：大幅提升生长机会
5. **层级拓扑可视化**：直观展示网络结构

### 预期结果

- ✅ Epoch 5 变化率：27% → **<10%**
- ✅ 剪枝/生长比例：10:1 → **2:1**
- ✅ 连接数变化：持续下降 → **波动上升**
- ✅ 训练稳定性：显著提升
- ✅ 可视化效果：更直观、更美观

### 下一步

运行优化后的训练，验证改进效果：
```bash
cd v1.0.2
python run_with_config.py
```

---

**创建日期**：2026-02-04
**版本**：v1.0.2 可塑性算法改进
**状态**：代码已更新，待验证
