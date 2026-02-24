# 快速运行指南

## 运行优化后的代码

### 方法1：使用配置文件运行（推荐）

```bash
cd projects/deepLerning/improved_plastic_net
python run_with_config.py
```

这将使用 `config.py` 中的所有参数运行实验。

### 方法2：直接运行实验脚本

```bash
cd projects/deepLerning/improved_plastic_net/experiments
python v1_0_3_mnist_baseline.py
```

### 方法3：使用批处理脚本（Windows）

```bash
cd projects/deepLerning/improved_plastic_net
run_experiment.bat
```

---

## 调整参数

### 修改神经元数量

编辑 `config.py`：

```python
NUM_NEURONS = 2000  # 从1000改为2000
```

其他参数会自动调整：
- BATCH_SIZE: 128 → 96
- PLASTICITY_INTERVAL: 40 → 60
- PROTECTION_PERIOD: 60 → 50

### 修改训练轮数

```python
EPOCHS = 20  # 从10改为20
```

### 修改初始稀疏度

```python
INITIAL_SPARSITY = 0.7  # 从0.6改为0.7（更稀疏）
```

---

## 查看结果

### 训练过程中

实时输出会显示：
- 每个epoch的训练和测试准确率
- 可塑性更新信息（剪枝/生长的连接数）
- 当前稀疏度和连接数
- 变化率

### 训练完成后

生成的文件：
1. **v1_0_3_optimized_plastic_net_mnist.pth** - 模型权重
2. **v1_0_3_optimized_plastic_net_mnist_performance.json** - 性能数据
3. **training_history.png** - 训练历史图表
4. **topology.png** - 网络拓扑结构
5. **connection_age.png** - 连接年龄分布

---

## 验证优化效果

### 检查可塑性活跃度

在训练输出中查找：
```
Pruned: XX | Added: YY
```

期望值：
- Pruned: 10-30个
- Added: 10-20个
- 变化率: 8-12%（1000神经元）或 5-10%（2000+神经元）

### 检查初始化

在训练开始时查找：
```
[INFO] Topology-aware init: XXXXX connections from YYYYY valid candidates
```

这表示智能初始化正在工作。

### 检查准确率

期望值：
- 1000神经元：≥ 90%
- 2000神经元：≥ 91%
- 3000神经元：≥ 92%

---

## 常见问题

### Q: 内存不足怎么办？

A: 减少批次大小或神经元数量：
```python
BATCH_SIZE = 64  # 从128减少
NUM_NEURONS = 800  # 从1000减少
```

### Q: 训练太慢怎么办？

A: 增加可塑性更新间隔：
```python
PLASTICITY_INTERVAL = 60  # 从40增加
```

### Q: 可塑性变化太激进怎么办？

A: 调整阈值：
```python
GROWTH_THRESHOLD = 0.25  # 从0.2增加
PRUNE_THRESHOLD = 0.012  # 从0.015减少
PLASTICITY_INTERVAL = 50  # 从40增加
```

### Q: 准确率不够高怎么办？

A: 增加训练轮数和神经元数量：
```python
EPOCHS = 15  # 从10增加
NUM_NEURONS = 1500  # 从1000增加
```

---

## 监控GPU使用

### 实时监控

```bash
# 每秒更新一次
nvidia-smi -l 1
```

### 查看内存使用

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## 对比不同配置

### 测试1000神经元

```python
# config.py
NUM_NEURONS = 1000
```

运行并记录结果。

### 测试2000神经元

```python
# config.py
NUM_NEURONS = 2000
```

运行并对比：
- 训练时间
- 准确率
- 内存使用
- 可塑性活跃度

---

## 保存实验结果

### 重命名输出文件

```bash
# 保存1000神经元的结果
mv v1_0_3_optimized_plastic_net_mnist.pth results_1000neurons.pth
mv training_history.png results_1000neurons_history.png

# 保存2000神经元的结果
mv v1_0_3_optimized_plastic_net_mnist.pth results_2000neurons.pth
mv training_history.png results_2000neurons_history.png
```

### 创建结果目录

```bash
mkdir results
mv *.pth *.png *.json results/
```

---

## 下一步

1. **验证基础功能**：运行1000神经元配置，确认所有优化正常工作
2. **测试扩展性**：尝试2000和3000神经元配置
3. **调优参数**：根据结果微调参数
4. **对比分析**：比较不同配置的性能

---

**提示**：首次运行会下载Fashion-MNIST数据集，可能需要几分钟。

