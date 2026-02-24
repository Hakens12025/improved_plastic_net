# 快速开始指南

## 项目已完成！

改进版神经可塑性网络已经完全实现并测试通过。

## 文件清单

### 核心代码
- ✓ `models/plastic_net.py` - 核心网络模型
- ✓ `models/topology_manager.py` - 拓扑管理器
- ✓ `models/connection_manager.py` - 连接管理器
- ✓ `training/engine.py` - 训练引擎
- ✓ `utils/visualization.py` - 可视化工具

### 实验脚本
- ✓ `experiments/mnist_baseline.py` - MNIST完整实验
- ✓ `simple_example.py` - 简单使用示例
- ✓ `test_quick.py` - 快速测试脚本

### 文档
- ✓ `claude.md` - 详细设计文档
- ✓ `README.md` - 项目说明
- ✓ `IMPLEMENTATION_SUMMARY.md` - 实现总结
- ✓ `requirements.txt` - 依赖项

## 三种使用方式

### 方式1：快速测试（推荐先运行）

验证所有模块是否正常工作：

```bash
cd C:\Users\21472\projects\deeplerning\improved_plastic_net
python test_quick.py
```

预期输出：
```
[OK] All modules imported successfully
[OK] Model created successfully
[OK] Forward pass successful
[OK] Neuroplasticity mechanism working
[SUCCESS] All tests passed!
```

### 方式2：简单示例

运行一个快速的演示（1个epoch，6000个样本）：

```bash
cd C:\Users\21472\projects\deeplerning\improved_plastic_net
python simple_example.py
```

这会：
- 创建一个小型模型（200神经元）
- 训练1个epoch
- 生成可视化图表

### 方式3：完整MNIST实验

运行完整的MNIST基准测试（5个epoch，60000个样本）：

```bash
cd C:\Users\21472\projects\deeplerning\improved_plastic_net\experiments
python mnist_baseline.py
```

这会：
- 创建完整模型（400神经元）
- 训练5个epoch
- 在测试集上评估
- 生成所有可视化
- 保存训练好的模型

## 预期结果

### 测试准确率
- **目标**：≥ 98%
- **训练时间**：约5-10分钟（CPU）

### 网络演化
- **初始连接数**：约40000
- **最终连接数**：约20000-30000
- **稀疏度**：60-80%
- **变化率**：5-10%（持续）

### 生成的文件
- `training_history.png` - 训练历史曲线
- `topology.png` - 网络拓扑结构
- `connection_age.png` - 连接年龄分布
- `improved_plastic_net_mnist.pth` - 训练好的模型

## 核心特性

### 1. 双向动态拓扑
- ✓ 剪枝无效连接
- ✓ 生成新连接
- ✓ 持续演化（5-10%变化率）

### 2. 最小作用量原理
- ✓ 相对拓扑距离（2-3跳）
- ✓ 共同激活检测
- ✓ 优先级排序

### 3. 连接生命周期
- ✓ 保护期机制（75步）
- ✓ 年龄追踪
- ✓ 快速淘汰

### 4. 自动调节
- ✓ 动态阈值调整
- ✓ 变化率控制
- ✓ 反馈机制

## 参数调整

### 如果想要更激进的演化
```python
model = ImprovedPlasticNet(
    num_neurons=400,
    input_dim=784,
    output_dim=10,
    initial_sparsity=0.3  # 更少的初始连接
)
# 在模型内部调整
model.protection_period = 50  # 更短的保护期
model.target_change_rate_max = 0.15  # 允许更高的变化率
```

### 如果想要更稳定的网络
```python
model = ImprovedPlasticNet(
    num_neurons=400,
    input_dim=784,
    output_dim=10,
    initial_sparsity=0.7  # 更多的初始连接
)
# 在模型内部调整
model.protection_period = 100  # 更长的保护期
model.target_change_rate_max = 0.08  # 限制变化率
```

### 如果想要更快的训练
```python
engine = ImprovedEvolutionEngine(
    model=model,
    device=device,
    lr=0.002,  # 更高的学习率
    plasticity_interval=100  # 更少的拓扑更新
)
```

## 故障排除

### 问题1：内存不足
**解决方案**：
- 减少神经元数量（如300）
- 减少batch_size（如64）
- 增加initial_sparsity（如0.7）

### 问题2：训练太慢
**解决方案**：
- 使用GPU（如果可用）
- 增加plasticity_interval（如100）
- 减少num_workers（如2）

### 问题3：准确率不理想
**解决方案**：
- 增加神经元数量（如500）
- 增加训练轮数（如10）
- 调整学习率（如0.0005）

### 问题4：连接变化太剧烈
**解决方案**：
- 增加保护期
- 降低target_change_rate_max
- 提高growth_threshold

## 下一步

### 立即可做
1. ✓ 运行快速测试
2. ✓ 运行简单示例
3. ✓ 运行完整实验
4. ✓ 查看可视化结果
5. ✓ 调整参数实验

### 进阶实验
1. 在Fashion-MNIST上测试
2. 尝试不同的神经元数量
3. 调整可塑性参数
4. 对比不同的演化策略

### 未来扩展
1. 视频序列训练
2. 多网络集成
3. 在线学习

## 获取帮助

### 查看详细文档
```bash
# 设计理念和技术细节
cat claude.md

# 实现总结
cat IMPLEMENTATION_SUMMARY.md

# 快速说明
cat README.md
```

### 查看代码注释
所有核心文件都有详细的注释和文档字符串。

## 总结

✓ 所有代码已实现并测试通过
✓ 文档完整详细
✓ 可以立即运行实验
✓ 支持灵活的参数调整

**祝实验顺利！**

---

**项目位置**：`C:\Users\21472\projects\deeplerning\improved_plastic_net`
**创建日期**：2026-02-02
**状态**：✓ 完成并可用
