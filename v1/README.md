# 改进版神经可塑性网络

一个受大脑神经可塑性启发的深度学习网络，实现了双向动态拓扑调整、最小作用量原理和持续演化机制。

## 项目结构

```
improved_plastic_net/
├── README.md                        # 项目总览（本文件）
├── VERSION_MANAGEMENT_GUIDE.md      # 版本管理规范
├── requirements.txt                 # 共享依赖
├── data/                            # 共享数据集
│
├── v1.0.2/                          # 版本 1.0.2（当前稳定版）
│   ├── README.md                    # 版本说明
│   ├── VERSION_INFO.md              # 版本信息
│   ├── config.py                    # 配置文件
│   ├── run_with_config.py           # 运行脚本
│   ├── models/                      # 模型代码
│   ├── training/                    # 训练代码
│   ├── experiments/                 # 实验脚本
│   ├── utils/                       # 工具函数
│   ├── results/                     # 训练结果
│   └── docs/                        # 版本文档
│
└── notebooks/                       # Jupyter notebooks
```

## 快速开始

### 运行 v1.0.2

```bash
cd v1.0.2
python run_with_config.py
```

详细说明请查看 `v1.0.2/README.md`

## 版本说明

### v1.0.2（当前版本）
- **状态**: 稳定版
- **发布日期**: 2026-02-03
- **主要特性**:
  - 智能初始化（基于拓扑距离）
  - 动态参数扩展（支持 1000-3000 神经元）
  - 训练优化（学习率调度 + 数据增强）
  - 性能优化（向量化操作 + GPU加速）
- **性能指标**:
  - 测试准确率: 88.53%
  - 训练时间: ~30s/epoch
  - 最终连接数: 267,416

## 版本管理

本项目采用版本隔离策略，每个版本使用独立目录：

- ✅ 每个版本完全独立，可独立运行
- ✅ 便于版本对比和回退
- ✅ 避免代码混乱

详细规范请查看 `VERSION_MANAGEMENT_GUIDE.md`

## 依赖项

- Python 3.8+
- PyTorch 1.9+
- torchvision
- matplotlib
- networkx
- numpy

安装依赖：
```bash
pip install -r requirements.txt
```

## 文档

### 项目文档
- `VERSION_MANAGEMENT_GUIDE.md` - 版本管理规范
- `PROJECT_SUMMARY.md` - 项目总结
- `VERSION_COMPARISON.md` - 版本对比

### v1.0.2 文档
- `v1.0.2/README.md` - 版本说明
- `v1.0.2/docs/OPTIMIZATION_IMPLEMENTATION_SUMMARY.md` - 优化实施总结
- `v1.0.2/docs/QUICK_RUN_GUIDE.md` - 快速运行指南
- `v1.0.2/docs/TRAINING_CONFIG.md` - 训练配置说明

## 核心特性

### 1. 动态拓扑调整
- 基于信用分数的连接剪枝
- 基于共激活的连接生长
- 拓扑距离约束（2-3跳）

### 2. 智能初始化
- 基于拓扑距离的连接采样
- 确保初始连接符合约束
- 提高可塑性效率

### 3. 持续演化
- 自适应阈值调整
- 目标变化率控制（5-10%）
- 连接保护期机制

### 4. 性能优化
- 向量化剪枝和生长操作
- 预计算拓扑距离
- GPU加速训练
- 混合精度训练

## 使用示例

### 基础训练
```bash
cd v1.0.2
python run_with_config.py
```

### 自定义配置
编辑 `v1.0.2/config.py`：
```python
NUM_NEURONS = 2000          # 神经元数量
EPOCHS = 20                 # 训练轮数
INITIAL_SPARSITY = 0.4      # 初始稀疏度
```

### 查看结果
训练完成后，查看 `v1.0.2/results/` 目录：
- `comprehensive_analysis.png` - 综合分析图
- `*.pth` - 模型权重
- `*_performance.json` - 性能数据

## 性能基准

### Fashion-MNIST
- 测试准确率: 88.53%
- 训练时间: ~30s/epoch (1000神经元, GPU)
- 最终稀疏度: 46.46%

## 已知问题

- 准确率略低于预期（目标 90%+）
- 需要进一步调优参数

改进建议请查看 `v1.0.2/docs/OPTIMIZATION_IMPLEMENTATION_SUMMARY.md`

## 引用

如果您在研究中使用了本项目，请引用：

```
改进版神经可塑性网络
基于最小作用量原理和持续演化机制的动态拓扑神经网络
2026
```

## 许可证

本项目仅供学习和研究使用。

---

**最后更新**: 2026-02-04
**当前版本**: v1.0.2
**项目状态**: 活跃开发中
