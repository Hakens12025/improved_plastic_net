# AGENTS.md - AI 编程 Agent 指南

本文档为在改进版神经可塑性网络项目中工作的 AI 编程 Agent 提供指导规范。

## 项目概述

本项目是一个受大脑神经可塑性启发的深度学习网络，实现了双向动态拓扑调整、最小作用量原理和持续演化机制。项目采用版本隔离策略，每个主要版本都有独立的目录（如 `v1.0.2/`）。

## 构建与运行命令

### 运行主实验 (v1.0.2)
```bash
cd v1.0.2 && python run_with_config.py
```

### 使用自定义配置运行
```bash
cd v1.0.2 && python run_with_config.py --epochs 10 --num_neurons 1500
```

### 运行特定实验脚本
```bash
cd v1.0.2/experiments && python v1_0_2_mnist_baseline.py
```

### 安装依赖
```bash
pip install -r requirements.txt
```

所需依赖包：
- torch >= 1.9.0
- torchvision >= 0.10.0
- matplotlib >= 3.3.0
- networkx >= 2.5
- numpy >= 1.19.0

## 代码风格规范

### 导入顺序
- 标准库导入在前，其次是第三方库，最后是本地导入
- 按类型分组，组间用空行分隔
- 使用绝对导入进行包内导入

```python
import torch
import torch.nn as nn
from typing import Tuple, List
from collections import deque

from .v1_0_2_topology_manager import OptimizedTopologyManager
from .v1_0_2_connection_manager import OptimizedConnectionManager
```

### 命名规范
- **类名**：帕斯卡命名法 (PascalCase)，例如 `OptimizedPlasticNet`
- **函数/变量**：蛇形命名法 (snake_case)，例如 `apply_neuroplasticity`
- **常量**：全大写下划线分隔 (UPPER_SNAKE_CASE)，例如 `PRUNE_THRESHOLD`
- **私有成员**：前导下划线，例如 `_temp_buffer`
- **类型变量**：帕斯卡命名法 (PascalCase)，例如 `Tensor`

### 类型提示
- 所有函数参数和返回值必须使用类型提示
- 从 `typing` 模块导入类型
- 使用 `Optional[T]` 表示可空值，`List[T]` 表示列表，`Tuple[T, U]` 表示元组

```python
def forward_optimized(self, x: torch.Tensor) -> torch.Tensor:
def apply_neuroplasticity_optimized(self) -> Tuple[int, int]:
def get_statistics(self) -> dict:
```

### 文档字符串
- 所有文档字符串使用三引号
- 包含 Args 部分说明参数
- 包含 Returns 部分说明返回值
- 代码解释使用中文注释，代码元素使用英文

```python
def __init__(
    self,
    num_neurons: int,
    input_dim: int,
    output_dim: int,
    iterations: int = 5
):
    """
    初始化优化版网络

    Args:
        num_neurons: 内部神经元数量
        input_dim: 输入维度
        output_dim: 输出维度
        iterations: 内部迭代次数

    Returns:
        输出张量 (batch_size, output_dim)
    """
```

### 错误处理
- 推理操作使用 `with torch.no_grad():`
- 外部资源访问使用 try-except
- 尽早验证输入并抛出描述性错误
- 为错误条件使用有意义的变量名

```python
if not can_prune_mask.any():
    return 0

if len(prune_indices) == 0:
    return 0
```

### 代码结构
- 每个函数最多约 100 行
- 使用版本前缀文件：`v1_0_2_*.py`
- 主要代码放在 `models/`、`training/`、`utils/` 目录
- 实验脚本放在 `experiments/` 目录
- 结果保存在 `results/` 或 `experiments/` 目录

### 性能优化标记
- 使用表情符号标记优化：🚀 表示性能优化，🔥 表示问题修复
- 在注释中说明向量化操作
- 在安全的情况下使用原地操作 (`.half()`、`.mul_()`)

```python
# 🚀 优化1：使用更紧凑的数据类型
self.credit_score = self.credit_score.half()

# 🔥 改进：先生长，再剪枝
added = self._grow_connections_vectorized()
```

### Git 工作流
- 重大更改创建新的版本目录
- 使用格式：`v1.0.X/`
- 提交信息使用英文祈使句，说明"为什么"
- 永远不要提交生成的文件 (.pth、.png、.json 结果)

## 项目结构

```
improved_plastic_net/
├── v1.0.2/                    # 当前稳定版本
│   ├── models/                # 神经网络模型
│   ├── training/              # 训练引擎
│   ├── experiments/           # 实验脚本
│   ├── utils/                 # 工具函数
│   ├── results/               # 输出结果
│   ├── config.py              # 配置文件
│   └── run_with_config.py     # 入口脚本
├── data/                      # 共享数据集 (MNIST, FashionMNIST)
└── requirements.txt           # 共享依赖
```

## 版本管理
- 每个版本完全独立，有自己的目录
- 使用版本前缀模块名：`v1_0_2_*`
- 新功能 → 新版本号 → 新目录
- 详情参见 `VERSION_MANAGEMENT_GUIDE.md`
