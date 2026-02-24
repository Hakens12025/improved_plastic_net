# GPU环境使用说明

## 环境要求

- **Conda环境**: `pt_gpu`
- **数据集**: Fashion-MNIST（自动下载）
- **GPU**: 支持CUDA的NVIDIA GPU（推荐）

## 快速开始

### Windows系统

双击运行 `run_experiment.bat` 或在命令行中执行：

```cmd
run_experiment.bat
```

### Linux/Mac系统

在终端中执行：

```bash
bash run_experiment.sh
```

或者：

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

## 手动运行

如果启动脚本无法使用，可以手动执行以下步骤：

### 1. 激活conda环境

```bash
conda activate pt_gpu
```

### 2. 检查GPU状态

```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

### 3. 运行实验

```bash
cd experiments
python v1_0_2_mnist_baseline.py
```

## 版本说明

### v1.0.2 优化版（推荐）

- **文件**: `experiments/v1_0_2_mnist_baseline.py`
- **性能**: 6-12倍训练速度提升
- **内存**: 4-8倍内存节省
- **特性**: 向量化操作、预计算拓扑、优化前向传播

### v1.0.0 基线版

- **文件**: `experiments/mnist_baseline.py`
- **用途**: 对比测试

## 预期结果

### 训练性能（400神经元，5轮）

- **训练时间**: 25-100秒（GPU）
- **测试准确率**: ≥ 98%
- **最终稀疏度**: 60-80%
- **连接变化率**: 5-10%

### 输出文件

训练完成后会在项目根目录生成：

- `training_history.png` - 训练历史曲线
- `topology.png` - 网络拓扑结构
- `connection_age.png` - 连接年龄分布
- `improved_plastic_net_fashion_mnist.pth` - 训练好的模型

## 参数调整

如需调整训练参数，编辑实验脚本中的参数：

```python
run_mnist_experiment_v1_0_2(
    num_neurons=400,        # 神经元数量
    epochs=5,               # 训练轮数
    batch_size=128,         # 批次大小
    lr=0.001,              # 学习率
    plasticity_interval=50, # 可塑性更新间隔
    device='auto'          # 'auto', 'cuda', 'cpu'
)
```

## 故障排除

### 问题1: conda命令未找到

**解决方案**: 确保已安装Anaconda或Miniconda，并将conda添加到系统PATH

### 问题2: pt_gpu环境不存在

**解决方案**: 创建pt_gpu环境

```bash
conda create -n pt_gpu python=3.8
conda activate pt_gpu
pip install torch torchvision matplotlib networkx numpy
```

### 问题3: CUDA不可用

**解决方案**:
- 检查GPU驱动是否安装
- 安装支持CUDA的PyTorch版本：
  ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  ```

### 问题4: 内存不足

**解决方案**: 减少神经元数量或批次大小

```python
num_neurons=200  # 从400减少到200
batch_size=64    # 从128减少到64
```

## 项目结构

```
improved_plastic_net/
├── models/                      # 核心模型
│   ├── v1_0_2_plastic_net.py   # 优化版网络
│   ├── v1_0_2_topology_manager.py
│   └── v1_0_2_connection_manager.py
├── training/                    # 训练引擎
│   └── v1_0_2_engine.py
├── experiments/                 # 实验脚本
│   ├── v1_0_2_mnist_baseline.py # 优化版实验
│   └── mnist_baseline.py        # 基线版实验
├── utils/                       # 工具
│   └── visualization.py
├── run_experiment.bat          # Windows启动脚本
├── run_experiment.sh           # Linux/Mac启动脚本
└── GPU_USAGE.md               # 本文档
```

## 更多信息

- **详细设计**: 查看 `claude.md`
- **实现总结**: 查看 `IMPLEMENTATION_SUMMARY.md`
- **快速开始**: 查看 `QUICK_START.md`
- **v1.0.2说明**: 查看 `v1.0.2_README.md`
