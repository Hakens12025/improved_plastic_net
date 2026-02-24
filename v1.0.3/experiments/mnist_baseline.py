"""
Fashion-MNIST基准实验 - 改进版神经可塑性网络

运行此脚本以训练和评估改进版网络
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import ImprovedPlasticNet
from training import ImprovedEvolutionEngine
from utils import plot_all_visualizations


def run_fashion_mnist_experiment(
    num_neurons: int = 400,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    plasticity_interval: int = 200,  # 减少更新频率：50 -> 200
    device: str = 'auto'
):
    """
    运行MNIST实验

    Args:
        num_neurons: 内部神经元数量
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        plasticity_interval: 神经可塑性更新间隔
        device: 计算设备 ('auto', 'cuda', 'cpu')
    """
    print("="*80)
    print("改进版神经可塑性网络 - Fashion-MNIST基准实验")
    print("="*80)

    # 设备选择
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"\n使用设备: {device}")

    # 数据准备
    print("\n准备数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.FashionMNIST(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )

    test_set = torchvision.datasets.FashionMNIST(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 设置为0避免Windows多进程问题
        pin_memory=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
        num_workers=0  # 设置为0避免Windows多进程问题
    )

    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(test_set)}")

    # 创建模型
    print(f"\n创建模型...")
    print(f"  内部神经元数: {num_neurons}")
    print(f"  输入维度: 784 (28x28)")
    print(f"  输出维度: 10")

    model = ImprovedPlasticNet(
        num_neurons=num_neurons,
        input_dim=784,
        output_dim=10,
        iterations=5,
        initial_sparsity=0.5
    )

    print(f"  初始连接数: {model.adj_mask.sum().item():.0f}")
    print(f"  初始稀疏度: {model.get_sparsity():.2%}")

    # 创建训练引擎
    print(f"\n创建训练引擎...")
    engine = ImprovedEvolutionEngine(
        model=model,
        device=device,
        lr=lr,
        plasticity_interval=plasticity_interval
    )

    # 训练
    print(f"\n开始训练...")
    engine.train_and_evolve(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        verbose=True
    )

    # 最终测试
    print(f"\n最终测试...")
    final_test_acc = engine.test(test_loader)
    print(f"最终测试准确率: {final_test_acc:.2f}%")

    # 生成可视化
    print(f"\n生成可视化图表...")
    epoch_stats = engine.get_epoch_stats()
    plot_all_visualizations(model, epoch_stats, output_dir=".")

    # 保存模型
    print(f"\n保存模型...")
    engine.save_model("improved_plastic_net_fashion_mnist.pth")

    # 打印最终统计
    print(f"\n{'='*80}")
    print("实验完成！最终统计:")
    print(f"{'='*80}")
    final_stats = model.get_statistics()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return model, engine


if __name__ == "__main__":
    # 运行实验
    model, engine = run_fashion_mnist_experiment(
        num_neurons=1000,
        epochs=5,
        batch_size=128,
        lr=0.001,
        plasticity_interval=50,
        device='auto'
    )
