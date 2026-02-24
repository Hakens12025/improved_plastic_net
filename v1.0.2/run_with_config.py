"""
使用配置文件运行实验 - 方便参数调整

直接运行此脚本，参数从config.py读取
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from config import (
    NUM_NEURONS, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    PLASTICITY_INTERVAL, DEVICE
)

# 导入实验函数
from experiments.v1_0_2_mnist_baseline import run_mnist_experiment_v1_0_2


def main():
    """主函数：使用配置文件运行实验"""

    print("=" * 80)
    print("使用配置文件运行实验")
    print("=" * 80)
    print(f"\n当前配置:")
    print(f"  神经元数量: {NUM_NEURONS}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  可塑性更新间隔: {PLASTICITY_INTERVAL}")
    print(f"  设备: {DEVICE}")
    print("\n提示: 修改 config.py 文件可以调整这些参数")
    print("=" * 80)

    # 运行实验
    model, engine, perf_summary = run_mnist_experiment_v1_0_2(
        num_neurons=NUM_NEURONS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        plasticity_interval=PLASTICITY_INTERVAL,
        device=DEVICE
    )

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)

    return model, engine, perf_summary


if __name__ == "__main__":
    main()
