# 训练配置文件
# 修改此文件中的参数来调整训练设置

# ========================================
# 模型配置
# ========================================

# 神经元数量（内部隐藏层）
# 优化：增加神经元数量，提升模型容量
NUM_NEURONS = 1500

# 内部迭代次数（每次前向传播的迭代轮数）
ITERATIONS = 5

# 初始稀疏度（0.75表示初始时75%的连接被剪枝，只保留25%）
# 优化：大幅提高初始稀疏度，避免 Epoch 5 的大规模剪枝
# 让网络从稀疏状态开始，通过可塑性机制逐步生长需要的连接
INITIAL_SPARSITY = 0.75

# ========================================
# 训练配置
# ========================================

# 训练轮数
# 优化：增加训练轮数，让网络有更多时间学习和调整拓扑
EPOCHS = 15

# 批次大小
# Stage 3 优化：根据神经元数量动态调整
if NUM_NEURONS >= 3000:
    BATCH_SIZE = 64
elif NUM_NEURONS >= 2000:
    BATCH_SIZE = 96
else:
    BATCH_SIZE = 128

# 学习率
# 优化：降低学习率，配合更多训练轮数，实现更稳定的收敛
LEARNING_RATE = 0.0008

# 神经可塑性更新间隔（每多少个batch更新一次拓扑）
# Stage 1 优化：适度增加更新频率
# Stage 3 优化：根据神经元数量动态调整
if NUM_NEURONS >= 3000:
    PLASTICITY_INTERVAL = 80
elif NUM_NEURONS >= 2000:
    PLASTICITY_INTERVAL = 60
else:
    PLASTICITY_INTERVAL = 40

# 设备选择 ('auto', 'cuda', 'cpu')
DEVICE = 'auto'

# ========================================
# 神经可塑性参数
# ========================================

# 连接保护期（新连接在多少步内不会被剪枝）
# Stage 1 优化：温和缩短保护期
# Stage 3 优化：根据神经元数量动态调整
if NUM_NEURONS >= 3000:
    PROTECTION_PERIOD = 40
elif NUM_NEURONS >= 2000:
    PROTECTION_PERIOD = 50
else:
    PROTECTION_PERIOD = 60

# 剪枝阈值（初始值，会动态调整）
# 优化：降低剪枝阈值，避免过于激进的剪枝（如 Epoch 5 的大规模剪枝）
PRUNE_THRESHOLD = 0.01

# 生长阈值（初始值，会动态调整）
# 优化：大幅降低生长阈值，增加生长机会，平衡剪枝和生长
GROWTH_THRESHOLD = 0.1

# 目标变化率范围
TARGET_CHANGE_RATE_MIN = 0.05  # 5%
TARGET_CHANGE_RATE_MAX = 0.10  # 10%

# ========================================
# 拓扑距离参数
# ========================================

# 允许连接的最小相对距离
MIN_DISTANCE = 2

# 允许连接的最大相对距离
MAX_DISTANCE = 3

# ========================================
# 性能优化参数
# ========================================

# 是否使用混合精度训练
USE_AMP = True

# DataLoader工作进程数（Windows建议设为0）
NUM_WORKERS = 0

# 是否使用pin_memory（GPU训练时建议True）
PIN_MEMORY = True

# ========================================
# 输出配置
# ========================================

# 模型保存路径
MODEL_SAVE_PATH = 'v1_0_2_optimized_plastic_net_fashion_mnist.pth'

# 性能数据保存路径
PERFORMANCE_SAVE_PATH = 'v1_0_2_optimized_plastic_net_fashion_mnist_performance.json'

# 是否显示详细训练日志
VERBOSE = True

# ========================================
# 预期性能指标（参考）
# ========================================

# 预期训练时间（1000神经元，10轮，GPU）
# - 总训练时间: 100-300秒
# - 平均每轮: 10-30秒
# - 测试准确率: ≥ 98%
# - 最终稀疏度: 60-80%
# - 连接变化率: 5-10%

# ========================================
# 参数调优建议
# ========================================

# 如果训练太慢：
# - 减少 NUM_NEURONS (1000 -> 600)
# - 增加 PLASTICITY_INTERVAL (50 -> 100)
# - 减少 EPOCHS (10 -> 5)

# 如果准确率不够：
# - 增加 NUM_NEURONS (1000 -> 1500)
# - 增加 ITERATIONS (5 -> 7)
# - 增加 EPOCHS (10 -> 15)
# - 调整 LEARNING_RATE (0.001 -> 0.0005)

# 如果连接变化率太高（>15%）：
# - 增加 PROTECTION_PERIOD (75 -> 100)
# - 提高 GROWTH_THRESHOLD (0.3 -> 0.5)

# 如果连接变化率太低（<3%）：
# - 减少 PROTECTION_PERIOD (75 -> 50)
# - 降低 GROWTH_THRESHOLD (0.3 -> 0.2)
# - 减少 PLASTICITY_INTERVAL (50 -> 30)

# 如果内存不足：
# - 减少 NUM_NEURONS (1000 -> 600)
# - 减少 BATCH_SIZE (128 -> 64)
# - 增加 INITIAL_SPARSITY (0.5 -> 0.7)
