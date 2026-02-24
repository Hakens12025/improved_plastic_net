"""
可视化工具 - 网络拓扑和训练过程可视化

核心功能：
1. 连接演化曲线
2. 变化率曲线
3. 拓扑结构图
4. 连接年龄分布
"""

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
import torch
from typing import List, Dict
import platform

# 设置中文字体支持
def setup_chinese_font():
    """配置matplotlib的中文字体"""
    system = platform.system()
    if system == 'Windows':
        # Windows系统使用微软雅黑
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
    else:  # Linux
        matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False

# 初始化字体设置
setup_chinese_font()


def plot_training_history(epoch_stats: List[Dict], save_path: str = None):
    """
    绘制训练历史

    Args:
        epoch_stats: epoch统计信息列表
        save_path: 保存路径（可选）
    """
    if not epoch_stats:
        print("没有统计数据可绘制")
        return

    epochs = [s['epoch'] for s in epoch_stats]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('训练历史', fontsize=16, fontweight='bold')

    # 1. 损失和准确率
    ax = axes[0, 0]
    ax.plot(epochs, [s['train_loss'] for s in epoch_stats], 'b-', label='Train Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练损失')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 准确率
    ax = axes[0, 1]
    ax.plot(epochs, [s['train_acc'] for s in epoch_stats], 'g-', label='Train Acc', linewidth=2)
    if epoch_stats[0].get('test_acc') is not None:
        ax.plot(epochs, [s['test_acc'] for s in epoch_stats], 'r-', label='Test Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('准确率')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 连接数量变化
    ax = axes[0, 2]
    ax.plot(epochs, [s['total_connections'] for s in epoch_stats], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('连接数')
    ax.set_title('总连接数演化')
    ax.grid(True, alpha=0.3)

    # 4. 剪枝和新增
    ax = axes[1, 0]
    ax.plot(epochs, [s['pruned_last'] for s in epoch_stats], 'r-', label='剪枝', linewidth=2)
    ax.plot(epochs, [s['added_last'] for s in epoch_stats], 'g-', label='新增', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('数量')
    ax.set_title('连接变化（剪枝 vs 新增）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 变化率
    ax = axes[1, 1]
    change_rates = [s['change_rate'] * 100 for s in epoch_stats]
    ax.plot(epochs, change_rates, 'orange', linewidth=2)
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='目标下限 (5%)')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='目标上限 (10%)')
    ax.fill_between(epochs, 5, 10, alpha=0.2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('变化率 (%)')
    ax.set_title('连接变化率（目标: 5-10%）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 稀疏度
    ax = axes[1, 2]
    sparsity = [s['sparsity'] * 100 for s in epoch_stats]
    ax.plot(epochs, sparsity, 'brown', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('稀疏度 (%)')
    ax.set_title('网络稀疏度')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")

    plt.show()


def plot_topology(model, max_neurons: int = 50, save_path: str = None):
    """
    绘制网络拓扑结构（改进版：层级结构 + 连接度着色）

    Args:
        model: 神经网络模型
        max_neurons: 最大显示神经元数
        save_path: 保存路径（可选）
    """
    adj_mask = model.adj_mask.cpu().numpy()
    weights = model.weights.data.cpu().numpy()
    n = min(model.num_neurons, max_neurons)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])
    fig.suptitle(f'网络拓扑结构 (显示前{n}个神经元)', fontsize=16, fontweight='bold')

    # ==================== 左图：邻接矩阵热力图 ====================
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(adj_mask[:n, :n], cmap='Blues', interpolation='nearest')
    ax1.set_title('邻接矩阵热力图', fontsize=12, fontweight='bold')
    ax1.set_xlabel('目标神经元')
    ax1.set_ylabel('源神经元')
    ax1.grid(False)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('连接强度')

    # ==================== 右图：层级网络图（类似MLP） ====================
    ax2 = fig.add_subplot(gs[0, 1])

    # 构建有向图
    G = nx.DiGraph()

    # 添加边（带权重）
    for i in range(n):
        for j in range(n):
            if adj_mask[i, j] > 0:
                G.add_edge(i, j, weight=abs(weights[i, j]))

    if G.number_of_edges() == 0:
        ax2.text(0.5, 0.5, '没有活跃连接', ha='center', va='center', fontsize=14)
        ax2.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return

    # 🔥 计算每个神经元的连接度（入度 + 出度）
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    total_degree = {node: in_degree.get(node, 0) + out_degree.get(node, 0) for node in G.nodes()}

    # 归一化连接度到 [0, 1]
    max_degree = max(total_degree.values()) if total_degree else 1
    min_degree = min(total_degree.values()) if total_degree else 0
    degree_range = max_degree - min_degree if max_degree > min_degree else 1

    normalized_degrees = {
        node: (total_degree[node] - min_degree) / degree_range
        for node in G.nodes()
    }

    # 🔥 层级布局（模仿MLP）
    # 将神经元分成若干层（根据拓扑距离）
    num_layers = 5  # 分成5层
    layer_size = n // num_layers + 1

    pos = {}
    for i in range(n):
        layer = i // layer_size
        pos_in_layer = i % layer_size
        # x: 层级位置（横向）
        # y: 层内位置（纵向）
        x = layer * 2.0
        y = pos_in_layer - layer_size / 2.0
        pos[i] = (x, y)

    # 🔥 根据连接度着色：蓝色（少连接）→ 红色（多连接）
    node_colors = []
    for node in G.nodes():
        degree_norm = normalized_degrees[node]
        # 使用蓝-白-红渐变
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
        node_colors.append((r, g, b))

    # 节点大小也根据连接度调整
    node_sizes = [100 + 400 * normalized_degrees[node] for node in G.nodes()]

    # 🔥 边的粗细根据权重调整
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 3 * (w / max_weight) for w in edge_weights]

    # 绘制网络
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax2, alpha=0.9)

    # 🔥 允许跨层连接（体现拓扑距离特性）
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                           width=edge_widths, alpha=0.4,
                           arrows=True, arrowsize=15,
                           arrowstyle='->', ax=ax2,
                           connectionstyle='arc3,rad=0.1')  # 弧形连线，更美观

    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax2)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='低连接度'),
        Patch(facecolor='white', edgecolor='black', label='中等连接度'),
        Patch(facecolor='red', label='高连接度')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax2.set_title(f'层级网络图 (连接数: {G.number_of_edges()})', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 添加统计信息
    stats_text = f"节点数: {G.number_of_nodes()}\n"
    stats_text += f"连接数: {G.number_of_edges()}\n"
    stats_text += f"平均连接度: {sum(total_degree.values()) / len(total_degree):.1f}\n"
    stats_text += f"最大连接度: {max_degree}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"拓扑结构图已保存到: {save_path}")

    plt.show()

    # 打印连接度统计
    print(f"\n连接度统计:")
    print(f"  最高连接度神经元: {max(total_degree, key=total_degree.get)} (连接度: {max_degree})")
    print(f"  最低连接度神经元: {min(total_degree, key=total_degree.get)} (连接度: {min_degree})")
    print(f"  平均连接度: {sum(total_degree.values()) / len(total_degree):.2f}")


def plot_connection_age_distribution(model, save_path: str = None):
    """
    绘制连接年龄分布

    Args:
        model: 神经网络模型
        save_path: 保存路径（可选）
    """
    connection_age = model.connection_manager.connection_age.cpu().numpy()
    adj_mask = model.adj_mask.cpu().numpy()

    # 只统计活跃连接的年龄
    active_ages = connection_age[adj_mask > 0]

    if len(active_ages) == 0:
        print("没有活跃连接")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('连接年龄分析', fontsize=14, fontweight='bold')

    # 1. 年龄分布直方图
    ax1.hist(active_ages, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=model.connection_manager.protection_period,
                color='red', linestyle='--', linewidth=2,
                label=f'保护期 ({model.connection_manager.protection_period}步)')
    ax1.set_xlabel('连接年龄（步数）')
    ax1.set_ylabel('连接数量')
    ax1.set_title('连接年龄分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 累积分布
    sorted_ages = np.sort(active_ages)
    cumulative = np.arange(1, len(sorted_ages) + 1) / len(sorted_ages) * 100

    ax2.plot(sorted_ages, cumulative, linewidth=2, color='green')
    ax2.axvline(x=model.connection_manager.protection_period,
                color='red', linestyle='--', linewidth=2,
                label=f'保护期')
    ax2.set_xlabel('连接年龄（步数）')
    ax2.set_ylabel('累积百分比 (%)')
    ax2.set_title('连接年龄累积分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"连接年龄分布图已保存到: {save_path}")

    plt.show()

    # 打印统计信息
    print(f"\n连接年龄统计:")
    print(f"  总连接数: {len(active_ages)}")
    print(f"  平均年龄: {active_ages.mean():.1f}步")
    print(f"  中位数年龄: {np.median(active_ages):.1f}步")
    print(f"  最大年龄: {active_ages.max()}步")
    print(f"  保护期内连接: {(active_ages < model.connection_manager.protection_period).sum()}")


def plot_all_visualizations(model, epoch_stats: List[Dict], output_dir: str = "."):
    """
    生成所有可视化图表（合并到一张大图）

    Args:
        model: 神经网络模型
        epoch_stats: epoch统计信息列表
        output_dir: 输出目录
    """
    import os

    print("\n生成综合可视化图表...")

    # 创建一个大的图形，包含所有子图
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # ==================== 第一行：训练历史（6个子图） ====================
    # 1. 训练损失
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = [s['epoch'] for s in epoch_stats]
    ax1.plot(epochs, [s['train_loss'] for s in epoch_stats], 'b-', label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('训练损失', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 准确率
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [s['train_acc'] for s in epoch_stats], 'g-', label='Train Acc', linewidth=2)
    if epoch_stats[0].get('test_acc') is not None:
        ax2.plot(epochs, [s['test_acc'] for s in epoch_stats], 'r-', label='Test Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.set_title('准确率', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 连接数量变化
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, [s['total_connections'] for s in epoch_stats], 'purple', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('连接数', fontsize=10)
    ax3.set_title('总连接数演化', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. 剪枝和新增
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(epochs, [s['pruned_last'] for s in epoch_stats], 'r-', label='剪枝', linewidth=2)
    ax4.plot(epochs, [s['added_last'] for s in epoch_stats], 'g-', label='新增', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('数量', fontsize=10)
    ax4.set_title('连接变化（剪枝 vs 新增）', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ==================== 第二行：变化率和稀疏度 + 拓扑结构 ====================
    # 5. 变化率
    ax5 = fig.add_subplot(gs[1, 0])
    change_rates = [s['change_rate'] * 100 for s in epoch_stats]
    ax5.plot(epochs, change_rates, 'orange', linewidth=2)
    ax5.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='目标下限 (5%)')
    ax5.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='目标上限 (10%)')
    ax5.fill_between(epochs, 5, 10, alpha=0.2, color='green')
    ax5.set_xlabel('Epoch', fontsize=10)
    ax5.set_ylabel('变化率 (%)', fontsize=10)
    ax5.set_title('连接变化率（目标: 5-10%）', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 稀疏度
    ax6 = fig.add_subplot(gs[1, 1])
    sparsity = [s['sparsity'] * 100 for s in epoch_stats]
    ax6.plot(epochs, sparsity, 'brown', linewidth=2)
    ax6.set_xlabel('Epoch', fontsize=10)
    ax6.set_ylabel('稀疏度 (%)', fontsize=10)
    ax6.set_title('网络稀疏度', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. 邻接矩阵热力图
    ax7 = fig.add_subplot(gs[1, 2])
    adj_mask = model.adj_mask.cpu().numpy()
    n = min(model.num_neurons, 50)
    im = ax7.imshow(adj_mask[:n, :n], cmap='Blues', interpolation='nearest')
    ax7.set_title(f'邻接矩阵热力图 (前{n}个神经元)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('目标神经元', fontsize=10)
    ax7.set_ylabel('源神经元', fontsize=10)
    plt.colorbar(im, ax=ax7, label='连接强度')

    # 8. 网络图
    ax8 = fig.add_subplot(gs[1, 3])
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if adj_mask[i, j] > 0:
                G.add_edge(i, j)
    try:
        pos = nx.spring_layout(G, k=2, iterations=50)
    except:
        pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=200, ax=ax8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True, arrowsize=8, ax=ax8)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax8)
    ax8.set_title(f'网络图 (连接数: {G.number_of_edges()})', fontsize=12, fontweight='bold')
    ax8.axis('off')

    # ==================== 第三行：连接年龄分析 ====================
    connection_age = model.connection_manager.connection_age.cpu().numpy()
    active_ages = connection_age[adj_mask > 0]

    if len(active_ages) > 0:
        # 9. 年龄分布直方图
        ax9 = fig.add_subplot(gs[2, 0:2])
        ax9.hist(active_ages, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax9.axvline(x=model.connection_manager.protection_period,
                    color='red', linestyle='--', linewidth=2,
                    label=f'保护期 ({model.connection_manager.protection_period}步)')
        ax9.set_xlabel('连接年龄（步数）', fontsize=10)
        ax9.set_ylabel('连接数量', fontsize=10)
        ax9.set_title('连接年龄分布', fontsize=12, fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        # 10. 累积分布
        ax10 = fig.add_subplot(gs[2, 2:4])
        sorted_ages = np.sort(active_ages)
        cumulative = np.arange(1, len(sorted_ages) + 1) / len(sorted_ages) * 100
        ax10.plot(sorted_ages, cumulative, linewidth=2, color='green')
        ax10.axvline(x=model.connection_manager.protection_period,
                     color='red', linestyle='--', linewidth=2, label='保护期')
        ax10.set_xlabel('连接年龄（步数）', fontsize=10)
        ax10.set_ylabel('累积百分比 (%)', fontsize=10)
        ax10.set_title('连接年龄累积分布', fontsize=12, fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)

        # 打印统计信息
        print(f"\n连接年龄统计:")
        print(f"  总连接数: {len(active_ages)}")
        print(f"  平均年龄: {active_ages.mean():.1f}步")
        print(f"  中位数年龄: {np.median(active_ages):.1f}步")
        print(f"  最大年龄: {active_ages.max()}步")
        print(f"  保护期内连接: {(active_ages < model.connection_manager.protection_period).sum()}")

    # 添加总标题
    fig.suptitle('神经可塑性网络 - 综合训练分析', fontsize=18, fontweight='bold', y=0.995)

    # 保存图表
    save_path = os.path.join(output_dir, "comprehensive_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 综合分析图已保存到: {save_path}")

    # 显示图表（只显示一次）
    plt.show()

    print(f"✅ 所有可视化完成！")
