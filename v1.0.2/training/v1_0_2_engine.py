"""
优化版演化训练引擎 v1.0.2 - 性能优化的训练引擎

核心优化：
1. 内存管理优化，减少临时张量分配
2. 批量处理，提高GPU利用率
3. 性能监控，添加详细的性能指标
4. 智能调度，减少不必要的状态更新
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Optional
import numpy as np


class OptimizedEvolutionEngine:
    """优化版演化训练引擎 v1.0.2"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 0.001,
        plasticity_interval: int = 50
    ):
        """
        初始化优化版训练引擎

        Args:
            model: 神经网络模型
            device: 计算设备
            lr: 学习率
            plasticity_interval: 神经可塑性更新间隔
        """
        self.model = model.to(device)
        self.device = device
        self.plasticity_interval = plasticity_interval

        # 🚀 优化1：使用更高效的优化器设置
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # AdamW通常更稳定
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑提高泛化

        # 🚀 Stage 4 优化：添加学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # 每5个epoch重启一次
            T_mult=2,
            eta_min=lr * 0.01
        )

        # 🚀 优化2：改进的混合精度训练支持
        self.use_amp = (device.type == 'cuda')
        if self.use_amp:
            try:
                # 尝试新版本的torch.amp
                from torch.amp import autocast, GradScaler
                self.autocast = lambda: autocast(device_type='cuda', enabled=True)
                self.scaler = GradScaler(enabled=True)
            except (ImportError, AttributeError):
                # 回退到旧版本
                from torch.cuda.amp import autocast, GradScaler
                self.autocast = lambda: autocast(enabled=True)
                self.scaler = GradScaler(enabled=True)
        else:
            # CPU模式：使用空的上下文管理器（不影响梯度计算）
            from contextlib import nullcontext
            self.autocast = lambda: nullcontext()
            self.scaler = None

        # 🚀 优化3：预分配统计缓冲区
        self.epoch_stats = []
        self.performance_metrics = {
            'batch_times': [],
            'plasticity_times': [],
            'forward_times': [],
            'backward_times': []
        }

        # 🚀 优化4：智能调度标志
        self.last_plasticity_time = 0
        self.performance_log_interval = 100  # 每100个batch记录一次性能

    def train_and_evolve_optimized(
        self,
        train_loader,
        test_loader=None,
        epochs: int = 5,
        verbose: bool = True
    ):
        """
        🚀 优化版训练和演化 - 性能监控和智能调度

        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
            verbose: 是否显示详细信息
        """
        if verbose:
            print(f"\n[INFO] Optimized Engine Started. Device: {self.device} | AMP: {self.use_amp}")
            print(f"[INFO] Plasticity update interval: {self.plasticity_interval} batches")
            print(f"[INFO] Optimizations: Vectorized pruning | Smart caching | Memory optimization")
            print("-" * 80)

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train one epoch (optimized version)
            train_loss, train_acc = self._train_epoch_optimized(
                train_loader, epoch, epochs, verbose
            )

            # Test
            test_acc = self.test(test_loader) if test_loader else 0.0

            epoch_time = time.time() - epoch_start
            
            # Get detailed performance metrics
            stats = self.model.get_statistics()
            stats.update({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'epoch_time': epoch_time,
                'avg_batch_time': np.mean(self.performance_metrics['batch_times']) if self.performance_metrics['batch_times'] else 0,
                'avg_plasticity_time': np.mean(self.performance_metrics['plasticity_times']) if self.performance_metrics['plasticity_times'] else 0,
                'memory_optimization': 'enabled' if hasattr(self.model, 'connection_manager') else 'disabled'
            })
            self.epoch_stats.append(stats)

            # Clear current epoch performance metrics
            self._clear_performance_metrics()

            if verbose:
                self._print_epoch_summary_optimized(stats)

    def train_and_evolve(self, train_loader, test_loader=None, epochs: int = 5, verbose: bool = True):
        """兼容性方法：调用优化版实现"""
        return self.train_and_evolve_optimized(train_loader, test_loader, epochs, verbose)

    def _train_epoch_optimized(self, train_loader, epoch, total_epochs, verbose):
        """
        🚀 优化版epoch训练 - 性能监控和内存优化

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            total_epochs: 总epoch数
            verbose: 是否显示详细信息

        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(train_loader)

        # 🚀 预分配一些临时变量
        batch_start_time = time.time()

        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            # 🚀 优化：非阻塞数据传输
            imgs = imgs.view(imgs.size(0), -1).to(self.device, non_blocking=True)
            lbls = lbls.to(self.device, non_blocking=True)

            # 🚀 记录前向传播时间
            forward_start = time.time()

            # 清零梯度
            self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            # 🚀 优化的混合精度前向传播
            with self.autocast():
                outputs = self.model(imgs)
                loss = self.criterion(outputs, lbls)

            # 🚀 记录反向传播时间
            backward_start = time.time()

            # 反向传播（优化版）
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # 🚀 Stage 4 优化：更新学习率
            self.scheduler.step()

            # 🚀 性能指标记录
            current_time = time.time()
            forward_time = backward_start - forward_start
            backward_time = current_time - backward_start
            batch_time = current_time - batch_start_time

            self.performance_metrics['forward_times'].append(forward_time)
            self.performance_metrics['backward_times'].append(backward_time)
            self.performance_metrics['batch_times'].append(batch_time)

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

            # 🚀 智能神经可塑性更新 - 性能监控版本
            if batch_idx % self.plasticity_interval == 0 and batch_idx > 0:
                plasticity_start = time.time()
                
                # 执行优化版剪枝和生长
                pruned, added = self.model.apply_neuroplasticity()
                
                plasticity_time = time.time() - plasticity_start
                self.performance_metrics['plasticity_times'].append(plasticity_time)

                if verbose:
                    # 🚀 优化：减少频繁的状态查询
                    current_acc = 100.0 * correct / total
                    current_sparsity = self.model.get_sparsity()
                    current_connections = self.model.adj_mask.sum().item()

                    print(f"[Epoch {epoch+1}/{total_epochs}] Batch {batch_idx}/{num_batches} | "
                          f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}% | "
                          f"Pruned: {pruned} | Added: {added} | "
                          f"Connections: {current_connections:.0f} | Sparsity: {current_sparsity:.2%} | "
                          f"Plasticity: {plasticity_time:.3f}s")

                # 🚀 智能调度：如果可塑性更新太慢，动态调整间隔
                if plasticity_time > 1.0 and self.plasticity_interval < 200:
                    self.plasticity_interval = min(200, self.plasticity_interval + 10)
                    if verbose:
                        print(f"[INFO] 🚀 动态调整可塑性间隔为: {self.plasticity_interval}")

            # 🚀 性能日志记录（减少频率）
            elif verbose and batch_idx % self.performance_log_interval == 0:
                current_acc = 100.0 * correct / total
                print(f"[Epoch {epoch+1}/{total_epochs}] Batch {batch_idx}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}% | "
                      f"Batch: {batch_time:.3f}s", end='\r')

            batch_start_time = time.time()

        avg_loss = total_loss / num_batches
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def test(self, test_loader) -> float:
        """
        测试模型性能

        Args:
            test_loader: 测试数据加载器

        Returns:
            测试准确率
        """
        self.model.eval()
        correct = 0
        total = 0
        
        # 🚀 优化：禁用梯度计算和使用更高效的评估
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.view(imgs.size(0), -1).to(self.device, non_blocking=True)
                lbls = lbls.to(self.device, non_blocking=True)
                
                # 🚀 使用混合精度评估（如果可用）
                with self.autocast():
                    outputs = self.model(imgs)
                
                _, predicted = torch.max(outputs, 1)
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()
                
        return 100.0 * correct / total

    def _print_epoch_summary_optimized(self, stats: dict):
        """
        🚀 优化版epoch总结 - 包含性能指标

        Args:
            stats: 统计信息字典
        """
        print(f"\n\n{'='*30} Epoch {stats['epoch']} Summary {'='*30}")
        print(f"[INFO] Train accuracy: {stats['train_acc']:.2f}% | Test accuracy: {stats['test_acc']:.2f}%")
        print(f"[INFO] Current sparsity: {stats['sparsity']:.2%} | Total connections: {stats['total_connections']}")
        print(f"[INFO] Total time: {stats['epoch_time']:.2f}s | Avg batch time: {stats['avg_batch_time']:.3f}s")
        
        if stats['avg_plasticity_time'] > 0:
            print(f"[INFO] Avg plasticity time: {stats['avg_plasticity_time']:.3f}s")
        
        if 'memory_optimization' in stats:
            print(f"[INFO] Memory optimization: {stats['memory_optimization']}")
            
        print(f"{'='*75}\n")

    def _clear_performance_metrics(self):
        """清理当前epoch的性能指标"""
        for key in self.performance_metrics:
            self.performance_metrics[key].clear()

    def get_epoch_stats(self):
        """获取epoch统计信息"""
        return self.epoch_stats

    def get_performance_summary(self) -> dict:
        """
        🚀 获取性能总结

        Returns:
            性能总结字典
        """
        if not self.epoch_stats:
            return {}

        # 计算平均性能指标
        avg_batch_times = []
        avg_plasticity_times = []
        
        for epoch_stat in self.epoch_stats:
            if 'avg_batch_time' in epoch_stat:
                avg_batch_times.append(epoch_stat['avg_batch_time'])
            if 'avg_plasticity_time' in epoch_stat:
                avg_plasticity_times.append(epoch_stat['avg_plasticity_time'])

        return {
            'total_epochs': len(self.epoch_stats),
            'avg_epoch_time': np.mean([s['epoch_time'] for s in self.epoch_stats]),
            'avg_batch_time': np.mean(avg_batch_times) if avg_batch_times else 0,
            'avg_plasticity_time': np.mean(avg_plasticity_times) if avg_plasticity_times else 0,
            'final_accuracy': self.epoch_stats[-1]['train_acc'] if self.epoch_stats else 0,
            'final_sparsity': self.epoch_stats[-1]['sparsity'] if self.epoch_stats else 0
        }

    def save_model(self, path: str):
        """
        保存模型和训练状态

        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch_stats': self.epoch_stats,
            'performance_metrics': self.performance_metrics,
            'plasticity_interval': self.plasticity_interval,
            'version': 'v1.0.2_optimized'
        }, path)
        
        # 🚀 同时保存性能总结
        perf_path = path.replace('.pth', '_performance.json')
        import json
        with open(perf_path, 'w') as f:
            json.dump(self.get_performance_summary(), f, indent=2)
        
        print(f"[INFO] 🚀 模型和性能数据已保存到: {path} 和 {perf_path}")