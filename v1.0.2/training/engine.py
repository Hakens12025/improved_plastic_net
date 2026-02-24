import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Optional

class ImprovedEvolutionEngine:
    """改进版演化训练引擎"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 0.001,
        plasticity_interval: int = 50
    ):
        self.model = model.to(device)
        self.device = device
        self.plasticity_interval = plasticity_interval

        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        # 混合精度训练 (AMP)
        # 兼容 torch<2.0：优先 torch.amp，失败则回退 torch.cuda.amp
        self.use_amp = (device.type == 'cuda')
        try:
            from torch.amp import autocast, GradScaler
            self._autocast = autocast
            self._autocast_device_type = True
            self.scaler = GradScaler(enabled=self.use_amp)
        except ImportError:
            from torch.cuda.amp import autocast, GradScaler
            self._autocast = autocast
            self._autocast_device_type = False
            self.scaler = GradScaler(enabled=self.use_amp)

        self.epoch_stats = []

    def train_and_evolve(
        self,
        train_loader,
        test_loader=None,
        epochs: int = 5,
        verbose: bool = True
    ):
        if verbose:
            print(f"\n[INFO] 引擎启动。设备: {self.device} | AMP 开启: {self.use_amp}")
            print(f"[INFO] 可塑性更新间隔: {self.plasticity_interval} batches")
            print("-" * 80)

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练一个 epoch
            train_loss, train_acc = self._train_epoch(train_loader, epoch, epochs, verbose)

            # 测试
            test_acc = self.test(test_loader) if test_loader else 0.0

            epoch_time = time.time() - epoch_start
            
            # 获取模型统计
            stats = self.model.get_statistics()
            stats.update({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'epoch_time': epoch_time
            })
            self.epoch_stats.append(stats)

            if verbose:
                self._print_epoch_summary(stats)

    def _train_epoch(self, train_loader, epoch, total_epochs, verbose):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(train_loader)

        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            # 数据移动到设备并展平
            imgs = imgs.view(imgs.size(0), -1).to(self.device, non_blocking=True)
            lbls = lbls.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # 混合精度前向传播
            if self._autocast_device_type:
                autocast_ctx = self._autocast(device_type='cuda', enabled=self.use_amp)
            else:
                autocast_ctx = self._autocast(enabled=self.use_amp)
            with autocast_ctx:
                outputs = self.model(imgs)
                loss = self.criterion(outputs, lbls)

            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

            # --- 神经可塑性更新 (这是最耗时的部分) ---
            if batch_idx % self.plasticity_interval == 0 and batch_idx > 0:
                # 执行剪枝和生长
                pruned, added = self.model.apply_neuroplasticity()

                if verbose:
                    # 获取当前统计信息
                    current_acc = 100.0 * correct / total if total > 0 else 0.0
                    current_sparsity = self.model.get_sparsity()
                    current_connections = self.model.adj_mask.sum().item()

                    # 打印详细进度（每次都换行，不覆盖）
                    print(f"[Epoch {epoch+1}/{total_epochs}] Batch {batch_idx}/{num_batches} | "
                          f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}% | "
                          f"Pruned: {pruned} | Added: {added} | "
                          f"Connections: {current_connections:.0f} | Sparsity: {current_sparsity:.2%}")

            # 即使不是可塑性更新的batch，也定期打印进度
            elif verbose and batch_idx % 10 == 0:
                current_acc = 100.0 * correct / total if total > 0 else 0.0
                print(f"[Epoch {epoch+1}/{total_epochs}] Batch {batch_idx}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%", end='\r')

        avg_loss = total_loss / num_batches
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def test(self, test_loader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.view(imgs.size(0), -1).to(self.device)
                lbls = lbls.to(self.device)
                outputs = self.model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()
        return 100.0 * correct / total

    def _print_epoch_summary(self, stats: dict):
        print(f"\n\n{'='*30} Epoch {stats['epoch']} Summary {'='*30}")
        print(f"训练准确率: {stats['train_acc']:.2f}% | 测试准确率: {stats['test_acc']:.2f}%")
        print(f"当前稀疏度: {stats['sparsity']:.2%} | 连接总数: {stats['total_connections']}")
        print(f"耗时: {stats['epoch_time']:.2f}s")
        print(f"{'='*75}\n")

    def get_epoch_stats(self):
        return self.epoch_stats

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch_stats': self.epoch_stats
        }, path)
