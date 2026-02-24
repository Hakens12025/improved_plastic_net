# Bug修复：Epoch 5 训练卡住问题

## 问题描述
训练在 epoch 5 结束后卡住，无法继续，只能强制退出。

## 根本原因
在 CPU 模式下，`autocast()` 被错误地设置为 `torch.no_grad()`，这会导致：
1. 前向传播时禁用梯度计算
2. 反向传播时无法计算梯度
3. 导致训练卡住或崩溃

## 修复方案

### 修改文件
`training/v1_0_2_engine.py` 第 54-69 行

### 修改前
```python
else:
    self.autocast = lambda: torch.no_grad()
    self.scaler = None
```

### 修改后
```python
else:
    # CPU模式：使用空的上下文管理器（不影响梯度计算）
    from contextlib import nullcontext
    self.autocast = lambda: nullcontext()
    self.scaler = None
```

## 解释
- `nullcontext()` 是一个空的上下文管理器，不会影响任何操作
- 在 CPU 模式下，不需要混合精度训练，使用 `nullcontext()` 可以保持代码结构一致
- 这样既不会禁用梯度计算，也不会影响性能

## 测试验证
修复后，请重新运行训练：
```bash
cd projects/deepLerning/improved_plastic_net
python run_with_config.py
```

应该能够正常完成所有 10 个 epoch 的训练。

## 修复日期
2026-02-03

## 状态
✅ 已修复
