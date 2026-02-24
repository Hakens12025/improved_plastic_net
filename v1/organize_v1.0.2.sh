#!/bin/bash

# 创建 v1.0.2 目录结构
mkdir -p v1.0.2/{models,training,experiments,utils,results,docs}

# 移动代码文件
echo "移动代码文件..."
cp -r models/* v1.0.2/models/ 2>/dev/null
cp -r training/* v1.0.2/training/ 2>/dev/null
cp -r experiments/* v1.0.2/experiments/ 2>/dev/null
cp -r utils/* v1.0.2/utils/ 2>/dev/null

# 移动配置和运行脚本
echo "移动配置文件..."
cp config.py v1.0.2/
cp run_with_config.py v1.0.2/
cp run_experiment.bat v1.0.2/
cp run_experiment.sh v1.0.2/
cp requirements.txt v1.0.2/

# 移动文档
echo "移动文档..."
cp v1.0.2_README.md v1.0.2/README.md
cp OPTIMIZATION_IMPLEMENTATION_SUMMARY.md v1.0.2/docs/
cp QUICK_RUN_GUIDE.md v1.0.2/docs/
cp TRAINING_CONFIG.md v1.0.2/docs/
cp CONFIG_UPDATE_SUMMARY.md v1.0.2/docs/
cp BUGFIX_EPOCH5_HANG.md v1.0.2/docs/
cp BUGFIX_BATCH400_HANG.md v1.0.2/docs/

# 移动训练结果
echo "移动训练结果..."
mv *.png v1.0.2/results/ 2>/dev/null
mv *.pth v1.0.2/results/ 2>/dev/null
mv *_performance.json v1.0.2/results/ 2>/dev/null

# 创建版本信息文件
cat > v1.0.2/VERSION_INFO.md << 'VEOF'
# 版本信息

**版本号**: v1.0.2
**发布日期**: 2026-02-03
**状态**: 稳定版

## 主要特性

1. **智能初始化** - 基于拓扑距离的连接初始化
2. **动态参数扩展** - 支持 1000-3000 神经元
3. **训练优化** - 学习率调度 + 数据增强
4. **性能优化** - 向量化操作 + GPU加速

## 运行方法

```bash
cd v1.0.2
python run_with_config.py
```

## 性能指标

- 测试准确率: 88.53%
- 训练时间: ~30s/epoch
- 最终连接数: 267,416
- 稀疏度: 46.46%

## 已知问题

- 准确率略低于预期（目标 90%+）
- 需要进一步调优参数

## 改进建议

参考 `docs/OPTIMIZATION_IMPLEMENTATION_SUMMARY.md`
VEOF

echo "✅ v1.0.2 文件整理完成！"
