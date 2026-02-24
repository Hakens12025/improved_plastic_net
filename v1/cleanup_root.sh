#!/bin/bash

echo "清理根目录，保留共享文件..."

# 删除已移动到 v1.0.2 的文件
rm -f config.py
rm -f run_with_config.py
rm -f run_experiment.bat
rm -f run_experiment.sh
rm -f v1.0.2_README.md
rm -f OPTIMIZATION_IMPLEMENTATION_SUMMARY.md
rm -f QUICK_RUN_GUIDE.md
rm -f TRAINING_CONFIG.md
rm -f CONFIG_UPDATE_SUMMARY.md
rm -f BUGFIX_EPOCH5_HANG.md
rm -f BUGFIX_BATCH400_HANG.md

# 删除代码目录（已复制到 v1.0.2）
rm -rf models
rm -rf training
rm -rf experiments
rm -rf utils

# 删除 __pycache__
rm -rf __pycache__

echo "✅ 根目录清理完成！"
echo ""
echo "保留的共享文件："
ls -1 *.md 2>/dev/null
ls -1 *.txt 2>/dev/null
