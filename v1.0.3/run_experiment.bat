@echo off
REM ========================================
REM 改进版神经可塑性网络 - GPU训练启动脚本
REM ========================================

echo ========================================
echo 激活 pt_gpu Conda 环境
echo ========================================

REM 激活conda环境
call conda activate pt_gpu

if %ERRORLEVEL% NEQ 0 (
    echo [错误] 无法激活 pt_gpu 环境
    echo 请确保已安装conda并创建了pt_gpu环境
    pause
    exit /b 1
)

echo.
echo [成功] pt_gpu 环境已激活
echo.

REM 检查CUDA是否可用
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ========================================
echo 运行 Fashion-MNIST 实验 (v1.0.3优化版)
echo 配置: 1000神经元, 10轮训练
echo ========================================
echo.

cd experiments
python v1_0_3_mnist_baseline.py

echo.
echo ========================================
echo 训练完成！
echo ========================================
pause

