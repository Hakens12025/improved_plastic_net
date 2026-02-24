"""
优化版训练模块 v1.0.3 - 版本化初始化

提供优化版训练引擎的统一接口
"""

from .v1_0_3_engine import OptimizedEvolutionEngine

# 版本信息
__version__ = "1.0.3"
__title__ = "Optimized Evolution Training Engine"
__description__ = "Performance-optimized training engine with intelligent scheduling"

# 导出主要类
__all__ = [
    'OptimizedEvolutionEngine',
    '__version__',
    '__title__',
    '__description__'
]

# 兼容性别名
ImprovedEvolutionEngine = OptimizedEvolutionEngine
