"""
优化版模型模块 v1.0.3 - 版本化初始化

提供优化版神经可塑性网络的统一接口
"""

from .v1_0_3_plastic_net import OptimizedPlasticNet
from .v1_0_3_topology_manager import OptimizedTopologyManager
from .v1_0_3_connection_manager import OptimizedConnectionManager

# 版本信息
__version__ = "1.0.3"
__title__ = "Optimized Improved Plastic Network"
__description__ = "Performance-optimized neural plasticity network with vectorized operations"

# 导出主要类
__all__ = [
    'OptimizedPlasticNet',
    'OptimizedTopologyManager', 
    'OptimizedConnectionManager',
    '__version__',
    '__title__',
    '__description__'
]

# 兼容性别名
ImprovedPlasticNet = OptimizedPlasticNet
TopologyManager = OptimizedTopologyManager
ConnectionManager = OptimizedConnectionManager
