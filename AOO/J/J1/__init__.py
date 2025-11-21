"""
J1数学计算工具模块

该模块提供了全面的数学计算功能，包括高级数学运算、金融数学计算、
统计数学工具、优化数学工具等。支持异步处理和缓存机制。

主要组件：
- AdvancedMath: 高级数学运算（矩阵运算、微积分、线性代数）
- FinancialMath: 金融数学计算（期权定价、风险度量、收益率计算）
- StatisticalMath: 统计数学工具（概率分布、假设检验、置信区间）
- OptimizationMath: 优化数学工具（梯度下降、牛顿法、拉格朗日乘数法）
- MathematicalTools: 主工具类，整合所有功能

作者：J1数学计算工具团队
版本：1.0.0
日期：2025-11-06
"""

# 导入主要类和异常
from .MathematicalTools import (
    # 异常类
    MathError,
    MatrixError,
    FinancialMathError,
    StatisticalError,
    OptimizationError,
    CacheError,
    
    # 数据结构
    MathResult,
    OptimizationResult,
    StatisticalTestResult,
    
    # 核心类
    CacheManager,
    AsyncMathProcessor,
    AdvancedMath,
    FinancialMath,
    StatisticalMath,
    OptimizationMath,
    MathematicalTools
)

# 版本信息
__version__ = "1.0.0"
__author__ = "J1数学计算工具团队"
__email__ = "support@j1-math-tools.com"

# 公开API
__all__ = [
    # 异常类
    'MathError',
    'MatrixError', 
    'FinancialMathError',
    'StatisticalError',
    'OptimizationError',
    'CacheError',
    
    # 数据结构
    'MathResult',
    'OptimizationResult', 
    'StatisticalTestResult',
    
    # 核心类
    'CacheManager',
    'AsyncMathProcessor',
    'AdvancedMath',
    'FinancialMath',
    'StatisticalMath',
    'OptimizationMath',
    'MathematicalTools'
]

# 模块级便捷函数
def create_math_tools(cache_size: int = 1000, cache_ttl: int = 3600, 
                     max_workers: int = 4) -> MathematicalTools:
    """
    创建数学工具实例的便捷函数
    
    Args:
        cache_size: 缓存大小
        cache_ttl: 缓存生存时间（秒）
        max_workers: 异步处理最大工作线程数
        
    Returns:
        数学工具实例
    """
    return MathematicalTools(
        cache_size=cache_size,
        cache_ttl=cache_ttl,
        max_workers=max_workers
    )

def get_version() -> str:
    """获取模块版本信息"""
    return __version__

# 向后兼容性别名
MathTools = MathematicalTools  # 兼容性别名