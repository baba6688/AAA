"""
Z3自定义算法插槽系统

这个包提供了Z3求解器的自定义算法插槽功能，包括：
- 算法插槽管理
- 算法注册和注销
- 算法执行和调用
- 算法配置和验证
- 算法统计和优化
- 算法文档管理

主要类：
- CustomAlgorithmSlot: 主要的算法插槽管理器
- AlgorithmRegistry: 算法注册表
- AlgorithmExecutor: 算法执行器
- AlgorithmConfig: 算法配置管理器
- AlgorithmValidator: 算法验证器
- AlgorithmStatistics: 算法统计器
- AlgorithmOptimizer: 算法优化器
- AlgorithmDocumentation: 算法文档管理器
"""

from .CustomAlgorithmSlot import (
    CustomAlgorithmSlot,
    AlgorithmRegistry,
    AlgorithmExecutor,
    AlgorithmConfig,
    AlgorithmValidator,
    AlgorithmStatistics,
    AlgorithmOptimizer,
    AlgorithmDocumentation,
    AlgorithmSlot,
    AlgorithmInfo,
    AlgorithmResult,
    ExecutionContext,
    ValidationResult,
    OptimizationResult
)

__version__ = "1.0.0"
__author__ = "Z3 Algorithm Slot Team"

__all__ = [
    'CustomAlgorithmSlot',
    'AlgorithmRegistry',
    'AlgorithmExecutor',
    'AlgorithmConfig',
    'AlgorithmValidator',
    'AlgorithmStatistics',
    'AlgorithmOptimizer',
    'AlgorithmDocumentation',
    'AlgorithmSlot',
    'AlgorithmInfo',
    'AlgorithmResult',
    'ExecutionContext',
    'ValidationResult',
    'OptimizationResult'
]