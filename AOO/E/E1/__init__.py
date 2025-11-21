"""
E1策略生成器模块

该模块实现了智能交易策略生成器，具备以下功能：
- 多类型交易策略生成（趋势跟踪、均值回归、动量、套利等）
- 基于机器学习的策略生成
- 参数化策略模板和框架
- 策略组合和混合生成
- 策略个性化定制
- 策略可行性评估
- 策略代码自动生成
"""

from .StrategyGenerator import (
    StrategyGenerator,
    StrategyTemplate,
    GeneticAlgorithm,
    DeepRLStrategy,
    StrategyEvaluator,
    CodeGenerator
)

__all__ = [
    'StrategyGenerator',
    'StrategyTemplate', 
    'GeneticAlgorithm',
    'DeepRLStrategy',
    'StrategyEvaluator',
    'CodeGenerator'
]

__version__ = '1.0.0'