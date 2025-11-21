"""
H6自适应优化器模块
实现自适应参数优化、策略学习、性能提升、环境适应、效果评估、历史跟踪和报告生成
"""

from .AdaptiveOptimizer import (
    AdaptiveOptimizer,
    OptimizationConfig,
    PerformanceMetrics,
    OptimizationResult,
    ParameterOptimizer,
    StrategyLearner,
    PerformanceOptimizer,
    EnvironmentAdapter,
    EffectivenessEvaluator,
    HistoryTracker,
    ReportGenerator
)

__all__ = [
    'AdaptiveOptimizer',
    'OptimizationConfig',
    'PerformanceMetrics', 
    'OptimizationResult',
    'ParameterOptimizer',
    'StrategyLearner',
    'PerformanceOptimizer',
    'EnvironmentAdapter',
    'EffectivenessEvaluator',
    'HistoryTracker',
    'ReportGenerator'
]

__version__ = '1.0.0'
__author__ = 'H6 Adaptive Optimizer Team'