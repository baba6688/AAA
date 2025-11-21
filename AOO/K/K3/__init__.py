"""
K3策略配置管理器模块
==================

该模块提供了完整的策略配置管理功能，包括策略参数管理、模板管理、
优化配置、回测配置、风险管理、性能评估、版本控制等功能。

主要组件：
- StrategyConfigurationManager: 主要的配置管理器
- StrategyConfig: 策略配置数据类
- StrategyTemplate: 策略模板
- StrategyLibrary: 策略库
- ParameterOptimizer: 参数优化器
- BacktestConfig: 回测配置
- RiskManager: 风险管理器
- PerformanceEvaluator: 性能评估器
- VersionManager: 版本管理器

作者: AI Assistant
版本: 1.0.0
日期: 2025-11-06
"""

from .StrategyConfigurationManager import (
    StrategyConfigurationManager,
    StrategyConfig,
    StrategyTemplate,
    StrategyLibrary,
    ParameterOptimizer,
    BacktestConfig,
    RiskManager,
    PerformanceEvaluator,
    VersionManager,
    AsyncConfigProcessor,
    ConfigError,
    ValidationError,
    OptimizationError,
    BacktestError,
    RiskError,
    PerformanceError,
    VersionError,
    AsyncProcessingError,
    StrategyParameter,
    StrategyLogic,
    StrategyConstraints,
    OptimizationResult,
    RiskMetrics,
    PerformanceMetrics,
    VersionInfo,
    StrategyType,
    OptimizationMethod,
    RiskLevel,
    BacktestMode,
    PerformanceMetric,
    VersionStatus
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    'StrategyConfigurationManager',
    'StrategyConfig',
    'StrategyTemplate', 
    'StrategyLibrary',
    'ParameterOptimizer',
    'BacktestConfig',
    'RiskManager',
    'PerformanceEvaluator',
    'VersionManager',
    'AsyncConfigProcessor',
    'ConfigError',
    'ValidationError',
    'OptimizationError',
    'BacktestError',
    'RiskError',
    'PerformanceError',
    'VersionError',
    'AsyncProcessingError',
    'StrategyParameter',
    'StrategyLogic',
    'StrategyConstraints',
    'OptimizationResult',
    'RiskMetrics',
    'PerformanceMetrics',
    'VersionInfo',
    'StrategyType',
    'OptimizationMethod',
    'RiskLevel',
    'BacktestMode',
    'PerformanceMetric',
    'VersionStatus'
]