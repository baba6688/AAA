"""
C5 规则引擎模块
实现规则管理、执行、冲突检测、优化等功能

主要类:
- RuleType: 规则类型枚举
- ConflictResolutionStrategy: 冲突解决策略枚举
- RuleMetadata: 规则元数据
- RuleExecutionResult: 规则执行结果
- PerformanceMetrics: 性能指标
- Rule: 规则基类
- IfThenRule: 如果-那么规则
- DecisionTableRule: 决策表规则
- DecisionTreeRule: 决策树规则
- FuzzyRule: 模糊规则
- TemporalRule: 时态规则
- RuleConflictDetector: 规则冲突检测器
- RuleOptimizer: 规则优化器
- RuleValidator: 规则验证器
- RuleVersionManager: 规则版本管理器
- RuleEngine: 规则引擎

版本: 1.0.0
作者: AI量化系统
"""

from .RuleEngine import (
    RuleType,
    ConflictResolutionStrategy,
    RuleMetadata,
    RuleExecutionResult,
    PerformanceMetrics,
    Rule,
    IfThenRule,
    DecisionTableRule,
    DecisionTreeRule,
    FuzzyRule,
    TemporalRule,
    RuleConflictDetector,
    RuleOptimizer,
    RuleValidator,
    RuleVersionManager,
    RuleEngine
)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    'RuleType',
    'ConflictResolutionStrategy',
    'RuleMetadata',
    'RuleExecutionResult',
    'PerformanceMetrics',
    'Rule',
    'IfThenRule',
    'DecisionTableRule',
    'DecisionTreeRule',
    'FuzzyRule',
    'TemporalRule',
    'RuleConflictDetector',
    'RuleOptimizer',
    'RuleValidator',
    'RuleVersionManager',
    'RuleEngine'
]