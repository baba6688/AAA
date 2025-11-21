"""
D7状态自检器模块
================

实现系统状态全面检查、健康状态评估、问题诊断和定位等功能。

主要组件：
- SystemHealthChecker: 系统健康检查器
- StateAnalyzer: 状态分析器
- ProblemDiagnostic: 问题诊断器
- StateReporter: 状态报告器
- StateOptimizer: 状态优化器
- StateMonitor: 状态监控器
- StateHistoryManager: 状态历史管理器
"""

from .SelfChecker import (
    SystemHealthChecker,
    StateAnalyzer, 
    ProblemDiagnostic,
    StateReporter,
    StateOptimizer,
    StateMonitor,
    StateHistoryManager,
    SelfChecker
)

__all__ = [
    'SystemHealthChecker',
    'StateAnalyzer',
    'ProblemDiagnostic', 
    'StateReporter',
    'StateOptimizer',
    'StateMonitor',
    'StateHistoryManager',
    'SelfChecker'
]

__version__ = '1.0.0'
__author__ = 'D7状态自检器开发团队'