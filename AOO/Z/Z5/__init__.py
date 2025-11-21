"""
Z5实验性功能插槽系统

该模块提供了完整的实验性功能管理解决方案，包括：
- 实验功能管理
- 特性开关和灰度发布
- A/B测试框架
- 实验配置管理
- 实验监控和统计
- 实验报告生成
- 实验安全控制

主要类：
- ExperimentalFeatureSlot: 实验性功能插槽主类
- FeatureToggle: 特性开关管理
- ABTestManager: A/B测试管理器
- ExperimentConfig: 实验配置管理
- ExperimentMonitor: 实验监控器
- ExperimentReporter: 实验报告生成器
- ExperimentSecurity: 实验安全控制器
"""

from .ExperimentalFeatureSlot import (
    ExperimentalFeatureSlot,
    FeatureToggle,
    ABTestManager,
    ExperimentConfig,
    ExperimentMonitor,
    ExperimentReporter,
    ExperimentSecurity,
    ExperimentResult,
    FeatureFlag,
    ABTestGroup,
    ExperimentMetrics
)

__version__ = "1.0.0"
__author__ = "Z5 Team"

__all__ = [
    "ExperimentalFeatureSlot",
    "FeatureToggle", 
    "ABTestManager",
    "ExperimentConfig",
    "ExperimentMonitor",
    "ExperimentReporter",
    "ExperimentSecurity",
    "ExperimentResult",
    "FeatureFlag",
    "ABTestGroup",
    "ExperimentMetrics"
]