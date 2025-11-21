#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3系统升级器模块

提供完整的系统升级管理功能，包括：
- 系统升级需求分析
- 升级方案设计和优化
- 升级过程管理和监控
- 升级效果评估和验证
- 升级风险评估和控制
- 升级历史管理和跟踪
- 升级策略优化和改进


版本: 1.0.0
"""

from .SystemUpgrader import (
    SystemUpgrader,
    SystemUpgradeRequirementAnalyzer,
    UpgradePlanDesigner,
    UpgradeProcessManager,
    UpgradeEffectivenessEvaluator,
    UpgradeRiskAssessor,
    UpgradeHistoryManager,
    UpgradeStrategyOptimizer,
    SystemComponent,
    UpgradeRequirement,
    UpgradePlan,
    UpgradeExecution,
    UpgradeMetrics,
    UpgradeStatus,
    RiskLevel,
    UpgradeType
)

__version__ = "1.0.0"
__author__ = "AI系统"
__all__ = [
    "SystemUpgrader",
    "SystemUpgradeRequirementAnalyzer", 
    "UpgradePlanDesigner",
    "UpgradeProcessManager",
    "UpgradeEffectivenessEvaluator",
    "UpgradeRiskAssessor",
    "UpgradeHistoryManager",
    "UpgradeStrategyOptimizer",
    "SystemComponent",
    "UpgradeRequirement",
    "UpgradePlan",
    "UpgradeExecution",
    "UpgradeMetrics",
    "UpgradeStatus",
    "RiskLevel",
    "UpgradeType"
]