#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G3风险控制器模块

实现风险控制和管理的核心功能，包括：
1. 风险级别、类型和警报状态枚举
2. 风险事件、指标和合规规则
3. 风险识别、评估、测量引擎
4. 风险控制、监控、报告引擎
5. 风险模型验证和合规引擎
6. 综合风险控制器
"""

from .RiskController import (
    # 枚举类
    RiskLevel,
    RiskType,
    AlertStatus,
    # 核心数据类
    RiskEvent,
    RiskMetric,
    ComplianceRule,
    # 引擎类
    RiskIdentificationEngine,
    RiskAssessmentEngine,
    RiskMeasurementEngine,
    RiskControlEngine,
    RiskMonitoringEngine,
    RiskReportingEngine,
    RiskModelValidationEngine,
    RiskComplianceEngine,
    # 主控制器
    RiskController
)

__version__ = "1.0.0"
__author__ = "G3 Team"

__all__ = [
    # 枚举类
    "RiskLevel",
    "RiskType",
    "AlertStatus",
    # 核心数据类
    "RiskEvent",
    "RiskMetric",
    "ComplianceRule",
    # 引擎类
    "RiskIdentificationEngine",
    "RiskAssessmentEngine",
    "RiskMeasurementEngine",
    "RiskControlEngine",
    "RiskMonitoringEngine",
    "RiskReportingEngine",
    "RiskModelValidationEngine",
    "RiskComplianceEngine",
    # 主控制器
    "RiskController"
]