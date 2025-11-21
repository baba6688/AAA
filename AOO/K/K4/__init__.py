#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K4风险配置处理器模块

该模块提供全面的风险配置管理功能，包括市场风险、信用风险、操作风险
等各种风险类型的配置管理、监控、告警、对冲和报告功能。

导出类:
- RiskConfigurationProcessor: 风险配置处理器主类
- AlertSystem: 告警系统
- RiskDatabase: 风险配置数据库管理器
- RiskValidator: 风险配置验证器

导出枚举:
- RiskType: 风险类型枚举
- RiskLevel: 风险等级枚举
- AlertSeverity: 告警严重程度枚举
- HedgeStrategy: 对冲策略枚举
- ReportFormat: 报告格式枚举

导出数据类:
- RiskThreshold: 风险阈值配置
- RiskLimit: 风险限制配置
- RiskModel: 风险模型配置
- MarketRiskConfig: 市场风险配置
- CreditRiskConfig: 信用风险配置
- OperationalRiskConfig: 操作风险配置
- MonitoringConfig: 风险监控配置
- AlertConfig: 告警配置
- HedgeConfig: 风险对冲配置
- ReportConfig: 风险报告配置

导出异常类:
- RiskConfigurationError: 风险配置异常基类
- ValidationError: 配置验证异常
- ProcessingError: 处理异常
- DatabaseError: 数据库异常

作者: K4风险配置处理器开发团队
版本: 1.0.0
创建时间: 2025-11-06
"""

# 导入所有公共类和函数
from .RiskConfigurationProcessor import (
    # 主类
    RiskConfigurationProcessor,
    
    # 工具类
    AlertSystem,
    RiskDatabase,
    RiskValidator,
    
    # 枚举类
    RiskType,
    RiskLevel,
    AlertSeverity,
    HedgeStrategy,
    ReportFormat,
    
    # 数据类
    RiskThreshold,
    RiskLimit,
    RiskModel,
    MarketRiskConfig,
    CreditRiskConfig,
    OperationalRiskConfig,
    MonitoringConfig,
    AlertConfig,
    HedgeConfig,
    ReportConfig,
    
    # 异常类
    RiskConfigurationError,
    ValidationError,
    ProcessingError,
    DatabaseError
)

# 定义模块级别的公共接口
__all__ = [
    # 主类
    'RiskConfigurationProcessor',
    
    # 工具类
    'AlertSystem',
    'RiskDatabase',
    'RiskValidator',
    
    # 枚举类
    'RiskType',
    'RiskLevel',
    'AlertSeverity',
    'HedgeStrategy',
    'ReportFormat',
    
    # 数据类
    'RiskThreshold',
    'RiskLimit',
    'RiskModel',
    'MarketRiskConfig',
    'CreditRiskConfig',
    'OperationalRiskConfig',
    'MonitoringConfig',
    'AlertConfig',
    'HedgeConfig',
    'ReportConfig',
    
    # 异常类
    'RiskConfigurationError',
    'ValidationError',
    'ProcessingError',
    'DatabaseError'
]

# 模块元数据
__version__ = '1.0.0'
__author__ = 'K4风险配置处理器开发团队'
__email__ = 'k4-team@company.com'
__description__ = 'K4风险配置处理器 - 提供全面的风险配置管理功能'