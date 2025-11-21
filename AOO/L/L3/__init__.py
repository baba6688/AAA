"""
L3层级错误日志记录器模块

本模块提供了完整的错误日志记录功能，包括：
- 异常捕获和分类记录
- 错误堆栈跟踪和上下文信息
- 错误统计和分析
- 错误告警和通知机制
- 错误恢复和重试日志
- 错误解决和关闭流程
- 异步错误日志处理

主要类：
- ErrorLogger: 主要的错误日志记录器
- ErrorContext: 错误上下文信息
- ErrorStatistics: 错误统计分析
- AlertManager: 错误告警管理器
- RecoveryManager: 错误恢复管理器
- ErrorType: 错误类型枚举
- ErrorSeverity: 错误严重级别枚举
- ErrorStatus: 错误状态枚举
- AlertChannel: 告警渠道枚举
- RecoveryStrategy: 恢复策略枚举
"""

from ErrorLogger import (
    ErrorLogger,
    ErrorContext,
    ErrorStatistics,
    AlertManager,
    RecoveryManager,
    ErrorType,
    ErrorSeverity,
    ErrorStatus,
    AlertChannel,
    RecoveryStrategy
)

__all__ = [
    'ErrorLogger',
    'ErrorContext', 
    'ErrorStatistics',
    'AlertManager',
    'RecoveryManager',
    'ErrorType',
    'ErrorSeverity',
    'ErrorStatus',
    'AlertChannel',
    'RecoveryStrategy'
]

__version__ = '1.0.0'