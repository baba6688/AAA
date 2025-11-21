"""
L1层系统日志记录器模块

该模块提供完整的系统级日志管理功能，包括：
- 多级别日志记录
- 多输出目标支持
- 日志格式化和模板系统
- 日志轮转和归档管理
- 异步日志记录
- 日志过滤和搜索
- 完整的错误处理

"""

from .SystemLogger import (
    # 枚举类
    LogLevel,
    LogFormat,
    RotationStrategy,
    CompressionType,
    # 数据结构
    LogRecord,
    SystemLoggerConfig,
    # 格式化器
    LogFormatter,
    SimpleFormatter,
    DetailedFormatter,
    JsonFormatter,
    CustomFormatter,
    # 过滤器
    LogFilter,
    LevelFilter,
    PatternFilter,
    ModuleFilter,
    CompositeFilter,
    # 输出目标
    OutputTarget,
    FileTarget,
    ConsoleTarget,
    NetworkTarget,
    DatabaseTarget,
    # 管理器
    LogRotationManager,
    AsyncLogProcessor,
    LogSearchEngine,
    # 主要类
    SystemLogger,
    # 工厂函数
    create_system_logger,
    setup_system_logging
)

__version__ = "1.0.0"
__author__ = "System Logger Team"

__all__ = [
    # 枚举类
    "LogLevel",
    "LogFormat",
    "RotationStrategy",
    "CompressionType",
    # 数据结构
    "LogRecord",
    "SystemLoggerConfig",
    # 格式化器
    "LogFormatter",
    "SimpleFormatter",
    "DetailedFormatter",
    "JsonFormatter",
    "CustomFormatter",
    # 过滤器
    "LogFilter",
    "LevelFilter",
    "PatternFilter",
    "ModuleFilter",
    "CompositeFilter",
    # 输出目标
    "OutputTarget",
    "FileTarget",
    "ConsoleTarget",
    "NetworkTarget",
    "DatabaseTarget",
    # 管理器
    "LogRotationManager",
    "AsyncLogProcessor",
    "LogSearchEngine",
    # 主要类
    "SystemLogger",
    # 工厂函数
    "create_system_logger",
    "setup_system_logging"
]