"""
K2模块配置处理器包

该包提供了完整的模块配置管理功能，包括：
- 模块级配置管理
- 配置模板和默认配置
- 配置继承和覆盖机制
- 动态加载和卸载
- 依赖解析和验证
- 热更新和实时生效
- 异步处理和批处理
- 完整的错误处理和日志记录

作者: 智能量化系统
版本: 1.0.0
创建时间: 2025-11-06
"""

from .ModuleConfigurationProcessor import (
    # 核心类和枚举
    ModuleConfigurationProcessor,
    ModuleConfig,
    ConfigTemplate,
    ConfigDependency,
    
    # 配置类型和状态枚举
    ConfigType,
    ConfigStatus,
    ConfigEvent,
    ConfigPriority,
    ConfigScope,
    
    # 异常类
    ConfigError,
    ConfigValidationError,
    ConfigDependencyError,
    ConfigLoadError,
    ConfigUpdateError,
    ConfigNotFoundError,
    ConfigConflictError,
    ConfigTimeoutError,
    CircularDependencyError,
    
    # 核心组件
    ConfigValidator,
    ConfigLoader,
    ConfigInheritance,
    ConfigHotUpdate,
    AsyncConfigProcessor,
    BatchConfigProcessor,
    ConfigEventHandler,
    
    # 事件处理器
    LoggingConfigEventHandler,
    
    # 配置元数据
    ConfigMetadata,
    
    # 类型别名
    ConfigDict,
    ModuleId,
    ConfigId,
    TemplateId,
    DependencyId,
    HandlerId
)

__version__ = "1.0.0"
__author__ = "智能量化系统"
__description__ = "K2模块配置处理器 - 完整的模块配置管理解决方案"

__all__ = [
    # 核心类
    "ModuleConfigurationProcessor",
    "ModuleConfig", 
    "ConfigTemplate",
    "ConfigDependency",
    
    # 枚举类
    "ConfigType",
    "ConfigStatus", 
    "ConfigEvent",
    "ConfigPriority",
    "ConfigScope",
    
    # 异常类
    "ConfigError",
    "ConfigValidationError",
    "ConfigDependencyError",
    "ConfigLoadError", 
    "ConfigUpdateError",
    "ConfigNotFoundError",
    "ConfigConflictError",
    "ConfigTimeoutError",
    "CircularDependencyError",
    
    # 组件类
    "ConfigValidator",
    "ConfigLoader",
    "ConfigInheritance",
    "ConfigHotUpdate",
    "AsyncConfigProcessor",
    "BatchConfigProcessor",
    "ConfigEventHandler",
    
    # 事件处理器
    "LoggingConfigEventHandler",
    
    # 数据类
    "ConfigMetadata",
    
    # 类型别名
    "ConfigDict",
    "ModuleId",
    "ConfigId", 
    "TemplateId",
    "DependencyId",
    "HandlerId"
]