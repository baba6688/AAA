"""
K8环境配置处理器模块

该模块提供了一套完整的Kubernetes环境配置管理解决方案，支持开发、测试、生产环境的
配置管理、环境变量管理、依赖管理、环境切换、监控和健康检查等功能。

主要导出类和函数：
- EnvironmentConfigurationProcessor: 主环境配置处理器类
- ExtendedEnvironmentConfigurationProcessor: 扩展环境配置处理器类
- EnvironmentType: 环境类型枚举
- ConfigStatus: 配置状态枚举
- EnvironmentConfig: 环境配置数据类
- 各种环境处理器类和管理器类

作者：K8配置管理团队
版本：1.0.0
"""

# 主类和扩展类
from .EnvironmentConfigurationProcessor import (
    EnvironmentConfigurationProcessor,
    ExtendedEnvironmentConfigurationProcessor,
)

# 枚举类
from .EnvironmentConfigurationProcessor import (
    EnvironmentType,
    ConfigStatus,
    DependencyStatus,
    HealthStatus,
)

# 数据类
from .EnvironmentConfigurationProcessor import (
    EnvironmentConfig,
    DependencyInfo,
    HealthCheckResult,
)

# 异常类
from .EnvironmentConfigurationProcessor import (
    ConfigurationError,
    DependencyError,
    HealthCheckError,
)

# 环境处理器类
from .EnvironmentConfigurationProcessor import (
    EnvironmentProcessor,
    DevelopmentEnvironmentProcessor,
    TestingEnvironmentProcessor,
    ProductionEnvironmentProcessor,
    StagingEnvironmentProcessor,
)

# 管理器类
from .EnvironmentConfigurationProcessor import (
    VariableManager,
    DependencyManager,
    HealthChecker,
    EnvironmentMigrator,
    ConfigurationValidator,
    ConfigurationBackupManager,
    PerformanceMonitor,
    ConfigurationAuditor,
)

# 工具函数
from .EnvironmentConfigurationProcessor import (
    create_development_config,
    create_testing_config,
    create_production_config,
    create_staging_config,
)

# 类型别名
from .EnvironmentConfigurationProcessor import (
    ConfigDict,
    VariableDict,
    DependencyDict,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "K8配置管理团队"

__all__ = [
    # 主类
    "EnvironmentConfigurationProcessor",
    "ExtendedEnvironmentConfigurationProcessor",
    
    # 枚举
    "EnvironmentType",
    "ConfigStatus", 
    "DependencyStatus",
    "HealthStatus",
    
    # 数据类
    "EnvironmentConfig",
    "DependencyInfo",
    "HealthCheckResult",
    
    # 异常
    "ConfigurationError",
    "DependencyError",
    "HealthCheckError",
    
    # 处理器
    "EnvironmentProcessor",
    "DevelopmentEnvironmentProcessor",
    "TestingEnvironmentProcessor",
    "ProductionEnvironmentProcessor",
    "StagingEnvironmentProcessor",
    
    # 管理器
    "VariableManager",
    "DependencyManager",
    "HealthChecker",
    "EnvironmentMigrator",
    "ConfigurationValidator",
    "ConfigurationBackupManager",
    "PerformanceMonitor",
    "ConfigurationAuditor",
    
    # 工具函数
    "create_development_config",
    "create_testing_config",
    "create_production_config",
    "create_staging_config",
    
    # 类型别名
    "ConfigDict",
    "VariableDict",
    "DependencyDict",
    
    # 版本信息
    "__version__",
    "__author__",
]