"""
K1系统配置管理器模块

提供完整的系统配置管理功能，包括配置加载、验证、缓存、加密等。
"""

# 主要的公共接口类和函数
from .SystemConfigurationManager import (
    # 主要配置管理器
    SystemConfigurationManager,
    ExtendedSystemConfigurationManager,
    UltimateSystemConfigurationManager,
    
    # 核心组件类
    ConfigurationValidator,
    ConfigurationHistory,
    ConfigurationCache,
    ConfigurationEncryptor,
    ConfigurationLoader,
    
    # 配置模板和验证相关
    ConfigurationTemplate,
    ConfigurationMerger,
    ConfigurationDependencyManager,
    ConfigurationPerformanceMonitor,
    AdvancedConfigurationValidator,
    ConfigurationEnvironmentManager,
    ConfigurationBackupManager,
    ConfigurationHealthChecker,
    
    # 配置导入导出相关
    ConfigurationImporter,
    ConfigurationExporter,
    ConfigurationComparator,
    
    # 格式处理器
    TOMLFormatHandler,
    PropertiesFormatHandler,
    AdvancedValidationRules,
    ConfigurationSchemaValidator,
    
    # 测试和基准相关
    ConfigurationBenchmark,
    ConfigurationManagerTestSuite,
)

# 异常类
from .SystemConfigurationManager import (
    ConfigurationError,
    ConfigurationValidationError,
    ConfigurationLoadError,
    ConfigurationSaveError,
    ConfigurationEncryptionError,
    ConfigurationCacheError,
)

# 枚举类
from .SystemConfigurationManager import (
    ConfigurationFormat,
    LogLevel,
)

# 数据类
from .SystemConfigurationManager import (
    ConfigurationMetadata,
    ConfigurationValidationRule,
    ConfigurationChange,
)

# 类型定义
from .SystemConfigurationManager import (
    T,
    ConfigValue,
    ConfigurationData,
)

# 定义模块的公共接口
__all__ = [
    # 主要配置管理器类
    'SystemConfigurationManager',
    'ExtendedSystemConfigurationManager', 
    'UltimateSystemConfigurationManager',
    
    # 核心组件类
    'ConfigurationValidator',
    'ConfigurationHistory',
    'ConfigurationCache',
    'ConfigurationEncryptor',
    'ConfigurationLoader',
    
    # 配置管理组件
    'ConfigurationTemplate',
    'ConfigurationMerger',
    'ConfigurationDependencyManager',
    'ConfigurationPerformanceMonitor',
    'AdvancedConfigurationValidator',
    'ConfigurationEnvironmentManager',
    'ConfigurationBackupManager',
    'ConfigurationHealthChecker',
    
    # 导入导出组件
    'ConfigurationImporter',
    'ConfigurationExporter',
    'ConfigurationComparator',
    
    # 格式处理器
    'TOMLFormatHandler',
    'PropertiesFormatHandler',
    'AdvancedValidationRules',
    'ConfigurationSchemaValidator',
    
    # 工具类
    'ConfigurationBenchmark',
    'ConfigurationManagerTestSuite',
    
    # 异常类
    'ConfigurationError',
    'ConfigurationValidationError',
    'ConfigurationLoadError',
    'ConfigurationSaveError',
    'ConfigurationEncryptionError',
    'ConfigurationCacheError',
    
    # 枚举类
    'ConfigurationFormat',
    'LogLevel',
    
    # 数据类
    'ConfigurationMetadata',
    'ConfigurationValidationRule',
    'ConfigurationChange',
    
    # 类型定义
    'T',
    'ConfigValue',
    'ConfigurationData',
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'K1系统开发团队'
__email__ = 'k1-team@example.com'
__description__ = 'K1系统配置管理器 - 提供完整的配置管理功能'

# 便捷导入函数
def create_config_manager(config_dir: str = "configs", **kwargs):
    """
    创建系统配置管理器实例的便捷函数
    
    Args:
        config_dir: 配置文件目录
        **kwargs: 其他初始化参数
        
    Returns:
        SystemConfigurationManager: 配置管理器实例
    """
    return SystemConfigurationManager(config_dir=config_dir, **kwargs)

def create_basic_config_manager():
    """
    创建基础配置管理器的便捷函数
    
    Returns:
        SystemConfigurationManager: 基础配置管理器实例
    """
    return SystemConfigurationManager()

# 向后兼容性
def get_version():
    """获取版本信息"""
    return __version__

def get_author():
    """获取作者信息"""
    return __author__