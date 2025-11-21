"""
K7学习配置管理器模块

这是一个全面的机器学习配置管理系统，支持模型配置、训练、验证、评估、
部署、更新和实验管理的完整生命周期。

主要功能:
- 机器学习模型配置管理
- 训练、验证、评估配置
- 模型部署和更新配置
- 实验配置管理
- 异步处理支持
- 配置验证和分析
- 模板系统
- 性能监控

作者: K7团队
版本: 1.0.0
创建日期: 2025-11-06
"""

# 核心异常类
from .LearningConfigurationManager import (
    ConfigurationError,
    ValidationError,
    DeploymentError,
    ExperimentError,
    AsyncProcessingError
)

# 状态枚举
from .LearningConfigurationManager import (
    ConfigurationStatus,
    DeploymentStatus,
    ExperimentStatus,
    ModelTypeEnum
)

# 基础配置类
from .LearningConfigurationManager import (
    BaseConfiguration
)

# ML配置类
from .LearningConfigurationManager import (
    MLModelConfiguration,
    TrainingConfiguration,
    ValidationConfiguration,
    EvaluationConfiguration,
    DeploymentConfiguration,
    ModelUpdateConfiguration,
    ExperimentConfiguration
)

# 扩展配置类
from .LearningConfigurationManager import (
    MonitoringConfiguration,
    DataPipelineConfiguration,
    ABTestConfiguration,
    ModelRegistryConfiguration,
    FeatureStoreConfiguration
)

# 核心处理器
from .LearningConfigurationManager import (
    AsyncConfigurationProcessor,
    ConfigurationValidator
)

# 核心管理器
from .LearningConfigurationManager import (
    ConfigurationManager
)

# 主要管理器类
from .LearningConfigurationManager import (
    LearningConfigurationManager
)

# 高级管理器
from .LearningConfigurationManager import (
    AdvancedConfigurationManager,
    ExtendedLearningConfigurationManager
)

# 工具类
from .LearningConfigurationManager import (
    ConfigurationTemplate,
    TemplateManager,
    AdvancedConfigurationValidator,
    ConfigurationAnalyzer,
    ConfigurationPerformanceMonitor,
    ConfigurationOptimizer
)

# 导出所有公共接口
__all__ = [
    # 异常类
    'ConfigurationError',
    'ValidationError', 
    'DeploymentError',
    'ExperimentError',
    'AsyncProcessingError',
    
    # 枚举
    'ConfigurationStatus',
    'DeploymentStatus',
    'ExperimentStatus',
    'ModelTypeEnum',
    
    # 基础配置
    'BaseConfiguration',
    
    # ML配置
    'MLModelConfiguration',
    'TrainingConfiguration',
    'ValidationConfiguration',
    'EvaluationConfiguration',
    'DeploymentConfiguration',
    'ModelUpdateConfiguration',
    'ExperimentConfiguration',
    
    # 扩展配置
    'MonitoringConfiguration',
    'DataPipelineConfiguration',
    'ABTestConfiguration',
    'ModelRegistryConfiguration',
    'FeatureStoreConfiguration',
    
    # 处理器
    'AsyncConfigurationProcessor',
    'ConfigurationValidator',
    
    # 管理器
    'ConfigurationManager',
    'LearningConfigurationManager',
    'AdvancedConfigurationManager',
    'ExtendedLearningConfigurationManager',
    
    # 工具类
    'ConfigurationTemplate',
    'TemplateManager',
    'AdvancedConfigurationValidator',
    'ConfigurationAnalyzer',
    'ConfigurationPerformanceMonitor',
    'ConfigurationOptimizer'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "K7团队"
__description__ = "K7学习配置管理器 - 全面的机器学习配置管理系统"

# 便捷导入函数
def create_learning_manager(storage_path="k7_configs"):
    """
    创建学习配置管理器的便捷函数
    
    Args:
        storage_path (str): 存储路径
        
    Returns:
        LearningConfigurationManager: 学习配置管理器实例
    """
    return LearningConfigurationManager(storage_path)

def create_extended_manager(storage_path="extended_configs"):
    """
    创建扩展学习配置管理器的便捷函数
    
    Args:
        storage_path (str): 存储路径
        
    Returns:
        ExtendedLearningConfigurationManager: 扩展学习配置管理器实例
    """
    return ExtendedLearningConfigurationManager(storage_path)

# 模块初始化日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"K7学习配置管理器模块已加载，版本: {__version__}")