"""
V区 - 机器学习模型版本控制系统
============================

V区是一个完整的机器学习模型生命周期管理平台，提供从模型训练到部署的全流程解决方案。

功能模块:
- V1: 模型训练器 (ModelTrainer) - 模型训练、超参数调优、分布式训练
- V2: 模型验证器 (ModelValidator) - 数据验证、结构验证、交叉验证
- V3: 模型评估器 (ModelEvaluator) - 性能评估、指标分析、可视化
- V4: 模型部署器 (ModelDeployer) - 模型部署、多格式支持、容器化
- V5: 模型版本控制器 (ModelVersionController) - 版本控制、分支管理、变更追踪
- V6: 模型监控器 (ModelMonitor) - 性能监控、数据漂移、衰减检测
- V7: 模型解释器 (ModelInterpreter) - 模型解释、SHAP、LIME分析
- V8: 模型转换器 (ModelConverter) - 框架转换、ONNX、TensorRT优化
- V9: 模型状态聚合器 (ModelStateAggregator) - 状态聚合、监控、报告生成

主要特性:
- 全生命周期管理：从训练到部署的完整流程
- 多框架支持：PyTorch、TensorFlow、sklearn、ONNX等
- 版本控制：完善的模型版本管理系统
- 自动化工作流：训练、验证、评估、部署自动化
- 模型监控：实时性能监控和数据漂移检测
- 模型解释：支持SHAP、LIME等多种解释方法
- 多格式转换：支持多种部署格式的转换优化
- 容器化部署：Docker容器化、负载均衡支持

版本: 1.0.0
作者: V区开发团队
许可证: MIT
"""

# ========== V1 - 模型训练器导出 ==========
from V.V1.ModelTrainer import (
    # 核心类和枚举
    TrainingConfig,
    TrainingResult,
    ModelTrainer,
    
    # 便利函数
    create_model_trainer
)

# ========== V2 - 模型验证器导出 ==========
from V.V2.ModelValidator import (
    # 核心类和枚举
    ValidationResult,
    ValidationConfig,
    ModelValidator,
    
    # 便利函数
    create_model_validator
)

# ========== V3 - 模型评估器导出 ==========
from V.V3.ModelEvaluator import (
    # 核心类
    ModelEvaluator,
    
    # 便利函数
    create_model_evaluator
)

# ========== V4 - 模型部署器导出 ==========
from V.V4.ModelDeployer import (
    # 核心枚举和类
    DeploymentFormat,
    DeploymentStatus,
    ServerType,
    DeploymentConfig,
    ModelDeployer,
    
    # 便利函数
    create_model_deployer
)

# ========== V5 - 模型版本控制器导出 ==========
from V.V5.ModelVersionController import (
    # 核心枚举和类
    VersionStatus,
    MergeStrategy,
    ModelMetadata,
    ModelVersionController,
    
    # 便利函数
    create_version_controller
)

# ========== V6 - 模型监控器导出 ==========
from V.V6.ModelMonitor import (
    # 核心类和枚举
    MonitoringConfig,
    MonitoringResult,
    ModelMonitor,
    
    # 便利函数
    create_model_monitor
)

# ========== V7 - 模型解释器导出 ==========
from V.V7.ModelInterpreter import (
    # 核心类和枚举
    ExplanationResult,
    QualityMetrics,
    ModelInterpreter,
    
    # 便利函数
    create_model_interpreter
)

# ========== V8 - 模型转换器导出 ==========
from V.V8.ModelConverter import (
    # 核心枚举和类
    ModelFramework,
    ConversionType,
    ModelConverter,
    
    # 便利函数
    create_model_converter
)

# ========== V9 - 模型状态聚合器导出 ==========
from V.V9.ModelStateAggregator import (
    # 核心枚举和类
    ModelStatus,
    AlertLevel,
    HealthStatus,
    ModelInfo,
    ModelStatusAggregator,
    
    # 便利函数
    create_status_aggregator
)

# ========== 主包导出列表 ==========
__all__ = [
    # V1 - 模型训练器
    'TrainingConfig', 'TrainingResult', 'ModelTrainer', 'create_model_trainer',
    
    # V2 - 模型验证器
    'ValidationResult', 'ValidationConfig', 'ModelValidator', 'create_model_validator',
    
    # V3 - 模型评估器
    'ModelEvaluator', 'create_model_evaluator',
    
    # V4 - 模型部署器
    'DeploymentFormat', 'DeploymentStatus', 'ServerType', 'DeploymentConfig',
    'ModelDeployer', 'create_model_deployer',
    
    # V5 - 模型版本控制器
    'VersionStatus', 'MergeStrategy', 'ModelMetadata', 'ModelVersionController',
    'create_version_controller',
    
    # V6 - 模型监控器
    'MonitoringConfig', 'MonitoringResult', 'ModelMonitor', 'create_model_monitor',
    
    # V7 - 模型解释器
    'ExplanationResult', 'QualityMetrics', 'ModelInterpreter', 'create_model_interpreter',
    
    # V8 - 模型转换器
    'ModelFramework', 'ConversionType', 'ModelConverter', 'create_model_converter',
    
    # V9 - 模型状态聚合器
    'ModelStatus', 'AlertLevel', 'HealthStatus', 'ModelInfo',
    'ModelStatusAggregator', 'create_status_aggregator'
]

# ========== 包信息 ==========
__version__ = "1.0.0"
__author__ = "V区开发团队"
__license__ = "MIT"
__email__ = "v-team@example.com"
__url__ = "https://github.com/company/v-ml-version-control"

# ========== 工厂函数 ==========
def create_ml_component(component_type: str, **kwargs):
    """
    创建机器学习组件的便利工厂函数
    
    Args:
        component_type: 组件类型 ('trainer', 'validator', 'evaluator', 'deployer', 'version', 'monitor', 'interpreter', 'converter', 'aggregator')
        **kwargs: 组件特定参数
    
    Returns:
        对应的机器学习组件实例
    
    Examples:
        # 创建模型训练器
        trainer = create_ml_component('trainer', config=TrainingConfig(...))
        
        # 创建模型评估器
        evaluator = create_ml_component('evaluator', task_type='classification')
        
        # 创建模型监控器
        monitor = create_ml_component('monitor', config=MonitoringConfig(...))
    """
    component_map = {
        'trainer': create_model_trainer,
        'validator': create_model_validator,
        'evaluator': create_model_evaluator,
        'deployer': create_model_deployer,
        'version': create_version_controller,
        'monitor': create_model_monitor,
        'interpreter': create_model_interpreter,
        'converter': create_model_converter,
        'aggregator': create_status_aggregator
    }
    
    if component_type not in component_map:
        raise ValueError(
            f"不支持的组件类型: {component_type}。支持的类型: {list(component_map.keys())}"
        )
    
    return component_map[component_type](**kwargs)

def get_available_modules():
    """获取所有可用的机器学习模块"""
    return {
        'V1': 'ModelTrainer - 模型训练器',
        'V2': 'ModelValidator - 模型验证器',
        'V3': 'ModelEvaluator - 模型评估器',
        'V4': 'ModelDeployer - 模型部署器',
        'V5': 'ModelVersionController - 模型版本控制器',
        'V6': 'ModelMonitor - 模型监控器',
        'V7': 'ModelInterpreter - 模型解释器',
        'V8': 'ModelConverter - 模型转换器',
        'V9': 'ModelStateAggregator - 模型状态聚合器'
    }

def get_module_info(module_name: str):
    """
    获取指定模块的详细信息
    
    Args:
        module_name: 模块名称 (V1-V9)
    
    Returns:
        模块信息字典
    
    Raises:
        ValueError: 无效的模块名称
    """
    module_info = {
        'V1': {
            'name': 'ModelTrainer',
            'description': '模型训练器，支持分布式训练、超参数调优、早停机制',
            'components': ['TrainingConfig', 'TrainingResult', 'ModelTrainer']
        },
        'V2': {
            'name': 'ModelValidator',
            'description': '模型验证器，支持数据验证、结构验证、交叉验证',
            'components': ['ValidationResult', 'ValidationConfig', 'ModelValidator']
        },
        'V3': {
            'name': 'ModelEvaluator',
            'description': '模型评估器，支持性能评估、可视化分析、报告生成',
            'components': ['ModelEvaluator']
        },
        'V4': {
            'name': 'ModelDeployer',
            'description': '模型部署器，支持多格式部署、容器化、负载均衡',
            'components': ['DeploymentConfig', 'ModelDeployer', 'DeploymentFormat']
        },
        'V5': {
            'name': 'ModelVersionController',
            'description': '模型版本控制器，支持版本管理、分支控制、变更追踪',
            'components': ['ModelMetadata', 'ModelVersionController', 'VersionStatus']
        },
        'V6': {
            'name': 'ModelMonitor',
            'description': '模型监控器，支持性能监控、数据漂移、衰减检测',
            'components': ['MonitoringConfig', 'MonitoringResult', 'ModelMonitor']
        },
        'V7': {
            'name': 'ModelInterpreter',
            'description': '模型解释器，支持SHAP、LIME、特征重要性分析',
            'components': ['ExplanationResult', 'QualityMetrics', 'ModelInterpreter']
        },
        'V8': {
            'name': 'ModelConverter',
            'description': '模型转换器，支持框架转换、ONNX、TensorRT优化',
            'components': ['ModelConverter', 'ModelFramework', 'ConversionType']
        },
        'V9': {
            'name': 'ModelStateAggregator',
            'description': '模型状态聚合器，支持状态聚合、监控、报告生成',
            'components': ['ModelStatusAggregator', 'HealthStatus', 'AlertLevel']
        }
    }
    
    if module_name not in module_info:
        raise ValueError(f"无效的模块名称: {module_name}。支持的模块: V1-V9")
    
    return module_info[module_name]

# ========== 工作流便利函数 ==========
def create_ml_pipeline(pipeline_type: str = "full", **kwargs):
    """
    创建机器学习工作流的便利函数
    
    Args:
        pipeline_type: 工作流类型 ('full', 'training', 'deployment', 'monitoring')
        **kwargs: 工作流特定参数
    
    Returns:
        机器学习工作流字典
    
    Examples:
        # 创建完整工作流
        pipeline = create_ml_pipeline('full', task_type='classification')
        
        # 创建训练工作流
        training_pipeline = create_ml_pipeline('training', model_type='random_forest')
    """
    if pipeline_type == "full":
        return {
            'trainer': create_ml_component('trainer', **kwargs),
            'validator': create_ml_component('validator'),
            'evaluator': create_ml_component('evaluator'),
            'deployer': create_ml_component('deployer'),
            'monitor': create_ml_component('monitor'),
            'version': create_ml_component('version')
        }
    elif pipeline_type == "training":
        return {
            'trainer': create_ml_component('trainer', **kwargs),
            'validator': create_ml_component('validator'),
            'evaluator': create_ml_component('evaluator')
        }
    elif pipeline_type == "deployment":
        return {
            'deployer': create_ml_component('deployer', **kwargs),
            'version': create_ml_component('version'),
            'monitor': create_ml_component('monitor')
        }
    elif pipeline_type == "monitoring":
        return {
            'monitor': create_ml_component('monitor', **kwargs),
            'aggregator': create_ml_component('aggregator'),
            'interpreter': create_ml_component('interpreter')
        }
    else:
        raise ValueError(f"不支持的工作流类型: {pipeline_type}")

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 显示可用模块
    print("V区 - 机器学习模型版本控制系统")
    print("=" * 50)
    
    modules = get_available_modules()
    for code, description in modules.items():
        print(f"{code}: {description}")
    
    print("\n使用示例:")
    print("from V import create_ml_component, create_ml_pipeline")
    print("trainer = create_ml_component('trainer')")
    print("evaluator = create_ml_component('evaluator', task_type='classification')")
    print("full_pipeline = create_ml_pipeline('full')")