"""
V2模块 - 模型验证器模块

这个模块包含全面的模型验证功能，用于验证机器学习模型的各种属性和性能指标。
支持数据验证、模型结构验证、参数验证、兼容性检查、交叉验证等功能。

主要功能：
1. 数据验证和清洗 - 验证输入数据的质量和完整性
2. 模型结构验证 - 验证模型架构的合理性
3. 参数验证和边界检查 - 验证模型参数的合法性
4. 模型兼容性检查 - 验证模型与环境的兼容性
5. 数据集分割和验证 - 智能的数据集分割策略
6. 交叉验证实现 - 多种交叉验证方法
7. 验证结果统计 - 详细的验证结果分析
8. 验证报告生成 - 专业的验证报告输出
"""

from .ModelValidator import (
    # 核心类和枚举
    ValidationResult,
    ValidationConfig,
    ModelValidator,
    
    # 便利函数
    create_model_validator
)

__all__ = [
    'ValidationResult',
    'ValidationConfig',
    'ModelValidator',
    'create_model_validator'
]

__version__ = '2.0.0'

# 便利函数
def create_model_validator(config: ValidationConfig = None, **kwargs):
    """
    创建模型验证器实例
    
    Args:
        config: 验证配置对象
        **kwargs: 验证器参数
    
    Returns:
        ModelValidator: 模型验证器实例
    
    Examples:
        from V2 import create_model_validator, ValidationConfig
        
        # 使用默认配置创建验证器
        validator = create_model_validator()
        
        # 使用自定义配置创建验证器
        config = ValidationConfig(
            validation_method='kfold',
            n_splits=5,
            random_state=42
        )
        validator = create_model_validator(config=config)
    """
    return ModelValidator(config=config, **kwargs)

# 验证类型
class ValidationTypes:
    """验证类型常量"""
    DATA_VALIDATION = "data_validation"
    MODEL_STRUCTURE = "model_structure"
    PARAMETER_VALIDATION = "parameter_validation"
    CROSS_VALIDATION = "cross_validation"
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"

# 验证方法
class ValidationMethods:
    """验证方法枚举"""
    TRAIN_TEST_SPLIT = "train_test_split"
    K_FOLD = "kfold"
    STRATIFIED_K_FOLD = "stratified_kfold"
    LEAVE_ONE_OUT = "leave_one_out"
    BOOTSTRAP = "bootstrap"

# 快速开始指南
QUICK_START = """
V2模型验证器快速开始：

1. 创建验证器：
   from V2 import create_model_validator
   validator = create_model_validator()

2. 验证模型：
   # 准备数据
   X, y = load_your_data()
   model = load_your_model()
   
   # 执行验证
   result = validator.validate_model(model, X, y)
   
3. 查看验证结果：
   print(f"验证是否通过: {result.is_valid}")
   print(f"验证得分: {result.validation_score:.3f}")
   
4. 生成验证报告：
   report = validator.generate_validation_report(result)
   print(report)

5. 检查数据质量：
   data_result = validator.validate_data(X, y)
   print(f"数据质量得分: {data_result.data_quality_score:.3f}")
"""

# 数据质量检查装饰器
def validate_data_quality(func):
    """数据质量检查装饰器"""
    def wrapper(*args, **kwargs):
        import numpy as np
        import pandas as pd
        
        # 获取数据参数
        data_args = [arg for arg in args if isinstance(arg, (np.ndarray, pd.DataFrame))]
        
        if not data_args:
            return func(*args, **kwargs)
        
        # 基本数据质量检查
        for data in data_args:
            if isinstance(data, (np.ndarray, pd.DataFrame)):
                if data.size == 0:
                    raise ValueError("数据不能为空")
                if np.any(np.isnan(data)):
                    print(f"警告: 数据包含 {np.isnan(data).sum()} 个NaN值")
        
        return func(*args, **kwargs)
    
    return wrapper

print("V2模型验证器已加载")