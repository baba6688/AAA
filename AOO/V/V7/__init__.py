"""
V7模块 - 模型解释器模块

实现完整的模型解释功能，包括SHAP、LIME、特征重要性分析等多种解释方法。

主要功能：
1. SHAP解释 - 使用SHAP值解释模型预测
2. LIME解释 - 使用LIME解释局部预测
3. 特征重要性 - 计算特征重要性排序
4. 模型解释质量评估 - 评估解释的保真度、稳定性
5. 可视化解释 - 生成解释结果的可视化
6. 批处理解释 - 支持批量数据解释
7. 解释报告生成 - 生成专业解释报告
8. 模型类型适配 - 支持不同类型的模型解释
"""

from .ModelInterpreter import (
    # 核心类和枚举
    ExplanationResult,
    QualityMetrics,
    ModelInterpreter,
    
    # 便利函数
    create_model_interpreter
)

__all__ = [
    'ExplanationResult',
    'QualityMetrics',
    'ModelInterpreter',
    'create_model_interpreter'
]

__version__ = '1.0.0'

# 便利函数
def create_model_interpreter(**kwargs):
    """
    创建模型解释器实例
    
    Args:
        **kwargs: 解释器参数
    
    Returns:
        ModelInterpreter: 模型解释器实例
    
    Examples:
        from V7 import create_model_interpreter
        
        # 创建基本解释器
        interpreter = create_model_interpreter()
        
        # 创建带自定义参数的解释器
        interpreter = create_model_interpreter(
            n_samples=1000,
            random_state=42
        )
    """
    return ModelInterpreter(**kwargs)

# 解释方法类型
class InterpretationMethods:
    """解释方法类型常量"""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    LIME_TEXT = "lime_text"

# 模型类型
class ModelTypes:
    """模型类型常量"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"

# 解释质量级别
class QualityLevels:
    """解释质量级别"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# 可视化类型
class VisualizationTypes:
    """可视化类型常量"""
    SUMMARY_PLOT = "summary_plot"
    WATERFALL_PLOT = "waterfall_plot"
    FORCE_PLOT = "force_plot"
    DEPENDENCE_PLOT = "dependence_plot"
    DECISION_PLOT = "decision_plot"
    HEATMAP = "heatmap"

# 快速开始指南
QUICK_START = """
V7模型解释器快速开始：

1. 创建解释器：
   from V7 import create_model_interpreter
   interpreter = create_model_interpreter()

2. 准备数据：
   # 加载模型和数据
   model = load_your_model()
   X, y = load_your_data()
   
   # 准备要解释的样本
   X_sample = X[:10]  # 选择要解释的样本

3. SHAP解释：
   # 全局解释
   shap_result = interpreter.explain_with_shap(model, X, method="auto")
   
   # 局部解释
   local_shap = interpreter.explain_sample_with_shap(model, X_sample[0])

4. LIME解释：
   lime_result = interpreter.explain_with_lime(model, X_sample, num_features=10)

5. 特征重要性：
   importance = interpreter.calculate_feature_importance(model, X, y)

6. 生成可视化：
   interpreter.plot_shap_summary(shap_result, save_path="shap_summary.png")
   interpreter.plot_shap_waterfall(shap_result, instance_idx=0)

7. 评估解释质量：
   quality_metrics = interpreter.evaluate_explanation_quality(
       shap_result, X_sample, model
   )
   print(f"解释质量得分: {quality_metrics.overall_score:.3f}")

8. 生成解释报告：
   report = interpreter.generate_explanation_report(
       shap_result, quality_metrics
   )
"""

# 解释器检查装饰器
def validate_interpretation(func):
    """解释功能验证装饰器"""
    def wrapper(*args, **kwargs):
        import numpy as np
        
        # 检查模型和数据参数
        for arg in args:
            if hasattr(arg, 'predict'):  # 可能是模型
                # 基本模型检查
                try:
                    test_input = np.random.rand(1, 5)  # 假设5个特征
                    pred = arg.predict(test_input)
                    print(f"模型预测测试通过: {pred}")
                except Exception as e:
                    print(f"模型预测测试失败: {e}")
        
        return func(*args, **kwargs)
    
    return wrapper

# 批量解释装饰器
def batch_interpretation(batch_size: int = 100):
    """批量解释装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import numpy as np
            
            # 获取要解释的数据
            X = None
            for arg in args:
                if isinstance(arg, np.ndarray) and len(arg.shape) == 2:
                    X = arg
                    break
            
            if X is None or len(X) <= batch_size:
                return func(*args, **kwargs)
            
            # 批量处理
            results = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_args = [batch if j == args.index(X) else arg for j, arg in enumerate(args)]
                batch_result = func(*batch_args, **{**kwargs, 'batch_info': f"batch_{i//batch_size + 1}"})
                results.append(batch_result)
            
            print(f"批量解释完成: {len(results)} 个批次")
            return results
        
        return wrapper
    return decorator

# 解释结果缓存装饰器
def cache_interpretation(func):
    """解释结果缓存装饰器"""
    def wrapper(*args, **kwargs):
        import hashlib
        import pickle
        import os
        
        # 生成缓存键
        cache_key = hashlib.md5(str(args).encode()).hexdigest()
        cache_file = f"./interpretation_cache/{cache_key}.pkl"
        
        # 检查缓存
        if os.path.exists(cache_file):
            print("使用缓存的解释结果")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 执行解释并缓存
        result = func(*args, **kwargs)
        
        # 确保缓存目录存在
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    
    return wrapper

print("V7模型解释器已加载")