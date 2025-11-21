"""
V3模型评估器模块
=================

提供全面的模型性能评估功能，支持分类和回归任务。

主要功能:
- 多指标评估（准确率、召回率、F1等）
- 混淆矩阵和ROC曲线可视化
- 回归评估指标（MAE、MSE、R2等）
- 分类和回归模型评估
- 模型性能比较
- 评估结果可视化
- 评估报告生成
- 评估结果存储
- 评估结果解释
"""

from .ModelEvaluator import (
    # 核心类
    ModelEvaluator,
    
    # 便利函数
    create_model_evaluator
)

__version__ = "3.0"
__author__ = "V3开发团队"
__email__ = "v3@example.com"

__all__ = [
    'ModelEvaluator',
    'create_model_evaluator'
]

# 便利函数
def create_model_evaluator(task_type: str = "classification", save_path: str = "./evaluation_results"):
    """
    创建模型评估器实例
    
    Args:
        task_type: 任务类型 ("classification" 或 "regression")
        save_path: 结果保存路径
    
    Returns:
        ModelEvaluator: 模型评估器实例
    
    Examples:
        from V3 import create_model_evaluator
        
        # 创建分类模型评估器
        evaluator = create_model_evaluator(task_type="classification")
        
        # 创建回归模型评估器
        evaluator = create_model_evaluator(task_type="regression")
    """
    return ModelEvaluator(task_type=task_type, save_path=save_path)

# 评估指标类型
class EvaluationMetrics:
    """评估指标类型常量"""
    # 分类指标
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    
    # 回归指标
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2_score"
    MAPE = "mape"

# 可视化类型
class VisualizationTypes:
    """可视化类型常量"""
    CONFUSION_MATRIX = "confusion_matrix"
    ROC_CURVE = "roc_curve"
    PRECISION_RECALL_CURVE = "precision_recall_curve"
    FEATURE_IMPORTANCE = "feature_importance"
    RESIDUAL_PLOT = "residual_plot"
    PREDICTION_VS_ACTUAL = "prediction_vs_actual"

# 快速开始指南
QUICK_START = """
V3模型评估器快速开始：

1. 创建评估器：
   from V3 import create_model_evaluator
   evaluator = create_model_evaluator(task_type="classification")

2. 评估分类模型：
   # 准备数据
   y_true, y_pred, y_prob = load_test_data()
   
   # 评估模型
   results = evaluator.evaluate_classification(
       y_true=y_true,
       y_pred=y_pred, 
       y_prob=y_prob,
       model_name="MyModel"
   )
   
3. 生成可视化：
   evaluator.plot_confusion_matrix("MyModel")
   evaluator.plot_roc_curve("MyModel")

4. 比较模型性能：
   comparison = evaluator.compare_models(
       model_names=["Model1", "Model2"],
       metric="accuracy"
   )
   
5. 生成评估报告：
   report_path = evaluator.generate_report("MyModel", "html")
"""

# 评估结果装饰器
def evaluate_model_performance(func):
    """模型性能评估装饰器"""
    def wrapper(*args, **kwargs):
        import time
        import numpy as np
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # 如果结果包含预测值和真实值，自动评估性能
        if hasattr(result, '__dict__'):
            # 检查是否有预测结果需要评估
            pred_key = None
            true_key = None
            
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    if 'pred' in key.lower():
                        pred_key = key
                    elif 'true' in key.lower() or 'actual' in key.lower():
                        true_key = key
            
            if pred_key and true_key:
                print(f"自动评估函数 {func.__name__} 的性能...")
                print(f"执行时间: {end_time - start_time:.3f}秒")
        
        return result
    
    return wrapper

print("V3模型评估器已加载")