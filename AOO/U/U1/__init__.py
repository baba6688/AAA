"""
U1机器学习算法库模块

包含完整的机器学习算法实现，包括监督学习、无监督学习和集成学习方法。

主要组件:
- ModelResult: 模型训练和预测结果数据类
- BaseMLAlgorithm: 机器学习算法基类
- LinearRegressionAlgorithm: 线性回归算法
- LogisticRegressionAlgorithm: 逻辑回归算法
- SVMAlgorithm: 支持向量机算法
- RandomForestAlgorithm: 随机森林算法
- GradientBoostingAlgorithm: 梯度提升算法
- NaiveBayesAlgorithm: 朴素贝叶斯算法
- KNNAlgorithm: K近邻算法
- DecisionTreeAlgorithm: 决策树算法
- EnsembleLearning: 集成学习方法
- FeatureSelection: 特征选择算法
- HyperparameterOptimizer: 超参数优化
- MLAlgorithmLibrary: 主算法库类

可用函数:
- create_sample_data: 创建示例数据
- run_comprehensive_test: 运行综合测试

作者: 智能量化系统
版本: 1.0.0
"""

from .MLAlgorithmLibrary import (
    ModelResult,
    BaseMLAlgorithm,
    LinearRegressionAlgorithm,
    LogisticRegressionAlgorithm,
    SVMAlgorithm,
    RandomForestAlgorithm,
    GradientBoostingAlgorithm,
    NaiveBayesAlgorithm,
    KNNAlgorithm,
    DecisionTreeAlgorithm,
    EnsembleLearning,
    FeatureSelection,
    HyperparameterOptimizer,
    MLAlgorithmLibrary,
    create_sample_data,
    run_comprehensive_test
)

__version__ = "1.0.0"
__author__ = "智能量化系统"

__all__ = [
    'ModelResult',
    'BaseMLAlgorithm',
    'LinearRegressionAlgorithm',
    'LogisticRegressionAlgorithm',
    'SVMAlgorithm',
    'RandomForestAlgorithm',
    'GradientBoostingAlgorithm',
    'NaiveBayesAlgorithm',
    'KNNAlgorithm',
    'DecisionTreeAlgorithm',
    'EnsembleLearning',
    'FeatureSelection',
    'HyperparameterOptimizer',
    'MLAlgorithmLibrary',
    'create_sample_data',
    'run_comprehensive_test'
]