"""
U6 - 时间序列算法库
===================

这是一个完整的时间序列分析库，包含多种经典和现代时间序列分析方法。

主要功能模块:
1. 预测模型：ARIMA、指数平滑、卡尔曼滤波、隐马尔可夫模型
2. 信号处理：季节性分解、傅里叶变换、小波分析
3. 异常检测：统计方法、孤立森林、高斯混合模型
4. 评估工具：预测精度指标、模型比较

类列表:
- ForecastResult: 预测结果数据类
- TimeSeriesModel: 时间序列模型基类
- ARIMAModel: ARIMA模型(自回归积分移动平均)
- SeasonalDecomposition: 季节性分解
- ExponentialSmoothing: 指数平滑方法
- KalmanFilter: 卡尔曼滤波
- HiddenMarkovModel: 隐马尔可夫模型
- FourierWaveletAnalysis: 傅里叶变换和小波分析
- AnomalyDetector: 异常检测
- ForecastMetrics: 预测评估指标
- TimeSeriesAlgorithmLibrary: 算法库主类

依赖包:
- numpy: 数值计算
- pandas: 数据处理
- scipy: 科学计算
- sklearn: 机器学习
- matplotlib: 可视化
- statsmodels: 统计模型

版本: 1.0.0
日期: 2025-11-14

使用示例:
---------
from U.U6 import TimeSeriesAlgorithmLibrary, ARIMAModel, ForecastMetrics

# 创建库实例
library = TimeSeriesAlgorithmLibrary()

# 创建ARIMA模型
model = library.create_arima(p=1, d=1, q=1)

# 拟合和预测
model.fit(data)
result = model.predict(steps=10)

# 评估预测结果
metrics = ForecastMetrics.comprehensive_evaluation(true_values, predictions)
"""

# 导入所有核心类和工具函数
from .TimeSeriesAlgorithmLibrary import (
    # 核心类
    ForecastResult,
    TimeSeriesModel,
    ARIMAModel,
    SeasonalDecomposition,
    ExponentialSmoothing,
    KalmanFilter,
    HiddenMarkovModel,
    FourierWaveletAnalysis,
    AnomalyDetector,
    ForecastMetrics,
    TimeSeriesAlgorithmLibrary,
    
    # 工具函数
    generate_sample_data,
    run_comprehensive_example,
    
    # 元信息
    __version__,
    __author__,
    __email__,
    __license__
)

# 重新定义__all__以包含模块元信息
__all__ = [
    # 核心类
    'ForecastResult',
    'TimeSeriesModel', 
    'ARIMAModel',
    'SeasonalDecomposition',
    'ExponentialSmoothing',
    'KalmanFilter',
    'HiddenMarkovModel',
    'FourierWaveletAnalysis',
    'AnomalyDetector',
    'ForecastMetrics',
    'TimeSeriesAlgorithmLibrary',
    
    # 工具函数
    'generate_sample_data',
    'run_comprehensive_example',
    
    # 元信息
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]

# 模块初始化信息
print("U6 - 时间序列算法库已加载")
print("版本:", __version__)
print("可用类:", ', '.join([item for item in __all__ if not item.startswith('_') and item not in ['__version__', '__author__', '__email__', '__license__']]))
print("输入 help(U.U6) 获取详细帮助信息")