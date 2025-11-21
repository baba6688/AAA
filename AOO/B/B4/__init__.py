"""
B4异常检测器包
Anomaly Detector
"""

from .AnomalyDetector import (
    AnomalyType,                          # 异常类型枚举
    SeverityLevel,                        # 严重级别枚举
    AlertStatus,                          # 告警状态枚举
    AnomalyEvent,                         # 异常事件
    AnomalyDatabase,                      # 异常数据库
    StatisticalAnomalyDetector,           # 统计异常检测器
    MachineLearningAnomalyDetector,       # 机器学习异常检测器
    PriceAnomalyDetector,                 # 价格异常检测器
    VolumeAnomalyDetector,                # 成交量异常检测器
    TechnicalIndicatorAnomalyDetector,    # 技术指标异常检测器
    CrossAssetAnomalyDetector,            # 跨资产异常检测器
    MarketStructureAnomalyDetector,       # 市场结构异常检测器
    AnomalyAlertSystem,                   # 异常告警系统
    AnomalyDetector                       # 异常检测器主类
)

__version__ = "1.0.0"
__author__ = "B4 Team"

__all__ = [
    'AnomalyType',
    'SeverityLevel',
    'AlertStatus',
    'AnomalyEvent',
    'AnomalyDatabase',
    'StatisticalAnomalyDetector',
    'MachineLearningAnomalyDetector',
    'PriceAnomalyDetector',
    'VolumeAnomalyDetector',
    'TechnicalIndicatorAnomalyDetector',
    'CrossAssetAnomalyDetector',
    'MarketStructureAnomalyDetector',
    'AnomalyAlertSystem',
    'AnomalyDetector'
]