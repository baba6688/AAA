#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L5学习日志记录器模块

这个模块提供了全面的机器学习实验日志记录功能，支持：
- 机器学习模型训练日志（训练进度、损失函数、验证指标）
- 模型评估日志（准确率、精确率、召回率、F1分数）
- 超参数调优日志（参数搜索、最优参数、收敛情况）
- 数据处理日志（数据加载、预处理、特征工程）
- 实验跟踪日志（实验配置、实验结果、对比分析）
- 模型版本日志（模型保存、模型加载、模型切换）
- 异步学习日志处理
- 完整的错误处理和日志记录

主要类：
- LearningLogger: 主要的学习日志记录器类
- 各种数据类和枚举类用于结构化数据
"""

from .LearningLogger import (
    # 主要类
    LearningLogger,
    
    # 枚举类
    LogLevel,
    ExperimentStatus,
    ModelVersionStatus,
    DataProcessingStage,
    HyperparameterSearchStatus,
    
    # 数据类
    TrainingMetrics,
    EvaluationMetrics,
    HyperparameterConfig,
    HyperparameterTrial,
    DataProcessingLog,
    ExperimentConfig,
    ExperimentResult,
    ModelVersion,
    AsyncLogEntry,
    
    # 异常类
    LearningLoggerError,
    DatabaseError,
    ExperimentError,
    ModelVersionError,
    AsyncProcessingError,
    
    # 存储后端
    LogStorageBackend,
    SQLiteStorageBackend
)

__version__ = "1.0.0"
__author__ = "L5学习日志记录器开发团队"

__all__ = [
    # 主要类
    "LearningLogger",
    
    # 枚举类
    "LogLevel",
    "ExperimentStatus", 
    "ModelVersionStatus",
    "DataProcessingStage",
    "HyperparameterSearchStatus",
    
    # 数据类
    "TrainingMetrics",
    "EvaluationMetrics",
    "HyperparameterConfig",
    "HyperparameterTrial",
    "DataProcessingLog",
    "ExperimentConfig",
    "ExperimentResult",
    "ModelVersion",
    "AsyncLogEntry",
    
    # 异常类
    "LearningLoggerError",
    "DatabaseError",
    "ExperimentError",
    "ModelVersionError",
    "AsyncProcessingError",
    
    # 存储后端
    "LogStorageBackend",
    "SQLiteStorageBackend"
]