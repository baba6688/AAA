#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J2数据处理工具模块

该模块提供了全面的数据处理功能，包括数据清洗、转换、聚合、验证等。
支持异步处理、内存优化和大数据处理。

主要组件:
- DataProcessingTools: 基础数据处理工具
- ExtendedDataProcessingTools: 扩展数据处理工具
- CompleteDataProcessingTools: 完整数据处理工具
- DataCleaner: 数据清洗工具
- DataTransformer: 数据转换工具
- DataAggregator: 数据聚合工具
- DataValidator: 数据验证工具
- AsyncDataPipeline: 异步数据处理管道
- MemoryOptimizedProcessor: 内存优化处理器
- BigDataProcessor: 大数据处理器
- AdvancedAnalytics: 高级分析工具
- DataProfiler: 数据画像工具
- DataExporter: 数据导出工具
- DataQualityAssessment: 数据质量评估工具
- DataMonitoring: 数据监控工具

版本: 1.0.0
作者: J2数据处理团队
创建日期: 2025-11-06
"""

try:
    from .DataProcessingTools import (
        # 基础类和异常
        ProcessingError,
        ValidationError,
        MemoryError,
        ProcessingStats,
        ValidationRule,
        BaseProcessor,
        
        # 核心处理器
        DataCleaner,
        DataTransformer,
        DataAggregator,
        DataValidator,
        
        # 管道和优化
        AsyncDataPipeline,
        MemoryOptimizedProcessor,
        BigDataProcessor,
        
        # 高级功能
        AdvancedAnalytics,
        DataProfiler,
        DataExporter,
        DataQualityAssessment,
        DataMonitoring,
        PerformanceMonitor,
        
        # 主工具类
        DataProcessingTools,
        ExtendedDataProcessingTools,
        CompleteDataProcessingTools,
        
        # 工具函数
        create_sample_data,
        benchmark_processor,
        validate_data_schema,
        create_data_pipeline,
        parallel_process_data
    )
except ImportError as e:
    import warnings
    warnings.warn(f"导入J2模块组件时出现警告: {e}", ImportWarning)

__version__ = "1.0.0"
__author__ = "J2数据处理团队"
__email__ = "j2-team@example.com"
__license__ = "MIT"

# 模块级别的配置
DEFAULT_CONFIG = {
    'chunk_size': 10000,
    'memory_limit_mb': 512,
    'use_multiprocessing': True,
    'max_workers': None,
    'log_level': 'INFO'
}

# 导出所有公共接口
__all__ = [
    # 异常类
    'ProcessingError',
    'ValidationError', 
    'MemoryError',
    
    # 数据类
    'ProcessingStats',
    'ValidationRule',
    
    # 基础类
    'BaseProcessor',
    
    # 处理器类
    'DataCleaner',
    'DataTransformer', 
    'DataAggregator',
    'DataValidator',
    'MemoryOptimizedProcessor',
    'BigDataProcessor',
    'AdvancedAnalytics',
    'DataProfiler',
    'DataExporter',
    'DataQualityAssessment',
    'DataMonitoring',
    'PerformanceMonitor',
    
    # 管道类
    'AsyncDataPipeline',
    
    # 主工具类
    'DataProcessingTools',
    'ExtendedDataProcessingTools',
    'CompleteDataProcessingTools',
    
    # 工具函数
    'create_sample_data',
    'benchmark_processor',
    'validate_data_schema',
    'create_data_pipeline',
    'parallel_process_data',
    
    # 模块信息
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    'DEFAULT_CONFIG'
]

# 模块初始化日志
import logging

logger = logging.getLogger(__name__)
logger.info(f"J2数据处理工具模块已加载，版本: {__version__}")

# 设置默认日志级别
logging.getLogger(__name__).setLevel(logging.INFO)