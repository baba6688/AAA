#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I3 数据接口适配器模块

提供统一的数据访问接口，支持多种数据源、格式转换、
数据验证、批量处理、增量同步、缓存、重试、血缘追踪和性能监控。

Author: AI Assistant
Date: 2025-11-05
Version: 1.0.0
"""

from .DataInterfaceAdapter import (
    DataInterfaceAdapter,
    DataRecord,
    DataSourceType,
    DataFormat,
    DataQuality,
    PerformanceMetrics,
    DatabaseAdapter,
    APIAdapter,
    FileAdapter,
    DataValidator,
    DataConverter,
    DataCache,
    RetryManager,
    DataLineageTracker,
    PerformanceMonitor,
    TestDataInterfaceAdapter
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "DataInterfaceAdapter",
    "DataRecord", 
    "DataSourceType",
    "DataFormat",
    "DataQuality",
    "PerformanceMetrics",
    "DatabaseAdapter",
    "APIAdapter", 
    "FileAdapter",
    "DataValidator",
    "DataConverter",
    "DataCache",
    "RetryManager",
    "DataLineageTracker",
    "PerformanceMonitor",
    "TestDataInterfaceAdapter"
]