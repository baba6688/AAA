#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y5存储优化器包

一个全面的存储性能优化工具，提供存储性能优化、空间优化、访问优化等功能。

主要功能:
- 存储性能优化（存储读写性能优化）
- 空间优化（存储空间优化和压缩）
- 访问优化（存储访问路径优化）
- 数据压缩（数据压缩和存储优化）
- 存储分析（存储使用情况分析）
- 优化建议（存储优化建议和方案）
- 优化监控（优化效果监控）
- 优化报告（优化结果报告）

版本: 1.0.0
作者: Y5 Team
许可证: MIT License
"""

__version__ = "1.0.0"
__author__ = "Y5 Team"
__email__ = "y5-team@example.com"
__license__ = "MIT License"

# 导入主要类和函数
from .StorageOptimizer import (
    StorageOptimizer,
    StorageMetrics,
    OptimizationReport
)

# 导入演示函数
from .StorageOptimizer import demo_storage_optimizer

# 包级别的公共接口
__all__ = [
    'StorageOptimizer',
    'StorageMetrics', 
    'OptimizationReport',
    'demo_storage_optimizer'
]

# 包信息
PACKAGE_INFO = {
    'name': 'Y5 Storage Optimizer',
    'version': __version__,
    'description': '全面的存储性能优化工具',
    'author': __author__,
    'license': __license__,
    'features': [
        '存储性能优化',
        '空间优化和压缩',
        '访问路径优化',
        '数据压缩',
        '存储分析',
        '优化建议',
        '实时监控',
        '详细报告'
    ]
}