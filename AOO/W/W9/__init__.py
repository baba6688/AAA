#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W9网络状态聚合器包

这是一个综合性的网络状态监控系统，提供：

核心组件：
- NetworkModuleStatus: 网络模块状态数据结构
- NetworkAlert: 网络预警数据结构
- StatusCollector: 状态收集器 - 从各个网络模块收集状态信息
- DataAggregator: 数据聚合器 - 聚合多个网络模块的结果
- StatusAnalyzer: 状态分析器 - 分析网络状态和趋势
- AlertManager: 预警管理器 - 网络异常时预警
- HistoryManager: 历史记录管理器 - 保存历史网络状态
- ReportGenerator: 报告生成器 - 生成综合网络状态报告
- Dashboard: 仪表板 - 提供可视化的网络状态仪表板
- NetworkStatusAggregator: 网络状态聚合器主类

功能特性：
- 多协议状态监控 (Ping, HTTP, TCP, System)
- 实时状态聚合与分析
- 智能预警机制
- 历史数据存储
- 可视化仪表板
- 综合报告生成

作者: W9网络状态聚合器团队
版本: 1.0.0
创建时间: 2025-11-06
更新时间: 2025-11-14
"""

# 导入所有核心类
from .NetworkStatusAggregator import (
    NetworkModuleStatus,
    NetworkAlert,
    StatusCollector,
    DataAggregator,
    StatusAnalyzer,
    AlertManager,
    HistoryManager,
    ReportGenerator,
    Dashboard,
    NetworkStatusAggregator
)

# 导入辅助函数
from .NetworkStatusAggregator import create_example_aggregator

# 包版本信息
__version__ = "1.0.0"
__author__ = "W9网络状态聚合器团队"
__email__ = "support@w9-network.com"
__license__ = "MIT"

# 导出所有公共接口
__all__ = [
    # 数据结构
    "NetworkModuleStatus",
    "NetworkAlert",
    
    # 核心组件
    "StatusCollector",
    "DataAggregator", 
    "StatusAnalyzer",
    "AlertManager",
    "HistoryManager",
    "ReportGenerator",
    "Dashboard",
    
    # 主聚合器类
    "NetworkStatusAggregator",
    
    # 辅助函数
    "create_example_aggregator"
]

# 模块初始化信息
__init_msg__ = f"""
W9网络状态聚合器 v{__version__}
作者: {__author__}
许可证: {__license__}

可用组件: {len(__all__)} 个
- 状态监控: StatusCollector, DataAggregator, StatusAnalyzer
- 预警系统: AlertManager, NetworkAlert  
- 数据管理: HistoryManager, NetworkModuleStatus
- 可视化: Dashboard, ReportGenerator
- 集成接口: NetworkStatusAggregator

使用示例:
    from W.W9 import NetworkStatusAggregator
    
    aggregator = NetworkStatusAggregator()
    aggregator.register_module("google_dns", "ping", {{"target": "8.8.8.8"}})
    status = aggregator.get_current_status()
    print(f"网络状态: {{status['overall_status']}}")
    
# 示例代码已在上方文档字符串中展示
"""

# 打印初始化信息（可选）
try:
    print(__init_msg__)
except:
    pass  # 在某些环境下可能无法输出