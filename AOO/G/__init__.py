#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G区域 - 执行控制系统模块

本模块实现了交易执行控制的核心功能，包括：

G1 - 智能决策引擎：智能决策系统，支持决策标准、风险评估、效果预测和决策解释
G2 - 动态适配器：环境动态适配系统，包含多种检测器、策略和优化算法
G3 - 风险控制器：全面风险控制和管理系统，支持风险识别、评估、控制和报告
G4 - 交易执行器：智能交易执行系统，支持订单优化、成本控制和性能监控
G5 - 仓位管理器：智能仓位管理系统，支持仓位计算、风险评估和组合优化
G6 - 订单管理器：完整订单管理系统，支持订单创建、验证、路由和执行监控
G7 - 执行监控器：实时执行监控系统，支持性能评估、异常检测和优化建议
G8 - 决策解释器：决策解释和说明系统，支持决策追踪、报告和可视化
G9 - 执行状态聚合器：多模块执行状态融合系统，支持状态评估和智能预警

每个模块都提供完整的API接口，支持独立使用或组合应用。
"""

from . import G1, G2, G3, G4, G5, G6, G7, G8, G9

__version__ = "1.0.0"
__author__ = "G区域开发团队"

__all__ = [
    "G1",  # 智能决策引擎
    "G2",  # 动态适配器
    "G3",  # 风险控制器
    "G4",  # 交易执行器
    "G5",  # 仓位管理器
    "G6",  # 订单管理器
    "G7",  # 执行监控器
    "G8",  # 决策解释器
    "G9"   # 执行状态聚合器
]

# 模块文档和说明
module_descriptions = {
    "G1": "智能决策引擎 - Decision Making & Risk Assessment",
    "G2": "动态适配器 - Dynamic Adaptation & Optimization", 
    "G3": "风险控制器 - Risk Control & Management",
    "G4": "交易执行器 - Trading Execution & Order Management",
    "G5": "仓位管理器 - Position Management & Optimization",
    "G6": "订单管理器 - Order Management & Routing",
    "G7": "执行监控器 - Execution Monitoring & Analytics",
    "G8": "决策解释器 - Decision Explanation & Visualization",
    "G9": "执行状态聚合器 - State Aggregation & Fusion"
}

def get_module_info():
    """获取G区域模块信息"""
    return {
        "total_modules": 9,
        "module_descriptions": module_descriptions,
        "version": __version__,
        "author": __author__
    }