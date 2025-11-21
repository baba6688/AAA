#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G4交易执行器模块

实现智能交易执行的核心功能，包括：
1. 订单类型、方向、状态、执行策略枚举
2. 交易信号、订单、执行结果数据类
3. 交易所连接器和模拟连接器
4. 订单优化器和成本优化器
5. 执行监控器和异常处理器
6. 性能评估器和交易执行器
"""

from .TradingExecutor import (
    # 枚举类
    OrderType,
    OrderSide,
    OrderStatus,
    ExecutionStrategy,
    # 核心数据类
    TradingSignal,
    Order,
    ExecutionResult,
    ExchangeInfo,
    # 连接器类
    ExchangeConnector,
    MockExchangeConnector,
    # 优化器和监控器
    OrderOptimizer,
    CostOptimizer,
    ExecutionMonitor,
    ExceptionHandler,
    PerformanceEvaluator,
    # 主执行器
    TradingExecutor
)

__version__ = "1.0.0"
__author__ = "G4 Team"

__all__ = [
    # 枚举类
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "ExecutionStrategy",
    # 核心数据类
    "TradingSignal",
    "Order",
    "ExecutionResult",
    "ExchangeInfo",
    # 连接器类
    "ExchangeConnector",
    "MockExchangeConnector",
    # 优化器和监控器
    "OrderOptimizer",
    "CostOptimizer",
    "ExecutionMonitor",
    "ExceptionHandler",
    "PerformanceEvaluator",
    # 主执行器
    "TradingExecutor"
]