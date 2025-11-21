#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G6订单管理器模块
提供完整的订单管理功能，包括订单创建、验证、路由、执行、监控等
"""

from .OrderManager import (
    OrderManager,
    Order,
    OrderStatus,
    OrderType,
    Priority,
    OrderValidationResult,
    OrderMetrics,
    OrderValidator,
    OrderRouter,
    OrderExecutionMonitor,
    OrderExceptionHandler,
    OrderAnalytics
)

__version__ = "1.0.0"
__author__ = "G6 Order Management System"

__all__ = [
    'OrderManager',
    'Order',
    'OrderStatus',
    'OrderType',
    'Priority',
    'OrderValidationResult',
    'OrderMetrics',
    'OrderValidator',
    'OrderRouter',
    'OrderExecutionMonitor',
    'OrderExceptionHandler',
    'OrderAnalytics'
]