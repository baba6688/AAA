#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4交易所连接器包

多交易所连接管理系统，支持主流交易所的统一接入和管理
"""

from .ExchangeConnector import (
    ExchangeType,
    ExchangeStatus,
    APIKey,
    ExchangeConfig,
    HealthStatus,
    RateLimiter,
    ConnectionPool,
    ExchangeConnector,
    BinanceConnector,
    OKXConnector,
    HuobiConnector,
    GateConnector,
    ExchangeManager,
    DataAggregator
)

__version__ = "1.0.0"
__author__ = "A4 Exchange Connector Team"

__all__ = [
    "ExchangeType",
    "ExchangeStatus", 
    "APIKey",
    "ExchangeConfig",
    "HealthStatus",
    "RateLimiter",
    "ConnectionPool",
    "ExchangeConnector",
    "BinanceConnector",
    "OKXConnector", 
    "HuobiConnector",
    "GateConnector",
    "ExchangeManager",
    "DataAggregator"
]