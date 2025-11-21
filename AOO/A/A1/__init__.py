"""
A1市场数据采集器包
"""

from .MarketDataCollector import (
    MarketDataCollector,
    ExchangeConfig,
    MarketData,
    DataValidator,
    DataCache,
    DataStorage,
    WebSocketManager
)

__version__ = "1.0.0"
__author__ = "A1 Team"

__all__ = [
    'MarketDataCollector',
    'ExchangeConfig', 
    'MarketData',
    'DataValidator',
    'DataCache',
    'DataStorage',
    'WebSocketManager'
]