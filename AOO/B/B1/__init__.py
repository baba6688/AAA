"""
B1市场感知引擎包
Market Perception Engine
"""

from .MarketPerceptionEngine import (
    MarketPhase,           # 市场阶段枚举
    RiskLevel,             # 风险级别枚举
    OpportunityType,       # 机会类型枚举
    MarketFeatures,        # 市场特征类
    MarketPerceptionResult, # 市场感知结果
    AlertConfig,           # 告警配置
    MarketPerceptionEngine # 市场感知引擎主类
)

__version__ = "1.0.0"
__author__ = "B1 Team"

__all__ = [
    'MarketPhase',
    'RiskLevel', 
    'OpportunityType',
    'MarketFeatures',
    'MarketPerceptionResult',
    'AlertConfig',
    'MarketPerceptionEngine'
]