"""
B6波动率分析器包
Volatility Analyzer
"""

from .VolatilityAnalyzer import (
    VolatilityModel,            # 波动率模型
    VolatilitySignal,           # 波动率信号
    BlackScholesCalculator,     # Black-Scholes期权定价计算器
    ImpliedVolatilityCalculator, # 隐含波动率计算器
    GARCHModel,                 # GARCH模型
    EWMA,                       # 指数加权移动平均
    HistoricalVolatility,       # 历史波动率
    VolatilitySmileAnalyzer,    # 波动率微笑分析器
    VolatilityArbitrage,        # 波动率套利
    VolatilityRiskManagement,   # 波动率风险管理
    VolatilityStrategy,         # 波动率策略
    VolatilityAnalyzer          # 波动率分析器主类
)

__version__ = "1.0.0"
__author__ = "B6 Team"

__all__ = [
    'VolatilityModel',
    'VolatilitySignal',
    'BlackScholesCalculator',
    'ImpliedVolatilityCalculator',
    'GARCHModel',
    'EWMA',
    'HistoricalVolatility',
    'VolatilitySmileAnalyzer',
    'VolatilityArbitrage',
    'VolatilityRiskManagement',
    'VolatilityStrategy',
    'VolatilityAnalyzer'
]