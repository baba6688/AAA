"""
B2模式识别器包
Pattern Recognizer
"""

from .PatternRecognizer import (
    PatternResult,                  # 模式识别结果
    PatternSignal,                  # 模式信号
    PricePatternRecognizer,         # 价格模式识别器
    CandlestickPatternRecognizer,   # 蜡烛图模式识别器
    VolumePatternRecognizer,        # 成交量模式识别器
    DeepLearningPatternModel,       # 深度学习模式模型
    CrossMarketPatternAnalyzer,     # 跨市场模式分析器
    PatternValidator,               # 模式验证器
    PatternSignalGenerator,         # 模式信号生成器
    PatternMonitor,                 # 模式监控器
    PatternBacktester,              # 模式回测器
    PatternRecognizer               # 模式识别器主类
)

__version__ = "1.0.0"
__author__ = "B2 Team"

__all__ = [
    'PatternResult',
    'PatternSignal',
    'PricePatternRecognizer',
    'CandlestickPatternRecognizer', 
    'VolumePatternRecognizer',
    'DeepLearningPatternModel',
    'CrossMarketPatternAnalyzer',
    'PatternValidator',
    'PatternSignalGenerator',
    'PatternMonitor',
    'PatternBacktester',
    'PatternRecognizer'
]