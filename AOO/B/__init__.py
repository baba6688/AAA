"""
B区市场认知分析包
Market Cognition Analysis Package

该包提供了完整的市场认知分析功能，包括：
- 市场感知引擎 (B1)
- 模式识别器 (B2) 
- 趋势分析器 (B3)
- 异常检测器 (B4)
- 市场结构分析器 (B5)
- 波动率分析器 (B6)
- 流动性分析器 (B7)
- 成交量分析器 (B8)
- 认知状态聚合器 (B9)
"""

# B1 - 市场感知引擎
from B.B1 import (
    MarketPhase,
    RiskLevel,
    OpportunityType,
    MarketFeatures,
    MarketPerceptionResult,
    AlertConfig,
    MarketPerceptionEngine
)

# B2 - 模式识别器
from B.B2 import (
    PatternResult,
    PatternSignal,
    PricePatternRecognizer,
    CandlestickPatternRecognizer,
    VolumePatternRecognizer,
    DeepLearningPatternModel,
    CrossMarketPatternAnalyzer,
    PatternValidator,
    PatternSignalGenerator,
    PatternMonitor,
    PatternBacktester,
    PatternRecognizer
)

# B3 - 趋势分析器
from B.B3 import (
    TrendDirection,
    TrendStrength,
    SignalType,
    TrendData,
    TrendAnalysis,
    TradingSignal,
    TrendAlert,
    TrendAnalyzer
)

# B4 - 异常检测器
from B.B4 import (
    AnomalyType,
    SeverityLevel,
    AlertStatus,
    AnomalyEvent,
    AnomalyDatabase,
    StatisticalAnomalyDetector,
    MachineLearningAnomalyDetector,
    PriceAnomalyDetector,
    VolumeAnomalyDetector,
    TechnicalIndicatorAnomalyDetector,
    CrossAssetAnomalyDetector,
    MarketStructureAnomalyDetector,
    AnomalyAlertSystem,
    AnomalyDetector
)

# B5 - 市场结构分析器
from B.B5 import (
    OrderBookLevel,
    TradeEvent,
    MarketStructureMetrics,
    OrderBookAnalyzer,
    TradeAnalyzer,
    MarketEfficiencyAnalyzer,
    MarketManipulationDetector,
    MarketStructureAnalyzer
)

# B6 - 波动率分析器
from B.B6 import (
    VolatilityModel,
    VolatilitySignal,
    BlackScholesCalculator,
    ImpliedVolatilityCalculator,
    GARCHModel,
    EWMA,
    HistoricalVolatility,
    VolatilitySmileAnalyzer,
    VolatilityArbitrage,
    VolatilityRiskManagement,
    VolatilityStrategy,
    VolatilityAnalyzer
)

# B7 - 流动性分析器
from B.B7 import (
    LiquidityRiskLevel,
    LiquidityCrisisLevel,
    LiquidityMetrics,
    LiquidityRiskAssessment,
    LiquidityAnalyzer
)

# B8 - 成交量分析器
from B.B8 import (
    VolumePattern,
    VolumeAnomaly,
    DivergenceType,
    VolumeSignal,
    VolumeAnalysisResult,
    VolumeAnalyzer
)

# B9 - 认知状态聚合器
from B.B9 import (
    CognitiveStateLevel,
    ConsistencyStatus,
    PerceptionResult,
    CognitiveState,
    CognitiveReport,
    CognitionStateAggregator
)

# 包信息
__version__ = "1.0.0"
__author__ = "B区开发团队"
__description__ = "市场认知分析工具包"

# 导出所有模块
__all__ = [
    # B1模块
    'MarketPhase',
    'RiskLevel',
    'OpportunityType',
    'MarketFeatures',
    'MarketPerceptionResult',
    'AlertConfig',
    'MarketPerceptionEngine',
    
    # B2模块
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
    'PatternRecognizer',
    
    # B3模块
    'TrendDirection',
    'TrendStrength',
    'SignalType',
    'TrendData',
    'TrendAnalysis',
    'TradingSignal',
    'TrendAlert',
    'TrendAnalyzer',
    
    # B4模块
    'AnomalyType',
    'SeverityLevel',
    'AlertStatus',
    'AnomalyEvent',
    'AnomalyDatabase',
    'StatisticalAnomalyDetector',
    'MachineLearningAnomalyDetector',
    'PriceAnomalyDetector',
    'VolumeAnomalyDetector',
    'TechnicalIndicatorAnomalyDetector',
    'CrossAssetAnomalyDetector',
    'MarketStructureAnomalyDetector',
    'AnomalyAlertSystem',
    'AnomalyDetector',
    
    # B5模块
    'OrderBookLevel',
    'TradeEvent',
    'MarketStructureMetrics',
    'OrderBookAnalyzer',
    'TradeAnalyzer',
    'MarketEfficiencyAnalyzer',
    'MarketManipulationDetector',
    'MarketStructureAnalyzer',
    
    # B6模块
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
    'VolatilityAnalyzer',
    
    # B7模块
    'LiquidityRiskLevel',
    'LiquidityCrisisLevel',
    'LiquidityMetrics',
    'LiquidityRiskAssessment',
    'LiquidityAnalyzer',
    
    # B8模块
    'VolumePattern',
    'VolumeAnomaly',
    'DivergenceType',
    'VolumeSignal',
    'VolumeAnalysisResult',
    'VolumeAnalyzer',
    
    # B9模块
    'CognitiveStateLevel',
    'ConsistencyStatus',
    'PerceptionResult',
    'CognitiveState',
    'CognitiveReport',
    'CognitionStateAggregator'
]