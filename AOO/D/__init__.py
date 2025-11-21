"""
D区：自我认知模块
Self-Cognition Module

实现自我认知、元认知处理、能力边界评估、错误诊断、置信度评估、
性能监控、自我检查、适应性评估和自我意识状态聚合等功能。

D1 - 自我认知引擎
D2 - 元认知处理器
D3 - 能力边界评估器
D4 - 错误诊断器
D5 - 置信度评估器
D6 - 性能监控器
D7 - 状态自检器
D8 - 适应性评估器
D9 - 自我意识状态聚合器

版本: 1.0.0
作者: AI量化系统
"""

from . import D1, D2, D3, D4, D5, D6, D7, D8, D9

# 导入所有子模块的主要类
from D.D1 import SelfCognitionEngine
from D.D2 import MetaCognitionProcessor, CognitiveState, StrategyType, LoadLevel, CognitiveEvent, CognitiveStrategy, MetaCognitionMetrics, MetaCognitionKnowledgeBase, CognitiveLoadMonitor
from D.D3 import CapabilityBoundaryAssessor, CapabilityType, BoundaryType, CapabilityMetrics, BoundaryDefinition, CapabilityGap, DevelopmentPotential
from D.D4 import ErrorDiagnoser, ErrorLevel, ErrorCategory, ErrorPattern, ErrorInfo, ErrorAnalysis, ErrorPatternInfo
from D.D5 import ConfidenceAssessor, ConfidenceLevel, AlertType, ConfidenceMetrics, AlertInfo, ConfidenceModel, MultiDimensionalConfidenceCalculator, UncertaintyQuantifier, ConfidenceCalibrator, DynamicConfidenceAdjuster, ConfidenceAlertSystem, ConfidenceOptimizer
from D.D6 import PerformanceMonitor, PerformanceMetrics, PerformanceAlert, PerformanceReport, TrendAnalyzer, BottleneckDetector, OptimizationAdvisor
from D.D7 import SelfChecker, SystemHealthChecker, StateAnalyzer, ProblemDiagnostic, StateReporter, StateOptimizer, StateMonitor, StateHistoryManager
from D.D8 import AdaptabilityAssessor, AdaptabilityType, EnvironmentType, AdaptationStrategy, AdaptabilityMetrics, EnvironmentChange, AdaptationPerformance, AdaptationSuggestion, AdaptabilityModel
from D.D9 import SelfAwarenessStateAggregator, SelfAwarenessLevel, ConsciousnessConsistency, AwarenessPriority, AwarenessModule, SelfAwarenessState, AwarenessResult, SelfAwarenessReport, SelfAwarenessFusionEngine

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    # 模块引用
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
    
    # D1 自我认知引擎
    'SelfCognitionEngine',
    
    # D2 元认知处理器
    'MetaCognitionProcessor', 'CognitiveState', 'StrategyType', 'LoadLevel',
    'CognitiveEvent', 'CognitiveStrategy', 'MetaCognitionMetrics', 
    'MetaCognitionKnowledgeBase', 'CognitiveLoadMonitor',
    
    # D3 能力边界评估器
    'CapabilityBoundaryAssessor', 'CapabilityType', 'BoundaryType', 'CapabilityMetrics',
    'BoundaryDefinition', 'CapabilityGap', 'DevelopmentPotential',
    
    # D4 错误诊断器
    'ErrorDiagnoser', 'ErrorLevel', 'ErrorCategory', 'ErrorPattern', 'ErrorInfo',
    'ErrorAnalysis', 'ErrorPatternInfo',
    
    # D5 置信度评估器
    'ConfidenceAssessor', 'ConfidenceLevel', 'AlertType', 'ConfidenceMetrics',
    'AlertInfo', 'ConfidenceModel', 'MultiDimensionalConfidenceCalculator',
    'UncertaintyQuantifier', 'ConfidenceCalibrator', 'DynamicConfidenceAdjuster',
    'ConfidenceAlertSystem', 'ConfidenceOptimizer',
    
    # D6 性能监控器
    'PerformanceMonitor', 'PerformanceMetrics', 'PerformanceAlert', 'PerformanceReport',
    'TrendAnalyzer', 'BottleneckDetector', 'OptimizationAdvisor',
    
    # D7 状态自检器
    'SelfChecker', 'SystemHealthChecker', 'StateAnalyzer', 'ProblemDiagnostic', 
    'StateReporter', 'StateOptimizer', 'StateMonitor', 'StateHistoryManager',
    
    # D8 适应性评估器
    'AdaptabilityAssessor', 'AdaptabilityType', 'EnvironmentType',
    'AdaptationStrategy', 'AdaptabilityMetrics', 'EnvironmentChange',
    'AdaptationPerformance', 'AdaptationSuggestion', 'AdaptabilityModel',
    
    # D9 自我意识状态聚合器
    'SelfAwarenessStateAggregator', 'SelfAwarenessLevel', 'ConsciousnessConsistency',
    'AwarenessPriority', 'AwarenessModule', 'SelfAwarenessState',
    'AwarenessResult', 'SelfAwarenessReport', 'SelfAwarenessFusionEngine'
]