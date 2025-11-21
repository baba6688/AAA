"""
C区：核心推理组件
Core Reasoning Components

包含各种核心推理引擎和分析器，包括知识图谱构建、因果推理、
抽象学习、关联分析、规则引擎、推理链管理、概念映射、
知识验证和状态聚合等功能。

C1 - 知识图谱构建器
C2 - 因果推理引擎  
C3 - 抽象学习器
C4 - 关联分析器
C5 - 规则引擎
C6 - 推理链管理器
C7 - 概念映射器
C8 - 知识验证器
C9 - 知识状态聚合器

版本: 1.0.0
作者: AI量化系统
"""

from . import C1, C2, C3, C4, C5, C6, C7, C8, C9

# 导入所有子模块的主要类
from C.C1 import Entity, Relation, KnowledgeTriple, EntityExtractor, RelationExtractor, KnowledgeGraphBuilder
from C.C2 import CausalReasoningEngine, CausalEffect, CausalRelation, CausalAlgorithm
from C.C3 import AbstractLearner, Concept, ConceptRelation, ConceptVisualizer
from C.C4 import CorrelationAnalyzer
from C.C5 import (RuleEngine, RuleType, ConflictResolutionStrategy, Rule, RuleMetadata, RuleExecutionResult,
                 IfThenRule, DecisionTableRule, DecisionTreeRule, FuzzyRule, TemporalRule,
                 RuleConflictDetector, RuleOptimizer, RuleValidator, RuleVersionManager)
from C.C6 import (ReasoningChainManager, ReasoningType, NodeType, ConfidenceLevel,
                 ReasoningNode, ReasoningEdge, ReasoningPath, PerformanceMetrics)
from C.C7 import ConceptMapper, Concept, MappingResult
from C.C8 import KnowledgeValidator, ValidationResult, KnowledgeQualityReport, ValidationLevel, QualityScore
from C.C9 import (KnowledgeStateAggregator, KnowledgeSource, KnowledgeItem, KnowledgeState,
                 KnowledgeFusionEngine, KnowledgeStateEvaluator, KnowledgeCredibilityCalculator,
                 KnowledgePriorityRanker, KnowledgeStateHistory, KnowledgeStateReporter, KnowledgeStateAlerter)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    # 模块引用
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
    
    # C1 知识图谱构建器
    'Entity', 'Relation', 'KnowledgeTriple', 'EntityExtractor', 'RelationExtractor', 'KnowledgeGraphBuilder',
    
    # C2 因果推理引擎
    'CausalReasoningEngine', 'CausalEffect', 'CausalRelation', 'CausalAlgorithm',
    
    # C3 抽象学习器
    'AbstractLearner', 'Concept', 'ConceptRelation', 'ConceptVisualizer',
    
    # C4 关联分析器
    'CorrelationAnalyzer',
    
    # C5 规则引擎
    'RuleEngine', 'RuleType', 'ConflictResolutionStrategy', 'Rule', 'RuleMetadata', 'RuleExecutionResult',
    'IfThenRule', 'DecisionTableRule', 'DecisionTreeRule', 'FuzzyRule', 'TemporalRule',
    'RuleConflictDetector', 'RuleOptimizer', 'RuleValidator', 'RuleVersionManager',
    
    # C6 推理链管理器
    'ReasoningChainManager', 'ReasoningType', 'NodeType', 'ConfidenceLevel',
    'ReasoningNode', 'ReasoningEdge', 'ReasoningPath', 'PerformanceMetrics',
    
    # C7 概念映射器
    'ConceptMapper', 'MappingResult',
    
    # C8 知识验证器
    'KnowledgeValidator', 'ValidationResult', 'KnowledgeQualityReport', 'ValidationLevel', 'QualityScore',
    
    # C9 知识状态聚合器
    'KnowledgeStateAggregator', 'KnowledgeSource', 'KnowledgeItem', 'KnowledgeState',
    'KnowledgeFusionEngine', 'KnowledgeStateEvaluator', 'KnowledgeCredibilityCalculator',
    'KnowledgePriorityRanker', 'KnowledgeStateHistory', 'KnowledgeStateReporter', 'KnowledgeStateAlerter'
]