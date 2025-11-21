"""
F2策略学习器包

该包实现了一个完整的策略学习系统，包括：
1. 策略学习和改进算法（强化学习、进化算法、模仿学习等）
2. 策略性能分析和评估
3. 策略适应性和进化
4. 策略组合和融合
5. 策略效果跟踪和预测
6. 策略知识提取和管理
7. 策略学习策略优化
"""

from typing import Any
from .StrategyLearner import (
    StrategyLearner,
    StrategyType,
    LearningPhase,
    BaseStrategy,
    StrategyPerformance,
    LearningContext,
    StrategyState,
    ReinforcementLearning,
    EvolutionAlgorithm,
    ImitationLearning,
    PerformanceAnalyzer,
    StrategyCombiner,
    KnowledgeExtractor,
    StrategyOptimizer
)

from .reinforcement_learning import (
    QNetwork,
    PolicyNetwork,
    ValueNetwork,
    DQNLearner,
    PolicyGradientLearner,
    ActorCriticLearner
)

from .evolution_algorithm import (
    Individual,
    GeneticAlgorithm,
    EvolutionStrategies,
    DifferentialEvolution
)

from .imitation_learning import (
    ExpertTrajectory,
    BehavioralCloning,
    TrajectoryCloning
)

from .performance_analyzer import (
    PerformanceMetrics,
    AdvancedPerformanceAnalyzer
)

from .strategy_combiner import (
    CombinationResult,
    PortfolioConstraints,
    BaseCombiner,
    WeightedAverageCombiner,
    VotingCombiner,
    StackingCombiner,
    AdaptiveCombiner,
    RiskParityCombiner
)

from .knowledge_extractor import (
    KnowledgePattern,
    KnowledgeRule,
    KnowledgeInsight,
    KnowledgeNode,
    KnowledgeEdge,
    PatternRecognizer,
    RuleExtractor,
    InsightGenerator,
    KnowledgeGraphBuilder,
    KnowledgeBase
)

from .strategy_optimizer import (
    OptimizationResult,
    OptimizationConfig,
    BaseOptimizer,
    BayesianOptimizer,
    GridSearchOptimizer,
    EvolutionOptimizer,
    AdaptiveOptimizer
)

__version__ = "1.0.0"
__author__ = "F2 Strategy Learning System"

# 包级别配置
__all__ = [
    # 核心类
    'StrategyLearner',
    'StrategyType',
    'LearningPhase',
    'BaseStrategy',
    'StrategyPerformance',
    'LearningContext',
    'StrategyState',
    'ReinforcementLearning',
    'EvolutionAlgorithm',
    'ImitationLearning',
    'PerformanceAnalyzer',
    'StrategyCombiner',
    'KnowledgeExtractor',
    'StrategyOptimizer',
    
    # 强化学习
    'QNetwork',
    'PolicyNetwork',
    'ValueNetwork',
    'DQNLearner',
    'PolicyGradientLearner',
    'ActorCriticLearner',
    
    # 进化算法
    'Individual',
    'GeneticAlgorithm',
    'EvolutionStrategies',
    'DifferentialEvolution',
    
    # 模仿学习
    'ExpertTrajectory',
    'BehavioralCloning',
    'TrajectoryCloning',
    
    # 性能分析
    'PerformanceMetrics',
    'AdvancedPerformanceAnalyzer',
    
    # 策略组合
    'CombinationResult',
    'PortfolioConstraints',
    'BaseCombiner',
    'WeightedAverageCombiner',
    'VotingCombiner',
    'StackingCombiner',
    'AdaptiveCombiner',
    'RiskParityCombiner',
    
    # 知识提取
    'KnowledgePattern',
    'KnowledgeRule',
    'KnowledgeInsight',
    'KnowledgeNode',
    'KnowledgeEdge',
    'PatternRecognizer',
    'RuleExtractor',
    'InsightGenerator',
    'KnowledgeGraphBuilder',
    'KnowledgeBase',
    
    # 策略优化
    'OptimizationResult',
    'OptimizationConfig',
    'BaseOptimizer',
    'BayesianOptimizer',
    'GridSearchOptimizer',
    'EvolutionOptimizer',
    'AdaptiveOptimizer'
]

# 便捷函数
def create_strategy_learner(config_path: str = None, **kwargs) -> StrategyLearner:
    """
    创建策略学习器实例的便捷函数
    
    Args:
        config_path: 配置文件路径
        **kwargs: 配置参数
        
    Returns:
        StrategyLearner实例
    """
    return StrategyLearner(config_path=config_path, **kwargs)

def quick_start(data: Any, strategy_type: StrategyType = StrategyType.HYBRID) -> StrategyLearner:
    """
    快速启动函数
    
    Args:
        data: 训练数据
        strategy_type: 策略类型
        
    Returns:
        配置好的StrategyLearner实例
    """
    learner = StrategyLearner()
    learner.setup_default_strategies(strategy_type)
    learner.load_training_data(data)
    return learner