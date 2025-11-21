"""
U区 - 用户界面模块：算法库组件
User Interface Module - Algorithm Library Components

模块描述：
U区为用户界面模块提供全面的算法库支持，包含机器学习、深度学习、强化学习等
9个子模块，总计111个类和15,490行代码，提供完整的算法实现框架。

功能分类：
- U1: 机器学习算法库 (MLAlgorithmLibrary) - 14类经典ML算法
- U2: 深度学习算法库 (DLAlgorithmLibrary) - 15类深度学习模型
- U3: 强化学习算法库 (RLAlgorithmLibrary) - 20类强化学习算法
- U4: 优化算法库 (OptimizationAlgorithmLibrary) - 13类优化方法
- U5: 统计算法库 (StatisticalAlgorithmLibrary) - 1类统计算法
- U6: 时间序列算法库 (TimeSeriesAlgorithmLibrary) - 11类时序算法
- U7: 图算法库 (GraphAlgorithmLibrary) - 15类图算法
- U8: 聚类算法库 (ClusteringAlgorithmLibrary) - 13类聚类算法
- U9: 算法状态聚合器 (AlgorithmStateAggregator) - 9类状态管理

版本：v1.0.0
最后更新：2025-11-14
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "MiniMax Agent"

# 主模块导入
from .U1.MLAlgorithmLibrary import (
    ModelResult,
    BaseMLAlgorithm,
    LinearRegressionAlgorithm,
    LogisticRegressionAlgorithm,
    SVMAlgorithm,
    RandomForestAlgorithm,
    GradientBoostingAlgorithm,
    NaiveBayesAlgorithm,
    KNNAlgorithm,
    DecisionTreeAlgorithm,
    EnsembleLearning,
    FeatureSelection,
    HyperparameterOptimizer,
    MLAlgorithmLibrary
)

from .U2.DLAlgorithmLibrary import (
    BaseNeuralNetwork,
    ConvolutionalNeuralNetwork,
    RecurrentNeuralNetwork,
    MultiHeadAttention,
    TransformerBlock,
    Transformer,
    Autoencoder,
    Generator,
    Discriminator,
    GAN,
    VariationalAutoencoder,
    ReplayBuffer,
    DeepQNetwork,
    DQNAgent,
    ModelPruner,
    ModelQuantizer,
    ModelTrainer,
    DLAlgorithmLibrary
)

from .U3.RLAlgorithmLibrary import (
    RLEnvironment,
    GymEnvironment,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    QNetwork,
    DuelingQNetwork,
    PolicyNetwork,
    ContinuousPolicyNetwork,
    ValueNetwork,
    ActorNetwork,
    CriticNetwork,
    QLearning,
    DQN,
    PolicyGradient,
    ActorCritic,
    PPO,
    TRPO,
    DDPG,
    MultiAgentDQN,
    MultiAgentEnvironment,
    RLAlgorithmLibrary
)

from .U4.OptimizationAlgorithmLibrary import (
    OptimizationProblem,
    BaseOptimizer,
    GradientDescentOptimizer,
    GeneticAlgorithmOptimizer,
    ParticleSwarmOptimizer,
    SimulatedAnnealingOptimizer,
    AntColonyOptimizer,
    DifferentialEvolutionOptimizer,
    BayesianOptimizer,
    MultiObjectiveProblem,
    NSGA2Optimizer,
    HyperparameterTuner,
    OptimizationAlgorithmLibrary
)

from .U5.StatisticalAlgorithmLibrary import (
    StatisticalAlgorithmLibrary
)

from .U6.TimeSeriesAlgorithmLibrary import (
    ForecastResult,
    TimeSeriesModel,
    ARIMAModel,
    SeasonalDecomposition,
    ExponentialSmoothing,
    KalmanFilter,
    HiddenMarkovModel,
    FourierWaveletAnalysis,
    AnomalyDetector,
    ForecastMetrics,
    TimeSeriesAlgorithmLibrary
)

from .U7.GraphAlgorithmLibrary import (
    Edge,
    Node,
    Graph,
    TraversalAlgorithms,
    ShortestPathAlgorithms,
    MinimumSpanningTree,
    CommunityDetection,
    GraphNeuralNetwork,
    GraphSimilarity,
    GraphEmbedding,
    DynamicGraph,
    GraphVisualization,
    GraphAlgorithmLibrary
)

from .U8.ClusteringAlgorithmLibrary import (
    ClusteringResult,
    EvaluationMetrics,
    ClusteringAlgorithm,
    KMeansClustering,
    HierarchicalClustering,
    DBSCANClustering,
    GaussianMixtureClustering,
    SpectralClusteringAlgorithm,
    FuzzyCMeansClustering,
    ClusteringEvaluator,
    ClusteringVisualizer,
    ParameterOptimizer,
    ClusteringAlgorithmLibrary
)

from .U9.AlgorithmStateAggregator import (
    AlgorithmStatus,
    HealthStatus,
    AlgorithmMetrics,
    AlgorithmConfig,
    AlgorithmUsageStats,
    AlgorithmEffectiveness,
    AlgorithmStateAggregator
)

# 导出配置
__all__ = [
    # U1 - 机器学习算法库 (14类)
    "ModelResult", "BaseMLAlgorithm", "LinearRegressionAlgorithm",
    "LogisticRegressionAlgorithm", "SVMAlgorithm", "RandomForestAlgorithm",
    "GradientBoostingAlgorithm", "NaiveBayesAlgorithm", "KNNAlgorithm",
    "DecisionTreeAlgorithm", "EnsembleLearning", "FeatureSelection",
    "HyperparameterOptimizer", "MLAlgorithmLibrary",
    
    # U2 - 深度学习算法库 (15类)
    "BaseNeuralNetwork", "ConvolutionalNeuralNetwork", "RecurrentNeuralNetwork",
    "MultiHeadAttention", "TransformerBlock", "Transformer", "Autoencoder",
    "Generator", "Discriminator", "GAN", "VariationalAutoencoder",
    "ReplayBuffer", "DeepQNetwork", "DQNAgent", "ModelPruner",
    "ModelQuantizer", "ModelTrainer", "DLAlgorithmLibrary",
    
    # U3 - 强化学习算法库 (20类)
    "RLEnvironment", "GymEnvironment", "ReplayBuffer", "PrioritizedReplayBuffer",
    "QNetwork", "DuelingQNetwork", "PolicyNetwork", "ContinuousPolicyNetwork",
    "ValueNetwork", "ActorNetwork", "CriticNetwork", "QLearning", "DQN",
    "PolicyGradient", "ActorCritic", "PPO", "TRPO", "DDPG",
    "MultiAgentDQN", "MultiAgentEnvironment", "RLAlgorithmLibrary",
    
    # U4 - 优化算法库 (13类)
    "OptimizationProblem", "BaseOptimizer", "GradientDescentOptimizer",
    "GeneticAlgorithmOptimizer", "ParticleSwarmOptimizer",
    "SimulatedAnnealingOptimizer", "AntColonyOptimizer",
    "DifferentialEvolutionOptimizer", "BayesianOptimizer",
    "MultiObjectiveProblem", "NSGA2Optimizer", "HyperparameterTuner",
    "OptimizationAlgorithmLibrary",
    
    # U5 - 统计算法库 (1类)
    "StatisticalAlgorithmLibrary",
    
    # U6 - 时间序列算法库 (11类)
    "ForecastResult", "TimeSeriesModel", "ARIMAModel",
    "SeasonalDecomposition", "ExponentialSmoothing", "KalmanFilter",
    "HiddenMarkovModel", "FourierWaveletAnalysis", "AnomalyDetector",
    "ForecastMetrics", "TimeSeriesAlgorithmLibrary",
    
    # U7 - 图算法库 (15类)
    "Edge", "Node", "Graph", "TraversalAlgorithms",
    "ShortestPathAlgorithms", "MinimumSpanningTree", "CommunityDetection",
    "GraphNeuralNetwork", "GraphSimilarity", "GraphEmbedding",
    "DynamicGraph", "GraphVisualization", "GraphAlgorithmLibrary",
    
    # U8 - 聚类算法库 (13类)
    "ClusteringResult", "EvaluationMetrics", "ClusteringAlgorithm",
    "KMeansClustering", "HierarchicalClustering", "DBSCANClustering",
    "GaussianMixtureClustering", "SpectralClusteringAlgorithm",
    "FuzzyCMeansClustering", "ClusteringEvaluator", "ClusteringVisualizer",
    "ParameterOptimizer", "ClusteringAlgorithmLibrary",
    
    # U9 - 算法状态聚合器 (9类)
    "AlgorithmStatus", "HealthStatus", "AlgorithmMetrics",
    "AlgorithmConfig", "AlgorithmUsageStats", "AlgorithmEffectiveness",
    "AlgorithmStateAggregator"
]

# 模块信息
MODULE_INFO = {
    "name": "User Interface Module - Algorithm Library",
    "version": "1.0.0",
    "total_classes": 111,
    "total_lines": 15490,
    "sub_modules": {
        "U1": {"name": "Machine Learning Algorithm Library", "classes": 14},
        "U2": {"name": "Deep Learning Algorithm Library", "classes": 15},
        "U3": {"name": "Reinforcement Learning Algorithm Library", "classes": 20},
        "U4": {"name": "Optimization Algorithm Library", "classes": 13},
        "U5": {"name": "Statistical Algorithm Library", "classes": 1},
        "U6": {"name": "Time Series Algorithm Library", "classes": 11},
        "U7": {"name": "Graph Algorithm Library", "classes": 15},
        "U8": {"name": "Clustering Algorithm Library", "classes": 13},
        "U9": {"name": "Algorithm State Aggregator", "classes": 9}
    }
}

print(f"U区 - 用户界面模块已初始化，算法库总数: {MODULE_INFO['total_classes']} 类")