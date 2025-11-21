"""
U8聚类算法库模块

这是一个完整的聚类算法库，提供了多种聚类算法的实现和评估工具。
包含K-Means、层次聚类、DBSCAN、GMM、谱聚类、模糊C均值等算法，
以及完整的评估指标、可视化工具和参数优化功能。

主要功能模块：
- 数据类：聚类结果和评估指标
- 聚类算法：6种经典聚类算法实现
- 评估工具：多种聚类效果评估指标
- 可视化工具：聚类结果可视化
- 参数优化：自动参数调优
- 主库类：统一接口调用所有功能

Author: U8聚类算法库开发团队
Date: 2025-11-05
Version: 1.0.0
"""

# 导入所有核心类
from .ClusteringAlgorithmLibrary import (
    # 数据类
    ClusteringResult,
    EvaluationMetrics,
    
    # 抽象基类
    ClusteringAlgorithm,
    
    # 聚类算法实现
    KMeansClustering,
    HierarchicalClustering,
    DBSCANClustering,
    GaussianMixtureClustering,
    SpectralClusteringAlgorithm,
    FuzzyCMeansClustering,
    
    # 工具类
    ClusteringEvaluator,
    ClusteringVisualizer,
    ParameterOptimizer,
    
    # 主库类
    ClusteringAlgorithmLibrary,
    
    # 工具函数
    generate_sample_data,
    run_comprehensive_test
)

# 模块元信息
__version__ = "1.0.0"
__author__ = "U8聚类算法库开发团队"
__email__ = "u8-team@example.com"
__license__ = "MIT"

# 定义公共导出接口
__all__ = [
    # 数据类
    'ClusteringResult',
    'EvaluationMetrics',
    
    # 抽象基类
    'ClusteringAlgorithm',
    
    # 聚类算法
    'KMeansClustering',
    'HierarchicalClustering',
    'DBSCANClustering',
    'GaussianMixtureClustering',
    'SpectralClusteringAlgorithm',
    'FuzzyCMeansClustering',
    
    # 工具类
    'ClusteringEvaluator',
    'ClusteringVisualizer',
    'ParameterOptimizer',
    
    # 主库类
    'ClusteringAlgorithmLibrary',
    
    # 工具函数
    'generate_sample_data',
    'run_comprehensive_test'
]

# 便捷使用示例
def quick_demo():
    """快速演示函数"""
    print("=== U8聚类算法库快速演示 ===")
    
    # 生成示例数据
    X, true_labels = generate_sample_data(n_samples=200, n_features=2, n_clusters=3)
    print(f"生成数据形状: {X.shape}")
    
    # 初始化库
    library = ClusteringAlgorithmLibrary()
    
    # 运行K-Means聚类
    result = library.cluster(X, 'kmeans', n_clusters=3)
    print(f"聚类算法: {result.algorithm}")
    print(f"聚类数量: {result.n_clusters}")
    print(f"轮廓系数: {result.metrics.get('silhouette_score', 'N/A'):.3f}")
    
    # 评估聚类效果
    metrics = library.evaluate_clustering(X, result.labels, true_labels)
    print(f"调整兰德指数: {metrics.adjusted_rand_score:.3f}")
    
    print("=== 演示完成 ===")

# 模块初始化信息
if __name__ == "__main__":
    quick_demo()