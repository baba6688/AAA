# U8聚类算法库使用指南

## 概述

U8聚类算法库提供了完整的聚类分析工具，包含6种经典聚类算法、评估指标、可视化工具和参数优化功能。

## 快速开始

### 基本导入

```python
# 方法1: 导入所有类和函数
from U.U8 import *

# 方法2: 按需导入特定类
from U.U8 import ClusteringAlgorithmLibrary, KMeansClustering, DBSCANClustering

# 方法3: 导入库并使用
from U.U8 import ClusteringAlgorithmLibrary
library = ClusteringAlgorithmLibrary()
```

### 快速演示

```python
from U.U8 import quick_demo
quick_demo()  # 运行快速演示
```

## 核心类介绍

### 1. 数据类

#### ClusteringResult
聚类结果数据类，包含：
- `labels`: 聚类标签
- `centroids`: 聚类中心（如果适用）
- `n_clusters`: 聚类数量
- `algorithm`: 使用的算法名称
- `parameters`: 算法参数
- `metrics`: 评估指标
- `convergence_iterations`: 收敛迭代次数
- `execution_time`: 执行时间

#### EvaluationMetrics
聚类评估指标数据类，包含：
- `silhouette_score`: 轮廓系数
- `calinski_harabasz_score`: Calinski-Harabasz指数
- `davies_bouldin_score`: Davies-Bouldin指数
- `adjusted_rand_score`: 调整兰德指数
- `normalized_mutual_info_score`: 标准化互信息
- `inertia`: 惯性（K-Means特有）

### 2. 聚类算法

#### KMeansClustering
```python
from U.U8 import KMeansClustering

# 创建聚类器
kmeans = KMeansClustering(n_clusters=3, max_iter=300, random_state=42)

# 拟合并预测
result = kmeans.fit(X)
labels = kmeans.predict(X_new)
```

#### HierarchicalClustering
```python
from U.U8 import HierarchicalClustering

hierarchical = HierarchicalClustering(n_clusters=3, linkage='ward')
result = hierarchical.fit(X)
```

#### DBSCANClustering
```python
from U.U8 import DBSCANClustering

dbscan = DBSCANClustering(eps=0.5, min_samples=5)
result = dbscan.fit(X)
```

#### GaussianMixtureClustering
```python
from U.U8 import GaussianMixtureClustering

gmm = GaussianMixtureClustering(n_components=3, covariance_type='full')
result = gmm.fit(X)
labels = gmm.predict(X_new)
```

#### SpectralClusteringAlgorithm
```python
from U.U8 import SpectralClusteringAlgorithm

spectral = SpectralClusteringAlgorithm(n_clusters=3, affinity='rbf')
result = spectral.fit(X)
```

#### FuzzyCMeansClustering
```python
from U.U8 import FuzzyCMeansClustering

fuzzy_cmeans = FuzzyCMeansClustering(n_clusters=3, fuzziness=2.0)
result = fuzzy_cmeans.fit(X)
labels = fuzzy_cmeans.predict(X_new)
```

### 3. 工具类

#### ClusteringEvaluator
```python
from U.U8 import ClusteringEvaluator

# 评估聚类结果
metrics = ClusteringEvaluator.evaluate_clustering(X, labels, true_labels)

# 比较多个聚类结果
comparison_df = ClusteringEvaluator.compare_clusterings(results, X, true_labels)
```

#### ClusteringVisualizer
```python
from U.U8 import ClusteringVisualizer

visualizer = ClusteringVisualizer(figsize=(12, 8))

# 绘制聚类结果
visualizer.plot_clusters(X, result, title="聚类结果", save_path="clusters.png")

# 绘制树状图
visualizer.plot_dendrogram(X, method='ward', save_path="dendrogram.png")

# 绘制评估指标比较
visualizer.plot_evaluation_metrics(results, X, save_path="metrics.png")
```

#### ParameterOptimizer
```python
from U.U8 import ParameterOptimizer

optimizer = ParameterOptimizer('kmeans')

param_grid = {
    'n_clusters': [2, 3, 4, 5],
    'max_iter': [100, 200, 300],
    'tol': [1e-4, 1e-3]
}

best_params, best_score = optimizer.optimize_parameters(X, param_grid)
```

### 4. 主库类

#### ClusteringAlgorithmLibrary
```python
from U.U8 import ClusteringAlgorithmLibrary

# 初始化库
library = ClusteringAlgorithmLibrary()

# 运行聚类
result = library.cluster(X, 'kmeans', n_clusters=3)

# 比较多种算法
comparison_df = library.compare_algorithms(X, ['kmeans', 'hierarchical', 'dbscan'])

# 参数优化
best_params, best_result = library.optimize_parameters(X, 'kmeans', param_grid)

# 可视化结果
library.visualize_results(X, result, save_path="result.png")

# 评估聚类
metrics = library.evaluate_clustering(X, result.labels, true_labels)
```

## 完整使用示例

### 示例1: 基本聚类分析
```python
import numpy as np
from U.U8 import ClusteringAlgorithmLibrary, generate_sample_data

# 生成示例数据
X, true_labels = generate_sample_data(n_samples=300, n_features=2, n_clusters=3)

# 初始化库
library = ClusteringAlgorithmLibrary()

# 运行K-Means聚类
result = library.cluster(X, 'kmeans', n_clusters=3)

# 评估结果
metrics = library.evaluate_clustering(X, result.labels, true_labels)

print(f"轮廓系数: {metrics.silhouette_score:.3f}")
print(f"调整兰德指数: {metrics.adjusted_rand_score:.3f}")

# 可视化结果
library.visualize_results(X, result, save_path="kmeans_result.png")
```

### 示例2: 多算法比较
```python
from U.U8 import ClusteringAlgorithmLibrary, generate_sample_data

# 生成数据
X, true_labels = generate_sample_data(n_samples=200, n_features=4, n_clusters=3)

# 初始化库
library = ClusteringAlgorithmLibrary()

# 比较多种算法
algorithms = ['kmeans', 'hierarchical', 'dbscan', 'gmm', 'spectral', 'fuzzy_cmeans']
comparison_df = library.compare_algorithms(X, algorithms)

print("算法比较结果:")
print(comparison_df.round(3))

# 可视化比较结果
visualizer = ClusteringVisualizer()
visualizer.plot_evaluation_metrics(
    library.compare_algorithms(X, algorithms, 
                             {'dbscan': {'eps': 0.5, 'min_samples': 5}})._results,
    X
)
```

### 示例3: 参数优化
```python
from U.U8 import ClusteringAlgorithmLibrary, generate_sample_data

# 生成数据
X, _ = generate_sample_data(n_samples=150, n_features=2, n_clusters=4)

# 初始化库
library = ClusteringAlgorithmLibrary()

# 定义参数网格
param_grid = {
    'n_clusters': [2, 3, 4, 5],
    'max_iter': [100, 200],
    'tol': [1e-4, 1e-3]
}

# 优化K-Means参数
best_params, best_result = library.optimize_parameters(X, 'kmeans', param_grid)

print(f"最佳参数: {best_params}")
print(f"最佳轮廓系数: {best_result.metrics['silhouette_score']:.3f}")
```

### 示例4: 高级可视化
```python
import numpy as np
from U.U8 import ClusteringAlgorithmLibrary, ClusteringVisualizer

# 生成数据
np.random.seed(42)
X = np.random.randn(200, 2)
X[:50] += [3, 0]  # 第一个聚类
X[50:100] += [0, 3]  # 第二个聚类
X[100:150] += [3, 3]  # 第三个聚类

# 运行聚类
library = ClusteringAlgorithmLibrary()
result = library.cluster(X, 'kmeans', n_clusters=3)

# 创建可视化器
visualizer = ClusteringVisualizer(figsize=(15, 5))

# 绘制聚类结果
visualizer.plot_clusters(X, result, title="K-Means聚类结果", save_path="clusters.png")

# 绘制树状图
visualizer.plot_dendrogram(X, method='ward', save_path="dendrogram.png")

# 绘制评估指标比较
results = [library.cluster(X, algo, n_clusters=3) for algo in ['kmeans', 'hierarchical', 'gmm']]
visualizer.plot_evaluation_metrics(results, X, save_path="comparison.png")
```

## 工具函数

### generate_sample_data()
生成用于测试的样本数据：

```python
from U.U8 import generate_sample_data

X, y = generate_sample_data(
    n_samples=300,      # 样本数量
    n_features=2,       # 特征数量
    n_clusters=3,       # 聚类数量
    random_state=42     # 随机种子
)
```

### run_comprehensive_test()
运行完整的综合测试：

```python
from U.U8 import run_comprehensive_test

run_comprehensive_test()
```

## 注意事项

1. **数据预处理**: 建议在使用聚类算法前对数据进行标准化处理
2. **参数选择**: 不同算法需要不同的参数调优策略
3. **评估指标**: 使用多个指标综合评估聚类效果
4. **可视化**: 高维数据会自动使用PCA降维到2D进行可视化
5. **性能**: 大数据集建议使用适当的数据采样

## 故障排除

### 常见问题

1. **ImportError**: 确保所有依赖包已安装（numpy, sklearn, matplotlib, seaborn, pandas, scipy）
2. **MemoryError**: 大数据集可能需要分批处理
3. **ConvergenceWarning**: 增加迭代次数或调整收敛容忍度

### 依赖包安装
```bash
pip install numpy scikit-learn matplotlib seaborn pandas scipy
```

## 扩展和自定义

你可以继承`ClusteringAlgorithm`基类来创建自己的聚类算法：

```python
from U.U8 import ClusteringAlgorithm, ClusteringResult
import numpy as np

class MyCustomClustering(ClusteringAlgorithm):
    def __init__(self, custom_param=1.0):
        self.custom_param = custom_param
    
    def fit(self, X: np.ndarray, **kwargs) -> ClusteringResult:
        # 实现你的聚类逻辑
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # 实现预测逻辑
        pass
```

## 版本信息

- **版本**: 1.0.0
- **作者**: U8聚类算法库开发团队
- **日期**: 2025-11-05
- **许可**: MIT