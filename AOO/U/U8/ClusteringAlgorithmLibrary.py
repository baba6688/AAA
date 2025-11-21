"""
U8聚类算法库

这是一个完整的聚类算法库，提供了多种聚类算法的实现和评估工具。
包含K-Means、层次聚类、DBSCAN、GMM、谱聚类、模糊C均值等算法，
以及完整的评估指标、可视化工具和参数优化功能。

Author: U8聚类算法库开发团队
Date: 2025-11-05
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import pandas as pd


@dataclass
class ClusteringResult:
    """聚类结果数据类"""
    labels: np.ndarray  # 聚类标签
    centroids: Optional[np.ndarray]  # 聚类中心（如果适用）
    n_clusters: int  # 聚类数量
    algorithm: str  # 使用的算法名称
    parameters: Dict[str, Any]  # 算法参数
    metrics: Dict[str, float]  # 评估指标
    convergence_iterations: int  # 收敛迭代次数
    execution_time: float  # 执行时间（秒）


@dataclass
class EvaluationMetrics:
    """聚类评估指标数据类"""
    silhouette_score: float  # 轮廓系数
    calinski_harabasz_score: float  # Calinski-Harabasz指数
    davies_bouldin_score: float  # Davies-Bouldin指数
    adjusted_rand_score: Optional[float]  # 调整兰德指数（如果有真实标签）
    normalized_mutual_info_score: Optional[float]  # 标准化互信息（如果有真实标签）
    inertia: Optional[float]  # 惯性（K-Means特有）


class ClusteringAlgorithm(ABC):
    """聚类算法抽象基类"""
    
    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> ClusteringResult:
        """
        拟合聚类模型
        
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
            **kwargs: 其他参数
            
        Returns:
            ClusteringResult: 聚类结果
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据点的聚类标签
        
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
            
        Returns:
            np.ndarray: 聚类标签
        """
        pass


class KMeansClustering(ClusteringAlgorithm):
    """K-Means聚类算法实现"""
    
    def __init__(self, n_clusters: int = 8, max_iter: int = 300, 
                 tol: float = 1e-4, random_state: Optional[int] = None):
        """
        初始化K-Means聚类器
        
        Args:
            n_clusters: 聚类数量
            max_iter: 最大迭代次数
            tol: 收敛容忍度
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model = None
        self.centroids = None
    
    def fit(self, X: np.ndarray, **kwargs) -> ClusteringResult:
        """拟合K-Means模型"""
        import time
        start_time = time.time()
        
        # 初始化模型
        self.model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            **kwargs
        )
        
        # 拟合模型
        labels = self.model.fit_predict(X)
        self.centroids = self.model.cluster_centers_
        
        execution_time = time.time() - start_time
        
        # 计算评估指标
        metrics = self._calculate_metrics(X, labels)
        
        return ClusteringResult(
            labels=labels,
            centroids=self.centroids,
            n_clusters=self.n_clusters,
            algorithm="K-Means",
            parameters={
                "n_clusters": self.n_clusters,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "random_state": self.random_state
            },
            metrics=metrics,
            convergence_iterations=self.model.n_iter_,
            execution_time=execution_time
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新数据点的聚类标签"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)
    
    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """计算聚类评估指标"""
        metrics = {}
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = -1
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = -1
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = -1
        
        metrics['inertia'] = self.model.inertia_
        
        return metrics


class HierarchicalClustering(ClusteringAlgorithm):
    """层次聚类算法实现"""
    
    def __init__(self, n_clusters: int = 8, linkage: str = 'ward'):
        """
        初始化层次聚类器
        
        Args:
            n_clusters: 聚类数量
            linkage: 连接方法 ('ward', 'complete', 'average', 'single')
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = None
    
    def fit(self, X: np.ndarray, **kwargs) -> ClusteringResult:
        """拟合层次聚类模型"""
        import time
        start_time = time.time()
        
        # 初始化模型
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            **kwargs
        )
        
        # 拟合模型
        labels = self.model.fit_predict(X)
        
        execution_time = time.time() - start_time
        
        # 计算评估指标
        metrics = self._calculate_metrics(X, labels)
        
        return ClusteringResult(
            labels=labels,
            centroids=None,  # 层次聚类没有明确的中心点
            n_clusters=self.n_clusters,
            algorithm="Hierarchical",
            parameters={
                "n_clusters": self.n_clusters,
                "linkage": self.linkage
            },
            metrics=metrics,
            convergence_iterations=1,  # 层次聚类不需要迭代
            execution_time=execution_time
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """层次聚类不支持预测新数据点"""
        raise NotImplementedError("层次聚类不支持预测新数据点")
    
    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """计算聚类评估指标"""
        metrics = {}
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = -1
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = -1
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = -1
        
        return metrics


class DBSCANClustering(ClusteringAlgorithm):
    """DBSCAN密度聚类算法实现"""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        初始化DBSCAN聚类器
        
        Args:
            eps: 邻域半径
            min_samples: 核心点的最小样本数
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
    
    def fit(self, X: np.ndarray, **kwargs) -> ClusteringResult:
        """拟合DBSCAN模型"""
        import time
        start_time = time.time()
        
        # 初始化模型
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            **kwargs
        )
        
        # 拟合模型
        labels = self.model.fit_predict(X)
        
        execution_time = time.time() - start_time
        
        # 计算实际聚类数量（排除噪声点）
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        
        # 计算评估指标（只考虑非噪声点）
        if n_clusters > 1:
            mask = labels != -1
            if np.sum(mask) > 1:
                X_filtered = X[mask]
                labels_filtered = labels[mask]
                metrics = self._calculate_metrics(X_filtered, labels_filtered)
            else:
                metrics = {}
        else:
            metrics = {}
        
        return ClusteringResult(
            labels=labels,
            centroids=None,  # DBSCAN没有明确的中心点
            n_clusters=n_clusters,
            algorithm="DBSCAN",
            parameters={
                "eps": self.eps,
                "min_samples": self.min_samples
            },
            metrics=metrics,
            convergence_iterations=1,  # DBSCAN不需要迭代
            execution_time=execution_time
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN不支持预测新数据点"""
        raise NotImplementedError("DBSCAN不支持预测新数据点")
    
    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """计算聚类评估指标"""
        metrics = {}
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = -1
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = -1
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = -1
        
        return metrics


class GaussianMixtureClustering(ClusteringAlgorithm):
    """高斯混合模型聚类算法实现"""
    
    def __init__(self, n_components: int = 8, covariance_type: str = 'full',
                 max_iter: int = 100, random_state: Optional[int] = None):
        """
        初始化高斯混合模型聚类器
        
        Args:
            n_components: 高斯分布数量
            covariance_type: 协方差类型 ('full', 'tied', 'diag', 'spherical')
            max_iter: 最大迭代次数
            random_state: 随机种子
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
    
    def fit(self, X: np.ndarray, **kwargs) -> ClusteringResult:
        """拟合高斯混合模型"""
        import time
        start_time = time.time()
        
        # 初始化模型
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **kwargs
        )
        
        # 拟合模型
        self.model.fit(X)
        labels = self.model.predict(X)
        
        execution_time = time.time() - start_time
        
        # 计算评估指标
        metrics = self._calculate_metrics(X, labels)
        
        return ClusteringResult(
            labels=labels,
            centroids=self.model.means_,
            n_clusters=self.n_components,
            algorithm="Gaussian Mixture",
            parameters={
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "max_iter": self.max_iter,
                "random_state": self.random_state
            },
            metrics=metrics,
            convergence_iterations=self.model.n_iter_,
            execution_time=execution_time
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新数据点的聚类标签"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)
    
    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """计算聚类评估指标"""
        metrics = {}
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = -1
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = -1
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = -1
        
        metrics['aic'] = self.model.aic(X)
        metrics['bic'] = self.model.bic(X)
        metrics['log_likelihood'] = self.model.score(X)
        
        return metrics


class SpectralClusteringAlgorithm(ClusteringAlgorithm):
    """谱聚类算法实现"""
    
    def __init__(self, n_clusters: int = 8, affinity: str = 'rbf',
                 gamma: float = 1.0, random_state: Optional[int] = None):
        """
        初始化谱聚类器
        
        Args:
            n_clusters: 聚类数量
            affinity: 相似性矩阵类型 ('rbf', 'nearest_neighbors', 'precomputed')
            gamma: RBF核参数
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.model = None
    
    def fit(self, X: np.ndarray, **kwargs) -> ClusteringResult:
        """拟合谱聚类模型"""
        import time
        start_time = time.time()
        
        # 初始化模型
        self.model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=self.gamma,
            random_state=self.random_state,
            **kwargs
        )
        
        # 拟合模型
        labels = self.model.fit_predict(X)
        
        execution_time = time.time() - start_time
        
        # 计算评估指标
        metrics = self._calculate_metrics(X, labels)
        
        return ClusteringResult(
            labels=labels,
            centroids=None,  # 谱聚类没有明确的中心点
            n_clusters=self.n_clusters,
            algorithm="Spectral",
            parameters={
                "n_clusters": self.n_clusters,
                "affinity": self.affinity,
                "gamma": self.gamma,
                "random_state": self.random_state
            },
            metrics=metrics,
            convergence_iterations=1,  # 谱聚类不需要迭代
            execution_time=execution_time
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """谱聚类不支持预测新数据点"""
        raise NotImplementedError("谱聚类不支持预测新数据点")
    
    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """计算聚类评估指标"""
        metrics = {}
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = -1
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = -1
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = -1
        
        return metrics


class FuzzyCMeansClustering(ClusteringAlgorithm):
    """模糊C均值聚类算法实现"""
    
    def __init__(self, n_clusters: int = 8, max_iter: int = 100, 
                 fuzziness: float = 2.0, tol: float = 1e-4):
        """
        初始化模糊C均值聚类器
        
        Args:
            n_clusters: 聚类数量
            max_iter: 最大迭代次数
            fuzziness: 模糊化参数（m值）
            tol: 收敛容忍度
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.fuzziness = fuzziness
        self.tol = tol
        self.centroids = None
        self.u = None  # 隶属度矩阵
    
    def fit(self, X: np.ndarray, **kwargs) -> ClusteringResult:
        """拟合模糊C均值模型"""
        import time
        start_time = time.time()
        
        n_samples, n_features = X.shape
        
        # 初始化隶属度矩阵
        np.random.seed(42)
        self.u = np.random.dirichlet(np.ones(self.n_clusters), n_samples)
        
        # 迭代优化
        for iteration in range(self.max_iter):
            u_old = self.u.copy()
            
            # 更新聚类中心
            self.centroids = self._update_centroids(X)
            
            # 更新隶属度矩阵
            self.u = self._update_membership_matrix(X, self.centroids)
            
            # 检查收敛
            if np.linalg.norm(self.u - u_old) < self.tol:
                break
        
        # 获取最终标签（最大隶属度对应的聚类）
        labels = np.argmax(self.u, axis=1)
        
        execution_time = time.time() - start_time
        
        # 计算评估指标
        metrics = self._calculate_metrics(X, labels)
        
        return ClusteringResult(
            labels=labels,
            centroids=self.centroids,
            n_clusters=self.n_clusters,
            algorithm="Fuzzy C-Means",
            parameters={
                "n_clusters": self.n_clusters,
                "max_iter": self.max_iter,
                "fuzziness": self.fuzziness,
                "tol": self.tol
            },
            metrics=metrics,
            convergence_iterations=iteration + 1,
            execution_time=execution_time
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新数据点的聚类标签"""
        if self.centroids is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 计算新数据点与聚类中心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        
        # 避免除零错误
        distances = np.power(distances, 2.0 / (self.fuzziness - 1))
        distances[distances == 0] = 1e-10
        
        # 计算隶属度
        u_new = 1.0 / (distances / np.sum(distances, axis=1, keepdims=True))
        
        # 返回最大隶属度对应的聚类
        return np.argmax(u_new, axis=1)
    
    def _update_centroids(self, X: np.ndarray) -> np.ndarray:
        """更新聚类中心"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for j in range(self.n_clusters):
            numerator = np.sum((self.u[:, j] ** self.fuzziness)[:, np.newaxis] * X, axis=0)
            denominator = np.sum(self.u[:, j] ** self.fuzziness)
            centroids[j] = numerator / denominator
        
        return centroids
    
    def _update_membership_matrix(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """更新隶属度矩阵"""
        n_samples = X.shape[0]
        u_new = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                # 计算距离
                distance = np.linalg.norm(X[i] - centroids[j])
                
                if distance == 0:
                    u_new[i, j] = 1.0
                else:
                    # 计算隶属度
                    sum_term = 0.0
                    for k in range(self.n_clusters):
                        distance_k = np.linalg.norm(X[i] - centroids[k])
                        if distance_k == 0:
                            sum_term += 1.0
                        else:
                            sum_term += (distance / distance_k) ** (2.0 / (self.fuzziness - 1))
                    
                    u_new[i, j] = 1.0 / sum_term
        
        return u_new
    
    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """计算聚类评估指标"""
        metrics = {}
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = -1
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = -1
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = -1
        
        return metrics


class ClusteringEvaluator:
    """聚类评估器"""
    
    @staticmethod
    def evaluate_clustering(X: np.ndarray, labels: np.ndarray, 
                          true_labels: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """
        评估聚类结果
        
        Args:
            X: 输入数据
            labels: 聚类标签
            true_labels: 真实标签（可选）
            
        Returns:
            EvaluationMetrics: 评估指标
        """
        # 计算无监督评估指标
        silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
        calinski_harabasz = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else -1
        davies_bouldin = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else -1
        
        # 计算有监督评估指标（如果有真实标签）
        ari = None
        nmi = None
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, labels)
            nmi = normalized_mutual_info_score(true_labels, labels)
        
        return EvaluationMetrics(
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            adjusted_rand_score=ari,
            normalized_mutual_info_score=nmi,
            inertia=None
        )
    
    @staticmethod
    def compare_clusterings(results: List[ClusteringResult], 
                          X: np.ndarray, 
                          true_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        比较多个聚类结果
        
        Args:
            results: 聚类结果列表
            X: 输入数据
            true_labels: 真实标签（可选）
            
        Returns:
            pd.DataFrame: 比较结果表格
        """
        comparison_data = []
        
        for result in results:
            metrics = ClusteringEvaluator.evaluate_clustering(X, result.labels, true_labels)
            
            comparison_data.append({
                'Algorithm': result.algorithm,
                'N_Clusters': result.n_clusters,
                'Silhouette_Score': metrics.silhouette_score,
                'Calinski_Harabasz_Score': metrics.calinski_harabasz_score,
                'Davies_Bouldin_Score': metrics.davies_bouldin_score,
                'Adjusted_Rand_Score': metrics.adjusted_rand_score,
                'Normalized_Mutual_Info_Score': metrics.normalized_mutual_info_score,
                'Execution_Time': result.execution_time,
                'Convergence_Iterations': result.convergence_iterations
            })
        
        return pd.DataFrame(comparison_data)


class ClusteringVisualizer:
    """聚类结果可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_clusters(self, X: np.ndarray, result: ClusteringResult, 
                     title: Optional[str] = None, save_path: Optional[str] = None,
                     true_labels: Optional[np.ndarray] = None) -> None:
        """
        绘制聚类结果
        
        Args:
            X: 输入数据
            result: 聚类结果
            title: 图形标题
            save_path: 保存路径
            true_labels: 真实标签（用于比较）
        """
        # 如果数据维度大于2，使用PCA降维
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
        else:
            X_plot = X
        
        # 创建子图
        fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, figsize=self.figsize)
        if true_labels is None:
            axes = [axes]
        
        # 绘制聚类结果
        scatter1 = axes[0].scatter(X_plot[:, 0], X_plot[:, 1], c=result.labels, 
                                 cmap='viridis', alpha=0.7, s=50)
        axes[0].set_title(f'{result.algorithm} 聚类结果' if title is None else title)
        axes[0].set_xlabel('第一主成分' if X.shape[1] > 2 else '特征1')
        axes[0].set_ylabel('第二主成分' if X.shape[1] > 2 else '特征2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # 绘制聚类中心（如果存在）
        if result.centroids is not None:
            if result.centroids.shape[1] > 2:
                centroids_plot = pca.transform(result.centroids)
            else:
                centroids_plot = result.centroids
            axes[0].scatter(centroids_plot[:, 0], centroids_plot[:, 1], 
                          c='red', marker='x', s=200, linewidths=3, label='聚类中心')
            axes[0].legend()
        
        # 如果有真实标签，绘制真实标签
        if true_labels is not None:
            scatter2 = axes[1].scatter(X_plot[:, 0], X_plot[:, 1], c=true_labels, 
                                     cmap='viridis', alpha=0.7, s=50)
            axes[1].set_title('真实标签')
            axes[1].set_xlabel('第一主成分' if X.shape[1] > 2 else '特征1')
            axes[1].set_ylabel('第二主成分' if X.shape[1] > 2 else '特征2')
            plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_dendrogram(self, X: np.ndarray, method: str = 'ward', 
                       save_path: Optional[str] = None) -> None:
        """
        绘制层次聚类树状图
        
        Args:
            X: 输入数据
            method: 连接方法
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)
        
        # 计算链接矩阵
        linkage_matrix = linkage(X, method=method)
        
        # 绘制树状图
        dendrogram(linkage_matrix, truncate_mode='lastp', p=30, show_leaf_counts=True)
        plt.title(f'层次聚类树状图 ({method})')
        plt.xlabel('样本索引或聚类大小')
        plt.ylabel('距离')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_evaluation_metrics(self, results: List[ClusteringResult], 
                              X: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        绘制评估指标比较图
        
        Args:
            results: 聚类结果列表
            X: 输入数据
            save_path: 保存路径
        """
        # 比较结果
        comparison_df = ClusteringEvaluator.compare_clusterings(results, X)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        # 轮廓系数
        axes[0].bar(comparison_df['Algorithm'], comparison_df['Silhouette_Score'])
        axes[0].set_title('轮廓系数比较')
        axes[0].set_ylabel('轮廓系数')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz指数
        axes[1].bar(comparison_df['Algorithm'], comparison_df['Calinski_Harabasz_Score'])
        axes[1].set_title('Calinski-Harabasz指数比较')
        axes[1].set_ylabel('Calinski-Harabasz指数')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin指数
        axes[2].bar(comparison_df['Algorithm'], comparison_df['Davies_Bouldin_Score'])
        axes[2].set_title('Davies-Bouldin指数比较')
        axes[2].set_ylabel('Davies-Bouldin指数')
        axes[2].tick_params(axis='x', rotation=45)
        
        # 执行时间
        axes[3].bar(comparison_df['Algorithm'], comparison_df['Execution_Time'])
        axes[3].set_title('执行时间比较')
        axes[3].set_ylabel('执行时间 (秒)')
        axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ParameterOptimizer:
    """聚类参数优化器"""
    
    def __init__(self, algorithm: str):
        """
        初始化参数优化器
        
        Args:
            algorithm: 算法名称
        """
        self.algorithm = algorithm
    
    def optimize_parameters(self, X: np.ndarray, param_grid: Dict[str, List], 
                          scoring: str = 'silhouette_score', 
                          cv: int = 3) -> Tuple[Dict[str, Any], float]:
        """
        优化聚类参数
        
        Args:
            X: 输入数据
            param_grid: 参数网格
            scoring: 评分指标
            cv: 交叉验证折数
            
        Returns:
            Tuple[Dict[str, Any], float]: 最佳参数和对应分数
        """
        best_score = -np.inf
        best_params = {}
        
        # 生成参数组合
        param_combinations = list(ParameterGrid(param_grid))
        
        for params in param_combinations:
            try:
                # 根据算法类型创建聚类器
                clusterer = self._create_clusterer(params)
                
                # 拟合并评估
                result = clusterer.fit(X)
                score = self._get_score(result, scoring)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                warnings.warn(f"参数组合 {params} 评估失败: {str(e)}")
                continue
        
        return best_params, best_score
    
    def _create_clusterer(self, params: Dict[str, Any]) -> ClusteringAlgorithm:
        """根据参数创建聚类器"""
        if self.algorithm.lower() == 'kmeans':
            return KMeansClustering(**params)
        elif self.algorithm.lower() == 'hierarchical':
            return HierarchicalClustering(**params)
        elif self.algorithm.lower() == 'dbscan':
            return DBSCANClustering(**params)
        elif self.algorithm.lower() == 'gmm':
            return GaussianMixtureClustering(**params)
        elif self.algorithm.lower() == 'spectral':
            return SpectralClusteringAlgorithm(**params)
        elif self.algorithm.lower() == 'fuzzy_cmeans':
            return FuzzyCMeansClustering(**params)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
    
    def _get_score(self, result: ClusteringResult, scoring: str) -> float:
        """获取评分"""
        if scoring in result.metrics:
            return result.metrics[scoring]
        else:
            raise ValueError(f"未知的评分指标: {scoring}")


class ClusteringAlgorithmLibrary:
    """聚类算法库主类"""
    
    def __init__(self):
        """初始化聚类算法库"""
        self.algorithms = {
            'kmeans': KMeansClustering,
            'hierarchical': HierarchicalClustering,
            'dbscan': DBSCANClustering,
            'gmm': GaussianMixtureClustering,
            'spectral': SpectralClusteringAlgorithm,
            'fuzzy_cmeans': FuzzyCMeansClustering
        }
        self.evaluator = ClusteringEvaluator()
        self.visualizer = ClusteringVisualizer()
    
    def cluster(self, X: np.ndarray, algorithm: str, **kwargs) -> ClusteringResult:
        """
        执行聚类分析
        
        Args:
            X: 输入数据
            algorithm: 算法名称
            **kwargs: 算法参数
            
        Returns:
            ClusteringResult: 聚类结果
        """
        if algorithm.lower() not in self.algorithms:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        clusterer = self.algorithms[algorithm.lower()](**kwargs)
        return clusterer.fit(X)
    
    def compare_algorithms(self, X: np.ndarray, algorithms: List[str], 
                          params: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """
        比较多种聚类算法
        
        Args:
            X: 输入数据
            algorithms: 算法列表
            params: 算法参数字典
            
        Returns:
            pd.DataFrame: 比较结果
        """
        results = []
        
        for algo in algorithms:
            try:
                algo_params = params.get(algo, {}) if params else {}
                # 为不同算法设置默认参数
                if algo.lower() == 'dbscan':
                    default_params = {'eps': 0.5, 'min_samples': 5}
                elif algo.lower() == 'gmm':
                    default_params = {'n_components': 3}
                elif algo.lower() == 'kmeans':
                    default_params = {'n_clusters': 3}
                elif algo.lower() == 'hierarchical':
                    default_params = {'n_clusters': 3}
                elif algo.lower() == 'spectral':
                    default_params = {'n_clusters': 3}
                elif algo.lower() == 'fuzzy_cmeans':
                    default_params = {'n_clusters': 3}
                else:
                    default_params = {}
                
                # 合并默认参数和用户参数
                final_params = {**default_params, **algo_params}
                result = self.cluster(X, algo, **final_params)
                results.append(result)
            except Exception as e:
                warnings.warn(f"算法 {algo} 执行失败: {str(e)}")
                continue
        
        return self.evaluator.compare_clusterings(results, X)
    
    def optimize_parameters(self, X: np.ndarray, algorithm: str, 
                          param_grid: Dict[str, List]) -> Tuple[Dict[str, Any], ClusteringResult]:
        """
        优化聚类参数
        
        Args:
            X: 输入数据
            algorithm: 算法名称
            param_grid: 参数网格
            
        Returns:
            Tuple[Dict[str, Any], ClusteringResult]: 最佳参数和对应的聚类结果
        """
        optimizer = ParameterOptimizer(algorithm)
        best_params, best_score = optimizer.optimize_parameters(X, param_grid)
        
        # 使用最佳参数重新运行聚类
        result = self.cluster(X, algorithm, **best_params)
        
        return best_params, result
    
    def visualize_results(self, X: np.ndarray, result: ClusteringResult, 
                         save_path: Optional[str] = None,
                         true_labels: Optional[np.ndarray] = None) -> None:
        """
        可视化聚类结果
        
        Args:
            X: 输入数据
            result: 聚类结果
            save_path: 保存路径
            true_labels: 真实标签
        """
        self.visualizer.plot_clusters(X, result, save_path=save_path, true_labels=true_labels)
    
    def plot_dendrogram(self, X: np.ndarray, method: str = 'ward', 
                       save_path: Optional[str] = None) -> None:
        """
        绘制层次聚类树状图
        
        Args:
            X: 输入数据
            method: 连接方法
            save_path: 保存路径
        """
        self.visualizer.plot_dendrogram(X, method, save_path)
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray, 
                          true_labels: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """
        评估聚类结果
        
        Args:
            X: 输入数据
            labels: 聚类标签
            true_labels: 真实标签
            
        Returns:
            EvaluationMetrics: 评估指标
        """
        return self.evaluator.evaluate_clustering(X, labels, true_labels)


def generate_sample_data(n_samples: int = 300, n_features: int = 2, 
                        n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成样本数据用于测试
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        n_clusters: 聚类数量
        random_state: 随机种子
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 特征数据和真实标签
    """
    np.random.seed(random_state)
    
    # 生成聚类中心
    centers = np.random.randn(n_clusters, n_features) * 2
    
    # 为每个聚类生成数据
    X = []
    y = []
    
    for i in range(n_clusters):
        # 为每个聚类生成样本
        cluster_size = n_samples // n_clusters
        if i == n_clusters - 1:  # 最后一个聚类包含剩余的样本
            cluster_size += n_samples % n_clusters
        
        cluster_data = np.random.randn(cluster_size, n_features) + centers[i]
        X.append(cluster_data)
        y.append(np.full(cluster_size, i))
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # 随机打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def run_comprehensive_test():
    """运行综合测试"""
    print("=== U8聚类算法库综合测试 ===\n")
    
    # 生成测试数据
    print("1. 生成测试数据...")
    X, true_labels = generate_sample_data(n_samples=300, n_features=2, n_clusters=3)
    print(f"   数据形状: {X.shape}")
    print(f"   真实聚类数: {len(np.unique(true_labels))}")
    
    # 初始化聚类算法库
    library = ClusteringAlgorithmLibrary()
    
    # 测试各种聚类算法
    algorithms = ['kmeans', 'hierarchical', 'dbscan', 'gmm', 'spectral', 'fuzzy_cmeans']
    results = []
    
    print("\n2. 运行聚类算法...")
    for algo in algorithms:
        try:
            print(f"   运行 {algo}...")
            # 根据算法类型设置合适的参数
            if algo == 'dbscan':
                result = library.cluster(X, algo, eps=0.5, min_samples=5)
            elif algo == 'gmm':
                result = library.cluster(X, algo, n_components=3)
            else:
                result = library.cluster(X, algo, n_clusters=3)
            results.append(result)
            print(f"   - 聚类数: {result.n_clusters}")
            print(f"   - 轮廓系数: {result.metrics.get('silhouette_score', 'N/A'):.3f}")
            print(f"   - 执行时间: {result.execution_time:.3f}秒")
        except Exception as e:
            print(f"   - {algo} 执行失败: {str(e)}")
    
    # 比较算法结果
    print("\n3. 算法比较结果:")
    comparison_df = library.compare_algorithms(X, [r.algorithm.lower() for r in results])
    print(comparison_df.round(3))
    
    # 参数优化测试
    print("\n4. 参数优化测试 (K-Means)...")
    param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'max_iter': [100, 200, 300]
    }
    
    try:
        best_params, best_result = library.optimize_parameters(X, 'kmeans', param_grid)
        print(f"   最佳参数: {best_params}")
        print(f"   最佳轮廓系数: {best_result.metrics.get('silhouette_score', 'N/A'):.3f}")
    except Exception as e:
        print(f"   参数优化失败: {str(e)}")
    
    # 评估指标测试
    print("\n5. 评估指标测试...")
    for i, result in enumerate(results):
        metrics = library.evaluate_clustering(X, result.labels, true_labels)
        print(f"   {result.algorithm}:")
        print(f"     轮廓系数: {metrics.silhouette_score:.3f}")
        if metrics.adjusted_rand_score is not None:
            print(f"     调整兰德指数: {metrics.adjusted_rand_score:.3f}")
        if metrics.normalized_mutual_info_score is not None:
            print(f"     标准化互信息: {metrics.normalized_mutual_info_score:.3f}")
    
    print("\n=== 测试完成 ===")
    
    # 保存比较结果到文件
    try:
        comparison_df.to_csv('/workspace/D/AO/AOO/U/U8/clustering_comparison_results.csv', index=False)
        print("   比较结果已保存到 clustering_comparison_results.csv")
    except Exception as e:
        print(f"   保存结果失败: {str(e)}")


if __name__ == "__main__":
    # 运行综合测试
    run_comprehensive_test()