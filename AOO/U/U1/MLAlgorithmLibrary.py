"""
机器学习算法库 (MLAlgorithmLibrary)

实现多种机器学习算法，包括监督学习、无监督学习和集成学习方法。
支持完整的训练、预测、评估和超参数优化功能。

作者: 智能量化系统
版本: 1.0.0
日期: 2025-11-05
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """模型训练和预测结果"""
    predictions: np.ndarray
    accuracy: Optional[float]
    mse: Optional[float]
    r2_score: Optional[float]
    feature_importance: Optional[np.ndarray]
    model_params: Dict[str, Any]


class BaseMLAlgorithm(ABC):
    """机器学习算法基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseMLAlgorithm':
        """训练模型"""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
        
    def preprocess_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """数据预处理"""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        predictions = self.predict(X)
        
        results = {}
        
        # 计算准确率（分类任务）
        try:
            results['accuracy'] = accuracy_score(y, predictions)
        except:
            results['accuracy'] = None
            
        # 计算均方误差（回归任务）
        try:
            results['mse'] = mean_squared_error(y, predictions)
        except:
            results['mse'] = None
            
        # 计算R²分数
        try:
            results['r2_score'] = r2_score(y, predictions)
        except:
            results['r2_score'] = None
            
        return results


class LinearRegressionAlgorithm(BaseMLAlgorithm):
    """线性回归算法"""
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        super().__init__("LinearRegression")
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionAlgorithm':
        """训练线性回归模型"""
        X_scaled = self.preprocess_data(X, fit=True)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        X_scaled = self.preprocess_data(X, fit=False)
        return self.model.predict(X_scaled)


class LogisticRegressionAlgorithm(BaseMLAlgorithm):
    """逻辑回归算法"""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, solver: str = 'lbfgs'):
        super().__init__("LogisticRegression")
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionAlgorithm':
        """训练逻辑回归模型"""
        X_scaled = self.preprocess_data(X, fit=True)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        X_scaled = self.preprocess_data(X, fit=False)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        X_scaled = self.preprocess_data(X, fit=False)
        return self.model.predict_proba(X_scaled)


class SVMAlgorithm(BaseMLAlgorithm):
    """支持向量机算法"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', 
                 is_regression: bool = False):
        super().__init__("SVM")
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.is_regression = is_regression
        
        if is_regression:
            self.model = SVR(kernel=kernel, C=C, gamma=gamma)
        else:
            self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMAlgorithm':
        """训练SVM模型"""
        X_scaled = self.preprocess_data(X, fit=True)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        X_scaled = self.preprocess_data(X, fit=False)
        return self.model.predict(X_scaled)


class RandomForestAlgorithm(BaseMLAlgorithm):
    """随机森林算法"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 is_regression: bool = False, random_state: int = 42):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.is_regression = is_regression
        self.random_state = random_state
        
        if is_regression:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=random_state
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestAlgorithm':
        """训练随机森林模型"""
        X_scaled = self.preprocess_data(X, fit=True)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        X_scaled = self.preprocess_data(X, fit=False)
        return self.model.predict(X_scaled)
        
    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.feature_importances_


class GradientBoostingAlgorithm(BaseMLAlgorithm):
    """梯度提升算法"""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3, is_regression: bool = False, random_state: int = 42):
        super().__init__("GradientBoosting")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.is_regression = is_regression
        self.random_state = random_state
        
        if is_regression:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingAlgorithm':
        """训练梯度提升模型"""
        X_scaled = self.preprocess_data(X, fit=True)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        X_scaled = self.preprocess_data(X, fit=False)
        return self.model.predict(X_scaled)


class NaiveBayesAlgorithm(BaseMLAlgorithm):
    """朴素贝叶斯算法"""
    
    def __init__(self):
        super().__init__("NaiveBayes")
        self.model = GaussianNB()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesAlgorithm':
        """训练朴素贝叶斯模型"""
        X_scaled = self.preprocess_data(X, fit=True)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        X_scaled = self.preprocess_data(X, fit=False)
        return self.model.predict(X_scaled)


class KNNAlgorithm(BaseMLAlgorithm):
    """K近邻算法"""
    
    def __init__(self, n_neighbors: int = 5, is_regression: bool = False):
        super().__init__("KNN")
        self.n_neighbors = n_neighbors
        self.is_regression = is_regression
        
        if is_regression:
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        else:
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNAlgorithm':
        """训练KNN模型"""
        X_scaled = self.preprocess_data(X, fit=True)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        X_scaled = self.preprocess_data(X, fit=False)
        return self.model.predict(X_scaled)


class DecisionTreeAlgorithm(BaseMLAlgorithm):
    """决策树算法"""
    
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, 
                 is_regression: bool = False, random_state: int = 42):
        super().__init__("DecisionTree")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.is_regression = is_regression
        self.random_state = random_state
        
        if is_regression:
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
        else:
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeAlgorithm':
        """训练决策树模型"""
        X_scaled = self.preprocess_data(X, fit=True)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        X_scaled = self.preprocess_data(X, fit=False)
        return self.model.predict(X_scaled)


class EnsembleLearning:
    """集成学习方法"""
    
    def __init__(self, algorithms: List[BaseMLAlgorithm], voting: str = 'hard'):
        """
        初始化集成学习
        
        Args:
            algorithms: 算法列表
            voting: 投票方式 ('hard' 或 'soft')
        """
        self.algorithms = algorithms
        self.voting = voting
        self.ensemble_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleLearning':
        """训练集成模型"""
        estimators = []
        
        for algo in self.algorithms:
            algo.fit(X, y)
            estimators.append((algo.name, algo.model))
            
        if self.voting == 'soft':
            # 软投票需要所有基学习器支持概率预测
            self.ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
        else:
            # 硬投票
            self.ensemble_model = VotingClassifier(estimators=estimators, voting='hard')
            
        # 集成模型需要单独训练
        X_scaled = self.algorithms[0].preprocess_data(X, fit=True)
        self.ensemble_model.fit(X_scaled, y)
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.ensemble_model is None:
            raise ValueError("集成模型尚未训练，请先调用fit方法")
        X_scaled = self.algorithms[0].preprocess_data(X, fit=False)
        return self.ensemble_model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率（仅软投票支持）"""
        if self.ensemble_model is None:
            raise ValueError("集成模型尚未训练，请先调用fit方法")
        if self.voting != 'soft':
            raise ValueError("只有软投票模式支持概率预测")
        X_scaled = self.algorithms[0].preprocess_data(X, fit=False)
        return self.ensemble_model.predict_proba(X_scaled)


class FeatureSelection:
    """特征选择算法"""
    
    @staticmethod
    def select_k_best(X: np.ndarray, y: np.ndarray, k: int = 10, 
                     task_type: str = 'classification') -> Tuple[np.ndarray, List[int]]:
        """
        选择K个最佳特征
        
        Args:
            X: 特征矩阵
            y: 目标变量
            k: 选择特征数量
            task_type: 任务类型 ('classification' 或 'regression')
            
        Returns:
            降维后的特征矩阵和选择的特征索引
        """
        if task_type == 'classification':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(f_regression, k=k)
            
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        
        return X_selected, selected_features.tolist()
    
    @staticmethod
    def recursive_feature_elimination(X: np.ndarray, y: np.ndarray, 
                                     estimator: BaseMLAlgorithm, n_features: int = 10) -> Tuple[np.ndarray, List[int]]:
        """
        递归特征消除
        
        Args:
            X: 特征矩阵
            y: 目标变量
            estimator: 基学习器
            n_features: 选择的特征数量
            
        Returns:
            降维后的特征矩阵和选择的特征索引
        """
        selector = RFE(estimator.model, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        
        return X_selected, selected_features.tolist()


class HyperparameterOptimizer:
    """超参数优化"""
    
    def __init__(self, algorithm: BaseMLAlgorithm, param_grid: Dict[str, Any]):
        """
        初始化超参数优化器
        
        Args:
            algorithm: 机器学习算法
            param_grid: 参数网格
        """
        self.algorithm = algorithm
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None
        
    def grid_search(self, X: np.ndarray, y: np.ndarray, cv: int = 5, 
                   scoring: str = 'accuracy', n_jobs: int = -1) -> Dict[str, Any]:
        """
        网格搜索优化
        
        Args:
            X: 特征矩阵
            y: 目标变量
            cv: 交叉验证折数
            scoring: 评分标准
            n_jobs: 并行作业数
            
        Returns:
            最佳参数和分数
        """
        grid_search = GridSearchCV(
            estimator=self.algorithm.model,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0
        )
        
        X_scaled = self.algorithm.preprocess_data(X, fit=True)
        grid_search.fit(X_scaled, y)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': grid_search.cv_results_
        }
    
    def random_search(self, X: np.ndarray, y: np.ndarray, n_iter: int = 100, 
                     cv: int = 5, scoring: str = 'accuracy', n_jobs: int = -1) -> Dict[str, Any]:
        """
        随机搜索优化
        
        Args:
            X: 特征矩阵
            y: 目标变量
            n_iter: 迭代次数
            cv: 交叉验证折数
            scoring: 评分标准
            n_jobs: 并行作业数
            
        Returns:
            最佳参数和分数
        """
        random_search = RandomizedSearchCV(
            estimator=self.algorithm.model,
            param_distributions=self.param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42,
            verbose=0
        )
        
        X_scaled = self.algorithm.preprocess_data(X, fit=True)
        random_search.fit(X_scaled, y)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': random_search.cv_results_
        }


class MLAlgorithmLibrary:
    """机器学习算法库主类"""
    
    def __init__(self):
        """初始化机器学习算法库"""
        self.algorithms = {}
        self.results = {}
        
    def create_algorithm(self, algorithm_type: str, **kwargs) -> BaseMLAlgorithm:
        """
        创建机器学习算法实例
        
        Args:
            algorithm_type: 算法类型
            **kwargs: 算法参数
            
        Returns:
            算法实例
        """
        algorithm_map = {
            'linear_regression': LinearRegressionAlgorithm,
            'logistic_regression': LogisticRegressionAlgorithm,
            'svm': SVMAlgorithm,
            'random_forest': RandomForestAlgorithm,
            'gradient_boosting': GradientBoostingAlgorithm,
            'naive_bayes': NaiveBayesAlgorithm,
            'knn': KNNAlgorithm,
            'decision_tree': DecisionTreeAlgorithm
        }
        
        if algorithm_type not in algorithm_map:
            raise ValueError(f"不支持的算法类型: {algorithm_type}")
            
        algorithm = algorithm_map[algorithm_type](**kwargs)
        self.algorithms[algorithm_type] = algorithm
        return algorithm
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, algorithm_type: str, 
                          test_size: float = 0.2, random_state: int = 42, **kwargs) -> ModelResult:
        """
        训练和评估模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            algorithm_type: 算法类型
            test_size: 测试集比例
            random_state: 随机种子
            **kwargs: 算法参数
            
        Returns:
            模型结果
        """
        # 创建算法实例
        algorithm = self.create_algorithm(algorithm_type, **kwargs)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 训练模型
        algorithm.fit(X_train, y_train)
        
        # 预测
        predictions = algorithm.predict(X_test)
        
        # 评估
        evaluation = algorithm.evaluate(X_test, y_test)
        
        # 获取特征重要性（如果支持）
        feature_importance = None
        if hasattr(algorithm, 'get_feature_importance'):
            try:
                feature_importance = algorithm.get_feature_importance()
            except:
                pass
        
        # 创建结果
        result = ModelResult(
            predictions=predictions,
            accuracy=evaluation.get('accuracy'),
            mse=evaluation.get('mse'),
            r2_score=evaluation.get('r2_score'),
            feature_importance=feature_importance,
            model_params=algorithm.__dict__
        )
        
        self.results[algorithm_type] = result
        return result
    
    def compare_algorithms(self, X: np.ndarray, y: np.ndarray, 
                          algorithm_types: List[str], test_size: float = 0.2) -> pd.DataFrame:
        """
        比较多种算法性能
        
        Args:
            X: 特征矩阵
            y: 目标变量
            algorithm_types: 算法类型列表
            test_size: 测试集比例
            
        Returns:
            性能比较结果DataFrame
        """
        results = []
        
        for algo_type in algorithm_types:
            try:
                result = self.train_and_evaluate(X, y, algo_type, test_size)
                results.append({
                    'Algorithm': algo_type,
                    'Accuracy': result.accuracy,
                    'MSE': result.mse,
                    'R2_Score': result.r2_score
                })
            except Exception as e:
                print(f"算法 {algo_type} 执行失败: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def ensemble_predict(self, X: np.ndarray, algorithm_types: List[str], 
                        voting: str = 'hard') -> np.ndarray:
        """
        集成预测
        
        Args:
            X: 特征矩阵
            algorithm_types: 算法类型列表
            voting: 投票方式
            
        Returns:
            集成预测结果
        """
        algorithms = []
        for algo_type in algorithm_types:
            if algo_type in self.algorithms:
                algorithms.append(self.algorithms[algo_type])
            else:
                raise ValueError(f"算法 {algo_type} 未训练，请先调用train_and_evaluate")
        
        ensemble = EnsembleLearning(algorithms, voting=voting)
        
        # 重新训练集成模型
        # 这里需要重新获取训练数据，实际使用时应该保存训练数据
        raise NotImplementedError("集成预测需要保存训练数据，请单独调用EnsembleLearning.fit")
    
    def feature_selection(self, X: np.ndarray, y: np.ndarray, 
                         method: str = 'select_k_best', k: int = 10, 
                         algorithm_type: Optional[str] = None) -> Tuple[np.ndarray, List[int]]:
        """
        特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            method: 选择方法 ('select_k_best' 或 'rfe')
            k: 选择特征数量
            algorithm_type: 基学习器类型（仅RFE需要）
            
        Returns:
            降维后的特征矩阵和选择的特征索引
        """
        if method == 'select_k_best':
            # 自动判断任务类型
            if len(np.unique(y)) <= 20 and isinstance(y[0], (int, np.integer)):
                task_type = 'classification'
            else:
                task_type = 'regression'
            return FeatureSelection.select_k_best(X, y, k, task_type)
        
        elif method == 'rfe':
            if algorithm_type is None:
                raise ValueError("RFE方法需要指定algorithm_type")
            
            if algorithm_type not in self.algorithms:
                raise ValueError(f"算法 {algorithm_type} 未训练")
            
            estimator = self.algorithms[algorithm_type]
            return FeatureSelection.recursive_feature_elimination(X, y, estimator, k)
        
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
    
    def hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray, 
                                   algorithm_type: str, param_grid: Dict[str, Any],
                                   method: str = 'grid_search', **kwargs) -> Dict[str, Any]:
        """
        超参数优化
        
        Args:
            X: 特征矩阵
            y: 目标变量
            algorithm_type: 算法类型
            param_grid: 参数网格
            method: 优化方法 ('grid_search' 或 'random_search')
            **kwargs: 其他参数
            
        Returns:
            优化结果
        """
        if algorithm_type not in self.algorithms:
            self.create_algorithm(algorithm_type)
        
        algorithm = self.algorithms[algorithm_type]
        optimizer = HyperparameterOptimizer(algorithm, param_grid)
        
        if method == 'grid_search':
            return optimizer.grid_search(X, y, **kwargs)
        elif method == 'random_search':
            return optimizer.random_search(X, y, **kwargs)
        else:
            raise ValueError(f"不支持的优化方法: {method}")
    
    def get_algorithm_info(self, algorithm_type: str) -> Dict[str, Any]:
        """
        获取算法信息
        
        Args:
            algorithm_type: 算法类型
            
        Returns:
            算法信息字典
        """
        if algorithm_type not in self.algorithms:
            raise ValueError(f"算法 {algorithm_type} 不存在")
        
        algorithm = self.algorithms[algorithm_type]
        
        return {
            'name': algorithm.name,
            'type': type(algorithm).__name__,
            'is_fitted': algorithm.is_fitted,
            'parameters': algorithm.__dict__
        }
    
    def list_algorithms(self) -> List[str]:
        """列出所有可用的算法类型"""
        return [
            'linear_regression', 'logistic_regression', 'svm',
            'random_forest', 'gradient_boosting', 'naive_bayes',
            'knn', 'decision_tree'
        ]


def create_sample_data(n_samples: int = 1000, n_features: int = 10, 
                      task_type: str = 'classification', random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建示例数据
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        task_type: 任务类型 ('classification' 或 'regression')
        random_state: 随机种子
        
    Returns:
        特征矩阵和目标变量
    """
    np.random.seed(random_state)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成目标变量
    if task_type == 'classification':
        # 分类任务：基于特征线性组合生成标签
        weights = np.random.randn(n_features)
        linear_combination = X @ weights
        y = (linear_combination > np.median(linear_combination)).astype(int)
    else:
        # 回归任务：添加噪声
        weights = np.random.randn(n_features)
        y = X @ weights + 0.1 * np.random.randn(n_samples)
    
    return X, y


def run_comprehensive_test():
    """运行综合测试"""
    print("=== 机器学习算法库综合测试 ===\n")
    
    # 创建算法库实例
    ml_library = MLAlgorithmLibrary()
    
    # 创建示例数据
    print("1. 创建示例数据...")
    X_class, y_class = create_sample_data(n_samples=1000, n_features=10, 
                                         task_type='classification', random_state=42)
    X_reg, y_reg = create_sample_data(n_samples=1000, n_features=10, 
                                     task_type='regression', random_state=42)
    
    # 测试分类算法
    print("\n2. 测试分类算法...")
    classification_algorithms = ['logistic_regression', 'svm', 'random_forest', 
                               'naive_bayes', 'knn', 'decision_tree']
    
    for algo in classification_algorithms:
        try:
            result = ml_library.train_and_evaluate(X_class, y_class, algo)
            print(f"{algo}: 准确率 = {result.accuracy:.4f}")
        except Exception as e:
            print(f"{algo}: 执行失败 - {e}")
    
    # 测试回归算法
    print("\n3. 测试回归算法...")
    regression_algorithms = ['linear_regression', 'svm', 'random_forest', 'knn', 'decision_tree']
    
    for algo in regression_algorithms:
        try:
            # 为回归任务创建算法实例并训练评估
            if algo == 'linear_regression':
                result = ml_library.train_and_evaluate(X_reg, y_reg, algo)
            elif algo == 'svm':
                result = ml_library.train_and_evaluate(X_reg, y_reg, algo, is_regression=True)
            elif algo == 'random_forest':
                result = ml_library.train_and_evaluate(X_reg, y_reg, algo, is_regression=True)
            elif algo == 'knn':
                result = ml_library.train_and_evaluate(X_reg, y_reg, algo, is_regression=True)
            elif algo == 'decision_tree':
                result = ml_library.train_and_evaluate(X_reg, y_reg, algo, is_regression=True)
            
            print(f"{algo}: R² = {result.r2_score:.4f}, MSE = {result.mse:.4f}")
        except Exception as e:
            print(f"{algo}: 执行失败 - {e}")
    
    # 测试算法比较
    print("\n4. 算法性能比较...")
    comparison_results = ml_library.compare_algorithms(X_class, y_class, 
                                                     classification_algorithms[:4])
    print(comparison_results)
    
    # 测试特征选择
    print("\n5. 测试特征选择...")
    try:
        X_selected, selected_features = ml_library.feature_selection(
            X_class, y_class, method='select_k_best', k=5
        )
        print(f"选择的特征索引: {selected_features}")
        print(f"原始特征数: {X_class.shape[1]}, 选择后特征数: {X_selected.shape[1]}")
    except Exception as e:
        print(f"特征选择失败: {e}")
    
    # 测试超参数优化
    print("\n6. 测试超参数优化...")
    try:
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        
        optimization_result = ml_library.hyperparameter_optimization(
            X_class, y_class, 'svm', param_grid, method='grid_search'
        )
        print(f"最佳参数: {optimization_result['best_params']}")
        print(f"最佳分数: {optimization_result['best_score']:.4f}")
    except Exception as e:
        print(f"超参数优化失败: {e}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    # 运行综合测试
    run_comprehensive_test()