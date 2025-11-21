"""
V1模型训练器 - ModelTrainer.py

这是一个功能完整的模型训练器类，支持：
1. 统一模型训练接口
2. 自动数据预处理
3. 交叉验证训练
4. 早停机制和模型保存
5. 分布式训练支持
6. 超参数自动调优
7. 模型性能监控
8. 训练进度跟踪
9. 错误处理和恢复


版本: 1.0.0
日期: 2025-11-05
"""

import os
import json
import time
import pickle
import logging
import warnings
import traceback
from typing import (
    Any, Dict, List, Optional, Union, Callable, Tuple, 
    TypeVar, Generic, Iterator, Generator
)
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
import joblib

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 类型定义
T = TypeVar('T')
ModelType = TypeVar('ModelType')
MetricType = Dict[str, float]
TrainingResult = Dict[str, Any]


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 基础训练参数
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_size: float = 0.2
    
    # 交叉验证参数
    cv_folds: int = 5
    cv_random_state: int = 42
    
    # 早停参数
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # 模型保存参数
    save_best_model: bool = True
    save_frequency: int = 10
    model_save_path: str = "./models/"
    
    # 分布式训练参数
    distributed: bool = False
    n_workers: int = mp.cpu_count()
    
    # 超参数调优参数
    hyperparameter_tuning: bool = False
    tuning_method: str = "grid"  # "grid" or "random"
    tuning_cv_folds: int = 3
    tuning_n_iter: int = 50
    
    # 监控参数
    monitor_metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    progress_report_frequency: int = 1
    
    # 错误处理参数
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 预处理参数
    preprocessing: bool = True
    scaling_method: str = "standard"  # "standard", "minmax", "robust"
    handle_missing: str = "drop"  # "drop", "fill_mean", "fill_median", "fill_mode"
    
    # 其他参数
    random_state: int = 42
    verbose: bool = True
    save_logs: bool = True
    log_path: str = "./logs/"


@dataclass
class TrainingMetrics:
    """训练指标类"""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    training_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scaler = None
        self.feature_columns = []
        self.target_column = None
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """拟合并转换数据"""
        X_processed = X.copy()
        
        # 处理缺失值
        if self.config.handle_missing != "drop":
            X_processed = self._handle_missing_values(X_processed)
        
        # 记录特征列
        self.feature_columns = X_processed.columns.tolist()
        self.target_column = y.name if y is not None else None
        
        # 数据缩放
        if self.config.preprocessing:
            X_processed = self._scale_features(X_processed)
        
        # 移除缺失值
        if self.config.handle_missing == "drop":
            mask = ~(X_processed.isnull().any(axis=1) | (y.isnull() if y is not None else False))
            X_processed = X_processed[mask]
            if y is not None:
                y = y[mask]
        
        return X_processed, y
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        X_processed = X.copy()
        
        # 处理缺失值
        if self.config.handle_missing != "drop":
            X_processed = self._handle_missing_values(X_processed)
        
        # 数据缩放
        if self.config.preprocessing and self.scaler is not None:
            X_processed = self._scale_features(X_processed)
        
        return X_processed
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        X_processed = X.copy()
        
        for column in X_processed.columns:
            if X_processed[column].isnull().any():
                if self.config.handle_missing == "fill_mean":
                    X_processed[column].fillna(X_processed[column].mean(), inplace=True)
                elif self.config.handle_missing == "fill_median":
                    X_processed[column].fillna(X_processed[column].median(), inplace=True)
                elif self.config.handle_missing == "fill_mode":
                    X_processed[column].fillna(X_processed[column].mode()[0], inplace=True)
        
        return X_processed
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """缩放特征"""
        if self.scaler is None:
            if self.config.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.config.scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.config.scaling_method == "robust":
                self.scaler = RobustScaler()
            
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled


class ModelPerformanceMonitor:
    """模型性能监控器类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics_history: List[TrainingMetrics] = []
        self.best_metric_value = float('-inf') if 'accuracy' in config.monitor_metrics else float('inf')
        self.best_epoch = 0
        
    def update(self, metrics: TrainingMetrics) -> None:
        """更新指标"""
        self.metrics_history.append(metrics)
        
        # 检查是否是最佳模型
        primary_metric = self._get_primary_metric(metrics)
        if self._is_better_metric(primary_metric):
            self.best_metric_value = primary_metric
            self.best_epoch = metrics.epoch
    
    def _get_primary_metric(self, metrics: TrainingMetrics) -> float:
        """获取主要指标"""
        if 'accuracy' in self.config.monitor_metrics:
            return metrics.val_accuracy
        elif 'loss' in self.config.monitor_metrics:
            return -metrics.val_loss  # 负值，因为loss越小越好
        else:
            return metrics.val_accuracy
    
    def _is_better_metric(self, current_value: float) -> bool:
        """判断当前指标是否更好"""
        if 'accuracy' in self.config.monitor_metrics:
            return current_value > self.best_metric_value
        elif 'loss' in self.config.monitor_metrics:
            return current_value < self.best_metric_value
        else:
            return current_value > self.best_metric_value
    
    def should_early_stop(self) -> bool:
        """检查是否应该早停"""
        if not self.config.early_stopping or len(self.metrics_history) < self.config.early_stopping_patience:
            return False
        
        recent_metrics = self.metrics_history[-self.config.early_stopping_patience:]
        
        if 'accuracy' in self.config.monitor_metrics:
            return all(m.val_accuracy <= self.best_metric_value - self.config.early_stopping_min_delta 
                      for m in recent_metrics)
        elif 'loss' in self.config.monitor_metrics:
            return all(m.val_loss >= self.best_metric_value + self.config.early_stopping_min_delta 
                      for m in recent_metrics)
        
        return False
    
    def get_best_model_epoch(self) -> int:
        """获取最佳模型对应的epoch"""
        return self.best_epoch
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.metrics_history:
            return {}
        
        final_metrics = self.metrics_history[-1]
        
        return {
            "best_epoch": self.best_epoch,
            "best_metric_value": self.best_metric_value,
            "total_epochs": len(self.metrics_history),
            "final_train_loss": final_metrics.train_loss,
            "final_val_loss": final_metrics.val_loss,
            "final_train_accuracy": final_metrics.train_accuracy,
            "final_val_accuracy": final_metrics.val_accuracy,
            "total_training_time": sum(m.training_time for m in self.metrics_history)
        }


class HyperparameterTuner:
    """超参数调优器类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def tune(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        """调优超参数"""
        if not self.config.hyperparameter_tuning:
            return model, {}
        
        logger.info("开始超参数调优...")
        
        # 定义超参数搜索空间
        param_grids = self._get_param_grid(model)
        
        if not param_grids:
            logger.warning("未找到合适的超参数搜索空间")
            return model, {}
        
        # 选择搜索方法
        if self.config.tuning_method == "grid":
            search = GridSearchCV(
                model, param_grids, cv=self.config.tuning_cv_folds,
                scoring='accuracy', n_jobs=-1, verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, param_grids, n_iter=self.config.tuning_n_iter,
                cv=self.config.tuning_cv_folds, scoring='accuracy',
                n_jobs=-1, random_state=self.config.random_state, verbose=1
            )
        
        # 执行搜索
        search.fit(X, y)
        
        logger.info(f"最佳参数: {search.best_params_}")
        logger.info(f"最佳分数: {search.best_score_:.4f}")
        
        return search.best_estimator_, {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": search.cv_results_
        }
    
    def _get_param_grid(self, model: Any) -> Dict[str, List]:
        """获取超参数搜索空间"""
        model_name = model.__class__.__name__
        
        param_grids = {
            "RandomForestClassifier": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "GradientBoostingClassifier": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            "LogisticRegression": {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            "SVC": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        
        return param_grids.get(model_name, {})


class DistributedTrainer:
    """分布式训练器类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def train_distributed(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                         train_func: Callable) -> List[Any]:
        """分布式训练"""
        if not self.config.distributed:
            return [train_func(model, X, y)]
        
        logger.info(f"开始分布式训练，使用 {self.config.n_workers} 个工作进程")
        
        # 数据分割
        data_splits = self._split_data(X, y)
        
        models = []
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = []
            for i, (X_split, y_split) in enumerate(data_splits):
                future = executor.submit(train_func, model, X_split, y_split)
                futures.append((i, future))
            
            for i, future in futures:
                try:
                    trained_model = future.result()
                    models.append(trained_model)
                    logger.info(f"进程 {i} 训练完成")
                except Exception as e:
                    logger.error(f"进程 {i} 训练失败: {str(e)}")
        
        return models
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """分割数据用于分布式训练"""
        n_splits = self.config.n_workers
        split_size = len(X) // n_splits
        
        splits = []
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X)
            
            X_split = X.iloc[start_idx:end_idx]
            y_split = y.iloc[start_idx:end_idx]
            splits.append((X_split, y_split))
        
        return splits


class ModelTrainer:
    """V1模型训练器主类
    
    这是一个功能完整的模型训练器，支持：
    - 统一模型训练接口
    - 自动数据预处理
    - 交叉验证训练
    - 早停机制和模型保存
    - 分布式训练支持
    - 超参数自动调优
    - 模型性能监控
    - 训练进度跟踪
    - 错误处理和恢复
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """初始化训练器
        
        Args:
            config: 训练配置，如果为None则使用默认配置
        """
        self.config = config or TrainingConfig()
        self.preprocessor = DataPreprocessor(self.config)
        self.monitor = ModelPerformanceMonitor(self.config)
        self.tuner = HyperparameterTuner(self.config)
        self.distributed_trainer = DistributedTrainer(self.config)
        
        # 训练状态
        self.is_training = False
        self.current_epoch = 0
        self.start_time = None
        self.model = None
        self.training_history = []
        
        # 错误处理
        self.retry_count = 0
        
        # 初始化目录
        self._init_directories()
        
        logger.info("模型训练器初始化完成")
    
    def _init_directories(self) -> None:
        """初始化必要的目录"""
        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(self.config.log_path, exist_ok=True)
    
    def train(self, model: Any, X: pd.DataFrame, y: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> TrainingResult:
        """统一训练接口
        
        Args:
            model: 要训练的模型
            X: 训练特征数据
            y: 训练标签数据
            X_val: 验证特征数据（可选）
            y_val: 验证标签数据（可选）
            
        Returns:
            训练结果字典
        """
        if self.is_training:
            raise RuntimeError("训练正在进行中，请等待完成")
        
        try:
            self.is_training = True
            self.start_time = time.time()
            self.model = model
            
            logger.info("开始模型训练...")
            
            # 数据预处理
            X_processed, y_processed = self.preprocessor.fit_transform(X, y)
            
            # 数据分割
            if X_val is None or y_val is None:
                X_train, X_val_split, y_train, y_val_split = train_test_split(
                    X_processed, y_processed, 
                    test_size=self.config.validation_split,
                    random_state=self.config.random_state,
                    stratify=y_processed if len(np.unique(y_processed)) > 1 else None
                )
            else:
                X_train, y_train = X_processed, y_processed
                X_val_split, y_val_split = self.preprocessor.transform(X_val), y_val
            
            logger.info(f"训练集大小: {len(X_train)}")
            logger.info(f"验证集大小: {len(X_val_split)}")
            
            # 超参数调优
            if self.config.hyperparameter_tuning:
                model, tuning_results = self.tuner.tune(model, X_train, y_train)
                self.model = model
            else:
                tuning_results = {}
            
            # 交叉验证
            cv_results = {}
            if self.config.cv_folds > 1:
                cv_results = self._perform_cross_validation(model, X_train, y_train)
            
            # 分布式训练
            if self.config.distributed:
                trained_models = self.distributed_trainer.train_distributed(
                    model, X_train, y_train, self._single_train
                )
                # 使用第一个训练的模型（可以改为集成）
                model = trained_models[0]
            
            # 主训练循环
            training_results = self._training_loop(model, X_train, y_train, X_val_split, y_val_split)
            
            # 保存最佳模型
            if self.config.save_best_model:
                self._save_best_model()
            
            # 生成最终结果
            final_results = self._generate_final_results(
                model, training_results, cv_results, tuning_results
            )
            
            logger.info("模型训练完成")
            return final_results
            
        except Exception as e:
            logger.error(f"训练过程中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 错误恢复
            if self.retry_count < self.config.max_retries:
                self.retry_count += 1
                logger.info(f"尝试重试 ({self.retry_count}/{self.config.max_retries})...")
                time.sleep(self.config.retry_delay)
                return self.train(model, X, y, X_val, y_val)
            else:
                raise RuntimeError(f"训练失败，已达到最大重试次数: {str(e)}")
        
        finally:
            self.is_training = False
    
    def _perform_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """执行交叉验证"""
        logger.info(f"执行 {self.config.cv_folds} 折交叉验证")
        
        # 选择交叉验证策略
        if len(np.unique(y)) > 1:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                               random_state=self.config.random_state)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, 
                      random_state=self.config.random_state)
        
        # 执行交叉验证
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            "cv_scores": cv_scores.tolist(),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_folds": self.config.cv_folds
        }
        
        logger.info(f"交叉验证结果: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def _single_train(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        """单进程训练函数"""
        model.fit(X, y)
        return model
    
    def _training_loop(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """训练循环"""
        logger.info("开始训练循环...")
        
        best_val_score = float('-inf') if 'accuracy' in self.config.monitor_metrics else float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            
            try:
                # 训练模型
                model.fit(X_train, y_train)
                
                # 计算训练指标
                train_predictions = model.predict(X_train)
                train_loss = self._calculate_loss(y_train, train_predictions)
                train_accuracy = accuracy_score(y_train, train_predictions)
                
                # 计算验证指标
                val_predictions = model.predict(X_val)
                val_loss = self._calculate_loss(y_val, val_predictions)
                val_accuracy = accuracy_score(y_val, val_predictions)
                
                # 创建指标对象
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    learning_rate=self.config.learning_rate,
                    training_time=time.time() - epoch_start_time
                )
                
                # 更新监控器
                self.monitor.update(metrics)
                self.training_history.append(metrics)
                
                # 检查早停
                if self.config.early_stopping:
                    current_score = val_accuracy if 'accuracy' in self.config.monitor_metrics else -val_loss
                    
                    if current_score > best_val_score + self.config.early_stopping_min_delta:
                        best_val_score = current_score
                        patience_counter = 0
                        # 保存临时最佳模型
                        if self.config.save_best_model:
                            self._save_checkpoint(model, epoch)
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"早停触发，在第 {epoch} epoch")
                        break
                
                # 进度报告
                if epoch % self.config.progress_report_frequency == 0 or epoch == 1:
                    self._log_progress(metrics)
                
                # 定期保存
                if epoch % self.config.save_frequency == 0 and self.config.save_best_model:
                    self._save_checkpoint(model, epoch)
                
                self.current_epoch = epoch
                
            except Exception as e:
                logger.error(f"Epoch {epoch} 训练失败: {str(e)}")
                continue
        
        logger.info("训练循环完成")
        return {
            "total_epochs": self.current_epoch,
            "best_epoch": self.monitor.get_best_model_epoch(),
            "final_metrics": self.training_history[-1] if self.training_history else None
        }
    
    def _calculate_loss(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """计算损失函数值"""
        try:
            # 对于分类问题，使用1 - accuracy作为损失
            if len(np.unique(y_true)) <= 10:  # 假设分类问题
                return 1.0 - accuracy_score(y_true, y_pred)
            else:  # 回归问题
                return mean_squared_error(y_true, y_pred)
        except:
            return 0.0
    
    def _log_progress(self, metrics: TrainingMetrics) -> None:
        """记录训练进度"""
        if self.config.verbose:
            logger.info(
                f"Epoch {metrics.epoch}/{self.config.epochs} - "
                f"Train Loss: {metrics.train_loss:.4f}, "
                f"Train Acc: {metrics.train_accuracy:.4f}, "
                f"Val Loss: {metrics.val_loss:.4f}, "
                f"Val Acc: {metrics.val_accuracy:.4f}, "
                f"Time: {metrics.training_time:.2f}s"
            )
    
    def _save_checkpoint(self, model: Any, epoch: int) -> None:
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.config.model_save_path, 
            f"checkpoint_epoch_{epoch}.pkl"
        )
        try:
            joblib.dump(model, checkpoint_path)
            logger.debug(f"检查点已保存: {checkpoint_path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}")
    
    def _save_best_model(self) -> None:
        """保存最佳模型"""
        if self.model is None:
            return
        
        best_epoch = self.monitor.get_best_model_epoch()
        if best_epoch > 0:
            model_path = os.path.join(
                self.config.model_save_path, 
                f"best_model_epoch_{best_epoch}.pkl"
            )
            try:
                joblib.dump(self.model, model_path)
                logger.info(f"最佳模型已保存: {model_path}")
            except Exception as e:
                logger.error(f"保存最佳模型失败: {str(e)}")
    
    def _generate_final_results(self, model: Any, training_results: Dict,
                               cv_results: Dict, tuning_results: Dict) -> TrainingResult:
        """生成最终训练结果"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        results = {
            "success": True,
            "model": model,
            "training_time": total_time,
            "total_epochs": training_results.get("total_epochs", 0),
            "best_epoch": training_results.get("best_epoch", 0),
            "monitor_summary": self.monitor.get_summary(),
            "training_history": [asdict(metric) for metric in self.training_history],
            "preprocessor": self.preprocessor,
            "config": asdict(self.config)
        }
        
        # 添加交叉验证结果
        if cv_results:
            results["cross_validation"] = cv_results
        
        # 添加超参数调优结果
        if tuning_results:
            results["hyperparameter_tuning"] = tuning_results
        
        # 保存训练日志
        if self.config.save_logs:
            self._save_training_logs(results)
        
        return results
    
    def _save_training_logs(self, results: TrainingResult) -> None:
        """保存训练日志"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.config.log_path, f"training_log_{timestamp}.json")
            
            # 移除不可序列化的对象
            log_data = {k: v for k, v in results.items() 
                       if k not in ["model", "preprocessor"]}
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"训练日志已保存: {log_file}")
        except Exception as e:
            logger.error(f"保存训练日志失败: {str(e)}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """使用训练好的模型进行预测"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 train 方法")
        
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """使用训练好的模型进行概率预测"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 train 方法")
        
        if not hasattr(self.model, 'predict_proba'):
            raise RuntimeError("该模型不支持概率预测")
        
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> MetricType:
        """评估模型性能"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 train 方法")
        
        X_processed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_processed)
        
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y, predictions, average='weighted', zero_division=0)
        }
        
        # 如果是二分类，添加AUC
        if len(np.unique(y)) == 2:
            try:
                predictions_proba = self.model.predict_proba(X_processed)[:, 1]
                metrics["auc"] = roc_auc_score(y, predictions_proba)
            except:
                pass
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """保存训练好的模型"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 train 方法")
        
        try:
            model_data = {
                "model": self.model,
                "preprocessor": self.preprocessor,
                "config": asdict(self.config),
                "training_history": [asdict(metric) for metric in self.training_history]
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"模型已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """加载训练好的模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.preprocessor = model_data["preprocessor"]
            self.config = TrainingConfig(**model_data["config"])
            self.training_history = model_data.get("training_history", [])
            
            logger.info(f"模型已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def get_training_progress(self) -> Dict[str, Any]:
        """获取训练进度信息"""
        if not self.is_training and not self.training_history:
            return {"status": "not_started"}
        
        if self.is_training:
            progress = {
                "status": "training",
                "current_epoch": self.current_epoch,
                "total_epochs": self.config.epochs,
                "progress_percentage": (self.current_epoch / self.config.epochs) * 100,
                "elapsed_time": time.time() - self.start_time if self.start_time else 0,
                "estimated_remaining": self._estimate_remaining_time()
            }
        else:
            progress = {
                "status": "completed",
                "total_epochs": len(self.training_history),
                "total_time": sum(m.training_time for m in self.training_history),
                "best_epoch": self.monitor.get_best_model_epoch(),
                "final_metrics": asdict(self.training_history[-1]) if self.training_history else None
            }
        
        return progress
    
    def _estimate_remaining_time(self) -> float:
        """估算剩余时间"""
        if not self.is_training or not self.training_history or self.start_time is None:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        avg_time_per_epoch = elapsed_time / self.current_epoch
        remaining_epochs = self.config.epochs - self.current_epoch
        
        return avg_time_per_epoch * remaining_epochs
    
    def stop_training(self) -> None:
        """停止训练"""
        if self.is_training:
            self.is_training = False
            logger.info("训练已停止")
        else:
            logger.warning("当前没有正在进行的训练")


# 测试用例和示例
def create_sample_data(n_samples: int = 1000, n_features: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """创建示例数据"""
    np.random.seed(42)
    
    # 生成特征
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 生成标签（二分类）
    y = pd.Series(
        (X.iloc[:, 0] + X.iloc[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int),
        name='target'
    )
    
    # 添加一些缺失值
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    X.iloc[missing_indices, 0] = np.nan
    
    return X, y


def test_model_trainer():
    """测试模型训练器"""
    print("=== V1模型训练器测试 ===")
    
    # 创建示例数据
    print("1. 创建示例数据...")
    X, y = create_sample_data(n_samples=500, n_features=5)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布:\n{y.value_counts()}")
    
    # 创建配置
    config = TrainingConfig(
        epochs=50,
        batch_size=32,
        early_stopping=True,
        early_stopping_patience=5,
        cv_folds=3,
        hyperparameter_tuning=True,
        save_best_model=True,
        verbose=True
    )
    
    # 初始化训练器
    print("\n2. 初始化训练器...")
    trainer = ModelTrainer(config)
    
    # 创建模型
    print("\n3. 创建模型...")
    model = RandomForestClassifier(random_state=42)
    
    # 训练模型
    print("\n4. 开始训练...")
    results = trainer.train(model, X, y)
    
    # 显示结果
    print("\n5. 训练结果:")
    print(f"训练成功: {results['success']}")
    print(f"训练时间: {results['training_time']:.2f}秒")
    print(f"总epochs: {results['total_epochs']}")
    print(f"最佳epoch: {results['best_epoch']}")
    
    if 'cross_validation' in results:
        cv = results['cross_validation']
        print(f"交叉验证: {cv['cv_mean']:.4f} (+/- {cv['cv_std'] * 2:.4f})")
    
    if 'hyperparameter_tuning' in results:
        tuning = results['hyperparameter_tuning']
        print(f"最佳参数: {tuning['best_params']}")
        print(f"最佳分数: {tuning['best_score']:.4f}")
    
    # 测试预测
    print("\n6. 测试预测...")
    predictions = trainer.predict(X[:10])
    print(f"前10个预测结果: {predictions}")
    
    # 测试评估
    print("\n7. 模型评估...")
    metrics = trainer.evaluate(X, y)
    print(f"评估指标: {metrics}")
    
    # 测试保存和加载
    print("\n8. 测试模型保存和加载...")
    trainer.save_model("./test_model.pkl")
    
    new_trainer = ModelTrainer()
    new_trainer.load_model("./test_model.pkl")
    
    new_predictions = new_trainer.predict(X[:10])
    print(f"加载模型预测结果: {new_predictions}")
    print(f"预测一致性: {np.array_equal(predictions, new_predictions)}")
    
    # 获取训练进度
    print("\n9. 训练进度信息...")
    progress = trainer.get_training_progress()
    print(f"训练进度: {progress}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    # 运行测试
    test_model_trainer()