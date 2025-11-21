"""
F9学习状态聚合器
实现多模块学习状态融合、评估、验证、优化等功能
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningStatus(Enum):
    """学习状态枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    NORMAL = "normal"
    POOR = "poor"
    CRITICAL = "critical"


class LearningPriority(Enum):
    """学习优先级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


@dataclass
class ModuleLearningState:
    """模块学习状态数据结构"""
    module_id: str
    module_name: str
    accuracy: float
    loss: float
    convergence_rate: float
    learning_efficiency: float
    stability: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedLearningState:
    """聚合学习状态数据结构"""
    overall_score: float
    status: LearningStatus
    priority: LearningPriority
    module_states: List[ModuleLearningState]
    consistency_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class LearningHistoryRecord:
    """学习历史记录数据结构"""
    timestamp: datetime
    aggregated_state: AggregatedLearningState
    performance_metrics: Dict[str, float]
    anomalies_detected: List[str]


class LearningStateAggregator:
    """F9学习状态聚合器"""
    
    def __init__(self, 
                 history_window: int = 100,
                 consistency_threshold: float = 0.8,
                 performance_weights: Optional[Dict[str, float]] = None):
        """
        初始化学习状态聚合器
        
        Args:
            history_window: 历史记录窗口大小
            consistency_threshold: 一致性阈值
            performance_weights: 性能指标权重配置
        """
        self.history_window = history_window
        self.consistency_threshold = consistency_threshold
        self.performance_weights = performance_weights or {
            'accuracy': 0.3,
            'convergence_rate': 0.25,
            'learning_efficiency': 0.2,
            'stability': 0.15,
            'loss': 0.1
        }
        
        # 历史数据存储
        self.learning_history: deque = deque(maxlen=history_window)
        self.module_states_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_window)
        )
        
        # 状态管理
        self.current_state: Optional[AggregatedLearningState] = None
        self.alerts_history: List[Dict] = []
        self.optimization_history: List[Dict] = []
        
        # 异常检测配置
        self.anomaly_detection_config = {
            'z_score_threshold': 2.5,
            'iqr_multiplier': 1.5,
            'window_size': 20
        }
        
        # 预警配置
        self.alert_config = {
            'performance_drop_threshold': 0.15,
            'consistency_threshold': 0.7,
            'stability_threshold': 0.6
        }
        
        logger.info("学习状态聚合器初始化完成")
    
    def aggregate_learning_states(self, 
                                module_states: List[ModuleLearningState]) -> AggregatedLearningState:
        """
        聚合多个模块的学习状态
        
        Args:
            module_states: 模块学习状态列表
            
        Returns:
            AggregatedLearningState: 聚合后的学习状态
        """
        if not module_states:
            raise ValueError("模块状态列表不能为空")
        
        # 1. 计算综合性能分数
        overall_score = self._calculate_overall_performance_score(module_states)
        
        # 2. 确定学习状态
        status = self._determine_learning_status(overall_score)
        
        # 3. 确定学习优先级
        priority = self._determine_learning_priority(module_states, overall_score)
        
        # 4. 计算一致性分数
        consistency_score = self._calculate_consistency_score(module_states)
        
        # 5. 生成优化建议
        recommendations = self._generate_optimization_recommendations(
            module_states, overall_score, consistency_score
        )
        
        # 6. 创建聚合状态
        aggregated_state = AggregatedLearningState(
            overall_score=overall_score,
            status=status,
            priority=priority,
            module_states=module_states,
            consistency_score=consistency_score,
            recommendations=recommendations
        )
        
        # 7. 更新历史记录
        self._update_learning_history(aggregated_state, module_states)
        
        # 8. 更新当前状态
        self.current_state = aggregated_state
        
        logger.info(f"学习状态聚合完成 - 总体分数: {overall_score:.3f}, 状态: {status.value}")
        
        return aggregated_state
    
    def _calculate_overall_performance_score(self, 
                                           module_states: List[ModuleLearningState]) -> float:
        """
        计算综合性能分数
        
        Args:
            module_states: 模块学习状态列表
            
        Returns:
            float: 综合性能分数 [0, 1]
        """
        if not module_states:
            return 0.0
        
        scores = []
        
        for module in module_states:
            # 标准化各项指标
            normalized_metrics = self._normalize_module_metrics(module)
            
            # 计算加权分数
            weighted_score = (
                normalized_metrics['accuracy'] * self.performance_weights['accuracy'] +
                normalized_metrics['convergence_rate'] * self.performance_weights['convergence_rate'] +
                normalized_metrics['learning_efficiency'] * self.performance_weights['learning_efficiency'] +
                normalized_metrics['stability'] * self.performance_weights['stability'] +
                (1 - normalized_metrics['loss']) * self.performance_weights['loss']  # 损失越小越好
            )
            
            scores.append(weighted_score)
        
        # 返回所有模块的平均分数
        return np.mean(scores)
    
    def _normalize_module_metrics(self, module: ModuleLearningState) -> Dict[str, float]:
        """
        标准化模块指标
        
        Args:
            module: 模块学习状态
            
        Returns:
            Dict[str, float]: 标准化后的指标
        """
        # 使用历史数据或默认值进行标准化
        history = self.module_states_history.get(module.module_id, [])
        
        if len(history) < 5:
            # 历史数据不足，使用简单标准化
            return {
                'accuracy': np.clip(module.accuracy, 0, 1),
                'convergence_rate': np.clip(module.convergence_rate, 0, 1),
                'learning_efficiency': np.clip(module.learning_efficiency, 0, 1),
                'stability': np.clip(module.stability, 0, 1),
                'loss': np.clip(module.loss, 0, 2) / 2  # 假设损失范围为[0, 2]
            }
        
        # 使用历史数据进行标准化
        accuracies = [m.accuracy for m in history]
        convergence_rates = [m.convergence_rate for m in history]
        learning_efficiencies = [m.learning_efficiency for m in history]
        stabilities = [m.stability for m in history]
        losses = [m.loss for m in history]
        
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform([
            [module.accuracy],
            [module.convergence_rate],
            [module.learning_efficiency],
            [module.stability],
            [module.loss]
        ])
        
        return {
            'accuracy': normalized_data[0][0],
            'convergence_rate': normalized_data[1][0],
            'learning_efficiency': normalized_data[2][0],
            'stability': normalized_data[3][0],
            'loss': normalized_data[4][0]
        }
    
    def _determine_learning_status(self, overall_score: float) -> LearningStatus:
        """
        确定学习状态
        
        Args:
            overall_score: 综合性能分数
            
        Returns:
            LearningStatus: 学习状态
        """
        if overall_score >= 0.9:
            return LearningStatus.EXCELLENT
        elif overall_score >= 0.75:
            return LearningStatus.GOOD
        elif overall_score >= 0.6:
            return LearningStatus.NORMAL
        elif overall_score >= 0.4:
            return LearningStatus.POOR
        else:
            return LearningStatus.CRITICAL
    
    def _determine_learning_priority(self, 
                                   module_states: List[ModuleLearningState],
                                   overall_score: float) -> LearningPriority:
        """
        确定学习优先级
        
        Args:
            module_states: 模块学习状态列表
            overall_score: 综合性能分数
            
        Returns:
            LearningPriority: 学习优先级
        """
        # 基于综合分数和模块状态确定优先级
        if overall_score < 0.4:
            return LearningPriority.CRITICAL
        elif overall_score < 0.6:
            return LearningPriority.HIGH
        elif overall_score < 0.8:
            return LearningPriority.MEDIUM
        else:
            return LearningPriority.LOW
    
    def _calculate_consistency_score(self, module_states: List[ModuleLearningState]) -> float:
        """
        计算学习一致性分数
        
        Args:
            module_states: 模块学习状态列表
            
        Returns:
            float: 一致性分数 [0, 1]
        """
        if len(module_states) < 2:
            return 1.0
        
        # 提取性能指标
        accuracies = [m.accuracy for m in module_states]
        convergence_rates = [m.convergence_rate for m in module_states]
        learning_efficiencies = [m.learning_efficiency for m in module_states]
        
        # 计算变异系数（标准差/均值）
        cv_accuracy = np.std(accuracies) / (np.mean(accuracies) + 1e-8)
        cv_convergence = np.std(convergence_rates) / (np.mean(convergence_rates) + 1e-8)
        cv_efficiency = np.std(learning_efficiencies) / (np.mean(learning_efficiencies) + 1e-8)
        
        # 一致性分数 = 1 - 平均变异系数（变异系数越小，一致性越好）
        avg_cv = (cv_accuracy + cv_convergence + cv_efficiency) / 3
        consistency_score = 1 / (1 + avg_cv)
        
        return np.clip(consistency_score, 0, 1)
    
    def validate_learning_consistency(self, 
                                    module_states: List[ModuleLearningState]) -> Dict[str, Any]:
        """
        验证学习一致性
        
        Args:
            module_states: 模块学习状态列表
            
        Returns:
            Dict[str, Any]: 一致性验证结果
        """
        validation_results = {
            'is_consistent': False,
            'consistency_score': 0.0,
            'validation_details': {},
            'anomalies': [],
            'recommendations': []
        }
        
        # 计算一致性分数
        consistency_score = self._calculate_consistency_score(module_states)
        validation_results['consistency_score'] = consistency_score
        
        # 判断是否一致
        validation_results['is_consistent'] = consistency_score >= self.consistency_threshold
        
        # 详细验证
        validation_results['validation_details'] = self._detailed_consistency_validation(module_states)
        
        # 异常检测
        validation_results['anomalies'] = self._detect_anomalies(module_states)
        
        # 生成建议
        if not validation_results['is_consistent']:
            validation_results['recommendations'] = self._generate_consistency_recommendations(
                module_states, consistency_score
            )
        
        logger.info(f"学习一致性验证完成 - 一致性分数: {consistency_score:.3f}")
        
        return validation_results
    
    def _detailed_consistency_validation(self, module_states: List[ModuleLearningState]) -> Dict[str, Any]:
        """
        详细一致性验证
        
        Args:
            module_states: 模块学习状态列表
            
        Returns:
            Dict[str, Any]: 详细验证结果
        """
        if len(module_states) < 2:
            return {'message': '模块数量不足，无法进行详细验证'}
        
        details = {}
        
        # 性能指标一致性分析
        metrics = ['accuracy', 'convergence_rate', 'learning_efficiency', 'stability']
        
        for metric in metrics:
            values = [getattr(m, metric) for m in module_states]
            
            # 统计描述
            details[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values),
                'cv': np.std(values) / (np.mean(values) + 1e-8)
            }
            
            # 异常值检测
            outliers = self._detect_outliers(values)
            if outliers:
                details[metric]['outliers'] = outliers
        
        return details
    
    def _detect_outliers(self, values: List[float]) -> List[int]:
        """
        检测异常值
        
        Args:
            values: 数值列表
            
        Returns:
            List[int]: 异常值索引列表
        """
        if len(values) < 3:
            return []
        
        # 使用IQR方法检测异常值
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers
    
    def _detect_anomalies(self, module_states: List[ModuleLearningState]) -> List[str]:
        """
        检测学习异常
        
        Args:
            module_states: 模块学习状态列表
            
        Returns:
            List[str]: 检测到的异常列表
        """
        anomalies = []
        
        for module in module_states:
            module_anomalies = []
            
            # 检查性能指标异常
            if module.accuracy < 0.3:
                module_anomalies.append(f"模块 {module.module_name} 准确率过低 ({module.accuracy:.3f})")
            
            if module.loss > 1.5:
                module_anomalies.append(f"模块 {module.module_name} 损失过高 ({module.loss:.3f})")
            
            if module.stability < 0.4:
                module_anomalies.append(f"模块 {module.module_name} 稳定性不足 ({module.stability:.3f})")
            
            if module.learning_efficiency < 0.3:
                module_anomalies.append(f"模块 {module.module_name} 学习效率低下 ({module.learning_efficiency:.3f})")
            
            if module_anomalies:
                anomalies.extend(module_anomalies)
        
        return anomalies
    
    def _generate_consistency_recommendations(self, 
                                            module_states: List[ModuleLearningState],
                                            consistency_score: float) -> List[str]:
        """
        生成一致性改进建议
        
        Args:
            module_states: 模块学习状态列表
            consistency_score: 一致性分数
            
        Returns:
            List[str]: 改进建议列表
        """
        recommendations = []
        
        if consistency_score < 0.5:
            recommendations.append("模块间性能差异过大，建议进行性能平衡调整")
        
        # 基于具体指标生成建议
        metrics = ['accuracy', 'convergence_rate', 'learning_efficiency', 'stability']
        
        for metric in metrics:
            values = [getattr(m, metric) for m in module_states]
            cv = np.std(values) / (np.mean(values) + 1e-8)
            
            if cv > 0.3:  # 变异系数过大
                if metric == 'accuracy':
                    recommendations.append("模块准确率差异较大，建议统一训练策略或数据质量")
                elif metric == 'convergence_rate':
                    recommendations.append("模块收敛速度不一致，建议调整学习率或优化器参数")
                elif metric == 'learning_efficiency':
                    recommendations.append("模块学习效率差异明显，建议优化网络架构或训练流程")
                elif metric == 'stability':
                    recommendations.append("模块稳定性差异较大，建议增加正则化或调整训练策略")
        
        return recommendations
    
    def prioritize_learning_modules(self, 
                                  module_states: List[ModuleLearningState]) -> List[Tuple[str, LearningPriority, float]]:
        """
        学习模块优先级排序
        
        Args:
            module_states: 模块学习状态列表
            
        Returns:
            List[Tuple[str, LearningPriority, float]]: (模块ID, 优先级, 优先级分数) 列表
        """
        module_priorities = []
        
        for module in module_states:
            # 计算优先级分数
            priority_score = self._calculate_priority_score(module)
            
            # 确定优先级
            if priority_score >= 0.8:
                priority = LearningPriority.CRITICAL
            elif priority_score >= 0.6:
                priority = LearningPriority.HIGH
            elif priority_score >= 0.4:
                priority = LearningPriority.MEDIUM
            else:
                priority = LearningPriority.LOW
            
            module_priorities.append((module.module_id, priority, priority_score))
        
        # 按优先级分数降序排序
        module_priorities.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"学习模块优先级排序完成，共 {len(module_priorities)} 个模块")
        
        return module_priorities
    
    def _calculate_priority_score(self, module: ModuleLearningState) -> float:
        """
        计算模块优先级分数
        
        Args:
            module: 模块学习状态
            
        Returns:
            float: 优先级分数 [0, 1]
        """
        # 综合考虑性能指标
        performance_score = (
            module.accuracy * 0.3 +
            (1 - module.loss / 2) * 0.2 +  # 损失归一化
            module.convergence_rate * 0.2 +
            module.learning_efficiency * 0.2 +
            module.stability * 0.1
        )
        
        # 考虑时间因素（越新的状态权重越高）
        time_factor = 1.0  # 可以根据实际需要调整
        
        return performance_score * time_factor
    
    def record_learning_state(self, 
                            aggregated_state: AggregatedLearningState,
                            module_states: List[ModuleLearningState],
                            performance_metrics: Optional[Dict[str, float]] = None):
        """
        记录学习状态到历史
        
        Args:
            aggregated_state: 聚合学习状态
            module_states: 模块学习状态列表
            performance_metrics: 性能指标
        """
        # 创建历史记录
        history_record = LearningHistoryRecord(
            timestamp=datetime.now(),
            aggregated_state=aggregated_state,
            performance_metrics=performance_metrics or {},
            anomalies_detected=self._detect_anomalies(module_states)
        )
        
        # 添加到历史记录
        self.learning_history.append(history_record)
        
        # 更新模块状态历史
        for module in module_states:
            self.module_states_history[module.module_id].append(module)
        
        logger.info("学习状态记录已保存")
    
    def _update_learning_history(self, 
                               aggregated_state: AggregatedLearningState,
                               module_states: List[ModuleLearningState]):
        """
        更新学习历史记录
        
        Args:
            aggregated_state: 聚合学习状态
            module_states: 模块学习状态列表
        """
        performance_metrics = {
            'overall_score': aggregated_state.overall_score,
            'consistency_score': aggregated_state.consistency_score,
            'module_count': len(module_states)
        }
        
        self.record_learning_state(aggregated_state, module_states, performance_metrics)
    
    def generate_learning_report(self, 
                               time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        生成学习状态报告
        
        Args:
            time_range: 时间范围 (开始时间, 结束时间)
            
        Returns:
            Dict[str, Any]: 学习状态报告
        """
        # 筛选时间范围内的记录
        filtered_history = self._filter_history_by_time_range(time_range)
        
        if not filtered_history:
            return {'message': '指定时间范围内无历史记录'}
        
        report = {
            'report_generation_time': datetime.now(),
            'time_range': time_range or ('全部历史', '全部历史'),
            'summary': self._generate_summary_statistics(filtered_history),
            'performance_trends': self._analyze_performance_trends(filtered_history),
            'consistency_analysis': self._analyze_consistency_trends(filtered_history),
            'module_performance': self._analyze_module_performance(filtered_history),
            'anomaly_summary': self._summarize_anomalies(filtered_history),
            'recommendations': self._generate_report_recommendations(filtered_history)
        }
        
        logger.info("学习状态报告生成完成")
        
        return report
    
    def _filter_history_by_time_range(self, 
                                    time_range: Optional[Tuple[datetime, datetime]]) -> List[LearningHistoryRecord]:
        """
        按时间范围筛选历史记录
        
        Args:
            time_range: 时间范围
            
        Returns:
            List[LearningHistoryRecord]: 筛选后的历史记录
        """
        if not time_range:
            return list(self.learning_history)
        
        start_time, end_time = time_range
        filtered_records = []
        
        for record in self.learning_history:
            if start_time <= record.timestamp <= end_time:
                filtered_records.append(record)
        
        return filtered_records
    
    def _generate_summary_statistics(self, history: List[LearningHistoryRecord]) -> Dict[str, Any]:
        """
        生成汇总统计信息
        
        Args:
            history: 历史记录列表
            
        Returns:
            Dict[str, Any]: 汇总统计信息
        """
        if not history:
            return {}
        
        overall_scores = [record.aggregated_state.overall_score for record in history]
        consistency_scores = [record.aggregated_state.consistency_score for record in history]
        
        return {
            'total_records': len(history),
            'average_overall_score': np.mean(overall_scores),
            'max_overall_score': np.max(overall_scores),
            'min_overall_score': np.min(overall_scores),
            'average_consistency_score': np.mean(consistency_scores),
            'max_consistency_score': np.max(consistency_scores),
            'min_consistency_score': np.min(consistency_scores),
            'score_trend': self._calculate_trend(overall_scores)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        计算数值趋势
        
        Args:
            values: 数值列表
            
        Returns:
            str: 趋势描述
        """
        if len(values) < 2:
            return "数据不足"
        
        # 使用线性回归计算趋势
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        if abs(slope) < 0.001:
            return "稳定"
        elif slope > 0:
            return "上升"
        else:
            return "下降"
    
    def _analyze_performance_trends(self, history: List[LearningHistoryRecord]) -> Dict[str, Any]:
        """
        分析性能趋势
        
        Args:
            history: 历史记录列表
            
        Returns:
            Dict[str, Any]: 性能趋势分析
        """
        if len(history) < 2:
            return {'message': '历史数据不足，无法分析趋势'}
        
        # 提取性能数据
        timestamps = [record.timestamp for record in history]
        overall_scores = [record.aggregated_state.overall_score for record in history]
        
        # 计算趋势
        trend_analysis = {
            'overall_trend': self._calculate_trend(overall_scores),
            'volatility': np.std(overall_scores),
            'recent_performance': overall_scores[-5:] if len(overall_scores) >= 5 else overall_scores
        }
        
        return trend_analysis
    
    def _analyze_consistency_trends(self, history: List[LearningHistoryRecord]) -> Dict[str, Any]:
        """
        分析一致性趋势
        
        Args:
            history: 历史记录列表
            
        Returns:
            Dict[str, Any]: 一致性趋势分析
        """
        if not history:
            return {}
        
        consistency_scores = [record.aggregated_state.consistency_score for record in history]
        
        return {
            'average_consistency': np.mean(consistency_scores),
            'consistency_trend': self._calculate_trend(consistency_scores),
            'consistency_volatility': np.std(consistency_scores),
            'consistency_violations': sum(1 for score in consistency_scores if score < self.consistency_threshold)
        }
    
    def _analyze_module_performance(self, history: List[LearningHistoryRecord]) -> Dict[str, Any]:
        """
        分析模块性能
        
        Args:
            history: 历史记录列表
            
        Returns:
            Dict[str, Any]: 模块性能分析
        """
        module_performance = {}
        
        # 收集所有模块ID
        all_module_ids = set()
        for record in history:
            for module in record.aggregated_state.module_states:
                all_module_ids.add(module.module_id)
        
        # 分析每个模块的性能
        for module_id in all_module_ids:
            module_scores = []
            module_stabilities = []
            
            for record in history:
                for module in record.aggregated_state.module_states:
                    if module.module_id == module_id:
                        module_scores.append(module.accuracy)
                        module_stabilities.append(module.stability)
            
            if module_scores:
                module_performance[module_id] = {
                    'average_accuracy': np.mean(module_scores),
                    'accuracy_volatility': np.std(module_scores),
                    'average_stability': np.mean(module_stabilities),
                    'stability_trend': self._calculate_trend(module_stabilities)
                }
        
        return module_performance
    
    def _summarize_anomalies(self, history: List[LearningHistoryRecord]) -> Dict[str, Any]:
        """
        汇总异常信息
        
        Args:
            history: 历史记录列表
            
        Returns:
            Dict[str, Any]: 异常汇总信息
        """
        all_anomalies = []
        for record in history:
            all_anomalies.extend(record.anomalies_detected)
        
        anomaly_counts = defaultdict(int)
        for anomaly in all_anomalies:
            anomaly_counts[anomaly] += 1
        
        return {
            'total_anomalies': len(all_anomalies),
            'unique_anomalies': len(set(all_anomalies)),
            'most_common_anomalies': dict(sorted(anomaly_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _generate_report_recommendations(self, history: List[LearningHistoryRecord]) -> List[str]:
        """
        生成报告建议
        
        Args:
            history: 历史记录列表
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        if not history:
            return ["历史数据不足，无法生成具体建议"]
        
        # 基于汇总统计生成建议
        summary = self._generate_summary_statistics(history)
        
        if summary.get('average_overall_score', 0) < 0.6:
            recommendations.append("整体学习性能偏低，建议优化模型架构或训练策略")
        
        if summary.get('average_consistency_score', 1) < self.consistency_threshold:
            recommendations.append("模块间一致性不足，建议进行性能平衡调整")
        
        if summary.get('score_trend') == '下降':
            recommendations.append("学习性能呈下降趋势，建议检查训练过程和参数设置")
        
        return recommendations
    
    def check_learning_alerts(self) -> List[Dict[str, Any]]:
        """
        检查学习状态预警
        
        Returns:
            List[Dict[str, Any]]: 预警列表
        """
        alerts = []
        
        if not self.current_state:
            return alerts
        
        # 检查性能下降
        if len(self.learning_history) >= 2:
            current_score = self.current_state.overall_score
            previous_score = list(self.learning_history)[-2].aggregated_state.overall_score
            
            if current_score < previous_score * (1 - self.alert_config['performance_drop_threshold']):
                alerts.append({
                    'type': 'performance_drop',
                    'level': 'warning',
                    'message': f'学习性能下降 {((previous_score - current_score) / previous_score * 100):.1f}%',
                    'timestamp': datetime.now()
                })
        
        # 检查一致性警告
        if self.current_state.consistency_score < self.alert_config['consistency_threshold']:
            alerts.append({
                'type': 'consistency_warning',
                'level': 'warning',
                'message': f'模块间一致性不足 ({self.current_state.consistency_score:.3f})',
                'timestamp': datetime.now()
            })
        
        # 检查稳定性警告
        for module in self.current_state.module_states:
            if module.stability < self.alert_config['stability_threshold']:
                alerts.append({
                    'type': 'stability_warning',
                    'level': 'warning',
                    'message': f'模块 {module.module_name} 稳定性不足 ({module.stability:.3f})',
                    'timestamp': datetime.now()
                })
        
        # 检查关键状态
        if self.current_state.status == LearningStatus.CRITICAL:
            alerts.append({
                'type': 'critical_status',
                'level': 'critical',
                'message': '学习状态处于临界状态，需要立即关注',
                'timestamp': datetime.now()
            })
        
        # 保存预警历史
        self.alerts_history.extend(alerts)
        
        logger.info(f"学习状态预警检查完成，发现 {len(alerts)} 个预警")
        
        return alerts
    
    def generate_optimization_suggestions(self, 
                                        module_states: List[ModuleLearningState]) -> List[Dict[str, Any]]:
        """
        生成学习状态优化建议
        
        Args:
            module_states: 模块学习状态列表
            
        Returns:
            List[Dict[str, Any]]: 优化建议列表
        """
        suggestions = []
        
        # 分析每个模块的性能
        for module in module_states:
            module_suggestions = self._analyze_module_optimization(module)
            suggestions.extend(module_suggestions)
        
        # 全局优化建议
        global_suggestions = self._generate_global_optimization_suggestions(module_states)
        suggestions.extend(global_suggestions)
        
        # 保存优化历史
        optimization_record = {
            'timestamp': datetime.now(),
            'module_count': len(module_states),
            'suggestions': suggestions
        }
        self.optimization_history.append(optimization_record)
        
        logger.info(f"优化建议生成完成，共 {len(suggestions)} 条建议")
        
        return suggestions
    
    def _analyze_module_optimization(self, module: ModuleLearningState) -> List[Dict[str, Any]]:
        """
        分析模块优化建议
        
        Args:
            module: 模块学习状态
            
        Returns:
            List[Dict[str, Any]]: 模块优化建议列表
        """
        suggestions = []
        
        # 准确率优化建议
        if module.accuracy < 0.7:
            suggestions.append({
                'module_id': module.module_id,
                'type': 'accuracy_improvement',
                'priority': 'high',
                'suggestion': '提高模型准确率',
                'actions': [
                    '增加训练数据量',
                    '优化网络架构',
                    '调整超参数',
                    '使用数据增强技术'
                ]
            })
        
        # 收敛速度优化建议
        if module.convergence_rate < 0.6:
            suggestions.append({
                'module_id': module.module_id,
                'type': 'convergence_improvement',
                'priority': 'medium',
                'suggestion': '加快模型收敛',
                'actions': [
                    '调整学习率策略',
                    '优化梯度更新算法',
                    '改进损失函数',
                    '使用预训练模型'
                ]
            })
        
        # 学习效率优化建议
        if module.learning_efficiency < 0.6:
            suggestions.append({
                'module_id': module.module_id,
                'type': 'efficiency_improvement',
                'priority': 'medium',
                'suggestion': '提升学习效率',
                'actions': [
                    '优化特征提取',
                    '减少模型复杂度',
                    '使用更高效的优化器',
                    '并行化训练过程'
                ]
            })
        
        # 稳定性优化建议
        if module.stability < 0.6:
            suggestions.append({
                'module_id': module.module_id,
                'type': 'stability_improvement',
                'priority': 'high',
                'suggestion': '增强模型稳定性',
                'actions': [
                    '增加正则化',
                    '使用Dropout技术',
                    '调整批次大小',
                    '改进数据预处理'
                ]
            })
        
        return suggestions
    
    def _generate_global_optimization_suggestions(self, 
                                                module_states: List[ModuleLearningState]) -> List[Dict[str, Any]]:
        """
        生成全局优化建议
        
        Args:
            module_states: 模块学习状态列表
            
        Returns:
            List[Dict[str, Any]]: 全局优化建议列表
        """
        suggestions = []
        
        if len(module_states) < 2:
            return suggestions
        
        # 一致性优化
        consistency_score = self._calculate_consistency_score(module_states)
        if consistency_score < self.consistency_threshold:
            suggestions.append({
                'type': 'consistency_optimization',
                'priority': 'high',
                'suggestion': '提升模块间一致性',
                'actions': [
                    '统一训练策略',
                    '平衡各模块性能',
                    '标准化数据处理流程',
                    '协调超参数设置'
                ]
            })
        
        # 资源分配优化
        avg_performance = np.mean([module.accuracy for module in module_states])
        if avg_performance < 0.7:
            suggestions.append({
                'type': 'resource_optimization',
                'priority': 'medium',
                'suggestion': '优化资源分配',
                'actions': [
                    '重新分配计算资源',
                    '优化训练时间安排',
                    '改进并行策略',
                    '调整批次大小配置'
                ]
            })
        
        # 整体架构优化
        performance_variance = np.var([module.accuracy for module in module_states])
        if performance_variance > 0.1:
            suggestions.append({
                'type': 'architecture_optimization',
                'priority': 'medium',
                'suggestion': '优化整体架构',
                'actions': [
                    '重新设计模块接口',
                    '优化信息传递机制',
                    '改进协作策略',
                    '统一评估标准'
                ]
            })
        
        return suggestions
    
    def _generate_optimization_recommendations(self, 
                                             module_states: List[ModuleLearningState],
                                             overall_score: float,
                                             consistency_score: float) -> List[str]:
        """
        生成优化建议（简化版本）
        
        Args:
            module_states: 模块学习状态列表
            overall_score: 综合性能分数
            consistency_score: 一致性分数
            
        Returns:
            List[str]: 优化建议列表
        """
        recommendations = []
        
        if overall_score < 0.6:
            recommendations.append("整体性能需要提升，建议优化模型架构和训练策略")
        
        if consistency_score < self.consistency_threshold:
            recommendations.append("模块间一致性不足，建议进行性能平衡调整")
        
        # 基于模块状态的具体建议
        low_performance_modules = [m for m in module_states if m.accuracy < 0.6]
        if low_performance_modules:
            module_names = [m.module_name for m in low_performance_modules]
            recommendations.append(f"模块 {', '.join(module_names)} 性能偏低，需要重点优化")
        
        unstable_modules = [m for m in module_states if m.stability < 0.5]
        if unstable_modules:
            module_names = [m.module_name for m in unstable_modules]
            recommendations.append(f"模块 {', '.join(module_names)} 稳定性不足，建议增加正则化")
        
        return recommendations
    
    def get_learning_state_summary(self) -> Dict[str, Any]:
        """
        获取学习状态摘要
        
        Returns:
            Dict[str, Any]: 学习状态摘要
        """
        if not self.current_state:
            return {'message': '当前无学习状态数据'}
        
        summary = {
            'current_state': {
                'overall_score': self.current_state.overall_score,
                'status': self.current_state.status.value,
                'priority': self.current_state.priority.value,
                'consistency_score': self.current_state.consistency_score,
                'module_count': len(self.current_state.module_states)
            },
            'recent_trends': self._get_recent_trends(),
            'active_alerts': self.check_learning_alerts(),
            'recommendations': self.current_state.recommendations[:3]  # 只显示前3条建议
        }
        
        return summary
    
    def _get_recent_trends(self) -> Dict[str, Any]:
        """
        获取最近趋势
        
        Returns:
            Dict[str, Any]: 最近趋势信息
        """
        if len(self.learning_history) < 2:
            return {'message': '历史数据不足'}
        
        recent_records = list(self.learning_history)[-5:]  # 最近5条记录
        
        overall_scores = [r.aggregated_state.overall_score for r in recent_records]
        consistency_scores = [r.aggregated_state.consistency_score for r in recent_records]
        
        return {
            'score_trend': self._calculate_trend(overall_scores),
            'consistency_trend': self._calculate_trend(consistency_scores),
            'recent_avg_score': np.mean(overall_scores),
            'recent_avg_consistency': np.mean(consistency_scores)
        }
    
    def export_learning_data(self, 
                           format_type: str = 'json',
                           time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """
        导出学习数据
        
        Args:
            format_type: 导出格式 ('json', 'csv')
            time_range: 时间范围
            
        Returns:
            str: 导出的数据字符串
        """
        filtered_history = self._filter_history_by_time_range(time_range)
        
        if format_type == 'json':
            return self._export_to_json(filtered_history)
        elif format_type == 'csv':
            return self._export_to_csv(filtered_history)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")
    
    def _export_to_json(self, history: List[LearningHistoryRecord]) -> str:
        """
        导出为JSON格式
        
        Args:
            history: 历史记录列表
            
        Returns:
            str: JSON格式的数据
        """
        export_data = []
        
        for record in history:
            record_data = {
                'timestamp': record.timestamp.isoformat(),
                'overall_score': record.aggregated_state.overall_score,
                'status': record.aggregated_state.status.value,
                'consistency_score': record.aggregated_state.consistency_score,
                'modules': []
            }
            
            for module in record.aggregated_state.module_states:
                module_data = {
                    'module_id': module.module_id,
                    'module_name': module.module_name,
                    'accuracy': module.accuracy,
                    'loss': module.loss,
                    'convergence_rate': module.convergence_rate,
                    'learning_efficiency': module.learning_efficiency,
                    'stability': module.stability
                }
                record_data['modules'].append(module_data)
            
            export_data.append(record_data)
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def _export_to_csv(self, history: List[LearningHistoryRecord]) -> str:
        """
        导出为CSV格式
        
        Args:
            history: 历史记录列表
            
        Returns:
            str: CSV格式的数据
        """
        import io
        
        output = io.StringIO()
        
        # 写入头部
        output.write('timestamp,overall_score,status,consistency_score,module_id,module_name,')
        output.write('accuracy,loss,convergence_rate,learning_efficiency,stability\n')
        
        # 写入数据
        for record in history:
            timestamp = record.timestamp.isoformat()
            overall_score = record.aggregated_state.overall_score
            status = record.aggregated_state.status.value
            consistency_score = record.aggregated_state.consistency_score
            
            for module in record.aggregated_state.module_states:
                output.write(f'{timestamp},{overall_score},{status},{consistency_score},')
                output.write(f'{module.module_id},{module.module_name},')
                output.write(f'{module.accuracy},{module.loss},{module.convergence_rate},')
                output.write(f'{module.learning_efficiency},{module.stability}\n')
        
        return output.getvalue()
    
    def visualize_learning_trends(self, save_path: Optional[str] = None) -> None:
        """
        可视化学习趋势
        
        Args:
            save_path: 保存路径，如果为None则显示图表
        """
        if len(self.learning_history) < 2:
            logger.warning("历史数据不足，无法生成趋势图")
            return
        
        # 准备数据
        history_list = list(self.learning_history)
        timestamps = [record.timestamp for record in history_list]
        overall_scores = [record.aggregated_state.overall_score for record in history_list]
        consistency_scores = [record.aggregated_state.consistency_score for record in history_list]
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 总体性能趋势
        axes[0].plot(timestamps, overall_scores, marker='o', linewidth=2, markersize=4)
        axes[0].set_title('学习性能趋势', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('综合性能分数', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # 一致性趋势
        axes[1].plot(timestamps, consistency_scores, marker='s', color='orange', linewidth=2, markersize=4)
        axes[1].set_title('学习一致性趋势', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('一致性分数', fontsize=12)
        axes[1].set_xlabel('时间', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # 添加一致性阈值线
        axes[1].axhline(y=self.consistency_threshold, color='red', linestyle='--', 
                       label=f'一致性阈值 ({self.consistency_threshold})')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"趋势图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


# 使用示例
if __name__ == "__main__":
    # 创建学习状态聚合器
    aggregator = LearningStateAggregator(
        history_window=50,
        consistency_threshold=0.8
    )
    
    # 创建示例模块状态
    module_states = [
        ModuleLearningState(
            module_id="module_1",
            module_name="卷积神经网络",
            accuracy=0.85,
            loss=0.15,
            convergence_rate=0.78,
            learning_efficiency=0.82,
            stability=0.88
        ),
        ModuleLearningState(
            module_id="module_2",
            module_name="循环神经网络",
            accuracy=0.78,
            loss=0.22,
            convergence_rate=0.72,
            learning_efficiency=0.75,
            stability=0.80
        ),
        ModuleLearningState(
            module_id="module_3",
            module_name="注意力机制",
            accuracy=0.90,
            loss=0.10,
            convergence_rate=0.85,
            learning_efficiency=0.88,
            stability=0.92
        )
    ]
    
    # 聚合学习状态
    aggregated_state = aggregator.aggregate_learning_states(module_states)
    
    # 验证学习一致性
    consistency_validation = aggregator.validate_learning_consistency(module_states)
    
    # 优先级排序
    priorities = aggregator.prioritize_learning_modules(module_states)
    
    # 检查预警
    alerts = aggregator.check_learning_alerts()
    
    # 生成优化建议
    optimization_suggestions = aggregator.generate_optimization_suggestions(module_states)
    
    # 生成报告
    report = aggregator.generate_learning_report()
    
    # 获取状态摘要
    summary = aggregator.get_learning_state_summary()
    
    # 可视化趋势
    aggregator.visualize_learning_trends("learning_trends.png")
    
    print("F9学习状态聚合器演示完成")