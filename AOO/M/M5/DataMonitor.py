"""
M5 数据监控器实现

该模块提供全面的数据监控功能，包括数据质量、完整性、一致性、及时性、
数据量、安全性、访问监控以及异常检测等功能。

Author: AI Assistant
Date: 2025-11-05
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import statistics
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
import sqlite3
import threading
import warnings


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataQualityScore(Enum):
    """数据质量评分枚举"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 80-89
    FAIR = "fair"           # 70-79
    POOR = "poor"           # 60-69
    CRITICAL = "critical"   # <60


@dataclass
class MonitorResult:
    """监控结果数据类"""
    monitor_type: str
    status: str
    score: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    alert_level: AlertLevel = AlertLevel.INFO
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataQualityMetrics:
    """数据质量指标数据类"""
    completeness: float = 0.0      # 完整性
    accuracy: float = 0.0          # 准确性
    consistency: float = 0.0       # 一致性
    timeliness: float = 0.0        # 及时性
    validity: float = 0.0          # 有效性
    uniqueness: float = 0.0        # 唯一性
    overall_score: float = 0.0     # 总体评分
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'consistency': self.consistency,
            'timeliness': self.timeliness,
            'validity': self.validity,
            'uniqueness': self.uniqueness,
            'overall_score': self.overall_score
        }


@dataclass
class AnomalyDetectionResult:
    """异常检测结果数据类"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0


class DataMonitor:
    """
    数据监控器类
    
    提供全面的数据监控功能，包括数据质量监控、完整性监控、一致性监控、
    及时性监控、数据量监控、安全监控、访问监控、异常检测和监控报告等功能。
    
    Attributes:
        config (Dict[str, Any]): 监控配置
        logger (logging.Logger): 日志记录器
        encryption_key (bytes): 加密密钥
        alert_callbacks (List[Callable]): 告警回调函数
        monitor_history (List[MonitorResult]): 监控历史记录
        anomaly_cache (Dict[str, AnomalyDetectionResult]): 异常缓存
        access_log (List[Dict[str, Any]]): 访问日志
        data_registry (Dict[str, Any]): 数据注册表
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据监控器
        
        Args:
            config: 配置字典，包含监控参数
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # 告警回调函数列表
        self.alert_callbacks: List[Callable] = []
        
        # 监控历史记录
        self.monitor_history: List[MonitorResult] = []
        
        # 异常检测缓存
        self.anomaly_cache: Dict[str, AnomalyDetectionResult] = {}
        
        # 数据访问日志
        self.access_log: List[Dict[str, Any]] = []
        
        # 数据注册表
        self.data_registry: Dict[str, Dict[str, Any]] = {}
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 初始化数据库
        self._init_database()
        
        self.logger.info("数据监控器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'quality_thresholds': {
                'excellent': 90,
                'good': 80,
                'fair': 70,
                'poor': 60
            },
            'completeness_threshold': 0.95,
            'accuracy_threshold': 0.90,
            'consistency_threshold': 0.95,
            'timeliness_threshold': 0.90,
            'validity_threshold': 0.95,
            'uniqueness_threshold': 0.99,
            'anomaly_threshold': 0.8,
            'data_volume_alerts': {
                'min_records': 100,
                'max_records': 1000000
            },
            'security': {
                'enable_encryption': True,
                'access_log_retention_days': 30
            },
            'monitoring': {
                'auto_monitor': True,
                'monitor_interval_minutes': 60,
                'max_history_records': 1000
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('DataMonitor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """初始化SQLite数据库用于存储监控历史"""
        try:
            self.db_path = Path("monitor_history.db")
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 创建监控历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitor_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    monitor_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    alert_level TEXT NOT NULL,
                    recommendations TEXT
                )
            ''')
            
            # 创建数据注册表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_registry (
                    data_id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    schema TEXT,
                    last_updated TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
    
    # ==================== 数据质量监控 ====================
    
    def monitor_data_quality(self, data: Any, data_id: str) -> MonitorResult:
        """
        数据质量监控
        
        Args:
            data: 待监控的数据
            data_id: 数据标识符
            
        Returns:
            MonitorResult: 监控结果
        """
        try:
            self.logger.info(f"开始数据质量监控: {data_id}")
            
            # 注册数据
            self._register_data(data_id, data)
            
            # 计算各项质量指标
            metrics = self._calculate_quality_metrics(data)
            
            # 计算总体质量评分
            overall_score = self._calculate_overall_quality_score(metrics)
            
            # 确定状态和告警级别
            status, alert_level = self._determine_quality_status(overall_score)
            
            # 生成建议
            recommendations = self._generate_quality_recommendations(metrics)
            
            result = MonitorResult(
                monitor_type="data_quality",
                status=status,
                score=overall_score,
                message=f"数据质量监控完成，总体评分: {overall_score:.2f}",
                details={
                    'data_id': data_id,
                    'metrics': metrics.to_dict(),
                    'data_type': type(data).__name__,
                    'data_size': self._get_data_size(data)
                },
                alert_level=alert_level,
                recommendations=recommendations
            )
            
            # 保存监控结果
            self._save_monitor_result(result)
            
            # 检查是否需要告警
            if alert_level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]:
                self._trigger_alerts(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据质量监控失败: {e}")
            return MonitorResult(
                monitor_type="data_quality",
                status="error",
                score=0.0,
                message=f"数据质量监控失败: {str(e)}",
                alert_level=AlertLevel.ERROR
            )
    
    def _calculate_quality_metrics(self, data: Any) -> DataQualityMetrics:
        """计算数据质量指标"""
        metrics = DataQualityMetrics()
        
        try:
            # 计算完整性
            metrics.completeness = self._calculate_completeness(data)
            
            # 计算准确性
            metrics.accuracy = self._calculate_accuracy(data)
            
            # 计算一致性
            metrics.consistency = self._calculate_consistency(data)
            
            # 计算及时性
            metrics.timeliness = self._calculate_timeliness(data)
            
            # 计算有效性
            metrics.validity = self._calculate_validity(data)
            
            # 计算唯一性
            metrics.uniqueness = self._calculate_uniqueness(data)
            
        except Exception as e:
            self.logger.error(f"计算质量指标失败: {e}")
            # 设置默认值
            metrics.completeness = 0.0
            metrics.accuracy = 0.0
            metrics.consistency = 0.0
            metrics.timeliness = 0.0
            metrics.validity = 0.0
            metrics.uniqueness = 0.0
        
        return metrics
    
    def _calculate_completeness(self, data: Any) -> float:
        """计算数据完整性"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame的完整性计算
                total_cells = data.size
                non_null_cells = data.count().sum()
                return non_null_cells / total_cells if total_cells > 0 else 0.0
            
            elif isinstance(data, dict):
                # 字典的完整性计算
                total_fields = len(data)
                non_null_fields = sum(1 for v in data.values() if v is not None and v != '')
                return non_null_fields / total_fields if total_fields > 0 else 0.0
            
            elif isinstance(data, list):
                # 列表的完整性计算
                non_null_items = sum(1 for item in data if item is not None and item != '')
                return non_null_items / len(data) if data else 0.0
            
            else:
                # 其他类型假设完整性为1.0
                return 1.0
                
        except Exception as e:
            self.logger.error(f"计算完整性失败: {e}")
            return 0.0
    
    def _calculate_accuracy(self, data: Any) -> float:
        """计算数据准确性"""
        try:
            if isinstance(data, pd.DataFrame):
                # 数值列的准确性检查
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) == 0:
                    return 1.0
                
                accuracy_scores = []
                for col in numeric_columns:
                    # 检查是否有异常值
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    outlier_ratio = len(outliers) / len(data) if len(data) > 0 else 0
                    accuracy = 1.0 - outlier_ratio
                    accuracy_scores.append(accuracy)
                
                return statistics.mean(accuracy_scores) if accuracy_scores else 1.0
            
            else:
                # 对于其他类型，假设准确性为1.0
                return 1.0
                
        except Exception as e:
            self.logger.error(f"计算准确性失败: {e}")
            return 0.0
    
    def _calculate_consistency(self, data: Any) -> float:
        """计算数据一致性"""
        try:
            if isinstance(data, pd.DataFrame):
                # 检查数据类型一致性
                type_consistency_scores = []
                
                for col in data.columns:
                    # 检查同一列的数据类型是否一致
                    dtype_counts = data[col].dtype
                    # 这里简化处理，实际应该更复杂
                    type_consistency_scores.append(1.0)
                
                return statistics.mean(type_consistency_scores) if type_consistency_scores else 1.0
            
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"计算一致性失败: {e}")
            return 0.0
    
    def _calculate_timeliness(self, data: Any) -> float:
        """计算数据及时性"""
        try:
            # 检查数据的时间戳是否在合理范围内
            current_time = datetime.now()
            
            if isinstance(data, dict) and 'timestamp' in data:
                data_time = data['timestamp']
                if isinstance(data_time, str):
                    data_time = datetime.fromisoformat(data_time.replace('Z', '+00:00'))
                
                time_diff = abs((current_time - data_time).total_seconds())
                # 如果时间差小于1小时，认为是及时的
                return max(0.0, 1.0 - time_diff / 3600)
            
            # 对于没有时间戳的数据，假设及时性为1.0
            return 1.0
            
        except Exception as e:
            self.logger.error(f"计算及时性失败: {e}")
            return 0.0
    
    def _calculate_validity(self, data: Any) -> float:
        """计算数据有效性"""
        try:
            if isinstance(data, pd.DataFrame):
                # 检查数据格式是否符合预期
                validity_scores = []
                
                for col in data.columns:
                    # 简化处理：检查是否有空值
                    null_ratio = data[col].isnull().sum() / len(data)
                    validity = 1.0 - null_ratio
                    validity_scores.append(validity)
                
                return statistics.mean(validity_scores) if validity_scores else 1.0
            
            elif isinstance(data, dict):
                # 检查字典值的有效性
                valid_fields = 0
                total_fields = len(data)
                
                for key, value in data.items():
                    if value is not None and value != '':
                        valid_fields += 1
                
                return valid_fields / total_fields if total_fields > 0 else 0.0
            
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"计算有效性失败: {e}")
            return 0.0
    
    def _calculate_uniqueness(self, data: Any) -> float:
        """计算数据唯一性"""
        try:
            if isinstance(data, pd.DataFrame):
                # 检查重复行
                total_rows = len(data)
                unique_rows = len(data.drop_duplicates())
                return unique_rows / total_rows if total_rows > 0 else 1.0
            
            elif isinstance(data, list):
                # 检查重复元素
                unique_items = len(set(data))
                return unique_items / len(data) if data else 1.0
            
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"计算唯一性失败: {e}")
            return 0.0
    
    def _calculate_overall_quality_score(self, metrics: DataQualityMetrics) -> float:
        """计算总体质量评分"""
        weights = {
            'completeness': 0.20,
            'accuracy': 0.25,
            'consistency': 0.20,
            'timeliness': 0.15,
            'validity': 0.15,
            'uniqueness': 0.05
        }
        
        overall_score = (
            metrics.completeness * weights['completeness'] +
            metrics.accuracy * weights['accuracy'] +
            metrics.consistency * weights['consistency'] +
            metrics.timeliness * weights['timeliness'] +
            metrics.validity * weights['validity'] +
            metrics.uniqueness * weights['uniqueness']
        )
        
        metrics.overall_score = overall_score
        return overall_score
    
    def _determine_quality_status(self, score: float) -> Tuple[str, AlertLevel]:
        """确定质量状态和告警级别"""
        thresholds = self.config['quality_thresholds']
        
        if score >= thresholds['excellent']:
            return "excellent", AlertLevel.INFO
        elif score >= thresholds['good']:
            return "good", AlertLevel.INFO
        elif score >= thresholds['fair']:
            return "fair", AlertLevel.WARNING
        elif score >= thresholds['poor']:
            return "poor", AlertLevel.ERROR
        else:
            return "critical", AlertLevel.CRITICAL
    
    def _generate_quality_recommendations(self, metrics: DataQualityMetrics) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        if metrics.completeness < 0.95:
            recommendations.append("数据完整性不足，建议检查数据收集流程")
        
        if metrics.accuracy < 0.90:
            recommendations.append("数据准确性有待提高，建议增加数据验证规则")
        
        if metrics.consistency < 0.95:
            recommendations.append("数据一致性存在问题，建议统一数据标准")
        
        if metrics.timeliness < 0.90:
            recommendations.append("数据更新不够及时，建议优化数据更新频率")
        
        if metrics.validity < 0.95:
            recommendations.append("数据格式存在问题，建议加强数据格式验证")
        
        if metrics.uniqueness < 0.99:
            recommendations.append("存在重复数据，建议实施去重机制")
        
        return recommendations
    
    # ==================== 数据完整性监控 ====================
    
    def monitor_data_integrity(self, data: Any, expected_schema: Optional[Dict[str, Any]] = None) -> MonitorResult:
        """
        数据完整性监控
        
        Args:
            data: 待监控的数据
            expected_schema: 期望的数据模式
            
        Returns:
            MonitorResult: 监控结果
        """
        try:
            self.logger.info("开始数据完整性监控")
            
            integrity_issues = []
            integrity_score = 1.0
            
            # 检查数据结构完整性
            structure_score = self._check_structure_integrity(data, expected_schema)
            integrity_score *= structure_score['score']
            
            if structure_score['issues']:
                integrity_issues.extend(structure_score['issues'])
            
            # 检查引用完整性
            reference_score = self._check_reference_integrity(data)
            integrity_score *= reference_score['score']
            
            if reference_score['issues']:
                integrity_issues.extend(reference_score['issues'])
            
            # 检查约束完整性
            constraint_score = self._check_constraint_integrity(data)
            integrity_score *= constraint_score['score']
            
            if constraint_score['issues']:
                integrity_issues.extend(constraint_score['issues'])
            
            # 确定状态
            if integrity_score >= 0.95:
                status = "good"
                alert_level = AlertLevel.INFO
            elif integrity_score >= 0.80:
                status = "warning"
                alert_level = AlertLevel.WARNING
            else:
                status = "error"
                alert_level = AlertLevel.ERROR
            
            result = MonitorResult(
                monitor_type="data_integrity",
                status=status,
                score=integrity_score,
                message=f"数据完整性监控完成，评分: {integrity_score:.2f}",
                details={
                    'integrity_issues': integrity_issues,
                    'structure_score': structure_score['score'],
                    'reference_score': reference_score['score'],
                    'constraint_score': constraint_score['score']
                },
                alert_level=alert_level,
                recommendations=self._generate_integrity_recommendations(integrity_issues)
            )
            
            self._save_monitor_result(result)
            return result
            
        except Exception as e:
            self.logger.error(f"数据完整性监控失败: {e}")
            return MonitorResult(
                monitor_type="data_integrity",
                status="error",
                score=0.0,
                message=f"数据完整性监控失败: {str(e)}",
                alert_level=AlertLevel.ERROR
            )
    
    def _check_structure_integrity(self, data: Any, expected_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查结构完整性"""
        issues = []
        score = 1.0
        
        if expected_schema:
            # 根据期望模式检查结构
            if isinstance(data, dict):
                expected_fields = set(expected_schema.get('fields', {}).keys())
                actual_fields = set(data.keys())
                
                missing_fields = expected_fields - actual_fields
                extra_fields = actual_fields - expected_fields
                
                if missing_fields:
                    issues.append(f"缺少必要字段: {missing_fields}")
                    score *= 0.8
                
                if extra_fields:
                    issues.append(f"发现额外字段: {extra_fields}")
                    score *= 0.9
        
        return {'score': score, 'issues': issues}
    
    def _check_reference_integrity(self, data: Any) -> Dict[str, Any]:
        """检查引用完整性"""
        issues = []
        score = 1.0
        
        # 简化处理：检查外键引用
        if isinstance(data, pd.DataFrame):
            # 这里应该检查外键约束
            pass
        
        return {'score': score, 'issues': issues}
    
    def _check_constraint_integrity(self, data: Any) -> Dict[str, Any]:
        """检查约束完整性"""
        issues = []
        score = 1.0
        
        if isinstance(data, pd.DataFrame):
            # 检查各种约束
            for col in data.columns:
                # 检查非空约束
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"字段 {col} 存在 {null_count} 个空值")
                    score *= 0.9
        
        return {'score': score, 'issues': issues}
    
    def _generate_integrity_recommendations(self, issues: List[str]) -> List[str]:
        """生成完整性改进建议"""
        recommendations = []
        
        for issue in issues:
            if "缺少必要字段" in issue:
                recommendations.append("检查数据源，确保所有必要字段都有数据")
            elif "空值" in issue:
                recommendations.append("实施数据清洗，处理空值和缺失数据")
            elif "外键" in issue:
                recommendations.append("检查引用关系，确保外键约束的有效性")
        
        return recommendations
    
    # ==================== 数据一致性监控 ====================
    
    def monitor_data_consistency(self, datasets: List[Any], comparison_rules: Optional[Dict[str, Any]] = None) -> MonitorResult:
        """
        数据一致性监控
        
        Args:
            datasets: 数据集列表
            comparison_rules: 比较规则
            
        Returns:
            MonitorResult: 监控结果
        """
        try:
            self.logger.info("开始数据一致性监控")
            
            if len(datasets) < 2:
                return MonitorResult(
                    monitor_type="data_consistency",
                    status="warning",
                    score=1.0,
                    message="数据集数量不足，无法进行一致性检查",
                    alert_level=AlertLevel.WARNING
                )
            
            consistency_scores = []
            consistency_issues = []
            
            # 比较不同数据集之间的一致性
            for i in range(len(datasets)):
                for j in range(i + 1, len(datasets)):
                    comparison_result = self._compare_datasets(datasets[i], datasets[j], comparison_rules)
                    consistency_scores.append(comparison_result['score'])
                    
                    if comparison_result['issues']:
                        consistency_issues.extend(comparison_result['issues'])
            
            # 计算总体一致性评分
            overall_consistency = statistics.mean(consistency_scores) if consistency_scores else 1.0
            
            # 确定状态
            if overall_consistency >= 0.95:
                status = "good"
                alert_level = AlertLevel.INFO
            elif overall_consistency >= 0.80:
                status = "warning"
                alert_level = AlertLevel.WARNING
            else:
                status = "error"
                alert_level = AlertLevel.ERROR
            
            result = MonitorResult(
                monitor_type="data_consistency",
                status=status,
                score=overall_consistency,
                message=f"数据一致性监控完成，评分: {overall_consistency:.2f}",
                details={
                    'consistency_issues': consistency_issues,
                    'dataset_count': len(datasets),
                    'comparison_pairs': len(consistency_scores)
                },
                alert_level=alert_level,
                recommendations=self._generate_consistency_recommendations(consistency_issues)
            )
            
            self._save_monitor_result(result)
            return result
            
        except Exception as e:
            self.logger.error(f"数据一致性监控失败: {e}")
            return MonitorResult(
                monitor_type="data_consistency",
                status="error",
                score=0.0,
                message=f"数据一致性监控失败: {str(e)}",
                alert_level=AlertLevel.ERROR
            )
    
    def _compare_datasets(self, data1: Any, data2: Any, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """比较两个数据集的一致性"""
        issues = []
        score = 1.0
        
        try:
            # 如果都是DataFrame，进行列级比较
            if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
                # 比较列名
                cols1 = set(data1.columns)
                cols2 = set(data2.columns)
                
                if cols1 != cols2:
                    missing_cols = cols2 - cols1
                    extra_cols = cols1 - cols2
                    
                    if missing_cols:
                        issues.append(f"数据集2缺少列: {missing_cols}")
                        score *= 0.8
                    
                    if extra_cols:
                        issues.append(f"数据集2有多余列: {extra_cols}")
                        score *= 0.9
                
                # 比较数据类型
                common_cols = cols1 & cols2
                for col in common_cols:
                    if data1[col].dtype != data2[col].dtype:
                        issues.append(f"列 {col} 的数据类型不一致")
                        score *= 0.9
                
                # 比较统计信息
                for col in common_cols:
                    if data1[col].dtype in ['int64', 'float64']:
                        mean1 = data1[col].mean()
                        mean2 = data2[col].mean()
                        
                        if abs(mean1 - mean2) / max(abs(mean1), abs(mean2), 1) > 0.1:
                            issues.append(f"列 {col} 的均值差异较大")
                            score *= 0.9
            
            elif isinstance(data1, dict) and isinstance(data2, dict):
                # 字典比较
                keys1 = set(data1.keys())
                keys2 = set(data2.keys())
                
                if keys1 != keys2:
                    missing_keys = keys2 - keys1
                    extra_keys = keys1 - keys2
                    
                    if missing_keys:
                        issues.append(f"字典2缺少键: {missing_keys}")
                        score *= 0.8
                    
                    if extra_keys:
                        issues.append(f"字典2有多余键: {extra_keys}")
                        score *= 0.9
                
                # 比较值
                common_keys = keys1 & keys2
                for key in common_keys:
                    if data1[key] != data2[key]:
                        issues.append(f"键 {key} 的值不一致")
                        score *= 0.9
        
        except Exception as e:
            issues.append(f"比较过程中出错: {str(e)}")
            score *= 0.5
        
        return {'score': score, 'issues': issues}
    
    def _generate_consistency_recommendations(self, issues: List[str]) -> List[str]:
        """生成一致性改进建议"""
        recommendations = []
        
        for issue in issues:
            if "缺少列" in issue or "缺少键" in issue:
                recommendations.append("统一数据结构定义，确保所有数据源包含相同的字段")
            elif "数据类型不一致" in issue:
                recommendations.append("统一数据类型定义，确保相同字段使用相同的数据类型")
            elif "值不一致" in issue:
                recommendations.append("检查数据转换逻辑，确保数据处理的一致性")
        
        return recommendations
    
    # ==================== 数据及时性监控 ====================
    
    def monitor_data_timeliness(self, data_source: str, expected_update_frequency: timedelta) -> MonitorResult:
        """
        数据及时性监控
        
        Args:
            data_source: 数据源标识
            expected_update_frequency: 期望的更新频率
            
        Returns:
            MonitorResult: 监控结果
        """
        try:
            self.logger.info(f"开始数据及时性监控: {data_source}")
            
            # 获取数据源的最后更新时间
            last_update_time = self._get_last_update_time(data_source)
            
            if last_update_time is None:
                return MonitorResult(
                    monitor_type="data_timeliness",
                    status="error",
                    score=0.0,
                    message=f"数据源 {data_source} 没有更新记录",
                    alert_level=AlertLevel.ERROR
                )
            
            current_time = datetime.now()
            time_since_update = current_time - last_update_time
            
            # 计算及时性评分
            expected_seconds = expected_update_frequency.total_seconds()
            actual_seconds = time_since_update.total_seconds()
            
            if actual_seconds <= expected_seconds:
                timeliness_score = 1.0
                status = "good"
                alert_level = AlertLevel.INFO
            else:
                # 延迟越久，评分越低
                delay_ratio = actual_seconds / expected_seconds
                timeliness_score = max(0.0, 1.0 - (delay_ratio - 1.0) * 0.5)
                
                if delay_ratio <= 2.0:
                    status = "warning"
                    alert_level = AlertLevel.WARNING
                else:
                    status = "error"
                    alert_level = AlertLevel.ERROR
            
            # 生成建议
            recommendations = []
            if timeliness_score < 1.0:
                recommendations.append(f"数据更新延迟 {time_since_update}，建议检查数据更新流程")
                recommendations.append(f"期望更新频率: {expected_update_frequency}")
            
            result = MonitorResult(
                monitor_type="data_timeliness",
                status=status,
                score=timeliness_score,
                message=f"数据及时性监控完成，评分: {timeliness_score:.2f}",
                details={
                    'data_source': data_source,
                    'last_update_time': last_update_time.isoformat(),
                    'time_since_update_seconds': actual_seconds,
                    'expected_frequency_seconds': expected_seconds,
                    'delay_ratio': actual_seconds / expected_seconds
                },
                alert_level=alert_level,
                recommendations=recommendations
            )
            
            self._save_monitor_result(result)
            return result
            
        except Exception as e:
            self.logger.error(f"数据及时性监控失败: {e}")
            return MonitorResult(
                monitor_type="data_timeliness",
                status="error",
                score=0.0,
                message=f"数据及时性监控失败: {str(e)}",
                alert_level=AlertLevel.ERROR
            )
    
    def _get_last_update_time(self, data_source: str) -> Optional[datetime]:
        """获取数据源的最后更新时间"""
        try:
            # 从数据注册表中获取
            if data_source in self.data_registry:
                last_updated_str = self.data_registry[data_source].get('last_updated')
                if last_updated_str:
                    return datetime.fromisoformat(last_updated_str)
            
            # 简化处理：返回当前时间
            return datetime.now()
            
        except Exception as e:
            self.logger.error(f"获取最后更新时间失败: {e}")
            return None
    
    # ==================== 数据量监控 ====================
    
    def monitor_data_volume(self, data: Any, data_id: str) -> MonitorResult:
        """
        数据量监控
        
        Args:
            data: 待监控的数据
            data_id: 数据标识符
            
        Returns:
            MonitorResult: 监控结果
        """
        try:
            self.logger.info(f"开始数据量监控: {data_id}")
            
            # 计算数据量指标
            volume_metrics = self._calculate_volume_metrics(data)
            
            # 检查数据量是否在合理范围内
            min_records = self.config['data_volume_alerts']['min_records']
            max_records = self.config['data_volume_alerts']['max_records']
            
            record_count = volume_metrics['record_count']
            
            if record_count < min_records:
                status = "warning"
                alert_level = AlertLevel.WARNING
                message = f"数据量过少: {record_count} 条记录，少于最小阈值 {min_records}"
            elif record_count > max_records:
                status = "warning"
                alert_level = AlertLevel.WARNING
                message = f"数据量过大: {record_count} 条记录，超过最大阈值 {max_records}"
            else:
                status = "good"
                alert_level = AlertLevel.INFO
                message = f"数据量正常: {record_count} 条记录"
            
            # 计算数据量评分
            if record_count == 0:
                volume_score = 0.0
            elif record_count < min_records:
                volume_score = record_count / min_records
            elif record_count > max_records:
                volume_score = max(0.0, 1.0 - (record_count - max_records) / max_records)
            else:
                volume_score = 1.0
            
            recommendations = []
            if record_count < min_records:
                recommendations.append(f"数据量不足，建议增加数据收集量")
            elif record_count > max_records:
                recommendations.append(f"数据量过大，建议考虑数据归档或分片")
            
            result = MonitorResult(
                monitor_type="data_volume",
                status=status,
                score=volume_score,
                message=message,
                details={
                    'data_id': data_id,
                    'volume_metrics': volume_metrics,
                    'min_threshold': min_records,
                    'max_threshold': max_records
                },
                alert_level=alert_level,
                recommendations=recommendations
            )
            
            self._save_monitor_result(result)
            return result
            
        except Exception as e:
            self.logger.error(f"数据量监控失败: {e}")
            return MonitorResult(
                monitor_type="data_volume",
                status="error",
                score=0.0,
                message=f"数据量监控失败: {str(e)}",
                alert_level=AlertLevel.ERROR
            )
    
    def _calculate_volume_metrics(self, data: Any) -> Dict[str, Any]:
        """计算数据量指标"""
        metrics = {}
        
        try:
            if isinstance(data, pd.DataFrame):
                metrics['record_count'] = len(data)
                metrics['column_count'] = len(data.columns)
                metrics['memory_usage_mb'] = data.memory_usage(deep=True).sum() / 1024 / 1024
                metrics['file_size_mb'] = 0  # 需要文件系统信息
                
            elif isinstance(data, dict):
                metrics['record_count'] = 1
                metrics['field_count'] = len(data)
                metrics['memory_usage_mb'] = len(str(data)) / 1024 / 1024
                
            elif isinstance(data, list):
                metrics['record_count'] = len(data)
                metrics['memory_usage_mb'] = len(str(data)) / 1024 / 1024
                
            else:
                metrics['record_count'] = 1
                metrics['memory_usage_mb'] = len(str(data)) / 1024 / 1024
            
            metrics['data_type'] = type(data).__name__
            
        except Exception as e:
            self.logger.error(f"计算数据量指标失败: {e}")
            metrics = {
                'record_count': 0,
                'column_count': 0,
                'memory_usage_mb': 0.0,
                'file_size_mb': 0.0,
                'data_type': type(data).__name__
            }
        
        return metrics
    
    # ==================== 数据安全监控 ====================
    
    def monitor_data_security(self, data: Any, data_id: str, security_rules: Optional[Dict[str, Any]] = None) -> MonitorResult:
        """
        数据安全监控
        
        Args:
            data: 待监控的数据
            data_id: 数据标识符
            security_rules: 安全规则
            
        Returns:
            MonitorResult: 监控结果
        """
        try:
            self.logger.info(f"开始数据安全监控: {data_id}")
            
            security_issues = []
            security_score = 1.0
            
            # 检查敏感数据泄露
            sensitive_check = self._check_sensitive_data_exposure(data, security_rules)
            security_score *= sensitive_check['score']
            if sensitive_check['issues']:
                security_issues.extend(sensitive_check['issues'])
            
            # 检查数据加密
            encryption_check = self._check_data_encryption(data, security_rules)
            security_score *= encryption_check['score']
            if encryption_check['issues']:
                security_issues.extend(encryption_check['issues'])
            
            # 检查访问权限
            access_check = self._check_access_permissions(data_id, security_rules)
            security_score *= access_check['score']
            if access_check['issues']:
                security_issues.extend(access_check['issues'])
            
            # 检查数据完整性
            integrity_check = self._check_data_integrity_security(data)
            security_score *= integrity_check['score']
            if integrity_check['issues']:
                security_issues.extend(integrity_check['issues'])
            
            # 确定状态
            if security_score >= 0.95:
                status = "secure"
                alert_level = AlertLevel.INFO
            elif security_score >= 0.80:
                status = "warning"
                alert_level = AlertLevel.WARNING
            else:
                status = "vulnerable"
                alert_level = AlertLevel.CRITICAL
            
            recommendations = self._generate_security_recommendations(security_issues)
            
            result = MonitorResult(
                monitor_type="data_security",
                status=status,
                score=security_score,
                message=f"数据安全监控完成，评分: {security_score:.2f}",
                details={
                    'data_id': data_id,
                    'security_issues': security_issues,
                    'sensitive_check': sensitive_check,
                    'encryption_check': encryption_check,
                    'access_check': access_check,
                    'integrity_check': integrity_check
                },
                alert_level=alert_level,
                recommendations=recommendations
            )
            
            self._save_monitor_result(result)
            return result
            
        except Exception as e:
            self.logger.error(f"数据安全监控失败: {e}")
            return MonitorResult(
                monitor_type="data_security",
                status="error",
                score=0.0,
                message=f"数据安全监控失败: {str(e)}",
                alert_level=AlertLevel.ERROR
            )
    
    def _check_sensitive_data_exposure(self, data: Any, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查敏感数据泄露"""
        issues = []
        score = 1.0
        
        # 定义敏感字段模式
        sensitive_patterns = [
            'password', 'passwd', 'pwd',
            'ssn', 'social_security',
            'credit_card', 'card_number',
            'bank_account', 'account_number',
            'phone', 'mobile',
            'email', 'address'
        ]
        
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    key_lower = key.lower()
                    for pattern in sensitive_patterns:
                        if pattern in key_lower:
                            # 检查值是否被适当处理
                            if isinstance(value, str) and len(value) > 0:
                                if pattern in ['password', 'passwd', 'pwd']:
                                    if not self._is_password_masked(value):
                                        issues.append(f"发现未加密的密码字段: {key}")
                                        score *= 0.3
                                else:
                                    issues.append(f"发现敏感字段: {key}")
                                    score *= 0.7
            
            elif isinstance(data, pd.DataFrame):
                for col in data.columns:
                    col_lower = col.lower()
                    for pattern in sensitive_patterns:
                        if pattern in col_lower:
                            issues.append(f"发现敏感列: {col}")
                            score *= 0.7
                            
        except Exception as e:
            issues.append(f"敏感数据检查失败: {str(e)}")
            score *= 0.5
        
        return {'score': score, 'issues': issues}
    
    def _is_password_masked(self, password: str) -> bool:
        """检查密码是否被掩码处理"""
        if not password:
            return True
        
        # 检查是否包含明文密码特征
        if len(password) < 8:
            return False
        
        # 检查是否包含常见密码模式
        common_patterns = ['123', 'password', 'admin', 'test']
        password_lower = password.lower()
        
        for pattern in common_patterns:
            if pattern in password_lower:
                return False
        
        return True
    
    def _check_data_encryption(self, data: Any, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查数据加密"""
        issues = []
        score = 1.0
        
        # 如果启用了加密检查
        if self.config['security']['enable_encryption']:
            try:
                # 简化处理：检查数据是否经过加密处理
                if isinstance(data, str) and len(data) > 0:
                    # 简单的加密检查：Base64编码检测
                    try:
                        import base64
                        base64.b64decode(data)
                        # 如果能解码为Base64，可能是加密数据
                        score = 1.0
                    except:
                        # 如果不是Base64，可能是明文
                        score = 0.8
                        issues.append("数据可能未加密")
                
            except Exception as e:
                issues.append(f"加密检查失败: {str(e)}")
                score *= 0.5
        
        return {'score': score, 'issues': issues}
    
    def _check_access_permissions(self, data_id: str, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查访问权限"""
        issues = []
        score = 1.0
        
        try:
            # 检查数据是否在注册表中
            if data_id not in self.data_registry:
                issues.append(f"数据 {data_id} 未在注册表中登记")
                score *= 0.8
            
            # 检查访问日志中是否有异常访问
            recent_accesses = [log for log in self.access_log 
                             if log.get('data_id') == data_id and 
                             (datetime.now() - datetime.fromisoformat(log['timestamp'])).days < 1]
            
            if len(recent_accesses) == 0:
                issues.append(f"数据 {data_id} 最近24小时内无访问记录")
                score *= 0.9
                
        except Exception as e:
            issues.append(f"访问权限检查失败: {str(e)}")
            score *= 0.5
        
        return {'score': score, 'issues': issues}
    
    def _check_data_integrity_security(self, data: Any) -> Dict[str, Any]:
        """检查数据完整性安全"""
        issues = []
        score = 1.0
        
        try:
            # 计算数据哈希值
            data_hash = self._calculate_data_hash(data)
            
            # 检查数据是否被篡改
            # 这里简化处理，实际应该与预期的哈希值比较
            if data_hash is None:
                issues.append("无法计算数据哈希值")
                score *= 0.5
                
        except Exception as e:
            issues.append(f"数据完整性检查失败: {str(e)}")
            score *= 0.5
        
        return {'score': score, 'issues': issues}
    
    def _calculate_data_hash(self, data: Any) -> Optional[str]:
        """计算数据哈希值"""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception:
            return None
    
    def _generate_security_recommendations(self, issues: List[str]) -> List[str]:
        """生成安全改进建议"""
        recommendations = []
        
        for issue in issues:
            if "未加密的密码" in issue:
                recommendations.append("对敏感密码字段进行加密存储")
            elif "敏感字段" in issue:
                recommendations.append("对敏感数据进行脱敏处理或加密")
            elif "未加密" in issue:
                recommendations.append("启用数据加密机制")
            elif "未在注册表中" in issue:
                recommendations.append("将数据源注册到数据目录中")
            elif "无访问记录" in issue:
                recommendations.append("检查数据访问权限配置")
            elif "哈希值" in issue:
                recommendations.append("实施数据完整性校验机制")
        
        return recommendations
    
    # ==================== 数据访问监控 ====================
    
    def monitor_data_access(self, user_id: str, data_id: str, access_type: str) -> MonitorResult:
        """
        数据访问监控
        
        Args:
            user_id: 用户ID
            data_id: 数据ID
            access_type: 访问类型 (read/write/delete)
            
        Returns:
            MonitorResult: 监控结果
        """
        try:
            self.logger.info(f"开始数据访问监控: 用户 {user_id} 访问 {data_id}")
            
            # 记录访问
            self._log_access(user_id, data_id, access_type)
            
            # 分析访问模式
            access_analysis = self._analyze_access_patterns(user_id, data_id)
            
            # 检测异常访问
            anomaly_result = self._detect_access_anomalies(user_id, data_id, access_type)
            
            # 计算访问安全评分
            security_score = self._calculate_access_security_score(access_analysis, anomaly_result)
            
            # 确定状态
            if anomaly_result.is_anomaly:
                status = "anomaly"
                alert_level = AlertLevel.WARNING
                message = f"检测到异常访问模式: {anomaly_result.anomaly_type}"
            else:
                status = "normal"
                alert_level = AlertLevel.INFO
                message = "访问模式正常"
            
            recommendations = self._generate_access_recommendations(anomaly_result, access_analysis)
            
            result = MonitorResult(
                monitor_type="data_access",
                status=status,
                score=security_score,
                message=message,
                details={
                    'user_id': user_id,
                    'data_id': data_id,
                    'access_type': access_type,
                    'access_analysis': access_analysis,
                    'anomaly_detection': {
                        'is_anomaly': anomaly_result.is_anomaly,
                        'anomaly_type': anomaly_result.anomaly_type,
                        'anomaly_score': anomaly_result.anomaly_score,
                        'confidence': anomaly_result.confidence
                    }
                },
                alert_level=alert_level,
                recommendations=recommendations
            )
            
            self._save_monitor_result(result)
            return result
            
        except Exception as e:
            self.logger.error(f"数据访问监控失败: {e}")
            return MonitorResult(
                monitor_type="data_access",
                status="error",
                score=0.0,
                message=f"数据访问监控失败: {str(e)}",
                alert_level=AlertLevel.ERROR
            )
    
    def _log_access(self, user_id: str, data_id: str, access_type: str):
        """记录数据访问"""
        access_record = {
            'user_id': user_id,
            'data_id': data_id,
            'access_type': access_type,
            'timestamp': datetime.now().isoformat(),
            'ip_address': '127.0.0.1',  # 简化处理
            'user_agent': 'DataMonitor/1.0'  # 简化处理
        }
        
        with self._lock:
            self.access_log.append(access_record)
            
            # 限制日志大小
            max_logs = 10000
            if len(self.access_log) > max_logs:
                self.access_log = self.access_log[-max_logs:]
    
    def _analyze_access_patterns(self, user_id: str, data_id: str) -> Dict[str, Any]:
        """分析访问模式"""
        try:
            # 获取用户的历史访问记录
            user_accesses = [log for log in self.access_log if log['user_id'] == user_id]
            
            # 获取数据的历史访问记录
            data_accesses = [log for log in self.access_log if log['data_id'] == data_id]
            
            # 分析访问频率
            now = datetime.now()
            recent_accesses = [log for log in user_accesses 
                             if (now - datetime.fromisoformat(log['timestamp'])).days < 7]
            
            access_frequency = len(recent_accesses)
            
            # 分析访问时间模式
            access_hours = []
            for log in recent_accesses:
                access_time = datetime.fromisoformat(log['timestamp'])
                access_hours.append(access_time.hour)
            
            # 分析访问类型分布
            access_types = Counter([log['access_type'] for log in recent_accesses])
            
            analysis = {
                'user_id': user_id,
                'data_id': data_id,
                'total_accesses': len(user_accesses),
                'recent_accesses_7d': access_frequency,
                'access_frequency_per_day': access_frequency / 7 if access_frequency > 0 else 0,
                'access_hours': access_hours,
                'access_types': dict(access_types),
                'most_common_access_type': access_types.most_common(1)[0][0] if access_types else None,
                'first_access': min([log['timestamp'] for log in user_accesses]) if user_accesses else None,
                'last_access': max([log['timestamp'] for log in user_accesses]) if user_accesses else None
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"访问模式分析失败: {e}")
            return {'error': str(e)}
    
    def _detect_access_anomalies(self, user_id: str, data_id: str, access_type: str) -> AnomalyDetectionResult:
        """检测访问异常"""
        try:
            analysis = self._analyze_access_patterns(user_id, data_id)
            
            anomaly_score = 0.0
            anomaly_type = "normal"
            confidence = 0.0
            
            # 检测访问频率异常
            if analysis.get('access_frequency_per_day', 0) > 100:  # 每天超过100次访问
                anomaly_score += 0.3
                anomaly_type = "high_frequency_access"
                confidence = 0.8
            
            # 检测非工作时间访问
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:  # 夜间访问
                anomaly_score += 0.2
                if anomaly_type == "normal":
                    anomaly_type = "off_hours_access"
                confidence = max(confidence, 0.6)
            
            # 检测访问类型异常
            if access_type == 'delete' and analysis.get('recent_accesses_7d', 0) > 10:
                anomaly_score += 0.4
                anomaly_type = "frequent_delete_access"
                confidence = max(confidence, 0.9)
            
            # 检测首次访问异常
            if analysis.get('first_access') == analysis.get('last_access'):
                anomaly_score += 0.1
                if anomaly_type == "normal":
                    anomaly_type = "first_time_access"
                confidence = max(confidence, 0.5)
            
            is_anomaly = anomaly_score > self.config['anomaly_threshold']
            
            return AnomalyDetectionResult(
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                anomaly_type=anomaly_type,
                details=analysis,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"访问异常检测失败: {e}")
            return AnomalyDetectionResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="detection_error",
                details={'error': str(e)},
                confidence=0.0
            )
    
    def _calculate_access_security_score(self, analysis: Dict[str, Any], anomaly_result: AnomalyDetectionResult) -> float:
        """计算访问安全评分"""
        base_score = 1.0
        
        # 根据异常检测结果调整评分
        if anomaly_result.is_anomaly:
            base_score *= (1.0 - anomaly_result.anomaly_score)
        
        # 根据访问频率调整评分
        access_frequency = analysis.get('access_frequency_per_day', 0)
        if access_frequency > 1000:  # 极高的访问频率
            base_score *= 0.5
        elif access_frequency > 100:
            base_score *= 0.8
        
        return max(0.0, base_score)
    
    def _generate_access_recommendations(self, anomaly_result: AnomalyDetectionResult, analysis: Dict[str, Any]) -> List[str]:
        """生成访问监控建议"""
        recommendations = []
        
        if anomaly_result.is_anomaly:
            if anomaly_result.anomaly_type == "high_frequency_access":
                recommendations.append("检测到高频访问，建议检查用户访问权限和频率限制")
            elif anomaly_result.anomaly_type == "off_hours_access":
                recommendations.append("检测到非工作时间访问，建议审查访问时间合理性")
            elif anomaly_result.anomaly_type == "frequent_delete_access":
                recommendations.append("检测到频繁删除操作，建议加强删除权限控制")
            elif anomaly_result.anomaly_type == "first_time_access":
                recommendations.append("新用户首次访问，建议加强监控和权限验证")
        
        # 基于访问模式的建议
        access_frequency = analysis.get('access_frequency_per_day', 0)
        if access_frequency > 100:
            recommendations.append("用户访问频率较高，建议实施访问频率限制")
        
        return recommendations
    
    # ==================== 数据异常检测 ====================
    
    def detect_data_anomalies(self, data: Any, data_id: str, anomaly_types: Optional[List[str]] = None) -> AnomalyDetectionResult:
        """
        数据异常检测
        
        Args:
            data: 待检测的数据
            data_id: 数据标识符
            anomaly_types: 要检测的异常类型列表
            
        Returns:
            AnomalyDetectionResult: 异常检测结果
        """
        try:
            self.logger.info(f"开始数据异常检测: {data_id}")
            
            if anomaly_types is None:
                anomaly_types = ['statistical', 'pattern', 'volume', 'quality']
            
            anomaly_scores = []
            detected_anomalies = []
            
            # 统计异常检测
            if 'statistical' in anomaly_types:
                stat_result = self._detect_statistical_anomalies(data)
                anomaly_scores.append(stat_result['score'])
                if stat_result['is_anomaly']:
                    detected_anomalies.append(('statistical', stat_result))
            
            # 模式异常检测
            if 'pattern' in anomaly_types:
                pattern_result = self._detect_pattern_anomalies(data)
                anomaly_scores.append(pattern_result['score'])
                if pattern_result['is_anomaly']:
                    detected_anomalies.append(('pattern', pattern_result))
            
            # 数据量异常检测
            if 'volume' in anomaly_types:
                volume_result = self._detect_volume_anomalies(data, data_id)
                anomaly_scores.append(volume_result['score'])
                if volume_result['is_anomaly']:
                    detected_anomalies.append(('volume', volume_result))
            
            # 数据质量异常检测
            if 'quality' in anomaly_types:
                quality_result = self._detect_quality_anomalies(data)
                anomaly_scores.append(quality_result['score'])
                if quality_result['is_anomaly']:
                    detected_anomalies.append(('quality', quality_result))
            
            # 计算总体异常评分
            overall_score = statistics.mean(anomaly_scores) if anomaly_scores else 0.0
            is_anomaly = overall_score > self.config['anomaly_threshold']
            
            # 确定主要异常类型
            if detected_anomalies:
                main_anomaly_type = detected_anomalies[0][0]
                main_anomaly_details = detected_anomalies[0][1]
            else:
                main_anomaly_type = "normal"
                main_anomaly_details = {}
            
            # 计算置信度
            confidence = min(1.0, overall_score * 2) if is_anomaly else (1.0 - overall_score)
            
            result = AnomalyDetectionResult(
                is_anomaly=is_anomaly,
                anomaly_score=overall_score,
                anomaly_type=main_anomaly_type,
                details={
                    'data_id': data_id,
                    'detected_anomalies': detected_anomalies,
                    'anomaly_types_checked': anomaly_types,
                    'main_anomaly_details': main_anomaly_details
                },
                confidence=confidence
            )
            
            # 缓存结果
            with self._lock:
                self.anomaly_cache[data_id] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据异常检测失败: {e}")
            return AnomalyDetectionResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="detection_error",
                details={'error': str(e)},
                confidence=0.0
            )
    
    def _detect_statistical_anomalies(self, data: Any) -> Dict[str, Any]:
        """检测统计异常"""
        try:
            if isinstance(data, pd.DataFrame):
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                anomaly_scores = []
                
                for col in numeric_columns:
                    # 使用Z-score检测异常值
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outlier_ratio = (z_scores > 3).sum() / len(data)
                    anomaly_scores.append(outlier_ratio)
                
                # 计算总体异常评分
                overall_score = statistics.mean(anomaly_scores) if anomaly_scores else 0.0
                is_anomaly = overall_score > 0.05  # 超过5%的数据是异常值
                
                return {
                    'is_anomaly': is_anomaly,
                    'score': overall_score,
                    'method': 'z_score',
                    'outlier_ratio': overall_score,
                    'numeric_columns_checked': len(numeric_columns)
                }
            
            else:
                return {
                    'is_anomaly': False,
                    'score': 0.0,
                    'method': 'not_applicable',
                    'reason': '非DataFrame数据类型'
                }
                
        except Exception as e:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def _detect_pattern_anomalies(self, data: Any) -> Dict[str, Any]:
        """检测模式异常"""
        try:
            if isinstance(data, pd.DataFrame):
                # 检测缺失值模式异常
                missing_patterns = data.isnull()
                
                # 如果某行或某列缺失值过多，可能是异常
                row_missing_ratio = missing_patterns.sum(axis=1) / len(data.columns)
                col_missing_ratio = missing_patterns.sum(axis=0) / len(data)
                
                max_row_missing = row_missing_ratio.max()
                max_col_missing = col_missing_ratio.max()
                
                anomaly_score = max(max_row_missing, max_col_missing)
                is_anomaly = anomaly_score > 0.5  # 超过50%的值缺失
                
                return {
                    'is_anomaly': is_anomaly,
                    'score': anomaly_score,
                    'method': 'missing_pattern',
                    'max_row_missing_ratio': max_row_missing,
                    'max_col_missing_ratio': max_col_missing
                }
            
            else:
                return {
                    'is_anomaly': False,
                    'score': 0.0,
                    'method': 'not_applicable'
                }
                
        except Exception as e:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def _detect_volume_anomalies(self, data: Any, data_id: str) -> Dict[str, Any]:
        """检测数据量异常"""
        try:
            current_volume = self._get_data_size(data)
            
            # 获取历史数据量
            historical_volumes = self._get_historical_volumes(data_id)
            
            if len(historical_volumes) == 0:
                # 第一次记录，设为正常
                self._record_volume(data_id, current_volume)
                return {
                    'is_anomaly': False,
                    'score': 0.0,
                    'method': 'baseline',
                    'reason': '首次记录，建立基线'
                }
            
            # 计算历史统计信息
            mean_volume = statistics.mean(historical_volumes)
            std_volume = statistics.stdev(historical_volumes) if len(historical_volumes) > 1 else 0
            
            # 计算Z-score
            if std_volume > 0:
                z_score = abs(current_volume - mean_volume) / std_volume
                is_anomaly = z_score > 3
                anomaly_score = min(1.0, z_score / 3)
            else:
                # 标准差为0，使用简单的比例检测
                ratio = current_volume / mean_volume if mean_volume > 0 else 1
                is_anomaly = ratio > 2 or ratio < 0.5
                anomaly_score = abs(ratio - 1) if ratio > 1 else (1 - ratio)
            
            # 记录当前数据量
            self._record_volume(data_id, current_volume)
            
            return {
                'is_anomaly': is_anomaly,
                'score': anomaly_score,
                'method': 'volume_zscore',
                'current_volume': current_volume,
                'mean_volume': mean_volume,
                'std_volume': std_volume,
                'z_score': z_score if std_volume > 0 else 0,
                'volume_ratio': current_volume / mean_volume if mean_volume > 0 else 1
            }
            
        except Exception as e:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def _detect_quality_anomalies(self, data: Any) -> Dict[str, Any]:
        """检测数据质量异常"""
        try:
            # 使用质量监控功能检测异常
            metrics = self._calculate_quality_metrics(data)
            
            # 计算质量异常评分
            quality_scores = [
                metrics.completeness,
                metrics.accuracy,
                metrics.consistency,
                metrics.timeliness,
                metrics.validity,
                metrics.uniqueness
            ]
            
            avg_quality = statistics.mean(quality_scores)
            min_quality = min(quality_scores)
            
            # 如果平均质量或最低质量过低，认为是异常
            is_anomaly = avg_quality < 0.7 or min_quality < 0.5
            anomaly_score = 1.0 - avg_quality
            
            return {
                'is_anomaly': is_anomaly,
                'score': anomaly_score,
                'method': 'quality_threshold',
                'average_quality': avg_quality,
                'minimum_quality': min_quality,
                'quality_breakdown': {
                    'completeness': metrics.completeness,
                    'accuracy': metrics.accuracy,
                    'consistency': metrics.consistency,
                    'timeliness': metrics.timeliness,
                    'validity': metrics.validity,
                    'uniqueness': metrics.uniqueness
                }
            }
            
        except Exception as e:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def _get_data_size(self, data: Any) -> int:
        """获取数据大小"""
        try:
            if isinstance(data, pd.DataFrame):
                return len(data)
            elif isinstance(data, (list, dict)):
                return len(data)
            else:
                return 1
        except Exception:
            return 0
    
    def _get_historical_volumes(self, data_id: str) -> List[int]:
        """获取历史数据量"""
        # 简化处理：从监控历史中获取
        volumes = []
        for result in self.monitor_history:
            if result.monitor_type == "data_volume" and result.details.get('data_id') == data_id:
                volume_metrics = result.details.get('volume_metrics', {})
                record_count = volume_metrics.get('record_count', 0)
                volumes.append(record_count)
        return volumes
    
    def _record_volume(self, data_id: str, volume: int):
        """记录数据量"""
        # 简化处理：记录到内存中
        if not hasattr(self, '_volume_history'):
            self._volume_history = {}
        
        if data_id not in self._volume_history:
            self._volume_history[data_id] = []
        
        self._volume_history[data_id].append(volume)
        
        # 限制历史记录数量
        max_history = 100
        if len(self._volume_history[data_id]) > max_history:
            self._volume_history[data_id] = self._volume_history[data_id][-max_history:]
    
    # ==================== 数据监控报告 ====================
    
    def generate_monitoring_report(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        生成数据监控报告
        
        Args:
            time_range: 时间范围，如果为None则包含所有历史记录
            
        Returns:
            Dict[str, Any]: 监控报告
        """
        try:
            self.logger.info("开始生成数据监控报告")
            
            # 过滤时间范围内的监控结果
            if time_range:
                cutoff_time = datetime.now() - time_range
                filtered_results = [
                    result for result in self.monitor_history
                    if result.timestamp >= cutoff_time
                ]
            else:
                filtered_results = self.monitor_history
            
            # 统计各种监控类型的执行情况
            monitor_stats = self._calculate_monitor_statistics(filtered_results)
            
            # 分析监控趋势
            trend_analysis = self._analyze_monitoring_trends(filtered_results)
            
            # 生成告警汇总
            alert_summary = self._generate_alert_summary(filtered_results)
            
            # 生成建议汇总
            recommendations_summary = self._generate_recommendations_summary(filtered_results)
            
            # 生成数据质量总览
            quality_overview = self._generate_quality_overview(filtered_results)
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'time_range': time_range.__str__() if time_range else 'all_time',
                    'total_monitor_results': len(filtered_results),
                    'report_version': '1.0'
                },
                'monitor_statistics': monitor_stats,
                'trend_analysis': trend_analysis,
                'alert_summary': alert_summary,
                'quality_overview': quality_overview,
                'recommendations_summary': recommendations_summary,
                'detailed_results': [
                    {
                        'monitor_type': result.monitor_type,
                        'status': result.status,
                        'score': result.score,
                        'message': result.message,
                        'timestamp': result.timestamp.isoformat(),
                        'alert_level': result.alert_level.value,
                        'recommendations': result.recommendations
                    }
                    for result in filtered_results[-100:]  # 只包含最近100条详细结果
                ]
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成监控报告失败: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0'
            }
    
    def _calculate_monitor_statistics(self, results: List[MonitorResult]) -> Dict[str, Any]:
        """计算监控统计信息"""
        if not results:
            return {'error': '没有监控结果数据'}
        
        # 按监控类型分组
        by_type = defaultdict(list)
        for result in results:
            by_type[result.monitor_type].append(result)
        
        stats = {}
        
        for monitor_type, type_results in by_type.items():
            # 计算评分统计
            scores = [r.score for r in type_results]
            
            # 计算状态分布
            status_counts = Counter([r.status for r in type_results])
            
            # 计算告警级别分布
            alert_counts = Counter([r.alert_level.value for r in type_results])
            
            stats[monitor_type] = {
                'total_checks': len(type_results),
                'average_score': statistics.mean(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'score_std': statistics.stdev(scores) if len(scores) > 1 else 0,
                'status_distribution': dict(status_counts),
                'alert_level_distribution': dict(alert_counts),
                'latest_result': {
                    'score': type_results[-1].score,
                    'status': type_results[-1].status,
                    'timestamp': type_results[-1].timestamp.isoformat()
                }
            }
        
        return stats
    
    def _analyze_monitoring_trends(self, results: List[MonitorResult]) -> Dict[str, Any]:
        """分析监控趋势"""
        if len(results) < 2:
            return {'error': '数据点不足，无法分析趋势'}
        
        # 按时间排序
        sorted_results = sorted(results, key=lambda x: x.timestamp)
        
        # 分析每个监控类型的趋势
        trends = {}
        
        monitor_types = set([r.monitor_type for r in sorted_results])
        
        for monitor_type in monitor_types:
            type_results = [r for r in sorted_results if r.monitor_type == monitor_type]
            
            if len(type_results) < 2:
                continue
            
            # 计算评分趋势
            scores = [r.score for r in type_results]
            timestamps = [r.timestamp for r in type_results]
            
            # 简单的线性趋势计算
            if len(scores) >= 3:
                # 使用简单线性回归计算趋势
                x_values = list(range(len(scores)))
                n = len(scores)
                
                sum_x = sum(x_values)
                sum_y = sum(scores)
                sum_xy = sum(x * y for x, y in zip(x_values, scores))
                sum_x2 = sum(x * x for x in x_values)
                
                # 计算斜率
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                if slope > 0.1:
                    trend_direction = "improving"
                elif slope < -0.1:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
                
                trends[monitor_type] = {
                    'trend_direction': trend_direction,
                    'slope': slope,
                    'score_range': {
                        'min': min(scores),
                        'max': max(scores),
                        'latest': scores[-1]
                    },
                    'data_points': len(scores)
                }
        
        return trends
    
    def _generate_alert_summary(self, results: List[MonitorResult]) -> Dict[str, Any]:
        """生成告警汇总"""
        # 统计各级别告警数量
        alert_counts = Counter([r.alert_level.value for r in results])
        
        # 统计各状态数量
        status_counts = Counter([r.status for r in results])
        
        # 获取最近的告警
        recent_alerts = [
            {
                'monitor_type': r.monitor_type,
                'alert_level': r.alert_level.value,
                'message': r.message,
                'timestamp': r.timestamp.isoformat(),
                'score': r.score
            }
            for r in sorted(results, key=lambda x: x.timestamp, reverse=True)
            if r.alert_level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        ][:10]  # 最近10个告警
        
        return {
            'total_alerts': sum(alert_counts.values()),
            'alert_level_distribution': dict(alert_counts),
            'status_distribution': dict(status_counts),
            'recent_alerts': recent_alerts,
            'most_common_alert_level': alert_counts.most_common(1)[0][0] if alert_counts else 'none'
        }
    
    def _generate_recommendations_summary(self, results: List[MonitorResult]) -> Dict[str, Any]:
        """生成建议汇总"""
        all_recommendations = []
        
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # 统计建议频率
        recommendation_counts = Counter(all_recommendations)
        
        # 按监控类型分组建议
        recommendations_by_type = defaultdict(list)
        for result in results:
            recommendations_by_type[result.monitor_type].extend(result.recommendations)
        
        return {
            'total_recommendations': len(all_recommendations),
            'unique_recommendations': len(set(all_recommendations)),
            'most_common_recommendations': dict(recommendation_counts.most_common(10)),
            'recommendations_by_monitor_type': {
                k: list(set(v)) for k, v in recommendations_by_type.items()
            }
        }
    
    def _generate_quality_overview(self, results: List[MonitorResult]) -> Dict[str, Any]:
        """生成数据质量总览"""
        quality_results = [r for r in results if r.monitor_type == 'data_quality']
        
        if not quality_results:
            return {'error': '没有数据质量监控结果'}
        
        # 提取质量指标
        quality_metrics = []
        for result in quality_results:
            if 'metrics' in result.details:
                metrics = result.details['metrics']
                metrics['timestamp'] = result.timestamp.isoformat()
                metrics['score'] = result.score
                quality_metrics.append(metrics)
        
        if not quality_metrics:
            return {'error': '没有有效的质量指标数据'}
        
        # 计算质量指标统计
        metric_names = ['completeness', 'accuracy', 'consistency', 'timeliness', 'validity', 'uniqueness']
        
        quality_stats = {}
        for metric in metric_names:
            values = [m.get(metric, 0) for m in quality_metrics if metric in m]
            if values:
                quality_stats[metric] = {
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'latest': values[-1] if values else 0
                }
        
        # 总体质量趋势
        overall_scores = [m.get('overall_score', 0) for m in quality_metrics if 'overall_score' in m]
        
        return {
            'total_quality_checks': len(quality_results),
            'quality_metrics_statistics': quality_stats,
            'overall_quality_trend': {
                'average_score': statistics.mean(overall_scores) if overall_scores else 0,
                'latest_score': overall_scores[-1] if overall_scores else 0,
                'score_range': {
                    'min': min(overall_scores) if overall_scores else 0,
                    'max': max(overall_scores) if overall_scores else 0
                }
            },
            'quality_grade_distribution': self._calculate_quality_grades(overall_scores)
        }
    
    def _calculate_quality_grades(self, scores: List[float]) -> Dict[str, str]:
        """计算质量等级分布"""
        if not scores:
            return {}
        
        grade_counts = defaultdict(int)
        
        for score in scores:
            if score >= 90:
                grade_counts['excellent'] += 1
            elif score >= 80:
                grade_counts['good'] += 1
            elif score >= 70:
                grade_counts['fair'] += 1
            elif score >= 60:
                grade_counts['poor'] += 1
            else:
                grade_counts['critical'] += 1
        
        total = len(scores)
        return {
            grade: f"{count} ({count/total*100:.1f}%)"
            for grade, count in grade_counts.items()
        }
    
    # ==================== 辅助方法 ====================
    
    def _register_data(self, data_id: str, data: Any):
        """注册数据到数据注册表"""
        try:
            registry_entry = {
                'data_id': data_id,
                'data_type': type(data).__name__,
                'source': 'unknown',
                'schema': self._extract_schema(data),
                'last_updated': datetime.now().isoformat(),
                'metadata': {
                    'size': self._get_data_size(data),
                    'registered_at': datetime.now().isoformat()
                }
            }
            
            with self._lock:
                self.data_registry[data_id] = registry_entry
            
            # 保存到数据库
            self._save_to_database(data_id, registry_entry)
            
        except Exception as e:
            self.logger.error(f"数据注册失败: {e}")
    
    def _extract_schema(self, data: Any) -> Dict[str, Any]:
        """提取数据模式"""
        try:
            if isinstance(data, pd.DataFrame):
                schema = {
                    'type': 'DataFrame',
                    'columns': [
                        {
                            'name': col,
                            'dtype': str(data[col].dtype),
                            'non_null_count': int(data[col].count()),
                            'null_count': int(data[col].isnull().sum())
                        }
                        for col in data.columns
                    ]
                }
            elif isinstance(data, dict):
                schema = {
                    'type': 'dict',
                    'fields': {
                        k: {
                            'type': type(v).__name__,
                            'nullable': v is None
                        }
                        for k, v in data.items()
                    }
                }
            else:
                schema = {
                    'type': type(data).__name__,
                    'serializable': True
                }
            
            return schema
            
        except Exception as e:
            self.logger.error(f"模式提取失败: {e}")
            return {'type': 'unknown', 'error': str(e)}
    
    def _save_monitor_result(self, result: MonitorResult):
        """保存监控结果"""
        try:
            with self._lock:
                self.monitor_history.append(result)
                
                # 限制历史记录数量
                max_history = self.config['monitoring']['max_history_records']
                if len(self.monitor_history) > max_history:
                    self.monitor_history = self.monitor_history[-max_history:]
            
            # 保存到数据库
            self._save_result_to_database(result)
            
        except Exception as e:
            self.logger.error(f"保存监控结果失败: {e}")
    
    def _save_to_database(self, data_id: str, registry_entry: Dict[str, Any]):
        """保存到数据库"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO data_registry 
                (data_id, data_type, source, schema, last_updated, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data_id,
                registry_entry['data_type'],
                registry_entry['source'],
                json.dumps(registry_entry['schema']),
                registry_entry['last_updated'],
                json.dumps(registry_entry['metadata'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"数据库保存失败: {e}")
    
    def _save_result_to_database(self, result: MonitorResult):
        """保存结果到数据库"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO monitor_history 
                (monitor_type, status, score, message, timestamp, details, alert_level, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.monitor_type,
                result.status,
                result.score,
                result.message,
                result.timestamp.isoformat(),
                json.dumps(result.details),
                result.alert_level.value,
                json.dumps(result.recommendations)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"结果数据库保存失败: {e}")
    
    def _trigger_alerts(self, result: MonitorResult):
        """触发告警"""
        try:
            for callback in self.alert_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"告警回调执行失败: {e}")
            
            # 记录告警
            self.logger.warning(f"监控告警: {result.monitor_type} - {result.message}")
            
        except Exception as e:
            self.logger.error(f"告警触发失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[MonitorResult], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_history(self, monitor_type: Optional[str] = None, limit: int = 100) -> List[MonitorResult]:
        """获取监控历史"""
        try:
            results = self.monitor_history
            
            if monitor_type:
                results = [r for r in results if r.monitor_type == monitor_type]
            
            # 按时间倒序排列
            results = sorted(results, key=lambda x: x.timestamp, reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"获取监控历史失败: {e}")
            return []
    
    def export_report(self, report: Dict[str, Any], file_path: str, format: str = 'json'):
        """导出报告"""
        try:
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'txt':
                self._export_text_report(report, file_path)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"报告已导出到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"报告导出失败: {e}")
            raise
    
    def _export_text_report(self, report: Dict[str, Any], file_path: str):
        """导出文本格式报告"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("数据监控报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 报告元数据
            metadata = report.get('report_metadata', {})
            f.write(f"生成时间: {metadata.get('generated_at', 'N/A')}\n")
            f.write(f"时间范围: {metadata.get('time_range', 'N/A')}\n")
            f.write(f"总监控结果: {metadata.get('total_monitor_results', 0)}\n\n")
            
            # 监控统计
            stats = report.get('monitor_statistics', {})
            f.write("监控统计\n")
            f.write("-" * 30 + "\n")
            for monitor_type, stat in stats.items():
                f.write(f"{monitor_type}:\n")
                f.write(f"  总检查次数: {stat.get('total_checks', 0)}\n")
                f.write(f"  平均评分: {stat.get('average_score', 0):.2f}\n")
                f.write(f"  状态分布: {stat.get('status_distribution', {})}\n\n")
            
            # 告警汇总
            alert_summary = report.get('alert_summary', {})
            f.write("告警汇总\n")
            f.write("-" * 30 + "\n")
            f.write(f"总告警数: {alert_summary.get('total_alerts', 0)}\n")
            f.write(f"告警级别分布: {alert_summary.get('alert_level_distribution', {})}\n\n")
            
            # 建议汇总
            recommendations = report.get('recommendations_summary', {})
            f.write("建议汇总\n")
            f.write("-" * 30 + "\n")
            f.write(f"总建议数: {recommendations.get('total_recommendations', 0)}\n")
            common_recs = recommendations.get('most_common_recommendations', {})
            for rec, count in common_recs.items():
                f.write(f"  {rec}: {count}\n")
    
    def cleanup_old_records(self, retention_days: int = 30):
        """清理旧记录"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # 清理监控历史
            with self._lock:
                self.monitor_history = [
                    r for r in self.monitor_history 
                    if r.timestamp >= cutoff_date
                ]
            
            # 清理访问日志
            self.access_log = [
                log for log in self.access_log
                if datetime.fromisoformat(log['timestamp']) >= cutoff_date
            ]
            
            self.logger.info(f"已清理 {retention_days} 天前的记录")
            
        except Exception as e:
            self.logger.error(f"清理旧记录失败: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, 'db_path') and self.db_path.exists():
                # 清理临时数据库文件
                pass
        except Exception:
            pass


# ==================== 测试用例 ====================

def test_data_monitor():
    """数据监控器测试函数"""
    print("开始数据监控器测试...")
    
    # 创建监控器实例
    monitor = DataMonitor()
    
    # 测试数据
    test_data = pd.DataFrame({
        'id': range(100),
        'name': [f'user_{i}' for i in range(100)],
        'age': np.random.normal(30, 10, 100),
        'email': [f'user_{i}@example.com' for i in range(100)],
        'score': np.random.uniform(0, 100, 100)
    })
    
    # 添加一些异常数据
    test_data.loc[0, 'age'] = -5  # 异常年龄
    test_data.loc[1, 'email'] = ''  # 空邮箱
    test_data.loc[2:5, 'name'] = None  # 缺失姓名
    
    print("1. 测试数据质量监控...")
    quality_result = monitor.monitor_data_quality(test_data, "test_dataset")
    print(f"质量监控结果: {quality_result.status}, 评分: {quality_result.score:.2f}")
    
    print("\n2. 测试数据完整性监控...")
    expected_schema = {
        'fields': {
            'id': 'int',
            'name': 'str',
            'age': 'float',
            'email': 'str',
            'score': 'float'
        }
    }
    integrity_result = monitor.monitor_data_integrity(test_data, expected_schema)
    print(f"完整性监控结果: {integrity_result.status}, 评分: {integrity_result.score:.2f}")
    
    print("\n3. 测试数据一致性监控...")
    test_data2 = test_data.copy()
    test_data2.loc[0, 'age'] = 35  # 修改一个值
    consistency_result = monitor.monitor_data_consistency([test_data, test_data2])
    print(f"一致性监控结果: {consistency_result.status}, 评分: {consistency_result.score:.2f}")
    
    print("\n4. 测试数据及时性监控...")
    timeliness_result = monitor.monitor_data_timeliness("test_dataset", timedelta(hours=1))
    print(f"及时性监控结果: {timeliness_result.status}, 评分: {timeliness_result.score:.2f}")
    
    print("\n5. 测试数据量监控...")
    volume_result = monitor.monitor_data_volume(test_data, "test_dataset")
    print(f"数据量监控结果: {volume_result.status}, 评分: {volume_result.score:.2f}")
    
    print("\n6. 测试数据安全监控...")
    security_result = monitor.monitor_data_security(test_data, "test_dataset")
    print(f"安全监控结果: {security_result.status}, 评分: {security_result.score:.2f}")
    
    print("\n7. 测试数据访问监控...")
    access_result = monitor.monitor_data_access("user_123", "test_dataset", "read")
    print(f"访问监控结果: {access_result.status}, 评分: {access_result.score:.2f}")
    
    print("\n8. 测试数据异常检测...")
    anomaly_result = monitor.detect_data_anomalies(test_data, "test_dataset")
    print(f"异常检测结果: {'异常' if anomaly_result.is_anomaly else '正常'}, "
          f"异常评分: {anomaly_result.anomaly_score:.2f}, "
          f"异常类型: {anomaly_result.anomaly_type}")
    
    print("\n9. 测试监控报告生成...")
    report = monitor.generate_monitoring_report(timedelta(days=1))
    print(f"报告生成成功，包含 {len(report.get('detailed_results', []))} 条详细结果")
    
    print("\n10. 测试报告导出...")
    monitor.export_report(report, "monitoring_report.json", "json")
    monitor.export_report(report, "monitoring_report.txt", "txt")
    print("报告导出完成")
    
    print("\n数据监控器测试完成！")


if __name__ == "__main__":
    # 运行测试
    test_data_monitor()