"""
V6模型监控器
实现模型性能监控、数据漂移检测、模型衰减监控、预测质量监控等功能
"""

import json
import sqlite3
import logging
import warnings
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import pickle


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """监控配置类"""
    # 性能监控配置
    performance_threshold: float = 0.8  # 性能阈值
    performance_window_size: int = 100  # 性能评估窗口大小
    
    # 数据漂移检测配置
    drift_threshold: float = 0.05  # 漂移阈值
    drift_test_size: int = 1000  # 漂移检测样本大小
    drift_confidence_level: float = 0.95  # 置信水平
    
    # 模型衰减监控配置
    degradation_threshold: float = 0.1  # 衰减阈值
    degradation_check_interval: int = 3600  # 检查间隔(秒)
    
    # 预测质量监控配置
    quality_threshold: float = 0.75  # 质量阈值
    quality_metrics: List[str] = None  # 质量指标
    
    # 异常检测配置
    anomaly_threshold: float = 3.0  # 异常检测阈值(标准差倍数)
    anomaly_window_size: int = 50  # 异常检测窗口大小
    
    # 数据存储配置
    db_path: str = "model_monitor.db"  # 数据库路径
    retention_days: int = 30  # 数据保留天数
    
    # 告警配置
    alert_email: str = ""  # 告警邮箱
    alert_enabled: bool = True  # 是否启用告警
    alert_cooldown: int = 300  # 告警冷却时间(秒)
    
    # 仪表板配置
    dashboard_update_interval: int = 60  # 仪表板更新间隔(秒)
    dashboard_history_size: int = 1000  # 历史数据大小
    
    def __post_init__(self):
        """初始化后处理"""
        if self.quality_metrics is None:
            self.quality_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']


@dataclass
class MonitoringResult:
    """监控结果类"""
    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold: float
    status: str  # 'normal', 'warning', 'critical'
    details: Dict[str, Any]
    model_id: str


@dataclass
class AnomalyAlert:
    """异常告警类"""
    timestamp: datetime
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric_value: float
    threshold: float
    model_id: str
    details: Dict[str, Any]


class DataDriftDetector:
    """数据漂移检测器"""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初始化数据漂移检测器
        
        Args:
            confidence_level: 置信水平
        """
        self.confidence_level = confidence_level
        self.baseline_data = None
        self.baseline_stats = {}
        
    def set_baseline(self, data: np.ndarray) -> None:
        """
        设置基准数据
        
        Args:
            data: 基准数据数组
        """
        self.baseline_data = data
        self.baseline_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0)
        }
        
    def detect_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """
        检测数据漂移
        
        Args:
            current_data: 当前数据
            
        Returns:
            漂移检测结果
        """
        if self.baseline_data is None:
            raise ValueError("请先设置基准数据")
            
        results = {}
        
        # 统计测试
        for i in range(current_data.shape[1]):
            current_col = current_data[:, i]
            baseline_col = self.baseline_data[:, i]
            
            # Kolmogorov-Smirnov测试
            ks_stat, ks_p_value = stats.ks_2samp(baseline_col, current_col)
            
            # Welch's t-test
            t_stat, t_p_value = stats.ttest_ind(baseline_col, current_col, equal_var=False)
            
            # Mann-Whitney U test
            mw_stat, mw_p_value = stats.mannwhitneyu(baseline_col, current_col, alternative='two-sided')
            
            results[f'feature_{i}'] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                't_statistic': t_stat,
                't_p_value': t_p_value,
                'mw_statistic': mw_stat,
                'mw_p_value': mw_p_value,
                'drift_detected': ks_p_value < (1 - self.confidence_level)
            }
        
        # 总体漂移分数
        drift_scores = [result['ks_statistic'] for result in results.values()]
        overall_drift_score = np.mean(drift_scores)
        
        results['overall'] = {
            'drift_score': overall_drift_score,
            'drift_detected': overall_drift_score > 0.1,
            'features_drifted': sum(1 for r in results.values() if r['drift_detected']),
            'total_features': len(results)
        }
        
        return results


class ModelDegradationDetector:
    """模型衰减检测器"""
    
    def __init__(self, degradation_threshold: float = 0.1):
        """
        初始化模型衰减检测器
        
        Args:
            degradation_threshold: 衰减阈值
        """
        self.degradation_threshold = degradation_threshold
        self.performance_history = deque(maxlen=100)
        self.baseline_performance = None
        
    def set_baseline(self, performance_score: float) -> None:
        """
        设置基准性能
        
        Args:
            performance_score: 基准性能分数
        """
        self.baseline_performance = performance_score
        self.performance_history.append(performance_score)
        
    def detect_degradation(self, current_performance: float) -> Dict[str, Any]:
        """
        检测模型衰减
        
        Args:
            current_performance: 当前性能
            
        Returns:
            衰减检测结果
        """
        self.performance_history.append(current_performance)
        
        if self.baseline_performance is None:
            self.set_baseline(current_performance)
            return {'degradation_detected': False, 'degradation_score': 0.0}
        
        # 计算衰减分数
        degradation_score = (self.baseline_performance - current_performance) / self.baseline_performance
        
        # 趋势分析
        if len(self.performance_history) >= 5:
            recent_scores = list(self.performance_history)[-5:]
            trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        else:
            trend_slope = 0
            
        # 衰减检测
        degradation_detected = degradation_score > self.degradation_threshold
        trend_degradation = trend_slope < -0.01
        
        return {
            'degradation_score': degradation_score,
            'degradation_detected': degradation_detected,
            'trend_slope': trend_slope,
            'trend_degradation': trend_degradation,
            'current_performance': current_performance,
            'baseline_performance': self.baseline_performance,
            'performance_drop': self.baseline_performance - current_performance
        }


class PredictionQualityMonitor:
    """预测质量监控器"""
    
    def __init__(self, quality_threshold: float = 0.75):
        """
        初始化预测质量监控器
        
        Args:
            quality_threshold: 质量阈值
        """
        self.quality_threshold = quality_threshold
        self.prediction_history = deque(maxlen=1000)
        self.quality_scores = {}
        
    def evaluate_prediction_quality(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray, 
                                  y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估预测质量
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            
        Returns:
            质量指标字典
        """
        quality_metrics = {}
        
        # 分类指标
        if len(np.unique(y_true)) == 2:  # 二分类
            quality_metrics['accuracy'] = accuracy_score(y_true, y_pred)
            quality_metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            quality_metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            quality_metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            
            if y_prob is not None:
                # 对于二分类，使用正类的概率
                quality_metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:  # 多分类
            quality_metrics['accuracy'] = accuracy_score(y_true, y_pred)
            quality_metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            quality_metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            quality_metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            if y_prob is not None:
                quality_metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        
        # 置信度分析
        if y_prob is not None:
            max_probs = np.max(y_prob, axis=1)
            quality_metrics['avg_confidence'] = np.mean(max_probs)
            quality_metrics['low_confidence_ratio'] = np.mean(max_probs < 0.5)
        
        # 保存历史记录
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'metrics': quality_metrics
        })
        
        return quality_metrics
    
    def get_quality_trend(self, window_size: int = 50) -> Dict[str, Any]:
        """
        获取质量趋势
        
        Args:
            window_size: 窗口大小
            
        Returns:
            趋势分析结果
        """
        if len(self.prediction_history) < window_size:
            return {'trend': 'insufficient_data'}
        
        recent_predictions = list(self.prediction_history)[-window_size:]
        
        # 计算趋势
        trends = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            scores = [pred['metrics'].get(metric, 0) for pred in recent_predictions if metric in pred['metrics']]
            if len(scores) >= 10:
                trend_slope = np.polyfit(range(len(scores)), scores, 1)[0]
                trends[metric] = {
                    'slope': trend_slope,
                    'direction': 'improving' if trend_slope > 0.001 else 'declining' if trend_slope < -0.001 else 'stable'
                }
        
        return {
            'trends': trends,
            'window_size': window_size,
            'data_points': len(recent_predictions)
        }


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, threshold: float = 3.0, window_size: int = 50):
        """
        初始化异常检测器
        
        Args:
            threshold: 异常阈值(标准差倍数)
            window_size: 窗口大小
        """
        self.threshold = threshold
        self.window_size = window_size
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
        
    def detect_anomalies(self, metric_name: str, metric_value: float) -> Dict[str, Any]:
        """
        检测异常
        
        Args:
            metric_name: 指标名称
            metric_value: 指标值
            
        Returns:
            异常检测结果
        """
        self.data_windows[metric_name].append(metric_value)
        
        if len(self.data_windows[metric_name]) < 10:
            return {'anomaly_detected': False, 'reason': 'insufficient_data'}
        
        data = np.array(self.data_windows[metric_name])
        
        # 计算统计量
        mean_val = np.mean(data)
        std_val = np.std(data)
        z_score = abs(metric_value - mean_val) / std_val if std_val > 0 else 0
        
        # 异常检测
        is_anomaly = z_score > self.threshold
        
        # 分位数分析
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        is_outlier = metric_value < lower_bound or metric_value > upper_bound
        
        return {
            'anomaly_detected': is_anomaly,
            'outlier_detected': is_outlier,
            'z_score': z_score,
            'mean': mean_val,
            'std': std_val,
            'threshold': self.threshold,
            'iqr_bounds': (lower_bound, upper_bound),
            'severity': self._calculate_severity(z_score)
        }
    
    def _calculate_severity(self, z_score: float) -> str:
        """计算异常严重程度"""
        if z_score > 5:
            return 'critical'
        elif z_score > 4:
            return 'high'
        elif z_score > 3:
            return 'medium'
        else:
            return 'low'


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: MonitoringConfig):
        """
        初始化告警管理器
        
        Args:
            config: 监控配置
        """
        self.config = config
        self.alert_history = deque(maxlen=1000)
        self.last_alert_times = {}
        
    def send_alert(self, alert: AnomalyAlert) -> bool:
        """
        发送告警
        
        Args:
            alert: 告警对象
            
        Returns:
            是否成功发送
        """
        # 检查冷却时间
        alert_key = f"{alert.alert_type}_{alert.model_id}"
        current_time = datetime.now()
        
        if alert_key in self.last_alert_times:
            time_diff = (current_time - self.last_alert_times[alert_key]).total_seconds()
            if time_diff < self.config.alert_cooldown:
                logger.info(f"告警冷却中，跳过发送: {alert_key}")
                return False
        
        # 记录告警历史
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = current_time
        
        # 发送邮件告警
        if self.config.alert_enabled and self.config.alert_email:
            try:
                self._send_email_alert(alert)
                logger.info(f"邮件告警已发送: {alert.message}")
            except Exception as e:
                logger.error(f"发送邮件告警失败: {e}")
        
        # 记录日志
        logger.warning(f"告警: {alert.severity.upper()} - {alert.message}")
        
        return True
    
    def _send_email_alert(self, alert: AnomalyAlert) -> None:
        """
        发送邮件告警
        
        Args:
            alert: 告警对象
        """
        # 注意：实际使用时需要配置SMTP服务器
        msg = MIMEMultipart()
        msg['From'] = "model-monitor@example.com"
        msg['To'] = self.config.alert_email
        msg['Subject'] = f"[{alert.severity.upper()}] 模型监控告警"
        
        body = f"""
        模型监控告警通知
        
        告警类型: {alert.alert_type}
        严重程度: {alert.severity}
        模型ID: {alert.model_id}
        时间: {alert.timestamp}
        
        详情:
        {alert.message}
        
        指标值: {alert.metric_value}
        阈值: {alert.threshold}
        
        详细信息:
        {json.dumps(alert.details, indent=2, ensure_ascii=False)}
        """
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 这里需要实际的SMTP服务器配置
        # server = smtplib.SMTP('smtp.example.com', 587)
        # server.starttls()
        # server.login("username", "password")
        # server.send_message(msg)
        # server.quit()
        
        logger.info(f"邮件告警内容已准备: {alert.message}")


class MonitoringStorage:
    """监控数据存储"""
    
    def __init__(self, db_path: str):
        """
        初始化存储管理器
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.init_database()
        
    def init_database(self) -> None:
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    auc REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_drift_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    drift_score REAL,
                    p_value REAL,
                    drift_detected BOOLEAN,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_monitoring_result(self, result: MonitoringResult) -> None:
        """
        保存监控结果
        
        Args:
            result: 监控结果
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO monitoring_results 
                (timestamp, model_id, metric_name, metric_value, threshold, status, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.timestamp.isoformat(),
                result.model_id,
                result.metric_name,
                result.metric_value,
                result.threshold,
                result.status,
                json.dumps(result.details, ensure_ascii=False, default=str)
            ))
            conn.commit()
    
    def save_anomaly_alert(self, alert: AnomalyAlert) -> None:
        """
        保存异常告警
        
        Args:
            alert: 告警对象
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO anomaly_alerts 
                (timestamp, model_id, alert_type, severity, message, metric_value, threshold, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp.isoformat(),
                alert.model_id,
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.metric_value,
                alert.threshold,
                json.dumps(alert.details, ensure_ascii=False, default=str)
            ))
            conn.commit()
    
    def get_monitoring_history(self, 
                             model_id: str, 
                             metric_name: str, 
                             hours: int = 24) -> List[Dict[str, Any]]:
        """
        获取监控历史数据
        
        Args:
            model_id: 模型ID
            metric_name: 指标名称
            hours: 小时数
            
        Returns:
            历史数据列表
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT timestamp, metric_value, threshold, status, details
                FROM monitoring_results
                WHERE model_id = ? AND metric_name = ?
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp
            '''.format(hours), (model_id, metric_name))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'timestamp': row[0],
                    'metric_value': row[1],
                    'threshold': row[2],
                    'status': row[3],
                    'details': json.loads(row[4]) if row[4] else {}
                })
            
            return results


class MonitoringDashboard:
    """监控仪表板"""
    
    def __init__(self, storage: MonitoringStorage, config: MonitoringConfig):
        """
        初始化监控仪表板
        
        Args:
            storage: 存储管理器
            config: 监控配置
        """
        self.storage = storage
        self.config = config
        self.dashboard_data = {}
        
    def generate_dashboard_data(self, model_id: str) -> Dict[str, Any]:
        """
        生成仪表板数据
        
        Args:
            model_id: 模型ID
            
        Returns:
            仪表板数据
        """
        dashboard_data = {
            'model_id': model_id,
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': {},
            'drift_analysis': {},
            'anomaly_summary': {},
            'alerts': [],
            'trends': {}
        }
        
        # 获取性能指标
        for metric in self.config.quality_metrics:
            history = self.storage.get_monitoring_history(model_id, metric, hours=24)
            if history:
                values = [h['metric_value'] for h in history]
                dashboard_data['performance_metrics'][metric] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'declining'
                }
        
        # 获取告警信息
        with sqlite3.connect(self.storage.db_path) as conn:
            cursor = conn.execute('''
                SELECT timestamp, alert_type, severity, message
                FROM anomaly_alerts
                WHERE model_id = ? AND timestamp >= datetime('now', '-24 hours')
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (model_id,))
            
            dashboard_data['alerts'] = [
                {
                    'timestamp': row[0],
                    'type': row[1],
                    'severity': row[2],
                    'message': row[3]
                }
                for row in cursor.fetchall()
            ]
        
        return dashboard_data
    
    def create_performance_chart(self, 
                               model_id: str, 
                               metric_name: str, 
                               hours: int = 24) -> str:
        """
        创建性能图表
        
        Args:
            model_id: 模型ID
            metric_name: 指标名称
            hours: 小时数
            
        Returns:
            图表文件路径
        """
        history = self.storage.get_monitoring_history(model_id, metric_name, hours)
        
        if not history:
            return ""
        
        timestamps = [datetime.fromisoformat(h['timestamp']) for h in history]
        values = [h['metric_value'] for h in history]
        thresholds = [h['threshold'] for h in history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, label=metric_name, linewidth=2)
        plt.plot(timestamps, thresholds, label='阈值', linestyle='--', alpha=0.7)
        plt.title(f'{metric_name} 监控趋势 - {model_id}')
        plt.xlabel('时间')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = f"dashboard_{model_id}_{metric_name}_{int(time.time())}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path


class ModelMonitor:
    """V6模型监控器主类"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        初始化模型监控器
        
        Args:
            config: 监控配置
        """
        self.config = config or MonitoringConfig()
        
        # 初始化各个组件
        self.drift_detector = DataDriftDetector(self.config.drift_confidence_level)
        self.degradation_detector = ModelDegradationDetector(self.config.degradation_threshold)
        self.quality_monitor = PredictionQualityMonitor(self.config.quality_threshold)
        self.anomaly_detector = AnomalyDetector(self.config.anomaly_threshold, self.config.anomaly_window_size)
        self.alert_manager = AlertManager(self.config)
        self.storage = MonitoringStorage(self.config.db_path)
        self.dashboard = MonitoringDashboard(self.storage, self.config)
        
        # 监控状态
        self.is_monitoring = False
        self.monitored_models = {}
        self.monitoring_thread = None
        
        # 回调函数
        self.performance_callbacks = []
        self.drift_callbacks = []
        self.alert_callbacks = []
        
        logger.info("V6模型监控器初始化完成")
    
    def register_model(self, 
                      model_id: str, 
                      baseline_data: Optional[np.ndarray] = None,
                      baseline_performance: Optional[float] = None,
                      model_type: str = "classifier") -> None:
        """
        注册模型进行监控
        
        Args:
            model_id: 模型ID
            baseline_data: 基准数据
            baseline_performance: 基准性能
            model_type: 模型类型
        """
        self.monitored_models[model_id] = {
            'model_type': model_type,
            'registered_at': datetime.now(),
            'last_prediction_time': None,
            'prediction_count': 0,
            'status': 'active'
        }
        
        # 设置基准数据
        if baseline_data is not None:
            self.drift_detector.set_baseline(baseline_data)
            logger.info(f"已为模型 {model_id} 设置基准数据")
        
        # 设置基准性能
        if baseline_performance is not None:
            self.degradation_detector.set_baseline(baseline_performance)
            logger.info(f"已为模型 {model_id} 设置基准性能: {baseline_performance}")
        
        logger.info(f"模型 {model_id} 注册成功")
    
    def add_performance_callback(self, callback: Callable[[MonitoringResult], None]) -> None:
        """
        添加性能监控回调函数
        
        Args:
            callback: 回调函数
        """
        self.performance_callbacks.append(callback)
    
    def add_drift_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        添加漂移检测回调函数
        
        Args:
            callback: 回调函数
        """
        self.drift_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[AnomalyAlert], None]) -> None:
        """
        添加告警回调函数
        
        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)
    
    def monitor_prediction(self, 
                          model_id: str,
                          features: np.ndarray,
                          prediction: Union[np.ndarray, Any],
                          actual: Optional[np.ndarray] = None,
                          prediction_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        监控单次预测
        
        Args:
            model_id: 模型ID
            features: 特征数据
            prediction: 预测结果
            actual: 真实标签
            prediction_prob: 预测概率
            
        Returns:
            监控结果
        """
        if model_id not in self.monitored_models:
            raise ValueError(f"模型 {model_id} 未注册")
        
        results = {
            'model_id': model_id,
            'timestamp': datetime.now(),
            'performance_metrics': {},
            'drift_analysis': {},
            'anomalies': {},
            'alerts': []
        }
        
        # 更新模型状态
        self.monitored_models[model_id]['last_prediction_time'] = datetime.now()
        self.monitored_models[model_id]['prediction_count'] += 1
        
        # 性能监控
        if actual is not None:
            try:
                quality_metrics = self.quality_monitor.evaluate_prediction_quality(
                    actual, prediction, prediction_prob
                )
                
                for metric_name, metric_value in quality_metrics.items():
                    # 异常检测
                    anomaly_result = self.anomaly_detector.detect_anomalies(
                        f"{model_id}_{metric_name}", metric_value
                    )
                    
                    # 判断状态
                    threshold = self.config.quality_threshold
                    if metric_value < threshold * 0.8:
                        status = 'critical'
                    elif metric_value < threshold:
                        status = 'warning'
                    else:
                        status = 'normal'
                    
                    # 创建监控结果
                    monitoring_result = MonitoringResult(
                        timestamp=datetime.now(),
                        metric_name=f"{model_id}_{metric_name}",
                        metric_value=metric_value,
                        threshold=threshold,
                        status=status,
                        details=anomaly_result,
                        model_id=model_id
                    )
                    
                    # 保存结果
                    self.storage.save_monitoring_result(monitoring_result)
                    
                    # 检查是否需要告警
                    if status in ['warning', 'critical'] and anomaly_result.get('anomaly_detected', False):
                        alert = AnomalyAlert(
                            timestamp=datetime.now(),
                            alert_type='performance_degradation',
                            severity='high' if status == 'critical' else 'medium',
                            message=f'{metric_name} 性能下降: {metric_value:.4f} < {threshold:.4f}',
                            metric_value=metric_value,
                            threshold=threshold,
                            model_id=model_id,
                            details=anomaly_result
                        )
                        
                        if self.alert_manager.send_alert(alert):
                            self.storage.save_anomaly_alert(alert)
                            results['alerts'].append(alert)
                    
                    results['performance_metrics'][metric_name] = {
                        'value': metric_value,
                        'status': status,
                        'anomaly': anomaly_result
                    }
                
                # 模型衰减检测
                if 'accuracy' in quality_metrics:
                    degradation_result = self.degradation_detector.detect_degradation(
                        quality_metrics['accuracy']
                    )
                    
                    if degradation_result['degradation_detected']:
                        alert = AnomalyAlert(
                            timestamp=datetime.now(),
                            alert_type='model_degradation',
                            severity='high',
                            message=f"模型 {model_id} 性能衰减: {degradation_result['degradation_score']:.2%}",
                            metric_value=degradation_result['degradation_score'],
                            threshold=self.config.degradation_threshold,
                            model_id=model_id,
                            details=degradation_result
                        )
                        
                        if self.alert_manager.send_alert(alert):
                            self.storage.save_anomaly_alert(alert)
                            results['alerts'].append(alert)
                    
                    results['degradation_analysis'] = degradation_result
                
            except Exception as e:
                logger.error(f"性能监控失败: {e}")
        
        # 数据漂移检测
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # 定期进行漂移检测
            if self.monitored_models[model_id]['prediction_count'] % 100 == 0:
                drift_result = self.drift_detector.detect_drift(features)
                
                if drift_result['overall']['drift_detected']:
                    alert = AnomalyAlert(
                        timestamp=datetime.now(),
                        alert_type='data_drift',
                        severity='medium',
                        message=f"模型 {model_id} 检测到数据漂移: {drift_result['overall']['drift_score']:.4f}",
                        metric_value=drift_result['overall']['drift_score'],
                        threshold=self.config.drift_threshold,
                        model_id=model_id,
                        details=drift_result
                    )
                    
                    if self.alert_manager.send_alert(alert):
                        self.storage.save_anomaly_alert(alert)
                        results['alerts'].append(alert)
                
                results['drift_analysis'] = drift_result
                
                # 调用漂移回调
                for callback in self.drift_callbacks:
                    try:
                        callback(drift_result)
                    except Exception as e:
                        logger.error(f"漂移回调执行失败: {e}")
        
        except Exception as e:
            logger.error(f"数据漂移检测失败: {e}")
        
        # 调用性能回调
        for callback in self.performance_callbacks:
            try:
                callback(results)
            except Exception as e:
                logger.error(f"性能回调执行失败: {e}")
        
        # 调用告警回调
        for alert in results['alerts']:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"告警回调执行失败: {e}")
        
        return results
    
    def start_monitoring(self) -> None:
        """启动监控"""
        if self.is_monitoring:
            logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("模型监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("模型监控已停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                # 检查模型状态
                current_time = datetime.now()
                for model_id, model_info in self.monitored_models.items():
                    if model_info['status'] == 'active':
                        # 检查模型是否长时间无预测
                        if (model_info['last_prediction_time'] and 
                            (current_time - model_info['last_prediction_time']).total_seconds() > 3600):
                            logger.warning(f"模型 {model_id} 超过1小时无预测活动")
                
                # 清理过期数据
                self._cleanup_old_data()
                
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(60)
    
    def _cleanup_old_data(self) -> None:
        """清理过期数据"""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        with sqlite3.connect(self.storage.db_path) as conn:
            # 清理监控结果
            conn.execute(
                'DELETE FROM monitoring_results WHERE created_at < ?',
                (cutoff_date.isoformat(),)
            )
            
            # 清理告警
            conn.execute(
                'DELETE FROM anomaly_alerts WHERE created_at < ?',
                (cutoff_date.isoformat(),)
            )
            
            conn.commit()
    
    def generate_monitoring_report(self, 
                                 model_id: str, 
                                 hours: int = 24) -> Dict[str, Any]:
        """
        生成监控报告
        
        Args:
            model_id: 模型ID
            hours: 报告时间范围(小时)
            
        Returns:
            监控报告
        """
        report = {
            'model_id': model_id,
            'report_period': f'{hours} hours',
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'performance_analysis': {},
            'drift_analysis': {},
            'anomaly_summary': {},
            'recommendations': []
        }
        
        # 获取监控历史
        performance_history = {}
        for metric in self.config.quality_metrics:
            history = self.storage.get_monitoring_history(model_id, f"{model_id}_{metric}", hours)
            if history:
                values = [h['metric_value'] for h in history]
                performance_history[metric] = {
                    'values': values,
                    'avg': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'declining'
                }
        
        report['performance_analysis'] = performance_history
        
        # 获取告警统计
        with sqlite3.connect(self.storage.db_path) as conn:
            cursor = conn.execute('''
                SELECT alert_type, severity, COUNT(*) as count
                FROM anomaly_alerts
                WHERE model_id = ? AND timestamp >= datetime('now', '-{} hours')
                GROUP BY alert_type, severity
            '''.format(hours), (model_id,))
            
            alert_stats = defaultdict(lambda: defaultdict(int))
            for row in cursor.fetchall():
                alert_stats[row[0]][row[1]] = row[2]
            
            report['anomaly_summary'] = dict(alert_stats)
        
        # 生成建议
        recommendations = []
        
        # 性能建议
        for metric, data in performance_history.items():
            if data['avg'] < self.config.quality_threshold:
                recommendations.append(f"模型 {metric} 性能低于阈值，建议检查模型或数据质量")
            
            if data['trend'] == 'declining':
                recommendations.append(f"模型 {metric} 呈下降趋势，建议重新训练模型")
        
        # 告警建议
        total_alerts = sum(sum(severities.values()) for severities in report['anomaly_summary'].values())
        if total_alerts > 10:
            recommendations.append("告警数量较多，建议检查模型稳定性")
        
        report['recommendations'] = recommendations
        
        # 汇总信息
        report['summary'] = {
            'total_predictions': self.monitored_models.get(model_id, {}).get('prediction_count', 0),
            'total_alerts': total_alerts,
            'monitoring_period_hours': hours,
            'models_monitored': len(self.monitored_models)
        }
        
        return report
    
    def get_dashboard_data(self, model_id: str) -> Dict[str, Any]:
        """
        获取仪表板数据
        
        Args:
            model_id: 模型ID
            
        Returns:
            仪表板数据
        """
        return self.dashboard.generate_dashboard_data(model_id)
    
    def export_monitoring_data(self, 
                             model_id: str, 
                             format: str = 'json',
                             hours: int = 24) -> str:
        """
        导出监控数据
        
        Args:
            model_id: 模型ID
            format: 导出格式 ('json', 'csv')
            hours: 时间范围
            
        Returns:
            导出文件路径
        """
        report = self.generate_monitoring_report(model_id, hours)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"monitoring_report_{model_id}_{timestamp}.{format}"
        
        if format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'csv':
            # 导出性能数据为CSV
            performance_data = []
            for metric, data in report['performance_analysis'].items():
                for i, value in enumerate(data['values']):
                    performance_data.append({
                        'timestamp': datetime.now() - timedelta(hours=hours-i),
                        'model_id': model_id,
                        'metric': metric,
                        'value': value
                    })
            
            df = pd.DataFrame(performance_data)
            df.to_csv(filename, index=False)
        
        logger.info(f"监控数据已导出到: {filename}")
        return filename


# 测试用例
def test_model_monitor():
    """测试模型监控器"""
    print("开始测试V6模型监控器...")
    
    # 创建配置
    config = MonitoringConfig(
        performance_threshold=0.8,
        quality_threshold=0.75,
        alert_email="test@example.com",
        alert_enabled=True
    )
    
    # 创建监控器
    monitor = ModelMonitor(config)
    
    # 注册模型
    np.random.seed(42)
    baseline_data = np.random.normal(0, 1, (1000, 5))
    baseline_performance = 0.85
    
    monitor.register_model("test_model", baseline_data, baseline_performance)
    
    # 添加回调函数
    def performance_callback(result):
        print(f"性能监控回调: {result['model_id']}")
    
    def drift_callback(drift_result):
        print(f"漂移检测回调: 检测到漂移 = {drift_result['overall']['drift_detected']}")
    
    def alert_callback(alert):
        print(f"告警回调: {alert.severity} - {alert.message}")
    
    monitor.add_performance_callback(performance_callback)
    monitor.add_drift_callback(drift_callback)
    monitor.add_alert_callback(alert_callback)
    
    # 模拟预测数据
    print("模拟预测数据...")
    for i in range(50):
        # 生成测试数据
        features = np.random.normal(0, 1, (100, 5))
        
        # 模拟预测结果（有些预测故意设置为低质量）
        if i < 40:
            # 正常预测
            y_true = np.random.choice([0, 1], 100)
            y_pred = y_true.copy()
            # 添加一些噪声
            noise_mask = np.random.random(100) < 0.1
            y_pred[noise_mask] = 1 - y_pred[noise_mask]
            y_prob = np.random.random((100, 2))
            y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        else:
            # 模拟性能下降
            y_true = np.random.choice([0, 1], 100)
            y_pred = np.random.choice([0, 1], 100)  # 随机预测
            y_prob = np.random.random((100, 2))
            y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # 监控预测
        result = monitor.monitor_prediction(
            model_id="test_model",
            features=features,
            prediction=y_pred,
            actual=y_true,
            prediction_prob=y_prob
        )
        
        if i % 10 == 0:
            print(f"完成 {i+1} 次预测监控")
    
    # 生成报告
    print("\n生成监控报告...")
    report = monitor.generate_monitoring_report("test_model", hours=1)
    print("报告摘要:")
    print(f"- 总预测次数: {report['summary']['total_predictions']}")
    print(f"- 总告警数: {report['summary']['total_alerts']}")
    print(f"- 性能分析: {list(report['performance_analysis'].keys())}")
    print(f"- 建议: {report['recommendations']}")
    
    # 获取仪表板数据
    print("\n获取仪表板数据...")
    dashboard_data = monitor.get_dashboard_data("test_model")
    print(f"仪表板数据键: {list(dashboard_data.keys())}")
    
    # 导出数据
    print("\n导出监控数据...")
    json_file = monitor.export_monitoring_data("test_model", format='json', hours=1)
    csv_file = monitor.export_monitoring_data("test_model", format='csv', hours=1)
    print(f"JSON文件: {json_file}")
    print(f"CSV文件: {csv_file}")
    
    # 启动监控
    print("\n启动监控...")
    monitor.start_monitoring()
    time.sleep(5)  # 运行5秒
    monitor.stop_monitoring()
    
    print("\nV6模型监控器测试完成！")


if __name__ == "__main__":
    test_model_monitor()