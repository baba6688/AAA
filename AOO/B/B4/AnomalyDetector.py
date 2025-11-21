"""
B4异常检测器
实现多种异常检测算法，用于金融市场异常监控和预警

功能包括：
1. 价格异常波动检测
2. 成交量异常检测
3. 市场结构异常检测
4. 技术指标异常检测
5. 跨资产异常关联分析
6. 异常原因分析和归类
7. 异常事件预警和响应
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from collections import deque
import warnings

# 机器学习和统计库
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import joblib

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """异常类型枚举"""
    PRICE_SPIKE = "price_spike"           # 价格异常波动
    VOLUME_SPIKE = "volume_spike"         # 成交量异常
    MARKET_STRUCTURE = "market_structure" # 市场结构异常
    TECHNICAL_INDICATOR = "technical_indicator" # 技术指标异常
    CROSS_ASSET = "cross_asset"           # 跨资产异常
    SYSTEM_ERROR = "system_error"         # 系统错误

class SeverityLevel(Enum):
    """异常严重程度级别"""
    LOW = 1       # 低级
    MEDIUM = 2    # 中级
    HIGH = 3      # 高级
    CRITICAL = 4  # 严重

class AlertStatus(Enum):
    """预警状态"""
    PENDING = "pending"     # 待处理
    ACTIVE = "active"       # 活跃
    RESOLVED = "resolved"   # 已解决
    FALSE_ALARM = "false_alarm" # 误报

@dataclass
class AnomalyEvent:
    """异常事件数据类"""
    id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: SeverityLevel
    asset_symbol: str
    value: float
    threshold: float
    description: str
    confidence: float
    alert_status: AlertStatus = AlertStatus.PENDING
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'asset_symbol': self.asset_symbol,
            'value': self.value,
            'threshold': self.threshold,
            'description': self.description,
            'confidence': self.confidence,
            'alert_status': self.alert_status.value,
            'metadata': self.metadata or {}
        }

class AnomalyDatabase:
    """异常事件历史数据库管理"""
    
    def __init__(self, db_path: str = "anomaly_events.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    asset_symbol TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    alert_status TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON anomaly_events(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_asset_type ON anomaly_events(asset_symbol, anomaly_type)
            ''')
            
            conn.commit()
    
    def save_event(self, event: AnomalyEvent):
        """保存异常事件"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO anomaly_events 
                (id, timestamp, anomaly_type, severity, asset_symbol, value, 
                 threshold, description, confidence, alert_status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.id,
                event.timestamp.isoformat(),
                event.anomaly_type.value,
                event.severity.value,
                event.asset_symbol,
                event.value,
                event.threshold,
                event.description,
                event.confidence,
                event.alert_status.value,
                json.dumps(event.metadata or {})
            ))
            conn.commit()
    
    def get_events(self, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   asset_symbol: Optional[str] = None,
                   anomaly_type: Optional[AnomalyType] = None,
                   limit: int = 1000) -> List[Dict]:
        """获取异常事件"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM anomaly_events WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            if asset_symbol:
                query += " AND asset_symbol = ?"
                params.append(asset_symbol)
            
            if anomaly_type:
                query += " AND anomaly_type = ?"
                params.append(anomaly_type.value)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

class StatisticalAnomalyDetector:
    """统计方法异常检测器"""
    
    @staticmethod
    def z_score_detection(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Z-score异常检测"""
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        return z_scores > threshold
    
    @staticmethod
    def iqr_detection(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """IQR（四分位数间距）异常检测"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    @staticmethod
    def modified_z_score_detection(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """修正Z-score异常检测（基于中位数绝对偏差）"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold

class MachineLearningAnomalyDetector:
    """机器学习异常检测器"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 保留95%的方差
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.is_trained = False
    
    def train(self, data: np.ndarray):
        """训练异常检测模型"""
        try:
            # 数据标准化
            scaled_data = self.scaler.fit_transform(data)
            
            # PCA降维
            if scaled_data.shape[1] > 1:
                pca_data = self.pca.fit_transform(scaled_data)
            else:
                pca_data = scaled_data
            
            # 训练Isolation Forest
            self.isolation_forest.fit(pca_data)
            
            # 训练DBSCAN聚类
            self.dbscan.fit(pca_data)
            
            self.is_trained = True
            logger.info("异常检测模型训练完成")
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def predict_isolation_forest(self, data: np.ndarray) -> np.ndarray:
        """Isolation Forest预测"""
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train方法")
        
        scaled_data = self.scaler.transform(data)
        if scaled_data.shape[1] > 1:
            pca_data = self.pca.transform(scaled_data)
        else:
            pca_data = scaled_data
        
        predictions = self.isolation_forest.predict(pca_data)
        return predictions == -1  # -1表示异常，1表示正常
    
    def predict_dbscan(self, data: np.ndarray) -> np.ndarray:
        """DBSCAN异常检测"""
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train方法")
        
        scaled_data = self.scaler.transform(data)
        if scaled_data.shape[1] > 1:
            pca_data = self.pca.transform(scaled_data)
        else:
            pca_data = scaled_data
        
        cluster_labels = self.dbscan.fit_predict(pca_data)
        return cluster_labels == -1  # -1表示噪声点（异常）

class PriceAnomalyDetector:
    """价格异常波动检测器"""
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.statistical_detector = StatisticalAnomalyDetector()
        self.ml_detector = MachineLearningAnomalyDetector()
        self.price_history = {}
    
    def detect_price_spike(self, 
                          symbol: str, 
                          current_price: float, 
                          price_history: List[float]) -> Tuple[bool, float, str]:
        """检测价格异常波动"""
        if len(price_history) < self.lookback_window:
            return False, 0.0, "历史数据不足"
        
        price_array = np.array(price_history[-self.lookback_window:])
        
        # Z-score检测
        z_score_anomaly = self.statistical_detector.z_score_detection(price_array, threshold=2.5)
        
        # IQR检测
        iqr_anomaly = self.statistical_detector.iqr_detection(price_array, factor=2.0)
        
        # 价格变化率检测
        returns = np.diff(price_array) / price_array[:-1]
        return_anomaly = self.statistical_detector.z_score_detection(returns, threshold=3.0)
        
        # 综合判断
        is_anomaly = np.any(z_score_anomaly[-3:]) or np.any(iqr_anomaly[-3:]) or np.any(return_anomaly[-3:])
        
        if is_anomaly:
            # 计算异常程度
            current_return = (current_price - price_array[-1]) / price_array[-1]
            z_score = abs((current_return - np.mean(returns)) / np.std(returns)) if len(returns) > 1 else 0
            
            # 确定严重程度
            if abs(current_return) > 0.1:  # 10%以上变化
                severity = SeverityLevel.CRITICAL
            elif abs(current_return) > 0.05:  # 5%以上变化
                severity = SeverityLevel.HIGH
            elif abs(current_return) > 0.02:  # 2%以上变化
                severity = SeverityLevel.MEDIUM
            else:
                severity = SeverityLevel.LOW
            
            description = f"价格异常波动: {current_return:.2%}, Z-score: {z_score:.2f}"
            return True, z_score, description
        
        return False, 0.0, "价格正常"

class VolumeAnomalyDetector:
    """成交量异常检测器"""
    
    def __init__(self, lookback_window: int = 30):
        self.lookback_window = lookback_window
        self.statistical_detector = StatisticalAnomalyDetector()
    
    def detect_volume_spike(self, 
                           symbol: str, 
                           current_volume: float, 
                           volume_history: List[float]) -> Tuple[bool, float, str]:
        """检测成交量异常"""
        if len(volume_history) < self.lookback_window:
            return False, 0.0, "历史数据不足"
        
        volume_array = np.array(volume_history[-self.lookback_window:])
        
        # Z-score检测
        z_score_anomaly = self.statistical_detector.z_score_detection(volume_array, threshold=2.5)
        
        # IQR检测
        iqr_anomaly = self.statistical_detector.iqr_detection(volume_array, factor=2.0)
        
        # 成交量比率检测
        avg_volume = np.mean(volume_array[:-1])  # 排除当前成交量
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        is_anomaly = (np.any(z_score_anomaly[-3:]) or 
                     np.any(iqr_anomaly[-3:]) or 
                     volume_ratio > 3.0)  # 当前成交量是平均值的3倍以上
        
        if is_anomaly:
            # 计算Z-score
            z_score = abs((current_volume - np.mean(volume_array)) / np.std(volume_array)) if np.std(volume_array) > 0 else 0
            
            # 确定严重程度
            if volume_ratio > 10:  # 10倍以上
                severity = SeverityLevel.CRITICAL
            elif volume_ratio > 5:  # 5倍以上
                severity = SeverityLevel.HIGH
            elif volume_ratio > 3:  # 3倍以上
                severity = SeverityLevel.MEDIUM
            else:
                severity = SeverityLevel.LOW
            
            description = f"成交量异常: {volume_ratio:.1f}倍平均, Z-score: {z_score:.2f}"
            return True, z_score, description
        
        return False, 0.0, "成交量正常"

class TechnicalIndicatorAnomalyDetector:
    """技术指标异常检测器"""
    
    def __init__(self):
        self.statistical_detector = StatisticalAnomalyDetector()
    
    def detect_rsi_anomaly(self, rsi_value: float, rsi_history: List[float]) -> Tuple[bool, float, str]:
        """RSI异常检测"""
        if len(rsi_history) < 14:
            return False, 0.0, "RSI历史数据不足"
        
        # RSI极值检测
        is_overbought = rsi_value > 80
        is_oversold = rsi_value < 20
        
        # RSI突变检测
        rsi_array = np.array(rsi_history[-20:])
        rsi_change = np.diff(rsi_array)
        change_anomaly = self.statistical_detector.z_score_detection(rsi_change, threshold=2.5)
        
        is_anomaly = is_overbought or is_oversold or np.any(change_anomaly[-3:])
        
        if is_anomaly:
            z_score = abs((rsi_value - np.mean(rsi_array)) / np.std(rsi_array)) if np.std(rsi_array) > 0 else 0
            
            if is_overbought:
                description = f"RSI超买异常: {rsi_value:.1f}"
                severity = SeverityLevel.HIGH
            elif is_oversold:
                description = f"RSI超卖异常: {rsi_value:.1f}"
                severity = SeverityLevel.HIGH
            else:
                description = f"RSI突变异常: Z-score {z_score:.2f}"
                severity = SeverityLevel.MEDIUM
            
            return True, z_score, description
        
        return False, 0.0, "RSI正常"
    
    def detect_macd_anomaly(self, 
                           macd_line: float, 
                           signal_line: float, 
                           histogram: float,
                           history: List[Dict]) -> Tuple[bool, float, str]:
        """MACD异常检测"""
        if len(history) < 26:
            return False, 0.0, "MACD历史数据不足"
        
        # MACD线与信号线背离检测
        recent_macd = [h['macd'] for h in history[-10:]]
        recent_signal = [h['signal'] for h in history[-10:]]
        
        # 检测金叉死叉异常
        macd_cross_up = macd_line > signal_line and recent_macd[-2] <= recent_signal[-2]
        macd_cross_down = macd_line < signal_line and recent_macd[-2] >= recent_signal[-2]
        
        # 直方图异常变化
        histogram_array = np.array([h['histogram'] for h in history[-20:]])
        histogram_anomaly = self.statistical_detector.z_score_detection(histogram_array, threshold=2.5)
        
        is_anomaly = macd_cross_up or macd_cross_down or np.any(histogram_anomaly[-3:])
        
        if is_anomaly:
            z_score = abs((histogram - np.mean(histogram_array)) / np.std(histogram_array)) if np.std(histogram_array) > 0 else 0
            
            if macd_cross_up:
                description = "MACD金叉异常"
                severity = SeverityLevel.MEDIUM
            elif macd_cross_down:
                description = "MACD死叉异常"
                severity = SeverityLevel.MEDIUM
            else:
                description = f"MACD直方图异常: Z-score {z_score:.2f}"
                severity = SeverityLevel.LOW
            
            return True, z_score, description
        
        return False, 0.0, "MACD正常"

class CrossAssetAnomalyDetector:
    """跨资产异常关联分析器"""
    
    def __init__(self):
        self.correlation_window = 60  # 相关性计算窗口
        self.asset_correlations = {}
    
    def detect_correlation_breakdown(self, 
                                   asset1: str, 
                                   asset2: str, 
                                   price_data1: List[float], 
                                   price_data2: List[float]) -> Tuple[bool, float, str]:
        """检测相关性破裂"""
        if len(price_data1) < self.correlation_window or len(price_data2) < self.correlation_window:
            return False, 0.0, "历史数据不足"
        
        # 计算滚动相关性
        returns1 = np.diff(price_data1[-self.correlation_window:]) / price_data1[-self.correlation_window:-1]
        returns2 = np.diff(price_data2[-self.correlation_window:]) / price_data2[-self.correlation_window:-1]
        
        current_correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        # 存储历史相关性
        if f"{asset1}_{asset2}" not in self.asset_correlations:
            self.asset_correlations[f"{asset1}_{asset2}"] = deque(maxlen=30)
        
        self.asset_correlations[f"{asset1}_{asset2}"].append(current_correlation)
        
        # 检测相关性异常
        if len(self.asset_correlations[f"{asset1}_{asset2}"]) >= 10:
            historical_corrs = list(self.asset_correlations[f"{asset1}_{asset2}"])
            mean_corr = np.mean(historical_corrs[:-1])  # 排除当前相关性
            std_corr = np.std(historical_corrs[:-1])
            
            # 相关性偏离检测
            correlation_deviation = abs(current_correlation - mean_corr)
            is_anomaly = correlation_deviation > 2 * std_corr if std_corr > 0 else correlation_deviation > 0.5
            
            if is_anomaly:
                z_score = (current_correlation - mean_corr) / std_corr if std_corr > 0 else 0
                
                if abs(correlation_deviation) > 0.8:
                    severity = SeverityLevel.HIGH
                elif abs(correlation_deviation) > 0.5:
                    severity = SeverityLevel.MEDIUM
                else:
                    severity = SeverityLevel.LOW
                
                description = f"相关性异常: 当前{current_correlation:.3f}, 历史均值{mean_corr:.3f}, 偏离{correlation_deviation:.3f}"
                return True, z_score, description
        
        return False, 0.0, "相关性正常"

class MarketStructureAnomalyDetector:
    """市场结构异常检测器"""
    
    def __init__(self):
        self.orderbook_levels = 10
        self.spread_history = deque(maxlen=100)
        self.depth_history = deque(maxlen=100)
    
    def detect_spread_anomaly(self, bid: float, ask: float, mid_price: float) -> Tuple[bool, float, str]:
        """检测买卖价差异常"""
        if mid_price <= 0:
            return False, 0.0, "价格数据无效"
        
        spread = (ask - bid) / mid_price  # 相对价差
        self.spread_history.append(spread)
        
        if len(self.spread_history) < 20:
            return False, 0.0, "历史数据不足"
        
        # 检测价差异常
        spread_array = np.array(self.spread_history)
        mean_spread = np.mean(spread_array[:-1])
        std_spread = np.std(spread_array[:-1])
        
        spread_deviation = (spread - mean_spread) / std_spread if std_spread > 0 else 0
        is_anomaly = abs(spread_deviation) > 2.5
        
        if is_anomaly:
            if abs(spread_deviation) > 4:
                severity = SeverityLevel.HIGH
            elif abs(spread_deviation) > 3:
                severity = SeverityLevel.MEDIUM
            else:
                severity = SeverityLevel.LOW
            
            description = f"价差异常: 当前{spread:.4f}, 历史均值{mean_spread:.4f}, Z-score{spread_deviation:.2f}"
            return True, abs(spread_deviation), description
        
        return False, 0.0, "价差正常"

class AnomalyAlertSystem:
    """异常预警和响应系统"""
    
    def __init__(self, database: AnomalyDatabase):
        self.database = database
        self.active_alerts = {}
        self.alert_callbacks = []
        self.alert_cooldown = {}  # 防止重复预警
        
    def add_alert_callback(self, callback):
        """添加预警回调函数"""
        self.alert_callbacks.append(callback)
    
    def trigger_alert(self, event: AnomalyEvent):
        """触发预警"""
        # 检查冷却时间
        alert_key = f"{event.asset_symbol}_{event.anomaly_type.value}"
        if alert_key in self.alert_cooldown:
            time_since_last = (datetime.now() - self.alert_cooldown[alert_key]).total_seconds()
            if time_since_last < 300:  # 5分钟冷却
                return
        
        # 保存事件到数据库
        self.database.save_event(event)
        
        # 更新活跃预警
        self.active_alerts[event.id] = event
        self.alert_cooldown[alert_key] = datetime.now()
        
        # 调用预警回调
        for callback in self.alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"预警回调执行失败: {e}")
        
        logger.warning(f"异常预警: {event.description} (严重程度: {event.severity.name})")
    
    def resolve_alert(self, alert_id: str, status: AlertStatus = AlertStatus.RESOLVED):
        """解决预警"""
        if alert_id in self.active_alerts:
            event = self.active_alerts[alert_id]
            event.alert_status = status
            self.database.save_event(event)
            del self.active_alerts[alert_id]
            logger.info(f"预警已解决: {alert_id}")

class AnomalyDetector:
    """综合异常检测器主类"""
    
    def __init__(self, db_path: str = "anomaly_events.db"):
        # 初始化各个检测器
        self.price_detector = PriceAnomalyDetector()
        self.volume_detector = VolumeAnomalyDetector()
        self.technical_detector = TechnicalIndicatorAnomalyDetector()
        self.cross_asset_detector = CrossAssetAnomalyDetector()
        self.market_structure_detector = MarketStructureAnomalyDetector()
        
        # 初始化数据库和预警系统
        self.database = AnomalyDatabase(db_path)
        self.alert_system = AnomalyAlertSystem(self.database)
        
        # 数据存储
        self.market_data = {}
        self.is_running = False
        self.monitor_thread = None
        
        # 配置参数
        self.config = {
            'price_threshold': 2.5,
            'volume_threshold': 3.0,
            'monitor_interval': 1,  # 监控间隔（秒）
            'enable_cross_asset': True,
            'enable_market_structure': True
        }
    
    def configure(self, **kwargs):
        """配置参数"""
        self.config.update(kwargs)
        logger.info(f"异常检测器配置更新: {self.config}")
    
    def add_market_data(self, symbol: str, data: Dict):
        """添加市场数据"""
        if symbol not in self.market_data:
            self.market_data[symbol] = {
                'prices': deque(maxlen=1000),
                'volumes': deque(maxlen=1000),
                'technical_indicators': deque(maxlen=1000),
                'orderbook': None,
                'last_update': None
            }
        
        market_data = self.market_data[symbol]
        
        # 更新价格数据
        if 'price' in data:
            market_data['prices'].append(data['price'])
        
        # 更新成交量数据
        if 'volume' in data:
            market_data['volumes'].append(data['volume'])
        
        # 更新技术指标
        if 'technical_indicators' in data:
            market_data['technical_indicators'].append(data['technical_indicators'])
        
        # 更新订单簿
        if 'orderbook' in data:
            market_data['orderbook'] = data['orderbook']
        
        market_data['last_update'] = datetime.now()
    
    def detect_anomalies(self, symbol: str) -> List[AnomalyEvent]:
        """检测异常"""
        events = []
        
        if symbol not in self.market_data:
            return events
        
        market_data = self.market_data[symbol]
        
        # 1. 价格异常检测
        if len(market_data['prices']) >= 20:
            current_price = market_data['prices'][-1]
            price_history = list(market_data['prices'])
            
            is_anomaly, z_score, description = self.price_detector.detect_price_spike(
                symbol, current_price, price_history
            )
            
            if is_anomaly:
                event = AnomalyEvent(
                    id=f"{symbol}_price_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    severity=self._determine_severity(z_score, AnomalyType.PRICE_SPIKE),
                    asset_symbol=symbol,
                    value=current_price,
                    threshold=z_score,
                    description=description,
                    confidence=min(abs(z_score) / 5.0, 1.0)  # 置信度计算
                )
                events.append(event)
        
        # 2. 成交量异常检测
        if len(market_data['volumes']) >= 30:
            current_volume = market_data['volumes'][-1]
            volume_history = list(market_data['volumes'])
            
            is_anomaly, z_score, description = self.volume_detector.detect_volume_spike(
                symbol, current_volume, volume_history
            )
            
            if is_anomaly:
                event = AnomalyEvent(
                    id=f"{symbol}_volume_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    severity=self._determine_severity(z_score, AnomalyType.VOLUME_SPIKE),
                    asset_symbol=symbol,
                    value=current_volume,
                    threshold=z_score,
                    description=description,
                    confidence=min(abs(z_score) / 5.0, 1.0)
                )
                events.append(event)
        
        # 3. 技术指标异常检测
        if len(market_data['technical_indicators']) >= 20:
            current_ti = market_data['technical_indicators'][-1]
            
            # RSI异常检测
            if 'rsi' in current_ti and len(market_data['technical_indicators']) >= 14:
                rsi_history = [ti.get('rsi', 50) for ti in market_data['technical_indicators']]
                is_anomaly, z_score, description = self.technical_detector.detect_rsi_anomaly(
                    current_ti['rsi'], rsi_history
                )
                
                if is_anomaly:
                    event = AnomalyEvent(
                        id=f"{symbol}_rsi_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.TECHNICAL_INDICATOR,
                        severity=self._determine_severity(z_score, AnomalyType.TECHNICAL_INDICATOR),
                        asset_symbol=symbol,
                        value=current_ti['rsi'],
                        threshold=z_score,
                        description=f"RSI异常: {description}",
                        confidence=min(abs(z_score) / 5.0, 1.0)
                    )
                    events.append(event)
            
            # MACD异常检测
            if all(key in current_ti for key in ['macd', 'signal', 'histogram']):
                macd_history = [
                    {
                        'macd': ti.get('macd', 0),
                        'signal': ti.get('signal', 0),
                        'histogram': ti.get('histogram', 0)
                    }
                    for ti in market_data['technical_indicators']
                ]
                
                is_anomaly, z_score, description = self.technical_detector.detect_macd_anomaly(
                    current_ti['macd'], current_ti['signal'], current_ti['histogram'], macd_history
                )
                
                if is_anomaly:
                    event = AnomalyEvent(
                        id=f"{symbol}_macd_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.TECHNICAL_INDICATOR,
                        severity=self._determine_severity(z_score, AnomalyType.TECHNICAL_INDICATOR),
                        asset_symbol=symbol,
                        value=current_ti['histogram'],
                        threshold=z_score,
                        description=f"MACD异常: {description}",
                        confidence=min(abs(z_score) / 5.0, 1.0)
                    )
                    events.append(event)
        
        # 4. 市场结构异常检测
        if market_data['orderbook'] and self.config['enable_market_structure']:
            orderbook = market_data['orderbook']
            if 'bid' in orderbook and 'ask' in orderbook:
                mid_price = (orderbook['bid'] + orderbook['ask']) / 2
                is_anomaly, z_score, description = self.market_structure_detector.detect_spread_anomaly(
                    orderbook['bid'], orderbook['ask'], mid_price
                )
                
                if is_anomaly:
                    event = AnomalyEvent(
                        id=f"{symbol}_spread_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.MARKET_STRUCTURE,
                        severity=self._determine_severity(z_score, AnomalyType.MARKET_STRUCTURE),
                        asset_symbol=symbol,
                        value=(orderbook['ask'] - orderbook['bid']) / mid_price,
                        threshold=z_score,
                        description=description,
                        confidence=min(abs(z_score) / 5.0, 1.0)
                    )
                    events.append(event)
        
        return events
    
    def _determine_severity(self, z_score: float, anomaly_type: AnomalyType) -> SeverityLevel:
        """根据Z-score和异常类型确定严重程度"""
        abs_z_score = abs(z_score)
        
        # 不同异常类型的阈值可能不同
        if anomaly_type == AnomalyType.PRICE_SPIKE:
            if abs_z_score > 4:
                return SeverityLevel.CRITICAL
            elif abs_z_score > 3:
                return SeverityLevel.HIGH
            elif abs_z_score > 2:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.LOW
        elif anomaly_type == AnomalyType.VOLUME_SPIKE:
            if abs_z_score > 5:
                return SeverityLevel.CRITICAL
            elif abs_z_score > 3:
                return SeverityLevel.HIGH
            elif abs_z_score > 2:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.LOW
        else:
            if abs_z_score > 3:
                return SeverityLevel.HIGH
            elif abs_z_score > 2:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.LOW
    
    def start_monitoring(self):
        """开始实时监控"""
        if self.is_running:
            logger.warning("监控已在运行中")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("异常检测监控已启动")
    
    def stop_monitoring(self):
        """停止实时监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("异常检测监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 检测所有资产的异常
                for symbol in list(self.market_data.keys()):
                    events = self.detect_anomalies(symbol)
                    
                    # 触发预警
                    for event in events:
                        self.alert_system.trigger_alert(event)
                
                time.sleep(self.config['monitor_interval'])
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(5)  # 异常后等待5秒再继续
    
    def get_anomaly_history(self, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           asset_symbol: Optional[str] = None,
                           anomaly_type: Optional[AnomalyType] = None,
                           limit: int = 1000) -> List[Dict]:
        """获取异常历史记录"""
        return self.database.get_events(start_time, end_time, asset_symbol, anomaly_type, limit)
    
    def get_active_alerts(self) -> List[AnomalyEvent]:
        """获取活跃预警"""
        return list(self.alert_system.active_alerts.values())
    
    def analyze_anomaly_patterns(self, days: int = 30) -> Dict:
        """分析异常模式"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        events = self.get_anomaly_history(start_time, end_time)
        
        if not events:
            return {"message": "指定时间范围内无异常事件"}
        
        # 统计分析
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        analysis = {
            "总异常数": len(events),
            "按类型分布": df['anomaly_type'].value_counts().to_dict(),
            "按严重程度分布": df['severity'].value_counts().to_dict(),
            "按资产分布": df['asset_symbol'].value_counts().to_dict(),
            "按小时分布": df['hour'].value_counts().to_dict(),
            "按星期分布": df['day_of_week'].value_counts().to_dict(),
            "平均置信度": df['confidence'].mean(),
            "时间范围": {
                "开始": start_time.isoformat(),
                "结束": end_time.isoformat()
            }
        }
        
        return analysis
    
    def export_model(self, filepath: str):
        """导出训练好的模型"""
        try:
            # 这里可以保存机器学习模型
            model_data = {
                'price_detector_ml': self.price_detector.ml_detector,
                'config': self.config,
                'version': '1.0'
            }
            joblib.dump(model_data, filepath)
            logger.info(f"模型已导出到: {filepath}")
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            raise
    
    def import_model(self, filepath: str):
        """导入训练好的模型"""
        try:
            model_data = joblib.load(filepath)
            self.price_detector.ml_detector = model_data['price_detector_ml']
            self.config.update(model_data.get('config', {}))
            logger.info(f"模型已从 {filepath} 导入")
        except Exception as e:
            logger.error(f"模型导入失败: {e}")
            raise

# 使用示例和测试函数
def example_usage():
    """使用示例"""
    # 创建异常检测器
    detector = AnomalyDetector()
    
    # 配置参数
    detector.configure(
        price_threshold=2.0,
        volume_threshold=2.5,
        monitor_interval=2
    )
    
    # 添加预警回调
    def alert_callback(event: AnomalyEvent):
        print(f"🚨 异常预警: {event.description}")
        print(f"   资产: {event.asset_symbol}")
        print(f"   严重程度: {event.severity.name}")
        print(f"   置信度: {event.confidence:.2f}")
        print("-" * 50)
    
    detector.alert_system.add_alert_callback(alert_callback)
    
    # 模拟市场数据
    symbols = ['BTCUSDT', 'ETHUSDT', 'AAPL']
    
    # 生成模拟数据
    for i in range(100):
        for symbol in symbols:
            # 模拟价格数据（带一些异常）
            base_price = 50000 if symbol == 'BTCUSDT' else 3000 if symbol == 'ETHUSDT' else 150
            price = base_price * (1 + np.random.normal(0, 0.02))
            
            # 在第50步添加价格异常
            if i == 50 and symbol == 'BTCUSDT':
                price *= 1.15  # 15%的价格跳跃
            
            # 模拟成交量数据
            volume = np.random.lognormal(10, 1)
            
            # 在第70步添加成交量异常
            if i == 70 and symbol == 'ETHUSDT':
                volume *= 5  # 5倍成交量
            
            # 模拟技术指标
            rsi = 30 + np.random.normal(0, 10)
            macd = np.random.normal(0, 0.1)
            signal = np.random.normal(0, 0.1)
            histogram = macd - signal
            
            technical_indicators = {
                'rsi': max(0, min(100, rsi)),
                'macd': macd,
                'signal': signal,
                'histogram': histogram
            }
            
            # 模拟订单簿数据
            orderbook = {
                'bid': price * 0.999,
                'ask': price * 1.001
            }
            
            # 添加市场数据
            detector.add_market_data(symbol, {
                'price': price,
                'volume': volume,
                'technical_indicators': technical_indicators,
                'orderbook': orderbook
            })
    
    # 检测异常
    print("开始异常检测...")
    for symbol in symbols:
        events = detector.detect_anomalies(symbol)
        if events:
            print(f"\n{symbol} 检测到 {len(events)} 个异常:")
            for event in events:
                print(f"  - {event.description}")
    
    # 启动实时监控（示例中不实际运行）
    print("\n启动实时监控...")
    # detector.start_monitoring()
    
    # 分析异常模式
    print("\n分析异常模式...")
    patterns = detector.analyze_anomaly_patterns(days=1)
    print("异常模式分析结果:")
    for key, value in patterns.items():
        print(f"  {key}: {value}")
    
    print("\n异常检测器示例运行完成!")

if __name__ == "__main__":
    # 忽略一些警告
    warnings.filterwarnings('ignore')
    
    # 运行示例
    example_usage()