"""
B3趋势分析器 - 多时间周期趋势分析系统
实现多时间周期趋势分析、强度评估、转折点识别、持续性预测等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# 尝试导入科学计算库
try:
    from scipy import stats
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 尝试导入机器学习库
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TrendDirection(Enum):
    """趋势方向枚举"""
    STRONG_UP = "强烈上涨"
    UP = "上涨"
    SIDEWAYS = "横盘"
    DOWN = "下跌"
    STRONG_DOWN = "强烈下跌"


class TrendStrength(Enum):
    """趋势强度枚举"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1


class SignalType(Enum):
    """信号类型枚举"""
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"


@dataclass
class TrendData:
    """趋势数据结构"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    symbol: str
    timeframe: str
    direction: TrendDirection
    strength: TrendStrength
    confidence: float
    slope: float
    r_squared: float
    support_level: float
    resistance_level: float
    trend_score: float
    persistence_score: float
    divergence_score: float
    timestamp: datetime


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    timeframe: str
    signal_type: SignalType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    reason: str


@dataclass
class TrendAlert:
    """趋势预警"""
    symbol: str
    alert_type: str
    message: str
    severity: str
    timestamp: datetime
    price: float


class TrendAnalyzer:
    """B3趋势分析器主类"""
    
    def __init__(self, 
                 timeframes: List[str] = None,
                 enable_kalman: bool = True,
                 enable_ml: bool = True,
                 enable_visualization: bool = True):
        """
        初始化趋势分析器
        
        Args:
            timeframes: 时间周期列表
            enable_kalman: 是否启用卡尔曼滤波
            enable_ml: 是否启用机器学习
            enable_visualization: 是否启用可视化
        """
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '4h', '1d']
        self.enable_kalman = enable_kalman and HAS_SCIPY
        self.enable_ml = enable_ml and HAS_SKLEARN
        self.enable_visualization = enable_visualization and HAS_MATPLOTLIB
        
        # 数据存储
        self.data: Dict[str, Dict[str, List[TrendData]]] = {}
        self.analysis_cache: Dict[str, Dict[str, TrendAnalysis]] = {}
        self.signals: List[TradingSignal] = []
        self.alerts: List[TrendAlert] = []
        
        # 配置参数
        self.config = {
            'min_data_points': 20,
            'trend_threshold': 0.02,
            'strength_threshold': 0.5,
            'persistence_threshold': 0.6,
            'divergence_threshold': 0.3,
            'signal_confidence_threshold': 0.7,
            'kalman_process_noise': 1e-4,
            'kalman_measurement_noise': 1e-1,
            'regression_window': 50,
            'smoothing_factor': 0.3
        }
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 初始化卡尔曼滤波器
        if self.enable_kalman:
            self._init_kalman_filters()
    
    def _init_kalman_filters(self):
        """初始化卡尔曼滤波器"""
        self.kalman_filters = {}
        for timeframe in self.timeframes:
            self.kalman_filters[timeframe] = {
                'position': np.array([[0.0], [0.0]]),  # [位置, 速度]
                'covariance': np.eye(2) * 1.0,
                'process_noise': self.config['kalman_process_noise'],
                'measurement_noise': self.config['kalman_measurement_noise']
            }
    
    def add_data(self, symbol: str, timeframe: str, data: List[Dict]):
        """
        添加价格数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            data: 价格数据列表
        """
        if symbol not in self.data:
            self.data[symbol] = {}
        
        if timeframe not in self.data[symbol]:
            self.data[symbol][timeframe] = []
        
        # 转换数据格式
        for item in data:
            trend_data = TrendData(
                timestamp=datetime.fromisoformat(item['timestamp']),
                open=float(item['open']),
                high=float(item['high']),
                low=float(item['low']),
                close=float(item['close']),
                volume=float(item['volume']),
                symbol=symbol,
                timeframe=timeframe
            )
            self.data[symbol][timeframe].append(trend_data)
        
        # 按时间排序
        self.data[symbol][timeframe].sort(key=lambda x: x.timestamp)
        
        # 清理缓存
        if symbol in self.analysis_cache and timeframe in self.analysis_cache[symbol]:
            del self.analysis_cache[symbol][timeframe]
    
    def analyze_trend(self, symbol: str, timeframe: str) -> Optional[TrendAnalysis]:
        """
        分析指定交易对和时间周期的趋势
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            
        Returns:
            趋势分析结果
        """
        if symbol not in self.data or timeframe not in self.data[symbol]:
            self.logger.warning(f"数据不存在: {symbol} {timeframe}")
            return None
        
        data_points = self.data[symbol][timeframe]
        if len(data_points) < self.config['min_data_points']:
            self.logger.warning(f"数据点不足: {symbol} {timeframe}")
            return None
        
        # 获取价格序列
        prices = [dp.close for dp in data_points[-self.config['regression_window']:]]
        timestamps = [(dp.timestamp - data_points[0].timestamp).total_seconds() 
                     for dp in data_points[-self.config['regression_window']:]]
        
        # 多方法趋势分析
        linear_trend = self._linear_regression_analysis(timestamps, prices)
        exponential_trend = self._exponential_smoothing_analysis(prices)
        kalman_trend = self._kalman_filter_analysis(timeframe, prices)
        
        # 综合分析
        trend_analysis = self._combine_trend_analysis(
            symbol, timeframe, linear_trend, exponential_trend, kalman_trend, prices
        )
        
        # 缓存结果
        if symbol not in self.analysis_cache:
            self.analysis_cache[symbol] = {}
        self.analysis_cache[symbol][timeframe] = trend_analysis
        
        return trend_analysis
    
    def _linear_regression_analysis(self, timestamps: List[float], prices: List[float]) -> Dict:
        """线性回归趋势分析"""
        if not HAS_SKLEARN:
            return self._simple_linear_regression(timestamps, prices)
        
        # 准备数据
        X = np.array(timestamps).reshape(-1, 1)
        y = np.array(prices)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 线性回归
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # 预测
        y_pred = model.predict(X_scaled)
        
        # 计算指标
        slope = model.coef_[0]
        r_squared = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        return {
            'slope': slope,
            'r_squared': r_squared,
            'mse': mse,
            'predictions': y_pred,
            'method': 'linear_regression'
        }
    
    def _simple_linear_regression(self, x: List[float], y: List[float]) -> Dict:
        """简单线性回归（无依赖版本）"""
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # 计算斜率和截距
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        # 计算R²
        y_pred = [slope * xi + intercept for xi in x]
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'slope': slope,
            'r_squared': r_squared,
            'mse': np.mean([(y[i] - y_pred[i]) ** 2 for i in range(n)]),
            'predictions': y_pred,
            'method': 'simple_linear_regression'
        }
    
    def _exponential_smoothing_analysis(self, prices: List[float]) -> Dict:
        """指数平滑趋势分析"""
        alpha = self.config['smoothing_factor']
        
        # 简单指数平滑
        smoothed = [prices[0]]
        for i in range(1, len(prices)):
            smoothed_value = alpha * prices[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_value)
        
        # 计算趋势
        if len(smoothed) >= 2:
            trend = (smoothed[-1] - smoothed[0]) / len(smoothed)
        else:
            trend = 0
        
        # 计算平滑度
        volatility = np.std(prices)
        smoothness = 1 / (1 + volatility)
        
        return {
            'trend': trend,
            'smoothed': smoothed,
            'smoothness': smoothness,
            'method': 'exponential_smoothing'
        }
    
    def _kalman_filter_analysis(self, timeframe: str, prices: List[float]) -> Dict:
        """卡尔曼滤波趋势分析"""
        if not self.enable_kalman:
            return {'filtered': prices, 'trend': 0, 'method': 'no_kalman'}
        
        # 简化的卡尔曼滤波实现
        filtered_prices = []
        
        # 初始化参数
        x = prices[0] if prices else 0  # 状态估计
        P = 1.0  # 误差协方差
        Q = self.config['kalman_process_noise']  # 过程噪声
        R = self.config['kalman_measurement_noise']  # 测量噪声
        
        for price in prices:
            # 预测步骤
            x_pred = x  # 状态预测（假设速度为0）
            P_pred = P + Q  # 协方差预测
            
            # 更新步骤
            K = P_pred / (P_pred + R)  # 卡尔曼增益
            x = x_pred + K * (price - x_pred)  # 状态更新
            P = (1 - K) * P_pred  # 协方差更新
            
            filtered_prices.append(x)
        
        # 计算趋势
        if len(filtered_prices) >= 2:
            trend = (filtered_prices[-1] - filtered_prices[0]) / len(filtered_prices)
        else:
            trend = 0
        
        return {
            'filtered': filtered_prices,
            'trend': trend,
            'method': 'kalman_filter'
        }
    
    def _combine_trend_analysis(self, symbol: str, timeframe: str, 
                              linear: Dict, exponential: Dict, kalman: Dict,
                              prices: List[float]) -> TrendAnalysis:
        """综合多种方法进行趋势分析"""
        
        # 提取趋势方向和强度
        linear_direction = self._calculate_trend_direction(linear['slope'])
        exponential_direction = self._calculate_trend_direction(exponential['trend'])
        kalman_direction = self._calculate_trend_direction(kalman['trend'])
        
        # 计算综合趋势强度
        linear_strength = abs(linear['slope']) * linear['r_squared']
        exponential_strength = abs(exponential['trend']) * exponential['smoothness']
        kalman_strength = abs(kalman['trend'])
        
        # 权重平均
        weights = [0.4, 0.3, 0.3]  # 线性回归、指数平滑、卡尔曼滤波
        combined_strength = (weights[0] * linear_strength + 
                           weights[1] * exponential_strength + 
                           weights[2] * kalman_strength)
        
        # 确定主要趋势方向
        directions = [linear_direction, exponential_direction, kalman_direction]
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        main_direction = max(direction_counts, key=direction_counts.get)
        
        # 计算趋势质量评分
        trend_score = self._calculate_trend_score(linear, exponential, kalman)
        
        # 计算持续性评分
        persistence_score = self._calculate_persistence_score(prices)
        
        # 计算背离评分
        divergence_score = self._calculate_divergence_score(linear, exponential, kalman)
        
        # 计算置信度
        confidence = self._calculate_confidence(linear, exponential, kalman)
        
        # 确定趋势强度等级
        strength = self._determine_trend_strength(combined_strength)
        
        # 计算支撑阻力位
        support_level, resistance_level = self._calculate_support_resistance(prices)
        
        return TrendAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            direction=main_direction,
            strength=strength,
            confidence=confidence,
            slope=linear['slope'],
            r_squared=linear['r_squared'],
            support_level=support_level,
            resistance_level=resistance_level,
            trend_score=trend_score,
            persistence_score=persistence_score,
            divergence_score=divergence_score,
            timestamp=datetime.now()
        )
    
    def _calculate_trend_direction(self, slope: float) -> TrendDirection:
        """根据斜率计算趋势方向"""
        threshold = self.config['trend_threshold']
        
        if slope > threshold * 2:
            return TrendDirection.STRONG_UP
        elif slope > threshold:
            return TrendDirection.UP
        elif slope < -threshold * 2:
            return TrendDirection.STRONG_DOWN
        elif slope < -threshold:
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS
    
    def _calculate_trend_score(self, linear: Dict, exponential: Dict, kalman: Dict) -> float:
        """计算趋势质量评分"""
        # R²评分 (0-1)
        r2_score = min(linear['r_squared'], 1.0)
        
        # 平滑度评分 (0-1)
        smoothness_score = exponential['smoothness']
        
        # 一致性评分 (0-1)
        slopes = [linear['slope'], exponential['trend'], kalman['trend']]
        slope_consistency = 1 - (np.std(slopes) / (np.abs(np.mean(slopes)) + 1e-6))
        slope_consistency = max(0, min(1, slope_consistency))
        
        # 综合评分
        trend_score = (r2_score * 0.4 + smoothness_score * 0.3 + slope_consistency * 0.3)
        return trend_score
    
    def _calculate_persistence_score(self, prices: List[float]) -> float:
        """计算趋势持续性评分"""
        if len(prices) < 10:
            return 0.5
        
        # 计算价格变化的一致性
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # 计算变化方向的稳定性
        positive_changes = sum(1 for change in changes if change > 0)
        negative_changes = sum(1 for change in changes if change < 0)
        
        total_changes = len(changes)
        if total_changes == 0:
            return 0.5
        
        # 持续性 = 主要方向的比例
        persistence = max(positive_changes, negative_changes) / total_changes
        return persistence
    
    def _calculate_divergence_score(self, linear: Dict, exponential: Dict, kalman: Dict) -> float:
        """计算背离评分"""
        slopes = [linear['slope'], exponential['trend'], kalman['trend']]
        
        # 计算斜率的标准差相对于均值的比例
        mean_slope = np.mean(slopes)
        if abs(mean_slope) < 1e-6:
            return 1.0  # 完全背离
        
        std_slope = np.std(slopes)
        divergence = std_slope / (abs(mean_slope) + 1e-6)
        
        # 背离评分 = 1 / (1 + 背离程度)
        divergence_score = 1 / (1 + divergence)
        return divergence_score
    
    def _calculate_confidence(self, linear: Dict, exponential: Dict, kalman: Dict) -> float:
        """计算置信度"""
        # 基于R²的置信度
        r2_confidence = min(linear['r_squared'], 1.0)
        
        # 基于方法一致性的置信度
        slopes = [linear['slope'], exponential['trend'], kalman['trend']]
        slope_consistency = 1 - (np.std(slopes) / (np.abs(np.mean(slopes)) + 1e-6))
        slope_consistency = max(0, min(1, slope_consistency))
        
        # 基于数据质量的置信度
        data_confidence = min(1.0, len(linear['predictions']) / self.config['regression_window'])
        
        # 综合置信度
        confidence = (r2_confidence * 0.4 + slope_consistency * 0.4 + data_confidence * 0.2)
        return confidence
    
    def _determine_trend_strength(self, strength_value: float) -> TrendStrength:
        """确定趋势强度等级"""
        if strength_value > 0.8:
            return TrendStrength.VERY_STRONG
        elif strength_value > 0.6:
            return TrendStrength.STRONG
        elif strength_value > 0.4:
            return TrendStrength.MODERATE
        elif strength_value > 0.2:
            return TrendStrength.WEAK
        else:
            return TrendStrength.VERY_WEAK
    
    def _calculate_support_resistance(self, prices: List[float]) -> Tuple[float, float]:
        """计算支撑和阻力位"""
        if len(prices) < 10:
            current_price = prices[-1] if prices else 0
            return current_price * 0.98, current_price * 1.02
        
        # 计算价格区间
        recent_prices = prices[-20:]  # 最近20个价格点
        support_level = min(recent_prices)
        resistance_level = max(recent_prices)
        
        return support_level, resistance_level
    
    def analyze_multi_timeframe(self, symbol: str) -> Dict[str, TrendAnalysis]:
        """多时间周期趋势分析"""
        results = {}
        
        for timeframe in self.timeframes:
            analysis = self.analyze_trend(symbol, timeframe)
            if analysis:
                results[timeframe] = analysis
        
        return results
    
    def detect_trend_reversal(self, symbol: str, timeframe: str) -> List[TrendAlert]:
        """检测趋势转折点"""
        alerts = []
        
        analysis = self.analyze_trend(symbol, timeframe)
        if not analysis:
            return alerts
        
        # 获取历史分析
        historical = self._get_historical_analysis(symbol, timeframe)
        if len(historical) < 2:
            return alerts
        
        recent_analysis = historical[-1]
        previous_analysis = historical[-2]
        
        # 检测方向变化
        if (recent_analysis.direction != previous_analysis.direction and
            recent_analysis.confidence > 0.7):
            
            alert = TrendAlert(
                symbol=symbol,
                alert_type="趋势转折",
                message=f"{symbol} {timeframe} 趋势从 {previous_analysis.direction.value} "
                       f"转为 {recent_analysis.direction.value}",
                severity="HIGH" if recent_analysis.strength.value >= 4 else "MEDIUM",
                timestamp=datetime.now(),
                price=recent_analysis.slope
            )
            alerts.append(alert)
        
        # 检测强度变化
        strength_change = recent_analysis.strength.value - previous_analysis.strength.value
        if abs(strength_change) >= 2:
            alert = TrendAlert(
                symbol=symbol,
                alert_type="强度变化",
                message=f"{symbol} {timeframe} 趋势强度变化: {strength_change:+d}",
                severity="MEDIUM",
                timestamp=datetime.now(),
                price=recent_analysis.slope
            )
            alerts.append(alert)
        
        return alerts
    
    def _get_historical_analysis(self, symbol: str, timeframe: str, limit: int = 10) -> List[TrendAnalysis]:
        """获取历史分析结果（简化实现）"""
        # 这里应该从数据库或缓存中获取历史分析结果
        # 简化实现，返回当前分析结果
        current = self.analysis_cache.get(symbol, {}).get(timeframe)
        return [current] if current else []
    
    def generate_trading_signals(self, symbol: str) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []
        
        # 多时间周期分析
        multi_tf_analysis = self.analyze_multi_timeframe(symbol)
        
        if not multi_tf_analysis:
            return signals
        
        # 趋势共振分析
        trend_alignment = self._analyze_trend_alignment(multi_tf_analysis)
        
        # 生成信号
        for timeframe, analysis in multi_tf_analysis.items():
            if analysis.confidence < self.config['signal_confidence_threshold']:
                continue
            
            signal = self._generate_single_signal(symbol, timeframe, analysis, trend_alignment)
            if signal:
                signals.append(signal)
        
        # 存储信号
        self.signals.extend(signals)
        
        return signals
    
    def _analyze_trend_alignment(self, analysis_dict: Dict[str, TrendAnalysis]) -> Dict:
        """分析趋势共振"""
        directions = [analysis.direction for analysis in analysis_dict.values()]
        strengths = [analysis.strength.value for analysis in analysis_dict.values()]
        confidences = [analysis.confidence for analysis in analysis_dict.values()]
        
        # 计算一致性
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        main_direction = max(direction_counts, key=direction_counts.get)
        alignment_score = direction_counts[main_direction] / len(directions)
        
        # 计算平均强度和置信度
        avg_strength = np.mean(strengths)
        avg_confidence = np.mean(confidences)
        
        return {
            'main_direction': main_direction,
            'alignment_score': alignment_score,
            'avg_strength': avg_strength,
            'avg_confidence': avg_confidence,
            'timeframe_consistency': len(direction_counts) == 1  # 所有时间周期方向一致
        }
    
    def _generate_single_signal(self, symbol: str, timeframe: str, 
                              analysis: TrendAnalysis, alignment: Dict) -> Optional[TradingSignal]:
        """生成单个交易信号"""
        
        # 获取当前价格（简化）
        current_price = analysis.slope  # 实际应该从数据中获取
        
        # 计算信号强度
        signal_strength = (
            analysis.confidence * 0.3 +
            (analysis.strength.value / 5) * 0.3 +
            alignment['alignment_score'] * 0.2 +
            alignment['avg_confidence'] * 0.2
        )
        
        # 确定信号类型
        if analysis.direction in [TrendDirection.STRONG_UP, TrendDirection.UP]:
            if signal_strength > 0.8:
                signal_type = SignalType.STRONG_BUY
            elif signal_strength > 0.6:
                signal_type = SignalType.BUY
            else:
                signal_type = SignalType.HOLD
        elif analysis.direction in [TrendDirection.STRONG_DOWN, TrendDirection.DOWN]:
            if signal_strength > 0.8:
                signal_type = SignalType.STRONG_SELL
            elif signal_strength > 0.6:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
        else:
            signal_type = SignalType.HOLD
        
        # 生成信号
        if signal_type != SignalType.HOLD:
            entry_price = current_price
            stop_loss = analysis.support_level if analysis.direction in [TrendDirection.UP, TrendDirection.STRONG_UP] else analysis.resistance_level
            take_profit = analysis.resistance_level if analysis.direction in [TrendDirection.UP, TrendDirection.STRONG_UP] else analysis.support_level
            
            return TradingSignal(
                symbol=symbol,
                timeframe=timeframe,
                signal_type=signal_type,
                confidence=signal_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=signal_strength,  # 简化仓位大小计算
                timestamp=datetime.now(),
                reason=f"趋势{analysis.direction.value}, 强度{analysis.strength.name}, 置信度{analysis.confidence:.2f}"
            )
        
        return None
    
    def predict_trend_persistence(self, symbol: str, timeframe: str) -> Dict:
        """预测趋势持续性"""
        analysis = self.analyze_trend(symbol, timeframe)
        if not analysis:
            return {}
        
        # 基于历史数据预测持续性
        persistence_factors = {
            'trend_strength': analysis.strength.value / 5,
            'confidence': analysis.confidence,
            'persistence_score': analysis.persistence_score,
            'divergence_score': analysis.divergence_score,
            'r_squared': analysis.r_squared
        }
        
        # 计算持续性概率
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]
        persistence_probability = sum(
            factor * weight for factor, weight in zip(persistence_factors.values(), weights)
        )
        
        # 预测持续时间（基于历史波动率）
        estimated_duration = self._estimate_trend_duration(analysis)
        
        return {
            'probability': persistence_probability,
            'estimated_duration': estimated_duration,
            'factors': persistence_factors,
            'confidence': analysis.confidence
        }
    
    def _estimate_trend_duration(self, analysis: TrendAnalysis) -> str:
        """估算趋势持续时间"""
        # 基于趋势强度和置信度估算持续时间
        base_duration = {
            '1m': 30,   # 30分钟
            '5m': 120,  # 2小时
            '15m': 360, # 6小时
            '1h': 1440, # 24小时
            '4h': 4320, # 3天
            '1d': 10080 # 7天
        }.get(analysis.timeframe, 360)
        
        # 根据强度和置信度调整
        strength_factor = analysis.strength.value / 5
        confidence_factor = analysis.confidence
        
        adjusted_duration = base_duration * strength_factor * confidence_factor
        
        # 转换为可读格式
        if adjusted_duration < 60:
            return f"{int(adjusted_duration)}分钟"
        elif adjusted_duration < 1440:
            return f"{int(adjusted_duration/60)}小时"
        else:
            return f"{int(adjusted_duration/1440)}天"
    
    def analyze_divergence(self, symbol: str, timeframe: str) -> Dict:
        """分析趋势背离"""
        analysis = self.analyze_trend(symbol, timeframe)
        if not analysis:
            return {}
        
        # 计算价格与指标的背离
        price_trend = analysis.slope
        volume_trend = self._calculate_volume_trend(symbol, timeframe)
        
        # 计算背离程度
        if abs(price_trend) > 1e-6:
            divergence_ratio = abs(volume_trend - price_trend) / abs(price_trend)
        else:
            divergence_ratio = 0
        
        # 背离类型判断
        if price_trend > 0 and volume_trend < 0:
            divergence_type = "看跌背离"
        elif price_trend < 0 and volume_trend > 0:
            divergence_type = "看涨背离"
        else:
            divergence_type = "无明显背离"
        
        return {
            'divergence_type': divergence_type,
            'divergence_ratio': divergence_ratio,
            'price_trend': price_trend,
            'volume_trend': volume_trend,
            'severity': 'HIGH' if divergence_ratio > 0.5 else 'MEDIUM' if divergence_ratio > 0.3 else 'LOW'
        }
    
    def _calculate_volume_trend(self, symbol: str, timeframe: str) -> float:
        """计算成交量趋势"""
        if symbol not in self.data or timeframe not in self.data[symbol]:
            return 0
        
        data_points = self.data[symbol][timeframe]
        if len(data_points) < 10:
            return 0
        
        volumes = [dp.volume for dp in data_points[-20:]]
        timestamps = list(range(len(volumes)))
        
        # 计算成交量趋势
        if HAS_SKLEARN:
            X = np.array(timestamps).reshape(-1, 1)
            y = np.array(volumes)
            
            model = LinearRegression()
            model.fit(X, y)
            return model.coef_[0]
        else:
            # 简单线性回归
            n = len(volumes)
            x_mean = np.mean(timestamps)
            y_mean = np.mean(volumes)
            
            numerator = sum((timestamps[i] - x_mean) * (volumes[i] - y_mean) for i in range(n))
            denominator = sum((timestamps[i] - x_mean) ** 2 for i in range(n))
            
            return numerator / denominator if denominator != 0 else 0
    
    def create_visualization(self, symbol: str, timeframe: str, save_path: str = None) -> str:
        """创建趋势分析可视化"""
        if not self.enable_visualization:
            return "可视化功能不可用（缺少matplotlib）"
        
        analysis = self.analyze_trend(symbol, timeframe)
        if not analysis:
            return "无法生成可视化：缺少分析数据"
        
        # 获取数据
        data_points = self.data.get(symbol, {}).get(timeframe, [])
        if not data_points:
            return "无法生成可视化：缺少价格数据"
        
        prices = [dp.close for dp in data_points[-50:]]
        timestamps = [dp.timestamp for dp in data_points[-50:]]
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{symbol} {timeframe} 趋势分析', fontsize=16)
        
        # 1. 价格走势和趋势线
        ax1.plot(timestamps, prices, 'b-', linewidth=2, label='价格')
        
        # 添加趋势线
        if HAS_SKLEARN:
            X = np.arange(len(prices)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, prices)
            trend_line = model.predict(X)
            ax1.plot(timestamps, trend_line, 'r--', linewidth=2, label='趋势线')
        
        ax1.set_title(f'价格走势 (趋势: {analysis.direction.value})')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 趋势强度指标
        indicators = ['趋势强度', '置信度', '持续性', '背离评分']
        values = [
            analysis.strength.value / 5,
            analysis.confidence,
            analysis.persistence_score,
            analysis.divergence_score
        ]
        
        bars = ax2.bar(indicators, values, color=['red' if v < 0.3 else 'orange' if v < 0.6 else 'green' for v in values])
        ax2.set_title('趋势质量指标')
        ax2.set_ylabel('评分')
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. 支撑阻力位
        ax3.plot(timestamps, prices, 'b-', linewidth=2, label='价格')
        ax3.axhline(y=analysis.support_level, color='green', linestyle='--', label=f'支撑位 ({analysis.support_level:.2f})')
        ax3.axhline(y=analysis.resistance_level, color='red', linestyle='--', label=f'阻力位 ({analysis.resistance_level:.2f})')
        ax3.fill_between(timestamps, analysis.support_level, analysis.resistance_level, alpha=0.2, color='gray')
        ax3.set_title('支撑阻力位分析')
        ax3.set_ylabel('价格')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 趋势统计信息
        stats_text = f"""
        趋势方向: {analysis.direction.value}
        趋势强度: {analysis.strength.name}
        置信度: {analysis.confidence:.2f}
        R²: {analysis.r_squared:.3f}
        斜率: {analysis.slope:.4f}
        趋势评分: {analysis.trend_score:.2f}
        持续性评分: {analysis.persistence_score:.2f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_title('趋势统计信息')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"可视化图表已保存到: {save_path}"
        else:
            plt.show()
            return "可视化图表已显示"
    
    def get_market_summary(self, symbols: List[str]) -> Dict:
        """获取市场总体趋势摘要"""
        summary = {
            'timestamp': datetime.now(),
            'symbols_analyzed': len(symbols),
            'strong_uptrend': [],
            'weak_uptrend': [],
            'sideways': [],
            'weak_downtrend': [],
            'strong_downtrend': [],
            'high_confidence_signals': [],
            'market_sentiment': 'NEUTRAL'
        }
        
        for symbol in symbols:
            multi_tf_analysis = self.analyze_multi_timeframe(symbol)
            
            if not multi_tf_analysis:
                continue
            
            # 统计各方向的数量
            directions = [analysis.direction for analysis in multi_tf_analysis.values()]
            
            # 获取主要时间周期的分析（1h作为主周期）
            main_analysis = multi_tf_analysis.get('1h') or list(multi_tf_analysis.values())[0]
            
            if main_analysis.direction == TrendDirection.STRONG_UP:
                summary['strong_uptrend'].append(symbol)
            elif main_analysis.direction == TrendDirection.UP:
                summary['weak_uptrend'].append(symbol)
            elif main_analysis.direction == TrendDirection.SIDEWAYS:
                summary['sideways'].append(symbol)
            elif main_analysis.direction == TrendDirection.DOWN:
                summary['weak_downtrend'].append(symbol)
            elif main_analysis.direction == TrendDirection.STRONG_DOWN:
                summary['strong_downtrend'].append(symbol)
            
            # 高置信度信号
            if main_analysis.confidence > 0.8:
                summary['high_confidence_signals'].append({
                    'symbol': symbol,
                    'direction': main_analysis.direction.value,
                    'confidence': main_analysis.confidence
                })
        
        # 计算市场情绪
        total_symbols = len(symbols)
        if total_symbols > 0:
            bullish_ratio = (len(summary['strong_uptrend']) + len(summary['weak_uptrend'])) / total_symbols
            bearish_ratio = (len(summary['strong_downtrend']) + len(summary['weak_downtrend'])) / total_symbols
            
            if bullish_ratio > 0.6:
                summary['market_sentiment'] = 'BULLISH'
            elif bearish_ratio > 0.6:
                summary['market_sentiment'] = 'BEARISH'
            else:
                summary['market_sentiment'] = 'NEUTRAL'
        
        return summary
    
    def export_analysis_report(self, symbols: List[str], save_path: str) -> str:
        """导出分析报告"""
        report_data = []
        
        for symbol in symbols:
            multi_tf_analysis = self.analyze_multi_timeframe(symbol)
            
            for timeframe, analysis in multi_tf_analysis.items():
                report_data.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': analysis.direction.value,
                    'strength': analysis.strength.name,
                    'confidence': analysis.confidence,
                    'r_squared': analysis.r_squared,
                    'slope': analysis.slope,
                    'support_level': analysis.support_level,
                    'resistance_level': analysis.resistance_level,
                    'trend_score': analysis.trend_score,
                    'persistence_score': analysis.persistence_score,
                    'divergence_score': analysis.divergence_score,
                    'timestamp': analysis.timestamp
                })
        
        # 创建DataFrame并导出
        df = pd.DataFrame(report_data)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        return f"分析报告已导出到: {save_path}"
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'analyzer_version': '1.0.0',
            'supported_timeframes': self.timeframes,
            'kalman_filter_enabled': self.enable_kalman,
            'machine_learning_enabled': self.enable_ml,
            'visualization_enabled': self.enable_visualization,
            'data_symbols': list(self.data.keys()),
            'cached_analysis': sum(len(timeframes) for timeframes in self.analysis_cache.values()),
            'total_signals': len(self.signals),
            'total_alerts': len(self.alerts),
            'configuration': self.config
        }


# 使用示例
if __name__ == "__main__":
    # 创建趋势分析器
    analyzer = TrendAnalyzer(
        timeframes=['5m', '15m', '1h', '4h'],
        enable_kalman=True,
        enable_ml=True,
        enable_visualization=True
    )
    
    # 模拟数据
    import random
    from datetime import datetime, timedelta
    
    # 生成模拟价格数据
    base_price = 100
    price_data = []
    current_price = base_price
    
    for i in range(100):
        # 添加随机波动
        change = random.uniform(-2, 2)
        current_price += change
        
        price_data.append({
            'timestamp': (datetime.now() - timedelta(minutes=100-i)).isoformat(),
            'open': current_price - random.uniform(0, 1),
            'high': current_price + random.uniform(0, 1),
            'low': current_price - random.uniform(0, 1),
            'close': current_price,
            'volume': random.uniform(1000, 10000)
        })
    
    # 添加数据
    analyzer.add_data('BTCUSDT', '1h', price_data)
    
    # 执行分析
    analysis = analyzer.analyze_trend('BTCUSDT', '1h')
    if analysis:
        print(f"趋势分析结果:")
        print(f"方向: {analysis.direction.value}")
        print(f"强度: {analysis.strength.name}")
        print(f"置信度: {analysis.confidence:.2f}")
        print(f"R²: {analysis.r_squared:.3f}")
        print(f"支撑位: {analysis.support_level:.2f}")
        print(f"阻力位: {analysis.resistance_level:.2f}")
    
    # 生成交易信号
    signals = analyzer.generate_trading_signals('BTCUSDT')
    print(f"\n生成的交易信号数量: {len(signals)}")
    
    # 多时间周期分析
    multi_tf = analyzer.analyze_multi_timeframe('BTCUSDT')
    print(f"\n多时间周期分析:")
    for timeframe, analysis in multi_tf.items():
        print(f"{timeframe}: {analysis.direction.value} (置信度: {analysis.confidence:.2f})")
    
    # 趋势持续性预测
    persistence = analyzer.predict_trend_persistence('BTCUSDT', '1h')
    if persistence:
        print(f"\n趋势持续性预测:")
        print(f"持续概率: {persistence['probability']:.2f}")
        print(f"预计持续时间: {persistence['estimated_duration']}")
    
    # 背离分析
    divergence = analyzer.analyze_divergence('BTCUSDT', '1h')
    if divergence:
        print(f"\n背离分析:")
        print(f"背离类型: {divergence['divergence_type']}")
        print(f"背离程度: {divergence['divergence_ratio']:.2f}")
    
    # 创建可视化
    if HAS_MATPLOTLIB:
        viz_result = analyzer.create_visualization('BTCUSDT', '1h', 'trend_analysis.png')
        print(f"\n{viz_result}")
    
    # 市场摘要
    summary = analyzer.get_market_summary(['BTCUSDT'])
    print(f"\n市场摘要:")
    print(f"市场情绪: {summary['market_sentiment']}")
    print(f"强烈上涨: {len(summary['strong_uptrend'])}")
    print(f"上涨: {len(summary['weak_uptrend'])}")
    print(f"横盘: {len(summary['sideways'])}")
    print(f"下跌: {len(summary['weak_downtrend'])}")
    print(f"强烈下跌: {len(summary['strong_downtrend'])}")
    
    # 系统状态
    status = analyzer.get_system_status()
    print(f"\n系统状态:")
    print(f"版本: {status['analyzer_version']}")
    print(f"支持的时间周期: {status['supported_timeframes']}")
    print(f"卡尔曼滤波: {'启用' if status['kalman_filter_enabled'] else '禁用'}")
    print(f"机器学习: {'启用' if status['machine_learning_enabled'] else '禁用'}")
    print(f"可视化: {'启用' if status['visualization_enabled'] else '禁用'}")