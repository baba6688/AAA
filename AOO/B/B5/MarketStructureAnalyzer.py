"""
市场结构分析器 - Market Structure Analyzer

实现市场微观结构分析、市场参与者行为分析、价格发现机制分析等功能
基于高频数据和市场微观结构理论，提供全面的市场结构评估工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from collections import deque
import logging

# 科学计算库
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from scipy.stats import jarque_bera, kstest

# 可视化库
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrderBookLevel:
    """订单簿级别数据"""
    price: float
    quantity: float
    order_count: int
    timestamp: datetime

@dataclass
class TradeEvent:
    """交易事件数据"""
    price: float
    quantity: float
    timestamp: datetime
    trade_type: str  # 'buy' or 'sell'
    aggressor_side: str  # 'buyer' or 'seller'

@dataclass
class MarketStructureMetrics:
    """市场结构指标"""
    spread: float
    depth: float
    order_imbalance: float
    price_impact: float
    volatility: float
    efficiency_ratio: float
    market_impact_cost: float
    manipulation_score: float

class OrderBookAnalyzer:
    """订单簿分析器"""
    
    def __init__(self, max_levels: int = 10):
        self.max_levels = max_levels
        self.bid_levels: List[OrderBookLevel] = []
        self.ask_levels: List[OrderBookLevel] = []
        
    def update_order_book(self, bid_data: List[Tuple[float, float, int]], 
                         ask_data: List[Tuple[float, float, int]], 
                         timestamp: datetime):
        """更新订单簿数据"""
        self.bid_levels = [OrderBookLevel(price, qty, count, timestamp) 
                          for price, qty, count in bid_data]
        self.ask_levels = [OrderBookLevel(price, qty, count, timestamp) 
                          for price, qty, count in ask_data]
        
    def calculate_spread(self) -> float:
        """计算买卖价差"""
        if not self.bid_levels or not self.ask_levels:
            return 0.0
        return self.ask_levels[0].price - self.bid_levels[0].price
    
    def calculate_mid_price(self) -> float:
        """计算中间价"""
        if not self.bid_levels or not self.ask_levels:
            return 0.0
        return (self.bid_levels[0].price + self.ask_levels[0].price) / 2
    
    def calculate_depth(self, levels: int = 5) -> Dict[str, float]:
        """计算市场深度"""
        bid_depth = sum(level.quantity for level in self.bid_levels[:levels])
        ask_depth = sum(level.quantity for level in self.ask_levels[:levels])
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': bid_depth + ask_depth,
            'depth_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        }
    
    def calculate_order_imbalance(self) -> float:
        """计算订单不平衡度"""
        bid_volume = sum(level.quantity for level in self.bid_levels[:self.max_levels])
        ask_volume = sum(level.quantity for level in self.ask_levels[:self.max_levels])
        
        if bid_volume + ask_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    def analyze_price_levels(self) -> Dict[str, Any]:
        """分析价格分布"""
        bid_prices = [level.price for level in self.bid_levels]
        ask_prices = [level.price for level in self.ask_levels]
        
        return {
            'bid_price_levels': len(bid_prices),
            'ask_price_levels': len(ask_prices),
            'price_clustering': self._calculate_price_clustering(bid_prices + ask_prices),
            'tick_size_efficiency': self._calculate_tick_efficiency()
        }
    
    def _calculate_price_clustering(self, prices: List[float]) -> float:
        """计算价格聚类程度"""
        if len(prices) < 2:
            return 0.0
        
        # 计算相邻价格差的分布
        price_diffs = np.diff(sorted(prices))
        if len(price_diffs) == 0:
            return 0.0
        
        # 计算最小价格差的频率
        min_diff = np.min(price_diffs)
        clustering_ratio = np.sum(price_diffs <= min_diff * 1.1) / len(price_diffs)
        
        return clustering_ratio
    
    def _calculate_tick_efficiency(self) -> float:
        """计算报价tick效率"""
        if len(self.bid_levels) < 2 or len(self.ask_levels) < 2:
            return 0.0
        
        # 计算有效tick大小
        bid_ticks = np.diff([level.price for level in self.bid_levels[:5]])
        ask_ticks = np.diff([level.price for level in self.ask_levels[:5]])
        
        all_ticks = np.concatenate([bid_ticks, ask_ticks])
        if len(all_ticks) == 0:
            return 0.0
        
        # 计算tick大小的变异系数
        tick_cv = np.std(all_ticks) / np.mean(all_ticks) if np.mean(all_ticks) > 0 else 0
        
        return 1 / (1 + tick_cv)  # 效率越高，变异系数越小

class TradeAnalyzer:
    """交易分析器"""
    
    def __init__(self, window_size: int = 1000):
        self.trades: deque = deque(maxlen=window_size)
        self.price_history: deque = deque(maxlen=window_size)
        self.volume_history: deque = deque(maxlen=window_size)
        
    def add_trade(self, trade: TradeEvent):
        """添加交易事件"""
        self.trades.append(trade)
        self.price_history.append(trade.price)
        self.volume_history.append(trade.quantity)
    
    def calculate_price_impact(self, trade_size: float) -> float:
        """计算价格冲击"""
        if len(self.price_history) < 10:
            return 0.0
        
        # 计算冲击前的基准价格
        baseline_price = np.mean(list(self.price_history)[-20:-10]) if len(self.price_history) >= 20 else np.mean(list(self.price_history)[:-10])
        
        # 计算冲击后的价格变化
        current_price = self.price_history[-1]
        
        return (current_price - baseline_price) / baseline_price
    
    def analyze_trade_patterns(self) -> Dict[str, Any]:
        """分析交易模式"""
        if len(self.trades) < 10:
            return {}
        
        prices = list(self.price_history)
        volumes = list(self.volume_history)
        
        # 计算交易强度
        trade_intensity = len(self.trades) / max(1, (self.trades[-1].timestamp - self.trades[0].timestamp).total_seconds())
        
        # 计算成交量加权平均价格
        vwap = np.sum(np.array(prices) * np.array(volumes)) / np.sum(volumes) if np.sum(volumes) > 0 else 0
        
        # 计算价格序列的自相关性
        price_autocorr = np.corrcoef(prices[:-1], prices[1:])[0, 1] if len(prices) > 1 else 0
        
        # 计算成交量与价格变化的关系
        price_changes = np.diff(prices)
        volume_price_corr = np.corrcoef(volumes[1:], np.abs(price_changes))[0, 1] if len(price_changes) > 0 else 0
        
        return {
            'trade_intensity': trade_intensity,
            'vwap': vwap,
            'price_autocorrelation': price_autocorr,
            'volume_price_correlation': volume_price_corr,
            'average_trade_size': np.mean(volumes),
            'trade_size_volatility': np.std(volumes)
        }
    
    def detect_block_trades(self, threshold_percentile: float = 95) -> List[int]:
        """检测大额交易"""
        if len(self.volume_history) < 20:
            return []
        
        volumes = np.array(list(self.volume_history))
        threshold = np.percentile(volumes, threshold_percentile)
        
        return [i for i, vol in enumerate(volumes) if vol >= threshold]

class MarketEfficiencyAnalyzer:
    """市场效率分析器"""
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.price_history: deque = deque(maxlen=lookback_window)
        self.timestamp_history: deque = deque(maxlen=lookback_window)
        
    def add_price_data(self, price: float, timestamp: datetime):
        """添加价格数据"""
        self.price_history.append(price)
        self.timestamp_history.append(timestamp)
    
    def calculate_efficiency_ratio(self) -> float:
        """计算市场效率比率"""
        if len(self.price_history) < 10:
            return 0.0
        
        prices = np.array(list(self.price_history))
        
        # 计算总路径长度
        total_path_length = np.sum(np.abs(np.diff(prices)))
        
        # 计算直线距离
        direct_distance = abs(prices[-1] - prices[0])
        
        if direct_distance == 0:
            return 0.0
        
        return direct_distance / total_path_length
    
    def calculate_information_ratio(self) -> float:
        """计算信息比率"""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = np.array(list(self.price_history))
        returns = np.diff(np.log(prices))
        
        if len(returns) == 0:
            return 0.0
        
        # 计算信息比率（年化）
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # 假设高频数据，按秒计算年化
        annualization_factor = np.sqrt(365 * 24 * 3600)  # 按秒年化
        
        return (mean_return / std_return) * annualization_factor
    
    def test_random_walk(self) -> Dict[str, float]:
        """随机游走检验"""
        if len(self.price_history) < 30:
            return {}
        
        prices = np.array(list(self.price_history))
        returns = np.diff(np.log(prices))
        
        # ADF检验
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(returns)
            adf_result = {
                'adf_statistic': adf_stat,
                'adf_pvalue': adf_pvalue,
                'adf_critical_1pct': adf_critical['1%'],
                'adf_critical_5pct': adf_critical['5%'],
                'adf_critical_10pct': adf_critical['10%']
            }
        except ImportError:
            adf_result = {'adf_statistic': 0, 'adf_pvalue': 1}
        
        # Ljung-Box检验
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            ljung_box_result = acorr_ljungbox(returns, lags=min(10, len(returns)//4), return_df=True)
            ljung_box_stat = ljung_box_result['lb_stat'].iloc[-1]
            ljung_box_pvalue = ljung_box_result['lb_pvalue'].iloc[-1]
        except ImportError:
            ljung_box_stat = 0
            ljung_box_pvalue = 1
        
        return {
            **adf_result,
            'ljung_box_statistic': ljung_box_stat,
            'ljung_box_pvalue': ljung_box_pvalue,
            'is_random_walk': ljung_box_pvalue > 0.05
        }
    
    def calculate_hurst_exponent(self) -> float:
        """计算Hurst指数"""
        if len(self.price_history) < 50:
            return 0.5
        
        prices = np.array(list(self.price_history))
        
        # 计算累积离差
        mean_price = np.mean(prices)
        cumdev = np.cumsum(prices - mean_price)
        
        # 计算R/S统计量
        def rs_statistic(n):
            if n < 2:
                return 0
            cumdev_subset = cumdev[:n]
            R = np.max(cumdev_subset) - np.min(cumdev_subset)
            S = np.std(prices[:n])
            return R / S if S > 0 else 0
        
        # 计算不同时间尺度的R/S
        scales = np.logspace(1, np.log10(len(prices)//2), 10).astype(int)
        rs_values = [rs_statistic(s) for s in scales]
        
        # 线性回归计算Hurst指数
        log_scales = np.log(scales)
        log_rs = np.log(rs_values)
        
        if len(log_scales) > 1 and np.std(log_scales) > 0:
            hurst = np.polyfit(log_scales, log_rs, 1)[0]
        else:
            hurst = 0.5
        
        return max(0, min(1, hurst))

class MarketManipulationDetector:
    """市场操纵检测器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history: deque = deque(maxlen=window_size)
        self.volume_history: deque = deque(maxlen=window_size)
        self.timestamp_history: deque = deque(maxlen=window_size)
        
    def add_data_point(self, price: float, volume: float, timestamp: datetime):
        """添加数据点"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.timestamp_history.append(timestamp)
    
    def detect_pump_and_dump(self, price_threshold: float = 0.05, 
                           volume_threshold: float = 3.0) -> Dict[str, Any]:
        """检测拉高砸盘操纵"""
        if len(self.price_history) < 20:
            return {'manipulation_detected': False}
        
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        
        # 检测价格快速上涨
        price_changes = np.diff(prices) / prices[:-1]
        rapid_increases = price_changes > price_threshold
        
        # 检测成交量异常
        volume_mean = np.mean(volumes[:-1])
        volume_spikes = volumes[1:] > volume_threshold * volume_mean
        
        # 检测拉高砸盘模式
        pump_detected = np.any(rapid_increases & volume_spikes)
        
        # 如果检测到拉高，寻找对应的砸盘
        dump_detected = False
        if pump_detected:
            pump_index = np.where(rapid_increases & volume_spikes)[0][0]
            if pump_index < len(price_changes) - 5:
                subsequent_changes = price_changes[pump_index+1:pump_index+6]
                dump_detected = np.any(subsequent_changes < -price_threshold/2)
        
        return {
            'manipulation_detected': pump_detected and dump_detected,
            'pump_detected': pump_detected,
            'dump_detected': dump_detected,
            'manipulation_score': self._calculate_manipulation_score(prices, volumes)
        }
    
    def detect_spoofing(self, order_book_data: List[Dict]) -> Dict[str, Any]:
        """检测虚假下单（spoofing）"""
        if len(order_book_data) < 10:
            return {'spoofing_detected': False}
        
        # 分析订单簿变化模式
        price_movements = []
        volume_changes = []
        
        for i in range(1, len(order_book_data)):
            prev_data = order_book_data[i-1]
            curr_data = order_book_data[i]
            
            # 计算价格变化
            price_change = (curr_data['mid_price'] - prev_data['mid_price']) / prev_data['mid_price']
            price_movements.append(price_change)
            
            # 计算订单量变化
            volume_change = curr_data['total_volume'] / prev_data['total_volume'] if prev_data['total_volume'] > 0 else 1
            volume_changes.append(volume_change)
        
        # 检测虚假下单模式：大量订单但快速撤销
        price_movements = np.array(price_movements)
        volume_changes = np.array(volume_changes)
        
        # 寻找大量订单但价格变化很小的模式
        low_price_impact = np.abs(price_movements) < 0.001  # 价格影响很小
        high_volume_change = volume_changes > 2.0  # 订单量大幅增加
        
        spoofing_score = np.sum(low_price_impact & high_volume_change) / len(price_movements)
        
        return {
            'spoofing_detected': spoofing_score > 0.3,
            'spoofing_score': spoofing_score,
            'suspicious_periods': np.sum(low_price_impact & high_volume_change)
        }
    
    def detect_layering(self, price_levels: List[float], volumes: List[float]) -> Dict[str, Any]:
        """检测分层下单（layering）"""
        if len(price_levels) < 5 or len(volumes) < 5:
            return {'layering_detected': False}
        
        # 分析价格分层模式
        price_diffs = np.diff(price_levels)
        
        # 检测是否有规律的价格间隔
        unique_diffs = np.unique(np.round(price_diffs, 4))
        
        # 如果有多个相同的价格间隔，可能存在分层下单
        diff_counts = {}
        for diff in price_diffs:
            diff_rounded = round(diff, 4)
            diff_counts[diff_rounded] = diff_counts.get(diff_rounded, 0) + 1
        
        max_count = max(diff_counts.values()) if diff_counts else 0
        layering_score = max_count / len(price_diffs) if len(price_diffs) > 0 else 0
        
        return {
            'layering_detected': layering_score > 0.5,
            'layering_score': layering_score,
            'regular_spacing_count': max_count
        }
    
    def _calculate_manipulation_score(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """计算操纵分数"""
        if len(prices) < 10:
            return 0.0
        
        # 计算价格和成交量的异常程度
        price_volatility = np.std(np.diff(prices) / prices[:-1])
        volume_volatility = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
        
        # 计算相关性异常
        if len(prices) > 1:
            price_volume_corr = np.corrcoef(prices, volumes)[0, 1]
            correlation_anomaly = abs(price_volume_corr) > 0.8  # 异常高的相关性
        else:
            correlation_anomaly = False
        
        # 综合评分
        score = 0
        if price_volatility > 0.02:  # 价格波动异常
            score += 0.3
        if volume_volatility > 2.0:  # 成交量波动异常
            score += 0.3
        if correlation_anomaly:
            score += 0.4
        
        return min(1.0, score)

class MarketStructureAnalyzer:
    """市场结构分析器主类"""
    
    def __init__(self, initial_capital: float = 1000000, 
                 tick_size: float = 0.01,
                 max_order_book_levels: int = 10):
        self.initial_capital = initial_capital
        self.tick_size = tick_size
        self.max_order_book_levels = max_order_book_levels
        
        # 初始化各个分析器
        self.order_book_analyzer = OrderBookAnalyzer(max_order_book_levels)
        self.trade_analyzer = TradeAnalyzer()
        self.efficiency_analyzer = MarketEfficiencyAnalyzer()
        self.manipulation_detector = MarketManipulationDetector()
        
        # 数据存储
        self.market_data: List[Dict] = []
        self.structure_metrics: List[MarketStructureMetrics] = []
        
        # 监控状态
        self.is_monitoring = False
        self.last_update_time = None
        
    def update_market_data(self, bid_data: List[Tuple[float, float, int]], 
                          ask_data: List[Tuple[float, float, int]], 
                          trade_data: List[Tuple[float, float, str, str]],
                          timestamp: datetime):
        """更新市场数据"""
        # 更新订单簿
        self.order_book_analyzer.update_order_book(bid_data, ask_data, timestamp)
        
        # 更新交易数据
        for price, quantity, trade_type, aggressor_side in trade_data:
            trade_event = TradeEvent(price, quantity, timestamp, trade_type, aggressor_side)
            self.trade_analyzer.add_trade(trade_event)
        
        # 更新效率分析器
        mid_price = self.order_book_analyzer.calculate_mid_price()
        if mid_price > 0:
            self.efficiency_analyzer.add_price_data(mid_price, timestamp)
        
        # 更新操纵检测器
        total_volume = sum(qty for _, qty, _, _ in trade_data)
        if total_volume > 0:
            self.manipulation_detector.add_data_point(mid_price, total_volume, timestamp)
        
        # 计算市场结构指标
        metrics = self._calculate_market_structure_metrics()
        self.structure_metrics.append(metrics)
        
        # 存储原始数据
        self.market_data.append({
            'timestamp': timestamp,
            'bid_data': bid_data,
            'ask_data': ask_data,
            'trade_data': trade_data,
            'metrics': metrics
        })
        
        self.last_update_time = timestamp
        
    def _calculate_market_structure_metrics(self) -> MarketStructureMetrics:
        """计算市场结构指标"""
        # 基础指标
        spread = self.order_book_analyzer.calculate_spread()
        mid_price = self.order_book_analyzer.calculate_mid_price()
        
        # 市场深度
        depth_info = self.order_book_analyzer.calculate_depth()
        depth = depth_info['total_depth']
        
        # 订单不平衡度
        order_imbalance = self.order_book_analyzer.calculate_order_imbalance()
        
        # 价格冲击
        price_impact = self.trade_analyzer.calculate_price_impact(1.0)
        
        # 波动率（基于价格序列）
        if len(self.efficiency_analyzer.price_history) >= 20:
            prices = np.array(list(self.efficiency_analyzer.price_history))
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252 * 24 * 3600)  # 年化波动率
        else:
            volatility = 0.0
        
        # 市场效率
        efficiency_ratio = self.efficiency_analyzer.calculate_efficiency_ratio()
        
        # 市场冲击成本
        market_impact_cost = self._calculate_market_impact_cost()
        
        # 操纵分数
        manipulation_score = self.manipulation_detector._calculate_manipulation_score(
            np.array(list(self.manipulation_detector.price_history)),
            np.array(list(self.manipulation_detector.volume_history))
        )
        
        return MarketStructureMetrics(
            spread=spread,
            depth=depth,
            order_imbalance=order_imbalance,
            price_impact=price_impact,
            volatility=volatility,
            efficiency_ratio=efficiency_ratio,
            market_impact_cost=market_impact_cost,
            manipulation_score=manipulation_score
        )
    
    def _calculate_market_impact_cost(self) -> float:
        """计算市场冲击成本"""
        if len(self.structure_metrics) < 10:
            return 0.0
        
        recent_metrics = self.structure_metrics[-10:]
        
        # 基于买卖价差和深度的冲击成本模型
        avg_spread = np.mean([m.spread for m in recent_metrics])
        avg_depth = np.mean([m.depth for m in recent_metrics])
        
        if avg_depth == 0:
            return 0.0
        
        # 简化的冲击成本模型：冲击成本与价差成正比，与深度成反比
        impact_cost = avg_spread / (2 * avg_depth)
        
        return impact_cost
    
    def analyze_market_microstructure(self) -> Dict[str, Any]:
        """分析市场微观结构"""
        if not self.structure_metrics:
            return {}
        
        recent_metrics = self.structure_metrics[-min(100, len(self.structure_metrics)):]
        
        # 计算统计指标
        analysis = {
            'spread_analysis': {
                'mean_spread': np.mean([m.spread for m in recent_metrics]),
                'spread_volatility': np.std([m.spread for m in recent_metrics]),
                'spread_percentile_95': np.percentile([m.spread for m in recent_metrics], 95)
            },
            'depth_analysis': {
                'mean_depth': np.mean([m.depth for m in recent_metrics]),
                'depth_volatility': np.std([m.depth for m in recent_metrics]),
                'depth_trend': self._calculate_trend([m.depth for m in recent_metrics])
            },
            'order_imbalance_analysis': {
                'mean_imbalance': np.mean([m.order_imbalance for m in recent_metrics]),
                'imbalance_volatility': np.std([m.order_imbalance for m in recent_metrics]),
                'imbalance_persistence': self._calculate_persistence([m.order_imbalance for m in recent_metrics])
            },
            'price_impact_analysis': {
                'mean_impact': np.mean([abs(m.price_impact) for m in recent_metrics]),
                'impact_asymmetry': self._calculate_impact_asymmetry(recent_metrics)
            },
            'volatility_analysis': {
                'mean_volatility': np.mean([m.volatility for m in recent_metrics]),
                'volatility_clustering': self._detect_volatility_clustering(recent_metrics)
            }
        }
        
        return analysis
    
    def analyze_participant_structure(self) -> Dict[str, Any]:
        """分析市场参与者结构"""
        if len(self.trade_analyzer.trades) < 50:
            return {}
        
        # 分析交易模式
        trade_patterns = self.trade_analyzer.analyze_trade_patterns()
        
        # 检测大额交易
        block_trades = self.trade_analyzer.detect_block_trades()
        
        # 分析订单流
        order_flow_analysis = self._analyze_order_flow()
        
        return {
            'trade_patterns': trade_patterns,
            'block_trade_analysis': {
                'block_trade_count': len(block_trades),
                'block_trade_ratio': len(block_trades) / len(self.trade_analyzer.trades),
                'block_trade_timestamps': [self.trade_analyzer.trades[i].timestamp for i in block_trades]
            },
            'order_flow_analysis': order_flow_analysis,
            'participant_classification': self._classify_participants()
        }
    
    def _analyze_order_flow(self) -> Dict[str, Any]:
        """分析订单流"""
        if len(self.trade_analyzer.trades) < 20:
            return {}
        
        # 分析买卖压力
        buy_volume = sum(trade.quantity for trade in self.trade_analyzer.trades 
                        if trade.aggressor_side == 'buyer')
        sell_volume = sum(trade.quantity for trade in self.trade_analyzer.trades 
                         if trade.aggressor_side == 'seller')
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {}
        
        # 计算订单流不平衡度
        order_flow_imbalance = (buy_volume - sell_volume) / total_volume
        
        # 分析订单流持续性
        flow_persistence = self._calculate_flow_persistence()
        
        return {
            'buy_sell_ratio': buy_volume / sell_volume if sell_volume > 0 else float('inf'),
            'order_flow_imbalance': order_flow_imbalance,
            'flow_persistence': flow_persistence,
            'aggressive_trade_ratio': len([t for t in self.trade_analyzer.trades 
                                         if t.aggressor_side in ['buyer', 'seller']]) / len(self.trade_analyzer.trades)
        }
    
    def _classify_participants(self) -> Dict[str, Any]:
        """分类市场参与者"""
        if len(self.trade_analyzer.trades) < 30:
            return {}
        
        # 基于交易模式的简单分类
        volumes = [trade.quantity for trade in self.trade_analyzer.trades]
        trade_sizes = np.array(volumes)
        
        # 定义阈值
        small_trade_threshold = np.percentile(trade_sizes, 33)
        large_trade_threshold = np.percentile(trade_sizes, 90)
        
        # 分类交易
        small_trades = np.sum(trade_sizes <= small_trade_threshold)
        medium_trades = np.sum((trade_sizes > small_trade_threshold) & 
                              (trade_sizes <= large_trade_threshold))
        large_trades = np.sum(trade_sizes > large_trade_threshold)
        
        total_trades = len(trade_sizes)
        
        return {
            'retail_trader_ratio': small_trades / total_trades,
            'institutional_trader_ratio': large_trades / total_trades,
            'medium_trader_ratio': medium_trades / total_trades,
            'average_trade_size': np.mean(trade_sizes),
            'trade_size_distribution': {
                'small': small_trades,
                'medium': medium_trades,
                'large': large_trades
            }
        }
    
    def analyze_price_discovery(self) -> Dict[str, Any]:
        """分析价格发现机制"""
        if len(self.structure_metrics) < 20:
            return {}
        
        # 价格发现效率指标
        efficiency_ratio = self.efficiency_analyzer.calculate_efficiency_ratio()
        information_ratio = self.efficiency_analyzer.calculate_information_ratio()
        
        # 价格序列分析
        price_discovery_metrics = self._calculate_price_discovery_metrics()
        
        # 有效价差分析
        effective_spread_analysis = self._analyze_effective_spread()
        
        return {
            'efficiency_metrics': {
                'efficiency_ratio': efficiency_ratio,
                'information_ratio': information_ratio,
                'hurst_exponent': self.efficiency_analyzer.calculate_hurst_exponent()
            },
            'price_discovery_metrics': price_discovery_metrics,
            'effective_spread_analysis': effective_spread_analysis,
            'random_walk_tests': self.efficiency_analyzer.test_random_walk()
        }
    
    def _calculate_price_discovery_metrics(self) -> Dict[str, Any]:
        """计算价格发现指标"""
        if len(self.order_book_analyzer.bid_levels) == 0 or len(self.order_book_analyzer.ask_levels) == 0:
            return {}
        
        # 计算价格改进概率
        price_improvement_prob = self._calculate_price_improvement_probability()
        
        # 计算价格影响持续性
        impact_persistence = self._calculate_impact_persistence()
        
        return {
            'price_improvement_probability': price_improvement_prob,
            'impact_persistence': impact_persistence,
            'quote_update_frequency': self._calculate_quote_update_frequency()
        }
    
    def _analyze_effective_spread(self) -> Dict[str, Any]:
        """分析有效价差"""
        if len(self.trade_analyzer.trades) < 10:
            return {}
        
        # 计算实际交易价差
        trades = list(self.trade_analyzer.trades)
        mid_prices = []
        actual_spreads = []
        
        for trade in trades[-20:]:  # 最近20笔交易
            # 找到交易时间附近的报价
            closest_bid = self.order_book_analyzer.bid_levels[0].price if self.order_book_analyzer.bid_levels else trade.price
            closest_ask = self.order_book_analyzer.ask_levels[0].price if self.order_book_analyzer.ask_levels else trade.price
            
            mid_price = (closest_bid + closest_ask) / 2
            actual_spread = 2 * abs(trade.price - mid_price)
            
            mid_prices.append(mid_price)
            actual_spreads.append(actual_spread)
        
        return {
            'mean_effective_spread': np.mean(actual_spreads) if actual_spreads else 0,
            'effective_spread_volatility': np.std(actual_spreads) if actual_spreads else 0,
            'spread_ratio': np.mean(actual_spreads) / self.order_book_analyzer.calculate_spread() if self.order_book_analyzer.calculate_spread() > 0 else 0
        }
    
    def detect_market_manipulation(self) -> Dict[str, Any]:
        """检测市场操纵"""
        manipulation_results = {}
        
        # 检测拉高砸盘
        pump_dump_result = self.manipulation_detector.detect_pump_and_dump()
        manipulation_results['pump_and_dump'] = pump_dump_result
        
        # 检测虚假下单（需要订单簿历史数据）
        if len(self.market_data) >= 10:
            order_book_history = []
            for data in self.market_data[-10:]:
                order_book_history.append({
                    'mid_price': self.order_book_analyzer.calculate_mid_price(),
                    'total_volume': sum(qty for _, qty, _ in data['bid_data']) + 
                                   sum(qty for _, qty, _ in data['ask_data'])
                })
            
            spoofing_result = self.manipulation_detector.detect_spoofing(order_book_history)
            manipulation_results['spoofing'] = spoofing_result
        
        # 综合操纵评分
        all_scores = []
        for result in manipulation_results.values():
            if isinstance(result, dict) and 'manipulation_score' in result:
                all_scores.append(result['manipulation_score'])
            elif isinstance(result, dict) and 'spoofing_score' in result:
                all_scores.append(result['spoofing_score'])
        
        overall_manipulation_score = np.mean(all_scores) if all_scores else 0
        
        return {
            'manipulation_results': manipulation_results,
            'overall_manipulation_score': overall_manipulation_score,
            'manipulation_alerts': self._generate_manipulation_alerts(manipulation_results)
        }
    
    def monitor_market_structure_changes(self) -> Dict[str, Any]:
        """监控市场结构变化"""
        if len(self.structure_metrics) < 50:
            return {'status': 'insufficient_data'}
        
        # 获取历史数据作为基准
        historical_metrics = self.structure_metrics[:-10]  # 排除最近10个数据点
        recent_metrics = self.structure_metrics[-10:]
        
        # 计算结构变化指标
        structure_changes = {}
        
        for metric_name in ['spread', 'depth', 'order_imbalance', 'volatility']:
            historical_values = [getattr(m, metric_name) for m in historical_metrics]
            recent_values = [getattr(m, metric_name) for m in recent_metrics]
            
            # 计算变化幅度
            historical_mean = np.mean(historical_values)
            recent_mean = np.mean(recent_values)
            
            if historical_mean != 0:
                change_magnitude = (recent_mean - historical_mean) / historical_mean
                structure_changes[metric_name] = {
                    'change_magnitude': change_magnitude,
                    'change_significance': abs(change_magnitude) > 0.2,  # 20%以上变化认为显著
                    'historical_mean': historical_mean,
                    'recent_mean': recent_mean
                }
        
        # 生成预警
        alerts = []
        for metric, change_info in structure_changes.items():
            if change_info['change_significance']:
                direction = "增加" if change_info['change_magnitude'] > 0 else "减少"
                alerts.append(f"{metric}显著{direction}，变化幅度：{change_info['change_magnitude']:.2%}")
        
        return {
            'structure_changes': structure_changes,
            'alerts': alerts,
            'overall_stability': self._assess_market_stability(structure_changes),
            'recommended_actions': self._generate_recommended_actions(structure_changes)
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合分析报告"""
        report = {
            'timestamp': datetime.now(),
            'data_summary': {
                'total_data_points': len(self.market_data),
                'analysis_period': {
                    'start': self.market_data[0]['timestamp'] if self.market_data else None,
                    'end': self.market_data[-1]['timestamp'] if self.market_data else None
                }
            },
            'market_microstructure_analysis': self.analyze_market_microstructure(),
            'participant_structure_analysis': self.analyze_participant_structure(),
            'price_discovery_analysis': self.analyze_price_discovery(),
            'manipulation_detection': self.detect_market_manipulation(),
            'structure_monitoring': self.monitor_market_structure_changes(),
            'current_metrics': self.structure_metrics[-1].__dict__ if self.structure_metrics else {},
            'risk_assessment': self._assess_market_risks(),
            'recommendations': self._generate_investment_recommendations()
        }
        
        return report
    
    # 辅助方法
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def _calculate_persistence(self, values: List[float]) -> float:
        """计算持续性"""
        if len(values) < 3:
            return 0.0
        
        # 计算一阶自相关
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
        return autocorr if not np.isnan(autocorr) else 0.0
    
    def _calculate_impact_asymmetry(self, metrics: List[MarketStructureMetrics]) -> float:
        """计算冲击不对称性"""
        positive_impacts = [m.price_impact for m in metrics if m.price_impact > 0]
        negative_impacts = [abs(m.price_impact) for m in metrics if m.price_impact < 0]
        
        if not positive_impacts or not negative_impacts:
            return 0.0
        
        return np.mean(positive_impacts) - np.mean(negative_impacts)
    
    def _detect_volatility_clustering(self, metrics: List[MarketStructureMetrics]) -> float:
        """检测波动率聚集"""
        volatilities = [m.volatility for m in metrics]
        if len(volatilities) < 10:
            return 0.0
        
        # 计算波动率的滞后相关性
        high_vol_periods = np.array(volatilities) > np.mean(volatilities)
        if np.sum(high_vol_periods) < 2:
            return 0.0
        
        # 计算高波动期的聚集程度
        clustering_score = 0
        for i in range(len(high_vol_periods) - 1):
            if high_vol_periods[i] and high_vol_periods[i + 1]:
                clustering_score += 1
        
        return clustering_score / max(1, len(high_vol_periods) - 1)
    
    def _calculate_flow_persistence(self) -> float:
        """计算订单流持续性"""
        if len(self.trade_analyzer.trades) < 10:
            return 0.0
        
        # 简化的持续性计算
        recent_trades = list(self.trade_analyzer.trades)[-20:]
        buy_sell_sequence = [1 if trade.aggressor_side == 'buyer' else -1 
                           for trade in recent_trades]
        
        # 计算序列的自相关
        if len(buy_sell_sequence) > 1:
            return np.corrcoef(buy_sell_sequence[:-1], buy_sell_sequence[1:])[0, 1]
        return 0.0
    
    def _calculate_price_improvement_probability(self) -> float:
        """计算价格改进概率"""
        if len(self.trade_analyzer.trades) < 5:
            return 0.0
        
        improvements = 0
        total_trades = 0
        
        for i in range(1, min(len(self.trade_analyzer.trades), 20)):
            prev_trade = self.trade_analyzer.trades[-i-1]
            curr_trade = self.trade_analyzer.trades[-i]
            
            # 简化的价格改进判断
            if curr_trade.price > prev_trade.price and curr_trade.aggressor_side == 'buyer':
                improvements += 1
            elif curr_trade.price < prev_trade.price and curr_trade.aggressor_side == 'seller':
                improvements += 1
            
            total_trades += 1
        
        return improvements / total_trades if total_trades > 0 else 0.0
    
    def _calculate_impact_persistence(self) -> float:
        """计算冲击持续性"""
        if len(self.structure_metrics) < 10:
            return 0.0
        
        impacts = [abs(m.price_impact) for m in self.structure_metrics[-20:]]
        if len(impacts) < 3:
            return 0.0
        
        # 计算冲击的衰减速度
        decay_rate = 0
        for i in range(1, len(impacts)):
            if impacts[i-1] > 0:
                decay_rate += (impacts[i-1] - impacts[i]) / impacts[i-1]
        
        return decay_rate / (len(impacts) - 1)
    
    def _calculate_quote_update_frequency(self) -> float:
        """计算报价更新频率"""
        if len(self.market_data) < 2:
            return 0.0
        
        time_diffs = []
        for i in range(1, min(len(self.market_data), 20)):
            prev_time = self.market_data[i-1]['timestamp']
            curr_time = self.market_data[i]['timestamp']
            time_diff = (curr_time - prev_time).total_seconds()
            time_diffs.append(time_diff)
        
        return 1 / np.mean(time_diffs) if time_diffs and np.mean(time_diffs) > 0 else 0.0
    
    def _generate_manipulation_alerts(self, manipulation_results: Dict) -> List[str]:
        """生成操纵预警"""
        alerts = []
        
        for manipulation_type, result in manipulation_results.items():
            if isinstance(result, dict):
                if result.get('manipulation_detected', False):
                    alerts.append(f"检测到{manipulation_type}操纵行为")
                elif result.get('spoofing_detected', False):
                    alerts.append(f"检测到{manipulation_type}行为")
                elif result.get('layering_detected', False):
                    alerts.append(f"检测到{manipulation_type}行为")
        
        return alerts
    
    def _assess_market_stability(self, structure_changes: Dict) -> str:
        """评估市场稳定性"""
        if not structure_changes:
            return "数据不足"
        
        significant_changes = sum(1 for change in structure_changes.values() 
                                if change['change_significance'])
        
        if significant_changes >= 3:
            return "不稳定"
        elif significant_changes >= 1:
            return "轻微波动"
        else:
            return "稳定"
    
    def _generate_recommended_actions(self, structure_changes: Dict) -> List[str]:
        """生成建议操作"""
        actions = []
        
        for metric, change_info in structure_changes.items():
            if change_info['change_significance']:
                if metric == 'spread' and change_info['change_magnitude'] > 0:
                    actions.append("买卖价差扩大，建议降低交易频率")
                elif metric == 'depth' and change_info['change_magnitude'] < 0:
                    actions.append("市场深度下降，建议谨慎交易")
                elif metric == 'volatility' and change_info['change_magnitude'] > 0:
                    actions.append("波动率上升，建议增加风险控制")
        
        if not actions:
            actions.append("市场结构正常，无需特殊操作")
        
        return actions
    
    def _assess_market_risks(self) -> Dict[str, str]:
        """评估市场风险"""
        if not self.structure_metrics:
            return {"overall_risk": "数据不足"}
        
        recent_metrics = self.structure_metrics[-min(50, len(self.structure_metrics)):]
        
        # 计算风险指标
        avg_volatility = np.mean([m.volatility for m in recent_metrics])
        avg_spread = np.mean([m.spread for m in recent_metrics])
        avg_manipulation_score = np.mean([m.manipulation_score for m in recent_metrics])
        
        # 风险评级
        risk_level = "低"
        if avg_volatility > 0.3 or avg_spread > 0.02 or avg_manipulation_score > 0.5:
            risk_level = "高"
        elif avg_volatility > 0.2 or avg_spread > 0.01 or avg_manipulation_score > 0.3:
            risk_level = "中"
        
        return {
            "overall_risk": risk_level,
            "volatility_risk": "高" if avg_volatility > 0.3 else "中" if avg_volatility > 0.2 else "低",
            "liquidity_risk": "高" if avg_spread > 0.02 else "中" if avg_spread > 0.01 else "低",
            "manipulation_risk": "高" if avg_manipulation_score > 0.5 else "中" if avg_manipulation_score > 0.3 else "低"
        }
    
    def _generate_investment_recommendations(self) -> Dict[str, List[str]]:
        """生成投资建议"""
        recommendations = {
            "trading_strategy": [],
            "risk_management": [],
            "market_timing": []
        }
        
        if not self.structure_metrics:
            return recommendations
        
        recent_metrics = self.structure_metrics[-min(20, len(self.structure_metrics)):]
        
        # 基于市场结构指标生成建议
        avg_efficiency = np.mean([m.efficiency_ratio for m in recent_metrics])
        avg_spread = np.mean([m.spread for m in recent_metrics])
        avg_depth = np.mean([m.depth for m in recent_metrics])
        
        if avg_efficiency > 0.8:
            recommendations["trading_strategy"].append("市场效率较高，适合被动策略")
        else:
            recommendations["trading_strategy"].append("市场效率较低，存在套利机会")
        
        if avg_spread > 0.01:
            recommendations["trading_strategy"].append("价差较大，适合做市策略")
            recommendations["risk_management"].append("注意交易成本控制")
        else:
            recommendations["trading_strategy"].append("价差较小，适合高频交易")
        
        if avg_depth < 1000:
            recommendations["risk_management"].append("市场深度不足，大额交易需谨慎")
            recommendations["market_timing"].append("建议分批交易以减少市场冲击")
        
        return recommendations
    
    def visualize_market_structure(self, save_path: Optional[str] = None) -> None:
        """可视化市场结构分析结果"""
        if len(self.structure_metrics) < 10:
            print("数据不足，无法生成可视化图表")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('市场结构分析报告', fontsize=16, fontweight='bold')
        
        # 提取数据
        timestamps = [data['timestamp'] for data in self.market_data]
        spreads = [m.spread for m in self.structure_metrics]
        depths = [m.depth for m in self.structure_metrics]
        imbalances = [m.order_imbalance for m in self.structure_metrics]
        volatilities = [m.volatility for m in self.structure_metrics]
        manipulation_scores = [m.manipulation_score for m in self.structure_metrics]
        
        # 1. 买卖价差时间序列
        axes[0, 0].plot(timestamps, spreads, 'b-', linewidth=1.5)
        axes[0, 0].set_title('买卖价差时间序列')
        axes[0, 0].set_ylabel('价差')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 市场深度
        axes[0, 1].plot(timestamps, depths, 'g-', linewidth=1.5)
        axes[0, 1].set_title('市场深度')
        axes[0, 1].set_ylabel('深度')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 订单不平衡度
        axes[0, 2].plot(timestamps, imbalances, 'r-', linewidth=1.5)
        axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 2].set_title('订单不平衡度')
        axes[0, 2].set_ylabel('不平衡度')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 波动率
        axes[1, 0].plot(timestamps, volatilities, 'purple', linewidth=1.5)
        axes[1, 0].set_title('价格波动率')
        axes[1, 0].set_ylabel('波动率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 操纵分数
        axes[1, 1].plot(timestamps, manipulation_scores, 'orange', linewidth=1.5)
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='操纵阈值')
        axes[1, 1].set_title('市场操纵分数')
        axes[1, 1].set_ylabel('操纵分数')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 市场效率比率
        efficiency_ratios = [m.efficiency_ratio for m in self.structure_metrics]
        axes[1, 2].plot(timestamps, efficiency_ratios, 'brown', linewidth=1.5)
        axes[1, 2].set_title('市场效率比率')
        axes[1, 2].set_ylabel('效率比率')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 设置时间轴格式
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, len(timestamps)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
    
    def export_analysis_report(self, file_path: str) -> None:
        """导出分析报告"""
        report = self.generate_comprehensive_report()
        
        # 转换为DataFrame格式便于导出
        report_data = []
        
        # 基本信息
        report_data.append(['分析时间', report['timestamp']])
        report_data.append(['数据点数量', report['data_summary']['total_data_points']])
        
        # 当前指标
        if report['current_metrics']:
            for metric, value in report['current_metrics'].items():
                report_data.append([f'当前{metric}', value])
        
        # 市场结构分析
        micro_analysis = report['market_microstructure_analysis']
        if micro_analysis:
            for category, metrics in micro_analysis.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        report_data.append([f'{category}_{metric}', value])
        
        # 风险评估
        risk_assessment = report['risk_assessment']
        for risk_type, level in risk_assessment.items():
            report_data.append([f'{risk_type}_risk', level])
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(report_data, columns=['指标', '数值'])
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"分析报告已导出到: {file_path}")

# 使用示例和测试函数
def example_usage():
    """使用示例"""
    print("=== 市场结构分析器使用示例 ===")
    
    # 初始化分析器
    analyzer = MarketStructureAnalyzer(
        initial_capital=1000000,
        tick_size=0.01,
        max_order_book_levels=10
    )
    
    # 模拟市场数据
    import random
    from datetime import datetime, timedelta
    
    base_price = 100.0
    current_time = datetime.now()
    
    print("正在生成模拟市场数据...")
    
    for i in range(200):  # 生成200个时间点的数据
        # 模拟订单簿数据
        bid_data = []
        ask_data = []
        
        for level in range(5):
            bid_price = base_price - (level + 1) * 0.01 + random.uniform(-0.005, 0.005)
            ask_price = base_price + (level + 1) * 0.01 + random.uniform(-0.005, 0.005)
            bid_quantity = random.uniform(100, 1000)
            ask_quantity = random.uniform(100, 1000)
            bid_order_count = random.randint(1, 10)
            ask_order_count = random.randint(1, 10)
            
            bid_data.append((bid_price, bid_quantity, bid_order_count))
            ask_data.append((ask_price, ask_quantity, ask_order_count))
        
        # 模拟交易数据
        trade_data = []
        for _ in range(random.randint(1, 5)):
            trade_price = base_price + random.uniform(-0.02, 0.02)
            trade_quantity = random.uniform(50, 500)
            trade_type = random.choice(['buy', 'sell'])
            aggressor_side = random.choice(['buyer', 'seller'])
            trade_data.append((trade_price, trade_quantity, trade_type, aggressor_side))
        
        # 更新分析器
        timestamp = current_time + timedelta(seconds=i)
        analyzer.update_market_data(bid_data, ask_data, trade_data, timestamp)
        
        # 模拟价格变化
        base_price += random.uniform(-0.01, 0.01)
    
    print("数据分析完成，开始生成报告...")
    
    # 生成综合报告
    report = analyzer.generate_comprehensive_report()
    
    # 打印关键指标
    print("\n=== 分析结果摘要 ===")
    print(f"分析期间: {report['data_summary']['analysis_period']['start']} 到 {report['data_summary']['analysis_period']['end']}")
    print(f"数据点数量: {report['data_summary']['total_data_points']}")
    
    current_metrics = report['current_metrics']
    print(f"\n当前市场指标:")
    print(f"  买卖价差: {current_metrics.get('spread', 0):.4f}")
    print(f"  市场深度: {current_metrics.get('depth', 0):.2f}")
    print(f"  订单不平衡度: {current_metrics.get('order_imbalance', 0):.4f}")
    print(f"  价格冲击: {current_metrics.get('price_impact', 0):.4f}")
    print(f"  波动率: {current_metrics.get('volatility', 0):.4f}")
    print(f"  市场效率: {current_metrics.get('efficiency_ratio', 0):.4f}")
    print(f"  操纵分数: {current_metrics.get('manipulation_score', 0):.4f}")
    
    # 风险评估
    risk_assessment = report['risk_assessment']
    print(f"\n风险评估:")
    for risk_type, level in risk_assessment.items():
        print(f"  {risk_type}: {level}")
    
    # 投资建议
    recommendations = report['recommendations']
    print(f"\n投资建议:")
    for category, advice_list in recommendations.items():
        if advice_list:
            print(f"  {category}:")
            for advice in advice_list:
                print(f"    - {advice}")
    
    # 生成可视化图表
    print("\n正在生成可视化图表...")
    try:
        analyzer.visualize_market_structure('/workspace/market_structure_analysis.png')
        print("图表生成完成")
    except Exception as e:
        print(f"图表生成失败: {e}")
    
    # 导出报告
    print("\n正在导出分析报告...")
    try:
        analyzer.export_analysis_report('/workspace/market_structure_report.csv')
        print("报告导出完成")
    except Exception as e:
        print(f"报告导出失败: {e}")
    
    return analyzer, report

if __name__ == "__main__":
    # 运行示例
    analyzer, report = example_usage()
    
    print("\n=== 市场结构分析器初始化完成 ===")
    print("主要功能:")
    print("1. 市场微观结构分析 - 订单簿深度、买卖盘结构")
    print("2. 市场参与者结构分析 - 交易模式、参与者分类")
    print("3. 价格发现机制分析 - 效率指标、价格改进")
    print("4. 市场效率评估 - 随机游走检验、信息比率")
    print("5. 市场冲击成本分析 - 临时和永久冲击")
    print("6. 市场操纵检测 - 拉高砸盘、虚假下单")
    print("7. 市场结构变化监控 - 实时监控、预警系统")
    print("\n使用方法:")
    print("analyzer = MarketStructureAnalyzer()")
    print("analyzer.update_market_data(bid_data, ask_data, trade_data, timestamp)")
    print("report = analyzer.generate_comprehensive_report()")
    print("analyzer.visualize_market_structure()")