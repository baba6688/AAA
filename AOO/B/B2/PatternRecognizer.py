"""
B2模式识别器
实现多种金融市场的模式识别功能，包括价格模式、K线形态、成交量模式等
支持深度学习模型、实时监控和历史回测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 深度学习和机器学习
try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告：TensorFlow未安装，将使用基础模式识别")

# 技术分析库
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("警告：TA-Lib未安装，将使用基础技术指标")

# 可视化
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


@dataclass
class PatternResult:
    """模式识别结果"""
    pattern_type: str
    confidence: float
    start_idx: int
    end_idx: int
    strength: float
    direction: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class PatternSignal:
    """模式信号"""
    symbol: str
    pattern_type: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price: float
    timestamp: datetime
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = None


class PricePatternRecognizer:
    """价格模式识别器"""
    
    def __init__(self):
        self.patterns = {
            'head_and_shoulders': self._detect_head_shoulders,
            'inverse_head_shoulders': self._detect_inverse_head_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'triangle_ascending': self._detect_ascending_triangle,
            'triangle_descending': self._detect_descending_triangle,
            'triangle_symmetrical': self._detect_symmetrical_triangle,
            'flag': self._detect_flag,
            'pennant': self._detect_pennant,
            'rectangle': self._detect_rectangle
        }
    
    def detect_patterns(self, data: pd.DataFrame, min_strength: float = 0.6) -> List[PatternResult]:
        """检测所有价格模式"""
        results = []
        
        for pattern_name, pattern_func in self.patterns.items():
            try:
                patterns = pattern_func(data, min_strength)
                results.extend(patterns)
            except Exception as e:
                print(f"检测 {pattern_name} 时出错: {e}")
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    def _detect_head_shoulders(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测头肩顶模式"""
        results = []
        highs = data['high'].values
        
        # 寻找局部高点
        peaks = self._find_peaks(highs, prominence=0.02)
        
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # 检查头肩顶条件
            if (highs[head] > highs[left_shoulder] and 
                highs[head] > highs[right_shoulder] and
                abs(highs[left_shoulder] - highs[right_shoulder]) / highs[head] < 0.05):
                
                # 计算颈线
                neckline_start = data['low'].iloc[left_shoulder:head].min()
                neckline_end = data['low'].iloc[head:right_shoulder].min()
                neckline = (neckline_start + neckline_end) / 2
                
                # 计算强度和置信度
                shoulder_diff = abs(highs[left_shoulder] - highs[right_shoulder]) / highs[head]
                head_height = (highs[head] - neckline) / highs[head]
                confidence = max(0, 1 - shoulder_diff - head_height * 0.5)
                
                if confidence >= min_strength:
                    target_price = neckline - (highs[head] - neckline)
                    stop_loss = highs[head] * 1.02
                    
                    results.append(PatternResult(
                        pattern_type='head_and_shoulders',
                        confidence=confidence,
                        start_idx=left_shoulder,
                        end_idx=right_shoulder,
                        strength=head_height,
                        direction='BEARISH',
                        target_price=target_price,
                        stop_loss=stop_loss,
                        metadata={
                            'head_price': highs[head],
                            'neckline': neckline,
                            'left_shoulder': highs[left_shoulder],
                            'right_shoulder': highs[right_shoulder]
                        }
                    ))
        
        return results
    
    def _detect_inverse_head_shoulders(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测倒头肩顶模式"""
        results = []
        lows = data['low'].values
        
        # 寻找局部低点
        troughs = self._find_troughs(lows, prominence=0.02)
        
        for i in range(len(troughs) - 2):
            left_shoulder = troughs[i]
            head = troughs[i + 1]
            right_shoulder = troughs[i + 2]
            
            # 检查倒头肩顶条件
            if (lows[head] < lows[left_shoulder] and 
                lows[head] < lows[right_shoulder] and
                abs(lows[left_shoulder] - lows[right_shoulder]) / lows[head] < 0.05):
                
                # 计算颈线
                neckline_start = data['high'].iloc[left_shoulder:head].max()
                neckline_end = data['high'].iloc[head:right_shoulder].max()
                neckline = (neckline_start + neckline_end) / 2
                
                # 计算强度和置信度
                shoulder_diff = abs(lows[left_shoulder] - lows[right_shoulder]) / lows[head]
                head_depth = (neckline - lows[head]) / lows[head]
                confidence = max(0, 1 - shoulder_diff - head_depth * 0.5)
                
                if confidence >= min_strength:
                    target_price = neckline + (neckline - lows[head])
                    stop_loss = lows[head] * 0.98
                    
                    results.append(PatternResult(
                        pattern_type='inverse_head_shoulders',
                        confidence=confidence,
                        start_idx=left_shoulder,
                        end_idx=right_shoulder,
                        strength=head_depth,
                        direction='BULLISH',
                        target_price=target_price,
                        stop_loss=stop_loss,
                        metadata={
                            'head_price': lows[head],
                            'neckline': neckline,
                            'left_shoulder': lows[left_shoulder],
                            'right_shoulder': lows[right_shoulder]
                        }
                    ))
        
        return results
    
    def _detect_double_top(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测双顶模式"""
        results = []
        highs = data['high'].values
        
        peaks = self._find_peaks(highs, prominence=0.015)
        
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]
            
            # 检查双顶条件
            price_diff = abs(highs[peak1] - highs[peak2]) / highs[peak1]
            time_diff = peak2 - peak1
            
            if price_diff < 0.03 and 5 <= time_diff <= 50:  # 顶之间距离合理
                # 寻找颈线（两个峰之间的最低点）
                neckline_idx = np.argmin(data['low'].iloc[peak1:peak2+1])
                neckline = data['low'].iloc[peak1 + neckline_idx]
                
                # 计算置信度
                confidence = max(0, 1 - price_diff * 10 - time_diff * 0.01)
                
                if confidence >= min_strength:
                    target_price = neckline - (max(highs[peak1], highs[peak2]) - neckline)
                    stop_loss = max(highs[peak1], highs[peak2]) * 1.01
                    
                    results.append(PatternResult(
                        pattern_type='double_top',
                        confidence=confidence,
                        start_idx=peak1,
                        end_idx=peak2,
                        strength=price_diff,
                        direction='BEARISH',
                        target_price=target_price,
                        stop_loss=stop_loss,
                        metadata={
                            'peak1_price': highs[peak1],
                            'peak2_price': highs[peak2],
                            'neckline': neckline,
                            'time_span': time_diff
                        }
                    ))
        
        return results
    
    def _detect_double_bottom(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测双底模式"""
        results = []
        lows = data['low'].values
        
        troughs = self._find_troughs(lows, prominence=0.015)
        
        for i in range(len(troughs) - 1):
            trough1 = troughs[i]
            trough2 = troughs[i + 1]
            
            # 检查双底条件
            price_diff = abs(lows[trough1] - lows[trough2]) / lows[trough1]
            time_diff = trough2 - trough1
            
            if price_diff < 0.03 and 5 <= time_diff <= 50:
                # 寻找颈线（两个底之间的最高点）
                neckline_idx = np.argmax(data['high'].iloc[trough1:trough2+1])
                neckline = data['high'].iloc[trough1 + neckline_idx]
                
                # 计算置信度
                confidence = max(0, 1 - price_diff * 10 - time_diff * 0.01)
                
                if confidence >= min_strength:
                    target_price = neckline + (neckline - min(lows[trough1], lows[trough2]))
                    stop_loss = min(lows[trough1], lows[trough2]) * 0.99
                    
                    results.append(PatternResult(
                        pattern_type='double_bottom',
                        confidence=confidence,
                        start_idx=trough1,
                        end_idx=trough2,
                        strength=price_diff,
                        direction='BULLISH',
                        target_price=target_price,
                        stop_loss=stop_loss,
                        metadata={
                            'trough1_price': lows[trough1],
                            'trough2_price': lows[trough2],
                            'neckline': neckline,
                            'time_span': time_diff
                        }
                    ))
        
        return results
    
    def _detect_ascending_triangle(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测上升三角形"""
        results = []
        
        # 简化实现：寻找阻力线和支撑线
        recent_data = data.tail(50)  # 最近50个数据点
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # 寻找水平阻力线
        resistance_level = np.percentile(highs[-20:], 90)  # 最近20个高点的90%分位数
        
        # 检查阻力线附近的高点
        resistance_touches = 0
        for high in highs[-20:]:
            if abs(high - resistance_level) / resistance_level < 0.01:
                resistance_touches += 1
        
        # 检查上升趋势的支撑线
        if len(lows) >= 10:
            support_slope = np.polyfit(range(len(lows[-10:])), lows[-10:], 1)[0]
            
            if resistance_touches >= 3 and support_slope > 0:
                confidence = min(0.9, resistance_touches * 0.2 + support_slope * 10)
                
                if confidence >= min_strength:
                    target_price = resistance_level + (resistance_level - np.min(lows[-10:]))
                    
                    results.append(PatternResult(
                        pattern_type='triangle_ascending',
                        confidence=confidence,
                        start_idx=len(data) - 50,
                        end_idx=len(data) - 1,
                        strength=resistance_touches,
                        direction='BULLISH',
                        target_price=target_price,
                        stop_loss=np.min(lows[-10:]) * 0.98,
                        metadata={
                            'resistance_level': resistance_level,
                            'support_slope': support_slope,
                            'resistance_touches': resistance_touches
                        }
                    ))
        
        return results
    
    def _detect_descending_triangle(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测下降三角形"""
        results = []
        
        recent_data = data.tail(50)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # 寻找水平支撑线
        support_level = np.percentile(lows[-20:], 10)
        
        # 检查支撑线附近的低点
        support_touches = 0
        for low in lows[-20:]:
            if abs(low - support_level) / support_level < 0.01:
                support_touches += 1
        
        # 检查下降趋势的阻力线
        if len(highs) >= 10:
            resistance_slope = np.polyfit(range(len(highs[-10:])), highs[-10:], 1)[0]
            
            if support_touches >= 3 and resistance_slope < 0:
                confidence = min(0.9, support_touches * 0.2 - resistance_slope * 10)
                
                if confidence >= min_strength:
                    target_price = support_level - (np.max(highs[-10:]) - support_level)
                    
                    results.append(PatternResult(
                        pattern_type='triangle_descending',
                        confidence=confidence,
                        start_idx=len(data) - 50,
                        end_idx=len(data) - 1,
                        strength=support_touches,
                        direction='BEARISH',
                        target_price=target_price,
                        stop_loss=np.max(highs[-10:]) * 1.02,
                        metadata={
                            'support_level': support_level,
                            'resistance_slope': resistance_slope,
                            'support_touches': support_touches
                        }
                    ))
        
        return results
    
    def _detect_symmetrical_triangle(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测对称三角形"""
        results = []
        
        recent_data = data.tail(50)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        if len(highs) >= 10 and len(lows) >= 10:
            # 计算趋势线斜率
            resistance_slope = np.polyfit(range(len(highs[-10:])), highs[-10:], 1)[0]
            support_slope = np.polyfit(range(len(lows[-10:])), lows[-10:], 1)[0]
            
            # 对称三角形：阻力线下降，支撑线上升
            if resistance_slope < 0 and support_slope > 0:
                convergence = abs(resistance_slope + support_slope)
                confidence = min(0.9, convergence * 50)
                
                if confidence >= min_strength:
                    # 目标价格为三角形高度的2倍
                    triangle_height = np.max(highs[-10:]) - np.min(lows[-10:])
                    breakout_point = (np.max(highs[-10:]) + np.min(lows[-10:])) / 2
                    
                    results.append(PatternResult(
                        pattern_type='triangle_symmetrical',
                        confidence=confidence,
                        start_idx=len(data) - 50,
                        end_idx=len(data) - 1,
                        strength=convergence,
                        direction='NEUTRAL',
                        target_price=breakout_point + triangle_height,
                        metadata={
                            'resistance_slope': resistance_slope,
                            'support_slope': support_slope,
                            'triangle_height': triangle_height
                        }
                    ))
        
        return results
    
    def _detect_flag(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测旗形模式"""
        results = []
        
        # 简化实现：寻找强趋势后的整理形态
        if len(data) < 20:
            return results
        
        recent_data = data.tail(20)
        
        # 计算价格变化
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # 检查是否在合理范围内（旗形通常是强势后的整理）
        if 0.02 <= abs(price_change) <= 0.15:
            # 检查波动性是否降低
            volatility_current = recent_data['close'].pct_change().std()
            volatility_reference = data['close'].tail(40).pct_change().std()
            
            if volatility_current < volatility_reference * 0.7:
                direction = 'BULLISH' if price_change > 0 else 'BEARISH'
                confidence = min(0.8, (0.15 - abs(price_change)) * 10 + (1 - volatility_current/volatility_reference) * 0.5)
                
                if confidence >= min_strength:
                    results.append(PatternResult(
                        pattern_type='flag',
                        confidence=confidence,
                        start_idx=len(data) - 20,
                        end_idx=len(data) - 1,
                        strength=abs(price_change),
                        direction=direction,
                        metadata={
                            'price_change': price_change,
                            'volatility_reduction': 1 - volatility_current/volatility_reference
                        }
                    ))
        
        return results
    
    def _detect_pennant(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测尖旗形模式"""
        results = []
        
        # 尖旗形类似于旗形，但持续时间更短，收敛更快
        if len(data) < 15:
            return results
        
        recent_data = data.tail(15)
        
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        if 0.01 <= abs(price_change) <= 0.10:
            # 检查价格收敛
            price_range_start = recent_data['high'].iloc[0] - recent_data['low'].iloc[0]
            price_range_end = recent_data['high'].iloc[-1] - recent_data['low'].iloc[-1]
            
            if price_range_end < price_range_start * 0.7:  # 价格范围显著收敛
                direction = 'BULLISH' if price_change > 0 else 'BEARISH'
                convergence = 1 - price_range_end / price_range_start
                confidence = min(0.8, convergence * 2 + abs(price_change) * 5)
                
                if confidence >= min_strength:
                    results.append(PatternResult(
                        pattern_type='pennant',
                        confidence=confidence,
                        start_idx=len(data) - 15,
                        end_idx=len(data) - 1,
                        strength=convergence,
                        direction=direction,
                        metadata={
                            'price_change': price_change,
                            'convergence': convergence
                        }
                    ))
        
        return results
    
    def _detect_rectangle(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测矩形整理模式"""
        results = []
        
        if len(data) < 30:
            return results
        
        recent_data = data.tail(30)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # 计算支撑和阻力水平
        resistance = np.percentile(highs, 90)
        support = np.percentile(lows, 10)
        
        # 检查价格在支撑阻力间波动的频率
        touches_resistance = sum(1 for h in highs if abs(h - resistance) / resistance < 0.01)
        touches_support = sum(1 for l in lows if abs(l - support) / support < 0.01)
        
        if touches_resistance >= 3 and touches_support >= 3:
            rectangle_height = resistance - support
            confidence = min(0.8, (touches_resistance + touches_support) * 0.1)
            
            if confidence >= min_strength:
                results.append(PatternResult(
                    pattern_type='rectangle',
                    confidence=confidence,
                    start_idx=len(data) - 30,
                    end_idx=len(data) - 1,
                    strength=rectangle_height / support,
                    direction='NEUTRAL',
                    metadata={
                        'resistance': resistance,
                        'support': support,
                        'touches_resistance': touches_resistance,
                        'touches_support': touches_support
                    }
                ))
        
        return results
    
    def _find_peaks(self, data: np.ndarray, prominence: float = 0.02) -> List[int]:
        """寻找局部高点"""
        peaks = []
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and data[i] > data[i+1] and 
                data[i] - max(data[i-1], data[i+1]) > prominence * data[i]):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, data: np.ndarray, prominence: float = 0.02) -> List[int]:
        """寻找局部低点"""
        troughs = []
        for i in range(1, len(data) - 1):
            if (data[i] < data[i-1] and data[i] < data[i+1] and 
                min(data[i-1], data[i+1]) - data[i] > prominence * data[i]):
                troughs.append(i)
        return troughs


class CandlestickPatternRecognizer:
    """K线形态识别器"""
    
    def __init__(self):
        self.patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'shooting_star': self._detect_shooting_star,
            'engulfing_bullish': self._detect_bullish_engulfing,
            'engulfing_bearish': self._detect_bearish_engulfing,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star,
            'three_white_soldiers': self._detect_three_white_soldiers,
            'three_black_crows': self._detect_three_black_crows,
            'spinning_top': self._detect_spinning_top
        }
    
    def detect_patterns(self, data: pd.DataFrame, min_strength: float = 0.6) -> List[PatternResult]:
        """检测所有K线形态"""
        results = []
        
        for pattern_name, pattern_func in self.patterns.items():
            try:
                patterns = pattern_func(data, min_strength)
                results.extend(patterns)
            except Exception as e:
                print(f"检测 {pattern_name} 时出错: {e}")
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    def _detect_doji(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测十字星"""
        results = []
        
        for i in range(1, len(data)):
            row = data.iloc[i]
            body_size = abs(row['close'] - row['open'])
            total_range = row['high'] - row['low']
            
            if total_range > 0 and body_size / total_range < 0.1:  # 实体很小
                confidence = 1 - body_size / total_range
                
                if confidence >= min_strength:
                    direction = 'BULLISH' if row['close'] > row['open'] else 'BEARISH'
                    
                    results.append(PatternResult(
                        pattern_type='doji',
                        confidence=confidence,
                        start_idx=i,
                        end_idx=i,
                        strength=body_size / total_range,
                        direction='NEUTRAL',
                        metadata={
                            'body_size': body_size,
                            'total_range': total_range,
                            'close_open_ratio': body_size / total_range
                        }
                    ))
        
        return results
    
    def _detect_hammer(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测锤头线"""
        results = []
        
        for i in range(1, len(data)):
            row = data.iloc[i]
            body_size = abs(row['close'] - row['open'])
            upper_shadow = row['high'] - max(row['close'], row['open'])
            lower_shadow = min(row['close'], row['open']) - row['low']
            total_range = row['high'] - row['low']
            
            # 锤头线条件：下影线长，实体小，上影线短
            if (total_range > 0 and 
                lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5):
                
                confidence = min(0.9, (lower_shadow / total_range) * 2)
                
                if confidence >= min_strength:
                    results.append(PatternResult(
                        pattern_type='hammer',
                        confidence=confidence,
                        start_idx=i,
                        end_idx=i,
                        strength=lower_shadow / total_range,
                        direction='BULLISH',
                        metadata={
                            'body_size': body_size,
                            'lower_shadow': lower_shadow,
                            'upper_shadow': upper_shadow
                        }
                    ))
        
        return results
    
    def _detect_shooting_star(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测流星线"""
        results = []
        
        for i in range(1, len(data)):
            row = data.iloc[i]
            body_size = abs(row['close'] - row['open'])
            upper_shadow = row['high'] - max(row['close'], row['open'])
            lower_shadow = min(row['close'], row['open']) - row['low']
            total_range = row['high'] - row['low']
            
            # 流星线条件：上影线长，实体小，下影线短
            if (total_range > 0 and 
                upper_shadow > body_size * 2 and 
                lower_shadow < body_size * 0.5):
                
                confidence = min(0.9, (upper_shadow / total_range) * 2)
                
                if confidence >= min_strength:
                    results.append(PatternResult(
                        pattern_type='shooting_star',
                        confidence=confidence,
                        start_idx=i,
                        end_idx=i,
                        strength=upper_shadow / total_range,
                        direction='BEARISH',
                        metadata={
                            'body_size': body_size,
                            'upper_shadow': upper_shadow,
                            'lower_shadow': lower_shadow
                        }
                    ))
        
        return results
    
    def _detect_bullish_engulfing(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测看涨吞噬形态"""
        results = []
        
        for i in range(1, len(data)):
            prev_row = data.iloc[i-1]
            curr_row = data.iloc[i]
            
            # 前一根是阴线，后一根是阳线且吞噬前一根
            if (prev_row['close'] < prev_row['open'] and  # 前一根阴线
                curr_row['close'] > curr_row['open'] and  # 当前阳线
                curr_row['open'] < prev_row['close'] and  # 低开
                curr_row['close'] > prev_row['open']):    # 高收
                
                engulf_size = (curr_row['close'] - curr_row['open']) + (prev_row['open'] - prev_row['close'])
                prev_size = prev_row['open'] - prev_row['close']
                
                confidence = min(0.9, engulf_size / prev_size)
                
                if confidence >= min_strength:
                    results.append(PatternResult(
                        pattern_type='engulfing_bullish',
                        confidence=confidence,
                        start_idx=i-1,
                        end_idx=i,
                        strength=engulf_size / prev_size,
                        direction='BULLISH',
                        metadata={
                            'engulf_size': engulf_size,
                            'prev_size': prev_size
                        }
                    ))
        
        return results
    
    def _detect_bearish_engulfing(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测看跌吞噬形态"""
        results = []
        
        for i in range(1, len(data)):
            prev_row = data.iloc[i-1]
            curr_row = data.iloc[i]
            
            # 前一根是阳线，后一根是阴线且吞噬前一根
            if (prev_row['close'] > prev_row['open'] and  # 前一根阳线
                curr_row['close'] < curr_row['open'] and  # 当前阴线
                curr_row['open'] > prev_row['close'] and  # 高开
                curr_row['close'] < prev_row['open']):    # 低收
                
                engulf_size = (curr_row['open'] - curr_row['close']) + (prev_row['close'] - prev_row['open'])
                prev_size = prev_row['close'] - prev_row['open']
                
                confidence = min(0.9, engulf_size / prev_size)
                
                if confidence >= min_strength:
                    results.append(PatternResult(
                        pattern_type='engulfing_bearish',
                        confidence=confidence,
                        start_idx=i-1,
                        end_idx=i,
                        strength=engulf_size / prev_size,
                        direction='BEARISH',
                        metadata={
                            'engulf_size': engulf_size,
                            'prev_size': prev_size
                        }
                    ))
        
        return results
    
    def _detect_morning_star(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测晨星形态"""
        results = []
        
        for i in range(2, len(data)):
            row1 = data.iloc[i-2]
            row2 = data.iloc[i-1]
            row3 = data.iloc[i]
            
            # 晨星条件：1.长阴线 2.星形线 3.长阳线
            if (row1['close'] < row1['open'] * 0.98 and  # 长阴线
                abs(row2['close'] - row2['open']) < (row1['high'] - row1['low']) * 0.3 and  # 星形线
                row3['close'] > row3['open'] * 1.02 and  # 长阳线
                row3['close'] > (row1['open'] + row1['close']) / 2):  # 突破中点
                
                confidence = 0.8
                
                if confidence >= min_strength:
                    results.append(PatternResult(
                        pattern_type='morning_star',
                        confidence=confidence,
                        start_idx=i-2,
                        end_idx=i,
                        strength=1.0,
                        direction='BULLISH',
                        metadata={
                            'pattern_strength': 'strong'
                        }
                    ))
        
        return results
    
    def _detect_evening_star(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测黄昏星形态"""
        results = []
        
        for i in range(2, len(data)):
            row1 = data.iloc[i-2]
            row2 = data.iloc[i-1]
            row3 = data.iloc[i]
            
            # 黄昏星条件：1.长阳线 2.星形线 3.长阴线
            if (row1['close'] > row1['open'] * 1.02 and  # 长阳线
                abs(row2['close'] - row2['open']) < (row1['high'] - row1['low']) * 0.3 and  # 星形线
                row3['close'] < row3['open'] * 0.98 and  # 长阴线
                row3['close'] < (row1['open'] + row1['close']) / 2):  # 跌破中点
                
                confidence = 0.8
                
                if confidence >= min_strength:
                    results.append(PatternResult(
                        pattern_type='evening_star',
                        confidence=confidence,
                        start_idx=i-2,
                        end_idx=i,
                        strength=1.0,
                        direction='BEARISH',
                        metadata={
                            'pattern_strength': 'strong'
                        }
                    ))
        
        return results
    
    def _detect_three_white_soldiers(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测三白兵形态"""
        results = []
        
        for i in range(3, len(data)):
            rows = [data.iloc[i-2], data.iloc[i-1], data.iloc[i]]
            
            # 检查是否都是阳线且逐步上升
            if all(row['close'] > row['open'] for row in rows):
                # 检查收盘价是否逐步上升
                if (rows[0]['close'] < rows[1]['close'] < rows[2]['close'] and
                    all(abs(row['close'] - row['open']) > (row['high'] - row['low']) * 0.6 
                        for row in rows)):
                    
                    confidence = 0.85
                    
                    if confidence >= min_strength:
                        results.append(PatternResult(
                            pattern_type='three_white_soldiers',
                            confidence=confidence,
                            start_idx=i-2,
                            end_idx=i,
                            strength=1.0,
                            direction='BULLISH',
                            metadata={
                                'consecutive_rises': rows[2]['close'] - rows[0]['close']
                            }
                        ))
        
        return results
    
    def _detect_three_black_crows(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测三只乌鸦形态"""
        results = []
        
        for i in range(3, len(data)):
            rows = [data.iloc[i-2], data.iloc[i-1], data.iloc[i]]
            
            # 检查是否都是阴线且逐步下降
            if all(row['close'] < row['open'] for row in rows):
                # 检查收盘价是否逐步下降
                if (rows[0]['close'] > rows[1]['close'] > rows[2]['close'] and
                    all(abs(row['close'] - row['open']) > (row['high'] - row['low']) * 0.6 
                        for row in rows)):
                    
                    confidence = 0.85
                    
                    if confidence >= min_strength:
                        results.append(PatternResult(
                            pattern_type='three_black_crows',
                            confidence=confidence,
                            start_idx=i-2,
                            end_idx=i,
                            strength=1.0,
                            direction='BEARISH',
                            metadata={
                                'consecutive_declines': rows[0]['close'] - rows[2]['close']
                            }
                        ))
        
        return results
    
    def _detect_spinning_top(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测纺锤线"""
        results = []
        
        for i in range(1, len(data)):
            row = data.iloc[i]
            body_size = abs(row['close'] - row['open'])
            total_range = row['high'] - row['low']
            upper_shadow = row['high'] - max(row['close'], row['open'])
            lower_shadow = min(row['close'], row['open']) - row['low']
            
            # 纺锤线条件：实体小，上下影线相对较长且相近
            if (total_range > 0 and 
                body_size / total_range < 0.3 and
                abs(upper_shadow - lower_shadow) / total_range < 0.2):
                
                confidence = 1 - body_size / total_range
                
                if confidence >= min_strength:
                    results.append(PatternResult(
                        pattern_type='spinning_top',
                        confidence=confidence,
                        start_idx=i,
                        end_idx=i,
                        strength=body_size / total_range,
                        direction='NEUTRAL',
                        metadata={
                            'body_ratio': body_size / total_range,
                            'shadow_balance': 1 - abs(upper_shadow - lower_shadow) / total_range
                        }
                    ))
        
        return results


class VolumePatternRecognizer:
    """成交量模式识别器"""
    
    def __init__(self):
        self.volume_ma_periods = [5, 10, 20, 50]
    
    def detect_patterns(self, data: pd.DataFrame, min_strength: float = 0.6) -> List[PatternResult]:
        """检测成交量模式"""
        results = []
        
        # 计算成交量移动平均
        for period in self.volume_ma_periods:
            if len(data) >= period:
                data[f'volume_ma_{period}'] = data['volume'].rolling(window=period).mean()
        
        # 检测各种成交量模式
        results.extend(self._detect_volume_surge(data, min_strength))
        results.extend(self._detect_volume_divergence(data, min_strength))
        results.extend(self._detect_volume_climax(data, min_strength))
        results.extend(self._detect_volume_accumulation(data, min_strength))
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    def _detect_volume_surge(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测成交量激增"""
        results = []
        
        for period in self.volume_ma_periods:
            if f'volume_ma_{period}' in data.columns:
                for i in range(period, len(data)):
                    current_volume = data['volume'].iloc[i]
                    avg_volume = data[f'volume_ma_{period}'].iloc[i]
                    
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        
                        if volume_ratio > 2.0:  # 成交量是平均的2倍以上
                            confidence = min(0.9, (volume_ratio - 1) * 0.3)
                            
                            if confidence >= min_strength:
                                # 判断方向
                                price_change = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                                direction = 'BULLISH' if price_change > 0 else 'BEARISH'
                                
                                results.append(PatternResult(
                                    pattern_type=f'volume_surge_{period}',
                                    confidence=confidence,
                                    start_idx=max(0, i-5),
                                    end_idx=i,
                                    strength=volume_ratio,
                                    direction=direction,
                                    metadata={
                                        'volume_ratio': volume_ratio,
                                        'avg_volume': avg_volume,
                                        'current_volume': current_volume,
                                        'period': period
                                    }
                                ))
        
        return results
    
    def _detect_volume_divergence(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测量价背离"""
        results = []
        
        if len(data) < 20:
            return results
        
        # 计算价格趋势和成交量趋势
        price_change = data['close'].pct_change(10).iloc[-1]  # 10期价格变化
        volume_change = data['volume'].pct_change(10).iloc[-1]  # 10期成交量变化
        
        # 检查背离：价格和成交量趋势相反
        if abs(price_change) > 0.05 and abs(volume_change) > 0.3:
            if (price_change > 0 and volume_change < -0.3) or (price_change < 0 and volume_change > 0.3):
                confidence = min(0.8, abs(price_change) * 5 + abs(volume_change))
                
                if confidence >= min_strength:
                    direction = 'BEARISH' if price_change > 0 else 'BULLISH'
                    
                    results.append(PatternResult(
                        pattern_type='volume_divergence',
                        confidence=confidence,
                        start_idx=len(data) - 20,
                        end_idx=len(data) - 1,
                        strength=abs(price_change) + abs(volume_change),
                        direction=direction,
                        metadata={
                            'price_change': price_change,
                            'volume_change': volume_change
                        }
                    ))
        
        return results
    
    def _detect_volume_climax(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测成交量高潮"""
        results = []
        
        if len(data) < 10:
            return results
        
        recent_volumes = data['volume'].tail(10).values
        max_volume = np.max(recent_volumes)
        max_idx = np.argmax(recent_volumes)
        
        # 检查是否显著高于其他时期
        other_volumes = np.delete(recent_volumes, max_idx)
        avg_other_volume = np.mean(other_volumes)
        
        if max_volume > avg_other_volume * 3:
            confidence = min(0.9, max_volume / avg_other_volume * 0.2)
            
            if confidence >= min_strength:
                # 根据价格方向判断
                price_at_climax = data['close'].iloc[len(data) - 10 + max_idx]
                price_before = data['close'].iloc[len(data) - 15] if len(data) >= 15 else data['close'].iloc[0]
                price_after = data['close'].iloc[-1]
                
                # 检查后续价格走势
                if price_after > price_at_climax * 1.02:
                    direction = 'BULLISH'
                elif price_after < price_at_climax * 0.98:
                    direction = 'BEARISH'
                else:
                    direction = 'NEUTRAL'
                
                results.append(PatternResult(
                    pattern_type='volume_climax',
                    confidence=confidence,
                    start_idx=len(data) - 10 + max_idx,
                    end_idx=len(data) - 1,
                    strength=max_volume / avg_other_volume,
                    direction=direction,
                    metadata={
                        'max_volume': max_volume,
                        'avg_volume': avg_other_volume,
                        'volume_ratio': max_volume / avg_other_volume
                    }
                ))
        
        return results
    
    def _detect_volume_accumulation(self, data: pd.DataFrame, min_strength: float) -> List[PatternResult]:
        """检测成交量累积"""
        results = []
        
        if len(data) < 20:
            return results
        
        # 检查价格横盘整理期间成交量是否放大
        recent_data = data.tail(20)
        price_std = recent_data['close'].std()
        price_mean = recent_data['close'].mean()
        price_cv = price_std / price_mean  # 变异系数
        
        # 价格相对稳定但成交量放大
        if price_cv < 0.05:  # 价格变异系数小于5%
            avg_volume_recent = recent_data['volume'].mean()
            avg_volume_earlier = data['volume'].iloc[-40:-20].mean() if len(data) >= 40 else data['volume'].mean()
            
            if avg_volume_earlier > 0:
                volume_increase = avg_volume_recent / avg_volume_earlier
                
                if volume_increase > 1.5:
                    confidence = min(0.8, (volume_increase - 1) * 0.4)
                    
                    if confidence >= min_strength:
                        results.append(PatternResult(
                            pattern_type='volume_accumulation',
                            confidence=confidence,
                            start_idx=len(data) - 20,
                            end_idx=len(data) - 1,
                            strength=volume_increase,
                            direction='BULLISH',
                            metadata={
                                'volume_increase': volume_increase,
                                'price_stability': 1 - price_cv,
                                'avg_volume_recent': avg_volume_recent,
                                'avg_volume_earlier': avg_volume_earlier
                            }
                        ))
        
        return results


class DeepLearningPatternModel:
    """深度学习模式识别模型"""
    
    def __init__(self, model_type: str = 'cnn'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if TENSORFLOW_AVAILABLE:
            self._build_model()
        else:
            print("警告：TensorFlow不可用，将使用传统机器学习方法")
    
    def _build_model(self):
        """构建深度学习模型"""
        if not TENSORFLOW_AVAILABLE:
            return
        
        if self.model_type == 'cnn':
            # CNN模型用于图像化的K线数据
            self.model = keras.Sequential([
                keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 10, 4)),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation='softmax')  # 10种模式
            ])
            
        elif self.model_type == 'lstm':
            # LSTM模型用于时间序列数据
            self.model = keras.Sequential([
                keras.layers.LSTM(50, return_sequences=True, input_shape=(50, 4)),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(50, return_sequences=False),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(25),
                keras.layers.Dense(10, activation='softmax')
            ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_features(self, data: pd.DataFrame, window_size: int = 50) -> np.ndarray:
        """准备特征数据"""
        features = []
        
        # 技术指标特征
        if TALIB_AVAILABLE:
            features.extend([
                talib.SMA(data['close'], timeperiod=20),
                talib.EMA(data['close'], timeperiod=20),
                talib.RSI(data['close'], timeperiod=14),
                talib.MACD(data['close'])[0],
                talib.BBANDS(data['close'])[0]
            ])
        
        # 价格特征
        features.extend([
            data['high'] / data['close'] - 1,  # 上影线比例
            data['low'] / data['close'] - 1,   # 下影线比例
            (data['close'] - data['open']) / data['close'],  # 实体比例
            data['volume'] / data['volume'].rolling(20).mean()  # 成交量比率
        ])
        
        # 转换为numpy数组并处理NaN值
        feature_array = np.column_stack(features)
        feature_array = np.nan_to_num(feature_array)
        
        # 标准化
        if not self.is_trained:
            feature_array = self.scaler.fit_transform(feature_array)
        else:
            feature_array = self.scaler.transform(feature_array)
        
        return feature_array
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """训练模型"""
        if TENSORFLOW_AVAILABLE and self.model is not None:
            # 深度学习训练
            history = self.model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)
            self.is_trained = True
            return history
        else:
            # 使用传统机器学习方法
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100)
            self.model.fit(X, y)
            self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测模式"""
        if TENSORFLOW_AVAILABLE and self.model is not None and self.is_trained:
            return self.model.predict(X)
        elif hasattr(self, 'model') and self.is_trained:
            return self.model.predict_proba(X)
        else:
            # 返回随机预测（用于演示）
            return np.random.random((len(X), 10))
    
    def predict_patterns(self, data: pd.DataFrame) -> List[PatternResult]:
        """预测模式"""
        if not self.is_trained:
            print("模型尚未训练，返回空结果")
            return []
        
        features = self.prepare_features(data)
        
        if len(features) < 50:
            return []
        
        # 取最后50个时间步的数据
        recent_features = features[-50:].reshape(1, 50, -1)
        predictions = self.predict(recent_features)
        
        # 转换预测结果为PatternResult
        pattern_names = [
            'head_and_shoulders', 'inverse_head_shoulders', 'double_top', 'double_bottom',
            'triangle_ascending', 'triangle_descending', 'doji', 'hammer', 'engulfing_bullish', 'engulfing_bearish'
        ]
        
        results = []
        for i, prob in enumerate(predictions[0]):
            if prob > 0.3:  # 置信度阈值
                results.append(PatternResult(
                    pattern_type=pattern_names[i],
                    confidence=prob,
                    start_idx=len(data) - 50,
                    end_idx=len(data) - 1,
                    strength=prob,
                    direction='NEUTRAL',
                    metadata={'ml_probability': prob}
                ))
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)


class CrossMarketPatternAnalyzer:
    """跨市场模式关联分析器"""
    
    def __init__(self):
        self.correlation_window = 20
        self.market_data = {}
    
    def add_market_data(self, symbol: str, data: pd.DataFrame):
        """添加市场数据"""
        self.market_data[symbol] = data
    
    def analyze_correlations(self, symbol1: str, symbol2: str) -> Dict[str, float]:
        """分析两个市场的相关性"""
        if symbol1 not in self.market_data or symbol2 not in self.market_data:
            return {}
        
        data1 = self.market_data[symbol1]['close']
        data2 = self.market_data[symbol2]['close']
        
        # 计算不同时间窗口的相关性
        correlations = {}
        for window in [5, 10, 20, 50]:
            if len(data1) >= window and len(data2) >= window:
                corr = data1.tail(window).corr(data2.tail(window))
                correlations[f'correlation_{window}'] = corr
        
        return correlations
    
    def detect_cross_market_patterns(self, symbols: List[str]) -> List[PatternResult]:
        """检测跨市场模式"""
        results = []
        
        # 分析所有市场对的相关性
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlations = self.analyze_correlations(symbol1, symbol2)
                
                # 检测相关性突破
                for corr_key, corr_value in correlations.items():
                    if abs(corr_value) > 0.8:  # 强相关性
                        window = int(corr_key.split('_')[1])
                        
                        results.append(PatternResult(
                            pattern_type=f'cross_market_correlation_{symbol1}_{symbol2}',
                            confidence=abs(corr_value),
                            start_idx=len(self.market_data[symbol1]) - window,
                            end_idx=len(self.market_data[symbol1]) - 1,
                            strength=abs(corr_value),
                            direction='NEUTRAL',
                            metadata={
                                'symbol1': symbol1,
                                'symbol2': symbol2,
                                'correlation': corr_value,
                                'window': window
                            }
                        ))
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)


class PatternValidator:
    """模式有效性验证器"""
    
    def __init__(self):
        self.validation_rules = {
            'min_data_points': 10,
            'min_confidence': 0.6,
            'max_pattern_age': 100
        }
    
    def validate_pattern(self, pattern: PatternResult, data: pd.DataFrame) -> Tuple[bool, float]:
        """验证模式有效性"""
        # 检查数据点数量
        pattern_length = pattern.end_idx - pattern.start_idx + 1
        if pattern_length < self.validation_rules['min_data_points']:
            return False, 0.0
        
        # 检查置信度
        if pattern.confidence < self.validation_rules['min_confidence']:
            return False, 0.0
        
        # 检查模式年龄
        current_idx = len(data) - 1
        pattern_age = current_idx - pattern.end_idx
        if pattern_age > self.validation_rules['max_pattern_age']:
            return False, 0.0
        
        # 计算历史成功率（简化实现）
        success_rate = self._calculate_historical_success_rate(pattern, data)
        
        # 综合验证得分
        validation_score = (
            pattern.confidence * 0.4 +
            success_rate * 0.3 +
            (1 - pattern_age / self.validation_rules['max_pattern_age']) * 0.3
        )
        
        is_valid = validation_score > 0.6
        
        return is_valid, validation_score
    
    def _calculate_historical_success_rate(self, pattern: PatternResult, data: pd.DataFrame) -> float:
        """计算模式历史成功率"""
        # 简化实现：基于模式类型的平均成功率
        success_rates = {
            'head_and_shoulders': 0.65,
            'inverse_head_shoulders': 0.70,
            'double_top': 0.60,
            'double_bottom': 0.65,
            'triangle_ascending': 0.75,
            'triangle_descending': 0.70,
            'doji': 0.55,
            'hammer': 0.60,
            'engulfing_bullish': 0.68,
            'engulfing_bearish': 0.66
        }
        
        return success_rates.get(pattern.pattern_type, 0.5)
    
    def batch_validate(self, patterns: List[PatternResult], data: pd.DataFrame) -> List[Tuple[PatternResult, bool, float]]:
        """批量验证模式"""
        validated_patterns = []
        
        for pattern in patterns:
            is_valid, score = self.validate_pattern(pattern, data)
            validated_patterns.append((pattern, is_valid, score))
        
        return validated_patterns


class PatternSignalGenerator:
    """模式信号生成器"""
    
    def __init__(self):
        self.signal_rules = {
            'high_confidence_threshold': 0.8,
            'medium_confidence_threshold': 0.6,
            'position_sizing_rules': {
                'high_confidence': 0.1,  # 10%仓位
                'medium_confidence': 0.05  # 5%仓位
            }
        }
    
    def generate_signals(self, patterns: List[PatternResult], current_price: float, 
                        symbol: str) -> List[PatternSignal]:
        """基于模式生成交易信号"""
        signals = []
        
        for pattern in patterns:
            signal_type = self._determine_signal_type(pattern)
            confidence = pattern.confidence
            
            if signal_type != 'HOLD':
                # 生成信号
                signal = PatternSignal(
                    symbol=symbol,
                    pattern_type=pattern.pattern_type,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.now(),
                    target_price=pattern.target_price,
                    stop_loss=pattern.stop_loss,
                    metadata={
                        'pattern_strength': pattern.strength,
                        'pattern_direction': pattern.direction,
                        'pattern_start': pattern.start_idx,
                        'pattern_end': pattern.end_idx
                    }
                )
                
                signals.append(signal)
        
        # 按置信度排序
        return sorted(signals, key=lambda x: x.confidence, reverse=True)
    
    def _determine_signal_type(self, pattern: PatternResult) -> str:
        """确定信号类型"""
        if pattern.confidence >= self.signal_rules['high_confidence_threshold']:
            if pattern.direction == 'BULLISH':
                return 'BUY'
            elif pattern.direction == 'BEARISH':
                return 'SELL'
        
        elif pattern.confidence >= self.signal_rules['medium_confidence_threshold']:
            if pattern.direction == 'BULLISH':
                return 'BUY'
            elif pattern.direction == 'BEARISH':
                return 'SELL'
        
        return 'HOLD'


class PatternMonitor:
    """实时模式监控器"""
    
    def __init__(self, update_interval: int = 60):  # 默认60秒更新一次
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitored_symbols = []
        self.callbacks = []
        self.last_update = None
    
    def add_symbol(self, symbol: str):
        """添加监控标的"""
        if symbol not in self.monitored_symbols:
            self.monitored_symbols.append(symbol)
    
    def remove_symbol(self, symbol: str):
        """移除监控标的"""
        if symbol in self.monitored_symbols:
            self.monitored_symbols.remove(symbol)
    
    def add_callback(self, callback):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """开始监控"""
        self.is_monitoring = True
        self.last_update = datetime.now()
        print(f"开始监控模式识别，标的：{self.monitored_symbols}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        print("停止模式监控")
    
    def update(self, market_data: Dict[str, pd.DataFrame]):
        """更新监控数据"""
        if not self.is_monitoring:
            return
        
        current_time = datetime.now()
        
        # 检查是否需要更新
        if (self.last_update and 
            (current_time - self.last_update).seconds < self.update_interval):
            return
        
        # 对每个监控标的进行模式识别
        for symbol in self.monitored_symbols:
            if symbol in market_data:
                try:
                    # 这里应该调用PatternRecognizer进行识别
                    # 简化实现
                    patterns_detected = self._simulate_pattern_detection(market_data[symbol])
                    
                    # 触发回调
                    for callback in self.callbacks:
                        callback(symbol, patterns_detected, current_time)
                
                except Exception as e:
                    print(f"监控 {symbol} 时出错: {e}")
        
        self.last_update = current_time
    
    def _simulate_pattern_detection(self, data: pd.DataFrame) -> List[PatternResult]:
        """模拟模式检测（实际应用中应该调用真实的识别器）"""
        # 简化实现：随机生成一些模式用于演示
        import random
        
        if len(data) < 10:
            return []
        
        patterns = []
        pattern_types = ['doji', 'hammer', 'engulfing_bullish', 'engulfing_bearish']
        
        # 随机决定是否检测到模式
        if random.random() < 0.3:  # 30%概率检测到模式
            pattern_type = random.choice(pattern_types)
            confidence = random.uniform(0.6, 0.9)
            
            patterns.append(PatternResult(
                pattern_type=pattern_type,
                confidence=confidence,
                start_idx=len(data) - 5,
                end_idx=len(data) - 1,
                strength=random.uniform(0.5, 1.0),
                direction=random.choice(['BULLISH', 'BEARISH', 'NEUTRAL'])
            ))
        
        return patterns


class PatternBacktester:
    """模式历史回测器"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.trade_history = []
        self.performance_metrics = {}
    
    def backtest_patterns(self, patterns: List[PatternResult], data: pd.DataFrame, 
                         signals: List[PatternSignal]) -> Dict[str, Any]:
        """回测模式表现"""
        capital = self.initial_capital
        position = 0
        trades = []
        
        for signal in signals:
            if signal.signal_type in ['BUY', 'SELL'] and capital > 0:
                # 计算仓位大小
                position_size = capital * self._calculate_position_size(signal)
                
                # 执行交易
                if signal.signal_type == 'BUY':
                    shares = position_size / signal.price
                    capital -= position_size
                    position += shares
                    
                    trades.append({
                        'type': 'BUY',
                        'price': signal.price,
                        'shares': shares,
                        'value': position_size,
                        'timestamp': signal.timestamp,
                        'pattern': signal.pattern_type
                    })
                
                elif signal.signal_type == 'SELL' and position > 0:
                    sell_value = position * signal.price
                    capital += sell_value
                    
                    trades.append({
                        'type': 'SELL',
                        'price': signal.price,
                        'shares': position,
                        'value': sell_value,
                        'timestamp': signal.timestamp,
                        'pattern': signal.pattern_type
                    })
                    
                    position = 0
        
        # 计算最终收益
        final_value = capital + position * data['close'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 计算性能指标
        self.performance_metrics = self._calculate_performance_metrics(trades, data)
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'trades': trades,
            'performance_metrics': self.performance_metrics
        }
    
    def _calculate_position_size(self, signal: PatternSignal) -> float:
        """计算仓位大小"""
        if signal.confidence >= 0.8:
            return 0.1  # 10%
        elif signal.confidence >= 0.6:
            return 0.05  # 5%
        else:
            return 0.02  # 2%
    
    def _calculate_performance_metrics(self, trades: List[Dict], data: pd.DataFrame) -> Dict[str, float]:
        """计算性能指标"""
        if len(trades) < 2:
            return {'total_trades': 0, 'win_rate': 0, 'avg_return': 0}
        
        # 计算每笔交易的收益
        returns = []
        winning_trades = 0
        
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i+1]
                
                trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                returns.append(trade_return)
                
                if trade_return > 0:
                    winning_trades += 1
        
        win_rate = winning_trades / len(returns) if returns else 0
        avg_return = np.mean(returns) if returns else 0
        
        return {
            'total_trades': len(returns),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'max_return': max(returns) if returns else 0,
            'min_return': min(returns) if returns else 0,
            'return_std': np.std(returns) if len(returns) > 1 else 0
        }
    
    def generate_report(self) -> str:
        """生成回测报告"""
        if not self.performance_metrics:
            return "没有回测数据"
        
        report = f"""
模式识别回测报告
================
初始资金: ${self.initial_capital:,.2f}

交易统计:
- 总交易次数: {self.performance_metrics['total_trades']}
- 胜率: {self.performance_metrics['win_rate']:.2%}
- 平均收益: {self.performance_metrics['avg_return']:.2%}
- 最大收益: {self.performance_metrics['max_return']:.2%}
- 最大亏损: {self.performance_metrics['min_return']:.2%}
- 收益波动率: {self.performance_metrics['return_std']:.2%}

交易历史:
"""
        
        for trade in self.trade_history[-10:]:  # 显示最近10笔交易
            report += f"{trade['timestamp']}: {trade['type']} {trade['shares']:.2f} @ ${trade['price']:.2f}\n"
        
        return report


class PatternRecognizer:
    """主模式识别器"""
    
    def __init__(self, use_deep_learning: bool = True):
        self.price_recognizer = PricePatternRecognizer()
        self.candlestick_recognizer = CandlestickPatternRecognizer()
        self.volume_recognizer = VolumePatternRecognizer()
        self.validator = PatternValidator()
        self.signal_generator = PatternSignalGenerator()
        self.monitor = PatternMonitor()
        self.backtester = PatternBacktester()
        
        # 跨市场分析器
        self.cross_market_analyzer = CrossMarketPatternAnalyzer()
        
        # 深度学习模型
        if use_deep_learning and TENSORFLOW_AVAILABLE:
            self.deep_learning_model = DeepLearningPatternModel('cnn')
        else:
            self.deep_learning_model = None
        
        # 历史模式统计
        self.pattern_statistics = {}
        
    def recognize_patterns(self, data: pd.DataFrame, symbol: str = 'UNKNOWN',
                          min_confidence: float = 0.6) -> Dict[str, List[PatternResult]]:
        """识别所有类型的模式"""
        all_patterns = []
        
        # 价格模式识别
        price_patterns = self.price_recognizer.detect_patterns(data, min_confidence)
        all_patterns.extend(price_patterns)
        
        # K线形态识别
        candlestick_patterns = self.candlestick_recognizer.detect_patterns(data, min_confidence)
        all_patterns.extend(candlestick_patterns)
        
        # 成交量模式识别
        volume_patterns = self.volume_recognizer.detect_patterns(data, min_confidence)
        all_patterns.extend(volume_patterns)
        
        # 深度学习模型预测
        if self.deep_learning_model and self.deep_learning_model.is_trained:
            try:
                ml_patterns = self.deep_learning_model.predict_patterns(data)
                all_patterns.extend(ml_patterns)
            except Exception as e:
                print(f"深度学习模式预测失败: {e}")
        
        # 验证模式有效性
        validated_patterns = self.validator.batch_validate(all_patterns, data)
        valid_patterns = [pattern for pattern, is_valid, score in validated_patterns if is_valid]
        
        # 按类型分组
        pattern_groups = {
            'price_patterns': [p for p in valid_patterns if 'triangle' in p.pattern_type or 
                              'head' in p.pattern_type or 'double' in p.pattern_type or 
                              'flag' in p.pattern_type or 'pennant' in p.pattern_type or 
                              'rectangle' in p.pattern_type],
            'candlestick_patterns': [p for p in valid_patterns if any(candle in p.pattern_type 
                                   for candle in ['doji', 'hammer', 'shooting', 'engulfing', 
                                                 'star', 'soldiers', 'crows', 'spinning'])],
            'volume_patterns': [p for p in valid_patterns if 'volume' in p.pattern_type],
            'ml_patterns': [p for p in valid_patterns if 'ml_probability' in (p.metadata or {})],
            'all_patterns': valid_patterns
        }
        
        # 更新统计信息
        self._update_pattern_statistics(valid_patterns, symbol)
        
        return pattern_groups
    
    def generate_signals(self, patterns: Dict[str, List[PatternResult]], 
                        current_price: float, symbol: str) -> List[PatternSignal]:
        """生成交易信号"""
        all_patterns = patterns.get('all_patterns', [])
        return self.signal_generator.generate_signals(all_patterns, current_price, symbol)
    
    def add_cross_market_data(self, symbol: str, data: pd.DataFrame):
        """添加跨市场数据"""
        self.cross_market_analyzer.add_market_data(symbol, data)
    
    def analyze_cross_market_patterns(self, symbols: List[str]) -> List[PatternResult]:
        """分析跨市场模式"""
        return self.cross_market_analyzer.detect_cross_market_patterns(symbols)
    
    def start_real_time_monitoring(self, symbols: List[str], callbacks: List = None):
        """启动实时监控"""
        for symbol in symbols:
            self.monitor.add_symbol(symbol)
        
        if callbacks:
            for callback in callbacks:
                self.monitor.add_callback(callback)
        
        self.monitor.start_monitoring()
    
    def stop_real_time_monitoring(self):
        """停止实时监控"""
        self.monitor.stop_monitoring()
    
    def backtest_pattern_performance(self, data: pd.DataFrame, 
                                    patterns: Dict[str, List[PatternResult]]) -> Dict[str, Any]:
        """回测模式表现"""
        all_patterns = patterns.get('all_patterns', [])
        signals = self.generate_signals(patterns, data['close'].iloc[-1], 'BACKTEST')
        
        return self.backtester.backtest_patterns(all_patterns, data, signals)
    
    def get_pattern_statistics(self, symbol: str = None) -> Dict[str, Any]:
        """获取模式统计信息"""
        if symbol and symbol in self.pattern_statistics:
            return self.pattern_statistics[symbol]
        elif not symbol:
            return self.pattern_statistics
        else:
            return {}
    
    def _update_pattern_statistics(self, patterns: List[PatternResult], symbol: str):
        """更新模式统计信息"""
        if symbol not in self.pattern_statistics:
            self.pattern_statistics[symbol] = {
                'total_patterns': 0,
                'pattern_types': {},
                'avg_confidence': 0,
                'success_rate': 0
            }
        
        stats = self.pattern_statistics[symbol]
        stats['total_patterns'] += len(patterns)
        
        # 统计各类型模式数量
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in stats['pattern_types']:
                stats['pattern_types'][pattern_type] = 0
            stats['pattern_types'][pattern_type] += 1
        
        # 更新平均置信度
        if patterns:
            total_confidence = sum(p.confidence for p in patterns)
            stats['avg_confidence'] = total_confidence / len(patterns)
    
    def visualize_patterns(self, data: pd.DataFrame, patterns: List[PatternResult], 
                          save_path: str = None):
        """可视化模式识别结果"""
        if not PLOTTING_AVAILABLE:
            print("可视化库不可用")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 价格图和模式标记
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['close'], label='收盘价', alpha=0.7)
        
        # 标记模式
        for pattern in patterns:
            start_idx = pattern.start_idx
            end_idx = pattern.end_idx
            ax1.axvspan(start_idx, end_idx, alpha=0.3, 
                       color='green' if pattern.direction == 'BULLISH' else 
                             'red' if pattern.direction == 'BEARISH' else 'gray')
        
        ax1.set_title('价格模式和K线')
        ax1.legend()
        
        # 成交量图
        ax2 = axes[0, 1]
        ax2.bar(data.index, data['volume'], alpha=0.7)
        ax2.set_title('成交量')
        
        # 模式统计
        ax3 = axes[1, 0]
        if patterns:
            pattern_types = [p.pattern_type for p in patterns]
            pattern_counts = pd.Series(pattern_types).value_counts()
            ax3.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%')
        ax3.set_title('模式类型分布')
        
        # 置信度分布
        ax4 = axes[1, 1]
        if patterns:
            confidences = [p.confidence for p in patterns]
            ax4.hist(confidences, bins=10, alpha=0.7, edgecolor='black')
        ax4.set_title('模式置信度分布')
        ax4.set_xlabel('置信度')
        ax4.set_ylabel('频次')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_pattern_report(self, patterns: List[PatternResult], 
                             output_file: str = 'pattern_report.txt'):
        """导出模式报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("模式识别报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"识别到的模式总数: {len(patterns)}\n\n")
            
            # 按置信度排序
            sorted_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
            
            for i, pattern in enumerate(sorted_patterns, 1):
                f.write(f"模式 {i}: {pattern.pattern_type}\n")
                f.write(f"  置信度: {pattern.confidence:.2%}\n")
                f.write(f"  方向: {pattern.direction}\n")
                f.write(f"  强度: {pattern.strength:.2f}\n")
                f.write(f"  时间范围: {pattern.start_idx} - {pattern.end_idx}\n")
                
                if pattern.target_price:
                    f.write(f"  目标价位: {pattern.target_price:.2f}\n")
                if pattern.stop_loss:
                    f.write(f"  止损价位: {pattern.stop_loss:.2f}\n")
                
                if pattern.metadata:
                    f.write(f"  附加信息: {pattern.metadata}\n")
                
                f.write("\n" + "-" * 30 + "\n\n")
            
            # 统计信息
            if patterns:
                pattern_types = {}
                total_confidence = 0
                
                for pattern in patterns:
                    pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
                    total_confidence += pattern.confidence
                
                f.write("模式统计:\n")
                f.write(f"平均置信度: {total_confidence / len(patterns):.2%}\n")
                f.write("模式类型分布:\n")
                
                for pattern_type, count in pattern_types.items():
                    f.write(f"  {pattern_type}: {count} 次\n")


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # 生成模拟股价数据
    price_base = 100
    price_changes = np.random.normal(0, 0.02, 100)  # 2%日波动
    prices = [price_base]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 1, 100)
    })
    
    # 确保OHLC逻辑正确
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    # 创建模式识别器
    recognizer = PatternRecognizer(use_deep_learning=False)
    
    # 识别模式
    print("正在识别模式...")
    patterns = recognizer.recognize_patterns(data, 'TEST_SYMBOL')
    
    # 显示结果
    print(f"\n识别结果:")
    print(f"价格模式: {len(patterns['price_patterns'])} 个")
    print(f"K线形态: {len(patterns['candlestick_patterns'])} 个")
    print(f"成交量模式: {len(patterns['volume_patterns'])} 个")
    print(f"总模式数: {len(patterns['all_patterns'])} 个")
    
    # 生成交易信号
    if patterns['all_patterns']:
        signals = recognizer.generate_signals(patterns, data['close'].iloc[-1], 'TEST_SYMBOL')
        print(f"\n生成的交易信号: {len(signals)} 个")
        
        for signal in signals[:5]:  # 显示前5个信号
            print(f"信号: {signal.signal_type} {signal.pattern_type} "
                  f"(置信度: {signal.confidence:.2%})")
    
    # 回测表现
    print("\n正在回测模式表现...")
    backtest_result = recognizer.backtest_pattern_performance(data, patterns)
    print(f"回测结果: 总收益 {backtest_result['total_return']:.2%}")
    
    # 导出报告
    recognizer.export_pattern_report(patterns['all_patterns'], 'test_pattern_report.txt')
    print("\n模式报告已导出到 test_pattern_report.txt")
    
    # 可视化（如果可用）
    if PLOTTING_AVAILABLE:
        try:
            recognizer.visualize_patterns(data, patterns['all_patterns'], 'pattern_visualization.png')
            print("模式可视化图已保存到 pattern_visualization.png")
        except Exception as e:
            print(f"可视化失败: {e}")
    
    print("\n模式识别器测试完成!")