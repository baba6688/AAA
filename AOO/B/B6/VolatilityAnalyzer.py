"""
波动率分析器 - B6模块
===================

高级波动率分析系统，提供全面的波动率建模、预测和分析功能。

主要功能：
1. 历史波动率计算和建模
2. 隐含波动率分析
3. 波动率预测和建模（GARCH、EWMA等）
4. 波动率微笑和偏斜分析
5. 波动率套利机会识别
6. 波动率风险管理和对冲
7. 波动率策略信号生成


创建时间: 2025-11-05
版本: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, interpolate
from scipy.stats import norm, t, jarque_bera
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
# 尝试导入arch库，如果没有则使用fallback
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    arch_model = None
    ARCH_AVAILABLE = False
    
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VolatilityModel:
    """波动率模型配置"""
    name: str
    type: str  # 'GARCH', 'EWMA', 'HISTORICAL', 'IMPLIED'
    parameters: Dict[str, float]
    aic: float
    bic: float
    log_likelihood: float
    forecast: np.ndarray
    confidence_interval: Tuple[np.ndarray, np.ndarray]


@dataclass
class VolatilitySignal:
    """波动率交易信号"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY_VOLATILITY', 'SELL_VOLATILITY', 'HEDGE', 'ARBITRAGE'
    strength: float  # 0-1
    confidence: float  # 0-1
    target_volatility: float
    current_volatility: float
    expected_return: float
    risk_score: float
    recommended_action: str
    metadata: Dict[str, Any]


class BlackScholesCalculator:
    """Black-Scholes期权定价计算器"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """计算d1参数"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """计算d2参数"""
        return BlackScholesCalculator.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """计算看涨期权价格"""
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """计算看跌期权价格"""
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """计算Vega（对波动率的敏感性）"""
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1)


class ImpliedVolatilityCalculator:
    """隐含波动率计算器"""
    
    def __init__(self, max_iterations=100, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def calculate_implied_volatility(self, option_price, S, K, T, r, option_type='call'):
        """
        使用牛顿法计算隐含波动率
        
        参数:
        - option_price: 期权价格
        - S: 标的资产价格
        - K: 行权价
        - T: 到期时间（年）
        - r: 无风险利率
        - option_type: 期权类型 ('call' 或 'put')
        
        返回:
        - 隐含波动率
        """
        # 初始猜测
        sigma = 0.2
        
        for i in range(self.max_iterations):
            if option_type == 'call':
                price = BlackScholesCalculator.call_price(S, K, T, r, sigma)
                vega = BlackScholesCalculator.vega(S, K, T, r, sigma)
            else:
                price = BlackScholesCalculator.put_price(S, K, T, r, sigma)
                vega = BlackScholesCalculator.vega(S, K, T, r, sigma)
            
            price_diff = price - option_price
            
            if abs(price_diff) < self.tolerance:
                return sigma
            
            if vega == 0:
                break
            
            sigma = sigma - price_diff / vega
            
            # 确保sigma在合理范围内
            sigma = max(0.001, min(sigma, 5.0))
        
        return sigma if sigma > 0.001 else np.nan


class GARCHModel:
    """GARCH波动率模型"""
    
    def __init__(self, p=1, q=1, distribution='normal'):
        self.p = p
        self.q = q
        self.distribution = distribution
        self.model = None
        self.fitted_model = None
        self.residuals = None
        self.conditional_volatility = None
    
    def fit(self, returns):
        """拟合GARCH模型"""
        try:
            if not ARCH_AVAILABLE:
                # Fallback实现：使用简单的EWMA波动率
                logger.warning("arch库不可用，使用EWMA作为GARCH模型的fallback实现")
                self.fitted_model = self._ewma_fallback(returns)
                self.conditional_volatility = self.fitted_model['volatility']
                return self.fitted_model
                
            # 创建GARCH模型
            self.model = arch_model(
                returns * 100,  # 转换为百分比
                vol='GARCH',
                p=self.p,
                q=self.q,
                dist=self.distribution
            )
            
            # 拟合模型
            self.fitted_model = self.model.fit(disp='off')
            
            # 获取条件波动率
            self.conditional_volatility = self.fitted_model.conditional_volatility / 100
            
            # 获取残差
            self.residuals = self.fitted_model.resid / 100
            
            return self.fitted_model
            
        except Exception as e:
            logger.error(f"GARCH模型拟合失败: {e}")
            return None
    
    def forecast(self, horizon=1):
        """预测未来波动率"""
        if self.fitted_model is None:
            raise ValueError("模型未拟合")
        
        try:
            forecast = self.fitted_model.forecast(horizon=horizon, method='simulation')
            forecast_volatility = np.sqrt(forecast.variance.values[-1, :]) / 100
            
            return forecast_volatility
            
        except Exception as e:
            logger.error(f"GARCH模型预测失败: {e}")
            return None
    
    def get_model_info(self):
        """获取模型信息"""
        if self.fitted_model is None:
            return None
        
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.loglikelihood,
            'parameters': self.fitted_model.params.to_dict()
        }

    def _ewma_fallback(self, returns):
        """使用EWMA作为GARCH模型的fallback实现"""
        # 使用简单的方法计算波动率
        returns = np.array(returns)
        
        # 计算EWMA波动率
        lambda_param = 0.94
        volatility = []
        
        if len(returns) > 0:
            # 初始化：使用前几个观察值的均值作为初始波动率
            volatility.append(np.var(returns[:10] if len(returns) > 10 else returns))
            
            # 递归计算EWMA波动率
            for i in range(1, len(returns)):
                var_prev = volatility[-1]
                var_current = lambda_param * var_prev + (1 - lambda_param) * returns[i-1]**2
                volatility.append(var_current)
        
        volatility = np.array(volatility)
        forecast_volatility = np.sqrt(volatility[-1])
        
        return {
            'name': 'EWMA Fallback',
            'aic': np.inf,  # Fallback模型没有AIC
            'forecast': forecast_volatility,
            'volatility': volatility,
            'log_likelihood': np.nan
        }


class EWMA:
    """指数加权移动平均波动率模型"""
    
    def __init__(self, lambda_param=0.94):
        self.lambda_param = lambda_param
        self.volatility_history = []
    
    def calculate_volatility(self, returns, window=252):
        """计算EWMA波动率"""
        if len(returns) < 2:
            return np.array([])
        
        # 计算日收益率平方
        squared_returns = returns ** 2
        
        # 初始化
        ewma_vol = np.zeros(len(returns))
        ewma_vol[0] = squared_returns[0]
        
        # 计算EWMA
        for i in range(1, len(returns)):
            ewma_vol[i] = (self.lambda_param * ewma_vol[i-1] + 
                          (1 - self.lambda_param) * squared_returns[i-1])
        
        # 取平方根得到波动率
        ewma_vol = np.sqrt(ewma_vol)
        
        return ewma_vol
    
    def forecast(self, returns, horizon=1):
        """预测未来波动率"""
        if len(returns) == 0:
            return np.array([])
        
        # 计算当前EWMA波动率
        current_vol = self.calculate_volatility(returns)[-1]
        
        # EWMA预测假设未来波动率等于当前波动率
        forecast = np.full(horizon, current_vol)
        
        return forecast


class HistoricalVolatility:
    """历史波动率计算"""
    
    @staticmethod
    def simple_volatility(returns, window=252):
        """简单历史波动率"""
        if len(returns) < window:
            window = len(returns)
        
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    @staticmethod
    def realized_volatility(returns, window=252, intraday_data=None):
        """已实现波动率"""
        if intraday_data is not None:
            # 如果有日内数据，计算已实现波动率
            squared_returns = intraday_data['returns'] ** 2
            realized_vol = np.sqrt(squared_returns.rolling(window=window).sum() * 252)
            return realized_vol
        else:
            # 使用日数据计算
            return HistoricalVolatility.simple_volatility(returns, window)
    
    @staticmethod
    def parkinson_volatility(high, low, window=252):
        """Parkinson极值波动率估计器"""
        if len(high) != len(low):
            raise ValueError("高价和低价数据长度不匹配")
        
        log_hl = np.log(high / low) ** 2
        parkinson_vol = np.sqrt((log_hl.rolling(window=window).mean() * 252) / (4 * np.log(2)))
        return parkinson_vol
    
    @staticmethod
    def yang_zhang_volatility(open_price, high, low, close, window=252):
        """Yang-Zhang波动率估计器"""
        # 计算各种收益率
        log_ho = np.log(high / open_price)
        log_lo = np.log(low / open_price)
        log_co = np.log(close / open_price)
        log_oc = np.log(open_price / close.shift(1))
        log_cc = np.log(close / close.shift(1))
        
        # 计算各个组件
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        ro = log_co + 0.5 * (log_ho + log_lo)
        ro_sq = ro ** 2
        
        # 计算Yang-Zhang波动率
        yang_zhang_vol = np.sqrt(
            ro_sq.rolling(window=window).mean() * 252 +
            (0.34 * rs.rolling(window=window).mean() * 252) +
            (0.17 * log_oc.rolling(window=window).var() * 252)
        )
        
        return yang_zhang_vol


class VolatilitySmileAnalyzer:
    """波动率微笑和偏斜分析"""
    
    def __init__(self):
        self.implied_vol_calc = ImpliedVolatilityCalculator()
    
    def analyze_volatility_smile(self, option_chain_data):
        """
        分析波动率微笑
        
        参数:
        - option_chain_data: 期权链数据，包含行权价、到期时间和期权价格
        
        返回:
        - 波动率微笑分析结果
        """
        results = {
            'atm_volatility': None,
            'smile_curve': None,
            'skewness': None,
            'kurtosis': None,
            'volatility_surface': None
        }
        
        try:
            # 按到期时间分组
            for expiry, group in option_chain_data.groupby('expiry'):
                # 计算每个行权价的隐含波动率
                ivs = []
                strikes = []
                
                for _, row in group.iterrows():
                    try:
                        iv = self.implied_vol_calc.calculate_implied_volatility(
                            option_price=row['option_price'],
                            S=row['underlying_price'],
                            K=row['strike'],
                            T=row['time_to_expiry'],
                            r=row['risk_free_rate'],
                            option_type=row['option_type']
                        )
                        
                        if not np.isnan(iv):
                            ivs.append(iv)
                            strikes.append(row['strike'])
                    except:
                        continue
                
                if len(ivs) > 3:
                    # 拟合二次曲线（波动率微笑）
                    strikes = np.array(strikes)
                    ivs = np.array(ivs)
                    
                    # 归一化行权价
                    atm_price = np.median(strikes)
                    normalized_strikes = (strikes - atm_price) / atm_price
                    
                    # 二次拟合
                    coeffs = np.polyfit(normalized_strikes, ivs, 2)
                    smile_curve = np.poly1d(coeffs)
                    
                    # 计算偏斜和峰度
                    skewness = stats.skew(ivs)
                    kurtosis = stats.kurtosis(ivs)
                    
                    results[f'expiry_{expiry}'] = {
                        'atm_volatility': smile_curve(0),
                        'smile_curve': smile_curve,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'strikes': strikes,
                        'implied_vols': ivs
                    }
        
        except Exception as e:
            logger.error(f"波动率微笑分析失败: {e}")
        
        return results
    
    def calculate_volatility_surface(self, option_chain_data):
        """计算波动率曲面"""
        surface_data = []
        
        try:
            for _, row in option_chain_data.iterrows():
                try:
                    iv = self.implied_vol_calc.calculate_implied_volatility(
                        option_price=row['option_price'],
                        S=row['underlying_price'],
                        K=row['strike'],
                        T=row['time_to_expiry'],
                        r=row['risk_free_rate'],
                        option_type=row['option_type']
                    )
                    
                    if not np.isnan(iv):
                        surface_data.append({
                            'strike': row['strike'],
                            'time_to_expiry': row['time_to_expiry'],
                            'implied_volatility': iv,
                            'moneyness': row['strike'] / row['underlying_price']
                        })
                except:
                    continue
        
        except Exception as e:
            logger.error(f"波动率曲面计算失败: {e}")
        
        return pd.DataFrame(surface_data)


class VolatilityArbitrage:
    """波动率套利分析"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 默认无风险利率
    
    def identify_calendar_spreads(self, option_chain_data):
        """识别日历价差套利机会"""
        opportunities = []
        
        try:
            # 按行权价分组
            for strike, group in option_chain_data.groupby('strike'):
                if len(group) < 2:
                    continue
                
                # 按到期时间排序
                group = group.sort_values('time_to_expiry')
                
                # 检查相邻到期日的套利机会
                for i in range(len(group) - 1):
                    near_term = group.iloc[i]
                    far_term = group.iloc[i + 1]
                    
                    # 计算时间价值差异
                    time_value_diff = far_term['time_to_expiry'] - near_term['time_to_expiry']
                    
                    if time_value_diff > 0:
                        # 简单的套利检查：远期期权时间价值不应为负
                        if far_term['option_price'] < near_term['option_price']:
                            opportunities.append({
                                'type': 'Calendar Spread',
                                'strike': strike,
                                'near_expiry': near_term['time_to_expiry'],
                                'far_expiry': far_term['time_to_expiry'],
                                'near_price': near_term['option_price'],
                                'far_price': far_term['option_price'],
                                'time_value_diff': time_value_diff,
                                'arbitrage_signal': 'Potential Calendar Spread'
                            })
        
        except Exception as e:
            logger.error(f"日历价差分析失败: {e}")
        
        return opportunities
    
    def identify_straddles_strangles(self, option_chain_data):
        """识别跨式和宽跨式套利机会"""
        opportunities = []
        
        try:
            # 按到期时间分组
            for expiry, group in option_chain_data.groupby('time_to_expiry'):
                calls = group[group['option_type'] == 'call']
                puts = group[group['option_type'] == 'put']
                
                if len(calls) == 0 or len(puts) == 0:
                    continue
                
                # 寻找接近平值的期权
                underlying_price = calls['underlying_price'].iloc[0]
                atm_call = calls.iloc[(calls['strike'] - underlying_price).abs().argsort()[:1]]
                atm_put = puts.iloc[(puts['strike'] - underlying_price).abs().argsort()[:1]]
                
                if len(atm_call) > 0 and len(atm_put) > 0:
                    straddle_price = atm_call['option_price'].iloc[0] + atm_put['option_price'].iloc[0]
                    
                    # 检查跨式套利（简单检查：跨式价格不应超过内在价值）
                    intrinsic_value = max(0, underlying_price - atm_call['strike'].iloc[0])
                    
                    if straddle_price < intrinsic_value:
                        opportunities.append({
                            'type': 'Straddle Arbitrage',
                            'expiry': expiry,
                            'strike': atm_call['strike'].iloc[0],
                            'call_price': atm_call['option_price'].iloc[0],
                            'put_price': atm_put['option_price'].iloc[0],
                            'straddle_price': straddle_price,
                            'intrinsic_value': intrinsic_value,
                            'arbitrage_signal': 'Potential Straddle Arbitrage'
                        })
        
        except Exception as e:
            logger.error(f"跨式套利分析失败: {e}")
        
        return opportunities
    
    def calculate_volatility_risk_premium(self, historical_vol, implied_vol):
        """计算波动率风险溢价"""
        if len(historical_vol) != len(implied_vol):
            raise ValueError("历史波动率和隐含波动率数据长度不匹配")
        
        # 计算风险溢价
        volatility_risk_premium = implied_vol - historical_vol
        
        # 统计分析
        premium_stats = {
            'mean_premium': np.mean(volatility_risk_premium),
            'std_premium': np.std(volatility_risk_premium),
            'skewness': stats.skew(volatility_risk_premium),
            'kurtosis': stats.kurtosis(volatility_risk_premium),
            'percentiles': {
                '5%': np.percentile(volatility_risk_premium, 5),
                '25%': np.percentile(volatility_risk_premium, 25),
                '50%': np.percentile(volatility_risk_premium, 50),
                '75%': np.percentile(volatility_risk_premium, 75),
                '95%': np.percentile(volatility_risk_premium, 95)
            }
        }
        
        return volatility_risk_premium, premium_stats


class VolatilityRiskManagement:
    """波动率风险管理"""
    
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.var_model = None
    
    def calculate_portfolio_volatility(self, returns, weights=None):
        """计算投资组合波动率"""
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        # 计算协方差矩阵
        cov_matrix = returns.cov() * 252  # 年化
        
        # 计算投资组合波动率
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        return portfolio_volatility
    
    def calculate_volatility_var(self, returns, confidence_level=None):
        """计算波动率VaR"""
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # 计算历史波动率
        volatility = returns.rolling(window=252).std() * np.sqrt(252)
        
        # 计算VaR
        var = volatility * stats.norm.ppf(1 - confidence_level)
        
        return var
    
    def calculate_expected_shortfall(self, returns, confidence_level=None):
        """计算期望损失（Expected Shortfall）"""
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # 计算VaR
        var = self.calculate_volatility_var(returns, confidence_level)
        
        # 计算期望损失
        es = returns[returns <= var].mean()
        
        return es
    
    def calculate_volatility_budget(self, portfolio_volatility, target_volatility):
        """计算波动率预算"""
        if portfolio_volatility == 0:
            return 1.0
        
        volatility_ratio = target_volatility / portfolio_volatility
        
        # 确保比例在合理范围内
        volatility_ratio = max(0.1, min(volatility_ratio, 2.0))
        
        return volatility_ratio
    
    def hedge_volatility_risk(self, current_vol, target_vol, hedge_ratio=None):
        """对冲波动率风险"""
        if hedge_ratio is None:
            hedge_ratio = (target_vol - current_vol) / current_vol
        
        # 计算对冲头寸
        hedge_position = {
            'hedge_ratio': hedge_ratio,
            'direction': 'long' if hedge_ratio > 0 else 'short',
            'size': abs(hedge_ratio),
            'instrument': 'volatility_swap'  # 简化假设
        }
        
        return hedge_position


class VolatilityStrategy:
    """波动率策略信号生成"""
    
    def __init__(self):
        self.signals = []
        self.performance_metrics = {}
    
    def generate_mean_reversion_signal(self, current_vol, historical_vol, threshold=1.5):
        """生成均值回归信号"""
        if len(historical_vol) < 30:
            return None
        
        # 计算历史波动率的统计特征
        mean_vol = historical_vol.mean()
        std_vol = historical_vol.std()
        
        # 计算z-score
        z_score = (current_vol - mean_vol) / std_vol
        
        # 生成信号
        if z_score > threshold:
            # 当前波动率过高，预期回归
            signal = VolatilitySignal(
                timestamp=datetime.now(),
                symbol='UNKNOWN',
                signal_type='SELL_VOLATILITY',
                strength=min(abs(z_score) / (threshold * 2), 1.0),
                confidence=0.7,
                target_volatility=mean_vol,
                current_volatility=current_vol,
                expected_return=-0.1,  # 负期望收益
                risk_score=0.6,
                recommended_action='做空波动率',
                metadata={'z_score': z_score, 'mean_vol': mean_vol}
            )
            return signal
        
        elif z_score < -threshold:
            # 当前波动率过低，预期上升
            signal = VolatilitySignal(
                timestamp=datetime.now(),
                symbol='UNKNOWN',
                signal_type='BUY_VOLATILITY',
                strength=min(abs(z_score) / (threshold * 2), 1.0),
                confidence=0.7,
                target_volatility=mean_vol,
                current_volatility=current_vol,
                expected_return=0.1,  # 正期望收益
                risk_score=0.6,
                recommended_action='做多波动率',
                metadata={'z_score': z_score, 'mean_vol': mean_vol}
            )
            return signal
        
        return None
    
    def generate_trend_signal(self, vol_forecast, current_vol, confidence_threshold=0.8):
        """生成趋势信号"""
        if len(vol_forecast) == 0:
            return None
        
        # 计算趋势强度
        trend_strength = (vol_forecast[0] - current_vol) / current_vol
        
        # 生成信号
        if trend_strength > 0.1 and len(vol_forecast) > 5:
            # 波动率上升趋势
            signal = VolatilitySignal(
                timestamp=datetime.now(),
                symbol='UNKNOWN',
                signal_type='BUY_VOLATILITY',
                strength=min(trend_strength * 5, 1.0),
                confidence=confidence_threshold,
                target_volatility=vol_forecast[0],
                current_volatility=current_vol,
                expected_return=0.15,
                risk_score=0.7,
                recommended_action='做多波动率',
                metadata={'trend_strength': trend_strength, 'forecast': vol_forecast[0]}
            )
            return signal
        
        elif trend_strength < -0.1:
            # 波动率下降趋势
            signal = VolatilitySignal(
                timestamp=datetime.now(),
                symbol='UNKNOWN',
                signal_type='SELL_VOLATILITY',
                strength=min(abs(trend_strength) * 5, 1.0),
                confidence=confidence_threshold,
                target_volatility=vol_forecast[0],
                current_volatility=current_vol,
                expected_return=-0.15,
                risk_score=0.7,
                recommended_action='做空波动率',
                metadata={'trend_strength': trend_strength, 'forecast': vol_forecast[0]}
            )
            return signal
        
        return None
    
    def generate_arbitrage_signal(self, implied_vol, historical_vol, threshold=0.05):
        """生成套利信号"""
        vol_diff = implied_vol - historical_vol
        
        if abs(vol_diff) > threshold:
            if vol_diff > 0:
                # 隐含波动率高于历史波动率，做空波动率
                signal = VolatilitySignal(
                    timestamp=datetime.now(),
                    symbol='UNKNOWN',
                    signal_type='ARBITRAGE',
                    strength=min(abs(vol_diff) / (threshold * 2), 1.0),
                    confidence=0.8,
                    target_volatility=historical_vol,
                    current_volatility=implied_vol,
                    expected_return=-vol_diff * 0.5,
                    risk_score=0.5,
                    recommended_action='做空隐含波动率',
                    metadata={'implied_vol': implied_vol, 'historical_vol': historical_vol}
                )
                return signal
            
            else:
                # 隐含波动率低于历史波动率，做多波动率
                signal = VolatilitySignal(
                    timestamp=datetime.now(),
                    symbol='UNKNOWN',
                    signal_type='ARBITRAGE',
                    strength=min(abs(vol_diff) / (threshold * 2), 1.0),
                    confidence=0.8,
                    target_volatility=historical_vol,
                    current_volatility=implied_vol,
                    expected_return=abs(vol_diff) * 0.5,
                    risk_score=0.5,
                    recommended_action='做多隐含波动率',
                    metadata={'implied_vol': implied_vol, 'historical_vol': historical_vol}
                )
                return signal
        
        return None
    
    def backtest_strategy(self, signals, returns, initial_capital=100000):
        """回测策略"""
        if len(signals) == 0:
            return {}
        
        portfolio_value = [initial_capital]
        positions = []
        cash = initial_capital
        
        for signal in signals:
            if signal.signal_type == 'BUY_VOLATILITY':
                # 做多波动率
                position_size = cash * signal.strength * 0.1  # 10%仓位
                positions.append({
                    'type': 'long_vol',
                    'size': position_size,
                    'entry_price': signal.current_volatility,
                    'timestamp': signal.timestamp
                })
                cash -= position_size
            
            elif signal.signal_type == 'SELL_VOLATILITY':
                # 做空波动率
                position_size = cash * signal.strength * 0.1
                positions.append({
                    'type': 'short_vol',
                    'size': position_size,
                    'entry_price': signal.current_volatility,
                    'timestamp': signal.timestamp
                })
                cash += position_size
        
        # 计算最终投资组合价值
        final_value = cash + sum([pos['size'] for pos in positions])
        
        # 计算性能指标
        total_return = (final_value - initial_capital) / initial_capital
        max_drawdown = self.calculate_max_drawdown(portfolio_value)
        sharpe_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        performance = {
            'total_return': total_return,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(positions),
            'win_rate': 0.6  # 简化计算
        }
        
        return performance
    
    def calculate_max_drawdown(self, portfolio_values):
        """计算最大回撤"""
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown


class VolatilityAnalyzer:
    """主要波动率分析器类"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.garch_model = GARCHModel()
        self.ewma_model = EWMA()
        self.historical_vol = HistoricalVolatility()
        self.smile_analyzer = VolatilitySmileAnalyzer()
        self.arbitrage_analyzer = VolatilityArbitrage()
        self.risk_manager = VolatilityRiskManagement()
        self.strategy_generator = VolatilityStrategy()
        
        self.real_time_data = {}
        self.volatility_models = {}
        self.alerts = []
        self.is_running = False
        
        # 线程锁
        self.lock = threading.Lock()
        
        logger.info("波动率分析器初始化完成")
    
    def load_market_data(self, data):
        """加载市场数据"""
        try:
            with self.lock:
                self.real_time_data.update(data)
                logger.info(f"加载了 {len(data)} 条市场数据")
        
        except Exception as e:
            logger.error(f"加载市场数据失败: {e}")
    
    def calculate_historical_volatility(self, symbol, window=252):
        """计算历史波动率"""
        try:
            if symbol not in self.real_time_data:
                logger.warning(f"未找到 {symbol} 的数据")
                return None
            
            data = self.real_time_data[symbol]
            returns = data['returns']
            
            # 多种历史波动率计算方法
            results = {
                'simple_volatility': self.historical_vol.simple_volatility(returns, window),
                'realized_volatility': self.historical_vol.realized_volatility(returns, window),
                'parkinson_volatility': self.historical_vol.parkinson_volatility(
                    data['high'], data['low'], window
                ) if 'high' in data and 'low' in data else None,
                'yang_zhang_volatility': self.historical_vol.yang_zhang_volatility(
                    data.get('open', data['close'].shift(1)),
                    data.get('high', data['close']),
                    data.get('low', data['close']),
                    data['close'],
                    window
                ) if all(key in data for key in ['open', 'high', 'low', 'close']) else None
            }
            
            return results
        
        except Exception as e:
            logger.error(f"计算历史波动率失败: {e}")
            return None
    
    def fit_garch_model(self, symbol, p=1, q=1):
        """拟合GARCH模型"""
        try:
            if symbol not in self.real_time_data:
                return None
            
            data = self.real_time_data[symbol]
            returns = data['returns'].dropna()
            
            # 创建并拟合GARCH模型
            garch = GARCHModel(p=p, q=q)
            fitted_model = garch.fit(returns)
            
            if fitted_model is not None:
                # 保存模型
                model_info = garch.get_model_info()
                forecast = garch.forecast(horizon=30)
                
                volatility_model = VolatilityModel(
                    name=f"GARCH({p},{q})",
                    type="GARCH",
                    parameters=model_info['parameters'],
                    aic=model_info['aic'],
                    bic=model_info['bic'],
                    log_likelihood=model_info['log_likelihood'],
                    forecast=forecast,
                    confidence_interval=(forecast * 0.8, forecast * 1.2)  # 简化置信区间
                )
                
                self.volatility_models[symbol] = volatility_model
                logger.info(f"GARCH模型拟合成功: {symbol}")
                
                return volatility_model
            
        except Exception as e:
            logger.error(f"GARCH模型拟合失败: {e}")
            return None
    
    def calculate_implied_volatility(self, option_data):
        """计算隐含波动率"""
        try:
            iv_calc = ImpliedVolatilityCalculator()
            results = []
            
            for option in option_data:
                iv = iv_calc.calculate_implied_volatility(
                    option_price=option['option_price'],
                    S=option['underlying_price'],
                    K=option['strike'],
                    T=option['time_to_expiry'],
                    r=option['risk_free_rate'],
                    option_type=option['option_type']
                )
                
                if not np.isnan(iv):
                    results.append({
                        'symbol': option['symbol'],
                        'strike': option['strike'],
                        'expiry': option['expiry'],
                        'option_type': option['option_type'],
                        'implied_volatility': iv,
                        'option_price': option['option_price']
                    })
            
            return pd.DataFrame(results)
        
        except Exception as e:
            logger.error(f"隐含波动率计算失败: {e}")
            return pd.DataFrame()
    
    def analyze_volatility_smile(self, option_chain_data):
        """分析波动率微笑"""
        try:
            return self.smile_analyzer.analyze_volatility_smile(option_chain_data)
        
        except Exception as e:
            logger.error(f"波动率微笑分析失败: {e}")
            return {}
    
    def identify_arbitrage_opportunities(self, option_chain_data):
        """识别套利机会"""
        try:
            opportunities = []
            
            # 日历价差
            calendar_spreads = self.arbitrage_analyzer.identify_calendar_spreads(option_chain_data)
            opportunities.extend(calendar_spreads)
            
            # 跨式套利
            straddles = self.arbitrage_analyzer.identify_straddles_strangles(option_chain_data)
            opportunities.extend(straddles)
            
            return opportunities
        
        except Exception as e:
            logger.error(f"套利机会识别失败: {e}")
            return []
    
    def generate_trading_signals(self, symbol):
        """生成交易信号"""
        try:
            signals = []
            
            # 获取历史波动率
            hist_vol_data = self.calculate_historical_volatility(symbol)
            if hist_vol_data is None:
                return signals
            
            current_vol = hist_vol_data['simple_volatility'].iloc[-1]
            historical_vol_series = hist_vol_data['simple_volatility'].dropna()
            
            # 均值回归信号
            mean_reversion_signal = self.strategy_generator.generate_mean_reversion_signal(
                current_vol, historical_vol_series
            )
            if mean_reversion_signal:
                mean_reversion_signal.symbol = symbol
                signals.append(mean_reversion_signal)
            
            # 趋势信号
            if symbol in self.volatility_models:
                forecast = self.volatility_models[symbol].forecast
                trend_signal = self.strategy_generator.generate_trend_signal(
                    forecast, current_vol
                )
                if trend_signal:
                    trend_signal.symbol = symbol
                    signals.append(trend_signal)
            
            return signals
        
        except Exception as e:
            logger.error(f"交易信号生成失败: {e}")
            return []
    
    def assess_risk(self, portfolio_data):
        """风险评估"""
        try:
            risk_metrics = {}
            
            for symbol, data in portfolio_data.items():
                returns = data['returns']
                
                # 计算风险指标
                volatility = returns.std() * np.sqrt(252)
                var_95 = self.risk_manager.calculate_volatility_var(returns, 0.95)
                expected_shortfall = self.risk_manager.calculate_expected_shortfall(returns, 0.95)
                
                risk_metrics[symbol] = {
                    'volatility': volatility,
                    'var_95': var_95.iloc[-1] if len(var_95) > 0 else 0,
                    'expected_shortfall': expected_shortfall.iloc[-1] if len(expected_shortfall) > 0 else 0,
                    'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                }
            
            return risk_metrics
        
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return {}
    
    def create_volatility_visualization(self, symbol, save_path=None):
        """创建波动率可视化图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{symbol} 波动率分析', fontsize=16, fontweight='bold')
            
            # 获取数据
            hist_vol_data = self.calculate_historical_volatility(symbol)
            if hist_vol_data is None:
                return
            
            # 1. 历史波动率对比
            ax1 = axes[0, 0]
            if hist_vol_data['simple_volatility'] is not None:
                ax1.plot(hist_vol_data['simple_volatility'].index, 
                        hist_vol_data['simple_volatility'], 
                        label='简单历史波动率', linewidth=2)
            if hist_vol_data['parkinson_volatility'] is not None:
                ax1.plot(hist_vol_data['parkinson_volatility'].index, 
                        hist_vol_data['parkinson_volatility'], 
                        label='Parkinson波动率', linewidth=2, alpha=0.7)
            ax1.set_title('历史波动率对比')
            ax1.set_ylabel('年化波动率')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. GARCH模型预测
            ax2 = axes[0, 1]
            if symbol in self.volatility_models:
                model = self.volatility_models[symbol]
                current_vol = hist_vol_data['simple_volatility'].iloc[-1]
                
                # 历史波动率
                ax2.plot(hist_vol_data['simple_volatility'].index[-60:], 
                        hist_vol_data['simple_volatility'].iloc[-60:], 
                        label='历史波动率', linewidth=2)
                
                # 预测波动率
                forecast_dates = pd.date_range(start=datetime.now(), periods=len(model.forecast), freq='D')
                ax2.plot(forecast_dates, model.forecast, 
                        label=f'{model.name}预测', linewidth=2, linestyle='--')
                
                # 置信区间
                lower_ci, upper_ci = model.confidence_interval
                ax2.fill_between(forecast_dates, lower_ci, upper_ci, 
                               alpha=0.3, label='置信区间')
            
            ax2.set_title('波动率预测')
            ax2.set_ylabel('年化波动率')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 波动率分布
            ax3 = axes[1, 0]
            if hist_vol_data['simple_volatility'] is not None:
                vol_data = hist_vol_data['simple_volatility'].dropna()
                ax3.hist(vol_data, bins=30, alpha=0.7, density=True, label='波动率分布')
                
                # 拟合正态分布
                mu, sigma = norm.fit(vol_data)
                x = np.linspace(vol_data.min(), vol_data.max(), 100)
                ax3.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='拟合正态分布')
            
            ax3.set_title('波动率分布')
            ax3.set_xlabel('年化波动率')
            ax3.set_ylabel('密度')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 波动率风险指标
            ax4 = axes[1, 1]
            risk_data = self.assess_risk({symbol: self.real_time_data[symbol]})
            if symbol in risk_data:
                metrics = ['volatility', 'var_95', 'expected_shortfall']
                values = [risk_data[symbol][metric] for metric in metrics]
                labels = ['波动率', 'VaR(95%)', '期望损失']
                
                bars = ax4.bar(labels, values, alpha=0.7, color=['blue', 'red', 'orange'])
                ax4.set_title('风险指标')
                ax4.set_ylabel('风险值')
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
            
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"图表已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"创建可视化图表失败: {e}")
    
    def start_real_time_monitoring(self, symbols, update_interval=60):
        """启动实时监控"""
        self.is_running = True
        
        def monitor_loop():
            while self.is_running:
                try:
                    for symbol in symbols:
                        # 更新波动率模型
                        self.fit_garch_model(symbol)
                        
                        # 生成交易信号
                        signals = self.generate_trading_signals(symbol)
                        
                        # 检查风险预警
                        risk_data = self.assess_risk({symbol: self.real_time_data[symbol]})
                        
                        # 触发预警
                        self.check_risk_alerts(symbol, risk_data)
                    
                    time.sleep(update_interval)
                
                except Exception as e:
                    logger.error(f"实时监控错误: {e}")
                    time.sleep(10)
        
        # 在后台线程中运行监控
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        logger.info(f"实时监控已启动，监控标的: {symbols}")
    
    def stop_real_time_monitoring(self):
        """停止实时监控"""
        self.is_running = False
        logger.info("实时监控已停止")
    
    def check_risk_alerts(self, symbol, risk_data):
        """检查风险预警"""
        try:
            if symbol not in risk_data:
                return
            
            metrics = risk_data[symbol]
            
            # 波动率预警
            if metrics['volatility'] > 0.5:  # 50%年化波动率
                alert = {
                    'type': 'HIGH_VOLATILITY',
                    'symbol': symbol,
                    'message': f'{symbol} 波动率过高: {metrics["volatility"]:.2%}',
                    'timestamp': datetime.now(),
                    'severity': 'HIGH'
                }
                self.alerts.append(alert)
                logger.warning(f"高波动率预警: {alert['message']}")
            
            # VaR预警
            if metrics['var_95'] < -0.1:  # 10%日损失
                alert = {
                    'type': 'HIGH_VAR',
                    'symbol': symbol,
                    'message': f'{symbol} VaR过高: {metrics["var_95"]:.2%}',
                    'timestamp': datetime.now(),
                    'severity': 'MEDIUM'
                }
                self.alerts.append(alert)
                logger.warning(f"高VaR预警: {alert['message']}")
        
        except Exception as e:
            logger.error(f"风险预警检查失败: {e}")
    
    def generate_report(self, symbols):
        """生成波动率分析报告"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'summary': {},
                'models': {},
                'signals': [],
                'risk_assessment': {},
                'alerts': self.alerts[-10:],  # 最近10个预警
                'recommendations': []
            }
            
            for symbol in symbols:
                # 历史波动率
                hist_vol = self.calculate_historical_volatility(symbol)
                if hist_vol is not None and hist_vol['simple_volatility'] is not None:
                    current_vol = hist_vol['simple_volatility'].iloc[-1]
                    report['summary'][symbol] = {
                        'current_volatility': current_vol,
                        'volatility_percentile': stats.percentileofscore(
                            hist_vol['simple_volatility'].dropna(), current_vol
                        )
                    }
                
                # 模型信息
                if symbol in self.volatility_models:
                    model = self.volatility_models[symbol]
                    report['models'][symbol] = {
                        'model_name': model.name,
                        'aic': model.aic,
                        'bic': model.bic,
                        'forecast': model.forecast.tolist()
                    }
                
                # 交易信号
                signals = self.generate_trading_signals(symbol)
                report['signals'].extend([{
                    'symbol': s.symbol,
                    'type': s.signal_type,
                    'strength': s.strength,
                    'confidence': s.confidence,
                    'action': s.recommended_action
                } for s in signals])
                
                # 风险评估
                risk_data = self.assess_risk({symbol: self.real_time_data[symbol]})
                if risk_data:
                    report['risk_assessment'][symbol] = risk_data[symbol]
            
            # 生成建议
            report['recommendations'] = self.generate_recommendations(report)
            
            return report
        
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            return {}
    
    def generate_recommendations(self, report):
        """生成投资建议"""
        recommendations = []
        
        try:
            # 基于波动率水平
            for symbol, data in report['summary'].items():
                current_vol = data['current_volatility']
                percentile = data['volatility_percentile']
                
                if percentile > 90:
                    recommendations.append({
                        'type': 'HIGH_VOLATILITY',
                        'symbol': symbol,
                        'action': '考虑做空波动率或增加对冲',
                        'reason': f'当前波动率位于历史{percentile:.0f}%分位数',
                        'priority': 'HIGH'
                    })
                elif percentile < 10:
                    recommendations.append({
                        'type': 'LOW_VOLATILITY',
                        'symbol': symbol,
                        'action': '考虑做多波动率',
                        'reason': f'当前波动率位于历史{percentile:.0f}%分位数',
                        'priority': 'MEDIUM'
                    })
            
            # 基于模型预测
            for symbol, model_data in report['models'].items():
                if len(model_data['forecast']) > 0:
                    forecast_vol = model_data['forecast'][0]
                    current_vol = report['summary'][symbol]['current_volatility']
                    
                    if forecast_vol > current_vol * 1.2:
                        recommendations.append({
                            'type': 'VOLATILITY_TREND',
                            'symbol': symbol,
                            'action': '预期波动率上升，考虑做多波动率',
                            'reason': f'模型预测波动率将从{current_vol:.2%}上升到{forecast_vol:.2%}',
                            'priority': 'MEDIUM'
                        })
        
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
        
        return recommendations
    
    def export_data(self, filepath):
        """导出分析数据"""
        try:
            export_data = {
                'volatility_models': {},
                'signals': [],
                'alerts': self.alerts,
                'real_time_data': self.real_time_data
            }
            
            # 导出模型数据
            for symbol, model in self.volatility_models.items():
                export_data['volatility_models'][symbol] = {
                    'name': model.name,
                    'type': model.type,
                    'parameters': model.parameters,
                    'aic': model.aic,
                    'bic': model.bic,
                    'forecast': model.forecast.tolist()
                }
            
            # 导出到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"数据已导出到: {filepath}")
        
        except Exception as e:
            logger.error(f"数据导出失败: {e}")


# 示例使用和测试函数
def create_sample_data():
    """创建示例数据用于测试"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # 模拟价格序列（几何布朗运动）
    mu = 0.0002  # 日收益率
    sigma = 0.02  # 日波动率
    
    returns = np.random.normal(mu, sigma, n_days)
    prices = [100]  # 初始价格
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # 去掉初始价格
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'returns': returns
    })
    
    # 添加高低价（简化模拟）
    data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
    data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    
    return data


def demo_volatility_analyzer():
    """波动率分析器演示"""
    print("=== 波动率分析器演示 ===")
    
    # 创建分析器
    analyzer = VolatilityAnalyzer()
    
    # 创建示例数据
    sample_data = create_sample_data()
    
    # 加载数据
    analyzer.load_market_data({'AAPL': sample_data})
    
    print("1. 计算历史波动率...")
    hist_vol = analyzer.calculate_historical_volatility('AAPL')
    if hist_vol and hist_vol['simple_volatility'] is not None:
        current_vol = hist_vol['simple_volatility'].iloc[-1]
        print(f"   当前历史波动率: {current_vol:.2%}")
    
    print("2. 拟合GARCH模型...")
    garch_model = analyzer.fit_garch_model('AAPL', p=1, q=1)
    if garch_model:
        print(f"   模型: {garch_model.name}")
        print(f"   AIC: {garch_model.aic:.2f}")
        print(f"   30天预测波动率: {garch_model.forecast[0]:.2%}")
    
    print("3. 生成交易信号...")
    signals = analyzer.generate_trading_signals('AAPL')
    print(f"   生成信号数量: {len(signals)}")
    for signal in signals[:3]:  # 显示前3个信号
        print(f"   - {signal.signal_type}: {signal.recommended_action} (强度: {signal.strength:.2f})")
    
    print("4. 风险评估...")
    risk_data = analyzer.assess_risk({'AAPL': sample_data})
    if 'AAPL' in risk_data:
        risk = risk_data['AAPL']
        print(f"   年化波动率: {risk['volatility']:.2%}")
        print(f"   VaR(95%): {risk['var_95']:.2%}")
        print(f"   夏普比率: {risk['sharpe_ratio']:.2f}")
    
    print("5. 生成分析报告...")
    report = analyzer.generate_report(['AAPL'])
    print(f"   报告生成完成，包含 {len(report.get('recommendations', []))} 条建议")
    
    print("6. 创建可视化图表...")
    analyzer.create_volatility_visualization('AAPL', 'volatility_analysis.png')
    
    print("演示完成！")


if __name__ == "__main__":
    # 运行演示
    demo_volatility_analyzer()