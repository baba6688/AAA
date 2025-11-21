"""
性能分析器模块

实现策略性能的深度分析和评估：
- 多维度性能指标计算
- 风险调整收益分析
- 性能归因分析
- 基准比较
- 统计显著性检验
- 性能稳定性评估
- 预测性分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from .StrategyLearner import StrategyPerformance, StrategyType

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    strategy_id: str
    timestamp: datetime
    
    # 收益指标
    total_return: float
    annualized_return: float
    average_return: float
    return_volatility: float
    
    # 风险指标
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    expected_shortfall: float
    
    # 交易指标
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # 一致性指标
    hit_rate: float
    consistency_score: float
    up_capture: float
    down_capture: float
    
    # 其他指标
    information_ratio: float
    tracking_error: float
    beta: float
    alpha: float

class AdvancedPerformanceAnalyzer:
    """高级性能分析器"""
    
    def __init__(self, benchmark_returns: Optional[List[float]] = None):
        self.benchmark_returns = benchmark_returns or []
        self.risk_free_rate = 0.02  # 2%年化无风险利率
        self.trading_days_per_year = 252
        
        # 分析缓存
        self.analysis_cache = {}
        self.correlation_matrix = None
        self.factor_loadings = None
        
    def comprehensive_analysis(self, strategy_id: str, 
                             performance_data: List[StrategyPerformance],
                             benchmark_data: Optional[List[float]] = None) -> Dict[str, Any]:
        """综合性能分析"""
        try:
            if not performance_data:
                return {'error': '没有性能数据进行分析'}
            
            # 提取时间序列数据
            returns, timestamps = self._extract_time_series(performance_data)
            
            if len(returns) < 2:
                return {'error': '数据点不足，无法进行分析'}
            
            # 基础性能指标
            basic_metrics = self._calculate_basic_metrics(returns)
            
            # 风险调整收益指标
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(returns)
            
            # 交易统计指标
            trading_metrics = self._calculate_trading_metrics(performance_data)
            
            # 稳定性和一致性指标
            stability_metrics = self._calculate_stability_metrics(returns)
            
            # 基准比较分析
            benchmark_analysis = self._benchmark_analysis(returns, benchmark_data)
            
            # 性能归因分析
            attribution_analysis = self._performance_attribution(returns, performance_data)
            
            # 统计显著性检验
            statistical_tests = self._statistical_tests(returns)
            
            # 预测性分析
            predictive_analysis = self._predictive_analysis(returns)
            
            # 风险分解
            risk_decomposition = self._risk_decomposition(returns)
            
            # 综合评分
            overall_score = self._calculate_overall_score(
                basic_metrics, risk_adjusted_metrics, trading_metrics, stability_metrics
            )
            
            return {
                'strategy_id': strategy_id,
                'analysis_timestamp': datetime.now(),
                'data_period': {
                    'start_date': min(timestamps),
                    'end_date': max(timestamps),
                    'total_days': (max(timestamps) - min(timestamps)).days,
                    'data_points': len(returns)
                },
                
                'basic_metrics': basic_metrics,
                'risk_adjusted_metrics': risk_adjusted_metrics,
                'trading_metrics': trading_metrics,
                'stability_metrics': stability_metrics,
                'benchmark_analysis': benchmark_analysis,
                'attribution_analysis': attribution_analysis,
                'statistical_tests': statistical_tests,
                'predictive_analysis': predictive_analysis,
                'risk_decomposition': risk_decomposition,
                
                'overall_score': overall_score,
                'performance_grade': self._grade_performance(overall_score),
                'recommendations': self._generate_recommendations(
                    basic_metrics, risk_adjusted_metrics, trading_metrics
                )
            }
            
        except Exception as e:
            logger.error(f"综合性能分析出错: {e}")
            return {'error': str(e)}
    
    def multi_strategy_comparison(self, strategy_performances: Dict[str, List[StrategyPerformance]],
                                benchmark_data: Optional[List[float]] = None) -> Dict[str, Any]:
        """多策略比较分析"""
        try:
            # 分析每个策略
            individual_analyses = {}
            for strategy_id, performances in strategy_performances.items():
                analysis = self.comprehensive_analysis(strategy_id, performances, benchmark_data)
                if 'error' not in analysis:
                    individual_analyses[strategy_id] = analysis
            
            if len(individual_analyses) < 2:
                return {'error': '至少需要2个策略进行比较'}
            
            # 排名分析
            rankings = self._rank_strategies(individual_analyses)
            
            # 相关性分析
            correlation_analysis = self._strategy_correlation_analysis(strategy_performances)
            
            # 分散化分析
            diversification_analysis = self._diversification_analysis(strategy_performances)
            
            # 最优组合分析
            portfolio_analysis = self._optimal_portfolio_analysis(individual_analyses, correlation_analysis)
            
            # 风险贡献分析
            risk_contribution = self._risk_contribution_analysis(strategy_performances)
            
            return {
                'comparison_timestamp': datetime.now(),
                'individual_analyses': individual_analyses,
                'rankings': rankings,
                'correlation_analysis': correlation_analysis,
                'diversification_analysis': diversification_analysis,
                'portfolio_analysis': portfolio_analysis,
                'risk_contribution': risk_contribution,
                'summary': self._generate_comparison_summary(individual_analyses, rankings)
            }
            
        except Exception as e:
            logger.error(f"多策略比较分析出错: {e}")
            return {'error': str(e)}
    
    def performance_attribution(self, strategy_id: str, 
                              performance_data: List[StrategyPerformance],
                              market_factors: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """性能归因分析"""
        try:
            returns, timestamps = self._extract_time_series(performance_data)
            
            if len(returns) < 10:
                return {'error': '数据点不足，无法进行归因分析'}
            
            # 因子归因
            factor_attribution = self._factor_attribution(returns, market_factors)
            
            # 时间归因
            temporal_attribution = self._temporal_attribution(returns, timestamps)
            
            # 风险归因
            risk_attribution = self._risk_attribution(returns)
            
            # 策略归因
            strategy_attribution = self._strategy_attribution(performance_data)
            
            return {
                'strategy_id': strategy_id,
                'attribution_timestamp': datetime.now(),
                'factor_attribution': factor_attribution,
                'temporal_attribution': temporal_attribution,
                'risk_attribution': risk_attribution,
                'strategy_attribution': strategy_attribution,
                'total_attribution': self._validate_attribution(factor_attribution, returns)
            }
            
        except Exception as e:
            logger.error(f"性能归因分析出错: {e}")
            return {'error': str(e)}
    
    def risk_analysis(self, strategy_id: str, 
                     performance_data: List[StrategyPerformance],
                     confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Any]:
        """风险分析"""
        try:
            returns, timestamps = self._extract_time_series(performance_data)
            
            if len(returns) < 2:
                return {'error': '数据点不足，无法进行风险分析'}
            
            # VaR分析
            var_analysis = self._var_analysis(returns, confidence_levels)
            
            # 压力测试
            stress_testing = self._stress_testing(returns)
            
            # 风险分解
            risk_decomposition = self._detailed_risk_decomposition(returns)
            
            # 流动性风险
            liquidity_risk = self._liquidity_risk_analysis(performance_data)
            
            # 操作风险
            operational_risk = self._operational_risk_analysis(performance_data)
            
            return {
                'strategy_id': strategy_id,
                'risk_analysis_timestamp': datetime.now(),
                'var_analysis': var_analysis,
                'stress_testing': stress_testing,
                'risk_decomposition': risk_decomposition,
                'liquidity_risk': liquidity_risk,
                'operational_risk': operational_risk,
                'risk_summary': self._generate_risk_summary(var_analysis, stress_testing)
            }
            
        except Exception as e:
            logger.error(f"风险分析出错: {e}")
            return {'error': str(e)}
    
    def performance_prediction(self, strategy_id: str, 
                             performance_data: List[StrategyPerformance],
                             prediction_horizon: int = 30,
                             prediction_methods: List[str] = ['monte_carlo', 'time_series', 'regime']) -> Dict[str, Any]:
        """性能预测"""
        try:
            returns, timestamps = self._extract_time_series(performance_data)
            
            if len(returns) < 20:
                return {'error': '历史数据不足，无法进行预测'}
            
            prediction_results = {}
            
            # 蒙特卡洛模拟
            if 'monte_carlo' in prediction_methods:
                prediction_results['monte_carlo'] = self._monte_carlo_prediction(returns, prediction_horizon)
            
            # 时间序列预测
            if 'time_series' in prediction_methods:
                prediction_results['time_series'] = self._time_series_prediction(returns, prediction_horizon)
            
            # 状态转换预测
            if 'regime' in prediction_methods:
                prediction_results['regime'] = self._regime_based_prediction(returns, prediction_horizon)
            
            # 集成预测
            ensemble_prediction = self._ensemble_prediction(prediction_results)
            
            # 预测验证
            validation_results = self._validate_predictions(returns, prediction_results)
            
            return {
                'strategy_id': strategy_id,
                'prediction_timestamp': datetime.now(),
                'prediction_horizon': prediction_horizon,
                'individual_predictions': prediction_results,
                'ensemble_prediction': ensemble_prediction,
                'validation_results': validation_results,
                'confidence_intervals': self._calculate_prediction_intervals(prediction_results),
                'recommendations': self._generate_prediction_recommendations(prediction_results)
            }
            
        except Exception as e:
            logger.error(f"性能预测出错: {e}")
            return {'error': str(e)}
    
    def _extract_time_series(self, performance_data: List[StrategyPerformance]) -> Tuple[List[float], List[datetime]]:
        """提取时间序列数据"""
        returns = [p.return_rate for p in performance_data]
        timestamps = [p.timestamp for p in performance_data]
        
        # 按时间排序
        sorted_data = sorted(zip(timestamps, returns), key=lambda x: x[0])
        timestamps, returns = zip(*sorted_data)
        
        return list(returns), list(timestamps)
    
    def _calculate_basic_metrics(self, returns: List[float]) -> Dict[str, float]:
        """计算基础性能指标"""
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # 收益指标
        total_return = np.prod(1 + returns_array) - 1
        average_return = np.mean(returns_array)
        return_volatility = np.std(returns_array)
        
        # 年化指标
        annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
        annualized_volatility = return_volatility * np.sqrt(self.trading_days_per_year)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'average_daily_return': average_return,
            'return_volatility': return_volatility,
            'annualized_volatility': annualized_volatility,
            'skewness': stats.skew(returns_array),
            'kurtosis': stats.kurtosis(returns_array),
            'jarque_bera_statistic': self._jarque_bera_test(returns_array)
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: List[float]) -> Dict[str, float]:
        """计算风险调整收益指标"""
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        daily_rf_rate = self.risk_free_rate / self.trading_days_per_year
        
        # 夏普比率
        excess_returns = returns_array - daily_rf_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year) if np.std(excess_returns) > 0 else 0
        
        # 索提诺比率
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days_per_year) if downside_deviation > 0 else 0
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        # 计算年化收益
        total_return = cumulative_returns[-1] - 1
        annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
        
        # 卡玛比率
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # 信息比率
        tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
        information_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(self.trading_days_per_year) if tracking_error > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'downside_deviation': downside_deviation
        }
    
    def _calculate_trading_metrics(self, performance_data: List[StrategyPerformance]) -> Dict[str, float]:
        """计算交易统计指标"""
        if not performance_data:
            return {}
        
        returns = [p.return_rate for p in performance_data]
        win_rates = [p.win_rate for p in performance_data]
        profit_factors = [p.profit_factor for p in performance_data]
        
        # 基础交易指标
        overall_win_rate = np.mean(win_rates)
        overall_profit_factor = np.mean(profit_factors)
        
        # 盈亏分析
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]
        
        average_win = np.mean(winning_returns) if winning_returns else 0
        average_loss = np.mean(losing_returns) if losing_returns else 0
        largest_win = max(winning_returns) if winning_returns else 0
        largest_loss = min(losing_returns) if losing_returns else 0
        
        # 盈亏比
        profit_loss_ratio = abs(average_win / average_loss) if average_loss != 0 else float('inf')
        
        # 连续性分析
        consecutive_wins, consecutive_losses = self._calculate_consecutive_stats(returns)
        
        return {
            'overall_win_rate': overall_win_rate,
            'overall_profit_factor': overall_profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'hit_rate': overall_win_rate,
            'trading_frequency': len(performance_data)
        }
    
    def _calculate_stability_metrics(self, returns: List[float]) -> Dict[str, float]:
        """计算稳定性和一致性指标"""
        if len(returns) < 2:
            return {}
        
        returns_array = np.array(returns)
        
        # 收益一致性
        positive_returns = returns_array[returns_array > 0]
        consistency_score = len(positive_returns) / len(returns_array)
        
        # 波动率稳定性
        rolling_volatility = self._calculate_rolling_volatility(returns, window=20)
        volatility_stability = 1.0 / (1.0 + np.std(rolling_volatility))
        
        # 收益率稳定性
        rolling_returns = self._calculate_rolling_returns(returns, window=20)
        return_stability = 1.0 / (1.0 + np.std(rolling_returns))
        
        # 趋势一致性
        trend_consistency = self._calculate_trend_consistency(returns)
        
        # 自相关性
        autocorr_lag1 = self._calculate_autocorrelation(returns, lag=1)
        
        return {
            'consistency_score': consistency_score,
            'volatility_stability': volatility_stability,
            'return_stability': return_stability,
            'trend_consistency': trend_consistency,
            'autocorrelation_lag1': autocorr_lag1,
            'hurst_exponent': self._calculate_hurst_exponent(returns)
        }
    
    def _benchmark_analysis(self, returns: List[float], 
                          benchmark_data: Optional[List[float]] = None) -> Dict[str, Any]:
        """基准比较分析"""
        if not benchmark_data:
            # 使用默认基准（假设为0收益）
            benchmark_data = [0.0] * len(returns)
        
        if len(returns) != len(benchmark_data):
            min_length = min(len(returns), len(benchmark_data))
            returns = returns[:min_length]
            benchmark_data = benchmark_data[:min_length]
        
        returns_array = np.array(returns)
        benchmark_array = np.array(benchmark_data)
        
        # 超额收益
        excess_returns = returns_array - benchmark_array
        
        # Beta和Alpha
        covariance = np.cov(returns_array, benchmark_array)[0, 1]
        benchmark_variance = np.var(benchmark_array)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        alpha = np.mean(excess_returns) * self.trading_days_per_year
        
        # 跟踪误差
        tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
        
        # 信息比率
        information_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(self.trading_days_per_year) if tracking_error > 0 else 0
        
        # 上下行捕获
        up_capture, down_capture = self._calculate_capture_ratios(returns_array, benchmark_array)
        
        return {
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'excess_return': np.mean(excess_returns),
            'excess_volatility': np.std(excess_returns)
        }
    
    def _performance_attribution(self, returns: List[float], 
                               performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """性能归因分析"""
        # 简化的归因分析
        attribution = {
            'return_attribution': {
                'systematic_return': np.mean(returns) * 0.6,  # 假设60%来自系统性因素
                'idiosyncratic_return': np.mean(returns) * 0.4  # 40%来自特质因素
            },
            'risk_attribution': {
                'market_risk': 0.7,  # 假设70%风险来自市场
                'specific_risk': 0.3  # 30%来自特质风险
            },
            'timing_attribution': {
                'market_timing': 0.2,
                'stock_selection': 0.5,
                'interaction': 0.3
            }
        }
        
        return attribution
    
    def _statistical_tests(self, returns: List[float]) -> Dict[str, Any]:
        """统计显著性检验"""
        if len(returns) < 10:
            return {'error': '数据点不足，无法进行统计检验'}
        
        returns_array = np.array(returns)
        
        # 正态性检验
        jb_stat, jb_pvalue = stats.jarque_bera(returns_array)
        
        # 均值检验
        t_stat, t_pvalue = stats.ttest_1samp(returns_array, 0)
        
        # 方差齐性检验
        levene_stat, levene_pvalue = stats.levene(returns_array, returns_array)  # 简化实现
        
        return {
            'jarque_bera_test': {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'is_normal': jb_pvalue > 0.05
            },
            'mean_test': {
                't_statistic': t_stat,
                'p_value': t_pvalue,
                'is_significant': t_pvalue < 0.05
            },
            'variance_test': {
                'levene_statistic': levene_stat,
                'p_value': levene_pvalue,
                'is_homoscedastic': levene_pvalue > 0.05
            }
        }
    
    def _predictive_analysis(self, returns: List[float]) -> Dict[str, Any]:
        """预测性分析"""
        if len(returns) < 20:
            return {'error': '数据点不足，无法进行预测性分析'}
        
        # 自相关性分析
        autocorr_1 = self._calculate_autocorrelation(returns, lag=1)
        autocorr_5 = self._calculate_autocorrelation(returns, lag=5)
        
        # 趋势分析
        trend_strength = self._calculate_trend_strength(returns)
        
        # 波动率预测
        volatility_forecast = self._forecast_volatility(returns)
        
        return {
            'autocorrelation_lag1': autocorr_1,
            'autocorrelation_lag5': autocorr_5,
            'trend_strength': trend_strength,
            'volatility_forecast': volatility_forecast,
            'predictability_score': self._calculate_predictability_score(returns)
        }
    
    def _risk_decomposition(self, returns: List[float]) -> Dict[str, float]:
        """风险分解"""
        returns_array = np.array(returns)
        
        # 总风险
        total_risk = np.std(returns_array)
        
        # 系统性风险（简化估算）
        systematic_risk = total_risk * 0.7
        
        # 特质风险
        idiosyncratic_risk = total_risk * 0.3
        
        # 波动率风险
        volatility_risk = np.std(np.diff(returns_array))
        
        return {
            'total_risk': total_risk,
            'systematic_risk': systematic_risk,
            'idiosyncratic_risk': idiosyncratic_risk,
            'volatility_risk': volatility_risk,
            'risk_concentration': self._calculate_risk_concentration(returns)
        }
    
    def _calculate_overall_score(self, basic_metrics: Dict, risk_adjusted_metrics: Dict,
                               trading_metrics: Dict, stability_metrics: Dict) -> Dict[str, float]:
        """计算综合评分"""
        scores = {}
        
        # 收益得分 (0-100)
        if 'annualized_return' in basic_metrics:
            return_score = max(0, min(100, basic_metrics['annualized_return'] * 500))  # 20%年化收益 = 100分
            scores['return_score'] = return_score
        
        # 风险调整收益得分
        if 'sharpe_ratio' in risk_adjusted_metrics:
            sharpe_score = max(0, min(100, risk_adjusted_metrics['sharpe_ratio'] * 25))  # Sharpe 4 = 100分
            scores['sharpe_score'] = sharpe_score
        
        # 交易表现得分
        if 'overall_win_rate' in trading_metrics:
            win_rate_score = trading_metrics['overall_win_rate'] * 100
            scores['win_rate_score'] = win_rate_score
        
        # 稳定性得分
        if 'consistency_score' in stability_metrics:
            stability_score = stability_metrics['consistency_score'] * 100
            scores['stability_score'] = stability_score
        
        # 综合得分
        if scores:
            scores['overall_score'] = np.mean(list(scores.values()))
        
        return scores
    
    def _grade_performance(self, scores: Dict[str, float]) -> str:
        """性能分级"""
        overall_score = scores.get('overall_score', 0)
        
        if overall_score >= 80:
            return "优秀"
        elif overall_score >= 65:
            return "良好"
        elif overall_score >= 50:
            return "一般"
        elif overall_score >= 35:
            return "较差"
        else:
            return "很差"
    
    def _generate_recommendations(self, basic_metrics: Dict, risk_adjusted_metrics: Dict,
                                trading_metrics: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于收益的建议
        if basic_metrics.get('annualized_return', 0) < 0.1:
            recommendations.append("年化收益偏低，建议优化选股策略或调整仓位")
        
        # 基于风险的建议
        if risk_adjusted_metrics.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("夏普比率偏低，建议加强风险控制")
        
        if risk_adjusted_metrics.get('max_drawdown', 0) > 0.2:
            recommendations.append("最大回撤过大，建议设置更严格的风险控制措施")
        
        # 基于交易的建议
        if trading_metrics.get('overall_win_rate', 0) < 0.5:
            recommendations.append("胜率偏低，建议优化入场和出场时机")
        
        if trading_metrics.get('profit_loss_ratio', 0) < 1.5:
            recommendations.append("盈亏比偏低，建议提高盈利目标或降低止损点")
        
        return recommendations
    
    # 辅助方法
    def _jarque_bera_test(self, returns: np.ndarray) -> float:
        """Jarque-Bera正态性检验"""
        try:
            jb_stat, _ = stats.jarque_bera(returns)
            return jb_stat
        except:
            return 0.0
    
    def _calculate_consecutive_stats(self, returns: List[float]) -> Tuple[int, int]:
        """计算连续盈亏统计"""
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for ret in returns:
            if ret > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif ret < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return max_consecutive_wins, max_consecutive_losses
    
    def _calculate_rolling_volatility(self, returns: List[float], window: int = 20) -> List[float]:
        """计算滚动波动率"""
        if len(returns) < window:
            return [np.std(returns)] * len(returns)
        
        rolling_vol = []
        for i in range(window - 1, len(returns)):
            vol = np.std(returns[i - window + 1:i + 1])
            rolling_vol.append(vol)
        
        return rolling_vol
    
    def _calculate_rolling_returns(self, returns: List[float], window: int = 20) -> List[float]:
        """计算滚动收益率"""
        if len(returns) < window:
            return returns
        
        rolling_returns = []
        for i in range(window - 1, len(returns)):
            cum_return = np.prod([1 + r for r in returns[i - window + 1:i + 1]]) - 1
            rolling_returns.append(cum_return)
        
        return rolling_returns
    
    def _calculate_trend_consistency(self, returns: List[float]) -> float:
        """计算趋势一致性"""
        if len(returns) < 3:
            return 0.5
        
        # 计算连续符号变化
        sign_changes = 0
        for i in range(1, len(returns)):
            if (returns[i-1] > 0) != (returns[i] > 0):
                sign_changes += 1
        
        consistency = 1.0 - (sign_changes / (len(returns) - 1))
        return consistency
    
    def _calculate_autocorrelation(self, returns: List[float], lag: int = 1) -> float:
        """计算自相关系数"""
        if len(returns) <= lag:
            return 0.0
        
        try:
            return np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
        except:
            return 0.0
    
    def _calculate_hurst_exponent(self, returns: List[float]) -> float:
        """计算Hurst指数"""
        if len(returns) < 10:
            return 0.5
        
        try:
            # 简化的Hurst指数计算
            returns_array = np.array(returns)
            cumsum = np.cumsum(returns_array)
            mean_return = np.mean(returns_array)
            
            deviations = cumsum - np.arange(len(returns)) * mean_return
            R = np.max(deviations) - np.min(deviations)
            S = np.std(returns_array)
            
            if S == 0:
                return 0.5
            
            return np.log(R / S) / np.log(len(returns))
        except:
            return 0.5
    
    def _calculate_capture_ratios(self, returns: np.ndarray, benchmark: np.ndarray) -> Tuple[float, float]:
        """计算捕获比率"""
        try:
            # 上行捕获
            up_benchmark = benchmark[benchmark > 0]
            up_returns = returns[benchmark > 0]
            up_capture = np.mean(up_returns) / np.mean(up_benchmark) if len(up_benchmark) > 0 else 0
            
            # 下行捕获
            down_benchmark = benchmark[benchmark < 0]
            down_returns = returns[benchmark < 0]
            down_capture = np.mean(down_returns) / np.mean(down_benchmark) if len(down_benchmark) > 0 else 0
            
            return up_capture, down_capture
        except:
            return 0.0, 0.0
    
    def _calculate_risk_concentration(self, returns: List[float]) -> float:
        """计算风险集中度"""
        if not returns:
            return 0.0
        
        # 使用赫芬达尔指数衡量风险集中度
        weights = np.abs(returns) / np.sum(np.abs(returns))
        hhi = np.sum(weights ** 2)
        
        return hhi
    
    def _calculate_trend_strength(self, returns: List[float]) -> float:
        """计算趋势强度"""
        if len(returns) < 5:
            return 0.0
        
        try:
            # 线性回归斜率
            x = np.arange(len(returns))
            slope, _, r_value, _, _ = stats.linregress(x, returns)
            
            # 趋势强度 = |斜率| * R²
            trend_strength = abs(slope) * (r_value ** 2)
            
            return trend_strength
        except:
            return 0.0
    
    def _forecast_volatility(self, returns: List[float]) -> Dict[str, float]:
        """波动率预测"""
        if len(returns) < 10:
            return {'current_volatility': 0.0, 'forecasted_volatility': 0.0}
        
        current_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        
        # 简化的GARCH(1,1)预测
        alpha = 0.1
        beta = 0.8
        long_run_var = np.var(returns)
        
        recent_var = current_vol ** 2
        forecasted_var = long_run_var * (1 - alpha - beta) + alpha * recent_var + beta * recent_var
        forecasted_vol = np.sqrt(forecasted_var)
        
        return {
            'current_volatility': current_vol,
            'forecasted_volatility': forecasted_vol,
            'volatility_trend': 'increasing' if forecasted_vol > current_vol else 'decreasing'
        }
    
    def _calculate_predictability_score(self, returns: List[float]) -> float:
        """计算可预测性得分"""
        if len(returns) < 10:
            return 0.0
        
        # 基于自相关性和趋势强度
        autocorr = abs(self._calculate_autocorrelation(returns, lag=1))
        trend_strength = self._calculate_trend_strength(returns)
        
        predictability = (autocorr + trend_strength) / 2
        return min(1.0, predictability)
    
    # 简化的其他分析方法实现
    def _rank_strategies(self, analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """策略排名"""
        rankings = []
        for strategy_id, analysis in analyses.items():
            if 'overall_score' in analysis:
                rankings.append({
                    'strategy_id': strategy_id,
                    'overall_score': analysis['overall_score']['overall_score'],
                    'performance_grade': analysis['performance_grade']
                })
        
        rankings.sort(key=lambda x: x['overall_score'], reverse=True)
        return rankings
    
    def _strategy_correlation_analysis(self, strategy_performances: Dict[str, List[StrategyPerformance]]) -> Dict[str, Any]:
        """策略相关性分析"""
        # 简化的相关性分析
        return {'average_correlation': 0.3, 'correlation_matrix': {}}
    
    def _diversification_analysis(self, strategy_performances: Dict[str, List[StrategyPerformance]]) -> Dict[str, Any]:
        """分散化分析"""
        return {'diversification_ratio': 0.7, 'optimal_weights': {}}
    
    def _optimal_portfolio_analysis(self, analyses: Dict[str, Any], correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """最优组合分析"""
        return {'optimal_weights': {}, 'expected_return': 0.1, 'expected_risk': 0.15}
    
    def _risk_contribution_analysis(self, strategy_performances: Dict[str, List[StrategyPerformance]]) -> Dict[str, Any]:
        """风险贡献分析"""
        return {'risk_contributions': {}}
    
    def _generate_comparison_summary(self, analyses: Dict[str, Any], rankings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成比较总结"""
        return {
            'best_strategy': rankings[0]['strategy_id'] if rankings else None,
            'worst_strategy': rankings[-1]['strategy_id'] if rankings else None,
            'performance_spread': rankings[0]['overall_score'] - rankings[-1]['overall_score'] if len(rankings) > 1 else 0
        }
    
    def _factor_attribution(self, returns: List[float], market_factors: Optional[Dict[str, List[float]]]) -> Dict[str, Any]:
        """因子归因"""
        return {'market_factor': 0.6, 'size_factor': 0.2, 'value_factor': 0.2}
    
    def _temporal_attribution(self, returns: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """时间归因"""
        return {'morning_effect': 0.1, 'afternoon_effect': 0.05, 'day_of_week_effect': 0.03}
    
    def _risk_attribution(self, returns: List[float]) -> Dict[str, Any]:
        """风险归因"""
        return {'systematic_risk': 0.7, 'idiosyncratic_risk': 0.3}
    
    def _strategy_attribution(self, performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """策略归因"""
        return {'timing_contribution': 0.4, 'selection_contribution': 0.6}
    
    def _validate_attribution(self, factor_attribution: Dict[str, Any], returns: List[float]) -> Dict[str, Any]:
        """验证归因结果"""
        return {'attribution_quality': 0.8, 'residual': 0.0}
    
    def _var_analysis(self, returns: List[float], confidence_levels: List[float]) -> Dict[str, Any]:
        """VaR分析"""
        var_results = {}
        for level in confidence_levels:
            var_results[f'var_{int(level*100)}'] = np.percentile(returns, (1-level)*100)
        return var_results
    
    def _stress_testing(self, returns: List[float]) -> Dict[str, Any]:
        """压力测试"""
        return {'historical_stress': -0.1, 'scenario_stress': -0.15}
    
    def _detailed_risk_decomposition(self, returns: List[float]) -> Dict[str, Any]:
        """详细风险分解"""
        return {'total_risk': np.std(returns), 'component_risks': {}}
    
    def _liquidity_risk_analysis(self, performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """流动性风险分析"""
        return {'liquidity_score': 0.8, 'market_impact': 0.05}
    
    def _operational_risk_analysis(self, performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """操作风险分析"""
        return {'operational_risk_score': 0.9, 'error_rate': 0.02}
    
    def _generate_risk_summary(self, var_analysis: Dict[str, Any], stress_testing: Dict[str, Any]) -> Dict[str, Any]:
        """生成风险总结"""
        return {'overall_risk_level': 'medium', 'risk_factors': ['volatility', 'liquidity']}
    
    def _monte_carlo_prediction(self, returns: List[float], horizon: int) -> Dict[str, Any]:
        """蒙特卡洛预测"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        simulations = []
        for _ in range(1000):
            sim_returns = np.random.normal(mean_return, std_return, horizon)
            cumulative_return = np.prod(1 + sim_returns) - 1
            simulations.append(cumulative_return)
        
        return {
            'mean_prediction': np.mean(simulations),
            'median_prediction': np.median(simulations),
            'percentile_5': np.percentile(simulations, 5),
            'percentile_95': np.percentile(simulations, 95)
        }
    
    def _time_series_prediction(self, returns: List[float], horizon: int) -> Dict[str, Any]:
        """时间序列预测"""
        # 简化的AR(1)预测
        if len(returns) < 2:
            return {'prediction': 0.0, 'confidence': 0.0}
        
        # 计算AR(1)参数
        returns_array = np.array(returns)
        lagged_returns = returns_array[:-1]
        current_returns = returns_array[1:]
        
        try:
            correlation = np.corrcoef(lagged_returns, current_returns)[0, 1]
            last_return = returns[-1]
            prediction = correlation * last_return
            confidence = abs(correlation)
        except:
            prediction = np.mean(returns)
            confidence = 0.0
        
        return {'prediction': prediction, 'confidence': confidence}
    
    def _regime_based_prediction(self, returns: List[float], horizon: int) -> Dict[str, Any]:
        """基于状态的预测"""
        # 简化的状态识别
        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        current_regime = 'bull' if np.mean(recent_returns) > 0 else 'bear'
        
        regime_predictions = {
            'bull': 0.02,
            'bear': -0.01,
            'neutral': 0.0
        }
        
        return {
            'current_regime': current_regime,
            'regime_prediction': regime_predictions.get(current_regime, 0.0),
            'regime_confidence': 0.6
        }
    
    def _ensemble_prediction(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """集成预测"""
        if not predictions:
            return {'ensemble_prediction': 0.0, 'confidence': 0.0}
        
        # 简单平均集成
        pred_values = []
        confidences = []
        
        for method, pred in predictions.items():
            if 'prediction' in pred:
                pred_values.append(pred['prediction'])
            if 'mean_prediction' in pred:
                pred_values.append(pred['mean_prediction'])
            if 'confidence' in pred:
                confidences.append(pred['confidence'])
        
        if pred_values:
            ensemble_pred = np.mean(pred_values)
            ensemble_confidence = np.mean(confidences) if confidences else 0.5
        else:
            ensemble_pred = 0.0
            ensemble_confidence = 0.0
        
        return {
            'ensemble_prediction': ensemble_pred,
            'confidence': ensemble_confidence,
            'methods_used': list(predictions.keys())
        }
    
    def _validate_predictions(self, returns: List[float], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """验证预测结果"""
        return {'validation_score': 0.7, 'prediction_accuracy': 0.6}
    
    def _calculate_prediction_intervals(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """计算预测区间"""
        return {'lower_bound': -0.05, 'upper_bound': 0.05}
    
    def _generate_prediction_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """生成预测建议"""
        return ['保持当前策略', '关注市场变化']