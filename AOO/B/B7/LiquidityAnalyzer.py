#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B7流动性分析器
实现市场微观结构理论基础的流动性分析系统

功能模块：
1. 市场流动性评估和量化
2. 流动性风险评估
3. 流动性成本分析
4. 流动性提供者分析
5. 流动性季节性分析
6. 流动性危机预警
7. 流动性优化建议


创建时间: 2025-11-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LiquidityRiskLevel(Enum):
    """流动性风险等级"""
    VERY_LOW = "极低"
    LOW = "低"
    MEDIUM = "中等"
    HIGH = "高"
    VERY_HIGH = "极高"


class LiquidityCrisisLevel(Enum):
    """流动性危机等级"""
    NORMAL = "正常"
    WARNING = "预警"
    ALERT = "警报"
    EMERGENCY = "紧急"


@dataclass
class LiquidityMetrics:
    """流动性指标数据结构"""
    amihud_ratio: float
    roll_spread: float
    pastor_beta: float
    bid_ask_spread: float
    market_depth: float
    turnover_ratio: float
    price_impact: float
    trading_volume: float
    order_book_imbalance: float


@dataclass
class LiquidityRiskAssessment:
    """流动性风险评估结果"""
    risk_level: LiquidityRiskLevel
    risk_score: float
    key_risks: List[str]
    recommendations: List[str]
    timestamp: datetime


class LiquidityAnalyzer:
    """流动性分析器主类"""
    
    def __init__(self):
        """初始化流动性分析器"""
        self.market_data = None
        self.order_book_data = None
        self.trade_data = None
        self.liquidity_metrics_history = []
        self.crisis_threshold = 0.8
        self.warning_threshold = 0.6
        
    def load_market_data(self, data: pd.DataFrame):
        """加载市场数据"""
        """
        预期数据格式：
        - timestamp: 时间戳
        - open, high, low, close: 价格数据
        - volume: 成交量
        - bid_price, ask_price: 买卖价格
        - bid_size, ask_size: 买卖数量
        """
        self.market_data = data.copy()
        if 'timestamp' in data.columns:
            self.market_data['timestamp'] = pd.to_datetime(data['timestamp'])
            self.market_data = self.market_data.sort_values('timestamp')
        
        print(f"已加载市场数据，共 {len(data)} 条记录")
        
    def calculate_amihud_ratio(self, returns: pd.Series, volume: pd.Series) -> float:
        """
        计算Amihud流动性比率
        Amihud = |return| / volume
        """
        if len(returns) == 0 or len(volume) == 0:
            return 0.0
            
        abs_returns = np.abs(returns)
        amihud = np.mean(abs_returns / volume)
        return amihud
    
    def calculate_roll_spread(self, price_changes: pd.Series) -> float:
        """
        计算Roll有效价差
        基于价格序列的自协方差估计
        """
        if len(price_changes) < 2:
            return 0.0
            
        # 计算一阶自协方差
        autocov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
        
        # Roll价差估计
        roll_spread = 2 * np.sqrt(-autocov) if autocov < 0 else 0.0
        return roll_spread
    
    def calculate_pastor_stambaugh(self, returns: pd.Series, volume: pd.Series) -> float:
        """
        计算Pastor-Stambaugh流动性指标
        基于价格冲击与成交量的关系
        """
        if len(returns) < 2 or len(volume) < 2:
            return 0.0
            
        # 计算滞后成交量
        volume_lag = volume.shift(1).fillna(volume.mean())
        
        # 确保数据长度一致
        min_length = min(len(returns), len(volume_lag))
        returns_trimmed = returns.iloc[:min_length]
        volume_lag_trimmed = volume_lag.iloc[:min_length]
        
        # 回归分析：return = alpha + beta * volume_lag + error
        try:
            from sklearn.linear_model import LinearRegression
            X = volume_lag_trimmed.values.reshape(-1, 1)
            y = returns_trimmed.values
            
            # 移除NaN值
            mask = ~(np.isnan(X.flatten()) | np.isnan(y))
            X_clean = X[mask].reshape(-1, 1)
            y_clean = y[mask]
            
            if len(X_clean) > 1:
                model = LinearRegression()
                model.fit(X_clean, y_clean)
                return model.coef_[0]
        except ImportError:
            # 简单的相关性计算
            correlation = np.corrcoef(returns_trimmed.dropna(), volume_lag_trimmed.dropna())[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        return 0.0
    
    def calculate_bid_ask_spread(self, bid_price: float, ask_price: float) -> float:
        """计算买卖价差"""
        if bid_price <= 0 or ask_price <= 0:
            return 0.0
        return (ask_price - bid_price) / ((ask_price + bid_price) / 2)
    
    def calculate_market_depth(self, bid_size: float, ask_size: float) -> float:
        """计算市场深度"""
        return (bid_size + ask_size) / 2
    
    def calculate_price_impact(self, volume: pd.Series, returns: pd.Series) -> float:
        """计算价格冲击"""
        if len(volume) == 0 or len(returns) == 0:
            return 0.0
            
        # 确保数据长度一致
        min_length = min(len(returns), len(volume))
        returns_trimmed = returns.iloc[:min_length]
        volume_trimmed = volume.iloc[:min_length]
        
        # 计算价格冲击指标
        price_impact = np.corrcoef(np.abs(returns_trimmed), volume_trimmed)[0, 1]
        return price_impact if not np.isnan(price_impact) else 0.0
    
    def assess_liquidity_metrics(self) -> LiquidityMetrics:
        """评估当前流动性指标"""
        if self.market_data is None or len(self.market_data) == 0:
            raise ValueError("请先加载市场数据")
        
        data = self.market_data
        
        # 计算收益率
        returns = data['close'].pct_change().dropna()
        
        # 计算各种流动性指标
        amihud_ratio = self.calculate_amihud_ratio(returns, data['volume'])
        
        # 计算价格变化
        price_changes = data['close'].diff().dropna()
        roll_spread = self.calculate_roll_spread(price_changes)
        
        # Pastor-Stambaugh指标
        pastor_beta = self.calculate_pastor_stambaugh(returns, data['volume'])
        
        # 买卖价差
        bid_ask_spreads = []
        if all(col in data.columns for col in ['bid_price', 'ask_price']):
            bid_ask_spreads = [
                self.calculate_bid_ask_spread(bid, ask)
                for bid, ask in zip(data['bid_price'], data['ask_price'])
            ]
        avg_bid_ask_spread = np.mean(bid_ask_spreads) if bid_ask_spreads else 0.0
        
        # 市场深度
        market_depths = []
        if all(col in data.columns for col in ['bid_size', 'ask_size']):
            market_depths = [
                self.calculate_market_depth(bid_size, ask_size)
                for bid_size, ask_size in zip(data['bid_size'], data['ask_size'])
            ]
        avg_market_depth = np.mean(market_depths) if market_depths else 0.0
        
        # 换手率
        turnover_ratio = data['volume'].mean() / (data['volume'].std() + 1e-8)
        
        # 价格冲击
        price_impact = self.calculate_price_impact(data['volume'], returns)
        
        # 总成交量
        trading_volume = data['volume'].sum()
        
        # 订单簿不平衡
        order_book_imbalance = 0.0
        if market_depths:
            order_book_imbalance = np.std(market_depths) / (np.mean(market_depths) + 1e-8)
        
        metrics = LiquidityMetrics(
            amihud_ratio=amihud_ratio,
            roll_spread=roll_spread,
            pastor_beta=pastor_beta,
            bid_ask_spread=avg_bid_ask_spread,
            market_depth=avg_market_depth,
            turnover_ratio=turnover_ratio,
            price_impact=price_impact,
            trading_volume=trading_volume,
            order_book_imbalance=order_book_imbalance
        )
        
        return metrics
    
    def assess_liquidity_risk(self, metrics: LiquidityMetrics) -> LiquidityRiskAssessment:
        """评估流动性风险"""
        risk_score = 0.0
        key_risks = []
        recommendations = []
        
        # Amihud比率风险评估 (值越高流动性越差)
        if metrics.amihud_ratio > 0.1:
            risk_score += 0.3
            key_risks.append("Amihud比率过高，流动性较差")
            recommendations.append("增加市场参与度，提高交易活跃度")
        
        # Roll价差风险评估
        if metrics.roll_spread > 0.05:
            risk_score += 0.25
            key_risks.append("有效价差过大，交易成本高")
            recommendations.append("优化做市策略，减少价差")
        
        # Pastor指标风险评估
        if abs(metrics.pastor_beta) > 0.01:
            risk_score += 0.2
            key_risks.append("价格冲击敏感度高")
            recommendations.append("分批交易，降低单笔交易对价格的影响")
        
        # 买卖价差风险评估
        if metrics.bid_ask_spread > 0.02:
            risk_score += 0.2
            key_risks.append("买卖价差过大")
            recommendations.append("改善订单簿深度，平衡买卖力量")
        
        # 市场深度风险评估
        if metrics.market_depth < 1000:
            risk_score += 0.25
            key_risks.append("市场深度不足")
            recommendations.append("吸引更多流动性提供者")
        
        # 订单簿不平衡风险
        if metrics.order_book_imbalance > 0.5:
            risk_score += 0.15
            key_risks.append("订单簿严重不平衡")
            recommendations.append("平衡市场供给和需求")
        
        # 确定风险等级
        if risk_score >= 0.8:
            risk_level = LiquidityRiskLevel.VERY_HIGH
        elif risk_score >= 0.6:
            risk_level = LiquidityRiskLevel.HIGH
        elif risk_score >= 0.4:
            risk_level = LiquidityRiskLevel.MEDIUM
        elif risk_score >= 0.2:
            risk_level = LiquidityRiskLevel.LOW
        else:
            risk_level = LiquidityRiskLevel.VERY_LOW
        
        # 通用建议
        if not recommendations:
            recommendations.append("流动性状况良好，继续保持")
        
        return LiquidityRiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            key_risks=key_risks,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def analyze_liquidity_cost(self, metrics: LiquidityMetrics) -> Dict[str, float]:
        """分析流动性成本"""
        costs = {}
        
        # 显性成本：买卖价差
        costs['spread_cost'] = metrics.bid_ask_spread
        
        # 隐性成本：价格冲击
        costs['impact_cost'] = metrics.price_impact * 0.01  # 转换为百分比
        
        # 执行成本：基于Amihud比率
        costs['execution_cost'] = metrics.amihud_ratio
        
        # 总成本
        costs['total_cost'] = (costs['spread_cost'] + 
                             costs['impact_cost'] + 
                             costs['execution_cost']) / 3
        
        return costs
    
    def analyze_liquidity_providers(self) -> Dict[str, Any]:
        """分析流动性提供者"""
        if self.market_data is None:
            return {}
        
        # 分析交易模式
        data = self.market_data
        
        # 计算交易频率
        if 'timestamp' in data.columns:
            time_diffs = data['timestamp'].diff().dt.total_seconds()
            avg_time_between_trades = time_diffs.mean()
        else:
            avg_time_between_trades = 1.0
        
        # 分析交易量分布
        volume_stats = {
            'mean_volume': data['volume'].mean(),
            'median_volume': data['volume'].median(),
            'volume_volatility': data['volume'].std() / (data['volume'].mean() + 1e-8),
            'large_trade_ratio': (data['volume'] > data['volume'].quantile(0.9)).mean()
        }
        
        # 分析价格影响
        price_impact_analysis = {
            'high_impact_trades': 0,
            'medium_impact_trades': 0,
            'low_impact_trades': 0,
            'avg_price_change': data['close'].pct_change().abs().mean()
        }
        
        # 计算价格变化与成交量的关系
        if len(data) > 1:
            price_changes = data['close'].pct_change().abs()
            correlation = np.corrcoef(price_changes.dropna(), data['volume'][1:])[0, 1]
            price_impact_analysis['volume_price_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        provider_analysis = {
            'trading_frequency': avg_time_between_trades,
            'volume_characteristics': volume_stats,
            'price_impact_pattern': price_impact_analysis,
            'liquidity_concentration': volume_stats['large_trade_ratio']
        }
        
        return provider_analysis
    
    def analyze_liquidity_seasonality(self, period: str = 'daily') -> Dict[str, Any]:
        """分析流动性季节性"""
        if self.market_data is None or 'timestamp' not in self.market_data.columns:
            return {}
        
        data = self.market_data.copy()
        
        # 按时间维度分组
        if period == 'daily':
            data['time_group'] = data['timestamp'].dt.hour
        elif period == 'weekly':
            data['time_group'] = data['timestamp'].dt.dayofweek
        elif period == 'monthly':
            data['time_group'] = data['timestamp'].dt.day
        else:
            data['time_group'] = data['timestamp'].dt.hour
        
        # 计算各时间段的流动性指标
        liquidity_by_time = data.groupby('time_group').agg({
            'volume': ['mean', 'std'],
            'close': lambda x: x.pct_change().abs().mean()
        }).round(4)
        
        # 计算季节性指标
        seasonality_metrics = {
            'peak_liquidity_hours': liquidity_by_time['volume']['mean'].idxmax(),
            'low_liquidity_hours': liquidity_by_time['volume']['mean'].idxmin(),
            'liquidity_volatility': liquidity_by_time['volume']['std'].mean(),
            'price_volatility_by_time': liquidity_by_time['close']['<lambda>'].to_dict()
        }
        
        return seasonality_metrics
    
    def detect_liquidity_crisis(self, metrics: LiquidityMetrics) -> LiquidityCrisisLevel:
        """检测流动性危机"""
        crisis_score = 0.0
        
        # 危机指标检测
        if metrics.amihud_ratio > 0.2:  # Amihud比率异常高
            crisis_score += 0.3
        
        if metrics.bid_ask_spread > 0.05:  # 买卖价差异常大
            crisis_score += 0.25
        
        if metrics.market_depth < 500:  # 市场深度严重不足
            crisis_score += 0.25
        
        if metrics.turnover_ratio < 0.1:  # 换手率过低
            crisis_score += 0.2
        
        # 确定危机等级
        if crisis_score >= self.crisis_threshold:
            return LiquidityCrisisLevel.EMERGENCY
        elif crisis_score >= 0.7:
            return LiquidityCrisisLevel.ALERT
        elif crisis_score >= self.warning_threshold:
            return LiquidityCrisisLevel.WARNING
        else:
            return LiquidityCrisisLevel.NORMAL
    
    def generate_optimization_recommendations(self, 
                                           metrics: LiquidityMetrics,
                                           risk_assessment: LiquidityRiskAssessment,
                                           provider_analysis: Dict[str, Any]) -> List[str]:
        """生成流动性优化建议"""
        recommendations = []
        
        # 基于风险评估的建议
        recommendations.extend(risk_assessment.recommendations)
        
        # 基于流动性指标的建议
        if metrics.amihud_ratio > 0.05:
            recommendations.append("考虑实施算法交易策略，提高执行效率")
        
        if metrics.bid_ask_spread > 0.01:
            recommendations.append("优化做市算法，动态调整买卖价差")
        
        if metrics.market_depth < 1000:
            recommendations.append("增加流动性激励措施，吸引更多做市商")
        
        if metrics.price_impact > 0.5:
            recommendations.append("实施TWAP/VWAP等算法交易，减少市场冲击")
        
        # 基于流动性提供者的建议
        if provider_analysis.get('liquidity_concentration', 0) > 0.3:
            recommendations.append("分散流动性来源，减少对大额交易者的依赖")
        
        # 季节性建议
        seasonality = self.analyze_liquidity_seasonality()
        if seasonality:
            peak_hours = seasonality.get('peak_liquidity_hours', 0)
            recommendations.append(f"在{peak_hours}时段增加交易活动，利用高流动性时期")
        
        return recommendations
    
    def generate_liquidity_report(self) -> Dict[str, Any]:
        """生成流动性分析报告"""
        if self.market_data is None:
            raise ValueError("请先加载市场数据")
        
        # 获取当前指标
        metrics = self.assess_liquidity_metrics()
        risk_assessment = self.assess_liquidity_risk(metrics)
        cost_analysis = self.analyze_liquidity_cost(metrics)
        provider_analysis = self.analyze_liquidity_providers()
        seasonality_analysis = self.analyze_liquidity_seasonality()
        crisis_level = self.detect_liquidity_crisis(metrics)
        recommendations = self.generate_optimization_recommendations(
            metrics, risk_assessment, provider_analysis
        )
        
        # 保存历史记录
        self.liquidity_metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'risk_assessment': risk_assessment
        })
        
        report = {
            'analysis_time': datetime.now(),
            'liquidity_metrics': {
                'amihud_ratio': metrics.amihud_ratio,
                'roll_spread': metrics.roll_spread,
                'pastor_beta': metrics.pastor_beta,
                'bid_ask_spread': metrics.bid_ask_spread,
                'market_depth': metrics.market_depth,
                'turnover_ratio': metrics.turnover_ratio,
                'price_impact': metrics.price_impact,
                'trading_volume': metrics.trading_volume,
                'order_book_imbalance': metrics.order_book_imbalance
            },
            'risk_assessment': {
                'risk_level': risk_assessment.risk_level.value,
                'risk_score': risk_assessment.risk_score,
                'key_risks': risk_assessment.key_risks
            },
            'cost_analysis': cost_analysis,
            'provider_analysis': provider_analysis,
            'seasonality_analysis': seasonality_analysis,
            'crisis_monitoring': {
                'crisis_level': crisis_level.value,
                'is_crisis': crisis_level in [LiquidityCrisisLevel.ALERT, LiquidityCrisisLevel.EMERGENCY]
            },
            'optimization_recommendations': recommendations
        }
        
        return report
    
    def plot_liquidity_analysis(self, save_path: Optional[str] = None):
        """绘制流动性分析图表"""
        if self.market_data is None:
            raise ValueError("请先加载市场数据")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('流动性分析报告', fontsize=16, fontweight='bold')
        
        data = self.market_data
        
        # 1. 价格和成交量
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(data.index, data['close'], 'b-', label='收盘价', linewidth=1)
        ax1_twin.bar(data.index, data['volume'], alpha=0.3, color='orange', label='成交量')
        
        ax1.set_title('价格与成交量')
        ax1.set_ylabel('价格', color='b')
        ax1_twin.set_ylabel('成交量', color='orange')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. 流动性指标时间序列
        ax2 = axes[0, 1]
        returns = data['close'].pct_change()
        amihud_series = np.abs(returns) / (data['volume'] + 1e-8)
        
        ax2.plot(data.index, amihud_series, 'g-', label='Amihud比率')
        ax2.set_title('Amihud流动性比率')
        ax2.set_ylabel('Amihud比率')
        ax2.legend()
        
        # 3. 买卖价差
        ax3 = axes[0, 2]
        if all(col in data.columns for col in ['bid_price', 'ask_price']):
            spreads = (data['ask_price'] - data['bid_price']) / ((data['ask_price'] + data['bid_price']) / 2)
            ax3.plot(data.index, spreads, 'r-', label='买卖价差')
            ax3.set_title('买卖价差')
            ax3.set_ylabel('价差比例')
            ax3.legend()
        
        # 4. 市场深度
        ax4 = axes[1, 0]
        if all(col in data.columns for col in ['bid_size', 'ask_size']):
            depth = (data['bid_size'] + data['ask_size']) / 2
            ax4.plot(data.index, depth, 'purple', label='市场深度')
            ax4.set_title('市场深度')
            ax4.set_ylabel('深度')
            ax4.legend()
        
        # 5. 价格冲击
        ax5 = axes[1, 1]
        price_changes = np.abs(data['close'].pct_change())
        ax5.scatter(data['volume'][1:], price_changes[1:], alpha=0.6, c='blue')
        ax5.set_xlabel('成交量')
        ax5.set_ylabel('价格变化')
        ax5.set_title('成交量与价格冲击关系')
        
        # 6. 流动性风险等级分布
        ax6 = axes[1, 2]
        if self.liquidity_metrics_history:
            risk_scores = [item['risk_assessment'].risk_score for item in self.liquidity_metrics_history]
            ax6.hist(risk_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax6.set_title('流动性风险评分分布')
            ax6.set_xlabel('风险评分')
            ax6.set_ylabel('频次')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
    
    def real_time_monitoring(self, interval_seconds: int = 60):
        """实时流动性监控"""
        print(f"开始实时流动性监控，监控间隔: {interval_seconds}秒")
        print("按 Ctrl+C 停止监控")
        
        try:
            while True:
                # 模拟实时数据更新
                current_time = datetime.now()
                
                # 重新评估当前流动性状况
                if self.market_data is not None:
                    metrics = self.assess_liquidity_metrics()
                    crisis_level = self.detect_liquidity_crisis(metrics)
                    
                    print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 流动性监控报告")
                    print(f"Amihud比率: {metrics.amihud_ratio:.6f}")
                    print(f"买卖价差: {metrics.bid_ask_spread:.4f}")
                    print(f"市场深度: {metrics.market_depth:.2f}")
                    print(f"危机等级: {crisis_level.value}")
                    
                    if crisis_level in [LiquidityCrisisLevel.WARNING, LiquidityCrisisLevel.ALERT, LiquidityCrisisLevel.EMERGENCY]:
                        print("⚠️  警告：检测到流动性异常！")
                
                import time
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n实时监控已停止")


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """创建示例数据用于测试"""
    np.random.seed(42)
    
    # 生成时间序列
    start_date = datetime.now() - timedelta(days=30)
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='5min')
    
    # 生成价格数据 (几何布朗运动)
    dt = 1/288  # 5分钟间隔
    mu = 0.0001  # 漂移
    sigma = 0.02  # 波动率
    
    prices = [100.0]  # 初始价格
    for i in range(1, n_samples):
        random_shock = np.random.normal(0, 1)
        price_change = mu * dt + sigma * np.sqrt(dt) * random_shock
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 1.0))  # 确保价格为正
    
    # 生成成交量
    base_volume = 1000
    volumes = np.random.lognormal(np.log(base_volume), 0.5, n_samples)
    
    # 生成买卖价差数据
    bid_prices = [p * (1 - np.random.uniform(0.001, 0.005)) for p in prices]
    ask_prices = [p * (1 + np.random.uniform(0.001, 0.005)) for p in prices]
    
    # 生成订单簿深度
    bid_sizes = np.random.uniform(100, 1000, n_samples)
    ask_sizes = np.random.uniform(100, 1000, n_samples)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': volumes,
        'bid_price': bid_prices,
        'ask_price': ask_prices,
        'bid_size': bid_sizes,
        'ask_size': ask_sizes
    })
    
    return data


def main():
    """主函数 - 演示流动性分析器功能"""
    print("=== B7流动性分析器演示 ===\n")
    
    # 创建分析器实例
    analyzer = LiquidityAnalyzer()
    
    # 生成示例数据
    print("1. 生成示例市场数据...")
    sample_data = create_sample_data(1000)
    analyzer.load_market_data(sample_data)
    
    # 生成流动性分析报告
    print("\n2. 生成流动性分析报告...")
    report = analyzer.generate_liquidity_report()
    
    # 打印报告摘要
    print(f"\n=== 流动性分析报告摘要 ===")
    print(f"分析时间: {report['analysis_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n【流动性指标】")
    metrics = report['liquidity_metrics']
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\n【风险评估】")
    risk = report['risk_assessment']
    print(f"  风险等级: {risk['risk_level']}")
    print(f"  风险评分: {risk['risk_score']:.3f}")
    if risk['key_risks']:
        print(f"  主要风险: {', '.join(risk['key_risks'])}")
    
    print(f"\n【成本分析】")
    costs = report['cost_analysis']
    for key, value in costs.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\n【危机监控】")
    crisis = report['crisis_monitoring']
    print(f"  危机等级: {crisis['crisis_level']}")
    print(f"  是否危机: {crisis['is_crisis']}")
    
    print(f"\n【优化建议】")
    for i, rec in enumerate(report['optimization_recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # 绘制分析图表
    print(f"\n3. 生成流动性分析图表...")
    analyzer.plot_liquidity_analysis(save_path='liquidity_analysis_report.png')
    
    # 季节性分析
    print(f"\n4. 流动性季节性分析...")
    seasonality = analyzer.analyze_liquidity_seasonality()
    if seasonality:
        print(f"  高流动性时段: {seasonality.get('peak_liquidity_hours', 'N/A')}点")
        print(f"  低流动性时段: {seasonality.get('low_liquidity_hours', 'N/A')}点")
    
    print(f"\n=== 演示完成 ===")


if __name__ == "__main__":
    main()