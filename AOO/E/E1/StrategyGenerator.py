"""
E1策略生成器 - 智能交易策略自动生成系统

该模块实现了完整的策略生成框架，包括：
- 多类型交易策略生成（趋势跟踪、均值回归、动量、套利等）
- 基于机器学习的策略生成
- 参数化策略模板和框架
- 策略组合和混合生成
- 策略个性化定制
- 策略可行性评估
- 策略代码自动生成
"""

import numpy as np
import pandas as pd
import json
import ast
import inspect
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import random
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """策略类型枚举"""
    TREND_FOLLOWING = "趋势跟踪"
    MEAN_REVERSION = "均值回归"
    MOMENTUM = "动量策略"
    ARBITRAGE = "套利策略"
    BREAKOUT = "突破策略"
    PAIRS_TRADING = "配对交易"
    MARKET_MAKING = "做市策略"
    STATISTICAL_ARBITRAGE = "统计套利"
    ML_BASED = "机器学习策略"
    HYBRID = "混合策略"


class RiskLevel(Enum):
    """风险等级枚举"""
    CONSERVATIVE = "保守型"
    MODERATE = "稳健型"
    AGGRESSIVE = "激进型"
    VERY_AGGRESSIVE = "极激进型"


@dataclass
class StrategyParameters:
    """策略参数配置"""
    name: str
    strategy_type: StrategyType
    risk_level: RiskLevel
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""


@dataclass
class GeneratedStrategy:
    """生成的策略结果"""
    id: str
    name: str
    strategy_type: StrategyType
    parameters: StrategyParameters
    code: str
    performance_estimate: Dict[str, float]
    feasibility_score: float
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, parameters: StrategyParameters):
        self.parameters = parameters
        self.name = parameters.name
        self.strategy_type = parameters.strategy_type
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """获取参数空间定义"""
        pass


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成趋势跟踪信号"""
        signals = pd.Series(0, index=data.index)
        
        # 移动平均线策略
        short_ma = data['close'].rolling(window=5).mean()
        long_ma = data['close'].rolling(window=20).mean()
        
        signals[(short_ma > long_ma)] = 1  # 买入信号
        signals[(short_ma < long_ma)] = -1  # 卖出信号
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        return {
            'short_window': {'type': 'int', 'min': 3, 'max': 10},
            'long_window': {'type': 'int', 'min': 15, 'max': 30},
            'threshold': {'type': 'float', 'min': 0.01, 'max': 0.05}
        }


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成均值回归信号"""
        signals = pd.Series(0, index=data.index)
        
        # RSI策略
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals[rsi < 30] = 1  # 超卖，买入
        signals[rsi > 70] = -1  # 超买，卖出
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        return {
            'rsi_period': {'type': 'int', 'min': 10, 'max': 20},
            'oversold': {'type': 'int', 'min': 20, 'max': 35},
            'overbought': {'type': 'int', 'min': 65, 'max': 80}
        }


class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成动量信号"""
        signals = pd.Series(0, index=data.index)
        
        # 价格动量
        returns = data['close'].pct_change()
        momentum = returns.rolling(window=10).mean()
        
        signals[momentum > 0.02] = 1  # 强动量，买入
        signals[momentum < -0.02] = -1  # 负动量，卖出
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        return {
            'momentum_period': {'type': 'int', 'min': 5, 'max': 15},
            'threshold': {'type': 'float', 'min': 0.01, 'max': 0.05}
        }


class ArbitrageStrategy(BaseStrategy):
    """套利策略"""
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成套利信号"""
        signals = pd.Series(0, index=data.index)
        
        # 简单的价格差异套利
        if 'high' in data.columns and 'low' in data.columns:
            price_range = data['high'] - data['low']
            signals[price_range > price_range.quantile(0.8)] = 1
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        return {
            'lookback_period': {'type': 'int', 'min': 20, 'max': 50},
            'threshold_percentile': {'type': 'float', 'min': 0.7, 'max': 0.9}
        }


class StrategyTemplate:
    """策略模板类"""
    
    def __init__(self):
        self.templates = {
            StrategyType.TREND_FOLLOWING: TrendFollowingStrategy,
            StrategyType.MEAN_REVERSION: MeanReversionStrategy,
            StrategyType.MOMENTUM: MomentumStrategy,
            StrategyType.ARBITRAGE: ArbitrageStrategy,
        }
    
    def create_strategy(self, strategy_type: StrategyType, parameters: Dict[str, Any]) -> BaseStrategy:
        """创建策略实例"""
        if strategy_type in self.templates:
            strategy_params = StrategyParameters(
                name=f"{strategy_type.value}_strategy",
                strategy_type=strategy_type,
                risk_level=RiskLevel.MODERATE,
                parameters=parameters
            )
            return self.templates[strategy_type](strategy_params)
        else:
            raise ValueError(f"不支持的策略类型: {strategy_type}")
    
    def get_available_strategies(self) -> List[StrategyType]:
        """获取可用策略类型"""
        return list(self.templates.keys())


class GeneticAlgorithm:
    """遗传算法优化器"""
    
    def __init__(self, population_size: int = 50, generations: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(self, strategy_type: StrategyType, parameter_space: Dict[str, Any], 
                fitness_function, target_data: pd.DataFrame) -> Dict[str, Any]:
        """使用遗传算法优化策略参数"""
        logger.info(f"开始遗传算法优化，策略类型: {strategy_type.value}")
        
        # 初始化种群
        population = self._initialize_population(parameter_space)
        best_fitness = float('-inf')
        best_params = None
        
        for generation in range(self.generations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                fitness = fitness_function(individual, target_data)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = individual.copy()
            
            # 选择、交叉、变异
            population = self._evolve_population(population, fitness_scores, parameter_space)
            
            if generation % 20 == 0:
                logger.info(f"第 {generation} 代，最佳适应度: {best_fitness:.4f}")
        
        logger.info(f"遗传算法优化完成，最佳适应度: {best_fitness:.4f}")
        return best_params
    
    def _initialize_population(self, parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param_name, param_config in parameter_space.items():
                if param_config['type'] == 'int':
                    individual[param_name] = random.randint(param_config['min'], param_config['max'])
                elif param_config['type'] == 'float':
                    individual[param_name] = random.uniform(param_config['min'], param_config['max'])
            population.append(individual)
        return population
    
    def _evolve_population(self, population: List[Dict[str, Any]], fitness_scores: List[float],
                          parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """进化种群"""
        # 选择（轮盘赌选择）
        new_population = []
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # 如果所有适应度都是0，随机选择
            selected_indices = random.choices(range(len(population)), k=len(population))
        else:
            probabilities = [f / total_fitness for f in fitness_scores]
            selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        
        for i in range(0, len(population), 2):
            parent1 = population[selected_indices[i % len(selected_indices)]].copy()
            
            # 交叉
            if random.random() < self.crossover_rate and i + 1 < len(population):
                parent2 = population[selected_indices[(i + 1) % len(selected_indices)]].copy()
                child1, child2 = self._crossover(parent1, parent2, parameter_space)
            else:
                child1 = parent1.copy()
                if i + 1 < len(population):
                    child2 = population[selected_indices[(i + 1) % len(selected_indices)]].copy()
                else:
                    child2 = parent1.copy()
            
            # 变异
            child1 = self._mutate(child1, parameter_space)
            child2 = self._mutate(child2, parameter_space)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                   parameter_space: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """交叉操作"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param_name in parameter_space.keys():
            if random.random() < 0.5:
                child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()
        
        for param_name, param_config in parameter_space.items():
            if random.random() < self.mutation_rate:
                if param_config['type'] == 'int':
                    mutated[param_name] = random.randint(param_config['min'], param_config['max'])
                elif param_config['type'] == 'float':
                    mutated[param_name] = random.uniform(param_config['min'], param_config['max'])
        
        return mutated


class DeepRLStrategy(BaseStrategy):
    """基于深度强化学习的策略"""
    
    def __init__(self, parameters: StrategyParameters):
        super().__init__(parameters)
        # 简化的Q-learning实现
        self.q_table = {}
        self.learning_rate = parameters.parameters.get('learning_rate', 0.1)
        self.discount_factor = parameters.parameters.get('discount_factor', 0.9)
        self.epsilon = parameters.parameters.get('epsilon', 0.1)
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """使用强化学习生成信号"""
        signals = pd.Series(0, index=data.index)
        
        # 简化的状态定义：价格变化、波动率等
        for i in range(len(data)):
            if i < 20:  # 需要足够的历史数据
                continue
                
            # 构建状态
            price_change = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
            volatility = data['close'].iloc[i-20:i].pct_change().std()
            state = (round(price_change, 2), round(volatility, 4))
            
            # 选择动作（使用epsilon-greedy策略）
            if random.random() < self.epsilon:
                action = random.choice([-1, 0, 1])
            else:
                if state in self.q_table:
                    action = max(self.q_table[state], key=self.q_table[state].get)
                else:
                    action = 0
            
            signals.iloc[i] = action
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, Any]:
        return {
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
            'discount_factor': {'type': 'float', 'min': 0.8, 'max': 0.99},
            'epsilon': {'type': 'float', 'min': 0.01, 'max': 0.2}
        }


class StrategyEvaluator:
    """策略评估器"""
    
    def __init__(self):
        self.metrics = {
            'total_return': self._calculate_total_return,
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown,
            'win_rate': self._calculate_win_rate,
            'profit_factor': self._calculate_profit_factor,
            'calmar_ratio': self._calculate_calmar_ratio
        }
    
    def evaluate_strategy(self, strategy: BaseStrategy, data: pd.DataFrame, 
                         initial_capital: float = 100000) -> Dict[str, float]:
        """评估策略性能"""
        signals = strategy.generate_signal(data)
        returns = data['close'].pct_change()
        
        # 计算策略收益
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)
        
        # 计算累计收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 计算各种指标
        performance = {}
        for metric_name, metric_func in self.metrics.items():
            try:
                performance[metric_name] = metric_func(strategy_returns, cumulative_returns)
            except Exception as e:
                logger.warning(f"计算指标 {metric_name} 时出错: {e}")
                performance[metric_name] = 0.0
        
        return performance
    
    def _calculate_total_return(self, returns: pd.Series, cumulative_returns: pd.Series) -> float:
        """计算总收益率"""
        return (cumulative_returns.iloc[-1] - 1) * 100
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, cumulative_returns: pd.Series) -> float:
        """计算夏普比率"""
        if returns.std() == 0:
            return 0
        excess_returns = returns - 0.02/252  # 假设无风险利率为2%
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: pd.Series, cumulative_returns: pd.Series) -> float:
        """计算最大回撤"""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min() * 100
    
    def _calculate_win_rate(self, returns: pd.Series, cumulative_returns: pd.Series) -> float:
        """计算胜率"""
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        if total_trades == 0:
            return 0
        return (winning_trades / total_trades) * 100
    
    def _calculate_profit_factor(self, returns: pd.Series, cumulative_returns: pd.Series) -> float:
        """计算盈亏比"""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss
    
    def _calculate_calmar_ratio(self, returns: pd.Series, cumulative_returns: pd.Series) -> float:
        """计算卡玛比率"""
        total_return = self._calculate_total_return(returns, cumulative_returns)
        max_drawdown = abs(self._calculate_max_drawdown(returns, cumulative_returns))
        if max_drawdown == 0:
            return 0
        return total_return / max_drawdown
    
    def assess_feasibility(self, strategy: BaseStrategy, data: pd.DataFrame) -> float:
        """评估策略可行性"""
        performance = self.evaluate_strategy(strategy, data)
        
        # 简单的可行性评分算法
        score = 0
        
        # 总收益率权重30%
        if performance['total_return'] > 10:
            score += 30
        elif performance['total_return'] > 5:
            score += 20
        elif performance['total_return'] > 0:
            score += 10
        
        # 夏普比率权重25%
        if performance['sharpe_ratio'] > 1.5:
            score += 25
        elif performance['sharpe_ratio'] > 1.0:
            score += 20
        elif performance['sharpe_ratio'] > 0.5:
            score += 15
        
        # 最大回撤权重20%
        if abs(performance['max_drawdown']) < 5:
            score += 20
        elif abs(performance['max_drawdown']) < 10:
            score += 15
        elif abs(performance['max_drawdown']) < 20:
            score += 10
        
        # 胜率权重15%
        if performance['win_rate'] > 60:
            score += 15
        elif performance['win_rate'] > 50:
            score += 10
        elif performance['win_rate'] > 40:
            score += 5
        
        # 盈亏比权重10%
        if performance['profit_factor'] > 2.0:
            score += 10
        elif performance['profit_factor'] > 1.5:
            score += 7
        elif performance['profit_factor'] > 1.2:
            score += 5
        
        return min(score, 100)  # 最高100分


class CodeGenerator:
    """策略代码生成器"""
    
    def __init__(self):
        self.template_engine = StrategyTemplateEngine()
    
    def generate_strategy_code(self, strategy: BaseStrategy, strategy_type: StrategyType,
                              parameters: Dict[str, Any]) -> str:
        """生成策略代码"""
        if strategy_type == StrategyType.TREND_FOLLOWING:
            return self._generate_trend_following_code(parameters)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return self._generate_mean_reversion_code(parameters)
        elif strategy_type == StrategyType.MOMENTUM:
            return self._generate_momentum_code(parameters)
        elif strategy_type == StrategyType.ARBITRAGE:
            return self._generate_arbitrage_code(parameters)
        elif strategy_type == StrategyType.ML_BASED:
            return self._generate_ml_code(parameters)
        else:
            return self._generate_generic_code(strategy_type, parameters)
    
    def _generate_trend_following_code(self, parameters: Dict[str, Any]) -> str:
        """生成趋势跟踪策略代码"""
        template = '''
import pandas as pd
import numpy as np

class TrendFollowingStrategy:
    """
    趋势跟踪策略
    参数: {parameters}
    """
    
    def __init__(self):
        self.short_window = {short_window}
        self.long_window = {long_window}
        self.threshold = {threshold}
        
    def generate_signals(self, data):
        """
        生成交易信号
        
        Args:
            data: DataFrame，包含价格数据
            
        Returns:
            signals: Series，交易信号
        """
        signals = pd.Series(0, index=data.index, name='signals')
        
        # 计算移动平均线
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        # 生成信号
        data['signal'] = 0
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
        data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
        
        # 计算持仓
        data['position'] = data['signal'].shift(1)
        
        return data['position']
    
    def backtest(self, data, initial_capital=100000):
        """
        回测策略
        
        Args:
            data: DataFrame，价格数据
            initial_capital: float，初始资金
            
        Returns:
            results: dict，回测结果
        """
        signals = self.generate_signals(data)
        returns = data['close'].pct_change()
        strategy_returns = signals * returns
        
        # 计算累计收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 计算性能指标
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max() - 1).min()) * 100
        
        return {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }}

# 使用示例
if __name__ == "__main__":
    # 创建策略实例
    strategy = TrendFollowingStrategy()
    
    # 假设有价格数据
    # data = pd.read_csv('price_data.csv', index_col=0, parse_dates=True)
    # results = strategy.backtest(data)
    # print(f"总收益率: {{results['total_return']:.2f}}%")
'''
        
        return template.format(
            parameters=parameters,
            short_window=parameters.get('short_window', 5),
            long_window=parameters.get('long_window', 20),
            threshold=parameters.get('threshold', 0.02)
        )
    
    def _generate_mean_reversion_code(self, parameters: Dict[str, Any]) -> str:
        """生成均值回归策略代码"""
        template = '''
import pandas as pd
import numpy as np

class MeanReversionStrategy:
    """
    均值回归策略
    参数: {parameters}
    """
    
    def __init__(self):
        self.rsi_period = {rsi_period}
        self.oversold = {oversold}
        self.overbought = {overbought}
        
    def calculate_rsi(self, prices):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data):
        """生成交易信号"""
        signals = pd.Series(0, index=data.index, name='signals')
        
        # 计算RSI
        data['rsi'] = self.calculate_rsi(data['close'])
        
        # 生成信号
        data['signal'] = 0
        data.loc[data['rsi'] < self.oversold, 'signal'] = 1  # 超卖，买入
        data.loc[data['rsi'] > self.overbought, 'signal'] = -1  # 超买，卖出
        
        # 计算持仓
        data['position'] = data['signal'].shift(1)
        
        return data['position']
    
    def backtest(self, data, initial_capital=100000):
        """回测策略"""
        signals = self.generate_signals(data)
        returns = data['close'].pct_change()
        strategy_returns = signals * returns
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max() - 1).min()) * 100
        
        return {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }}

if __name__ == "__main__":
    strategy = MeanReversionStrategy()
'''
        
        return template.format(
            parameters=parameters,
            rsi_period=parameters.get('rsi_period', 14),
            oversold=parameters.get('oversold', 30),
            overbought=parameters.get('overbought', 70)
        )
    
    def _generate_momentum_code(self, parameters: Dict[str, Any]) -> str:
        """生成动量策略代码"""
        template = '''
import pandas as pd
import numpy as np

class MomentumStrategy:
    """
    动量策略
    参数: {parameters}
    """
    
    def __init__(self):
        self.momentum_period = {momentum_period}
        self.threshold = {threshold}
        
    def generate_signals(self, data):
        """生成交易信号"""
        signals = pd.Series(0, index=data.index, name='signals')
        
        # 计算动量
        returns = data['close'].pct_change()
        momentum = returns.rolling(window=self.momentum_period).mean()
        
        # 生成信号
        data['signal'] = 0
        data.loc[momentum > self.threshold, 'signal'] = 1  # 强动量，买入
        data.loc[momentum < -self.threshold, 'signal'] = -1  # 负动量，卖出
        
        data['position'] = data['signal'].shift(1)
        
        return data['position']
    
    def backtest(self, data, initial_capital=100000):
        """回测策略"""
        signals = self.generate_signals(data)
        returns = data['close'].pct_change()
        strategy_returns = signals * returns
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max() - 1).min()) * 100
        
        return {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }}

if __name__ == "__main__":
    strategy = MomentumStrategy()
'''
        
        return template.format(
            parameters=parameters,
            momentum_period=parameters.get('momentum_period', 10),
            threshold=parameters.get('threshold', 0.02)
        )
    
    def _generate_arbitrage_code(self, parameters: Dict[str, Any]) -> str:
        """生成套利策略代码"""
        template = '''
import pandas as pd
import numpy as np

class ArbitrageStrategy:
    """
    套利策略
    参数: {parameters}
    """
    
    def __init__(self):
        self.lookback_period = {lookback_period}
        self.threshold_percentile = {threshold_percentile}
        
    def generate_signals(self, data):
        """生成交易信号"""
        signals = pd.Series(0, index=data.index, name='signals')
        
        # 计算价格范围
        price_range = data['high'] - data['low']
        threshold = price_range.rolling(window=self.lookback_period).quantile(self.threshold_percentile)
        
        # 生成信号
        data['signal'] = 0
        data.loc[price_range > threshold, 'signal'] = 1
        
        data['position'] = data['signal'].shift(1)
        
        return data['position']
    
    def backtest(self, data, initial_capital=100000):
        """回测策略"""
        signals = self.generate_signals(data)
        returns = data['close'].pct_change()
        strategy_returns = signals * returns
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max() - 1).min()) * 100
        
        return {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }}

if __name__ == "__main__":
    strategy = ArbitrageStrategy()
'''
        
        return template.format(
            parameters=parameters,
            lookback_period=parameters.get('lookback_period', 30),
            threshold_percentile=parameters.get('threshold_percentile', 0.8)
        )
    
    def _generate_ml_code(self, parameters: Dict[str, Any]) -> str:
        """生成机器学习策略代码"""
        template = '''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class MLStrategy:
    """
    机器学习策略
    参数: {parameters}
    """
    
    def __init__(self):
        self.lookback = {lookback}
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def create_features(self, data):
        """创建特征"""
        features = pd.DataFrame()
        
        # 技术指标特征
        features['rsi'] = self.calculate_rsi(data['close'])
        features['macd'], features['macd_signal'] = self.calculate_macd(data['close'])
        features['bb_upper'], features['bb_lower'] = self.calculate_bollinger_bands(data['close'])
        
        # 价格特征
        features['returns'] = data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        return features.dropna()
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices):
        """计算MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def calculate_bollinger_bands(self, prices, period=20):
        """计算布林带"""
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper = rolling_mean + (rolling_std * 2)
        lower = rolling_mean - (rolling_std * 2)
        return upper, lower
    
    def create_labels(self, data):
        """创建标签（未来收益率的正负）"""
        future_returns = data['close'].shift(-5) / data['close'] - 1
        labels = (future_returns > 0).astype(int)
        return labels[:-5]  # 移除最后5个NaN值
    
    def train_model(self, data):
        """训练模型"""
        features = self.create_features(data)
        labels = self.create_labels(data)
        
        # 确保特征和标签长度一致
        min_length = min(len(features), len(labels))
        features = features.iloc[:min_length]
        labels = labels.iloc[:min_length]
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        print("模型评估报告:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def generate_signals(self, data):
        """生成交易信号"""
        features = self.create_features(data)
        
        # 预测
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        signals = pd.Series(0, index=data.index[:len(features)], name='signals')
        signals[probabilities > 0.6] = 1  # 买入
        signals[probabilities < 0.4] = -1  # 卖出
        
        return signals
    
    def backtest(self, data, initial_capital=100000):
        """回测策略"""
        signals = self.generate_signals(data)
        returns = data['close'].pct_change()
        strategy_returns = signals * returns
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max() - 1).min()) * 100
        
        return {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }}

if __name__ == "__main__":
    strategy = MLStrategy()
'''
        
        return template.format(
            parameters=parameters,
            lookback=parameters.get('lookback', 20)
        )
    
    def _generate_generic_code(self, strategy_type: StrategyType, parameters: Dict[str, Any]) -> str:
        """生成通用策略代码"""
        template = '''
import pandas as pd
import numpy as np

class GenericStrategy:
    """
    通用策略模板
    策略类型: {strategy_type}
    参数: {parameters}
    """
    
    def __init__(self):
        # 策略参数初始化
        {param_initialization}
        
    def generate_signals(self, data):
        """生成交易信号"""
        signals = pd.Series(0, index=data.index, name='signals')
        
        # 在这里实现具体的策略逻辑
        # 示例：简单的移动平均策略
        data['ma_short'] = data['close'].rolling(window=5).mean()
        data['ma_long'] = data['close'].rolling(window=20).mean()
        
        data['signal'] = 0
        data.loc[data['ma_short'] > data['ma_long'], 'signal'] = 1
        data.loc[data['ma_short'] < data['ma_long'], 'signal'] = -1
        
        data['position'] = data['signal'].shift(1)
        
        return data['position']
    
    def backtest(self, data, initial_capital=100000):
        """回测策略"""
        signals = self.generate_signals(data)
        returns = data['close'].pct_change()
        strategy_returns = signals * returns
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max() - 1).min()) * 100
        
        return {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }}

if __name__ == "__main__":
    strategy = GenericStrategy()
'''
        
        param_init = "\n        ".join([f"self.{k} = {v}" for k, v in parameters.items()])
        
        return template.format(
            strategy_type=strategy_type.value,
            parameters=parameters,
            param_initialization=param_init if param_init else "# 无特定参数"
        )


class StrategyTemplateEngine:
    """策略模板引擎"""
    
    def __init__(self):
        self.strategy_registry = {}
    
    def register_strategy(self, strategy_type: StrategyType, template_class):
        """注册策略模板"""
        self.strategy_registry[strategy_type] = template_class
    
    def get_strategy_template(self, strategy_type: StrategyType):
        """获取策略模板"""
        return self.strategy_registry.get(strategy_type)


class StrategyGenerator:
    """主要的策略生成器类"""
    
    def __init__(self):
        self.template = StrategyTemplate()
        self.evaluator = StrategyEvaluator()
        self.code_generator = CodeGenerator()
        self.genetic_algorithm = GeneticAlgorithm()
        
        # 注册深度强化学习策略
        self.template.templates[StrategyType.ML_BASED] = DeepRLStrategy
        
        logger.info("策略生成器初始化完成")
    
    def generate_strategy(self, strategy_type: StrategyType, risk_level: RiskLevel,
                         market_data: pd.DataFrame, custom_parameters: Optional[Dict[str, Any]] = None,
                         use_optimization: bool = True, optimization_method: str = 'genetic') -> GeneratedStrategy:
        """
        生成交易策略
        
        Args:
            strategy_type: 策略类型
            risk_level: 风险等级
            market_data: 市场数据
            custom_parameters: 自定义参数
            use_optimization: 是否使用优化
            optimization_method: 优化方法 ('genetic', 'random', 'grid')
            
        Returns:
            GeneratedStrategy: 生成的策略
        """
        logger.info(f"开始生成策略: {strategy_type.value}, 风险等级: {risk_level.value}")
        
        # 创建策略参数
        if custom_parameters:
            parameters = StrategyParameters(
                name=f"{strategy_type.value}_custom",
                strategy_type=strategy_type,
                risk_level=risk_level,
                parameters=custom_parameters
            )
        else:
            # 生成默认参数
            parameters = self._generate_default_parameters(strategy_type, risk_level)
        
        # 参数优化
        if use_optimization:
            parameters.parameters = self._optimize_parameters(strategy_type, parameters, market_data, optimization_method)
        
        # 创建策略实例
        strategy = self.template.create_strategy(strategy_type, parameters.parameters)
        
        # 评估策略
        performance = self.evaluator.evaluate_strategy(strategy, market_data)
        feasibility_score = self.evaluator.assess_feasibility(strategy, market_data)
        
        # 生成策略代码
        code = self.code_generator.generate_strategy_code(strategy, strategy_type, parameters.parameters)
        
        # 风险评估
        risk_assessment = self._assess_risk(strategy, performance)
        
        # 生成建议
        recommendations = self._generate_recommendations(strategy_type, performance, risk_assessment)
        
        # 创建生成的策略
        generated_strategy = GeneratedStrategy(
            id=f"{strategy_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"{strategy_type.value}策略",
            strategy_type=strategy_type,
            parameters=parameters,
            code=code,
            performance_estimate=performance,
            feasibility_score=feasibility_score,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
        
        logger.info(f"策略生成完成，可行性评分: {feasibility_score:.2f}")
        return generated_strategy
    
    def generate_hybrid_strategy(self, strategy_types: List[StrategyType], risk_level: RiskLevel,
                               market_data: pd.DataFrame, weights: Optional[List[float]] = None) -> GeneratedStrategy:
        """
        生成混合策略
        
        Args:
            strategy_types: 策略类型列表
            risk_level: 风险等级
            market_data: 市场数据
            weights: 权重列表
            
        Returns:
            GeneratedStrategy: 生成的混合策略
        """
        logger.info(f"生成混合策略，包含策略: {[s.value for s in strategy_types]}")
        
        if weights is None:
            weights = [1.0 / len(strategy_types)] * len(strategy_types)
        
        if len(weights) != len(strategy_types):
            raise ValueError("权重数量必须与策略数量相同")
        
        # 生成各个子策略
        sub_strategies = []
        for strategy_type in strategy_types:
            sub_strategy = self.generate_strategy(strategy_type, risk_level, market_data, use_optimization=False)
            sub_strategies.append(sub_strategy)
        
        # 组合策略逻辑（简化实现）
        combined_performance = {}
        for metric in sub_strategies[0].performance_estimate.keys():
            combined_performance[metric] = sum(
                strategy.performance_estimate[metric] * weight 
                for strategy, weight in zip(sub_strategies, weights)
            )
        
        # 组合可行性评分
        combined_feasibility = sum(
            strategy.feasibility_score * weight 
            for strategy, weight in zip(sub_strategies, weights)
        )
        
        # 生成混合策略代码
        hybrid_code = self._generate_hybrid_code(sub_strategies, weights)
        
        # 创建混合策略
        hybrid_strategy = GeneratedStrategy(
            id=f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="混合策略",
            strategy_type=StrategyType.HYBRID,
            parameters=StrategyParameters(
                name="hybrid_strategy",
                strategy_type=StrategyType.HYBRID,
                risk_level=risk_level,
                parameters={'component_strategies': strategy_types, 'weights': weights}
            ),
            code=hybrid_code,
            performance_estimate=combined_performance,
            feasibility_score=combined_feasibility,
            risk_assessment=self._assess_hybrid_risk(sub_strategies, weights),
            recommendations=self._generate_hybrid_recommendations(sub_strategies, weights)
        )
        
        return hybrid_strategy
    
    def personalize_strategy(self, base_strategy: GeneratedStrategy, user_preferences: Dict[str, Any],
                           market_data: pd.DataFrame) -> GeneratedStrategy:
        """
        个性化定制策略
        
        Args:
            base_strategy: 基础策略
            user_preferences: 用户偏好
            market_data: 市场数据
            
        Returns:
            GeneratedStrategy: 个性化策略
        """
        logger.info("开始个性化策略定制")
        
        # 根据用户偏好调整参数
        personalized_params = base_strategy.parameters.parameters.copy()
        
        # 风险偏好调整
        if 'risk_tolerance' in user_preferences:
            risk_tolerance = user_preferences['risk_tolerance']
            if risk_tolerance == 'conservative':
                # 保守型：降低参数敏感度
                personalized_params['threshold'] = personalized_params.get('threshold', 0.02) * 0.5
            elif risk_tolerance == 'aggressive':
                # 激进型：增加参数敏感度
                personalized_params['threshold'] = personalized_params.get('threshold', 0.02) * 1.5
        
        # 交易频率偏好
        if 'trading_frequency' in user_preferences:
            freq = user_preferences['trading_frequency']
            if freq == 'low':
                personalized_params['short_window'] = personalized_params.get('short_window', 5) * 2
            elif freq == 'high':
                personalized_params['short_window'] = personalized_params.get('short_window', 5) * 0.5
        
        # 生成个性化策略
        personalized_strategy = self.generate_strategy(
            base_strategy.strategy_type,
            base_strategy.parameters.risk_level,
            market_data,
            custom_parameters=personalized_params,
            use_optimization=True
        )
        
        # 添加个性化标识
        personalized_strategy.name = f"{base_strategy.name}_个性化"
        personalized_strategy.id = f"{base_strategy.id}_personalized"
        
        return personalized_strategy
    
    def _generate_default_parameters(self, strategy_type: StrategyType, risk_level: RiskLevel) -> StrategyParameters:
        """生成默认参数"""
        default_params = {
            StrategyType.TREND_FOLLOWING: {'short_window': 5, 'long_window': 20, 'threshold': 0.02},
            StrategyType.MEAN_REVERSION: {'rsi_period': 14, 'oversold': 30, 'overbought': 70},
            StrategyType.MOMENTUM: {'momentum_period': 10, 'threshold': 0.02},
            StrategyType.ARBITRAGE: {'lookback_period': 30, 'threshold_percentile': 0.8},
            StrategyType.ML_BASED: {'learning_rate': 0.1, 'discount_factor': 0.9, 'epsilon': 0.1}
        }
        
        params = default_params.get(strategy_type, {})
        
        # 根据风险等级调整参数
        if risk_level == RiskLevel.CONSERVATIVE:
            # 保守型：更宽松的阈值
            if 'threshold' in params:
                params['threshold'] *= 0.7
        elif risk_level == RiskLevel.AGGRESSIVE:
            # 激进型：更严格的阈值
            if 'threshold' in params:
                params['threshold'] *= 1.3
        
        return StrategyParameters(
            name=f"{strategy_type.value}_default",
            strategy_type=strategy_type,
            risk_level=risk_level,
            parameters=params
        )
    
    def _optimize_parameters(self, strategy_type: StrategyType, parameters: StrategyParameters,
                           market_data: pd.DataFrame, method: str) -> Dict[str, Any]:
        """优化策略参数"""
        strategy = self.template.create_strategy(strategy_type, parameters.parameters)
        parameter_space = strategy.get_parameter_space()
        
        if method == 'genetic':
            # 定义适应度函数
            def fitness_function(params, data):
                try:
                    temp_strategy = self.template.create_strategy(strategy_type, params)
                    performance = self.evaluator.evaluate_strategy(temp_strategy, data)
                    # 综合评分：收益-风险惩罚
                    score = performance['total_return'] - abs(performance['max_drawdown']) * 0.5
                    return max(score, 0)  # 确保分数非负
                except:
                    return 0
            
            best_params = self.genetic_algorithm.optimize(
                strategy_type, parameter_space, fitness_function, market_data
            )
            return best_params
        
        elif method == 'random':
            # 随机搜索
            best_score = float('-inf')
            best_params = parameters.parameters.copy()
            
            for _ in range(50):  # 随机尝试50次
                random_params = {}
                for param_name, param_config in parameter_space.items():
                    if param_config['type'] == 'int':
                        random_params[param_name] = random.randint(param_config['min'], param_config['max'])
                    elif param_config['type'] == 'float':
                        random_params[param_name] = random.uniform(param_config['min'], param_config['max'])
                
                try:
                    temp_strategy = self.template.create_strategy(strategy_type, random_params)
                    performance = self.evaluator.evaluate_strategy(temp_strategy, market_data)
                    score = performance['total_return'] - abs(performance['max_drawdown']) * 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_params = random_params
                except:
                    continue
            
            return best_params
        
        else:
            logger.warning(f"未知的优化方法: {method}，使用默认参数")
            return parameters.parameters
    
    def _assess_risk(self, strategy: BaseStrategy, performance: Dict[str, float]) -> Dict[str, Any]:
        """评估策略风险"""
        risk_assessment = {
            'volatility_risk': 'medium',
            'drawdown_risk': 'medium',
            'market_risk': 'medium',
            'overall_risk_level': 'medium'
        }
        
        # 波动率风险
        if performance.get('sharpe_ratio', 0) < 0.5:
            risk_assessment['volatility_risk'] = 'high'
        elif performance.get('sharpe_ratio', 0) > 1.5:
            risk_assessment['volatility_risk'] = 'low'
        
        # 回撤风险
        if abs(performance.get('max_drawdown', 0)) > 20:
            risk_assessment['drawdown_risk'] = 'high'
        elif abs(performance.get('max_drawdown', 0)) < 10:
            risk_assessment['drawdown_risk'] = 'low'
        
        # 综合风险等级
        risk_scores = {
            'volatility_risk': {'low': 1, 'medium': 2, 'high': 3},
            'drawdown_risk': {'low': 1, 'medium': 2, 'high': 3},
            'market_risk': {'low': 1, 'medium': 2, 'high': 3}
        }
        
        avg_risk = (
            risk_scores['volatility_risk'][risk_assessment['volatility_risk']] +
            risk_scores['drawdown_risk'][risk_assessment['drawdown_risk']] +
            risk_scores['market_risk'][risk_assessment['market_risk']]
        ) / 3
        
        if avg_risk < 1.5:
            risk_assessment['overall_risk_level'] = 'low'
        elif avg_risk > 2.5:
            risk_assessment['overall_risk_level'] = 'high'
        
        return risk_assessment
    
    def _generate_recommendations(self, strategy_type: StrategyType, performance: Dict[str, float],
                                risk_assessment: Dict[str, Any]) -> List[str]:
        """生成策略建议"""
        recommendations = []
        
        # 性能相关建议
        if performance.get('total_return', 0) < 5:
            recommendations.append("考虑调整策略参数以提高收益率")
        
        if performance.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("夏普比率较低，建议优化风险调整后收益")
        
        if abs(performance.get('max_drawdown', 0)) > 15:
            recommendations.append("最大回撤较大，建议增加止损机制")
        
        if performance.get('win_rate', 0) < 50:
            recommendations.append("胜率较低，建议优化入场和出场条件")
        
        # 风险相关建议
        if risk_assessment.get('overall_risk_level') == 'high':
            recommendations.append("策略整体风险较高，建议降低仓位或增加对冲")
        
        if risk_assessment.get('volatility_risk') == 'high':
            recommendations.append("波动率风险较高，建议使用波动率过滤")
        
        # 策略特定建议
        if strategy_type == StrategyType.TREND_FOLLOWING:
            recommendations.append("趋势跟踪策略在震荡市场中表现可能较差")
        elif strategy_type == StrategyType.MEAN_REVERSION:
            recommendations.append("均值回归策略在趋势市场中需要谨慎使用")
        elif strategy_type == StrategyType.MOMENTUM:
            recommendations.append("动量策略需要关注市场流动性")
        
        # 通用建议
        recommendations.append("建议进行充分的回测验证")
        recommendations.append("考虑在不同市场环境下测试策略稳定性")
        recommendations.append("定期重新评估和调整策略参数")
        
        return recommendations
    
    def _generate_hybrid_code(self, sub_strategies: List[GeneratedStrategy], weights: List[float]) -> str:
        """生成混合策略代码"""
        template = '''
import pandas as pd
import numpy as np

class HybridStrategy:
    """
    混合策略
    组件策略: {component_names}
    权重: {weights}
    """
    
    def __init__(self):
        self.weights = {weights}
        self.component_strategies = {component_count}
        
    def generate_signals(self, data):
        """生成混合交易信号"""
        signals = pd.Series(0, index=data.index, name='signals')
        
        # 这里应该包含各个组件策略的信号生成逻辑
        # 为简化示例，使用简单的平均方法
        
        # 示例：假设有多个策略信号
        # strategy1_signals = self.strategy1.generate_signals(data)
        # strategy2_signals = self.strategy2.generate_signals(data)
        # ...
        
        # 组合信号（加权平均）
        # combined_signal = (strategy1_signals * self.weights[0] + 
        #                   strategy2_signals * self.weights[1] + ...)
        
        # 简化的实现
        signals = np.random.choice([-1, 0, 1], size=len(data), p=[0.1, 0.8, 0.1])
        
        return pd.Series(signals, index=data.index)
    
    def backtest(self, data, initial_capital=100000):
        """回测混合策略"""
        signals = self.generate_signals(data)
        returns = data['close'].pct_change()
        strategy_returns = signals * returns
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max() - 1).min()) * 100
        
        return {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }}

if __name__ == "__main__":
    strategy = HybridStrategy()
'''
        
        component_names = [s.name for s in sub_strategies]
        
        return template.format(
            component_names=component_names,
            weights=weights,
            component_count=len(sub_strategies)
        )
    
    def _assess_hybrid_risk(self, sub_strategies: List[GeneratedStrategy], weights: List[float]) -> Dict[str, Any]:
        """评估混合策略风险"""
        # 简单的风险组合评估
        total_risk = 0
        for strategy, weight in zip(sub_strategies, weights):
            strategy_risk_score = {
                'low': 1, 'medium': 2, 'high': 3
            }.get(strategy.risk_assessment.get('overall_risk_level', 'medium'), 2)
            total_risk += strategy_risk_score * weight
        
        avg_risk = total_risk / len(sub_strategies)
        
        if avg_risk < 1.5:
            overall_risk = 'low'
        elif avg_risk > 2.5:
            overall_risk = 'high'
        else:
            overall_risk = 'medium'
        
        return {
            'overall_risk_level': overall_risk,
            'component_risks': [s.risk_assessment.get('overall_risk_level', 'medium') for s in sub_strategies],
            'diversification_benefit': 'high' if len(sub_strategies) > 2 else 'medium'
        }
    
    def _generate_hybrid_recommendations(self, sub_strategies: List[GeneratedStrategy], 
                                       weights: List[float]) -> List[str]:
        """生成混合策略建议"""
        recommendations = []
        
        recommendations.append("混合策略通过多样化降低单一策略风险")
        recommendations.append("建议定期重新评估各组件策略的表现")
        recommendations.append("根据市场环境动态调整策略权重")
        
        # 检查组件策略的多样性
        strategy_types = set(s.strategy_type for s in sub_strategies)
        if len(strategy_types) < len(sub_strategies):
            recommendations.append("建议增加策略类型的多样性以提高分散化效果")
        
        # 权重建议
        if max(weights) > 0.7:
            recommendations.append("某些策略权重过高，建议重新平衡以降低集中度风险")
        
        return recommendations
    
    def save_strategy(self, strategy: GeneratedStrategy, filepath: str):
        """保存策略到文件"""
        strategy_dict = {
            'id': strategy.id,
            'name': strategy.name,
            'strategy_type': strategy.strategy_type.value,
            'parameters': {
                'name': strategy.parameters.name,
                'strategy_type': strategy.parameters.strategy_type.value,
                'risk_level': strategy.parameters.risk_level.value,
                'parameters': strategy.parameters.parameters,
                'description': strategy.parameters.description
            },
            'code': strategy.code,
            'performance_estimate': strategy.performance_estimate,
            'feasibility_score': strategy.feasibility_score,
            'risk_assessment': strategy.risk_assessment,
            'recommendations': strategy.recommendations,
            'created_at': strategy.created_at.isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(strategy_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"策略已保存到: {filepath}")
    
    def load_strategy(self, filepath: str) -> GeneratedStrategy:
        """从文件加载策略"""
        with open(filepath, 'r', encoding='utf-8') as f:
            strategy_dict = json.load(f)
        
        # 重建策略对象
        parameters = StrategyParameters(
            name=strategy_dict['parameters']['name'],
            strategy_type=StrategyType(strategy_dict['parameters']['strategy_type']),
            risk_level=RiskLevel(strategy_dict['parameters']['risk_level']),
            parameters=strategy_dict['parameters']['parameters'],
            description=strategy_dict['parameters']['description']
        )
        
        strategy = GeneratedStrategy(
            id=strategy_dict['id'],
            name=strategy_dict['name'],
            strategy_type=StrategyType(strategy_dict['strategy_type']),
            parameters=parameters,
            code=strategy_dict['code'],
            performance_estimate=strategy_dict['performance_estimate'],
            feasibility_score=strategy_dict['feasibility_score'],
            risk_assessment=strategy_dict['risk_assessment'],
            recommendations=strategy_dict['recommendations'],
            created_at=datetime.fromisoformat(strategy_dict['created_at'])
        )
        
        logger.info(f"策略已从文件加载: {filepath}")
        return strategy


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建策略生成器
    generator = StrategyGenerator()
    
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    price_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.02),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.02) + np.random.rand(1000) * 0.01,
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.02) - np.random.rand(1000) * 0.01,
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.02),
        'volume': np.random.randint(1000000, 10000000, 1000)
    }, index=dates)
    
    print("=== E1策略生成器演示 ===")
    
    # 1. 生成单个策略
    print("\n1. 生成趋势跟踪策略:")
    trend_strategy = generator.generate_strategy(
        strategy_type=StrategyType.TREND_FOLLOWING,
        risk_level=RiskLevel.MODERATE,
        market_data=price_data,
        use_optimization=True
    )
    
    print(f"策略名称: {trend_strategy.name}")
    print(f"可行性评分: {trend_strategy.feasibility_score:.2f}")
    print(f"预期收益率: {trend_strategy.performance_estimate['total_return']:.2f}%")
    print(f"夏普比率: {trend_strategy.performance_estimate['sharpe_ratio']:.2f}")
    print(f"最大回撤: {trend_strategy.performance_estimate['max_drawdown']:.2f}%")
    
    # 2. 生成均值回归策略
    print("\n2. 生成均值回归策略:")
    mean_rev_strategy = generator.generate_strategy(
        strategy_type=StrategyType.MEAN_REVERSION,
        risk_level=RiskLevel.CONSERVATIVE,
        market_data=price_data,
        use_optimization=True
    )
    
    print(f"策略名称: {mean_rev_strategy.name}")
    print(f"可行性评分: {mean_rev_strategy.feasibility_score:.2f}")
    
    # 3. 生成混合策略
    print("\n3. 生成混合策略:")
    hybrid_strategy = generator.generate_hybrid_strategy(
        strategy_types=[StrategyType.TREND_FOLLOWING, StrategyType.MEAN_REVERSION],
        risk_level=RiskLevel.MODERATE,
        market_data=price_data,
        weights=[0.6, 0.4]
    )
    
    print(f"策略名称: {hybrid_strategy.name}")
    print(f"可行性评分: {hybrid_strategy.feasibility_score:.2f}")
    
    # 4. 个性化策略
    print("\n4. 个性化策略定制:")
    user_preferences = {
        'risk_tolerance': 'conservative',
        'trading_frequency': 'low'
    }
    
    personalized_strategy = generator.personalize_strategy(
        base_strategy=trend_strategy,
        user_preferences=user_preferences,
        market_data=price_data
    )
    
    print(f"个性化策略名称: {personalized_strategy.name}")
    print(f"可行性评分: {personalized_strategy.feasibility_score:.2f}")
    
    # 5. 保存策略
    print("\n5. 保存策略:")
    generator.save_strategy(trend_strategy, '/workspace/trend_strategy.json')
    generator.save_strategy(mean_rev_strategy, '/workspace/mean_reversion_strategy.json')
    
    # 6. 显示策略代码示例
    print("\n6. 策略代码预览 (前20行):")
    code_lines = trend_strategy.code.split('\n')[:20]
    for i, line in enumerate(code_lines, 1):
        print(f"{i:2d}: {line}")
    
    print("\n=== 策略生成完成 ===")
    print(f"共生成 {3} 个策略")
    print("策略文件已保存到工作目录")