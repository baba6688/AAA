"""
F2策略学习器 - 核心策略学习系统

该模块实现了一个完整的策略学习系统，包括：
1. 策略学习和改进算法（强化学习、进化算法、模仿学习等）
2. 策略性能分析和评估
3. 策略适应性和进化
4. 策略组合和融合
5. 策略效果跟踪和预测
6. 策略知识提取和管理
7. 策略学习策略优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict, deque
import threading
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """策略类型枚举"""
    REINFORCEMENT = "reinforcement"
    EVOLUTION = "evolution"
    IMITATION = "imitation"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"

class LearningPhase(Enum):
    """学习阶段枚举"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    ADAPTATION = "adaptation"
    EVOLUTION = "evolution"

@dataclass
class StrategyState:
    """策略状态数据类"""
    strategy_id: str
    strategy_type: StrategyType
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningContext:
    """学习上下文数据类"""
    environment_state: Dict[str, Any]
    historical_performance: List[float]
    current_objective: str
    constraints: Dict[str, Any]
    risk_tolerance: float = 0.5
    time_horizon: int = 100

@dataclass
class StrategyPerformance:
    """策略性能数据类"""
    strategy_id: str
    timestamp: datetime
    return_rate: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    additional_metrics: Dict[str, float] = field(default_factory=dict)

class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, strategy_id: str, strategy_type: StrategyType):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.state = StrategyState(
            strategy_id=strategy_id,
            strategy_type=strategy_type
        )
        self.is_active = False
        
    @abstractmethod
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """执行学习过程"""
        pass
    
    @abstractmethod
    def predict(self, state: Dict[str, Any]) -> Any:
        """执行预测/决策"""
        pass
    
    @abstractmethod
    def update_performance(self, performance: StrategyPerformance):
        """更新性能指标"""
        pass
    
    def get_state(self) -> StrategyState:
        """获取策略状态"""
        return self.state
    
    def activate(self):
        """激活策略"""
        self.is_active = True
        logger.info(f"策略 {self.strategy_id} 已激活")
    
    def deactivate(self):
        """停用策略"""
        self.is_active = False
        logger.info(f"策略 {self.strategy_id} 已停用")

class ReinforcementLearning(BaseStrategy):
    """强化学习策略实现"""
    
    def __init__(self, strategy_id: str, learning_rate: float = 0.01, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        super().__init__(strategy_id, StrategyType.REINFORCEMENT)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = deque(maxlen=10000)
        self.policy = {}
        
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """强化学习训练过程"""
        try:
            # 从经验缓冲区中采样
            if len(self.experience_buffer) > 100:
                batch_size = min(32, len(self.experience_buffer))
                batch = random.sample(list(self.experience_buffer), batch_size)
                
                for state, action, reward, next_state in batch:
                    # Q-learning更新
                    current_q = self.q_table[state][action]
                    max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
                    td_target = reward + self.discount_factor * max_next_q
                    td_error = td_target - current_q
                    
                    self.q_table[state][action] += self.learning_rate * td_error
            
            # 探索-利用平衡
            if random.random() < self.epsilon:
                action = self._explore(context.environment_state)
            else:
                action = self._exploit(context.environment_state)
            
            # 更新策略
            self.policy = dict(self.q_table)
            
            # 记录学习结果
            learning_result = {
                'action': action,
                'q_values': dict(self.q_table[json.dumps(context.environment_state)]),
                'exploration_rate': self.epsilon,
                'learning_progress': len(self.experience_buffer)
            }
            
            return learning_result
            
        except Exception as e:
            logger.error(f"强化学习过程出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """基于Q表进行预测"""
        try:
            state_key = json.dumps(state, sort_keys=True)
            if state_key in self.q_table and self.q_table[state_key]:
                # 选择Q值最大的动作
                action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                confidence = self.q_table[state_key][action]
                return {
                    'action': action,
                    'confidence': confidence,
                    'q_values': dict(self.q_table[state_key])
                }
            else:
                # 如果没有历史数据，返回随机动作
                return {
                    'action': self._generate_random_action(),
                    'confidence': 0.0,
                    'q_values': {}
                }
        except Exception as e:
            logger.error(f"预测过程出错: {e}")
            return {'action': None, 'confidence': 0.0, 'error': str(e)}
    
    def update_performance(self, performance: StrategyPerformance):
        """更新性能指标"""
        try:
            self.state.performance_metrics.update({
                'return_rate': performance.return_rate,
                'sharpe_ratio': performance.sharpe_ratio,
                'max_drawdown': performance.max_drawdown,
                'win_rate': performance.win_rate
            })
            
            # 更新成功率
            if performance.return_rate > 0:
                self.state.success_rate = (self.state.success_rate * self.state.usage_count + 1) / (self.state.usage_count + 1)
            else:
                self.state.success_rate = (self.state.success_rate * self.state.usage_count) / (self.state.usage_count + 1)
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
            # 动态调整学习参数
            if self.state.success_rate > 0.7:
                self.epsilon = max(0.01, self.epsilon * 0.99)  # 减少探索
            elif self.state.success_rate < 0.3:
                self.epsilon = min(0.3, self.epsilon * 1.01)   # 增加探索
                
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def add_experience(self, state: Dict[str, Any], action: Any, reward: float, next_state: Dict[str, Any]):
        """添加经验到缓冲区"""
        experience = (
            json.dumps(state, sort_keys=True),
            str(action),
            reward,
            json.dumps(next_state, sort_keys=True)
        )
        self.experience_buffer.append(experience)
    
    def _explore(self, state: Dict[str, Any]) -> Any:
        """探索动作"""
        return self._generate_random_action()
    
    def _exploit(self, state: Dict[str, Any]) -> Any:
        """利用已知最优动作"""
        state_key = json.dumps(state, sort_keys=True)
        if state_key in self.q_table and self.q_table[state_key]:
            return max(self.q_table[state_key], key=self.q_table[state_key].get)
        else:
            return self._generate_random_action()
    
    def _generate_random_action(self) -> Any:
        """生成随机动作"""
        actions = ['buy', 'sell', 'hold', 'increase_position', 'decrease_position']
        return random.choice(actions)

class EvolutionAlgorithm(BaseStrategy):
    """进化算法策略实现"""
    
    def __init__(self, strategy_id: str, population_size: int = 50, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        super().__init__(strategy_id, StrategyType.EVOLUTION)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
        self.fitness_history = []
        
        # 初始化种群
        self._initialize_population()
    
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """进化学习过程"""
        try:
            # 评估种群适应度
            fitness_scores = []
            for individual in self.population:
                fitness = self._evaluate_fitness(individual, context)
                fitness_scores.append(fitness)
            
            # 记录适应度历史
            self.fitness_history.append({
                'generation': self.generation,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'worst_fitness': min(fitness_scores)
            })
            
            # 选择、交叉、变异
            new_population = []
            
            # 精英保留
            elite_count = max(1, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择父代
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(self.population[parent1], self.population[parent2])
                else:
                    child1, child2 = self.population[parent1].copy(), self.population[parent2].copy()
                
                # 变异
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 更新种群
            self.population = new_population[:self.population_size]
            self.generation += 1
            
            # 返回学习结果
            best_individual = self.population[np.argmax(fitness_scores)]
            return {
                'generation': self.generation,
                'best_fitness': max(fitness_scores),
                'best_individual': best_individual,
                'population_diversity': self._calculate_diversity(),
                'convergence_rate': self._calculate_convergence()
            }
            
        except Exception as e:
            logger.error(f"进化学习过程出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """基于最优个体进行预测"""
        try:
            if not self.population:
                return {'action': 'hold', 'confidence': 0.0}
            
            # 选择最优个体（简化版，实际应该基于适应度）
            best_individual = self.population[0]
            
            # 基于个体参数生成动作
            action = self._decode_individual(best_individual, state)
            
            return {
                'action': action,
                'confidence': 0.8,  # 进化算法通常有较高置信度
                'individual': best_individual,
                'generation': self.generation
            }
            
        except Exception as e:
            logger.error(f"进化预测过程出错: {e}")
            return {'action': None, 'confidence': 0.0, 'error': str(e)}
    
    def update_performance(self, performance: StrategyPerformance):
        """更新性能指标"""
        try:
            self.state.performance_metrics.update({
                'return_rate': performance.return_rate,
                'sharpe_ratio': performance.sharpe_ratio,
                'max_drawdown': performance.max_drawdown,
                'win_rate': performance.win_rate
            })
            
            # 进化算法关注长期表现
            if performance.return_rate > 0:
                self.state.success_rate = (self.state.success_rate * 0.9 + 0.1) if self.state.usage_count > 0 else 1.0
            else:
                self.state.success_rate = (self.state.success_rate * 0.9) if self.state.usage_count > 0 else 0.0
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def _initialize_population(self):
        """初始化种群"""
        for _ in range(self.population_size):
            individual = {
                'buy_threshold': random.uniform(0.1, 0.9),
                'sell_threshold': random.uniform(0.1, 0.9),
                'stop_loss': random.uniform(0.05, 0.2),
                'take_profit': random.uniform(0.1, 0.5),
                'position_size': random.uniform(0.1, 1.0)
            }
            self.population.append(individual)
    
    def _evaluate_fitness(self, individual: Dict[str, float], context: LearningContext) -> float:
        """评估个体适应度"""
        try:
            # 基于历史性能计算适应度
            if context.historical_performance:
                # 简化的适应度计算：结合回报率和稳定性
                returns = context.historical_performance[-20:]  # 最近20个周期
                avg_return = np.mean(returns)
                return_stability = 1.0 / (1.0 + np.std(returns))  # 稳定性越高越好
                risk_adjusted_return = avg_return * return_stability
                
                # 结合个体参数的影响
                parameter_bonus = (individual['take_profit'] - individual['stop_loss']) * 0.1
                
                fitness = risk_adjusted_return + parameter_bonus
                return max(0.0, fitness)  # 确保适应度非负
            else:
                return 0.5  # 默认适应度
                
        except Exception as e:
            logger.error(f"评估适应度出错: {e}")
            return 0.0
    
    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> int:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return winner_idx
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """交叉操作"""
        child1, child2 = {}, {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        return child1, child2
    
    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """变异操作"""
        mutated = individual.copy()
        for key in mutated.keys():
            if random.random() < 0.1:  # 10%变异概率
                if key in ['buy_threshold', 'sell_threshold']:
                    mutated[key] = np.clip(mutated[key] + random.uniform(-0.1, 0.1), 0.0, 1.0)
                else:
                    mutated[key] = max(0.01, mutated[key] + random.uniform(-0.05, 0.05))
        return mutated
    
    def _decode_individual(self, individual: Dict[str, float], state: Dict[str, Any]) -> str:
        """解码个体为动作"""
        try:
            # 简化的解码逻辑
            market_signal = state.get('market_signal', 0.5)
            
            if market_signal > individual['buy_threshold']:
                return 'buy'
            elif market_signal < individual['sell_threshold']:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            logger.error(f"解码个体出错: {e}")
            return 'hold'
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) < 2:
            return 0.0
        
        diversity_scores = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.sqrt(sum((self.population[i][k] - self.population[j][k]) ** 2 for k in self.population[i].keys()))
                diversity_scores.append(distance)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_convergence(self) -> float:
        """计算收敛率"""
        if len(self.fitness_history) < 2:
            return 0.0
        
        recent_fitness = [h['best_fitness'] for h in self.fitness_history[-5:]]
        if len(recent_fitness) < 2:
            return 0.0
        
        # 计算最近几代最佳适应度的变化率
        changes = [abs(recent_fitness[i] - recent_fitness[i-1]) for i in range(1, len(recent_fitness))]
        avg_change = np.mean(changes)
        
        # 收敛率 = 1 - 平均变化率（变化越小，收敛率越高）
        convergence_rate = max(0.0, 1.0 - avg_change)
        return convergence_rate

class ImitationLearning(BaseStrategy):
    """模仿学习策略实现"""
    
    def __init__(self, strategy_id: str, expert_data: Optional[List[Dict]] = None):
        super().__init__(strategy_id, StrategyType.IMITATION)
        self.expert_data = expert_data or []
        self.model = None
        self.feature_weights = {}
        self.action_patterns = defaultdict(int)
        
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """模仿学习过程"""
        try:
            if not self.expert_data:
                return {'error': '没有专家数据进行学习'}
            
            # 从专家数据中学习模式
            self._extract_patterns()
            
            # 学习特征权重
            self._learn_feature_weights()
            
            # 生成模仿策略
            imitation_strategy = self._generate_imitation_strategy(context)
            
            return {
                'strategy': imitation_strategy,
                'pattern_count': len(self.action_patterns),
                'feature_weights': self.feature_weights,
                'confidence': self._calculate_confidence()
            }
            
        except Exception as e:
            logger.error(f"模仿学习过程出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """基于模仿的预测"""
        try:
            if not self.action_patterns:
                return {'action': 'hold', 'confidence': 0.0}
            
            # 找到最相似的历史状态
            similar_state = self._find_similar_state(state)
            
            if similar_state:
                action = similar_state['action']
                confidence = self._calculate_action_confidence(action, state)
            else:
                # 使用最频繁的动作
                action = max(self.action_patterns, key=self.action_patterns.get)
                confidence = self.action_patterns[action] / sum(self.action_patterns.values())
            
            return {
                'action': action,
                'confidence': confidence,
                'similar_state': similar_state,
                'pattern_usage': dict(self.action_patterns)
            }
            
        except Exception as e:
            logger.error(f"模仿预测过程出错: {e}")
            return {'action': None, 'confidence': 0.0, 'error': str(e)}
    
    def update_performance(self, performance: StrategyPerformance):
        """更新性能指标"""
        try:
            self.state.performance_metrics.update({
                'return_rate': performance.return_rate,
                'sharpe_ratio': performance.sharpe_ratio,
                'max_drawdown': performance.max_drawdown,
                'win_rate': performance.win_rate
            })
            
            # 模仿学习关注与专家的相似度
            if performance.return_rate > 0:
                self.state.success_rate = min(1.0, self.state.success_rate + 0.01)
            else:
                self.state.success_rate = max(0.0, self.state.success_rate - 0.01)
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def add_expert_data(self, expert_data: List[Dict]):
        """添加专家数据"""
        self.expert_data.extend(expert_data)
        logger.info(f"添加了 {len(expert_data)} 条专家数据")
    
    def _extract_patterns(self):
        """提取动作模式"""
        self.action_patterns.clear()
        
        for data_point in self.expert_data:
            action = data_point.get('action', 'hold')
            self.action_patterns[action] += 1
    
    def _learn_feature_weights(self):
        """学习特征权重"""
        self.feature_weights.clear()
        
        if not self.expert_data:
            return
        
        # 简化的特征权重学习
        feature_importance = defaultdict(float)
        total_samples = len(self.expert_data)
        
        for data_point in self.expert_data:
            features = data_point.get('features', {})
            action = data_point.get('action', 'hold')
            
            # 计算特征重要性（基于动作分布）
            for feature, value in features.items():
                if isinstance(value, (int, float)):
                    feature_importance[feature] += abs(value) / total_samples
        
        # 归一化权重
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            self.feature_weights = {k: v/total_importance for k, v in feature_importance.items()}
        else:
            self.feature_weights = {k: 1.0/len(feature_importance) for k in feature_importance.keys()}
    
    def _generate_imitation_strategy(self, context: LearningContext) -> Dict[str, Any]:
        """生成模仿策略"""
        strategy = {
            'primary_action': max(self.action_patterns, key=self.action_patterns.get),
            'confidence_threshold': 0.6,
            'fallback_actions': list(self.action_patterns.keys())[:3],
            'feature_focus': list(self.feature_weights.keys())[:5]
        }
        return strategy
    
    def _find_similar_state(self, current_state: Dict[str, Any]) -> Optional[Dict]:
        """找到相似的历史状态"""
        if not self.expert_data:
            return None
        
        best_similarity = 0.0
        best_match = None
        
        for data_point in self.expert_data:
            historical_state = data_point.get('state', {})
            similarity = self._calculate_similarity(current_state, historical_state)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = data_point
        
        return best_match if best_similarity > 0.5 else None
    
    def _calculate_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """计算状态相似度"""
        if not state1 or not state2:
            return 0.0
        
        common_keys = set(state1.keys()) & set(state2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = state1[key], state2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值相似度
                max_val = max(abs(val1), abs(val2), 1.0)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(similarity)
            elif val1 == val2:
                # 分类相似度
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_action_confidence(self, action: str, state: Dict[str, Any]) -> float:
        """计算动作置信度"""
        base_confidence = self.action_patterns.get(action, 0) / sum(self.action_patterns.values())
        
        # 根据特征权重调整置信度
        feature_bonus = 0.0
        for feature, weight in self.feature_weights.items():
            if feature in state:
                feature_bonus += weight * 0.1
        
        confidence = min(1.0, base_confidence + feature_bonus)
        return confidence
    
    def _calculate_confidence(self) -> float:
        """计算整体置信度"""
        if not self.expert_data:
            return 0.0
        
        # 基于数据质量和模式一致性计算置信度
        data_quality = min(1.0, len(self.expert_data) / 1000)  # 数据量
        pattern_consistency = 1.0 - (np.std(list(self.action_patterns.values())) / np.mean(list(self.action_patterns.values())) if self.action_patterns else 0)
        
        confidence = (data_quality + pattern_consistency) / 2
        return max(0.0, min(1.0, confidence))

class PerformanceAnalyzer:
    """策略性能分析器"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.benchmark_data = None
        self.analysis_cache = {}
        
    def analyze_performance(self, strategy_id: str, performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """分析策略性能"""
        try:
            if not performance_data:
                return {'error': '没有性能数据进行分析'}
            
            # 基本统计指标
            returns = [p.return_rate for p in performance_data]
            volatilities = [p.volatility for p in performance_data]
            
            analysis = {
                'strategy_id': strategy_id,
                'analysis_timestamp': datetime.now(),
                'data_points': len(performance_data),
                'time_period': self._calculate_time_period(performance_data),
                
                # 收益指标
                'total_return': self._calculate_total_return(returns),
                'annualized_return': self._calculate_annualized_return(returns),
                'average_return': np.mean(returns),
                'return_std': np.std(returns),
                
                # 风险指标
                'volatility': np.mean(volatilities),
                'max_drawdown': max([p.max_drawdown for p in performance_data]),
                'var_95': self._calculate_var(returns, 0.95),
                'var_99': self._calculate_var(returns, 0.99),
                
                # 风险调整收益
                'sharpe_ratio': self._calculate_sharpe_ratio(returns, np.mean(volatilities)),
                'sortino_ratio': self._calculate_sortino_ratio(returns),
                'calmar_ratio': self._calculate_calmar_ratio(returns),
                
                # 交易指标
                'win_rate': np.mean([p.win_rate for p in performance_data]),
                'profit_factor': np.mean([p.profit_factor for p in performance_data]),
                'average_win': self._calculate_average_win(performance_data),
                'average_loss': self._calculate_average_loss(performance_data),
                
                # 一致性指标
                'return_consistency': self._calculate_consistency(returns),
                'up_capture': self._calculate_up_capture(returns),
                'down_capture': self._calculate_down_capture(returns)
            }
            
            # 性能分级
            analysis['performance_grade'] = self._grade_performance(analysis)
            
            # 风险评估
            analysis['risk_level'] = self._assess_risk_level(analysis)
            
            # 改进建议
            analysis['improvement_suggestions'] = self._generate_improvement_suggestions(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"性能分析出错: {e}")
            return {'error': str(e)}
    
    def compare_strategies(self, strategy_performances: Dict[str, List[StrategyPerformance]]) -> Dict[str, Any]:
        """比较多个策略的性能"""
        try:
            comparison_results = {}
            
            # 分析每个策略
            for strategy_id, performances in strategy_performances.items():
                comparison_results[strategy_id] = self.analyze_performance(strategy_id, performances)
            
            # 排名和比较
            ranking = self._rank_strategies(comparison_results)
            
            # 相关性分析
            correlations = self._calculate_strategy_correlations(strategy_performances)
            
            # 组合优化建议
            portfolio_suggestions = self._generate_portfolio_suggestions(comparison_results, correlations)
            
            return {
                'individual_analysis': comparison_results,
                'rankings': ranking,
                'correlations': correlations,
                'portfolio_suggestions': portfolio_suggestions,
                'comparison_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"策略比较出错: {e}")
            return {'error': str(e)}
    
    def predict_future_performance(self, strategy_id: str, performance_data: List[StrategyPerformance], 
                                 prediction_horizon: int = 30) -> Dict[str, Any]:
        """预测策略未来性能"""
        try:
            if len(performance_data) < 10:
                return {'error': '历史数据不足，无法进行预测'}
            
            # 提取时间序列
            returns = [p.return_rate for p in performance_data]
            timestamps = [p.timestamp for p in performance_data]
            
            # 趋势分析
            trend_analysis = self._analyze_trend(returns)
            
            # 波动率预测
            volatility_prediction = self._predict_volatility(returns)
            
            # 风险预测
            risk_prediction = self._predict_risk(returns)
            
            # 蒙特卡洛模拟
            monte_carlo_results = self._monte_carlo_simulation(returns, prediction_horizon)
            
            return {
                'strategy_id': strategy_id,
                'prediction_horizon': prediction_horizon,
                'prediction_timestamp': datetime.now(),
                'trend_analysis': trend_analysis,
                'volatility_prediction': volatility_prediction,
                'risk_prediction': risk_prediction,
                'monte_carlo_results': monte_carlo_results,
                'confidence_intervals': self._calculate_confidence_intervals(monte_carlo_results),
                'recommendation': self._generate_prediction_recommendation(trend_analysis, risk_prediction)
            }
            
        except Exception as e:
            logger.error(f"性能预测出错: {e}")
            return {'error': str(e)}
    
    def _calculate_total_return(self, returns: List[float]) -> float:
        """计算总回报率"""
        if not returns:
            return 0.0
        return np.prod([1 + r for r in returns]) - 1
    
    def _calculate_annualized_return(self, returns: List[float]) -> float:
        """计算年化回报率"""
        if not returns:
            return 0.0
        total_return = self._calculate_total_return(returns)
        periods_per_year = 252  # 假设一年252个交易日
        periods = len(returns)
        return (1 + total_return) ** (periods_per_year / periods) - 1
    
    def _calculate_var(self, returns: List[float], confidence_level: float) -> float:
        """计算风险价值(VaR)"""
        if not returns:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if not returns or np.std(returns) == 0:
            return 0.0
        excess_returns = np.mean(returns) - risk_free_rate / 252  # 日化无风险利率
        return excess_returns / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        if not returns:
            return 0.0
        excess_returns = np.mean(returns) - risk_free_rate / 252
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return float('inf')
        downside_deviation = np.std(downside_returns)
        return excess_returns / downside_deviation * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """计算卡玛比率"""
        total_return = self._calculate_total_return(returns)
        max_drawdown = abs(min([self._calculate_drawdown(returns[:i+1]) for i in range(len(returns))]))
        if max_drawdown == 0:
            return float('inf')
        return total_return / max_drawdown
    
    def _calculate_drawdown(self, returns: List[float]) -> float:
        """计算回撤"""
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_average_win(self, performances: List[StrategyPerformance]) -> float:
        """计算平均盈利"""
        wins = [p.return_rate for p in performances if p.return_rate > 0]
        return np.mean(wins) if wins else 0.0
    
    def _calculate_average_loss(self, performances: List[StrategyPerformance]) -> float:
        """计算平均亏损"""
        losses = [p.return_rate for p in performances if p.return_rate < 0]
        return np.mean(losses) if losses else 0.0
    
    def _calculate_consistency(self, returns: List[float]) -> float:
        """计算收益一致性"""
        if not returns:
            return 0.0
        positive_returns = [r for r in returns if r > 0]
        return len(positive_returns) / len(returns)
    
    def _calculate_up_capture(self, returns: List[float]) -> float:
        """计算上行捕获率"""
        # 简化实现，实际需要基准数据
        return min(1.0, np.mean([r for r in returns if r > 0]) * 10) if returns else 0.0
    
    def _calculate_down_capture(self, returns: List[float]) -> float:
        """计算下行捕获率"""
        # 简化实现，实际需要基准数据
        return min(1.0, abs(np.mean([r for r in returns if r < 0])) * 10) if returns else 0.0
    
    def _grade_performance(self, analysis: Dict[str, Any]) -> str:
        """性能分级"""
        sharpe = analysis.get('sharpe_ratio', 0)
        total_return = analysis.get('total_return', 0)
        max_drawdown = analysis.get('max_drawdown', 1)
        
        score = 0
        if sharpe > 2: score += 3
        elif sharpe > 1: score += 2
        elif sharpe > 0.5: score += 1
        
        if total_return > 0.2: score += 2
        elif total_return > 0.1: score += 1
        
        if max_drawdown < 0.1: score += 2
        elif max_drawdown < 0.2: score += 1
        
        if score >= 6: return "优秀"
        elif score >= 4: return "良好"
        elif score >= 2: return "一般"
        else: return "较差"
    
    def _assess_risk_level(self, analysis: Dict[str, Any]) -> str:
        """风险等级评估"""
        volatility = analysis.get('volatility', 0)
        max_drawdown = analysis.get('max_drawdown', 0)
        
        risk_score = volatility + max_drawdown
        
        if risk_score < 0.15: return "低风险"
        elif risk_score < 0.3: return "中等风险"
        else: return "高风险"
    
    def _generate_improvement_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        sharpe = analysis.get('sharpe_ratio', 0)
        if sharpe < 1:
            suggestions.append("建议优化风险调整收益，考虑降低波动率")
        
        win_rate = analysis.get('win_rate', 0)
        if win_rate < 0.5:
            suggestions.append("胜率偏低，建议改进入场和出场时机")
        
        max_drawdown = analysis.get('max_drawdown', 0)
        if max_drawdown > 0.2:
            suggestions.append("最大回撤较大，建议加强风险管理")
        
        return suggestions
    
    def _calculate_time_period(self, performances: List[StrategyPerformance]) -> Dict[str, Any]:
        """计算时间周期"""
        if not performances:
            return {}
        
        timestamps = [p.timestamp for p in performances]
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = end_time - start_time
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration_days': duration.days,
            'duration_hours': duration.total_seconds() / 3600
        }
    
    def _rank_strategies(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """策略排名"""
        rankings = []
        for strategy_id, analysis in analysis_results.items():
            if 'error' not in analysis:
                score = (
                    analysis.get('sharpe_ratio', 0) * 0.3 +
                    analysis.get('total_return', 0) * 0.3 +
                    (1 - analysis.get('max_drawdown', 1)) * 0.2 +
                    analysis.get('win_rate', 0) * 0.2
                )
                rankings.append({
                    'strategy_id': strategy_id,
                    'composite_score': score,
                    'performance_grade': analysis.get('performance_grade', '未知')
                })
        
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        return rankings
    
    def _calculate_strategy_correlations(self, strategy_performances: Dict[str, List[StrategyPerformance]]) -> Dict[str, Dict[str, float]]:
        """计算策略相关性"""
        correlations = {}
        strategy_ids = list(strategy_performances.keys())
        
        for i, strategy1 in enumerate(strategy_ids):
            correlations[strategy1] = {}
            for j, strategy2 in enumerate(strategy_ids):
                if i != j:
                    returns1 = [p.return_rate for p in strategy_performances[strategy1]]
                    returns2 = [p.return_rate for p in strategy_performances[strategy2]]
                    
                    min_length = min(len(returns1), len(returns2))
                    if min_length > 1:
                        corr = np.corrcoef(returns1[:min_length], returns2[:min_length])[0, 1]
                        correlations[strategy1][strategy2] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlations[strategy1][strategy2] = 0.0
                else:
                    correlations[strategy1][strategy2] = 1.0
        
        return correlations
    
    def _generate_portfolio_suggestions(self, analysis_results: Dict[str, Any], 
                                      correlations: Dict[str, Dict[str, float]]) -> List[str]:
        """生成组合建议"""
        suggestions = []
        
        # 基于相关性建议分散化
        high_corr_pairs = []
        for strategy1, corr_dict in correlations.items():
            for strategy2, corr in corr_dict.items():
                if strategy1 != strategy2 and corr > 0.7:
                    high_corr_pairs.append((strategy1, strategy2, corr))
        
        if high_corr_pairs:
            suggestions.append("检测到高相关性策略，建议选择其中一个或调整权重")
        
        # 基于性能建议权重分配
        performance_scores = []
        for strategy_id, analysis in analysis_results.items():
            if 'error' not in analysis:
                score = analysis.get('sharpe_ratio', 0) + analysis.get('total_return', 0)
                performance_scores.append((strategy_id, score))
        
        if performance_scores:
            best_strategy = max(performance_scores, key=lambda x: x[1])
            suggestions.append(f"建议重点关注策略 {best_strategy[0]}，其综合表现最佳")
        
        return suggestions
    
    def _analyze_trend(self, returns: List[float]) -> Dict[str, Any]:
        """趋势分析"""
        if len(returns) < 5:
            return {'trend': 'insufficient_data'}
        
        # 线性回归趋势
        x = np.arange(len(returns))
        slope = np.polyfit(x, returns, 1)[0]
        
        if slope > 0.001:
            trend = 'upward'
        elif slope < -0.001:
            trend = 'downward'
        else:
            trend = 'sideways'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': np.corrcoef(x, returns)[0, 1] ** 2 if len(returns) > 1 else 0
        }
    
    def _predict_volatility(self, returns: List[float]) -> Dict[str, Any]:
        """波动率预测"""
        if not returns:
            return {'predicted_volatility': 0.0}
        
        recent_volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        historical_volatility = np.std(returns)
        
        return {
            'current_volatility': historical_volatility,
            'recent_volatility': recent_volatility,
            'predicted_volatility': (historical_volatility + recent_volatility) / 2,
            'volatility_trend': 'increasing' if recent_volatility > historical_volatility else 'decreasing'
        }
    
    def _predict_risk(self, returns: List[float]) -> Dict[str, Any]:
        """风险预测"""
        if not returns:
            return {'risk_level': 'unknown'}
        
        current_var = np.percentile(returns, 5)
        expected_shortfall = np.mean([r for r in returns if r <= current_var])
        
        risk_level = 'high' if current_var < -0.05 else 'medium' if current_var < -0.02 else 'low'
        
        return {
            'current_var_5': current_var,
            'expected_shortfall': expected_shortfall,
            'risk_level': risk_level,
            'risk_trend': 'increasing' if len(returns) > 10 and np.std(returns[-10:]) > np.std(returns[:-10]) else 'stable'
        }
    
    def _monte_carlo_simulation(self, returns: List[float], horizon: int, num_simulations: int = 1000) -> Dict[str, Any]:
        """蒙特卡洛模拟"""
        if not returns:
            return {'simulations': []}
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        simulations = []
        for _ in range(num_simulations):
            # 生成随机收益率序列
            simulated_returns = np.random.normal(mean_return, std_return, horizon)
            cumulative_return = np.prod([1 + r for r in simulated_returns]) - 1
            simulations.append(cumulative_return)
        
        return {
            'simulations': simulations[:100],  # 只返回前100个模拟结果
            'mean_projection': np.mean(simulations),
            'median_projection': np.median(simulations),
            'percentile_5': np.percentile(simulations, 5),
            'percentile_95': np.percentile(simulations, 95)
        }
    
    def _calculate_confidence_intervals(self, monte_carlo_results: Dict[str, Any]) -> Dict[str, float]:
        """计算置信区间"""
        simulations = monte_carlo_results.get('simulations', [])
        if not simulations:
            return {}
        
        return {
            'lower_bound_90': np.percentile(simulations, 5),
            'upper_bound_90': np.percentile(simulations, 95),
            'lower_bound_95': np.percentile(simulations, 2.5),
            'upper_bound_95': np.percentile(simulations, 97.5)
        }
    
    def _generate_prediction_recommendation(self, trend_analysis: Dict[str, Any], 
                                          risk_prediction: Dict[str, Any]) -> str:
        """生成预测建议"""
        trend = trend_analysis.get('trend', 'unknown')
        risk_level = risk_prediction.get('risk_level', 'unknown')
        
        if trend == 'upward' and risk_level == 'low':
            return "趋势向好且风险较低，建议保持当前策略"
        elif trend == 'downward' and risk_level == 'high':
            return "趋势向下且风险较高，建议减少仓位或暂停交易"
        elif trend == 'sideways':
            return "市场横盘整理，建议采用区间交易策略"
        else:
            return "市场状况复杂，建议密切关注并适时调整策略"

class StrategyCombiner:
    """策略组合器"""
    
    def __init__(self):
        self.combination_methods = {
            'weighted_average': self._weighted_average_combination,
            'voting': self._voting_combination,
            'stacking': self._stacking_combination,
            'adaptive': self._adaptive_combination
        }
        self.weight_history = defaultdict(list)
        self.performance_tracker = {}
        
    def combine_strategies(self, strategies: List[BaseStrategy], method: str = 'weighted_average',
                         context: Optional[LearningContext] = None) -> Dict[str, Any]:
        """组合多个策略"""
        try:
            if len(strategies) < 2:
                return {'error': '至少需要2个策略进行组合'}
            
            if method not in self.combination_methods:
                return {'error': f'不支持的组合方法: {method}'}
            
            # 获取策略预测结果
            predictions = []
            for strategy in strategies:
                if strategy.is_active:
                    if context:
                        prediction = strategy.predict(context.environment_state)
                    else:
                        prediction = strategy.predict({})
                    predictions.append({
                        'strategy': strategy,
                        'prediction': prediction
                    })
            
            if not predictions:
                return {'error': '没有可用的活跃策略'}
            
            # 执行组合
            combination_result = self.combination_methods[method](predictions, context)
            
            # 更新权重历史
            if 'weights' in combination_result:
                timestamp = datetime.now()
                for i, strategy in enumerate(strategies):
                    if i < len(combination_result['weights']):
                        self.weight_history[strategy.strategy_id].append({
                            'timestamp': timestamp,
                            'weight': combination_result['weights'][i]
                        })
            
            return combination_result
            
        except Exception as e:
            logger.error(f"策略组合出错: {e}")
            return {'error': str(e)}
    
    def _weighted_average_combination(self, predictions: List[Dict], 
                                    context: Optional[LearningContext]) -> Dict[str, Any]:
        """加权平均组合"""
        # 计算权重
        weights = []
        total_weight = 0
        
        for pred_dict in predictions:
            strategy = pred_dict['strategy']
            prediction = pred_dict['prediction']
            
            # 基于性能和置信度计算权重
            performance_score = strategy.state.success_rate
            confidence = prediction.get('confidence', 0.5)
            
            # 动态调整权重
            if context and context.risk_tolerance < 0.5:
                # 保守投资者偏向稳定策略
                weight = performance_score * 0.7 + confidence * 0.3
            else:
                # 激进投资者偏向高性能策略
                weight = performance_score * 0.3 + confidence * 0.7
            
            weights.append(weight)
            total_weight += weight
        
        # 归一化权重
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # 组合预测
        combined_action = self._combine_actions([p['prediction'] for p in predictions], weights)
        
        # 计算组合置信度
        combined_confidence = sum(
            pred['prediction'].get('confidence', 0.5) * weights[i] 
            for i, pred in enumerate(predictions)
        )
        
        return {
            'combined_action': combined_action,
            'combined_confidence': combined_confidence,
            'weights': weights,
            'method': 'weighted_average',
            'individual_predictions': [p['prediction'] for p in predictions]
        }
    
    def _voting_combination(self, predictions: List[Dict], 
                          context: Optional[LearningContext]) -> Dict[str, Any]:
        """投票组合"""
        action_votes = defaultdict(float)
        
        for i, pred_dict in enumerate(predictions):
            prediction = pred_dict['prediction']
            action = prediction.get('action', 'hold')
            confidence = prediction.get('confidence', 0.5)
            
            # 投票权重基于置信度
            action_votes[action] += confidence
        
        # 选择得票最多的动作
        if action_votes:
            best_action = max(action_votes, key=action_votes.get)
            total_votes = sum(action_votes.values())
            voting_confidence = action_votes[best_action] / total_votes if total_votes > 0 else 0
        else:
            best_action = 'hold'
            voting_confidence = 0.0
        
        return {
            'combined_action': best_action,
            'combined_confidence': voting_confidence,
            'votes': dict(action_votes),
            'method': 'voting',
            'individual_predictions': [p['prediction'] for p in predictions]
        }
    
    def _stacking_combination(self, predictions: List[Dict], 
                            context: Optional[LearningContext]) -> Dict[str, Any]:
        """堆叠组合（元学习）"""
        # 简化的堆叠实现
        # 在实际应用中，这里会训练一个元学习器
        
        features = []
        targets = []
        
        # 提取特征（简化版）
        for pred_dict in predictions:
            prediction = pred_dict['prediction']
            features.append([
                prediction.get('confidence', 0.0),
                pred_dict['strategy'].state.success_rate,
                pred_dict['strategy'].state.usage_count
            ])
        
        # 简化的元学习：基于历史性能选择最佳策略
        if context and context.historical_performance:
            best_strategy_idx = np.argmax([
                pred_dict['strategy'].state.success_rate 
                for pred_dict in predictions
            ])
            
            best_prediction = predictions[best_strategy_idx]['prediction']
            
            return {
                'combined_action': best_prediction.get('action', 'hold'),
                'combined_confidence': best_prediction.get('confidence', 0.5),
                'selected_strategy': predictions[best_strategy_idx]['strategy'].strategy_id,
                'method': 'stacking',
                'individual_predictions': [p['prediction'] for p in predictions]
            }
        else:
            # 回退到加权平均
            return self._weighted_average_combination(predictions, context)
    
    def _adaptive_combination(self, predictions: List[Dict], 
                            context: Optional[LearningContext]) -> Dict[str, Any]:
        """自适应组合"""
        # 根据市场环境动态调整组合方法
        
        if not context:
            return self._weighted_average_combination(predictions, context)
        
        market_volatility = context.environment_state.get('volatility', 0.1)
        trend_strength = context.environment_state.get('trend_strength', 0.5)
        
        # 动态选择组合方法
        if market_volatility > 0.2:  # 高波动环境
            method = 'voting'  # 投票更稳健
        elif trend_strength > 0.7:  # 强趋势环境
            method = 'stacking'  # 堆叠能更好捕捉趋势
        else:
            method = 'weighted_average'  # 默认使用加权平均
        
        # 执行选定的组合方法
        if method == 'voting':
            return self._voting_combination(predictions, context)
        elif method == 'stacking':
            return self._stacking_combination(predictions, context)
        else:
            return self._weighted_average_combination(predictions, context)
    
    def _combine_actions(self, predictions: List[Dict], weights: List[float]) -> str:
        """组合动作"""
        action_scores = defaultdict(float)
        
        for i, prediction in enumerate(predictions):
            action = prediction.get('action', 'hold')
            confidence = prediction.get('confidence', 0.5)
            action_scores[action] += confidence * weights[i]
        
        if action_scores:
            return max(action_scores, key=action_scores.get)
        else:
            return 'hold'
    
    def optimize_weights(self, strategy_performances: Dict[str, List[StrategyPerformance]], 
                        objective: str = 'sharpe_ratio') -> Dict[str, Any]:
        """优化策略权重"""
        try:
            # 计算每个策略的目标指标
            strategy_scores = {}
            analyzer = PerformanceAnalyzer()
            
            for strategy_id, performances in strategy_performances.items():
                analysis = analyzer.analyze_performance(strategy_id, performances)
                if 'error' not in analysis:
                    if objective == 'sharpe_ratio':
                        strategy_scores[strategy_id] = analysis.get('sharpe_ratio', 0)
                    elif objective == 'total_return':
                        strategy_scores[strategy_id] = analysis.get('total_return', 0)
                    elif objective == 'calmar_ratio':
                        strategy_scores[strategy_id] = analysis.get('calmar_ratio', 0)
                    else:
                        strategy_scores[strategy_id] = analysis.get('sharpe_ratio', 0)
            
            if not strategy_scores:
                return {'error': '没有有效的性能数据'}
            
            # 基于得分的权重分配
            total_score = sum(strategy_scores.values())
            if total_score > 0:
                optimized_weights = {
                    strategy_id: score / total_score 
                    for strategy_id, score in strategy_scores.items()
                }
            else:
                # 平均分配
                weight = 1.0 / len(strategy_scores)
                optimized_weights = {
                    strategy_id: weight 
                    for strategy_id in strategy_scores.keys()
                }
            
            # 计算预期组合性能
            expected_performance = sum(
                score * optimized_weights[strategy_id] 
                for strategy_id, score in strategy_scores.items()
            )
            
            return {
                'optimized_weights': optimized_weights,
                'expected_performance': expected_performance,
                'objective': objective,
                'optimization_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"权重优化出错: {e}")
            return {'error': str(e)}

class KnowledgeExtractor:
    """策略知识提取器"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.patterns = defaultdict(list)
        self.rules = []
        self.insights = []
        
    def extract_knowledge(self, strategies: List[BaseStrategy], 
                         performance_data: Dict[str, List[StrategyPerformance]]) -> Dict[str, Any]:
        """从策略和性能数据中提取知识"""
        try:
            # 提取策略模式
            strategy_patterns = self._extract_strategy_patterns(strategies)
            
            # 提取性能模式
            performance_patterns = self._extract_performance_patterns(performance_data)
            
            # 生成规则
            generated_rules = self._generate_rules(strategy_patterns, performance_patterns)
            
            # 提取洞察
            insights = self._extract_insights(strategies, performance_data)
            
            # 构建知识库
            knowledge_base = {
                'strategy_patterns': strategy_patterns,
                'performance_patterns': performance_patterns,
                'rules': generated_rules,
                'insights': insights,
                'extraction_timestamp': datetime.now(),
                'knowledge_quality': self._assess_knowledge_quality()
            }
            
            return knowledge_base
            
        except Exception as e:
            logger.error(f"知识提取出错: {e}")
            return {'error': str(e)}
    
    def _extract_strategy_patterns(self, strategies: List[BaseStrategy]) -> Dict[str, Any]:
        """提取策略模式"""
        patterns = {
            'strategy_types': defaultdict(int),
            'performance_distribution': defaultdict(list),
            'usage_patterns': defaultdict(int),
            'success_factors': []
        }
        
        for strategy in strategies:
            # 策略类型分布
            patterns['strategy_types'][strategy.strategy_type.value] += 1
            
            # 使用模式
            patterns['usage_patterns'][strategy.strategy_type.value] += strategy.state.usage_count
            
            # 成功因素
            if strategy.state.success_rate > 0.7:
                patterns['success_factors'].append({
                    'strategy_id': strategy.strategy_id,
                    'type': strategy.strategy_type.value,
                    'success_rate': strategy.state.success_rate,
                    'usage_count': strategy.state.usage_count
                })
        
        return dict(patterns)
    
    def _extract_performance_patterns(self, performance_data: Dict[str, List[StrategyPerformance]]) -> Dict[str, Any]:
        """提取性能模式"""
        patterns = {
            'return_patterns': defaultdict(list),
            'risk_patterns': defaultdict(list),
            'consistency_patterns': defaultdict(list),
            'correlation_patterns': {}
        }
        
        strategy_ids = list(performance_data.keys())
        
        for strategy_id, performances in performance_data.items():
            returns = [p.return_rate for p in performances]
            volatilities = [p.volatility for p in performances]
            win_rates = [p.win_rate for p in performances]
            
            patterns['return_patterns'][strategy_id] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns)
            }
            
            patterns['risk_patterns'][strategy_id] = {
                'mean_volatility': np.mean(volatilities),
                'max_drawdown': max([p.max_drawdown for p in performances]),
                'var_95': np.percentile(returns, 5)
            }
            
            patterns['consistency_patterns'][strategy_id] = {
                'win_rate': np.mean(win_rates),
                'profit_factor': np.mean([p.profit_factor for p in performances]),
                'consistency_score': len([r for r in returns if r > 0]) / len(returns)
            }
        
        # 计算策略间相关性
        for i, strategy1 in enumerate(strategy_ids):
            for j, strategy2 in enumerate(strategy_ids):
                if i != j:
                    returns1 = [p.return_rate for p in performance_data[strategy1]]
                    returns2 = [p.return_rate for p in performance_data[strategy2]]
                    
                    min_length = min(len(returns1), len(returns2))
                    if min_length > 1:
                        corr = np.corrcoef(returns1[:min_length], returns2[:min_length])[0, 1]
                        if not np.isnan(corr):
                            key = f"{strategy1}_vs_{strategy2}"
                            patterns['correlation_patterns'][key] = corr
        
        return patterns
    
    def _generate_rules(self, strategy_patterns: Dict, performance_patterns: Dict) -> List[Dict[str, Any]]:
        """生成规则"""
        rules = []
        
        # 基于策略类型的规则
        strategy_types = strategy_patterns.get('strategy_types', {})
        if strategy_types:
            dominant_type = max(strategy_types, key=strategy_types.get)
            rules.append({
                'rule_id': f"dominance_rule_{dominant_type}",
                'type': 'strategy_dominance',
                'description': f"{dominant_type} 策略在当前环境中占主导地位",
                'confidence': strategy_types[dominant_type] / sum(strategy_types.values()),
                'action': f"优先考虑使用 {dominant_type} 类型策略"
            })
        
        # 基于性能的规则
        performance_data = performance_patterns.get('return_patterns', {})
        if performance_data:
            best_strategy = max(performance_data.keys(), 
                              key=lambda s: performance_data[s]['mean'])
            best_return = performance_data[best_strategy]['mean']
            
            rules.append({
                'rule_id': f"best_performance_rule_{best_strategy}",
                'type': 'performance_ranking',
                'description': f"策略 {best_strategy} 表现最佳，平均回报率 {best_return:.4f}",
                'confidence': 0.8,
                'action': f"建议增加 {best_strategy} 策略的权重"
            })
        
        # 风险控制规则
        risk_data = performance_patterns.get('risk_patterns', {})
        high_risk_strategies = []
        for strategy_id, risk_info in risk_data.items():
            if risk_info.get('mean_volatility', 0) > 0.3:
                high_risk_strategies.append(strategy_id)
        
        if high_risk_strategies:
            rules.append({
                'rule_id': "risk_control_rule",
                'type': 'risk_management',
                'description': f"检测到高风险策略: {', '.join(high_risk_strategies)}",
                'confidence': 0.9,
                'action': "建议对这些策略设置更严格的风险控制"
            })
        
        return rules
    
    def _extract_insights(self, strategies: List[BaseStrategy], 
                        performance_data: Dict[str, List[StrategyPerformance]]) -> List[Dict[str, Any]]:
        """提取洞察"""
        insights = []
        
        # 策略多样性洞察
        strategy_types = set(strategy.strategy_type for strategy in strategies)
        if len(strategy_types) >= 3:
            insights.append({
                'insight_id': 'diversity_insight',
                'category': 'strategy_diversity',
                'description': '策略组合具有良好的多样性，涵盖多种学习范式',
                'importance': 'high',
                'recommendation': '保持策略多样性有助于提高整体稳健性'
            })
        
        # 性能稳定性洞察
        performance_stability = {}
        for strategy_id, performances in performance_data.items():
            returns = [p.return_rate for p in performances]
            if len(returns) > 1:
                stability = 1.0 / (1.0 + np.std(returns))
                performance_stability[strategy_id] = stability
        
        if performance_stability:
            most_stable = max(performance_stability, key=performance_stability.get)
            insights.append({
                'insight_id': 'stability_insight',
                'category': 'performance_stability',
                'description': f'策略 {most_stable} 具有最佳的性能稳定性',
                'importance': 'medium',
                'recommendation': '可以考虑将稳定策略作为核心持仓'
            })
        
        # 学习效率洞察
        learning_efficiency = {}
        for strategy in strategies:
            if strategy.state.usage_count > 0:
                efficiency = strategy.state.success_rate / max(1, strategy.state.usage_count / 100)
                learning_efficiency[strategy.strategy_id] = efficiency
        
        if learning_efficiency:
            most_efficient = max(learning_efficiency, key=learning_efficiency.get)
            insights.append({
                'insight_id': 'efficiency_insight',
                'category': 'learning_efficiency',
                'description': f'策略 {most_efficient} 具有最高的学习效率',
                'importance': 'medium',
                'recommendation': '可以重点关注和学习该策略的优化方法'
            })
        
        return insights
    
    def _assess_knowledge_quality(self) -> Dict[str, Any]:
        """评估知识质量"""
        quality_score = 0.0
        quality_factors = {}
        
        # 数据完整性
        if self.patterns:
            quality_score += 0.3
            quality_factors['data_completeness'] = 1.0
        else:
            quality_factors['data_completeness'] = 0.0
        
        # 规则质量
        if len(self.rules) > 0:
            quality_score += 0.3
            quality_factors['rule_quality'] = min(1.0, len(self.rules) / 10)
        else:
            quality_factors['rule_quality'] = 0.0
        
        # 洞察深度
        if len(self.insights) > 0:
            quality_score += 0.2
            quality_factors['insight_depth'] = min(1.0, len(self.insights) / 5)
        else:
            quality_factors['insight_depth'] = 0.0
        
        # 知识一致性
        quality_factors['consistency'] = 0.8  # 简化实现
        quality_score += 0.2 * quality_factors['consistency']
        
        return {
            'overall_quality': quality_score,
            'quality_factors': quality_factors,
            'quality_level': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.5 else 'low'
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """计算偏度"""
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean([((x - mean) / std) ** 3 for x in data])
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """计算峰度"""
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean([((x - mean) / std) ** 4 for x in data]) - 3

class StrategyOptimizer:
    """策略学习优化器"""
    
    def __init__(self):
        self.optimization_history = []
        self.hyperparameter_ranges = {
            'learning_rate': (0.001, 0.1),
            'discount_factor': (0.8, 0.99),
            'epsilon': (0.01, 0.3),
            'population_size': (20, 100),
            'mutation_rate': (0.01, 0.2),
            'crossover_rate': (0.5, 1.0)
        }
        self.optimization_lock = threading.Lock()
        
    def optimize_hyperparameters(self, strategy: BaseStrategy, 
                               performance_data: List[StrategyPerformance],
                               optimization_objective: str = 'sharpe_ratio') -> Dict[str, Any]:
        """优化超参数"""
        with self.optimization_lock:
            try:
                if not performance_data:
                    return {'error': '没有性能数据用于优化'}
                
                # 当前性能基线
                current_performance = self._evaluate_performance(performance_data, optimization_objective)
                
                # 定义优化空间
                if isinstance(strategy, ReinforcementLearning):
                    optimization_space = self._define_rl_optimization_space()
                elif isinstance(strategy, EvolutionAlgorithm):
                    optimization_space = self._define_ea_optimization_space()
                else:
                    optimization_space = {}
                
                if not optimization_space:
                    return {'error': '不支持的策略类型'}
                
                # 执行优化
                optimization_result = self._bayesian_optimization(
                    strategy, performance_data, optimization_space, optimization_objective
                )
                
                # 更新策略参数
                if 'optimal_params' in optimization_result:
                    self._update_strategy_params(strategy, optimization_result['optimal_params'])
                
                # 记录优化历史
                optimization_record = {
                    'strategy_id': strategy.strategy_id,
                    'optimization_timestamp': datetime.now(),
                    'objective': optimization_objective,
                    'initial_performance': current_performance,
                    'optimized_performance': optimization_result.get('best_performance', current_performance),
                    'improvement': optimization_result.get('best_performance', current_performance) - current_performance,
                    'optimal_params': optimization_result.get('optimal_params', {}),
                    'optimization_method': 'bayesian_optimization'
                }
                
                self.optimization_history.append(optimization_record)
                
                return optimization_result
                
            except Exception as e:
                logger.error(f"超参数优化出错: {e}")
                return {'error': str(e)}
    
    def optimize_learning_rate(self, strategy: BaseStrategy, 
                             performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """优化学习率"""
        try:
            # 测试不同的学习率
            learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
            performance_scores = []
            
            for lr in learning_rates:
                # 模拟不同学习率的性能
                simulated_performance = self._simulate_learning_rate_performance(
                    strategy, performance_data, lr
                )
                performance_scores.append(simulated_performance)
            
            # 选择最佳学习率
            best_idx = np.argmax(performance_scores)
            optimal_lr = learning_rates[best_idx]
            
            return {
                'optimal_learning_rate': optimal_lr,
                'performance_scores': dict(zip(learning_rates, performance_scores)),
                'improvement': performance_scores[best_idx] - performance_scores[0] if performance_scores else 0,
                'optimization_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"学习率优化出错: {e}")
            return {'error': str(e)}
    
    def optimize_exploration_exploitation_balance(self, strategy: BaseStrategy,
                                                performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """优化探索-利用平衡"""
        try:
            if not isinstance(strategy, ReinforcementLearning):
                return {'error': '该方法只适用于强化学习策略'}
            
            # 测试不同的epsilon值
            epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3]
            balance_scores = []
            
            for epsilon in epsilon_values:
                # 模拟不同探索率的效果
                exploration_score = self._simulate_exploration_effect(strategy, performance_data, epsilon)
                balance_scores.append(exploration_score)
            
            # 选择最佳平衡点
            best_idx = np.argmax(balance_scores)
            optimal_epsilon = epsilon_values[best_idx]
            
            return {
                'optimal_epsilon': optimal_epsilon,
                'balance_scores': dict(zip(epsilon_values, balance_scores)),
                'recommendation': self._generate_balance_recommendation(optimal_epsilon),
                'optimization_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"探索-利用平衡优化出错: {e}")
            return {'error': str(e)}
    
    def adaptive_optimization(self, strategy: BaseStrategy, 
                            performance_data: List[StrategyPerformance],
                            market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """自适应优化"""
        try:
            # 根据市场条件调整优化策略
            volatility = market_conditions.get('volatility', 0.1)
            trend_strength = market_conditions.get('trend_strength', 0.5)
            
            if volatility > 0.3:  # 高波动环境
                optimization_focus = 'risk_management'
                optimization_method = 'conservative_optimization'
            elif trend_strength > 0.7:  # 强趋势环境
                optimization_focus = 'trend_following'
                optimization_method = 'aggressive_optimization'
            else:  # 平稳环境
                optimization_focus = 'consistency'
                optimization_method = 'balanced_optimization'
            
            # 执行相应的优化
            if optimization_method == 'conservative_optimization':
                result = self._conservative_optimization(strategy, performance_data)
            elif optimization_method == 'aggressive_optimization':
                result = self._aggressive_optimization(strategy, performance_data)
            else:
                result = self._balanced_optimization(strategy, performance_data)
            
            result['optimization_focus'] = optimization_focus
            result['market_conditions'] = market_conditions
            result['adaptive_timestamp'] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"自适应优化出错: {e}")
            return {'error': str(e)}
    
    def _evaluate_performance(self, performance_data: List[StrategyPerformance], 
                            objective: str) -> float:
        """评估性能"""
        if not performance_data:
            return 0.0
        
        returns = [p.return_rate for p in performance_data]
        
        if objective == 'sharpe_ratio':
            if np.std(returns) == 0:
                return 0.0
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
        elif objective == 'total_return':
            return np.prod([1 + r for r in returns]) - 1
        elif objective == 'calmar_ratio':
            total_return = np.prod([1 + r for r in returns]) - 1
            max_drawdown = abs(min([self._calculate_drawdown(returns[:i+1]) for i in range(len(returns))]))
            return total_return / max_drawdown if max_drawdown > 0 else 0
        else:
            return np.mean(returns)
    
    def _define_rl_optimization_space(self) -> Dict[str, Any]:
        """定义强化学习优化空间"""
        return {
            'learning_rate': (0.001, 0.1),
            'discount_factor': (0.8, 0.99),
            'epsilon': (0.01, 0.3),
            'batch_size': (16, 128),
            'target_update_freq': (100, 1000)
        }
    
    def _define_ea_optimization_space(self) -> Dict[str, Any]:
        """定义进化算法优化空间"""
        return {
            'population_size': (20, 100),
            'mutation_rate': (0.01, 0.2),
            'crossover_rate': (0.5, 1.0),
            'elite_size': (1, 10),
            'tournament_size': (2, 10)
        }
    
    def _bayesian_optimization(self, strategy: BaseStrategy, 
                             performance_data: List[StrategyPerformance],
                             optimization_space: Dict[str, Any],
                             objective: str) -> Dict[str, Any]:
        """贝叶斯优化（简化实现）"""
        # 简化的贝叶斯优化实现
        best_performance = float('-inf')
        best_params = {}
        
        # 随机搜索
        num_iterations = 20
        for _ in range(num_iterations):
            # 生成随机参数
            params = {}
            for param_name, (min_val, max_val) in optimization_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = random.uniform(min_val, max_val)
            
            # 模拟性能评估
            simulated_performance = self._simulate_parameter_performance(
                strategy, performance_data, params
            )
            
            if simulated_performance > best_performance:
                best_performance = simulated_performance
                best_params = params
        
        return {
            'optimal_params': best_params,
            'best_performance': best_performance,
            'optimization_iterations': num_iterations,
            'method': 'bayesian_optimization'
        }
    
    def _update_strategy_params(self, strategy: BaseStrategy, params: Dict[str, Any]):
        """更新策略参数"""
        try:
            if isinstance(strategy, ReinforcementLearning):
                if 'learning_rate' in params:
                    strategy.learning_rate = params['learning_rate']
                if 'discount_factor' in params:
                    strategy.discount_factor = params['discount_factor']
                if 'epsilon' in params:
                    strategy.epsilon = params['epsilon']
            
            elif isinstance(strategy, EvolutionAlgorithm):
                if 'population_size' in params:
                    strategy.population_size = params['population_size']
                if 'mutation_rate' in params:
                    strategy.mutation_rate = params['mutation_rate']
                if 'crossover_rate' in params:
                    strategy.crossover_rate = params['crossover_rate']
            
            logger.info(f"策略 {strategy.strategy_id} 参数已更新")
            
        except Exception as e:
            logger.error(f"更新策略参数出错: {e}")
    
    def _simulate_learning_rate_performance(self, strategy: BaseStrategy,
                                          performance_data: List[StrategyPerformance],
                                          learning_rate: float) -> float:
        """模拟学习率性能"""
        # 简化的性能模拟
        base_performance = self._evaluate_performance(performance_data, 'sharpe_ratio')
        
        # 学习率对性能的影响（钟形曲线）
        optimal_lr = 0.01
        performance_factor = 1.0 - abs(learning_rate - optimal_lr) / optimal_lr
        
        return base_performance * performance_factor
    
    def _simulate_exploration_effect(self, strategy: ReinforcementLearning,
                                   performance_data: List[StrategyPerformance],
                                   epsilon: float) -> float:
        """模拟探索效果"""
        base_performance = self._evaluate_performance(performance_data, 'sharpe_ratio')
        
        # 探索率对性能的影响
        if epsilon < 0.05:  # 过度利用
            return base_performance * 0.8
        elif epsilon > 0.25:  # 过度探索
            return base_performance * 0.7
        else:  # 平衡状态
            return base_performance
    
    def _generate_balance_recommendation(self, epsilon: float) -> str:
        """生成平衡建议"""
        if epsilon < 0.05:
            return "建议增加探索率以避免局部最优"
        elif epsilon > 0.25:
            return "建议降低探索率以提高收敛速度"
        else:
            return "当前探索-利用平衡较为合理"
    
    def _conservative_optimization(self, strategy: BaseStrategy,
                                 performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """保守优化"""
        # 保守优化关注风险控制
        return {
            'optimization_approach': 'conservative',
            'focus': 'risk_management',
            'recommended_changes': {
                'reduce_learning_rate': True,
                'increase_exploration': True,
                'tighten_risk_controls': True
            },
            'expected_outcome': '降低波动性，提高稳定性'
        }
    
    def _aggressive_optimization(self, strategy: BaseStrategy,
                               performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """激进优化"""
        # 激进优化关注收益最大化
        return {
            'optimization_approach': 'aggressive',
            'focus': 'return_maximization',
            'recommended_changes': {
                'increase_learning_rate': True,
                'reduce_exploration': True,
                'focus_on_trends': True
            },
            'expected_outcome': '提高收益潜力，承担更高风险'
        }
    
    def _balanced_optimization(self, strategy: BaseStrategy,
                             performance_data: List[StrategyPerformance]) -> Dict[str, Any]:
        """平衡优化"""
        # 平衡优化兼顾收益和风险
        return {
            'optimization_approach': 'balanced',
            'focus': 'risk_return_balance',
            'recommended_changes': {
                'maintain_learning_rate': True,
                'moderate_exploration': True,
                'balanced_risk_controls': True
            },
            'expected_outcome': '在风险和收益间找到最佳平衡点'
        }
    
    def _simulate_parameter_performance(self, strategy: BaseStrategy,
                                      performance_data: List[StrategyPerformance],
                                      params: Dict[str, Any]) -> float:
        """模拟参数性能"""
        # 简化的参数性能模拟
        base_performance = self._evaluate_performance(performance_data, 'sharpe_ratio')
        
        # 基于参数计算性能调整因子
        adjustment_factor = 1.0
        
        for param_name, param_value in params.items():
            if param_name == 'learning_rate':
                # 学习率优化
                optimal_lr = 0.01
                adjustment_factor *= (1.0 - abs(param_value - optimal_lr) / optimal_lr)
            elif param_name == 'epsilon':
                # 探索率优化
                if param_value < 0.05 or param_value > 0.25:
                    adjustment_factor *= 0.9
            elif param_name == 'population_size':
                # 种群大小优化
                if 30 <= param_value <= 70:
                    adjustment_factor *= 1.1
        
        return base_performance * adjustment_factor
    
    def _calculate_drawdown(self, returns: List[float]) -> float:
        """计算回撤"""
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

class StrategyLearner:
    """策略学习器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.strategies = {}
        self.learning_phase = LearningPhase.EXPLORATION
        self.performance_analyzer = PerformanceAnalyzer()
        self.strategy_combiner = StrategyCombiner()
        self.knowledge_extractor = KnowledgeExtractor()
        self.strategy_optimizer = StrategyOptimizer()
        
        # 数据存储
        self.performance_history = defaultdict(list)
        self.learning_context_history = []
        self.knowledge_base = {}
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 初始化策略
        self._initialize_strategies()
        
        logger.info("策略学习器初始化完成")
    
    def _initialize_strategies(self):
        """初始化策略"""
        # 创建不同类型的策略
        rl_strategy = ReinforcementLearning("RL_Strategy_1")
        ea_strategy = EvolutionAlgorithm("EA_Strategy_1")
        il_strategy = ImitationLearning("IL_Strategy_1")
        
        # 添加专家数据（示例）
        expert_data = [
            {'state': {'market_signal': 0.7}, 'action': 'buy', 'features': {'momentum': 0.8}},
            {'state': {'market_signal': 0.3}, 'action': 'sell', 'features': {'momentum': -0.6}},
            {'state': {'market_signal': 0.5}, 'action': 'hold', 'features': {'momentum': 0.1}}
        ]
        il_strategy.add_expert_data(expert_data)
        
        # 注册策略
        self.strategies[rl_strategy.strategy_id] = rl_strategy
        self.strategies[ea_strategy.strategy_id] = ea_strategy
        self.strategies[il_strategy.strategy_id] = il_strategy
        
        # 激活策略
        for strategy in self.strategies.values():
            strategy.activate()
    
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """执行策略学习"""
        with self.lock:
            try:
                logger.info(f"开始策略学习，学习阶段: {self.learning_phase.value}")
                
                # 学习结果汇总
                learning_results = {}
                
                # 对每个策略执行学习
                for strategy_id, strategy in self.strategies.items():
                    if strategy.is_active:
                        result = strategy.learn(context)
                        learning_results[strategy_id] = result
                
                # 组合策略学习
                if len(self.strategies) > 1:
                    combination_result = self.strategy_combiner.combine_strategies(
                        list(self.strategies.values()), 
                        method='adaptive',
                        context=context
                    )
                    learning_results['combination'] = combination_result
                
                # 知识提取
                if self.performance_history:
                    knowledge = self.knowledge_extractor.extract_knowledge(
                        list(self.strategies.values()),
                        dict(self.performance_history)
                    )
                    learning_results['knowledge'] = knowledge
                    self.knowledge_base.update(knowledge)
                
                # 更新学习阶段
                self._update_learning_phase(context)
                
                # 记录学习上下文
                self.learning_context_history.append(context)
                
                logger.info("策略学习完成")
                return {
                    'learning_results': learning_results,
                    'current_phase': self.learning_phase.value,
                    'active_strategies': len([s for s in self.strategies.values() if s.is_active]),
                    'learning_timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"策略学习出错: {e}")
                return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any], method: str = 'adaptive') -> Dict[str, Any]:
        """执行策略预测"""
        with self.lock:
            try:
                logger.info(f"开始策略预测，方法: {method}")
                
                # 获取各策略预测
                predictions = {}
                for strategy_id, strategy in self.strategies.items():
                    if strategy.is_active:
                        prediction = strategy.predict(state)
                        predictions[strategy_id] = prediction
                
                if not predictions:
                    return {'error': '没有可用的活跃策略'}
                
                # 组合预测
                if len(predictions) > 1:
                    combination_result = self.strategy_combiner.combine_strategies(
                        [self.strategies[pid] for pid in predictions.keys()],
                        method=method
                    )
                    final_prediction = {
                        'combined_action': combination_result.get('combined_action', 'hold'),
                        'combined_confidence': combination_result.get('combined_confidence', 0.0),
                        'individual_predictions': predictions,
                        'combination_method': method
                    }
                else:
                    # 单个策略预测
                    strategy_id = list(predictions.keys())[0]
                    final_prediction = predictions[strategy_id].copy()
                    final_prediction['strategy_id'] = strategy_id
                
                # 添加元信息
                final_prediction.update({
                    'prediction_timestamp': datetime.now(),
                    'active_strategies': len(predictions),
                    'learning_phase': self.learning_phase.value
                })
                
                logger.info(f"策略预测完成: {final_prediction.get('combined_action', 'hold')}")
                return final_prediction
                
            except Exception as e:
                logger.error(f"策略预测出错: {e}")
                return {'error': str(e)}
    
    def update_performance(self, strategy_id: str, performance: StrategyPerformance):
        """更新策略性能"""
        with self.lock:
            try:
                if strategy_id in self.strategies:
                    strategy = self.strategies[strategy_id]
                    strategy.update_performance(performance)
                    
                    # 记录性能历史
                    self.performance_history[strategy_id].append(performance)
                    
                    # 定期优化
                    if len(self.performance_history[strategy_id]) % 10 == 0:
                        self._periodic_optimization(strategy_id)
                    
                    logger.info(f"策略 {strategy_id} 性能已更新")
                else:
                    logger.warning(f"策略 {strategy_id} 不存在")
                    
            except Exception as e:
                logger.error(f"更新策略性能出错: {e}")
    
    def analyze_performance(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """分析策略性能"""
        try:
            if strategy_id:
                if strategy_id in self.performance_history:
                    return self.performance_analyzer.analyze_performance(
                        strategy_id, 
                        self.performance_history[strategy_id]
                    )
                else:
                    return {'error': f'策略 {strategy_id} 没有性能数据'}
            else:
                # 分析所有策略
                all_performances = {}
                for sid, performances in self.performance_history.items():
                    if performances:
                        all_performances[sid] = performances
                
                if all_performances:
                    return self.performance_analyzer.compare_strategies(all_performances)
                else:
                    return {'error': '没有可用的性能数据'}
                    
        except Exception as e:
            logger.error(f"性能分析出错: {e}")
            return {'error': str(e)}
    
    def predict_future_performance(self, strategy_id: str, 
                                 prediction_horizon: int = 30) -> Dict[str, Any]:
        """预测策略未来性能"""
        try:
            if strategy_id in self.performance_history:
                return self.performance_analyzer.predict_future_performance(
                    strategy_id,
                    self.performance_history[strategy_id],
                    prediction_horizon
                )
            else:
                return {'error': f'策略 {strategy_id} 没有足够的历史数据'}
                
        except Exception as e:
            logger.error(f"性能预测出错: {e}")
            return {'error': str(e)}
    
    def optimize_strategies(self, optimization_objective: str = 'sharpe_ratio') -> Dict[str, Any]:
        """优化策略"""
        with self.lock:
            try:
                logger.info(f"开始策略优化，目标: {optimization_objective}")
                
                optimization_results = {}
                
                for strategy_id, strategy in self.strategies.items():
                    if strategy_id in self.performance_history and self.performance_history[strategy_id]:
                        # 优化超参数
                        hyperparam_result = self.strategy_optimizer.optimize_hyperparameters(
                            strategy,
                            self.performance_history[strategy_id],
                            optimization_objective
                        )
                        
                        # 优化学习率
                        learning_rate_result = self.strategy_optimizer.optimize_learning_rate(
                            strategy,
                            self.performance_history[strategy_id]
                        )
                        
                        optimization_results[strategy_id] = {
                            'hyperparameter_optimization': hyperparam_result,
                            'learning_rate_optimization': learning_rate_result
                        }
                
                # 优化组合权重
                if len(self.strategies) > 1:
                    weight_optimization = self.strategy_combiner.optimize_weights(
                        dict(self.performance_history),
                        optimization_objective
                    )
                    optimization_results['weight_optimization'] = weight_optimization
                
                logger.info("策略优化完成")
                return {
                    'optimization_results': optimization_results,
                    'optimization_timestamp': datetime.now(),
                    'objective': optimization_objective
                }
                
            except Exception as e:
                logger.error(f"策略优化出错: {e}")
                return {'error': str(e)}
    
    def get_knowledge_base(self) -> Dict[str, Any]:
        """获取知识库"""
        return {
            'knowledge_base': self.knowledge_base,
            'learning_history': len(self.learning_context_history),
            'performance_history': {sid: len(perfs) for sid, perfs in self.performance_history.items()},
            'active_strategies': len([s for s in self.strategies.values() if s.is_active]),
            'current_phase': self.learning_phase.value,
            'knowledge_timestamp': datetime.now()
        }
    
    def export_knowledge(self, filepath: str) -> bool:
        """导出知识库"""
        try:
            export_data = {
                'knowledge_base': self.knowledge_base,
                'strategies': {
                    sid: {
                        'strategy_type': strategy.strategy_type.value,
                        'state': strategy.get_state().__dict__,
                        'performance_history': [
                            {
                                'timestamp': p.timestamp.isoformat(),
                                'return_rate': p.return_rate,
                                'volatility': p.volatility,
                                'sharpe_ratio': p.sharpe_ratio,
                                'max_drawdown': p.max_drawdown,
                                'win_rate': p.win_rate,
                                'profit_factor': p.profit_factor
                            } for p in performances
                        ]
                    }
                    for sid, strategy in self.strategies.items()
                    for performances in [self.performance_history.get(sid, [])]
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"知识库已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出知识库出错: {e}")
            return False
    
    def _update_learning_phase(self, context: LearningContext):
        """更新学习阶段"""
        try:
            # 基于性能和环境动态调整学习阶段
            if self.performance_history:
                recent_performance = []
                for performances in self.performance_history.values():
                    if performances:
                        recent_performance.extend([p.return_rate for p in performances[-10:]])
                
                if recent_performance:
                    avg_recent_return = np.mean(recent_performance)
                    
                    if avg_recent_return > 0.02:  # 表现良好
                        if self.learning_phase == LearningPhase.EXPLORATION:
                            self.learning_phase = LearningPhase.EXPLOITATION
                    elif avg_recent_return < -0.01:  # 表现较差
                        if self.learning_phase == LearningPhase.EXPLOITATION:
                            self.learning_phase = LearningPhase.ADAPTATION
                    elif avg_recent_return < -0.05:  # 表现很差
                        self.learning_phase = LearningPhase.EVOLUTION
            
            # 基于环境变化调整
            market_volatility = context.environment_state.get('volatility', 0.1)
            if market_volatility > 0.3:
                self.learning_phase = LearningPhase.ADAPTATION
            
        except Exception as e:
            logger.error(f"更新学习阶段出错: {e}")
    
    def _periodic_optimization(self, strategy_id: str):
        """定期优化"""
        try:
            if strategy_id in self.strategies and strategy_id in self.performance_history:
                strategy = self.strategies[strategy_id]
                performances = self.performance_history[strategy_id]
                
                # 优化超参数
                optimization_result = self.strategy_optimizer.optimize_hyperparameters(
                    strategy,
                    performances,
                    'sharpe_ratio'
                )
                
                logger.info(f"策略 {strategy_id} 定期优化完成")
                
        except Exception as e:
            logger.error(f"定期优化出错: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'system_status': 'active',
            'active_strategies': len([s for s in self.strategies.values() if s.is_active]),
            'total_strategies': len(self.strategies),
            'current_learning_phase': self.learning_phase.value,
            'performance_data_points': sum(len(perfs) for perfs in self.performance_history.values()),
            'knowledge_base_size': len(self.knowledge_base),
            'last_update': datetime.now().isoformat(),
            'uptime_hours': (datetime.now() - getattr(self, 'start_time', datetime.now())).total_seconds() / 3600
        }

# 创建全局实例
_strategy_learner_instance = None

def get_strategy_learner(config: Optional[Dict[str, Any]] = None) -> StrategyLearner:
    """获取策略学习器实例（单例模式）"""
    global _strategy_learner_instance
    if _strategy_learner_instance is None:
        _strategy_learner_instance = StrategyLearner(config)
    return _strategy_learner_instance

# 使用示例
if __name__ == "__main__":
    # 创建策略学习器
    learner = get_strategy_learner()
    
    # 创建学习上下文
    context = LearningContext(
        environment_state={
            'market_signal': 0.6,
            'volatility': 0.15,
            'trend_strength': 0.7
        },
        historical_performance=[0.01, -0.02, 0.03, 0.01, -0.01],
        current_objective='maximize_sharpe_ratio',
        constraints={'max_drawdown': 0.1},
        risk_tolerance=0.6,
        time_horizon=100
    )
    
    # 执行学习
    learning_result = learner.learn(context)
    print("学习结果:", learning_result)
    
    # 执行预测
    prediction = learner.predict({'market_signal': 0.7, 'volatility': 0.1})
    print("预测结果:", prediction)
    
    # 更新性能
    performance = StrategyPerformance(
        strategy_id="RL_Strategy_1",
        timestamp=datetime.now(),
        return_rate=0.02,
        volatility=0.15,
        sharpe_ratio=1.2,
        max_drawdown=0.05,
        win_rate=0.6,
        profit_factor=1.5
    )
    learner.update_performance("RL_Strategy_1", performance)
    
    # 分析性能
    performance_analysis = learner.analyze_performance()
    print("性能分析:", performance_analysis)
    
    # 优化策略
    optimization_result = learner.optimize_strategies()
    print("优化结果:", optimization_result)
    
    # 获取知识库
    knowledge = learner.get_knowledge_base()
    print("知识库:", knowledge)