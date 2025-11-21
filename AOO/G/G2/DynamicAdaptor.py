"""
G2动态适应器
实现环境变化检测、适应性策略调整、动态参数优化等功能
"""

import json
import logging
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import sqlite3
import pickle
from concurrent.futures import ThreadPoolExecutor
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnvironmentState:
    """环境状态数据类"""
    timestamp: datetime
    market_conditions: Dict[str, float]
    system_performance: Dict[str, float]
    external_factors: Dict[str, Any]
    volatility_index: float
    trend_direction: str
    liquidity_level: float
    risk_level: float
    confidence_score: float = 0.0


@dataclass
class AdaptationAction:
    """适应性动作数据类"""
    action_id: str
    timestamp: datetime
    action_type: str
    parameters: Dict[str, Any]
    target_component: str
    priority: int
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    accuracy: float
    efficiency: float
    stability: float
    adaptability_score: float
    success_rate: float
    error_rate: float
    response_time: float
    resource_usage: Dict[str, float]


class EnvironmentMonitor:
    """环境变化检测器"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.environment_history = deque(maxlen=1000)
        self.change_detectors = {}
        self.alert_callbacks = []
        
        # 初始化检测器
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """初始化各种变化检测器"""
        self.change_detectors = {
            'volatility': VolatilityDetector(),
            'trend': TrendChangeDetector(),
            'liquidity': LiquidityDetector(),
            'risk': RiskLevelDetector(),
            'performance': PerformanceDetector()
        }
    
    def add_alert_callback(self, callback: Callable[[EnvironmentState, str], None]):
        """添加环境变化回调函数"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """开始环境监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("环境监控已启动")
    
    def stop_monitoring(self):
        """停止环境监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("环境监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                current_state = self._collect_environment_state()
                self.environment_history.append(current_state)
                
                # 检测变化
                changes = self._detect_changes(current_state)
                
                # 触发回调
                for change_type, change_data in changes.items():
                    for callback in self.alert_callbacks:
                        callback(current_state, change_type)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"环境监控错误: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_environment_state(self) -> EnvironmentState:
        """收集当前环境状态"""
        # 模拟环境数据收集
        return EnvironmentState(
            timestamp=datetime.now(),
            market_conditions={
                'price_volatility': np.random.normal(0.02, 0.01),
                'volume_trend': np.random.normal(1000, 200),
                'momentum': np.random.normal(0.5, 0.2)
            },
            system_performance={
                'cpu_usage': np.random.normal(50, 10),
                'memory_usage': np.random.normal(60, 15),
                'response_time': np.random.normal(100, 20)
            },
            external_factors={
                'market_sentiment': np.random.normal(0.5, 0.3),
                'news_impact': np.random.uniform(0, 1),
                'regulatory_changes': np.random.choice([0, 1], p=[0.9, 0.1])
            },
            volatility_index=np.random.normal(0.2, 0.05),
            trend_direction=np.random.choice(['up', 'down', 'sideways']),
            liquidity_level=np.random.normal(0.7, 0.15),
            risk_level=np.random.normal(0.3, 0.1)
        )
    
    def _detect_changes(self, current_state: EnvironmentState) -> Dict[str, Any]:
        """检测环境变化"""
        changes = {}
        
        if len(self.environment_history) < 2:
            return changes
        
        previous_state = self.environment_history[-2]
        
        # 使用各种检测器检测变化
        for detector_name, detector in self.change_detectors.items():
            try:
                change_result = detector.detect(previous_state, current_state)
                if change_result['changed']:
                    changes[detector_name] = change_result
            except Exception as e:
                logger.error(f"检测器 {detector_name} 错误: {e}")
        
        return changes
    
    def get_current_state(self) -> Optional[EnvironmentState]:
        """获取当前环境状态"""
        return self.environment_history[-1] if self.environment_history else None
    
    def get_state_history(self, hours: int = 24) -> List[EnvironmentState]:
        """获取历史环境状态"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [state for state in self.environment_history 
                if state.timestamp >= cutoff_time]


class BaseDetector(ABC):
    """变化检测器基类"""
    
    @abstractmethod
    def detect(self, previous_state: EnvironmentState, 
               current_state: EnvironmentState) -> Dict[str, Any]:
        """检测变化"""
        pass


class VolatilityDetector(BaseDetector):
    """波动率变化检测器"""
    
    def detect(self, previous_state: EnvironmentState, 
               current_state: EnvironmentState) -> Dict[str, Any]:
        volatility_change = abs(current_state.volatility_index - 
                               previous_state.volatility_index)
        
        return {
            'changed': volatility_change > 0.05,
            'change_magnitude': volatility_change,
            'previous_value': previous_state.volatility_index,
            'current_value': current_state.volatility_index,
            'threshold': 0.05
        }


class TrendChangeDetector(BaseDetector):
    """趋势变化检测器"""
    
    def detect(self, previous_state: EnvironmentState, 
               current_state: EnvironmentState) -> Dict[str, Any]:
        trend_changed = previous_state.trend_direction != current_state.trend_direction
        
        return {
            'changed': trend_changed,
            'previous_trend': previous_state.trend_direction,
            'current_trend': current_state.trend_direction
        }


class LiquidityDetector(BaseDetector):
    """流动性变化检测器"""
    
    def detect(self, previous_state: EnvironmentState, 
               current_state: EnvironmentState) -> Dict[str, Any]:
        liquidity_change = abs(current_state.liquidity_level - 
                              previous_state.liquidity_level)
        
        return {
            'changed': liquidity_change > 0.1,
            'change_magnitude': liquidity_change,
            'previous_value': previous_state.liquidity_level,
            'current_value': current_state.liquidity_level,
            'threshold': 0.1
        }


class RiskLevelDetector(BaseDetector):
    """风险水平变化检测器"""
    
    def detect(self, previous_state: EnvironmentState, 
               current_state: EnvironmentState) -> Dict[str, Any]:
        risk_change = abs(current_state.risk_level - previous_state.risk_level)
        
        return {
            'changed': risk_change > 0.15,
            'change_magnitude': risk_change,
            'previous_value': previous_state.risk_level,
            'current_value': current_state.risk_level,
            'threshold': 0.15
        }


class PerformanceDetector(BaseDetector):
    """性能变化检测器"""
    
    def detect(self, previous_state: EnvironmentState, 
               current_state: EnvironmentState) -> Dict[str, Any]:
        # 检测系统性能变化
        cpu_change = abs(current_state.system_performance['cpu_usage'] - 
                        previous_state.system_performance['cpu_usage'])
        memory_change = abs(current_state.system_performance['memory_usage'] - 
                           previous_state.system_performance['memory_usage'])
        
        performance_changed = cpu_change > 10 or memory_change > 15
        
        return {
            'changed': performance_changed,
            'cpu_change': cpu_change,
            'memory_change': memory_change,
            'previous_cpu': previous_state.system_performance['cpu_usage'],
            'current_cpu': current_state.system_performance['cpu_usage']
        }


class AdaptationStrategy:
    """适应性策略管理器"""
    
    def __init__(self):
        self.strategies = {}
        self.active_strategies = {}
        self.strategy_performance = defaultdict(list)
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """初始化适应性策略"""
        self.strategies = {
            'conservative': ConservativeStrategy(),
            'aggressive': AggressiveStrategy(),
            'balanced': BalancedStrategy(),
            'adaptive': AdaptiveStrategy(),
            'emergency': EmergencyStrategy()
        }
    
    def select_strategy(self, environment_state: EnvironmentState, 
                       change_type: str) -> str:
        """根据环境状态选择策略"""
        # 基于环境特征选择最适合的策略
        if environment_state.risk_level > 0.7:
            return 'conservative'
        elif environment_state.volatility_index > 0.3:
            return 'adaptive'
        elif environment_state.trend_direction == 'sideways':
            return 'balanced'
        elif environment_state.liquidity_level < 0.3:
            return 'emergency'
        else:
            return 'aggressive'
    
    def execute_strategy(self, strategy_name: str, environment_state: EnvironmentState,
                        change_data: Dict[str, Any]) -> AdaptationAction:
        """执行适应性策略"""
        if strategy_name not in self.strategies:
            raise ValueError(f"未知策略: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        action = strategy.execute(environment_state, change_data)
        
        # 记录策略执行
        self.strategy_performance[strategy_name].append({
            'timestamp': datetime.now(),
            'action': action,
            'environment_state': environment_state
        })
        
        return action
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, float]:
        """获取策略性能统计"""
        if strategy_name not in self.strategy_performance:
            return {}
        
        performances = self.strategy_performance[strategy_name]
        if not performances:
            return {}
        
        # 计算平均性能指标
        success_rates = [p['action'].result.get('success_rate', 0) 
                        for p in performances if p['action'].result]
        
        return {
            'total_executions': len(performances),
            'average_success_rate': np.mean(success_rates) if success_rates else 0,
            'recent_performance': np.mean(success_rates[-10:]) if len(success_rates) >= 10 else 0
        }


class BaseStrategy(ABC):
    """策略基类"""
    
    @abstractmethod
    def execute(self, environment_state: EnvironmentState, 
                change_data: Dict[str, Any]) -> AdaptationAction:
        """执行策略"""
        pass


class ConservativeStrategy(BaseStrategy):
    """保守策略"""
    
    def execute(self, environment_state: EnvironmentState, 
                change_data: Dict[str, Any]) -> AdaptationAction:
        # 保守策略：减少风险暴露，降低参数敏感度
        return AdaptationAction(
            action_id=f"conservative_{int(time.time())}",
            timestamp=datetime.now(),
            action_type="parameter_adjustment",
            parameters={
                'risk_reduction_factor': 0.8,
                'sensitivity_threshold': 0.9,
                'position_size_limit': 0.5
            },
            target_component="risk_management",
            priority=3
        )


class AggressiveStrategy(BaseStrategy):
    """激进策略"""
    
    def execute(self, environment_state: EnvironmentState, 
                change_data: Dict[str, Any]) -> AdaptationAction:
        # 激进策略：增加参数敏感度，扩大操作范围
        return AdaptationAction(
            action_id=f"aggressive_{int(time.time())}",
            timestamp=datetime.now(),
            action_type="parameter_adjustment",
            parameters={
                'sensitivity_multiplier': 1.5,
                'operation_range_expansion': 1.3,
                'response_speed': 1.2
            },
            target_component="operation_control",
            priority=2
        )


class BalancedStrategy(BaseStrategy):
    """平衡策略"""
    
    def execute(self, environment_state: EnvironmentState, 
                change_data: Dict[str, Any]) -> AdaptationAction:
        # 平衡策略：适度调整参数
        return AdaptationAction(
            action_id=f"balanced_{int(time.time())}",
            timestamp=datetime.now(),
            action_type="parameter_adjustment",
            parameters={
                'adjustment_factor': 1.0,
                'stability_boost': 1.1,
                'efficiency_optimization': 1.05
            },
            target_component="general_optimization",
            priority=2
        )


class AdaptiveStrategy(BaseStrategy):
    """自适应策略"""
    
    def execute(self, environment_state: EnvironmentState, 
                change_data: Dict[str, Any]) -> AdaptationAction:
        # 自适应策略：根据变化类型动态调整
        change_magnitude = change_data.get('change_magnitude', 0)
        
        return AdaptationAction(
            action_id=f"adaptive_{int(time.time())}",
            timestamp=datetime.now(),
            action_type="dynamic_adjustment",
            parameters={
                'adjustment_intensity': min(change_magnitude * 2, 1.0),
                'feedback_loop_enabled': True,
                'continuous_monitoring': True
            },
            target_component="adaptive_control",
            priority=1
        )


class EmergencyStrategy(BaseStrategy):
    """紧急策略"""
    
    def execute(self, environment_state: EnvironmentState, 
                change_data: Dict[str, Any]) -> AdaptationAction:
        # 紧急策略：快速响应和风险控制
        return AdaptationAction(
            action_id=f"emergency_{int(time.time())}",
            timestamp=datetime.now(),
            action_type="emergency_response",
            parameters={
                'immediate_stop_loss': True,
                'position_reduction': 0.7,
                'risk_freeze': True,
                'alert_level': 'critical'
            },
            target_component="emergency_control",
            priority=0
        )


class ParameterOptimizer:
    """动态参数优化器"""
    
    def __init__(self, optimization_interval: float = 5.0):
        self.optimization_interval = optimization_interval
        self.current_parameters = {}
        self.parameter_history = deque(maxlen=500)
        self.optimization_algorithms = {
            'genetic': GeneticOptimizer(),
            'gradient': GradientOptimizer(),
            'bayesian': BayesianOptimizer(),
            'grid_search': GridSearchOptimizer()
        }
        self.optimization_lock = threading.Lock()
    
    def optimize_parameters(self, target_metrics: Dict[str, float],
                           environment_state: EnvironmentState,
                           algorithm: str = 'adaptive') -> Dict[str, Any]:
        """优化参数"""
        with self.optimization_lock:
            try:
                # 选择优化算法
                if algorithm == 'adaptive':
                    algorithm = self._select_optimal_algorithm(environment_state)
                
                optimizer = self.optimization_algorithms[algorithm]
                
                # 执行优化
                optimization_result = optimizer.optimize(
                    current_params=self.current_parameters,
                    target_metrics=target_metrics,
                    environment_state=environment_state
                )
                
                # 更新参数
                if optimization_result['success']:
                    old_params = self.current_parameters.copy()
                    self.current_parameters.update(optimization_result['new_parameters'])
                    
                    # 记录优化历史
                    self.parameter_history.append({
                        'timestamp': datetime.now(),
                        'algorithm': algorithm,
                        'old_parameters': old_params,
                        'new_parameters': self.current_parameters.copy(),
                        'improvement': optimization_result.get('improvement', 0),
                        'environment_state': environment_state
                    })
                
                return optimization_result
                
            except Exception as e:
                logger.error(f"参数优化错误: {e}")
                return {'success': False, 'error': str(e)}
    
    def _select_optimal_algorithm(self, environment_state: EnvironmentState) -> str:
        """选择最优优化算法"""
        if environment_state.volatility_index > 0.25:
            return 'bayesian'
        elif environment_state.risk_level > 0.6:
            return 'genetic'
        elif len(self.parameter_history) < 10:
            return 'grid_search'
        else:
            return 'gradient'
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """获取当前参数"""
        return self.current_parameters.copy()
    
    def set_parameter(self, name: str, value: Any):
        """设置特定参数"""
        with self.optimization_lock:
            self.current_parameters[name] = value
    
    def get_optimization_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return list(self.parameter_history)[-limit:]


class BaseOptimizer(ABC):
    """优化器基类"""
    
    @abstractmethod
    def optimize(self, current_params: Dict[str, Any], 
                 target_metrics: Dict[str, float],
                 environment_state: EnvironmentState) -> Dict[str, Any]:
        """执行优化"""
        pass


class GeneticOptimizer(BaseOptimizer):
    """遗传算法优化器"""
    
    def optimize(self, current_params: Dict[str, Any], 
                 target_metrics: Dict[str, float],
                 environment_state: EnvironmentState) -> Dict[str, Any]:
        # 简化的遗传算法实现
        generations = 50
        population_size = 20
        
        # 初始化种群
        population = self._initialize_population(current_params, population_size)
        
        for generation in range(generations):
            # 评估适应度
            fitness_scores = [self._evaluate_fitness(individual, target_metrics) 
                            for individual in population]
            
            # 选择、交叉、变异
            population = self._evolve_population(population, fitness_scores)
        
        # 返回最优解
        best_individual = max(population, key=lambda x: self._evaluate_fitness(x, target_metrics))
        
        return {
            'success': True,
            'new_parameters': best_individual,
            'algorithm': 'genetic',
            'generations': generations,
            'improvement': max(fitness_scores) - min(fitness_scores)
        }
    
    def _initialize_population(self, base_params: Dict[str, Any], size: int) -> List[Dict[str, Any]]:
        """初始化种群"""
        population = []
        for _ in range(size):
            individual = {}
            for key, value in base_params.items():
                if isinstance(value, (int, float)):
                    individual[key] = value + np.random.normal(0, value * 0.1)
                else:
                    individual[key] = value
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, individual: Dict[str, Any], 
                         target_metrics: Dict[str, float]) -> float:
        """评估适应度"""
        # 简化的适应度函数
        fitness = 0
        for metric, target in target_metrics.items():
            if metric in individual:
                fitness += 1 / (1 + abs(individual[metric] - target))
        return fitness
    
    def _evolve_population(self, population: List[Dict[str, Any]], 
                          fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """进化种群"""
        # 简化的进化操作
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), 
                                                key=lambda pair: pair[0], reverse=True)]
        
        # 选择前50%作为父代
        parents = sorted_population[:len(population)//2]
        
        # 生成新种群
        new_population = parents.copy()
        while len(new_population) < len(population):
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)
        
        return new_population
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """交叉操作"""
        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if np.random.random() > 0.5 else parent2[key]
        return child
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()
        for key, value in mutated.items():
            if isinstance(value, (int, float)) and np.random.random() < 0.1:
                mutated[key] = value + np.random.normal(0, value * 0.05)
        return mutated


class GradientOptimizer(BaseOptimizer):
    """梯度下降优化器"""
    
    def optimize(self, current_params: Dict[str, Any], 
                 target_metrics: Dict[str, float],
                 environment_state: EnvironmentState) -> Dict[str, Any]:
        # 简化的梯度下降实现
        learning_rate = 0.01
        max_iterations = 100
        
        params = current_params.copy()
        
        for iteration in range(max_iterations):
            # 计算梯度
            gradients = self._compute_gradients(params, target_metrics)
            
            # 更新参数
            for key, gradient in gradients.items():
                if key in params and isinstance(params[key], (int, float)):
                    params[key] -= learning_rate * gradient
            
            # 检查收敛
            if np.all(np.abs(list(gradients.values())) < 1e-6):
                break
        
        return {
            'success': True,
            'new_parameters': params,
            'algorithm': 'gradient',
            'iterations': iteration + 1,
            'improvement': self._evaluate_fitness(params, target_metrics)
        }
    
    def _compute_gradients(self, params: Dict[str, Any], 
                          target_metrics: Dict[str, float]) -> Dict[str, float]:
        """计算梯度"""
        gradients = {}
        epsilon = 1e-8
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # 数值梯度计算
                original_value = value
                params[key] = original_value + epsilon
                forward_fitness = self._evaluate_fitness(params, target_metrics)
                
                params[key] = original_value - epsilon
                backward_fitness = self._evaluate_fitness(params, target_metrics)
                
                gradients[key] = (forward_fitness - backward_fitness) / (2 * epsilon)
                params[key] = original_value
            else:
                gradients[key] = 0
        
        return gradients
    
    def _evaluate_fitness(self, individual: Dict[str, Any], 
                         target_metrics: Dict[str, float]) -> float:
        """评估适应度"""
        fitness = 0
        for metric, target in target_metrics.items():
            if metric in individual:
                fitness += 1 / (1 + abs(individual[metric] - target))
        return fitness


class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器"""
    
    def optimize(self, current_params: Dict[str, Any], 
                 target_metrics: Dict[str, float],
                 environment_state: EnvironmentState) -> Dict[str, Any]:
        # 简化的贝叶斯优化实现
        # 在实际应用中，这里会使用高斯过程等更复杂的方法
        
        # 基于历史性能调整参数
        adjusted_params = current_params.copy()
        
        for key, value in adjusted_params.items():
            if isinstance(value, (int, float)):
                # 根据目标指标调整
                if key in target_metrics:
                    target = target_metrics[key]
                    adjustment = (target - value) * 0.1  # 小的调整步长
                    adjusted_params[key] = value + adjustment
        
        return {
            'success': True,
            'new_parameters': adjusted_params,
            'algorithm': 'bayesian',
            'improvement': self._evaluate_fitness(adjusted_params, target_metrics)
        }
    
    def _evaluate_fitness(self, individual: Dict[str, Any], 
                         target_metrics: Dict[str, float]) -> float:
        """评估适应度"""
        fitness = 0
        for metric, target in target_metrics.items():
            if metric in individual:
                fitness += 1 / (1 + abs(individual[metric] - target))
        return fitness


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def optimize(self, current_params: Dict[str, Any], 
                 target_metrics: Dict[str, float],
                 environment_state: EnvironmentState) -> Dict[str, Any]:
        # 简化的网格搜索实现
        best_params = current_params.copy()
        best_fitness = self._evaluate_fitness(best_params, target_metrics)
        
        # 搜索邻域参数
        search_range = 0.2  # 20%的搜索范围
        
        for key, value in current_params.items():
            if isinstance(value, (int, float)):
                for multiplier in [0.9, 0.95, 1.0, 1.05, 1.1]:
                    test_params = current_params.copy()
                    test_params[key] = value * multiplier
                    
                    fitness = self._evaluate_fitness(test_params, target_metrics)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_params = test_params.copy()
        
        return {
            'success': True,
            'new_parameters': best_params,
            'algorithm': 'grid_search',
            'improvement': best_fitness
        }
    
    def _evaluate_fitness(self, individual: Dict[str, Any], 
                         target_metrics: Dict[str, float]) -> float:
        """评估适应度"""
        fitness = 0
        for metric, target in target_metrics.items():
            if metric in individual:
                fitness += 1 / (1 + abs(individual[metric] - target))
        return fitness


class LearningEngine:
    """适应性学习引擎"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.learning_history = deque(maxlen=1000)
        self.learned_patterns = {}
        self.model_performance = defaultdict(list)
        self.feature_importance = defaultdict(float)
    
    def learn_from_environment(self, environment_state: EnvironmentState,
                              action_result: AdaptationAction) -> Dict[str, Any]:
        """从环境和动作结果中学习"""
        try:
            # 提取特征
            features = self._extract_features(environment_state, action_result)
            
            # 更新学习模式
            self._update_patterns(features, action_result)
            
            # 更新模型性能
            self._update_model_performance(action_result)
            
            # 计算特征重要性
            self._update_feature_importance(features, action_result)
            
            # 记录学习历史
            learning_record = {
                'timestamp': datetime.now(),
                'environment_state': environment_state,
                'action_result': action_result,
                'features': features,
                'patterns_updated': len(self.learned_patterns)
            }
            self.learning_history.append(learning_record)
            
            return {
                'success': True,
                'patterns_count': len(self.learned_patterns),
                'learning_rate': self.learning_rate,
                'feature_importance': dict(self.feature_importance)
            }
            
        except Exception as e:
            logger.error(f"学习过程错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_optimal_action(self, environment_state: EnvironmentState) -> Dict[str, Any]:
        """预测最优动作"""
        try:
            # 提取当前特征
            current_features = self._extract_simple_features(environment_state)
            
            # 查找相似模式
            similar_patterns = self._find_similar_patterns(current_features)
            
            if not similar_patterns:
                return {'action': 'balanced', 'confidence': 0.5}
            
            # 基于历史成功模式推荐动作
            best_pattern = max(similar_patterns, key=lambda x: x['success_rate'])
            
            return {
                'action': best_pattern['action_type'],
                'confidence': best_pattern['success_rate'],
                'reasoning': best_pattern['reasoning'],
                'similar_patterns_count': len(similar_patterns)
            }
            
        except Exception as e:
            logger.error(f"预测过程错误: {e}")
            return {'action': 'balanced', 'confidence': 0.3}
    
    def _extract_features(self, environment_state: EnvironmentState,
                         action_result: AdaptationAction) -> Dict[str, Any]:
        """提取特征"""
        return {
            'volatility_level': environment_state.volatility_index,
            'risk_level': environment_state.risk_level,
            'liquidity_level': environment_state.liquidity_level,
            'trend_direction': environment_state.trend_direction,
            'action_type': action_result.action_type,
            'action_priority': action_result.priority,
            'success_rate': action_result.result.get('success_rate', 0) if action_result.result else 0
        }
    
    def _extract_simple_features(self, environment_state: EnvironmentState) -> Dict[str, Any]:
        """提取简化特征"""
        return {
            'volatility': environment_state.volatility_index,
            'risk': environment_state.risk_level,
            'liquidity': environment_state.liquidity_level,
            'trend': environment_state.trend_direction
        }
    
    def _update_patterns(self, features: Dict[str, Any], 
                        action_result: AdaptationAction):
        """更新学习模式"""
        pattern_key = f"{features['action_type']}_{features['trend_direction']}"
        
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                'action_type': features['action_type'],
                'success_count': 0,
                'total_count': 0,
                'success_rate': 0,
                'avg_volatility': [],
                'avg_risk': [],
                'reasoning': f"基于{features['trend_direction']}趋势的{features['action_type']}策略"
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern['total_count'] += 1
        
        if action_result.result and action_result.result.get('success_rate', 0) > 0.5:
            pattern['success_count'] += 1
        
        pattern['success_rate'] = pattern['success_count'] / pattern['total_count']
        pattern['avg_volatility'].append(features['volatility_level'])
        pattern['avg_risk'].append(features['risk_level'])
    
    def _update_model_performance(self, action_result: AdaptationAction):
        """更新模型性能"""
        if action_result.result:
            performance_score = action_result.result.get('success_rate', 0)
            self.model_performance['overall'].append({
                'timestamp': datetime.now(),
                'score': performance_score
            })
    
    def _update_feature_importance(self, features: Dict[str, Any], 
                                  action_result: AdaptationAction):
        """更新特征重要性"""
        if action_result.result:
            success = action_result.result.get('success_rate', 0) > 0.5
            for feature, value in features.items():
                if isinstance(value, (int, float)):
                    # 简化的重要性更新
                    importance_change = 0.01 if success else -0.01
                    self.feature_importance[feature] += importance_change
    
    def _find_similar_patterns(self, current_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找相似模式"""
        similar_patterns = []
        
        for pattern_key, pattern_data in self.learned_patterns.items():
            similarity = self._calculate_similarity(current_features, pattern_data)
            if similarity > 0.7:  # 相似度阈值
                similar_patterns.append({
                    'action_type': pattern_data['action_type'],
                    'success_rate': pattern_data['success_rate'],
                    'reasoning': pattern_data['reasoning'],
                    'similarity': similarity
                })
        
        return sorted(similar_patterns, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_similarity(self, current_features: Dict[str, Any], 
                             pattern_data: Dict[str, Any]) -> float:
        """计算特征相似度"""
        # 简化的相似度计算
        similarity = 0
        
        if 'volatility' in current_features and pattern_data['avg_volatility']:
            avg_volatility = np.mean(pattern_data['avg_volatility'])
            volatility_diff = abs(current_features['volatility'] - avg_volatility)
            similarity += max(0, 1 - volatility_diff)
        
        if 'risk' in current_features and pattern_data['avg_risk']:
            avg_risk = np.mean(pattern_data['avg_risk'])
            risk_diff = abs(current_features['risk'] - avg_risk)
            similarity += max(0, 1 - risk_diff)
        
        return similarity / 2  # 归一化
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        total_patterns = len(self.learned_patterns)
        total_learning_records = len(self.learning_history)
        
        avg_success_rate = 0
        if self.model_performance['overall']:
            scores = [record['score'] for record in self.model_performance['overall']]
            avg_success_rate = np.mean(scores)
        
        return {
            'total_patterns': total_patterns,
            'total_learning_records': total_learning_records,
            'average_success_rate': avg_success_rate,
            'feature_importance': dict(self.feature_importance),
            'recent_performance': self.model_performance['overall'][-10:] if self.model_performance['overall'] else []
        }


class EffectEvaluator:
    """适应性效果评估器"""
    
    def __init__(self, evaluation_window: int = 100):
        self.evaluation_window = evaluation_window
        self.evaluation_history = deque(maxlen=500)
        self.performance_metrics = defaultdict(list)
        self.evaluation_criteria = {
            'accuracy': 0.3,
            'efficiency': 0.25,
            'stability': 0.2,
            'adaptability': 0.15,
            'resource_usage': 0.1
        }
    
    def evaluate_adaptation_effect(self, before_state: EnvironmentState,
                                  after_state: EnvironmentState,
                                  action: AdaptationAction,
                                  target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """评估适应性效果"""
        try:
            # 计算各项指标改善情况
            improvements = self._calculate_improvements(before_state, after_state, target_metrics)
            
            # 计算综合评分
            overall_score = self._calculate_overall_score(improvements)
            
            # 评估动作执行效果
            action_effectiveness = self._evaluate_action_effectiveness(action, improvements)
            
            # 生成评估报告
            evaluation_result = {
                'timestamp': datetime.now(),
                'overall_score': overall_score,
                'improvements': improvements,
                'action_effectiveness': action_effectiveness,
                'recommendations': self._generate_recommendations(improvements, overall_score),
                'before_state': before_state,
                'after_state': after_state,
                'action_taken': action
            }
            
            # 记录评估历史
            self.evaluation_history.append(evaluation_result)
            
            # 更新性能指标
            self.performance_metrics['overall_scores'].append(overall_score)
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"效果评估错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_improvements(self, before_state: EnvironmentState,
                               after_state: EnvironmentState,
                               target_metrics: Dict[str, float]) -> Dict[str, float]:
        """计算各项指标改善情况"""
        improvements = {}
        
        # 波动率改善
        volatility_improvement = (before_state.volatility_index - after_state.volatility_index) / before_state.volatility_index
        improvements['volatility_control'] = max(0, volatility_improvement)
        
        # 风险水平改善
        risk_improvement = (before_state.risk_level - after_state.risk_level) / before_state.risk_level
        improvements['risk_reduction'] = max(0, risk_improvement)
        
        # 流动性改善
        liquidity_improvement = (after_state.liquidity_level - before_state.liquidity_level) / before_state.liquidity_level
        improvements['liquidity_enhancement'] = max(0, liquidity_improvement)
        
        # 系统性能改善
        cpu_improvement = (before_state.system_performance['cpu_usage'] - after_state.system_performance['cpu_usage']) / before_state.system_performance['cpu_usage']
        improvements['cpu_efficiency'] = max(0, cpu_improvement)
        
        memory_improvement = (before_state.system_performance['memory_usage'] - after_state.system_performance['memory_usage']) / before_state.system_performance['memory_usage']
        improvements['memory_efficiency'] = max(0, memory_improvement)
        
        return improvements
    
    def _calculate_overall_score(self, improvements: Dict[str, float]) -> float:
        """计算综合评分"""
        weighted_score = 0
        
        for criterion, weight in self.evaluation_criteria.items():
            if criterion in improvements:
                weighted_score += improvements[criterion] * weight
            else:
                # 使用默认改善值
                weighted_score += 0.5 * weight
        
        return min(1.0, max(0.0, weighted_score))
    
    def _evaluate_action_effectiveness(self, action: AdaptationAction,
                                      improvements: Dict[str, float]) -> Dict[str, Any]:
        """评估动作执行效果"""
        # 基于改善情况评估动作效果
        avg_improvement = np.mean(list(improvements.values()))
        
        effectiveness_score = avg_improvement
        if action.result:
            success_rate = action.result.get('success_rate', 0)
            effectiveness_score = (effectiveness_score + success_rate) / 2
        
        return {
            'effectiveness_score': effectiveness_score,
            'success_indicators': sum(1 for v in improvements.values() if v > 0),
            'total_indicators': len(improvements),
            'action_type': action.action_type,
            'execution_time': action.execution_time
        }
    
    def _generate_recommendations(self, improvements: Dict[str, float], 
                                 overall_score: float) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if overall_score < 0.3:
            recommendations.append("建议重新评估适应性策略，当前效果不佳")
        
        if improvements.get('volatility_control', 0) < 0:
            recommendations.append("波动率控制需要加强，建议调整敏感度参数")
        
        if improvements.get('risk_reduction', 0) < 0:
            recommendations.append("风险水平未得到有效控制，建议采用更保守的策略")
        
        if improvements.get('cpu_efficiency', 0) < 0:
            recommendations.append("系统资源使用效率有待提升，建议优化算法性能")
        
        if len(recommendations) == 0:
            recommendations.append("当前适应性效果良好，建议继续保持现有策略")
        
        return recommendations
    
    def get_evaluation_summary(self, days: int = 7) -> Dict[str, Any]:
        """获取评估摘要"""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_evaluations = [eval_data for eval_data in self.evaluation_history 
                             if eval_data['timestamp'] >= cutoff_time]
        
        if not recent_evaluations:
            return {'message': '没有最近的评估数据'}
        
        scores = [eval_data['overall_score'] for eval_data in recent_evaluations]
        
        return {
            'evaluation_period_days': days,
            'total_evaluations': len(recent_evaluations),
            'average_score': np.mean(scores),
            'best_score': max(scores),
            'worst_score': min(scores),
            'score_trend': 'improving' if scores[-1] > scores[0] else 'declining',
            'recommendation_frequency': self._analyze_recommendations(recent_evaluations)
        }
    
    def _analyze_recommendations(self, evaluations: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析建议频率"""
        recommendation_counts = defaultdict(int)
        
        for evaluation in evaluations:
            for recommendation in evaluation.get('recommendations', []):
                recommendation_counts[recommendation] += 1
        
        return dict(recommendation_counts)


class HistoryTracker:
    """适应性历史跟踪器"""
    
    def __init__(self, db_path: str = "adaptation_history.db"):
        self.db_path = db_path
        self.init_database()
        self.memory_cache = deque(maxlen=1000)
        self.tracking_lock = threading.Lock()
    
    def init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS adaptation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    environment_state TEXT,
                    action_taken TEXT,
                    action_result TEXT,
                    evaluation_result TEXT,
                    performance_metrics TEXT,
                    parameters TEXT,
                    success_rate REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parameter_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    parameter_name TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    change_reason TEXT,
                    environment_context TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"数据库初始化错误: {e}")
    
    def track_adaptation_event(self, environment_state: EnvironmentState,
                              action: AdaptationAction,
                              evaluation_result: Dict[str, Any],
                              performance_metrics: PerformanceMetrics):
        """跟踪适应性事件"""
        with self.tracking_lock:
            try:
                # 保存到数据库
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO adaptation_history 
                    (timestamp, environment_state, action_taken, action_result, 
                     evaluation_result, performance_metrics, parameters, success_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    json.dumps(environment_state.__dict__, default=str),
                    json.dumps(action.__dict__, default=str),
                    json.dumps(action.result) if action.result else None,
                    json.dumps(evaluation_result, default=str),
                    json.dumps(performance_metrics.__dict__, default=str),
                    json.dumps({}),  # 参数信息
                    action.result.get('success_rate', 0) if action.result else 0
                ))
                
                conn.commit()
                conn.close()
                
                # 添加到内存缓存
                event_data = {
                    'timestamp': datetime.now(),
                    'environment_state': environment_state,
                    'action': action,
                    'evaluation': evaluation_result,
                    'performance': performance_metrics
                }
                self.memory_cache.append(event_data)
                
                logger.info(f"适应性事件已记录: {action.action_id}")
                
            except Exception as e:
                logger.error(f"跟踪事件错误: {e}")
    
    def track_parameter_change(self, parameter_name: str, old_value: Any,
                              new_value: Any, change_reason: str,
                              environment_context: Dict[str, Any]):
        """跟踪参数变化"""
        with self.tracking_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO parameter_evolution 
                    (timestamp, parameter_name, old_value, new_value, 
                     change_reason, environment_context)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    parameter_name,
                    str(old_value),
                    str(new_value),
                    change_reason,
                    json.dumps(environment_context, default=str)
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"参数变化跟踪错误: {e}")
    
    def get_adaptation_history(self, days: int = 30, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """获取适应性历史"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT * FROM adaptation_history 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (cutoff_time, limit))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            history_data = []
            for row in rows:
                record = dict(zip(columns, row))
                # 解析JSON字段
                for field in ['environment_state', 'action_taken', 'action_result', 
                             'evaluation_result', 'performance_metrics', 'parameters']:
                    if record[field]:
                        try:
                            record[field] = json.loads(record[field])
                        except:
                            pass
                history_data.append(record)
            
            conn.close()
            return history_data
            
        except Exception as e:
            logger.error(f"获取历史数据错误: {e}")
            return []
    
    def get_parameter_evolution(self, parameter_name: str = None,
                               days: int = 30) -> List[Dict[str, Any]]:
        """获取参数演化历史"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(days=days)
            
            if parameter_name:
                cursor.execute('''
                    SELECT * FROM parameter_evolution 
                    WHERE timestamp > ? AND parameter_name = ?
                    ORDER BY timestamp DESC
                ''', (cutoff_time, parameter_name))
            else:
                cursor.execute('''
                    SELECT * FROM parameter_evolution 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            evolution_data = []
            for row in rows:
                record = dict(zip(columns, row))
                # 解析JSON字段
                if record['environment_context']:
                    try:
                        record['environment_context'] = json.loads(record['environment_context'])
                    except:
                        pass
                evolution_data.append(record)
            
            conn.close()
            return evolution_data
            
        except Exception as e:
            logger.error(f"获取参数演化错误: {e}")
            return []
    
    def generate_analytics_report(self, days: int = 30) -> Dict[str, Any]:
        """生成分析报告"""
        try:
            history_data = self.get_adaptation_history(days)
            
            if not history_data:
                return {'message': '没有足够的历史数据进行分析'}
            
            # 计算统计指标
            total_adaptations = len(history_data)
            successful_adaptations = sum(1 for record in history_data 
                                       if record.get('success_rate', 0) > 0.5)
            
            success_rate = successful_adaptations / total_adaptations if total_adaptations > 0 else 0
            
            # 分析趋势
            recent_records = history_data[:10]  # 最近10条记录
            recent_success_rates = [record.get('success_rate', 0) for record in recent_records]
            
            trend = 'improving' if len(recent_success_rates) >= 2 and recent_success_rates[0] > recent_success_rates[-1] else 'stable'
            
            # 分析常用动作类型
            action_types = defaultdict(int)
            for record in history_data:
                if record.get('action_taken'):
                    try:
                        action_data = json.loads(record['action_taken'])
                        action_types[action_data.get('action_type', 'unknown')] += 1
                    except:
                        pass
            
            return {
                'analysis_period_days': days,
                'total_adaptations': total_adaptations,
                'success_rate': success_rate,
                'trend': trend,
                'most_common_actions': dict(sorted(action_types.items(), key=lambda x: x[1], reverse=True)),
                'average_success_rate': np.mean([record.get('success_rate', 0) for record in history_data]),
                'performance_summary': self._calculate_performance_summary(history_data)
            }
            
        except Exception as e:
            logger.error(f"生成分析报告错误: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_summary(self, history_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算性能摘要"""
        if not history_data:
            return {}
        
        success_rates = [record.get('success_rate', 0) for record in history_data]
        
        return {
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'min_success_rate': np.min(success_rates),
            'max_success_rate': np.max(success_rates),
            'median_success_rate': np.median(success_rates)
        }


class EarlyWarningSystem:
    """适应性预警系统"""
    
    def __init__(self, warning_thresholds: Dict[str, float] = None):
        self.warning_thresholds = warning_thresholds or {
            'volatility_spike': 0.5,
            'risk_escalation': 0.8,
            'performance_degradation': 0.3,
            'system_overload': 0.9,
            'adaptation_failure_rate': 0.4
        }
        self.active_warnings = {}
        self.warning_history = deque(maxlen=500)
        self.warning_callbacks = []
        self.monitoring_lock = threading.Lock()
    
    def add_warning_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加预警回调函数"""
        self.warning_callbacks.append(callback)
    
    def check_warnings(self, environment_state: EnvironmentState,
                      recent_performance: List[float],
                      adaptation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查预警条件"""
        with self.monitoring_lock:
            warnings_detected = []
            
            # 检查波动率激增
            if environment_state.volatility_index > self.warning_thresholds['volatility_spike']:
                warning = {
                    'type': 'volatility_spike',
                    'severity': 'high' if environment_state.volatility_index > 0.7 else 'medium',
                    'message': f'市场波动率激增: {environment_state.volatility_index:.3f}',
                    'timestamp': datetime.now(),
                    'data': {'current_volatility': environment_state.volatility_index}
                }
                warnings_detected.append(warning)
            
            # 检查风险升级
            if environment_state.risk_level > self.warning_thresholds['risk_escalation']:
                warning = {
                    'type': 'risk_escalation',
                    'severity': 'critical' if environment_state.risk_level > 0.9 else 'high',
                    'message': f'风险水平过高: {environment_state.risk_level:.3f}',
                    'timestamp': datetime.now(),
                    'data': {'current_risk': environment_state.risk_level}
                }
                warnings_detected.append(warning)
            
            # 检查性能退化
            if recent_performance and np.mean(recent_performance[-10:]) < self.warning_thresholds['performance_degradation']:
                warning = {
                    'type': 'performance_degradation',
                    'severity': 'medium',
                    'message': f'系统性能下降: 平均成功率 {np.mean(recent_performance[-10:]):.3f}',
                    'timestamp': datetime.now(),
                    'data': {'recent_performance': recent_performance[-10:]}
                }
                warnings_detected.append(warning)
            
            # 检查系统过载
            cpu_usage = environment_state.system_performance.get('cpu_usage', 0)
            memory_usage = environment_state.system_performance.get('memory_usage', 0)
            
            if (cpu_usage > self.warning_thresholds['system_overload'] or 
                memory_usage > self.warning_thresholds['system_overload']):
                warning = {
                    'type': 'system_overload',
                    'severity': 'high' if max(cpu_usage, memory_usage) > 0.95 else 'medium',
                    'message': f'系统资源过载: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%',
                    'timestamp': datetime.now(),
                    'data': {'cpu_usage': cpu_usage, 'memory_usage': memory_usage}
                }
                warnings_detected.append(warning)
            
            # 检查适应性失败率
            if adaptation_history:
                recent_failures = sum(1 for record in adaptation_history[-20:] 
                                    if record.get('success_rate', 1) < 0.3)
                failure_rate = recent_failures / min(len(adaptation_history), 20)
                
                if failure_rate > self.warning_thresholds['adaptation_failure_rate']:
                    warning = {
                        'type': 'adaptation_failure_rate',
                        'severity': 'high' if failure_rate > 0.6 else 'medium',
                        'message': f'适应性失败率过高: {failure_rate:.3f}',
                        'timestamp': datetime.now(),
                        'data': {'failure_rate': failure_rate, 'recent_failures': recent_failures}
                    }
                    warnings_detected.append(warning)
            
            # 处理预警
            for warning in warnings_detected:
                self._process_warning(warning)
            
            return warnings_detected
    
    def _process_warning(self, warning: Dict[str, Any]):
        """处理预警"""
        warning_id = f"{warning['type']}_{int(warning['timestamp'].timestamp())}"
        
        # 添加到活动预警
        self.active_warnings[warning_id] = warning
        
        # 添加到历史记录
        self.warning_history.append(warning)
        
        # 触发回调
        for callback in self.warning_callbacks:
            try:
                callback(warning)
            except Exception as e:
                logger.error(f"预警回调错误: {e}")
        
        logger.warning(f"预警触发: {warning['message']}")
    
    def resolve_warning(self, warning_id: str):
        """解决预警"""
        if warning_id in self.active_warnings:
            warning = self.active_warnings.pop(warning_id)
            warning['status'] = 'resolved'
            warning['resolved_at'] = datetime.now()
            self.warning_history.append(warning)
            logger.info(f"预警已解决: {warning_id}")
    
    def get_active_warnings(self) -> List[Dict[str, Any]]:
        """获取活动预警"""
        return list(self.active_warnings.values())
    
    def get_warning_statistics(self, days: int = 7) -> Dict[str, Any]:
        """获取预警统计"""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_warnings = [w for w in self.warning_history if w['timestamp'] >= cutoff_time]
        
        if not recent_warnings:
            return {'message': '没有最近的预警记录'}
        
        # 统计预警类型
        warning_types = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for warning in recent_warnings:
            warning_types[warning['type']] += 1
            severity_counts[warning.get('severity', 'unknown')] += 1
        
        return {
            'analysis_period_days': days,
            'total_warnings': len(recent_warnings),
            'warning_types': dict(warning_types),
            'severity_distribution': dict(severity_counts),
            'most_common_warning': max(warning_types.items(), key=lambda x: x[1])[0] if warning_types else None,
            'resolution_rate': self._calculate_resolution_rate(recent_warnings)
        }
    
    def _calculate_resolution_rate(self, warnings: List[Dict[str, Any]]) -> float:
        """计算解决率"""
        resolved_count = sum(1 for w in warnings if w.get('status') == 'resolved')
        return resolved_count / len(warnings) if warnings else 0


class DynamicAdaptor:
    """G2动态适应器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化各个组件
        self.environment_monitor = EnvironmentMonitor(
            monitoring_interval=self.config.get('monitoring_interval', 1.0)
        )
        self.adaptation_strategy = AdaptationStrategy()
        self.parameter_optimizer = ParameterOptimizer(
            optimization_interval=self.config.get('optimization_interval', 5.0)
        )
        self.learning_engine = LearningEngine(
            learning_rate=self.config.get('learning_rate', 0.01)
        )
        self.effect_evaluator = EffectEvaluator(
            evaluation_window=self.config.get('evaluation_window', 100)
        )
        self.history_tracker = HistoryTracker(
            db_path=self.config.get('history_db_path', 'adaptation_history.db')
        )
        self.early_warning_system = EarlyWarningSystem(
            warning_thresholds=self.config.get('warning_thresholds')
        )
        
        # 状态管理
        self.is_running = False
        self.adaptation_lock = threading.Lock()
        self.current_environment_state = None
        self.last_adaptation_time = None
        
        # 性能统计
        self.performance_stats = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'average_success_rate': 0.0,
            'last_evaluation_time': None
        }
        
        # 设置回调
        self._setup_callbacks()
        
        logger.info("G2动态适应器初始化完成")
    
    def _setup_callbacks(self):
        """设置组件间的回调函数"""
        # 环境监控回调
        self.environment_monitor.add_alert_callback(self._on_environment_change)
        
        # 预警系统回调
        self.early_warning_system.add_warning_callback(self._on_warning)
    
    def start(self):
        """启动动态适应器"""
        if not self.is_running:
            self.is_running = True
            
            # 启动环境监控
            self.environment_monitor.start_monitoring()
            
            # 启动定期优化任务
            self._start_optimization_task()
            
            logger.info("G2动态适应器已启动")
        else:
            logger.warning("动态适应器已经在运行中")
    
    def stop(self):
        """停止动态适应器"""
        if self.is_running:
            self.is_running = False
            
            # 停止环境监控
            self.environment_monitor.stop_monitoring()
            
            logger.info("G2动态适应器已停止")
        else:
            logger.warning("动态适应器未在运行")
    
    def _on_environment_change(self, environment_state: EnvironmentState, change_type: str):
        """环境变化回调"""
        try:
            with self.adaptation_lock:
                logger.info(f"检测到环境变化: {change_type}")
                
                # 选择适应性策略
                strategy_name = self.adaptation_strategy.select_strategy(environment_state, change_type)
                
                # 执行适应性策略
                change_data = {'change_type': change_type, 'timestamp': datetime.now()}
                action = self.adaptation_strategy.execute_strategy(strategy_name, environment_state, change_data)
                
                # 执行动作
                result = self._execute_adaptation_action(action, environment_state)
                action.result = result
                action.status = 'completed' if result.get('success', False) else 'failed'
                action.execution_time = time.time() - action.timestamp.timestamp()
                
                # 评估效果
                evaluation_result = self._evaluate_adaptation_effect(environment_state, action)
                
                # 学习
                self.learning_engine.learn_from_environment(environment_state, action)
                
                # 记录历史
                performance_metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    accuracy=result.get('accuracy', 0),
                    efficiency=result.get('efficiency', 0),
                    stability=result.get('stability', 0),
                    adaptability_score=evaluation_result.get('overall_score', 0),
                    success_rate=result.get('success_rate', 0),
                    error_rate=1 - result.get('success_rate', 0),
                    response_time=action.execution_time,
                    resource_usage=environment_state.system_performance
                )
                
                self.history_tracker.track_adaptation_event(
                    environment_state, action, evaluation_result, performance_metrics
                )
                
                # 更新统计
                self._update_performance_stats(result)
                
                self.last_adaptation_time = datetime.now()
                
                logger.info(f"适应性动作执行完成: {action.action_id}, 成功率: {result.get('success_rate', 0):.3f}")
                
        except Exception as e:
            logger.error(f"环境变化处理错误: {e}")
    
    def _on_warning(self, warning: Dict[str, Any]):
        """预警回调"""
        logger.warning(f"收到预警: {warning['message']}")
        
        # 可以在这里添加自动响应逻辑
        if warning['severity'] == 'critical':
            # 严重预警时触发紧急策略
            self._trigger_emergency_response(warning)
    
    def _execute_adaptation_action(self, action: AdaptationAction,
                                  environment_state: EnvironmentState) -> Dict[str, Any]:
        """执行适应性动作"""
        try:
            if action.action_type == "parameter_adjustment":
                # 参数调整
                target_metrics = {
                    'volatility_control': 0.8,
                    'risk_reduction': 0.7,
                    'efficiency': 0.9
                }
                
                optimization_result = self.parameter_optimizer.optimize_parameters(
                    target_metrics, environment_state, 'adaptive'
                )
                
                if optimization_result['success']:
                    # 记录参数变化
                    for param_name, new_value in optimization_result['new_parameters'].items():
                        if param_name in self.parameter_optimizer.current_parameters:
                            old_value = self.parameter_optimizer.current_parameters[param_name]
                            self.history_tracker.track_parameter_change(
                                param_name, old_value, new_value,
                                f"适应性调整 - {action.action_type}",
                                environment_state.__dict__
                            )
                
                return {
                    'success': optimization_result['success'],
                    'parameters_updated': len(optimization_result.get('new_parameters', {})),
                    'accuracy': 0.85,
                    'efficiency': 0.90,
                    'stability': 0.88,
                    'success_rate': 0.87 if optimization_result['success'] else 0.3
                }
            
            elif action.action_type == "dynamic_adjustment":
                # 动态调整
                return {
                    'success': True,
                    'adjustments_made': len(action.parameters),
                    'accuracy': 0.82,
                    'efficiency': 0.87,
                    'stability': 0.85,
                    'success_rate': 0.85
                }
            
            elif action.action_type == "emergency_response":
                # 紧急响应
                return {
                    'success': True,
                    'emergency_actions': ['stop_loss', 'position_reduction', 'risk_freeze'],
                    'accuracy': 0.95,
                    'efficiency': 0.98,
                    'stability': 0.92,
                    'success_rate': 0.93
                }
            
            else:
                # 默认处理
                return {
                    'success': True,
                    'default_action': True,
                    'accuracy': 0.75,
                    'efficiency': 0.80,
                    'stability': 0.78,
                    'success_rate': 0.78
                }
                
        except Exception as e:
            logger.error(f"执行适应性动作错误: {e}")
            return {
                'success': False,
                'error': str(e),
                'accuracy': 0.0,
                'efficiency': 0.0,
                'stability': 0.0,
                'success_rate': 0.0
            }
    
    def _evaluate_adaptation_effect(self, environment_state: EnvironmentState,
                                   action: AdaptationAction) -> Dict[str, Any]:
        """评估适应性效果"""
        try:
            # 获取之前的环境状态
            history = self.environment_monitor.get_state_history(hours=1)
            before_state = history[-2] if len(history) >= 2 else environment_state
            
            # 设置目标指标
            target_metrics = {
                'volatility_control': 0.8,
                'risk_reduction': 0.7,
                'efficiency': 0.9,
                'stability': 0.85
            }
            
            # 评估效果
            evaluation_result = self.effect_evaluator.evaluate_adaptation_effect(
                before_state, environment_state, action, target_metrics
            )
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"效果评估错误: {e}")
            return {'success': False, 'error': str(e)}
    
    def _trigger_emergency_response(self, warning: Dict[str, Any]):
        """触发紧急响应"""
        try:
            logger.critical(f"触发紧急响应: {warning['message']}")
            
            # 创建紧急动作
            emergency_action = AdaptationAction(
                action_id=f"emergency_{int(time.time())}",
                timestamp=datetime.now(),
                action_type="emergency_response",
                parameters={
                    'immediate_stop_loss': True,
                    'position_reduction': 0.8,
                    'risk_freeze': True,
                    'alert_level': 'critical'
                },
                target_component="emergency_control",
                priority=0
            )
            
            # 执行紧急动作
            result = self._execute_adaptation_action(emergency_action, self.current_environment_state)
            emergency_action.result = result
            
            # 记录紧急响应
            performance_metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                accuracy=result.get('accuracy', 0),
                efficiency=result.get('efficiency', 0),
                stability=result.get('stability', 0),
                adaptability_score=1.0,  # 紧急响应优先级最高
                success_rate=result.get('success_rate', 0),
                error_rate=1 - result.get('success_rate', 0),
                response_time=0.1,  # 紧急响应应该很快
                resource_usage=self.current_environment_state.system_performance if self.current_environment_state else {}
            )
            
            self.history_tracker.track_adaptation_event(
                self.current_environment_state, emergency_action, 
                {'emergency_response': True}, performance_metrics
            )
            
        except Exception as e:
            logger.error(f"紧急响应错误: {e}")
    
    def _start_optimization_task(self):
        """启动定期优化任务"""
        def optimization_loop():
            while self.is_running:
                try:
                    time.sleep(self.parameter_optimizer.optimization_interval)
                    
                    if self.current_environment_state:
                        # 定期优化参数
                        target_metrics = {
                            'volatility_control': 0.8,
                            'risk_reduction': 0.7,
                            'efficiency': 0.9
                        }
                        
                        optimization_result = self.parameter_optimizer.optimize_parameters(
                            target_metrics, self.current_environment_state, 'adaptive'
                        )
                        
                        if optimization_result['success']:
                            logger.info(f"定期优化完成: {len(optimization_result['new_parameters'])} 个参数已更新")
                
                except Exception as e:
                    logger.error(f"定期优化错误: {e}")
        
        optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        optimization_thread.start()
    
    def _update_performance_stats(self, result: Dict[str, Any]):
        """更新性能统计"""
        self.performance_stats['total_adaptations'] += 1
        
        if result.get('success_rate', 0) > 0.5:
            self.performance_stats['successful_adaptations'] += 1
        
        # 更新平均成功率
        total = self.performance_stats['total_adaptations']
        successful = self.performance_stats['successful_adaptations']
        self.performance_stats['average_success_rate'] = successful / total
        
        self.performance_stats['last_evaluation_time'] = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        current_state = self.environment_monitor.get_current_state()
        
        return {
            'is_running': self.is_running,
            'current_environment_state': current_state.__dict__ if current_state else None,
            'performance_stats': self.performance_stats,
            'active_warnings': self.early_warning_system.get_active_warnings(),
            'current_parameters': self.parameter_optimizer.get_current_parameters(),
            'last_adaptation_time': self.last_adaptation_time,
            'learning_statistics': self.learning_engine.get_learning_statistics(),
            'optimization_history_count': len(self.parameter_optimizer.parameter_history)
        }
    
    def get_adaptation_report(self, days: int = 7) -> Dict[str, Any]:
        """获取适应性报告"""
        try:
            # 获取历史数据
            history_data = self.history_tracker.get_adaptation_history(days)
            analytics_report = self.history_tracker.generate_analytics_report(days)
            warning_stats = self.early_warning_system.get_warning_statistics(days)
            evaluation_summary = self.effect_evaluator.get_evaluation_summary(days)
            
            return {
                'report_period_days': days,
                'generation_time': datetime.now(),
                'system_status': self.get_status(),
                'adaptation_analytics': analytics_report,
                'warning_statistics': warning_stats,
                'effectiveness_evaluation': evaluation_summary,
                'recent_adaptations': history_data[:10],  # 最近10条记录
                'performance_trends': self._analyze_performance_trends(history_data)
            }
            
        except Exception as e:
            logger.error(f"生成适应性报告错误: {e}")
            return {'error': str(e)}
    
    def _analyze_performance_trends(self, history_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析性能趋势"""
        if len(history_data) < 5:
            return {'message': '数据不足，无法分析趋势'}
        
        # 提取成功率数据
        success_rates = [record.get('success_rate', 0) for record in history_data]
        
        # 计算趋势
        recent_avg = np.mean(success_rates[:5])  # 最近5个
        earlier_avg = np.mean(success_rates[5:10]) if len(success_rates) >= 10 else recent_avg
        
        trend = 'improving' if recent_avg > earlier_avg else 'declining' if recent_avg < earlier_avg else 'stable'
        
        return {
            'trend_direction': trend,
            'recent_average_success_rate': recent_avg,
            'earlier_average_success_rate': earlier_avg,
            'improvement_magnitude': recent_avg - earlier_avg,
            'volatility': np.std(success_rates[:10]) if len(success_rates) >= 10 else 0
        }
    
    def force_adaptation(self, adaptation_type: str = 'balanced') -> Dict[str, Any]:
        """强制执行适应性动作"""
        try:
            with self.adaptation_lock:
                current_state = self.environment_monitor.get_current_state()
                if not current_state:
                    return {'success': False, 'error': '没有当前环境状态'}
                
                # 执行指定类型的适应性策略
                change_data = {'change_type': 'forced_adaptation', 'timestamp': datetime.now()}
                action = self.adaptation_strategy.execute_strategy(
                    adaptation_type, current_state, change_data
                )
                
                # 执行动作
                result = self._execute_adaptation_action(action, current_state)
                action.result = result
                action.status = 'completed' if result.get('success', False) else 'failed'
                action.execution_time = time.time() - action.timestamp.timestamp()
                
                # 评估效果
                evaluation_result = self._evaluate_adaptation_effect(current_state, action)
                
                return {
                    'success': True,
                    'action': action,
                    'result': result,
                    'evaluation': evaluation_result,
                    'message': f'强制适应性动作执行完成: {adaptation_type}'
                }
                
        except Exception as e:
            logger.error(f"强制适应性动作错误: {e}")
            return {'success': False, 'error': str(e)}


# 使用示例和测试函数
def example_usage():
    """使用示例"""
    # 创建动态适应器
    config = {
        'monitoring_interval': 2.0,
        'optimization_interval': 10.0,
        'learning_rate': 0.02,
        'warning_thresholds': {
            'volatility_spike': 0.4,
            'risk_escalation': 0.7,
            'performance_degradation': 0.4
        }
    }
    
    adaptor = DynamicAdaptor(config)
    
    try:
        # 启动适应器
        adaptor.start()
        
        # 运行一段时间
        time.sleep(30)
        
        # 获取状态
        status = adaptor.get_status()
        print("系统状态:", json.dumps(status, indent=2, default=str))
        
        # 获取报告
        report = adaptor.get_adaptation_report(days=1)
        print("适应性报告:", json.dumps(report, indent=2, default=str))
        
        # 强制执行适应性动作
        result = adaptor.force_adaptation('conservative')
        print("强制适应性结果:", json.dumps(result, indent=2, default=str))
        
    finally:
        # 停止适应器
        adaptor.stop()


if __name__ == "__main__":
    example_usage()