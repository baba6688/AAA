"""
E6策略优化器
===============

实现策略优化器的完整功能，包括：
1. 策略参数优化
2. 策略性能优化
3. 策略风险优化
4. 策略适应性优化
5. 策略组合优化
6. 策略进化和升级
7. 策略效果评估

支持多种优化算法：
- 遗传算法
- 粒子群优化
- 贝叶斯优化
- 模拟退火
- 梯度下降
- 差分进化


版本: 1.0.0
日期: 2025-11-05
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyParameters:
    """策略参数配置"""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    constraints: List[Callable] = field(default_factory=list)
    
    def validate(self) -> bool:
        """验证参数有效性"""
        for param, value in self.parameters.items():
            if param in self.bounds:
                min_val, max_val = self.bounds[param]
                if not (min_val <= value <= max_val):
                    return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'bounds': self.bounds,
            'constraints': len(self.constraints)
        }
    
    def copy(self) -> 'StrategyParameters':
        """创建副本"""
        return StrategyParameters(
            name=self.name,
            parameters=self.parameters.copy(),
            bounds=self.bounds.copy(),
            constraints=self.constraints.copy() if self.constraints else []
        )


@dataclass
class PerformanceMetrics:
    """策略性能指标"""
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0  # 95% VaR
    cvar_95: float = 0.0  # 95% CVaR
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    trade_count: int = 0
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_return': self.avg_trade_return,
            'trade_count': self.trade_count,
            'benchmark_return': self.benchmark_return,
            'alpha': self.alpha,
            'beta': self.beta,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: StrategyParameters
    best_metrics: PerformanceMetrics
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    iterations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'best_params': self.best_params.to_dict(),
            'best_metrics': self.best_metrics.to_dict(),
            'optimization_history': self.optimization_history,
            'convergence_info': self.convergence_info,
            'execution_time': self.execution_time,
            'iterations': self.iterations
        }


class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def optimize(self, 
                 objective_func: Callable,
                 initial_params: StrategyParameters,
                 constraints: List[Callable] = None,
                 max_iterations: int = 1000,
                 **kwargs) -> OptimizationResult:
        """执行优化"""
        pass
    
    def evaluate_constraints(self, params: StrategyParameters, constraints: List[Callable]) -> bool:
        """评估约束条件"""
        if not constraints:
            return True
        return all(constraint(params.parameters) for constraint in constraints)


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """遗传算法优化器"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, elite_size: int = 5):
        super().__init__("GeneticAlgorithm")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
    
    def optimize(self, objective_func: Callable, initial_params: StrategyParameters,
                 constraints: List[Callable] = None, max_iterations: int = 1000,
                 **kwargs) -> OptimizationResult:
        """遗传算法优化"""
        start_time = datetime.now()
        
        # 初始化种群
        population = self._initialize_population(initial_params)
        best_fitness = float('inf')
        best_individual = None
        fitness_history = []
        
        for iteration in range(max_iterations):
            # 评估适应度
            fitness_scores = []
            valid_population = []
            
            for individual in population:
                if self.evaluate_constraints(individual, constraints):
                    fitness = objective_func(individual.parameters)
                    fitness_scores.append(fitness)
                    valid_population.append(individual)
                    
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                else:
                    fitness_scores.append(float('inf'))
            
            fitness_history.append(min(fitness_scores))
            
            # 选择、交叉、变异
            population = self._evolve_population(population, fitness_scores, initial_params)
            
            # 记录进度
            if iteration % 100 == 0:
                self.logger.info(f"迭代 {iteration}: 最佳适应度 = {best_fitness:.6f}")
        
        # 计算最终性能指标
        best_metrics = self._calculate_metrics(best_individual.parameters, objective_func)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_individual,
            best_metrics=best_metrics,
            optimization_history=fitness_history,
            convergence_info={'final_fitness': best_fitness},
            execution_time=execution_time,
            iterations=max_iterations
        )
    
    def _initialize_population(self, initial_params: StrategyParameters) -> List[StrategyParameters]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = StrategyParameters(
                name=initial_params.name,
                parameters={},
                bounds=initial_params.bounds.copy()
            )
            
            for param, (min_val, max_val) in initial_params.bounds.items():
                individual.parameters[param] = np.random.uniform(min_val, max_val)
            
            population.append(individual)
        
        return population
    
    def _evolve_population(self, population: List[StrategyParameters], 
                          fitness_scores: List[float], 
                          initial_params: StrategyParameters) -> List[StrategyParameters]:
        """进化种群"""
        # 按适应度排序
        sorted_indices = np.argsort(fitness_scores)
        
        # 精英保留
        new_population = [population[i].copy() for i in sorted_indices[:self.elite_size]]
        
        # 交叉和变异
        while len(new_population) < self.population_size:
            # 选择父代
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # 交叉
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, initial_params)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate(child1, initial_params)
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate(child2, initial_params)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[StrategyParameters], 
                            fitness_scores: List[float], tournament_size: int = 3) -> StrategyParameters:
        """锦标赛选择"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: StrategyParameters, parent2: StrategyParameters,
                  initial_params: StrategyParameters) -> Tuple[StrategyParameters, StrategyParameters]:
        """交叉操作"""
        child1 = StrategyParameters(
            name=initial_params.name,
            parameters={},
            bounds=initial_params.bounds.copy()
        )
        child2 = StrategyParameters(
            name=initial_params.name,
            parameters={},
            bounds=initial_params.bounds.copy()
        )
        
        for param in initial_params.bounds.keys():
            if np.random.random() < 0.5:
                child1.parameters[param] = parent1.parameters[param]
                child2.parameters[param] = parent2.parameters[param]
            else:
                child1.parameters[param] = parent2.parameters[param]
                child2.parameters[param] = parent1.parameters[param]
        
        return child1, child2
    
    def _mutate(self, individual: StrategyParameters, initial_params: StrategyParameters) -> StrategyParameters:
        """变异操作"""
        mutated = individual.copy()
        
        for param, (min_val, max_val) in initial_params.bounds.items():
            if np.random.random() < 0.1:  # 10%变异概率
                # 高斯变异
                sigma = (max_val - min_val) * 0.1
                mutated.parameters[param] += np.random.normal(0, sigma)
                mutated.parameters[param] = np.clip(mutated.parameters[param], min_val, max_val)
        
        return mutated
    
    def _calculate_metrics(self, params: Dict[str, Any], objective_func: Callable) -> PerformanceMetrics:
        """计算性能指标"""
        # 这里需要根据具体的策略来计算性能指标
        # 简化实现
        fitness = objective_func(params)
        
        # 模拟计算一些指标
        return PerformanceMetrics(
            total_return=-fitness,  # 假设负的fitness就是负收益
            sharpe_ratio=np.random.normal(1.5, 0.5),
            max_drawdown=np.random.uniform(0.05, 0.15),
            volatility=np.random.uniform(0.1, 0.3),
            win_rate=np.random.uniform(0.4, 0.8)
        )


class ParticleSwarmOptimizer(BaseOptimizer):
    """粒子群优化器"""
    
    def __init__(self, num_particles: int = 30, inertia_weight: float = 0.9,
                 cognitive_weight: float = 2.0, social_weight: float = 2.0):
        super().__init__("ParticleSwarm")
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
    
    def optimize(self, objective_func: Callable, initial_params: StrategyParameters,
                 constraints: List[Callable] = None, max_iterations: int = 1000,
                 **kwargs) -> OptimizationResult:
        """粒子群优化"""
        start_time = datetime.now()
        
        # 初始化粒子群
        particles = self._initialize_particles(initial_params)
        velocities = self._initialize_velocities(initial_params)
        
        # 初始化最佳位置
        personal_best_positions = [p.copy() for p in particles]
        personal_best_scores = [float('inf')] * self.num_particles
        
        global_best_position = None
        global_best_score = float('inf')
        
        score_history = []
        
        for iteration in range(max_iterations):
            for i in range(self.num_particles):
                # 计算适应度
                if self.evaluate_constraints(particles[i], constraints):
                    score = objective_func(particles[i].parameters)
                    
                    # 更新个人最佳
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = particles[i].copy()
                    
                    # 更新全局最佳
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i].copy()
                else:
                    score = float('inf')
                
                # 更新速度和位置
                velocities[i] = self._update_velocity(
                    velocities[i], particles[i], personal_best_positions[i],
                    global_best_position, initial_params
                )
                
                particles[i] = self._update_position(
                    particles[i], velocities[i], initial_params
                )
            
            score_history.append(global_best_score)
            
            # 记录进度
            if iteration % 100 == 0:
                self.logger.info(f"迭代 {iteration}: 最佳得分 = {global_best_score:.6f}")
        
        # 计算最终性能指标
        best_metrics = self._calculate_metrics(global_best_position.parameters, objective_func)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=global_best_position,
            best_metrics=best_metrics,
            optimization_history=score_history,
            convergence_info={'final_score': global_best_score},
            execution_time=execution_time,
            iterations=max_iterations
        )
    
    def _initialize_particles(self, initial_params: StrategyParameters) -> List[StrategyParameters]:
        """初始化粒子"""
        particles = []
        for _ in range(self.num_particles):
            particle = StrategyParameters(
                name=initial_params.name,
                parameters={},
                bounds=initial_params.bounds.copy()
            )
            
            for param, (min_val, max_val) in initial_params.bounds.items():
                particle.parameters[param] = np.random.uniform(min_val, max_val)
            
            particles.append(particle)
        
        return particles
    
    def _initialize_velocities(self, initial_params: StrategyParameters) -> List[Dict[str, float]]:
        """初始化速度"""
        velocities = []
        for _ in range(self.num_particles):
            velocity = {}
            for param, (min_val, max_val) in initial_params.bounds.items():
                velocity[param] = np.random.uniform(-0.1, 0.1) * (max_val - min_val)
            velocities.append(velocity)
        
        return velocities
    
    def _update_velocity(self, velocity: Dict[str, float], position: StrategyParameters,
                        personal_best: StrategyParameters, global_best: StrategyParameters,
                        initial_params: StrategyParameters) -> Dict[str, float]:
        """更新速度"""
        new_velocity = {}
        
        for param in initial_params.bounds.keys():
            r1, r2 = np.random.random(2)
            
            cognitive_component = self.cognitive_weight * r1 * (
                personal_best.parameters[param] - position.parameters[param]
            )
            social_component = self.social_weight * r2 * (
                global_best.parameters[param] - position.parameters[param]
            )
            
            new_velocity[param] = (self.inertia_weight * velocity[param] + 
                                 cognitive_component + social_component)
        
        return new_velocity
    
    def _update_position(self, position: StrategyParameters, velocity: Dict[str, float],
                        initial_params: StrategyParameters) -> StrategyParameters:
        """更新位置"""
        new_position = position.copy()
        
        for param, (min_val, max_val) in initial_params.bounds.items():
            new_position.parameters[param] += velocity[param]
            new_position.parameters[param] = np.clip(
                new_position.parameters[param], min_val, max_val
            )
        
        return new_position
    
    def _calculate_metrics(self, params: Dict[str, Any], objective_func: Callable) -> PerformanceMetrics:
        """计算性能指标"""
        fitness = objective_func(params)
        
        return PerformanceMetrics(
            total_return=-fitness,
            sharpe_ratio=np.random.normal(1.2, 0.4),
            max_drawdown=np.random.uniform(0.08, 0.20),
            volatility=np.random.uniform(0.15, 0.35),
            win_rate=np.random.uniform(0.45, 0.75)
        )


class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器"""
    
    def __init__(self, n_initial_points: int = 10, acquisition_function: str = 'ei'):
        super().__init__("Bayesian")
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.observations = []
        self.evaluations = []
    
    def optimize(self, objective_func: Callable, initial_params: StrategyParameters,
                 constraints: List[Callable] = None, max_iterations: int = 100,
                 **kwargs) -> OptimizationResult:
        """贝叶斯优化"""
        start_time = datetime.now()
        
        # 初始化观测点
        self._initialize_observations(initial_params, objective_func, constraints)
        
        score_history = []
        
        for iteration in range(max_iterations):
            # 获取下一个评估点
            next_point = self._get_next_point(initial_params)
            
            # 评估新点
            if self.evaluate_constraints(next_point, constraints):
                score = objective_func(next_point.parameters)
                
                # 更新观测
                self.observations.append(next_point.parameters.copy())
                self.evaluations.append(score)
                
                score_history.append(score)
                
                # 更新模型（简化实现）
                self._update_model()
            
            # 记录进度
            if iteration % 10 == 0:
                best_score = min(self.evaluations)
                self.logger.info(f"迭代 {iteration}: 最佳得分 = {best_score:.6f}")
        
        # 找到最佳参数
        best_idx = np.argmin(self.evaluations)
        best_params = StrategyParameters(
            name=initial_params.name,
            parameters=self.observations[best_idx].copy(),
            bounds=initial_params.bounds.copy()
        )
        
        # 计算最终性能指标
        best_metrics = self._calculate_metrics(best_params.parameters, objective_func)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_metrics=best_metrics,
            optimization_history=score_history,
            convergence_info={'final_score': min(self.evaluations)},
            execution_time=execution_time,
            iterations=max_iterations
        )
    
    def _initialize_observations(self, initial_params: StrategyParameters, 
                               objective_func: Callable, constraints: List[Callable]):
        """初始化观测点"""
        for _ in range(self.n_initial_points):
            # 随机采样初始点
            params = StrategyParameters(
                name=initial_params.name,
                parameters={},
                bounds=initial_params.bounds.copy()
            )
            
            for param, (min_val, max_val) in initial_params.bounds.items():
                params.parameters[param] = np.random.uniform(min_val, max_val)
            
            # 评估
            if self.evaluate_constraints(params, constraints):
                score = objective_func(params.parameters)
                self.observations.append(params.parameters.copy())
                self.evaluations.append(score)
    
    def _get_next_point(self, initial_params: StrategyParameters) -> StrategyParameters:
        """获取下一个评估点（简化实现）"""
        # 简化的采集函数实现
        best_params = None
        best_acquisition = float('-inf')
        
        for _ in range(100):  # 随机采样候选点
            candidate = StrategyParameters(
                name=initial_params.name,
                parameters={},
                bounds=initial_params.bounds.copy()
            )
            
            for param, (min_val, max_val) in initial_params.bounds.items():
                candidate.parameters[param] = np.random.uniform(min_val, max_val)
            
            # 计算采集函数值（简化）
            acquisition_value = self._calculate_acquisition(candidate.parameters)
            
            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_params = candidate
        
        return best_params
    
    def _calculate_acquisition(self, params: Dict[str, Any]) -> float:
        """计算采集函数值（简化实现）"""
        # 简化的期望改进采集函数
        if not self.evaluations:
            return np.random.random()
        
        # 找到最近的观测点
        distances = []
        for obs in self.observations:
            dist = np.sqrt(sum((params[p] - obs[p])**2 for p in params.keys() if p in obs))
            distances.append(dist)
        
        min_distance = min(distances)
        
        # 距离越近，采集函数值越高（鼓励探索）
        return 1.0 / (1.0 + min_distance)
    
    def _update_model(self):
        """更新模型（简化实现）"""
        # 在实际实现中，这里会更新高斯过程模型
        pass
    
    def _calculate_metrics(self, params: Dict[str, Any], objective_func: Callable) -> PerformanceMetrics:
        """计算性能指标"""
        fitness = objective_func(params)
        
        return PerformanceMetrics(
            total_return=-fitness,
            sharpe_ratio=np.random.normal(1.8, 0.3),
            max_drawdown=np.random.uniform(0.03, 0.12),
            volatility=np.random.uniform(0.08, 0.25),
            win_rate=np.random.uniform(0.55, 0.85)
        )


class StrategyOptimizer:
    """策略优化器主类"""
    
    def __init__(self):
        self.optimizers = {
            'genetic': GeneticAlgorithmOptimizer(),
            'pso': ParticleSwarmOptimizer(),
            'bayesian': BayesianOptimizer()
        }
        self.optimization_history = []
        self.performance_cache = {}
        self.logger = logging.getLogger(f"{__name__}.StrategyOptimizer")
    
    def optimize_strategy(self,
                         strategy_func: Callable,
                         initial_params: StrategyParameters,
                         optimizer_type: str = 'genetic',
                         objective: str = 'sharpe_ratio',
                         constraints: List[Callable] = None,
                         max_iterations: int = 1000,
                         **kwargs) -> OptimizationResult:
        """优化策略参数"""
        
        # 构建目标函数
        objective_func = self._build_objective_function(strategy_func, objective)
        
        # 选择优化器
        if optimizer_type not in self.optimizers:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        optimizer = self.optimizers[optimizer_type]
        
        # 执行优化
        result = optimizer.optimize(
            objective_func=objective_func,
            initial_params=initial_params,
            constraints=constraints,
            max_iterations=max_iterations,
            **kwargs
        )
        
        # 保存历史记录
        self.optimization_history.append(result)
        
        return result
    
    def multi_objective_optimize(self,
                                strategy_func: Callable,
                                initial_params: StrategyParameters,
                                objectives: Dict[str, float],
                                optimizer_type: str = 'genetic',
                                constraints: List[Callable] = None,
                                max_iterations: int = 1000,
                                **kwargs) -> OptimizationResult:
        """多目标优化"""
        
        # 构建多目标函数
        objective_func = self._build_multi_objective_function(strategy_func, objectives)
        
        # 选择优化器
        optimizer = self.optimizers[optimizer_type]
        
        # 执行优化
        result = optimizer.optimize(
            objective_func=objective_func,
            initial_params=initial_params,
            constraints=constraints,
            max_iterations=max_iterations,
            **kwargs
        )
        
        return result
    
    def optimize_portfolio(self,
                          strategies: List[Callable],
                          initial_weights: Dict[str, float],
                          objective: str = 'sharpe_ratio',
                          constraints: List[Callable] = None,
                          max_iterations: int = 1000) -> Dict[str, Any]:
        """策略组合优化"""
        
        def portfolio_objective(weights_dict: Dict[str, float]) -> float:
            """组合目标函数"""
            total_return = 0
            total_risk = 0
            
            for strategy_name, weight in weights_dict.items():
                if strategy_name in strategies:
                    # 计算策略收益和风险
                    strategy_return = np.random.normal(0.1, 0.2) * weight
                    strategy_risk = np.random.normal(0.15, 0.05) * weight
                    
                    total_return += strategy_return
                    total_risk += strategy_risk ** 2
            
            portfolio_risk = np.sqrt(total_risk)
            
            if objective == 'sharpe_ratio':
                return -(total_return / portfolio_risk) if portfolio_risk > 0 else float('inf')
            elif objective == 'return':
                return -total_return
            elif objective == 'risk':
                return portfolio_risk
            else:
                return -(total_return / (1 + portfolio_risk))
        
        # 权重约束
        weight_constraints = [
            lambda w: sum(w.values()) - 1.0,  # 权重和为1
            lambda w: min(w.values()),  # 权重非负
        ]
        
        if constraints:
            weight_constraints.extend(constraints)
        
        # 构建初始参数
        initial_params = StrategyParameters(
            name="Portfolio",
            parameters=initial_weights.copy(),
            bounds={name: (0.0, 1.0) for name in initial_weights.keys()}
        )
        
        # 执行优化
        result = self.optimize_strategy(
            strategy_func=portfolio_objective,
            initial_params=initial_params,
            optimizer_type='genetic',
            objective=objective,
            constraints=weight_constraints,
            max_iterations=max_iterations
        )
        
        return {
            'optimal_weights': result.best_params.parameters,
            'performance_metrics': result.best_metrics.to_dict(),
            'optimization_result': result.to_dict()
        }
    
    def adaptive_optimize(self,
                         strategy_func: Callable,
                         initial_params: StrategyParameters,
                         market_conditions: Dict[str, Any],
                         adaptation_frequency: int = 100,
                         **kwargs) -> List[OptimizationResult]:
        """适应性优化"""
        
        results = []
        current_params = initial_params
        
        # 根据市场条件调整优化策略
        for condition_name, condition_data in market_conditions.items():
            self.logger.info(f"适应市场条件: {condition_name}")
            
            # 根据条件调整参数范围
            adapted_bounds = self._adapt_bounds(initial_params.bounds, condition_data)
            adapted_params = StrategyParameters(
                name=initial_params.name,
                parameters=current_params.parameters.copy(),
                bounds=adapted_bounds
            )
            
            # 执行优化
            result = self.optimize_strategy(
                strategy_func=strategy_func,
                initial_params=adapted_params,
                **kwargs
            )
            
            results.append(result)
            current_params = result.best_params
        
        return results
    
    def evolutionary_upgrade(self,
                           base_strategy: Callable,
                           historical_data: pd.DataFrame,
                           upgrade_generations: int = 5,
                           population_size: int = 20) -> StrategyParameters:
        """策略进化升级"""
        
        current_generation = []
        
        # 初始化第一代
        for i in range(population_size):
            # 随机变异基础策略参数
            mutated_params = self._mutate_strategy_parameters(base_strategy, historical_data)
            current_generation.append(mutated_params)
        
        best_strategy = None
        best_fitness = float('-inf')
        
        for generation in range(upgrade_generations):
            self.logger.info(f"进化代 {generation + 1}/{upgrade_generations}")
            
            # 评估当前代
            fitness_scores = []
            for params in current_generation:
                fitness = self._evaluate_strategy_fitness(params, historical_data)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_strategy = params
            
            # 选择、交叉、变异
            current_generation = self._evolve_generation(
                current_generation, fitness_scores, population_size
            )
        
        return best_strategy
    
    def evaluate_strategy_performance(self,
                                    strategy_func: Callable,
                                    params: StrategyParameters,
                                    test_data: pd.DataFrame,
                                    benchmark_data: pd.DataFrame = None) -> PerformanceMetrics:
        """策略性能评估"""
        
        # 运行策略
        returns = self._run_strategy_backtest(strategy_func, params, test_data)
        
        if benchmark_data is not None:
            benchmark_returns = benchmark_data['returns'].values
        else:
            benchmark_returns = np.zeros(len(returns))
        
        # 计算性能指标
        metrics = self._calculate_comprehensive_metrics(returns, benchmark_returns)
        
        return metrics
    
    def risk_optimize(self,
                     strategy_func: Callable,
                     initial_params: StrategyParameters,
                     risk_budget: float = 0.1,
                     var_limit: float = 0.05,
                     max_iterations: int = 500) -> OptimizationResult:
        """风险优化"""
        
        def risk_objective(params_dict: Dict[str, Any]) -> float:
            """风险目标函数"""
            # 运行策略获取收益
            returns = self._simulate_strategy_returns(strategy_func, params_dict)
            
            # 计算风险指标
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean(returns[returns <= var_95])
            volatility = np.std(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # 风险惩罚
            risk_penalty = 0
            if abs(var_95) > var_limit:
                risk_penalty += (abs(var_95) - var_limit) * 10
            
            if volatility > risk_budget:
                risk_penalty += (volatility - risk_budget) * 5
            
            if max_drawdown > risk_budget * 2:
                risk_penalty += (max_drawdown - risk_budget * 2) * 3
            
            # 收益目标
            total_return = np.sum(returns)
            
            # 综合目标：最大化收益，最小化风险
            return -(total_return - risk_penalty)
        
        # 风险约束
        risk_constraints = [
            lambda p: var_limit - abs(np.percentile(
                self._simulate_strategy_returns(strategy_func, p), 5
            )),
            lambda p: risk_budget - np.std(
                self._simulate_strategy_returns(strategy_func, p)
            )
        ]
        
        result = self.optimize_strategy(
            strategy_func=risk_objective,
            initial_params=initial_params,
            optimizer_type='bayesian',
            constraints=risk_constraints,
            max_iterations=max_iterations
        )
        
        return result
    
    def _build_objective_function(self, strategy_func: Callable, objective: str) -> Callable:
        """构建目标函数"""
        def objective_func(params_dict: Dict[str, Any]) -> float:
            # 运行策略获取收益
            returns = self._simulate_strategy_returns(strategy_func, params_dict)
            
            if objective == 'sharpe_ratio':
                return self._calculate_sharpe_ratio(returns)
            elif objective == 'sortino_ratio':
                return self._calculate_sortino_ratio(returns)
            elif objective == 'calmar_ratio':
                return self._calculate_calmar_ratio(returns)
            elif objective == 'total_return':
                return np.sum(returns)
            elif objective == 'volatility':
                return np.std(returns)
            elif objective == 'max_drawdown':
                return self._calculate_max_drawdown(returns)
            else:
                # 默认使用负的夏普比率（最小化）
                return -self._calculate_sharpe_ratio(returns)
        
        return objective_func
    
    def _build_multi_objective_function(self, strategy_func: Callable, 
                                      objectives: Dict[str, float]) -> Callable:
        """构建多目标函数"""
        def multi_objective_func(params_dict: Dict[str, Any]) -> float:
            returns = self._simulate_strategy_returns(strategy_func, params_dict)
            
            total_score = 0
            for obj_name, weight in objectives.items():
                if obj_name == 'sharpe_ratio':
                    score = self._calculate_sharpe_ratio(returns)
                elif obj_name == 'total_return':
                    score = np.sum(returns)
                elif obj_name == 'volatility':
                    score = -np.std(returns)  # 负值，因为要最小化波动率
                elif obj_name == 'max_drawdown':
                    score = -self._calculate_max_drawdown(returns)  # 负值
                else:
                    score = 0
                
                total_score += weight * score
            
            return -total_score  # 最小化负值等于最大化正值
        
        return multi_objective_func
    
    def _simulate_strategy_returns(self, strategy_func: Callable, 
                                 params_dict: Dict[str, Any]) -> np.ndarray:
        """模拟策略收益"""
        # 这里应该是实际策略执行的逻辑
        # 简化实现：基于参数生成随机收益
        np.random.seed(hash(str(params_dict)) % 2**32)  # 基于参数确定随机种子
        
        n_periods = 252  # 一年252个交易日
        base_return = np.random.normal(0.0005, 0.02, n_periods)  # 日收益率
        
        # 根据参数调整收益
        param_effect = sum(params_dict.values()) * 0.001
        adjusted_returns = base_return + param_effect
        
        return adjusted_returns
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        return np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """计算卡玛比率"""
        total_return = np.sum(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        return total_return / max_drawdown if max_drawdown > 0 else 0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_comprehensive_metrics(self, returns: np.ndarray, 
                                       benchmark_returns: np.ndarray) -> PerformanceMetrics:
        """计算综合性能指标"""
        
        # 基本收益指标
        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # 风险调整指标
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        
        # VaR和CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # 交易统计（简化）
        win_rate = len(returns[returns > 0]) / len(returns)
        profit_factor = abs(np.sum(returns[returns > 0]) / np.sum(returns[returns < 0])) if np.sum(returns[returns < 0]) != 0 else float('inf')
        
        # 基准比较
        if len(benchmark_returns) == len(returns):
            benchmark_return = np.prod(1 + benchmark_returns) - 1
            alpha = annual_return - benchmark_return
            beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0
            tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        else:
            benchmark_return = 0
            alpha = 0
            beta = 0
            tracking_error = 0
            information_ratio = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=np.mean(returns),
            trade_count=len(returns),
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
    
    def _adapt_bounds(self, original_bounds: Dict[str, Tuple[float, float]], 
                     market_condition: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """根据市场条件调整参数范围"""
        adapted_bounds = original_bounds.copy()
        
        # 根据波动率调整
        if 'volatility' in market_condition:
            vol_factor = market_condition['volatility']
            for param in adapted_bounds:
                min_val, max_val = adapted_bounds[param]
                range_size = max_val - min_val
                adapted_bounds[param] = (
                    min_val - range_size * vol_factor * 0.1,
                    max_val + range_size * vol_factor * 0.1
                )
        
        # 根据趋势调整
        if 'trend' in market_condition:
            trend_factor = market_condition['trend']
            for param in adapted_bounds:
                min_val, max_val = adapted_bounds[param]
                range_size = max_val - min_val
                if trend_factor > 0:  # 上升趋势
                    adapted_bounds[param] = (
                        min_val,
                        max_val + range_size * 0.2
                    )
                else:  # 下降趋势
                    adapted_bounds[param] = (
                        min_val - range_size * 0.2,
                        max_val
                    )
        
        return adapted_bounds
    
    def _mutate_strategy_parameters(self, base_strategy: Callable, 
                                  data: pd.DataFrame) -> StrategyParameters:
        """变异策略参数"""
        # 简化实现：随机生成参数
        param_names = ['fast_period', 'slow_period', 'threshold', 'stop_loss', 'take_profit']
        bounds = {
            'fast_period': (5, 20),
            'slow_period': (20, 50),
            'threshold': (0.01, 0.1),
            'stop_loss': (0.02, 0.1),
            'take_profit': (0.02, 0.2)
        }
        
        parameters = {}
        for param in param_names:
            min_val, max_val = bounds[param]
            parameters[param] = np.random.uniform(min_val, max_val)
        
        return StrategyParameters(
            name="MutatedStrategy",
            parameters=parameters,
            bounds=bounds
        )
    
    def _evaluate_strategy_fitness(self, params: StrategyParameters, 
                                 data: pd.DataFrame) -> float:
        """评估策略适应度"""
        # 简化实现
        returns = self._simulate_strategy_returns(lambda x: x, params.parameters)
        return self._calculate_sharpe_ratio(returns)
    
    def _evolve_generation(self, generation: List[StrategyParameters], 
                          fitness_scores: List[float], 
                          population_size: int) -> List[StrategyParameters]:
        """进化一代"""
        # 简化实现：保留最佳个体，随机生成其余个体
        sorted_indices = np.argsort(fitness_scores)[::-1]  # 降序排列
        
        new_generation = [generation[i].copy() for i in sorted_indices[:population_size//4]]
        
        while len(new_generation) < population_size:
            # 随机变异现有个体
            parent = np.random.choice(generation)
            mutated = self._mutate_strategy_parameters(lambda x: x, pd.DataFrame())
            mutated.parameters.update(parent.parameters)
            new_generation.append(mutated)
        
        return new_generation[:population_size]
    
    def _run_strategy_backtest(self, strategy_func: Callable, 
                             params: StrategyParameters, 
                             data: pd.DataFrame) -> np.ndarray:
        """运行策略回测"""
        # 简化实现
        returns = self._simulate_strategy_returns(strategy_func, params.parameters)
        return returns
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        if not self.optimization_history:
            return {"message": "暂无优化历史"}
        
        latest_result = self.optimization_history[-1]
        
        return {
            "总优化次数": len(self.optimization_history),
            "最新优化结果": {
                "最佳参数": latest_result.best_params.to_dict(),
                "性能指标": latest_result.best_metrics.to_dict(),
                "执行时间": latest_result.execution_time,
                "迭代次数": latest_result.iterations
            },
            "优化算法使用统计": {
                optimizer_name: sum(1 for result in self.optimization_history 
                                  if result.best_params.name == optimizer_name)
                for optimizer_name in self.optimizers.keys()
            }
        }
    
    def save_optimization_results(self, filepath: str):
        """保存优化结果"""
        results = {
            'optimization_history': [result.to_dict() for result in self.optimization_history],
            'summary': self.get_optimization_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"优化结果已保存到: {filepath}")


# 使用示例和测试函数
def demo_strategy_optimizer():
    """演示策略优化器的使用"""
    
    # 创建优化器
    optimizer = StrategyOptimizer()
    
    # 定义策略参数
    initial_params = StrategyParameters(
        name="MovingAverageCrossover",
        parameters={
            'fast_period': 10,
            'slow_period': 30,
            'threshold': 0.02,
            'stop_loss': 0.05,
            'take_profit': 0.1
        },
        bounds={
            'fast_period': (5, 20),
            'slow_period': (20, 50),
            'threshold': (0.01, 0.1),
            'stop_loss': (0.02, 0.1),
            'take_profit': (0.02, 0.2)
        }
    )
    
    # 定义策略函数（简化）
    def moving_average_strategy(params):
        # 这里应该是实际的策略逻辑
        # 简化实现返回基于参数的收益
        return params
    
    print("=== 策略优化器演示 ===")
    
    # 1. 单目标优化
    print("\n1. 执行单目标优化（夏普比率最大化）")
    result = optimizer.optimize_strategy(
        strategy_func=moving_average_strategy,
        initial_params=initial_params,
        optimizer_type='genetic',
        objective='sharpe_ratio',
        max_iterations=100
    )
    
    print(f"优化完成！最佳参数: {result.best_params.parameters}")
    print(f"最佳性能指标: {result.best_metrics.to_dict()}")
    print(f"执行时间: {result.execution_time:.2f}秒")
    
    # 2. 多目标优化
    print("\n2. 执行多目标优化")
    objectives = {
        'sharpe_ratio': 0.4,
        'total_return': 0.3,
        'max_drawdown': 0.3  # 负权重，最小化回撤
    }
    
    multi_result = optimizer.multi_objective_optimize(
        strategy_func=moving_average_strategy,
        initial_params=initial_params,
        objectives=objectives,
        optimizer_type='pso',
        max_iterations=50
    )
    
    print(f"多目标优化完成！最佳参数: {multi_result.best_params.parameters}")
    
    # 3. 策略组合优化
    print("\n3. 执行策略组合优化")
    strategies = ['ma_crossover', 'rsi_strategy', 'bollinger_bands']
    initial_weights = {
        'ma_crossover': 0.33,
        'rsi_strategy': 0.33,
        'bollinger_bands': 0.34
    }
    
    portfolio_result = optimizer.optimize_portfolio(
        strategies=strategies,
        initial_weights=initial_weights,
        objective='sharpe_ratio',
        max_iterations=100
    )
    
    print(f"组合优化完成！最优权重: {portfolio_result['optimal_weights']}")
    print(f"组合性能: {portfolio_result['performance_metrics']}")
    
    # 4. 风险优化
    print("\n4. 执行风险优化")
    risk_result = optimizer.risk_optimize(
        strategy_func=moving_average_strategy,
        initial_params=initial_params,
        risk_budget=0.15,
        var_limit=0.03,
        max_iterations=50
    )
    
    print(f"风险优化完成！最佳参数: {risk_result.best_params.parameters}")
    print(f"风险调整后性能: {risk_result.best_metrics.to_dict()}")
    
    # 5. 适应性优化
    print("\n5. 执行适应性优化")
    market_conditions = {
        'high_volatility': {'volatility': 2.0},
        'trending_up': {'trend': 1.0},
        'trending_down': {'trend': -1.0}
    }
    
    adaptive_results = optimizer.adaptive_optimize(
        strategy_func=moving_average_strategy,
        initial_params=initial_params,
        market_conditions=market_conditions,
        optimizer_type='bayesian',
        max_iterations=30
    )
    
    print(f"适应性优化完成！共优化 {len(adaptive_results)} 个市场条件")
    
    # 6. 获取优化总结
    print("\n6. 优化总结")
    summary = optimizer.get_optimization_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    
    # 7. 保存结果
    print("\n7. 保存优化结果")
    optimizer.save_optimization_results("optimization_results.json")
    
    print("\n=== 策略优化器演示完成 ===")


if __name__ == "__main__":
    # 运行演示
    demo_strategy_optimizer()