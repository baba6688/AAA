"""
策略优化器模块

实现策略学习过程的优化：
- 超参数优化
- 学习率优化
- 探索-利用平衡优化
- 自适应优化
- 多目标优化
- 贝叶斯优化
- 进化优化
- 性能驱动优化
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from copy import deepcopy
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import json
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

from .StrategyLearner import BaseStrategy, StrategyType, LearningContext, StrategyPerformance

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """优化结果数据类"""
    optimization_id: str
    strategy_id: str
    method: str
    success: bool
    initial_performance: float
    optimized_performance: float
    improvement: float
    optimal_parameters: Dict[str, Any]
    optimization_time: float
    iterations: int
    convergence_achieved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationConfig:
    """优化配置"""
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    timeout_seconds: int = 300
    parallel_workers: int = 4
    random_seed: Optional[int] = None
    save_intermediate_results: bool = True
    validation_split: float = 0.2

class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, name: str, config: OptimizationConfig):
        self.name = name
        self.config = config
        self.optimization_history = []
        self.is_optimizing = False
        self.optimization_lock = threading.Lock()
        
        # 设置随机种子
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)
    
    @abstractmethod
    def optimize(self, strategy: BaseStrategy, performance_data: List[StrategyPerformance],
                objective_function: Callable, constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """执行优化"""
        pass
    
    def _evaluate_objective(self, parameters: Dict[str, Any], strategy: BaseStrategy,
                          performance_data: List[StrategyPerformance]) -> float:
        """评估目标函数"""
        try:
            # 创建策略副本进行测试
            test_strategy = self._create_test_strategy(strategy, parameters)
            
            # 模拟策略性能
            simulated_performance = self._simulate_strategy_performance(test_strategy, performance_data)
            
            return -simulated_performance  # 最小化负性能 = 最大化性能
            
        except Exception as e:
            logger.error(f"目标函数评估出错: {e}")
            return float('inf')
    
    def _create_test_strategy(self, original_strategy: BaseStrategy, 
                            parameters: Dict[str, Any]) -> BaseStrategy:
        """创建测试策略"""
        # 简化的策略复制和参数更新
        test_strategy = deepcopy(original_strategy)
        
        # 更新参数（具体实现取决于策略类型）
        if hasattr(test_strategy, 'learning_rate') and 'learning_rate' in parameters:
            test_strategy.learning_rate = parameters['learning_rate']
        
        if hasattr(test_strategy, 'epsilon') and 'epsilon' in parameters:
            test_strategy.epsilon = parameters['epsilon']
        
        if hasattr(test_strategy, 'discount_factor') and 'discount_factor' in parameters:
            test_strategy.discount_factor = parameters['discount_factor']
        
        return test_strategy
    
    def _simulate_strategy_performance(self, strategy: BaseStrategy,
                                     performance_data: List[StrategyPerformance]) -> float:
        """模拟策略性能"""
        if not performance_data:
            return 0.0
        
        # 简化的性能模拟
        base_performance = np.mean([p.return_rate for p in performance_data])
        
        # 基于策略类型调整
        if strategy.strategy_type == StrategyType.REINFORCEMENT:
            # 强化学习策略的性能调整
            if hasattr(strategy, 'learning_rate'):
                lr_factor = 1.0 / (1.0 + abs(strategy.learning_rate - 0.01) * 10)
                base_performance *= lr_factor
            
            if hasattr(strategy, 'epsilon'):
                epsilon_factor = 1.0 - abs(strategy.epsilon - 0.1) * 2
                base_performance *= max(0.1, epsilon_factor)
        
        elif strategy.strategy_type == StrategyType.EVOLUTION:
            # 进化算法策略的性能调整
            if hasattr(strategy, 'mutation_rate'):
                mutation_factor = 1.0 / (1.0 + abs(strategy.mutation_rate - 0.1) * 5)
                base_performance *= mutation_factor
        
        return max(0.0, base_performance)
    
    def _check_convergence(self, history: List[float]) -> bool:
        """检查收敛性"""
        if len(history) < 10:
            return False
        
        recent_values = history[-10:]
        return np.std(recent_values) < self.config.convergence_threshold

class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("bayesian_optimization", config)
        self.gp_model = None
        self.parameter_bounds = {}
        self.acquisition_function = 'expected_improvement'
        
    def optimize(self, strategy: BaseStrategy, performance_data: List[StrategyPerformance],
                objective_function: Callable, constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """贝叶斯优化"""
        with self.optimization_lock:
            start_time = datetime.now()
            optimization_id = f"bayesian_{strategy.strategy_id}_{int(start_time.timestamp())}"
            
            try:
                self.is_optimizing = True
                logger.info(f"开始贝叶斯优化: {optimization_id}")
                
                # 定义参数空间
                parameter_space = self._define_parameter_space(strategy)
                if not parameter_space:
                    return self._create_failed_result(optimization_id, strategy, "无法定义参数空间")
                
                # 初始采样
                initial_samples = self._generate_initial_samples(parameter_space, n_samples=10)
                
                # 评估初始样本
                sample_results = []
                for params in initial_samples:
                    try:
                        score = objective_function(params)
                        sample_results.append((params, score))
                    except Exception as e:
                        logger.warning(f"初始样本评估失败: {e}")
                        continue
                
                if not sample_results:
                    return self._create_failed_result(optimization_id, strategy, "初始样本评估全部失败")
                
                # 训练高斯过程模型
                X = np.array([list(params.values()) for params, _ in sample_results])
                y = np.array([score for _, score in sample_results])
                
                self._train_gp_model(X, y)
                
                # 迭代优化
                best_params = None
                best_score = float('inf')
                optimization_history = []
                
                for iteration in range(self.config.max_iterations):
                    # 获取下一个采样点
                    next_params = self._acquire_next_point(parameter_space)
                    
                    if next_params is None:
                        break
                    
                    # 评估新点
                    try:
                        score = objective_function(next_params)
                        optimization_history.append(score)
                        
                        # 更新最佳结果
                        if score < best_score:
                            best_score = score
                            best_params = next_params
                        
                        # 更新GP模型
                        X = np.vstack([X, list(next_params.values())])
                        y = np.append(y, score)
                        self._train_gp_model(X, y)
                        
                        logger.debug(f"迭代 {iteration+1}: 得分 = {score:.6f}, 最佳 = {best_score:.6f}")
                        
                        # 检查收敛
                        if self._check_convergence(optimization_history):
                            logger.info(f"贝叶斯优化收敛于迭代 {iteration+1}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"迭代 {iteration+1} 评估失败: {e}")
                        continue
                
                # 计算优化结果
                optimization_time = (datetime.now() - start_time).total_seconds()
                initial_performance = sample_results[0][1] if sample_results else 0.0
                optimized_performance = -best_score if best_score != float('inf') else 0.0
                improvement = optimized_performance - initial_performance
                
                result = OptimizationResult(
                    optimization_id=optimization_id,
                    strategy_id=strategy.strategy_id,
                    method=self.name,
                    success=best_params is not None,
                    initial_performance=initial_performance,
                    optimized_performance=optimized_performance,
                    improvement=improvement,
                    optimal_parameters=best_params or {},
                    optimization_time=optimization_time,
                    iterations=len(optimization_history),
                    convergence_achieved=self._check_convergence(optimization_history),
                    metadata={
                        'parameter_space': parameter_space,
                        'initial_samples': len(sample_results),
                        'acquisition_function': self.acquisition_function,
                        'gp_model_params': self._get_gp_model_params()
                    }
                )
                
                self.optimization_history.append(result)
                logger.info(f"贝叶斯优化完成: 改进 {improvement:.6f}")
                
                return result
                
            except Exception as e:
                logger.error(f"贝叶斯优化出错: {e}")
                return self._create_failed_result(optimization_id, strategy, str(e))
            
            finally:
                self.is_optimizing = False
    
    def _define_parameter_space(self, strategy: BaseStrategy) -> Dict[str, Tuple[float, float]]:
        """定义参数空间"""
        parameter_space = {}
        
        if strategy.strategy_type == StrategyType.REINFORCEMENT:
            parameter_space = {
                'learning_rate': (0.001, 0.1),
                'epsilon': (0.01, 0.3),
                'discount_factor': (0.8, 0.99)
            }
            
            # 根据具体策略类型调整
            if hasattr(strategy, 'batch_size'):
                parameter_space['batch_size'] = (16, 128)
            
        elif strategy.strategy_type == StrategyType.EVOLUTION:
            parameter_space = {
                'population_size': (20, 100),
                'mutation_rate': (0.01, 0.2),
                'crossover_rate': (0.5, 1.0)
            }
            
        return parameter_space
    
    def _generate_initial_samples(self, parameter_space: Dict[str, Tuple[float, float]], 
                                n_samples: int) -> List[Dict[str, float]]:
        """生成初始样本"""
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    sample[param_name] = random.randint(min_val, max_val)
                else:
                    sample[param_name] = random.uniform(min_val, max_val)
            samples.append(sample)
        return samples
    
    def _train_gp_model(self, X: np.ndarray, y: np.ndarray):
        """训练高斯过程模型"""
        try:
            # 定义核函数
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            
            # 创建GP模型
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self.config.random_seed
            )
            
            # 训练模型
            self.gp_model.fit(X, y)
            
        except Exception as e:
            logger.warning(f"GP模型训练失败: {e}")
            self.gp_model = None
    
    def _acquire_next_point(self, parameter_space: Dict[str, Tuple[float, float]]) -> Optional[Dict[str, float]]:
        """获取下一个采样点"""
        if self.gp_model is None:
            return None
        
        try:
            # 生成候选点
            n_candidates = 1000
            candidates = []
            
            for _ in range(n_candidates):
                candidate = {}
                for param_name, (min_val, max_val) in parameter_space.items():
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        candidate[param_name] = random.randint(min_val, max_val)
                    else:
                        candidate[param_name] = random.uniform(min_val, max_val)
                candidates.append(candidate)
            
            # 转换为数组
            candidate_array = np.array([list(c.values()) for c in candidates])
            
            # 获取预测均值和方差
            mu, sigma = self.gp_model.predict(candidate_array, return_std=True)
            
            # 计算采集函数值
            if self.acquisition_function == 'expected_improvement':
                acquisition_values = self._expected_improvement(mu, sigma)
            elif self.acquisition_function == 'upper_confidence_bound':
                acquisition_values = self._upper_confidence_bound(mu, sigma)
            else:
                acquisition_values = mu  # 默认使用均值
            
            # 选择最佳候选点
            best_idx = np.argmax(acquisition_values)
            return candidates[best_idx]
            
        except Exception as e:
            logger.warning(f"获取下一个采样点失败: {e}")
            return None
    
    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """期望改进采集函数"""
        mu_sample = mu
        sigma_sample = sigma
        
        with np.errstate(divide='warn'):
            improvement = mu_sample - np.max(mu_sample) - xi
            Z = improvement / sigma_sample
            ei = improvement * self._normal_cdf(Z) + sigma_sample * self._normal_pdf(Z)
            ei[sigma_sample == 0.0] = 0.0
        
        return ei
    
    def _upper_confidence_bound(self, mu: np.ndarray, sigma: np.ndarray, kappa: float = 2.576) -> np.ndarray:
        """上置信界采集函数"""
        return mu + kappa * sigma
    
    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """正态分布累积分布函数"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """正态分布概率密度函数"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _get_gp_model_params(self) -> Dict[str, Any]:
        """获取GP模型参数"""
        if self.gp_model is None:
            return {}
        
        try:
            return {
                'kernel': str(self.gp_model.kernel_),
                'log_marginal_likelihood': self.gp_model.log_marginal_likelihood_value_,
                'n_features': self.gp_model.n_features_in_
            }
        except:
            return {}

class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("grid_search", config)
        self.parameter_grids = {}
    
    def optimize(self, strategy: BaseStrategy, performance_data: List[StrategyPerformance],
                objective_function: Callable, constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """网格搜索优化"""
        with self.optimization_lock:
            start_time = datetime.now()
            optimization_id = f"grid_search_{strategy.strategy_id}_{int(start_time.timestamp())}"
            
            try:
                self.is_optimizing = True
                logger.info(f"开始网格搜索优化: {optimization_id}")
                
                # 定义参数网格
                parameter_grid = self._define_parameter_grid(strategy)
                if not parameter_grid:
                    return self._create_failed_result(optimization_id, strategy, "无法定义参数网格")
                
                # 生成所有参数组合
                param_combinations = self._generate_param_combinations(parameter_grid)
                logger.info(f"生成 {len(param_combinations)} 个参数组合")
                
                # 并行评估
                best_params = None
                best_score = float('inf')
                evaluation_results = []
                
                with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                    # 提交所有任务
                    future_to_params = {
                        executor.submit(objective_function, params): params 
                        for params in param_combinations
                    }
                    
                    # 收集结果
                    for future in as_completed(future_to_params):
                        params = future_to_params[future]
                        try:
                            score = future.result(timeout=self.config.timeout_seconds)
                            evaluation_results.append((params, score))
                            
                            if score < best_score:
                                best_score = score
                                best_params = params
                            
                            logger.debug(f"评估参数 {params}: 得分 = {score:.6f}")
                            
                        except Exception as e:
                            logger.warning(f"参数 {params} 评估失败: {e}")
                            continue
                
                # 计算优化结果
                optimization_time = (datetime.now() - start_time).total_seconds()
                initial_performance = evaluation_results[0][1] if evaluation_results else 0.0
                optimized_performance = -best_score if best_score != float('inf') else 0.0
                improvement = optimized_performance - initial_performance
                
                result = OptimizationResult(
                    optimization_id=optimization_id,
                    strategy_id=strategy.strategy_id,
                    method=self.name,
                    success=best_params is not None,
                    initial_performance=initial_performance,
                    optimized_performance=optimized_performance,
                    improvement=improvement,
                    optimal_parameters=best_params or {},
                    optimization_time=optimization_time,
                    iterations=len(evaluation_results),
                    convergence_achieved=True,  # 网格搜索总是收敛的
                    metadata={
                        'parameter_grid': parameter_grid,
                        'total_combinations': len(param_combinations),
                        'evaluated_combinations': len(evaluation_results),
                        'grid_efficiency': len(evaluation_results) / len(param_combinations) if param_combinations else 0
                    }
                )
                
                self.optimization_history.append(result)
                logger.info(f"网格搜索优化完成: 评估了 {len(evaluation_results)} 个组合")
                
                return result
                
            except Exception as e:
                logger.error(f"网格搜索优化出错: {e}")
                return self._create_failed_result(optimization_id, strategy, str(e))
            
            finally:
                self.is_optimizing = False
    
    def _define_parameter_grid(self, strategy: BaseStrategy) -> Dict[str, List[Any]]:
        """定义参数网格"""
        parameter_grid = {}
        
        if strategy.strategy_type == StrategyType.REINFORCEMENT:
            parameter_grid = {
                'learning_rate': [0.001, 0.01, 0.05, 0.1],
                'epsilon': [0.01, 0.05, 0.1, 0.2, 0.3],
                'discount_factor': [0.8, 0.85, 0.9, 0.95, 0.99]
            }
            
        elif strategy.strategy_type == StrategyType.EVOLUTION:
            parameter_grid = {
                'population_size': [20, 30, 50, 70, 100],
                'mutation_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'crossover_rate': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        
        return parameter_grid
    
    def _generate_param_combinations(self, parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """生成参数组合"""
        if not parameter_grid:
            return []
        
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        combinations = []
        
        # 使用itertools.product生成所有组合
        from itertools import product
        for values in product(*param_values):
            combination = dict(zip(param_names, values))
            combinations.append(combination)
        
        return combinations

class EvolutionOptimizer(BaseOptimizer):
    """进化优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("evolution_optimizer", config)
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
    def optimize(self, strategy: BaseStrategy, performance_data: List[StrategyPerformance],
                objective_function: Callable, constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """进化优化"""
        with self.optimization_lock:
            start_time = datetime.now()
            optimization_id = f"evolution_{strategy.strategy_id}_{int(start_time.timestamp())}"
            
            try:
                self.is_optimizing = True
                logger.info(f"开始进化优化: {optimization_id}")
                
                # 定义参数空间
                parameter_space = self._define_parameter_space(strategy)
                if not parameter_space:
                    return self._create_failed_result(optimization_id, strategy, "无法定义参数空间")
                
                # 初始化种群
                population = self._initialize_population(parameter_space, self.population_size)
                
                best_individual = None
                best_fitness = float('inf')
                fitness_history = []
                
                for generation in range(self.config.max_iterations):
                    # 评估种群
                    fitness_scores = []
                    for individual in population:
                        try:
                            fitness = objective_function(individual)
                            fitness_scores.append(fitness)
                            
                            if fitness < best_fitness:
                                best_fitness = fitness
                                best_individual = individual.copy()
                                
                        except Exception as e:
                            logger.warning(f"个体评估失败: {e}")
                            fitness_scores.append(float('inf'))
                    
                    fitness_history.append(best_fitness)
                    
                    logger.debug(f"第 {generation+1} 代: 最佳适应度 = {best_fitness:.6f}")
                    
                    # 检查收敛
                    if self._check_convergence(fitness_history):
                        logger.info(f"进化优化收敛于第 {generation+1} 代")
                        break
                    
                    # 选择、交叉、变异
                    population = self._evolve_population(population, fitness_scores, parameter_space)
                
                # 计算优化结果
                optimization_time = (datetime.now() - start_time).total_seconds()
                initial_performance = fitness_history[0] if fitness_history else 0.0
                optimized_performance = -best_fitness if best_fitness != float('inf') else 0.0
                improvement = optimized_performance - initial_performance
                
                result = OptimizationResult(
                    optimization_id=optimization_id,
                    strategy_id=strategy.strategy_id,
                    method=self.name,
                    success=best_individual is not None,
                    initial_performance=initial_performance,
                    optimized_performance=optimized_performance,
                    improvement=improvement,
                    optimal_parameters=best_individual or {},
                    optimization_time=optimization_time,
                    iterations=len(fitness_history),
                    convergence_achieved=self._check_convergence(fitness_history),
                    metadata={
                        'parameter_space': parameter_space,
                        'population_size': self.population_size,
                        'mutation_rate': self.mutation_rate,
                        'crossover_rate': self.crossover_rate,
                        'final_fitness_history': fitness_history[-10:]  # 保存最后10代的适应度
                    }
                )
                
                self.optimization_history.append(result)
                logger.info(f"进化优化完成: 改进 {improvement:.6f}")
                
                return result
                
            except Exception as e:
                logger.error(f"进化优化出错: {e}")
                return self._create_failed_result(optimization_id, strategy, str(e))
            
            finally:
                self.is_optimizing = False
    
    def _define_parameter_space(self, strategy: BaseStrategy) -> Dict[str, Tuple[float, float]]:
        """定义参数空间"""
        parameter_space = {}
        
        if strategy.strategy_type == StrategyType.REINFORCEMENT:
            parameter_space = {
                'learning_rate': (0.001, 0.1),
                'epsilon': (0.01, 0.3),
                'discount_factor': (0.8, 0.99)
            }
            
        elif strategy.strategy_type == StrategyType.EVOLUTION:
            parameter_space = {
                'population_size': (20, 100),
                'mutation_rate': (0.01, 0.2),
                'crossover_rate': (0.5, 1.0)
            }
        
        return parameter_space
    
    def _initialize_population(self, parameter_space: Dict[str, Tuple[float, float]], 
                             population_size: int) -> List[Dict[str, float]]:
        """初始化种群"""
        population = []
        
        for _ in range(population_size):
            individual = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    individual[param_name] = random.randint(min_val, max_val)
                else:
                    individual[param_name] = random.uniform(min_val, max_val)
            population.append(individual)
        
        return population
    
    def _evolve_population(self, population: List[Dict[str, float]], 
                         fitness_scores: List[float],
                         parameter_space: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
        """进化种群"""
        # 排序并选择精英
        sorted_indices = np.argsort(fitness_scores)
        new_population = []
        
        # 保留精英
        for i in range(min(self.elite_size, len(population))):
            idx = sorted_indices[i]
            new_population.append(population[idx].copy())
        
        # 生成新个体
        while len(new_population) < len(population):
            # 锦标赛选择
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, parameter_space)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1, parameter_space)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2, parameter_space)
            
            new_population.extend([child1, child2])
        
        return new_population[:len(population)]
    
    def _tournament_selection(self, population: List[Dict[str, float]], 
                            fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, float]:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float],
                  parameter_space: Dict[str, Tuple[float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """交叉操作"""
        child1, child2 = {}, {}
        
        for param_name in parent1.keys():
            if random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, float], 
               parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """变异操作"""
        mutated = individual.copy()
        
        for param_name, value in mutated.items():
            if param_name in parameter_space:
                min_val, max_val = parameter_space[param_name]
                mutation_strength = (max_val - min_val) * 0.1
                
                if isinstance(value, int):
                    mutation = random.randint(-int(mutation_strength), int(mutation_strength))
                    mutated[param_name] = max(min_val, min(max_val, value + mutation))
                else:
                    mutation = random.gauss(0, mutation_strength)
                    mutated[param_name] = max(min_val, min(max_val, value + mutation))
        
        return mutated

class AdaptiveOptimizer(BaseOptimizer):
    """自适应优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("adaptive_optimizer", config)
        self.optimization_strategies = [
            BayesianOptimizer(config),
            GridSearchOptimizer(config),
            EvolutionOptimizer(config)
        ]
        self.strategy_performance = defaultdict(list)
        self.current_strategy_idx = 0
        self.adaptation_threshold = 10  # 切换策略的阈值
    
    def optimize(self, strategy: BaseStrategy, performance_data: List[StrategyPerformance],
                objective_function: Callable, constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """自适应优化"""
        with self.optimization_lock:
            start_time = datetime.now()
            optimization_id = f"adaptive_{strategy.strategy_id}_{int(start_time.timestamp())}"
            
            try:
                self.is_optimizing = True
                logger.info(f"开始自适应优化: {optimization_id}")
                
                # 选择当前优化策略
                current_strategy = self._select_optimization_strategy()
                logger.info(f"选择优化策略: {current_strategy.name}")
                
                # 执行优化
                result = current_strategy.optimize(strategy, performance_data, objective_function, constraints)
                
                # 更新策略性能记录
                if result.success:
                    self.strategy_performance[current_strategy.name].append({
                        'improvement': result.improvement,
                        'optimization_time': result.optimization_time,
                        'timestamp': datetime.now()
                    })
                    
                    # 保持历史长度
                    history = self.strategy_performance[current_strategy.name]
                    if len(history) > 20:
                        self.strategy_performance[current_strategy.name] = history[-20:]
                
                # 根据结果调整策略选择
                self._adapt_strategy_selection(result)
                
                # 添加自适应元数据
                result.metadata.update({
                    'selected_strategy': current_strategy.name,
                    'strategy_performance_history': len(self.strategy_performance[current_strategy.name]),
                    'available_strategies': [s.name for s in self.optimization_strategies]
                })
                
                logger.info(f"自适应优化完成，使用策略: {current_strategy.name}")
                
                return result
                
            except Exception as e:
                logger.error(f"自适应优化出错: {e}")
                return self._create_failed_result(optimization_id, strategy, str(e))
            
            finally:
                self.is_optimizing = False
    
    def _select_optimization_strategy(self) -> BaseOptimizer:
        """选择优化策略"""
        if not self.strategy_performance:
            # 首次使用，返回第一个策略
            return self.optimization_strategies[self.current_strategy_idx]
        
        # 基于历史性能选择最佳策略
        strategy_scores = {}
        
        for strategy in self.optimization_strategies:
            history = self.strategy_performance[strategy.name]
            if history:
                # 计算平均改进和平均时间
                improvements = [h['improvement'] for h in history]
                times = [h['optimization_time'] for h in history]
                
                avg_improvement = np.mean(improvements)
                avg_time = np.mean(times)
                
                # 综合评分：改进/时间比
                score = avg_improvement / max(avg_time, 1e-6)
                strategy_scores[strategy.name] = score
            else:
                strategy_scores[strategy.name] = 0.0
        
        # 选择得分最高的策略
        best_strategy_name = max(strategy_scores, key=strategy_scores.get)
        
        # 找到对应的策略对象
        for strategy in self.optimization_strategies:
            if strategy.name == best_strategy_name:
                return strategy
        
        # 如果没找到，返回当前策略
        return self.optimization_strategies[self.current_strategy_idx]
    
    def _adapt_strategy_selection(self, result: OptimizationResult):
        """根据优化结果调整策略选择"""
        if not result.success:
            # 如果优化失败，尝试下一个策略
            self.current_strategy_idx = (self.current_strategy_idx + 1) % len(self.optimization_strategies)
            logger.info(f"优化失败，切换到策略索引: {self.current_strategy_idx}")
        
        # 如果某个策略连续失败，降低其优先级
        strategy_name = result.metadata.get('selected_strategy')
        if strategy_name and not result.success:
            # 简化的失败处理逻辑
            recent_failures = sum(
                1 for h in self.strategy_performance[strategy_name][-5:]
                if h['improvement'] <= 0
            )
            
            if recent_failures >= 3:
                # 降低该策略的权重（通过调整选择概率）
                logger.warning(f"策略 {strategy_name} 连续失败，考虑切换策略")

class StrategyOptimizer:
    """策略优化器主类"""
    
    def __init__(self, default_config: Optional[OptimizationConfig] = None):
        self.default_config = default_config or OptimizationConfig()
        self.optimizers = {
            'bayesian': BayesianOptimizer(self.default_config),
            'grid_search': GridSearchOptimizer(self.default_config),
            'evolution': EvolutionOptimizer(self.default_config),
            'adaptive': AdaptiveOptimizer(self.default_config)
        }
        self.optimization_history = []
        self.performance_cache = {}
        
    def optimize_strategy(self, strategy: BaseStrategy, 
                        performance_data: List[StrategyPerformance],
                        optimization_method: str = 'adaptive',
                        objective_function: Optional[Callable] = None,
                        constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """优化策略"""
        try:
            if optimization_method not in self.optimizers:
                raise ValueError(f"不支持的优化方法: {optimization_method}")
            
            # 使用默认目标函数
            if objective_function is None:
                objective_function = self._default_objective_function
            
            optimizer = self.optimizers[optimization_method]
            result = optimizer.optimize(strategy, performance_data, objective_function, constraints)
            
            # 记录优化历史
            self.optimization_history.append(result)
            
            # 更新策略参数
            if result.success and result.optimal_parameters:
                self._update_strategy_parameters(strategy, result.optimal_parameters)
            
            logger.info(f"策略 {strategy.strategy_id} 优化完成，方法: {optimization_method}")
            
            return result
            
        except Exception as e:
            logger.error(f"策略优化出错: {e}")
            return OptimizationResult(
                optimization_id=f"error_{strategy.strategy_id}_{datetime.now().timestamp()}",
                strategy_id=strategy.strategy_id,
                method=optimization_method,
                success=False,
                initial_performance=0.0,
                optimized_performance=0.0,
                improvement=0.0,
                optimal_parameters={},
                optimization_time=0.0,
                iterations=0,
                convergence_achieved=False,
                metadata={'error': str(e)}
            )
    
    def optimize_learning_rate(self, strategy: BaseStrategy, 
                             performance_data: List[StrategyPerformance],
                             learning_rates: Optional[List[float]] = None) -> OptimizationResult:
        """优化学习率"""
        if learning_rates is None:
            learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        def learning_rate_objective(params):
            lr = params['learning_rate']
            # 创建测试策略
            test_strategy = deepcopy(strategy)
            if hasattr(test_strategy, 'learning_rate'):
                test_strategy.learning_rate = lr
            
            # 模拟性能
            return -self._simulate_strategy_performance(test_strategy, performance_data)
        
        # 创建学习率参数空间
        parameter_space = {'learning_rate': learning_rates}
        
        # 使用网格搜索
        optimizer = self.optimizers['grid_search']
        original_parameter_grid = optimizer._define_parameter_grid(strategy)
        optimizer.parameter_grids = {'learning_rate': learning_rates}
        
        # 临时修改参数网格
        original_method = optimizer.optimize
        def custom_optimize(s, perf_data, obj_func, constraints=None):
            # 生成学习率组合
            combinations = [{'learning_rate': lr} for lr in learning_rates]
            
            best_lr = None
            best_score = float('inf')
            
            for lr in learning_rates:
                try:
                    score = learning_rate_objective({'learning_rate': lr})
                    if score < best_score:
                        best_score = score
                        best_lr = lr
                except:
                    continue
            
            return OptimizationResult(
                optimization_id=f"lr_opt_{strategy.strategy_id}",
                strategy_id=strategy.strategy_id,
                method="learning_rate_optimization",
                success=best_lr is not None,
                initial_performance=0.0,
                optimized_performance=-best_score if best_score != float('inf') else 0.0,
                improvement=0.0,
                optimal_parameters={'learning_rate': best_lr} if best_lr else {},
                optimization_time=0.0,
                iterations=len(learning_rates),
                convergence_achieved=True,
                metadata={'tested_rates': learning_rates}
            )
        
        optimizer.optimize = custom_optimize
        result = optimizer.optimize(strategy, performance_data, learning_rate_objective, None)
        
        # 恢复原始方法
        optimizer.optimize = original_method
        
        return result
    
    def optimize_exploration_exploitation(self, strategy: BaseStrategy,
                                        performance_data: List[StrategyPerformance],
                                        epsilon_range: Tuple[float, float] = (0.01, 0.3)) -> OptimizationResult:
        """优化探索-利用平衡"""
        if not hasattr(strategy, 'epsilon'):
            return OptimizationResult(
                optimization_id=f"ee_opt_{strategy.strategy_id}",
                strategy_id=strategy.strategy_id,
                method="exploration_exploitation_optimization",
                success=False,
                initial_performance=0.0,
                optimized_performance=0.0,
                improvement=0.0,
                optimal_parameters={},
                optimization_time=0.0,
                iterations=0,
                convergence_achieved=False,
                metadata={'error': '策略不支持epsilon参数'}
            )
        
        def ee_objective(params):
            epsilon = params['epsilon']
            test_strategy = deepcopy(strategy)
            test_strategy.epsilon = epsilon
            
            return -self._simulate_strategy_performance(test_strategy, performance_data)
        
        # 使用贝叶斯优化
        optimizer = self.optimizers['bayesian']
        
        # 临时修改参数空间
        original_method = optimizer.optimize
        def custom_optimize(s, perf_data, obj_func, constraints=None):
            min_eps, max_eps = epsilon_range
            
            # 测试几个epsilon值
            test_epsilons = np.linspace(min_eps, max_eps, 10)
            best_eps = None
            best_score = float('inf')
            
            for eps in test_epsilons:
                try:
                    score = ee_objective({'epsilon': eps})
                    if score < best_score:
                        best_score = score
                        best_eps = eps
                except:
                    continue
            
            return OptimizationResult(
                optimization_id=f"ee_opt_{strategy.strategy_id}",
                strategy_id=strategy.strategy_id,
                method="exploration_exploitation_optimization",
                success=best_eps is not None,
                initial_performance=0.0,
                optimized_performance=-best_score if best_score != float('inf') else 0.0,
                improvement=0.0,
                optimal_parameters={'epsilon': best_eps} if best_eps else {},
                optimization_time=0.0,
                iterations=len(test_epsilons),
                convergence_achieved=True,
                metadata={'epsilon_range': epsilon_range}
            )
        
        optimizer.optimize = custom_optimize
        result = optimizer.optimize(strategy, performance_data, ee_objective, None)
        
        # 恢复原始方法
        optimizer.optimize = original_method
        
        return result
    
    def multi_objective_optimize(self, strategy: BaseStrategy,
                               performance_data: List[StrategyPerformance],
                               objectives: Dict[str, Callable],
                               weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """多目标优化"""
        if weights is None:
            weights = {obj: 1.0 for obj in objectives}
        
        def multi_objective_function(params):
            scores = []
            for obj_name, obj_func in objectives.items():
                try:
                    score = obj_func(params)
                    scores.append(score * weights.get(obj_name, 1.0))
                except:
                    scores.append(0.0)
            
            return -np.sum(scores)  # 最小化负加权得分和
        
        # 使用进化优化进行多目标优化
        optimizer = self.optimizers['evolution']
        return optimizer.optimize(strategy, performance_data, multi_objective_function)
    
    def batch_optimize(self, strategies: List[BaseStrategy],
                      performance_data: Dict[str, List[StrategyPerformance]],
                      optimization_method: str = 'adaptive') -> Dict[str, OptimizationResult]:
        """批量优化"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.default_config.parallel_workers) as executor:
            # 提交所有优化任务
            future_to_strategy = {
                executor.submit(
                    self.optimize_strategy, 
                    strategy, 
                    performance_data.get(strategy.strategy_id, []),
                    optimization_method
                ): strategy 
                for strategy in strategies
            }
            
            # 收集结果
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result(timeout=self.default_config.timeout_seconds)
                    results[strategy.strategy_id] = result
                except Exception as e:
                    logger.error(f"策略 {strategy.strategy_id} 批量优化失败: {e}")
                    results[strategy.strategy_id] = self._create_failed_result(
                        f"batch_error_{strategy.strategy_id}", strategy, str(e)
                    )
        
        logger.info(f"批量优化完成，成功 {sum(1 for r in results.values() if r.success)}/{len(strategies)}")
        
        return results
    
    def _default_objective_function(self, parameters: Dict[str, Any]) -> float:
        """默认目标函数"""
        return 0.0  # 需要在具体优化中重写
    
    def _simulate_strategy_performance(self, strategy: BaseStrategy,
                                     performance_data: List[StrategyPerformance]) -> float:
        """模拟策略性能"""
        if not performance_data:
            return 0.0
        
        # 基础性能
        base_performance = np.mean([p.return_rate for p in performance_data])
        
        # 基于策略类型调整
        if strategy.strategy_type == StrategyType.REINFORCEMENT:
            if hasattr(strategy, 'learning_rate'):
                lr_factor = 1.0 / (1.0 + abs(strategy.learning_rate - 0.01) * 10)
                base_performance *= lr_factor
        
        return max(0.0, base_performance)
    
    def _update_strategy_parameters(self, strategy: BaseStrategy, parameters: Dict[str, Any]):
        """更新策略参数"""
        try:
            for param_name, value in parameters.items():
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, value)
                    logger.info(f"更新策略 {strategy.strategy_id} 参数 {param_name} = {value}")
        except Exception as e:
            logger.error(f"更新策略参数出错: {e}")
    
    def _create_failed_result(self, optimization_id: str, strategy: BaseStrategy, 
                            error_message: str) -> OptimizationResult:
        """创建失败的优化结果"""
        return OptimizationResult(
            optimization_id=optimization_id,
            strategy_id=strategy.strategy_id,
            method="unknown",
            success=False,
            initial_performance=0.0,
            optimized_performance=0.0,
            improvement=0.0,
            optimal_parameters={},
            optimization_time=0.0,
            iterations=0,
            convergence_achieved=False,
            metadata={'error': error_message}
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        if not self.optimization_history:
            return {'message': '没有优化历史'}
        
        successful_optimizations = [r for r in self.optimization_history if r.success]
        
        stats = {
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': len(successful_optimizations),
            'success_rate': len(successful_optimizations) / len(self.optimization_history),
            'average_improvement': np.mean([r.improvement for r in successful_optimizations]) if successful_optimizations else 0.0,
            'average_optimization_time': np.mean([r.optimization_time for r in self.optimization_history]),
            'method_performance': {},
            'recent_optimizations': [r.optimization_id for r in self.optimization_history[-10:]]
        }
        
        # 按方法统计性能
        method_stats = defaultdict(lambda: {'count': 0, 'success_rate': 0, 'avg_improvement': 0})
        
        for result in self.optimization_history:
            method = result.method
            method_stats[method]['count'] += 1
            if result.success:
                method_stats[method]['success_rate'] += 1
                method_stats[method]['avg_improvement'] += result.improvement
        
        for method, stats_dict in method_stats.items():
            if stats_dict['count'] > 0:
                stats_dict['success_rate'] /= stats_dict['count']
                stats_dict['avg_improvement'] /= stats_dict['count']
        
        stats['method_performance'] = dict(method_stats)
        
        return stats
    
    def export_optimization_history(self, filepath: str) -> bool:
        """导出优化历史"""
        try:
            export_data = {
                'optimization_history': [
                    {
                        'optimization_id': r.optimization_id,
                        'strategy_id': r.strategy_id,
                        'method': r.method,
                        'success': r.success,
                        'initial_performance': r.initial_performance,
                        'optimized_performance': r.optimized_performance,
                        'improvement': r.improvement,
                        'optimal_parameters': r.optimal_parameters,
                        'optimization_time': r.optimization_time,
                        'iterations': r.iterations,
                        'convergence_achieved': r.convergence_achieved,
                        'metadata': r.metadata,
                        'timestamp': datetime.now().isoformat()
                    }
                    for r in self.optimization_history
                ],
                'export_timestamp': datetime.now().isoformat(),
                'statistics': self.get_optimization_statistics()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"导出优化历史出错: {e}")
            return False