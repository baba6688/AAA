"""
F1参数学习器
实现多种参数学习算法和优化策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import json
import logging
from datetime import datetime, timedelta
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """参数空间定义"""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Optional[Tuple[float, float]] = None
    values: Optional[List[Any]] = None
    prior: Optional[Dict[str, Any]] = None


@dataclass
class ParameterSet:
    """参数集合"""
    params: Dict[str, Any]
    score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningResult:
    """学习结果"""
    best_params: Dict[str, Any]
    best_score: float
    history: List[ParameterSet]
    convergence_info: Dict[str, Any]
    learning_time: float


class ParameterOptimizer(ABC):
    """参数优化器基类"""
    
    def __init__(self, parameter_space: Dict[str, ParameterSpace]):
        self.parameter_space = parameter_space
        self.history = []
        
    @abstractmethod
    def optimize(self, objective_function: Callable, max_iterations: int = 100) -> ParameterSet:
        """优化参数"""
        pass
    
    def _validate_params(self, params: Dict[str, Any]) -> bool:
        """验证参数是否在定义的空间内"""
        for param_name, value in params.items():
            if param_name not in self.parameter_space:
                return False
            
            space = self.parameter_space[param_name]
            if space.param_type == 'continuous':
                if not (space.bounds[0] <= value <= space.bounds[1]):
                    return False
            elif space.param_type == 'discrete':
                if value not in space.values:
                    return False
            elif space.param_type == 'categorical':
                if value not in space.values:
                    return False
        
        return True


class GridSearchOptimizer(ParameterOptimizer):
    """网格搜索优化器"""
    
    def __init__(self, parameter_space: Dict[str, ParameterSpace], resolution: int = 10):
        super().__init__(parameter_space)
        self.resolution = resolution
        
    def optimize(self, objective_function: Callable, max_iterations: int = None) -> ParameterSet:
        """网格搜索优化"""
        logger.info("开始网格搜索优化")
        
        # 生成网格点
        grid_points = self._generate_grid()
        
        best_params = None
        best_score = float('-inf')
        
        for i, params in enumerate(grid_points):
            if max_iterations and i >= max_iterations:
                break
                
            try:
                score = objective_function(params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
                self.history.append(ParameterSet(params, score))
                
            except Exception as e:
                logger.warning(f"参数评估失败: {params}, 错误: {e}")
                continue
        
        return ParameterSet(best_params, best_score)
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """生成网格点"""
        grid_points = []
        
        # 创建参数组合
        param_grids = {}
        for name, space in self.parameter_space.items():
            if space.param_type == 'continuous':
                param_grids[name] = np.linspace(space.bounds[0], space.bounds[1], self.resolution)
            elif space.param_type == 'discrete':
                param_grids[name] = space.values
            elif space.param_type == 'categorical':
                param_grids[name] = space.values
        
        # 生成笛卡尔积
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        for combination in self._cartesian_product(param_values):
            params = dict(zip(param_names, combination))
            grid_points.append(params)
        
        return grid_points
    
    def _cartesian_product(self, lists: List[List]) -> List[Tuple]:
        """计算笛卡尔积"""
        if not lists:
            return [()]
        
        result = [[]]
        for lst in lists:
            result = [x + [y] for x in result for y in lst]
        
        return [tuple(x) for x in result]


class RandomSearchOptimizer(ParameterOptimizer):
    """随机搜索优化器"""
    
    def __init__(self, parameter_space: Dict[str, ParameterSpace], random_state: int = 42):
        super().__init__(parameter_space)
        self.random_state = random_state
        np.random.seed(random_state)
        
    def optimize(self, objective_function: Callable, max_iterations: int = 100) -> ParameterSet:
        """随机搜索优化"""
        logger.info(f"开始随机搜索优化，最大迭代次数: {max_iterations}")
        
        best_params = None
        best_score = float('-inf')
        
        for i in range(max_iterations):
            # 随机生成参数
            params = self._generate_random_params()
            
            try:
                score = objective_function(params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
                self.history.append(ParameterSet(params, score))
                
            except Exception as e:
                logger.warning(f"参数评估失败: {params}, 错误: {e}")
                continue
        
        return ParameterSet(best_params, best_score)
    
    def _generate_random_params(self) -> Dict[str, Any]:
        """随机生成参数"""
        params = {}
        
        for name, space in self.parameter_space.items():
            if space.param_type == 'continuous':
                params[name] = np.random.uniform(space.bounds[0], space.bounds[1])
            elif space.param_type == 'discrete':
                params[name] = np.random.choice(space.values)
            elif space.param_type == 'categorical':
                params[name] = np.random.choice(space.values)
        
        return params


class BayesianOptimizer(ParameterOptimizer):
    """贝叶斯优化器（简化版）"""
    
    def __init__(self, parameter_space: Dict[str, ParameterSpace], acquisition_function: str = 'ei'):
        super().__init__(parameter_space)
        self.acquisition_function = acquisition_function
        self.X_evaluated = []
        self.y_evaluated = []
        
    def optimize(self, objective_function: Callable, max_iterations: int = 50) -> ParameterSet:
        """贝叶斯优化"""
        logger.info(f"开始贝叶斯优化，最大迭代次数: {max_iterations}")
        
        # 初始化：随机采样几个点
        n_initial = min(5, max_iterations // 4)
        
        for i in range(n_initial):
            params = self._generate_random_params()
            try:
                score = objective_function(params)
                self.X_evaluated.append(params)
                self.y_evaluated.append(score)
                self.history.append(ParameterSet(params, score))
            except Exception as e:
                logger.warning(f"初始采样失败: {params}, 错误: {e}")
        
        # 贝叶斯优化循环
        for i in range(max_iterations - n_initial):
            # 选择下一个采样点
            next_params = self._select_next_point()
            
            try:
                score = objective_function(next_params)
                self.X_evaluated.append(next_params)
                self.y_evaluated.append(score)
                self.history.append(ParameterSet(next_params, score))
                
            except Exception as e:
                logger.warning(f"贝叶斯优化采样失败: {next_params}, 错误: {e}")
                # 随机采样一个点
                next_params = self._generate_random_params()
                score = objective_function(next_params)
                self.X_evaluated.append(next_params)
                self.y_evaluated.append(score)
                self.history.append(ParameterSet(next_params, score))
        
        # 返回最佳参数
        best_idx = np.argmax(self.y_evaluated)
        best_params = self.X_evaluated[best_idx]
        best_score = self.y_evaluated[best_idx]
        
        return ParameterSet(best_params, best_score)
    
    def _select_next_point(self) -> Dict[str, Any]:
        """选择下一个采样点（简化版）"""
        # 简化的获取函数：使用高斯过程代理模型
        # 在实际实现中，这里应该使用更复杂的获取函数
        
        if len(self.X_evaluated) < 2:
            return self._generate_random_params()
        
        # 简单的探索策略：在历史最佳点附近搜索
        best_idx = np.argmax(self.y_evaluated)
        best_params = self.X_evaluated[best_idx]
        
        # 在最佳点附近添加噪声
        next_params = {}
        for name, space in self.parameter_space.items():
            if space.param_type == 'continuous':
                noise = np.random.normal(0, 0.1)
                value = best_params[name] + noise
                value = np.clip(value, space.bounds[0], space.bounds[1])
                next_params[name] = value
            else:
                next_params[name] = best_params[name]
        
        return next_params
    
    def _generate_random_params(self) -> Dict[str, Any]:
        """随机生成参数"""
        params = {}
        
        for name, space in self.parameter_space.items():
            if space.param_type == 'continuous':
                params[name] = np.random.uniform(space.bounds[0], space.bounds[1])
            elif space.param_type == 'discrete':
                params[name] = np.random.choice(space.values)
            elif space.param_type == 'categorical':
                params[name] = np.random.choice(space.values)
        
        return params


class ParameterLearner:
    """参数学习器主类"""
    
    def __init__(self, parameter_space: Dict[str, ParameterSpace]):
        self.parameter_space = parameter_space
        self.optimizers = {
            'grid_search': GridSearchOptimizer(parameter_space),
            'random_search': RandomSearchOptimizer(parameter_space),
            'bayesian': BayesianOptimizer(parameter_space)
        }
        self.learning_history = deque(maxlen=1000)
        self.adaptation_history = deque(maxlen=100)
        self.performance_metrics = defaultdict(list)
        
    def learn_parameters(self, 
                        objective_function: Callable,
                        strategy: str = 'auto',
                        max_iterations: int = 100,
                        validation_function: Optional[Callable] = None,
                        early_stopping_patience: int = 10) -> LearningResult:
        """学习最优参数"""
        
        logger.info(f"开始参数学习，策略: {strategy}, 最大迭代次数: {max_iterations}")
        
        start_time = datetime.now()
        
        # 选择优化策略
        if strategy == 'auto':
            strategy = self._select_best_strategy()
            logger.info(f"自动选择策略: {strategy}")
        
        optimizer = self.optimizers[strategy]
        
        # 执行优化
        best_result = optimizer.optimize(objective_function, max_iterations)
        
        # 验证结果
        if validation_function:
            validation_score = validation_function(best_result.params)
            best_result.score = validation_score
        
        learning_time = (datetime.now() - start_time).total_seconds()
        
        # 创建学习结果
        result = LearningResult(
            best_params=best_result.params,
            best_score=best_result.score,
            history=list(optimizer.history),
            convergence_info=self._analyze_convergence(optimizer.history),
            learning_time=learning_time
        )
        
        # 保存学习历史
        self.learning_history.append(result)
        
        logger.info(f"参数学习完成，最佳得分: {best_result.score:.4f}, 用时: {learning_time:.2f}秒")
        
        return result
    
    def adapt_parameters(self, 
                        current_params: Dict[str, Any],
                        feedback: Dict[str, float],
                        adaptation_rate: float = 0.1) -> Dict[str, Any]:
        """参数自适应调整"""
        
        logger.info("开始参数自适应调整")
        
        adapted_params = current_params.copy()
        
        for param_name, adjustment in feedback.items():
            if param_name in self.parameter_space:
                space = self.parameter_space[param_name]
                
                if space.param_type == 'continuous':
                    # 连续参数的梯度下降式调整
                    current_value = adapted_params[param_name]
                    new_value = current_value + adaptation_rate * adjustment
                    
                    # 约束到参数空间
                    if space.bounds:
                        new_value = np.clip(new_value, space.bounds[0], space.bounds[1])
                    
                    adapted_params[param_name] = new_value
                
                elif space.param_type in ['discrete', 'categorical']:
                    # 离散/分类参数的选择调整
                    if abs(adjustment) > 0.5:  # 阈值判断
                        # 选择不同的值
                        available_values = space.values
                        current_idx = available_values.index(adapted_params[param_name])
                        
                        if adjustment > 0:
                            new_idx = min(current_idx + 1, len(available_values) - 1)
                        else:
                            new_idx = max(current_idx - 1, 0)
                        
                        adapted_params[param_name] = available_values[new_idx]
        
        # 记录适应历史
        adaptation_record = {
            'original_params': current_params,
            'adapted_params': adapted_params,
            'feedback': feedback,
            'timestamp': datetime.now()
        }
        self.adaptation_history.append(adaptation_record)
        
        logger.info("参数自适应调整完成")
        return adapted_params
    
    def evaluate_parameter_combination(self, 
                                     params: Dict[str, Any],
                                     evaluation_function: Callable,
                                     cross_validation_folds: int = 5) -> Dict[str, float]:
        """评估参数组合效果"""
        
        scores = []
        
        for fold in range(cross_validation_folds):
            try:
                score = evaluation_function(params, fold)
                scores.append(score)
            except Exception as e:
                logger.warning(f"交叉验证第{fold}折失败: {e}")
                continue
        
        if not scores:
            return {'mean_score': 0.0, 'std_score': 0.0, 'cv_scores': []}
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'cv_scores': scores,
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }
    
    def track_parameter_history(self, 
                              params_history: List[Dict[str, Any]],
                              scores_history: List[float]) -> Dict[str, Any]:
        """跟踪参数历史"""
        
        df = pd.DataFrame(params_history)
        df['score'] = scores_history
        
        # 计算统计信息
        stats = {
            'parameter_statistics': {},
            'score_statistics': {
                'mean': np.mean(scores_history),
                'std': np.std(scores_history),
                'min': np.min(scores_history),
                'max': np.max(scores_history)
            },
            'parameter_correlations': {},
            'convergence_analysis': self._analyze_convergence_history(df, scores_history)
        }
        
        # 只对数值型参数计算相关性
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(df) > 1 and len(numeric_columns) > 1:
            stats['parameter_correlations'] = df[numeric_columns].corr()['score'].drop('score').to_dict()
        
        # 参数统计
        for param in df.columns:
            if param != 'score':
                if pd.api.types.is_numeric_dtype(df[param]):
                    stats['parameter_statistics'][param] = {
                        'mean': df[param].mean(),
                        'std': df[param].std(),
                        'min': df[param].min(),
                        'max': df[param].max()
                    }
                else:
                    # 对于分类参数，提供不同的统计信息
                    stats['parameter_statistics'][param] = {
                        'unique_values': df[param].unique().tolist(),
                        'value_counts': df[param].value_counts().to_dict(),
                        'most_frequent': df[param].mode().iloc[0] if not df[param].mode().empty else None
                    }
        
        return stats
    
    def optimize_parameter_combinations(self, 
                                      base_params: Dict[str, Any],
                                      combination_strategy: str = 'genetic',
                                      population_size: int = 50,
                                      generations: int = 20) -> Dict[str, Any]:
        """参数组合优化"""
        
        logger.info(f"开始参数组合优化，策略: {combination_strategy}")
        
        if combination_strategy == 'genetic':
            return self._genetic_optimization(base_params, population_size, generations)
        elif combination_strategy == 'simulated_annealing':
            return self._simulated_annealing_optimization(base_params)
        else:
            raise ValueError(f"不支持的组合优化策略: {combination_strategy}")
    
    def _genetic_optimization(self, 
                            base_params: Dict[str, Any],
                            population_size: int,
                            generations: int) -> Dict[str, Any]:
        """遗传算法优化"""
        
        # 初始化种群
        population = self._initialize_population(base_params, population_size)
        
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(generations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                try:
                    # 这里需要目标函数，实际使用时应该传入
                    fitness = self._evaluate_individual(individual)
                    fitness_scores.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                        
                except Exception as e:
                    fitness_scores.append(float('-inf'))
            
            # 选择、交叉、变异
            population = self._evolve_population(population, fitness_scores)
            
            logger.info(f"第{generation+1}代，最佳适应度: {best_fitness:.4f}")
        
        return {
            'best_params': best_individual,
            'best_fitness': best_fitness,
            'final_population': population,
            'generations': generations
        }
    
    def _simulated_annealing_optimization(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """模拟退火优化"""
        
        current_params = base_params.copy()
        current_fitness = self._evaluate_individual(current_params)
        
        best_params = current_params.copy()
        best_fitness = current_fitness
        
        # 模拟退火参数
        initial_temperature = 100.0
        final_temperature = 0.01
        cooling_rate = 0.95
        
        temperature = initial_temperature
        iteration = 0
        
        while temperature > final_temperature:
            # 生成邻域解
            neighbor_params = self._generate_neighbor(current_params)
            
            try:
                neighbor_fitness = self._evaluate_individual(neighbor_params)
                
                # 接受准则
                delta = neighbor_fitness - current_fitness
                if delta > 0 or np.random.random() < np.exp(delta / temperature):
                    current_params = neighbor_params
                    current_fitness = neighbor_fitness
                    
                    if neighbor_fitness > best_fitness:
                        best_params = neighbor_params.copy()
                        best_fitness = neighbor_fitness
            
            except Exception as e:
                logger.warning(f"邻域解评估失败: {e}")
            
            temperature *= cooling_rate
            iteration += 1
        
        return {
            'best_params': best_params,
            'best_fitness': best_fitness,
            'iterations': iteration,
            'final_temperature': temperature
        }
    
    def _initialize_population(self, base_params: Dict[str, Any], population_size: int) -> List[Dict[str, Any]]:
        """初始化种群"""
        population = []
        
        for _ in range(population_size):
            individual = {}
            for name, space in self.parameter_space.items():
                if space.param_type == 'continuous':
                    individual[name] = np.random.uniform(space.bounds[0], space.bounds[1])
                elif space.param_type == 'discrete':
                    individual[name] = np.random.choice(space.values)
                elif space.param_type == 'categorical':
                    individual[name] = np.random.choice(space.values)
            
            population.append(individual)
        
        return population
    
    def _evaluate_individual(self, individual: Dict[str, Any]) -> float:
        """评估个体适应度（需要目标函数）"""
        # 这里应该调用实际的目标函数
        # 为了示例，返回随机值
        return np.random.random()
    
    def _evolve_population(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """进化种群"""
        # 简化的进化操作
        # 实际实现中应该包括选择、交叉、变异等操作
        
        # 选择（轮盘赌选择）
        selected = self._roulette_wheel_selection(population, fitness_scores)
        
        # 交叉和变异
        new_population = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            child1, child2 = self._crossover(parent1, parent2)
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:len(population)]
    
    def _roulette_wheel_selection(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """轮盘赌选择"""
        # 转换为正数
        min_fitness = min(fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-10 for f in fitness_scores]
        
        # 计算选择概率
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        # 选择
        selected = []
        for _ in range(len(population)):
            r = np.random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(population[i])
                    break
        
        return selected
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """交叉操作"""
        child1, child2 = {}, {}
        
        for name in parent1.keys():
            if np.random.random() < 0.5:
                child1[name] = parent1[name]
                child2[name] = parent2[name]
            else:
                child1[name] = parent2[name]
                child2[name] = parent1[name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()
        
        for name, space in self.parameter_space.items():
            if np.random.random() < 0.1:  # 变异概率
                if space.param_type == 'continuous':
                    noise = np.random.normal(0, 0.1)
                    value = mutated[name] + noise
                    value = np.clip(value, space.bounds[0], space.bounds[1])
                    mutated[name] = value
                elif space.param_type == 'discrete':
                    mutated[name] = np.random.choice(space.values)
                elif space.param_type == 'categorical':
                    mutated[name] = np.random.choice(space.values)
        
        return mutated
    
    def _generate_neighbor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """生成邻域解"""
        neighbor = params.copy()
        
        # 随机选择一个参数进行微调
        param_name = np.random.choice(list(params.keys()))
        space = self.parameter_space[param_name]
        
        if space.param_type == 'continuous':
            noise = np.random.normal(0, 0.05)
            neighbor[param_name] += noise
            neighbor[param_name] = np.clip(neighbor[param_name], space.bounds[0], space.bounds[1])
        else:
            neighbor[param_name] = np.random.choice(space.values)
        
        return neighbor
    
    def _select_best_strategy(self) -> str:
        """选择最佳学习策略"""
        # 基于参数空间特征选择策略
        
        continuous_params = sum(1 for space in self.parameter_space.values() 
                               if space.param_type == 'continuous')
        discrete_params = sum(1 for space in self.parameter_space.values() 
                             if space.param_type == 'discrete')
        categorical_params = sum(1 for space in self.parameter_space.values() 
                                if space.param_type == 'categorical')
        
        total_params = len(self.parameter_space)
        
        # 简单的策略选择逻辑
        if total_params <= 3 and continuous_params == total_params:
            return 'grid_search'
        elif total_params > 10:
            return 'random_search'
        else:
            return 'bayesian'
    
    def _analyze_convergence(self, history: List[ParameterSet]) -> Dict[str, Any]:
        """分析收敛性"""
        if len(history) < 2:
            return {'converged': False, 'reason': 'insufficient_history'}
        
        scores = [h.score for h in history]
        
        # 检查最近几次迭代的改善
        recent_improvements = []
        window_size = min(10, len(scores) // 4)
        
        for i in range(window_size, len(scores)):
            recent_scores = scores[i-window_size:i]
            if len(set(recent_scores)) > 1:
                improvement = max(recent_scores) - min(recent_scores)
                recent_improvements.append(improvement)
        
        if not recent_improvements:
            return {'converged': True, 'reason': 'no_improvement_needed'}
        
        # 判断收敛
        avg_improvement = np.mean(recent_improvements)
        threshold = 0.001
        
        return {
            'converged': avg_improvement < threshold,
            'avg_improvement': avg_improvement,
            'threshold': threshold,
            'reason': 'low_improvement' if avg_improvement < threshold else 'active_search'
        }
    
    def _analyze_convergence_history(self, df: pd.DataFrame, scores: List[float]) -> Dict[str, Any]:
        """分析历史收敛性"""
        if len(scores) < 2:
            return {'status': 'insufficient_data'}
        
        # 计算移动平均
        window = min(10, len(scores) // 3)
        moving_avg = pd.Series(scores).rolling(window=window).mean()
        
        # 检查趋势
        recent_trend = moving_avg.iloc[-1] - moving_avg.iloc[-window-1] if len(moving_avg) > window else 0
        
        return {
            'recent_trend': recent_trend,
            'moving_average_final': moving_avg.iloc[-1] if not pd.isna(moving_avg.iloc[-1]) else scores[-1],
            'stability': np.std(scores[-window:]) if len(scores) >= window else 0
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        if not self.learning_history:
            return {'status': 'no_learning_history'}
        
        recent_results = list(self.learning_history)[-10:]  # 最近10次学习
        
        return {
            'total_learning_sessions': len(self.learning_history),
            'recent_performance': {
                'best_scores': [r.best_score for r in recent_results],
                'avg_learning_time': np.mean([r.learning_time for r in recent_results]),
                'convergence_rate': sum(1 for r in recent_results if r.convergence_info.get('converged', False)) / len(recent_results)
            },
            'adaptation_stats': {
                'total_adaptations': len(self.adaptation_history),
                'recent_adaptations': list(self.adaptation_history)[-5:]
            }
        }
    
    def export_history(self, filepath: str):
        """导出学习历史"""
        
        def convert_numpy_types(obj):
            """转换numpy类型为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.str_):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        history_data = {
            'learning_history': [
                {
                    'best_params': convert_numpy_types(result.best_params),
                    'best_score': float(result.best_score),
                    'learning_time': result.learning_time,
                    'convergence_info': convert_numpy_types(result.convergence_info),
                    'timestamp': datetime.now().isoformat()
                }
                for result in self.learning_history
            ],
            'adaptation_history': [
                {
                    'original_params': convert_numpy_types(record['original_params']),
                    'adapted_params': convert_numpy_types(record['adapted_params']),
                    'feedback': convert_numpy_types(record['feedback']),
                    'timestamp': record['timestamp'].isoformat()
                }
                for record in self.adaptation_history
            ],
            'parameter_space': {
                name: {
                    'param_type': space.param_type,
                    'bounds': space.bounds,
                    'values': space.values,
                    'prior': space.prior
                }
                for name, space in self.parameter_space.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"学习历史已导出到: {filepath}")


# 使用示例
if __name__ == "__main__":
    # 定义参数空间
    parameter_space = {
        'learning_rate': ParameterSpace('learning_rate', 'continuous', bounds=(0.001, 0.1)),
        'batch_size': ParameterSpace('batch_size', 'discrete', values=[16, 32, 64, 128]),
        'optimizer': ParameterSpace('optimizer', 'categorical', values=['adam', 'sgd', 'rmsprop'])
    }
    
    # 创建参数学习器
    learner = ParameterLearner(parameter_space)
    
    # 定义目标函数（示例）
    def objective_function(params):
        # 模拟参数评估
        score = -((params['learning_rate'] - 0.01) ** 2 + 
                 (params['batch_size'] - 32) ** 2 / 1000 +
                 (params['optimizer'] == 'adam') * 0.1)
        return score
    
    # 学习参数
    result = learner.learn_parameters(
        objective_function=objective_function,
        strategy='bayesian',
        max_iterations=50
    )
    
    print(f"最佳参数: {result.best_params}")
    print(f"最佳得分: {result.best_score:.4f}")
    print(f"学习时间: {result.learning_time:.2f}秒")
    
    # 参数自适应
    feedback = {'learning_rate': 0.01, 'batch_size': -0.5}
    adapted_params = learner.adapt_parameters(result.best_params, feedback)
    print(f"适应后参数: {adapted_params}")
    
    # 导出历史
    learner.export_history('/workspace/parameter_learning_history.json')