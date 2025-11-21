"""
J5优化工具模块

提供完整的优化算法集合，包括经典优化、智能优化、约束优化、
多目标优化、贝叶斯优化等功能。支持并行计算和分布式优化。

作者: J5系统
版本: 1.0.0
日期: 2025-11-06
"""

import numpy as np
import logging
import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Protocol
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import time
from functools import wraps
from collections import defaultdict
import json
import pickle
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationError(Exception):
    """优化算法异常基类"""
    pass


class ConvergenceError(OptimizationError):
    """收敛失败异常"""
    pass


class ConstraintViolationError(OptimizationError):
    """约束违反异常"""
    pass


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    x: np.ndarray
    fval: float
    success: bool
    iterations: int
    message: str
    convergence_history: List[float] = None
    constraint_violations: List[float] = None
    execution_time: float = 0.0


@dataclass
class MultiObjectiveResult:
    """多目标优化结果"""
    pareto_front: np.ndarray
    pareto_values: np.ndarray
    pareto_solutions: np.ndarray
    convergence_history: List[List[float]] = None
    hypervolume: float = 0.0
    execution_time: float = 0.0


def timer(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if hasattr(result, 'execution_time'):
            result.execution_time = end_time - start_time
        return result
    return wrapper


def check_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> bool:
    """检查变量是否在边界内"""
    if bounds is None:
        return True
    for i, (lower, upper) in enumerate(bounds):
        if x[i] < lower or x[i] > upper:
            return False
    return True


def project_to_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """将变量投影到边界内"""
    if bounds is None:
        return x
    projected = x.copy()
    for i, (lower, upper) in enumerate(bounds):
        projected[i] = np.clip(projected[i], lower, upper)
    return projected


class Optimizer(ABC):
    """优化器基类"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None, 
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 verbose: bool = True):
        """
        初始化优化器
        
        Args:
            bounds: 变量边界 [(min1, max1), (min2, max2), ...]
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            verbose: 是否输出详细信息
        """
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
    @abstractmethod
    def optimize(self, func: Callable, x0: np.ndarray) -> OptimizationResult:
        """执行优化"""
        pass
        
    def log_iteration(self, iteration: int, fval: float, x: np.ndarray):
        """记录迭代信息"""
        if self.verbose:
            logger.info(f"迭代 {iteration}: f(x) = {fval:.6f}, x = {x}")


class ClassicalOptimizer(Optimizer):
    """经典优化算法基类"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 learning_rate: float = 0.01, verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.learning_rate = learning_rate
        
    def numerical_gradient(self, func: Callable, x: np.ndarray, 
                          epsilon: float = 1e-8) -> np.ndarray:
        """数值计算梯度"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
        return grad
        
    def numerical_hessian(self, func: Callable, x: np.ndarray,
                         epsilon: float = 1e-6) -> np.ndarray:
        """数值计算Hessian矩阵"""
        n = len(x)
        hessian = np.zeros((n, n))
        f_x = func(x)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # 对角线元素
                    x_pp = x.copy()
                    x_mm = x.copy()
                    x_pp[i] += 2 * epsilon
                    x_mm[i] -= 2 * epsilon
                    hessian[i, j] = (func(x_pp) - 2 * f_x + func(x_mm)) / (4 * epsilon**2)
                else:
                    # 非对角线元素
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()
                    x_pp[i] += epsilon
                    x_pp[j] += epsilon
                    x_pm[i] += epsilon
                    x_pm[j] -= epsilon
                    x_mp[i] -= epsilon
                    x_mp[j] += epsilon
                    x_mm[i] -= epsilon
                    x_mm[j] -= epsilon
                    
                    hessian[i, j] = (func(x_pp) - func(x_pm) - 
                                   func(x_mp) + func(x_mm)) / (4 * epsilon**2)
        return hessian


class GradientDescentOptimizer(ClassicalOptimizer):
    """梯度下降优化器"""
    
    @timer
    def optimize(self, func: Callable, x0: np.ndarray) -> OptimizationResult:
        """梯度下降优化"""
        x = x0.copy()
        convergence_history = []
        
        try:
            for iteration in range(self.max_iterations):
                # 计算梯度
                grad = self.numerical_gradient(func, x)
                grad_norm = np.linalg.norm(grad)
                
                if grad_norm < self.tolerance:
                    fval = func(x)
                    self.log_iteration(iteration, fval, x)
                    return OptimizationResult(
                        x=x, fval=fval, success=True, iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                # 更新参数
                x = x - self.learning_rate * grad
                if self.bounds:
                    x = project_to_bounds(x, self.bounds)
                
                fval = func(x)
                convergence_history.append(fval)
                self.log_iteration(iteration, fval, x)
                
        except Exception as e:
            logger.error(f"梯度下降优化失败: {e}")
            return OptimizationResult(
                x=x, fval=func(x), success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        return OptimizationResult(
            x=x, fval=func(x), success=False, iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )


class NewtonOptimizer(ClassicalOptimizer):
    """牛顿法优化器"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, 0.01, verbose)
        
    @timer
    def optimize(self, func: Callable, x0: np.ndarray) -> OptimizationResult:
        """牛顿法优化"""
        x = x0.copy()
        convergence_history = []
        
        try:
            for iteration in range(self.max_iterations):
                # 计算梯度和Hessian矩阵
                grad = self.numerical_gradient(func, x)
                grad_norm = np.linalg.norm(grad)
                
                if grad_norm < self.tolerance:
                    fval = func(x)
                    self.log_iteration(iteration, fval, x)
                    return OptimizationResult(
                        x=x, fval=fval, success=True, iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                # 计算Hessian矩阵
                hessian = self.numerical_hessian(func, x)
                
                try:
                    # 求解牛顿方向
                    delta_x = np.linalg.solve(hessian, -grad)
                except np.linalg.LinAlgError:
                    # 如果Hessian矩阵奇异，使用梯度下降
                    delta_x = -grad
                
                # 更新参数
                x = x + delta_x
                if self.bounds:
                    x = project_to_bounds(x, self.bounds)
                
                fval = func(x)
                convergence_history.append(fval)
                self.log_iteration(iteration, fval, x)
                
        except Exception as e:
            logger.error(f"牛顿法优化失败: {e}")
            return OptimizationResult(
                x=x, fval=func(x), success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        return OptimizationResult(
            x=x, fval=func(x), success=False, iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )


class ConjugateGradientOptimizer(ClassicalOptimizer):
    """共轭梯度法优化器"""
    
    @timer
    def optimize(self, func: Callable, x0: np.ndarray) -> OptimizationResult:
        """共轭梯度法优化"""
        x = x0.copy()
        convergence_history = []
        
        try:
            # 初始梯度
            grad = self.numerical_gradient(func, x)
            d = -grad  # 初始搜索方向
            grad_norm = np.linalg.norm(grad)
            
            for iteration in range(self.max_iterations):
                if grad_norm < self.tolerance:
                    fval = func(x)
                    self.log_iteration(iteration, fval, x)
                    return OptimizationResult(
                        x=x, fval=fval, success=True, iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                # 线搜索寻找最优步长
                alpha = self._line_search(func, x, d)
                
                # 更新参数
                x_new = x + alpha * d
                if self.bounds:
                    x_new = project_to_bounds(x_new, self.bounds)
                
                # 计算新梯度
                grad_new = self.numerical_gradient(func, x_new)
                grad_norm_new = np.linalg.norm(grad_new)
                
                # 计算共轭系数 (Polak-Ribiere公式)
                beta = (grad_norm_new**2 - np.dot(grad_new, grad)) / (grad_norm**2 + 1e-12)
                beta = max(0, beta)  # 确保非负
                
                # 更新搜索方向
                d = -grad_new + beta * d
                
                # 更新当前点
                x = x_new
                grad = grad_new
                grad_norm = grad_norm_new
                
                fval = func(x)
                convergence_history.append(fval)
                self.log_iteration(iteration, fval, x)
                
        except Exception as e:
            logger.error(f"共轭梯度法优化失败: {e}")
            return OptimizationResult(
                x=x, fval=func(x), success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        return OptimizationResult(
            x=x, fval=func(x), success=False, iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )
    
    def _line_search(self, func: Callable, x: np.ndarray, d: np.ndarray) -> float:
        """简单线搜索"""
        alpha = 1.0
        c = 1e-4
        rho = 0.5
        
        f_x = func(x)
        grad = self.numerical_gradient(func, x)
        
        while True:
            x_new = x + alpha * d
            if self.bounds:
                x_new = project_to_bounds(x_new, self.bounds)
            
            f_new = func(x_new)
            
            # Armijo条件
            if f_new <= f_x + c * alpha * np.dot(grad, d):
                return alpha
            
            alpha *= rho
            if alpha < 1e-10:
                return alpha


class IntelligentOptimizer(Optimizer):
    """智能优化算法基类"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 population_size: int = 50, verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.population_size = population_size
        
    def initialize_population(self) -> np.ndarray:
        """初始化种群"""
        if self.bounds is None:
            return np.random.randn(self.population_size, 1)
        
        population = np.zeros((self.population_size, len(self.bounds)))
        for i, (lower, upper) in enumerate(self.bounds):
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        
        return population


class GeneticAlgorithm(IntelligentOptimizer):
    """遗传算法优化器"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, elite_ratio: float = 0.1,
                 verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, population_size, verbose)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        
    @timer
    def optimize(self, func: Callable, x0: np.ndarray = None) -> OptimizationResult:
        """遗传算法优化"""
        # 初始化种群
        if x0 is not None:
            population = self.initialize_population()
            # 用初始解替换最差个体
            fitness = self._evaluate_population(func, population)
            worst_idx = np.argmin(fitness)
            population[worst_idx] = x0
        else:
            population = self.initialize_population()
        
        convergence_history = []
        best_fitness_history = []
        
        try:
            for iteration in range(self.max_iterations):
                # 评估种群
                fitness = self._evaluate_population(func, population)
                best_idx = np.argmax(fitness)
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx].copy()
                
                convergence_history.append(best_fitness)
                best_fitness_history.append(best_fitness)
                
                # 检查收敛
                if iteration > 0 and abs(best_fitness_history[-2] - best_fitness_history[-1]) < self.tolerance:
                    self.log_iteration(iteration, best_fitness, best_solution)
                    return OptimizationResult(
                        x=best_solution, fval=best_fitness, success=True,
                        iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                # 选择
                selected = self._selection(population, fitness)
                
                # 交叉
                offspring = self._crossover(selected)
                
                # 变异
                mutated = self._mutation(offspring)
                
                # 精英保留
                elite_size = int(self.population_size * self.elite_ratio)
                elite_indices = np.argsort(fitness)[-elite_size:]
                
                # 组合新种群
                new_population = np.zeros_like(population)
                new_population[:elite_size] = population[elite_indices]
                new_population[elite_size:] = mutated[:self.population_size - elite_size]
                
                population = new_population
                
                self.log_iteration(iteration, best_fitness, best_solution)
                
        except Exception as e:
            logger.error(f"遗传算法优化失败: {e}")
            best_idx = np.argmax(self._evaluate_population(func, population))
            return OptimizationResult(
                x=population[best_idx], fval=func(population[best_idx]),
                success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        # 返回最终结果
        fitness = self._evaluate_population(func, population)
        best_idx = np.argmax(fitness)
        return OptimizationResult(
            x=population[best_idx], fval=fitness[best_idx], success=False,
            iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )
    
    def _evaluate_population(self, func: Callable, population: np.ndarray) -> np.ndarray:
        """评估种群适应度"""
        fitness = np.zeros(len(population))
        for i, individual in enumerate(population):
            try:
                fitness[i] = -func(individual)  # 最小化问题转为最大化
            except Exception as e:
                logger.warning(f"适应度计算失败 for individual {i}: {e}")
                fitness[i] = -1e10  # 惩罚无效解
        return fitness
    
    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """选择操作（轮盘赌选择）"""
        # 归一化适应度
        fitness_sum = np.sum(fitness)
        if fitness_sum <= 0:
            probabilities = np.ones(len(fitness)) / len(fitness)
        else:
            probabilities = fitness / fitness_sum
        
        # 轮盘赌选择
        selected = np.zeros_like(population)
        for i in range(len(population)):
            selected_idx = np.random.choice(len(population), p=probabilities)
            selected[i] = population[selected_idx]
        
        return selected
    
    def _crossover(self, population: np.ndarray) -> np.ndarray:
        """交叉操作"""
        offspring = np.zeros_like(population)
        
        for i in range(0, len(population), 2):
            if i + 1 < len(population) and np.random.random() < self.crossover_rate:
                # 单点交叉
                crossover_point = np.random.randint(1, population.shape[1])
                offspring[i, :crossover_point] = population[i, :crossover_point]
                offspring[i, crossover_point:] = population[i + 1, crossover_point:]
                offspring[i + 1, :crossover_point] = population[i + 1, :crossover_point]
                offspring[i + 1, crossover_point:] = population[i, crossover_point:]
            else:
                offspring[i] = population[i]
                if i + 1 < len(population):
                    offspring[i + 1] = population[i + 1]
        
        return offspring
    
    def _mutation(self, population: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = population.copy()
        
        for i in range(len(population)):
            for j in range(population.shape[1]):
                if np.random.random() < self.mutation_rate:
                    if self.bounds:
                        lower, upper = self.bounds[j]
                        range_size = upper - lower
                        mutated[i, j] += np.random.normal(0, range_size * 0.1)
                        mutated[i, j] = np.clip(mutated[i, j], lower, upper)
                    else:
                        mutated[i, j] += np.random.normal(0, 0.1)
        
        return mutated


class ParticleSwarmOptimizer(IntelligentOptimizer):
    """粒子群优化器"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 population_size: int = 50, inertia_weight: float = 0.9,
                 cognitive_coefficient: float = 2.0, social_coefficient: float = 2.0,
                 verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, population_size, verbose)
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        
    @timer
    def optimize(self, func: Callable, x0: np.ndarray = None) -> OptimizationResult:
        """粒子群优化"""
        # 初始化粒子群
        particles = self.initialize_population()
        velocities = np.zeros_like(particles)
        
        # 个体最优和全局最优
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        
        # 全局最优
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_value = personal_best_values[global_best_idx]
        
        convergence_history = []
        
        try:
            for iteration in range(self.max_iterations):
                for i in range(self.population_size):
                    # 更新速度
                    r1, r2 = np.random.random(), np.random.random()
                    
                    velocities[i] = (self.inertia_weight * velocities[i] +
                                   self.cognitive_coefficient * r1 *
                                   (personal_best_positions[i] - particles[i]) +
                                   self.social_coefficient * r2 *
                                   (global_best_position - particles[i]))
                    
                    # 更新位置
                    particles[i] += velocities[i]
                    
                    # 边界处理
                    if self.bounds:
                        particles[i] = project_to_bounds(particles[i], self.bounds)
                        # 速度边界处理
                        for j, (lower, upper) in enumerate(self.bounds):
                            if particles[i, j] <= lower or particles[i, j] >= upper:
                                velocities[i, j] *= -0.5
                    
                    # 评估新位置
                    current_value = func(particles[i])
                    
                    # 更新个体最优
                    if current_value < personal_best_values[i]:
                        personal_best_values[i] = current_value
                        personal_best_positions[i] = particles[i].copy()
                        
                        # 更新全局最优
                        if current_value < global_best_value:
                            global_best_value = current_value
                            global_best_position = particles[i].copy()
                
                convergence_history.append(global_best_value)
                
                # 检查收敛
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < self.tolerance:
                    self.log_iteration(iteration, global_best_value, global_best_position)
                    return OptimizationResult(
                        x=global_best_position, fval=global_best_value, success=True,
                        iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                self.log_iteration(iteration, global_best_value, global_best_position)
                
        except Exception as e:
            logger.error(f"粒子群优化失败: {e}")
            return OptimizationResult(
                x=global_best_position, fval=global_best_value,
                success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        return OptimizationResult(
            x=global_best_position, fval=global_best_value, success=False,
            iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )


class SimulatedAnnealingOptimizer(Optimizer):
    """模拟退火优化器"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 initial_temperature: float = 1000.0, cooling_rate: float = 0.95,
                 min_temperature: float = 1e-8, verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        
    @timer
    def optimize(self, func: Callable, x0: np.ndarray) -> OptimizationResult:
        """模拟退火优化"""
        current_solution = x0.copy()
        current_value = func(current_solution)
        
        best_solution = current_solution.copy()
        best_value = current_value
        
        temperature = self.initial_temperature
        convergence_history = []
        
        try:
            for iteration in range(self.max_iterations):
                # 生成邻域解
                neighbor = self._generate_neighbor(current_solution)
                if self.bounds:
                    neighbor = project_to_bounds(neighbor, self.bounds)
                
                neighbor_value = func(neighbor)
                
                # 接受准则
                if neighbor_value < current_value:
                    # 更好的解，直接接受
                    current_solution = neighbor
                    current_value = neighbor_value
                    
                    if current_value < best_value:
                        best_solution = current_solution.copy()
                        best_value = current_value
                else:
                    # 较差的解，按概率接受
                    delta = neighbor_value - current_value
                    probability = np.exp(-delta / temperature)
                    
                    if np.random.random() < probability:
                        current_solution = neighbor
                        current_value = neighbor_value
                
                convergence_history.append(best_value)
                
                # 降温
                temperature *= self.cooling_rate
                
                # 检查收敛
                if temperature < self.min_temperature:
                    self.log_iteration(iteration, best_value, best_solution)
                    return OptimizationResult(
                        x=best_solution, fval=best_value, success=True,
                        iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代（温度过低）",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                # 检查函数值变化
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < self.tolerance:
                    self.log_iteration(iteration, best_value, best_solution)
                    return OptimizationResult(
                        x=best_solution, fval=best_value, success=True,
                        iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                self.log_iteration(iteration, best_value, best_solution)
                
        except Exception as e:
            logger.error(f"模拟退火优化失败: {e}")
            return OptimizationResult(
                x=best_solution, fval=best_value,
                success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        return OptimizationResult(
            x=best_solution, fval=best_value, success=False,
            iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )
    
    def _generate_neighbor(self, x: np.ndarray) -> np.ndarray:
        """生成邻域解"""
        neighbor = x.copy()
        step_size = 0.1
        
        # 随机选择几个维度进行扰动
        n_dims = len(x)
        n_changes = np.random.randint(1, min(5, n_dims + 1))
        indices = np.random.choice(n_dims, n_changes, replace=False)
        
        for idx in indices:
            neighbor[idx] += np.random.normal(0, step_size)
        
        return neighbor


class ConstraintOptimizer(Optimizer):
    """约束优化基类"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 constraints: List[Callable] = None, verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.constraints = constraints or []
        
    def check_constraints(self, x: np.ndarray) -> bool:
        """检查约束是否满足"""
        for constraint in self.constraints:
            if constraint(x) > 0:  # 假设约束为 g(x) <= 0
                return False
        return True
        
    def penalty_function(self, x: np.ndarray, penalty_weight: float = 1000.0) -> float:
        """计算惩罚函数"""
        penalty = 0.0
        for constraint in self.constraints:
            violation = max(0, constraint(x))
            penalty += penalty_weight * violation**2
        return penalty


class LagrangeMultiplierMethod(ConstraintOptimizer):
    """拉格朗日乘数法"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 constraints: List[Callable] = None,
                 equality_constraints: List[Callable] = None,
                 verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, constraints, verbose)
        self.equality_constraints = equality_constraints or []
        
    @timer
    def optimize(self, func: Callable, x0: np.ndarray) -> OptimizationResult:
        """拉格朗日乘数法优化"""
        x = x0.copy()
        n_vars = len(x)
        n_constraints = len(self.constraints)
        n_eq_constraints = len(self.equality_constraints)
        
        # 初始化拉格朗日乘数
        lambda_ineq = np.ones(n_constraints)
        lambda_eq = np.ones(n_eq_constraints)
        
        convergence_history = []
        
        try:
            for iteration in range(self.max_iterations):
                # 构建拉格朗日函数
                def lagrangian(x_vec):
                    L = func(x_vec)
                    # 不等式约束
                    for i, constraint in enumerate(self.constraints):
                        violation = constraint(x_vec)
                        if violation > 0:
                            L += lambda_ineq[i] * violation
                    # 等式约束
                    for i, constraint in enumerate(self.equality_constraints):
                        L += lambda_eq[i] * constraint(x_vec)
                    return L
                
                # 使用无约束优化器求解
                optimizer = NewtonOptimizer(self.bounds, 50, self.tolerance, self.verbose)
                result = optimizer.optimize(lagrangian, x)
                
                if not result.success:
                    logger.warning(f"第 {iteration} 次迭代内点优化失败")
                    break
                
                x_new = result.x
                
                # 更新拉格朗日乘数
                for i, constraint in enumerate(self.constraints):
                    lambda_ineq[i] = max(0, lambda_ineq[i] + 0.1 * constraint(x_new))
                
                for i, constraint in enumerate(self.equality_constraints):
                    lambda_eq[i] += 0.1 * constraint(x_new)
                
                # 检查收敛
                x_change = np.linalg.norm(x_new - x)
                if x_change < self.tolerance:
                    convergence_history.append(func(x_new))
                    self.log_iteration(iteration, func(x_new), x_new)
                    return OptimizationResult(
                        x=x_new, fval=func(x_new), success=True,
                        iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                x = x_new
                convergence_history.append(func(x))
                self.log_iteration(iteration, func(x), x)
                
        except Exception as e:
            logger.error(f"拉格朗日乘数法优化失败: {e}")
            return OptimizationResult(
                x=x, fval=func(x), success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        return OptimizationResult(
            x=x, fval=func(x), success=False, iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )


class PenaltyFunctionMethod(ConstraintOptimizer):
    """惩罚函数法"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 constraints: List[Callable] = None,
                 penalty_weight: float = 1000.0, penalty_increase: float = 10.0,
                 verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, constraints, verbose)
        self.penalty_weight = penalty_weight
        self.penalty_increase = penalty_increase
        
    @timer
    def optimize(self, func: Callable, x0: np.ndarray) -> OptimizationResult:
        """惩罚函数法优化"""
        x = x0.copy()
        penalty_weight = self.penalty_weight
        convergence_history = []
        
        try:
            for iteration in range(self.max_iterations):
                # 构建惩罚目标函数
                def penalized_func(x_vec):
                    return func(x_vec) + self.penalty_function(x_vec, penalty_weight)
                
                # 使用无约束优化器
                optimizer = GradientDescentOptimizer(
                    self.bounds, 50, self.tolerance, 0.01, self.verbose
                )
                result = optimizer.optimize(penalized_func, x)
                
                if not result.success:
                    logger.warning(f"第 {iteration} 次迭代内点优化失败")
                    break
                
                x_new = result.x
                
                # 检查约束满足情况
                if self.check_constraints(x_new):
                    convergence_history.append(func(x_new))
                    self.log_iteration(iteration, func(x_new), x_new)
                    return OptimizationResult(
                        x=x_new, fval=func(x_new), success=True,
                        iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                # 增加惩罚权重
                penalty_weight *= self.penalty_increase
                x = x_new
                
                convergence_history.append(func(x))
                self.log_iteration(iteration, func(x), x)
                
        except Exception as e:
            logger.error(f"惩罚函数法优化失败: {e}")
            return OptimizationResult(
                x=x, fval=func(x), success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        return OptimizationResult(
            x=x, fval=func(x), success=False, iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )


class BarrierFunctionMethod(ConstraintOptimizer):
    """障碍函数法"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-6,
                 constraints: List[Callable] = None,
                 barrier_weight: float = 1.0, barrier_increase: float = 10.0,
                 verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, constraints, verbose)
        self.barrier_weight = barrier_weight
        self.barrier_increase = barrier_increase
        
    @timer
    def optimize(self, func: Callable, x0: np.ndarray) -> OptimizationResult:
        """障碍函数法优化"""
        x = x0.copy()
        barrier_weight = self.barrier_weight
        convergence_history = []
        
        try:
            for iteration in range(self.max_iterations):
                # 构建障碍目标函数
                def barrier_func(x_vec):
                    barrier_value = 0.0
                    for constraint in self.constraints:
                        violation = constraint(x_vec)
                        if violation >= 0:  # 在边界上
                            return float('inf')
                        barrier_value += barrier_weight / violation
                    return func(x_vec) + barrier_value
                
                # 使用无约束优化器
                optimizer = GradientDescentOptimizer(
                    self.bounds, 50, self.tolerance, 0.01, self.verbose
                )
                result = optimizer.optimize(barrier_func, x)
                
                if not result.success:
                    logger.warning(f"第 {iteration} 次迭代内点优化失败")
                    break
                
                x_new = result.x
                
                # 检查是否仍在可行域内
                if not self.check_constraints(x_new):
                    logger.warning(f"第 {iteration} 次迭代解不在可行域内")
                    break
                
                convergence_history.append(func(x_new))
                
                # 检查收敛
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < self.tolerance:
                    self.log_iteration(iteration, func(x_new), x_new)
                    return OptimizationResult(
                        x=x_new, fval=func(x_new), success=True,
                        iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                # 减少障碍权重
                barrier_weight /= self.barrier_increase
                x = x_new
                
                self.log_iteration(iteration, func(x), x)
                
        except Exception as e:
            logger.error(f"障碍函数法优化失败: {e}")
            return OptimizationResult(
                x=x, fval=func(x), success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        return OptimizationResult(
            x=x, fval=func(x), success=False, iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )


class MultiObjectiveOptimizer(ABC):
    """多目标优化基类"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, population_size: int = 100,
                 verbose: bool = True):
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.verbose = verbose
        
    @abstractmethod
    def optimize(self, objectives: List[Callable], x0: np.ndarray = None) -> MultiObjectiveResult:
        """执行多目标优化"""
        pass
        
    def log_iteration(self, iteration: int, hypervolume: float):
        """记录迭代信息"""
        if self.verbose:
            logger.info(f"迭代 {iteration}: 超体积 = {hypervolume:.6f}")


class NSGA2Optimizer(MultiObjectiveOptimizer):
    """NSGA-II多目标优化器"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, population_size: int = 100,
                 crossover_probability: float = 0.9, mutation_probability: float = 0.1,
                 eta_crossover: float = 20.0, eta_mutation: float = 20.0,
                 verbose: bool = True):
        super().__init__(bounds, max_iterations, population_size, verbose)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation
        
    @timer
    def optimize(self, objectives: List[Callable], x0: np.ndarray = None) -> MultiObjectiveResult:
        """NSGA-II优化"""
        n_objectives = len(objectives)
        n_variables = len(self.bounds) if self.bounds else 1
        
        # 初始化种群
        if x0 is not None:
            population = self._initialize_population()
            # 用初始解替换最差个体
            objectives_values = self._evaluate_objectives(objectives, population)
            crowding_distances = self._calculate_crowding_distance(objectives_values)
            worst_idx = np.argmin(crowding_distances)
            population[worst_idx] = x0
        else:
            population = self._initialize_population()
        
        convergence_history = []
        
        try:
            for iteration in range(self.max_iterations):
                # 评估目标函数
                objectives_values = self._evaluate_objectives(objectives, population)
                
                # 非支配排序
                fronts = self._non_dominated_sort(objectives_values)
                
                # 计算拥挤距离
                crowding_distances = self._calculate_crowding_distance(objectives_values)
                
                # 选择、交叉、变异
                offspring = self._genetic_operations(population, objectives_values, 
                                                   fronts, crowding_distances)
                
                # 合并父代和子代
                combined_population = np.vstack([population, offspring])
                combined_objectives = np.vstack([objectives_values, 
                                               self._evaluate_objectives(objectives, offspring)])
                
                # 环境选择
                population, objectives_values = self._environmental_selection(
                    combined_population, combined_objectives)
                
                # 计算超体积
                hypervolume = self._calculate_hypervolume(objectives_values)
                convergence_history.append(hypervolume)
                
                self.log_iteration(iteration, hypervolume)
                
                # 简单收敛检查
                if iteration > 10:
                    recent_hv = convergence_history[-10:]
                    if np.std(recent_hv) < 1e-6:
                        break
                
        except Exception as e:
            logger.error(f"NSGA-II优化失败: {e}")
            objectives_values = self._evaluate_objectives(objectives, population)
        
        # 提取Pareto前沿
        fronts = self._non_dominated_sort(objectives_values)
        pareto_front_indices = fronts[0]
        
        pareto_front = objectives_values[pareto_front_indices]
        pareto_solutions = population[pareto_front_indices]
        
        return MultiObjectiveResult(
            pareto_front=pareto_front,
            pareto_values=objectives_values[pareto_front_indices],
            pareto_solutions=pareto_solutions,
            convergence_history=convergence_history,
            hypervolume=self._calculate_hypervolume(pareto_front),
            execution_time=0.0
        )
    
    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        if self.bounds is None:
            return np.random.randn(self.population_size, 1)
        
        population = np.zeros((self.population_size, len(self.bounds)))
        for i, (lower, upper) in enumerate(self.bounds):
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        
        return population
    
    def _evaluate_objectives(self, objectives: List[Callable], 
                           population: np.ndarray) -> np.ndarray:
        """评估目标函数"""
        n_objectives = len(objectives)
        objectives_values = np.zeros((len(population), n_objectives))
        
        for i, individual in enumerate(population):
            for j, objective in enumerate(objectives):
                try:
                    objectives_values[i, j] = objective(individual)
                except Exception as e:
                    logger.warning(f"目标函数 {j} 计算失败 for individual {i}: {e}")
                    objectives_values[i, j] = 1e10  # 惩罚无效解
        
        return objectives_values
    
    def _non_dominated_sort(self, objectives_values: np.ndarray) -> List[np.ndarray]:
        """非支配排序"""
        n = len(objectives_values)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives_values[i], objectives_values[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives_values[j], objectives_values[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return [np.array(front) for front in fronts if front]
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断解1是否支配解2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _calculate_crowding_distance(self, objectives_values: np.ndarray) -> np.ndarray:
        """计算拥挤距离"""
        n = len(objectives_values)
        n_objectives = objectives_values.shape[1]
        crowding_distance = np.zeros(n)
        
        for m in range(n_objectives):
            # 按目标函数值排序
            sorted_indices = np.argsort(objectives_values[:, m])
            
            # 边界点设为无穷大
            crowding_distance[sorted_indices[0]] = float('inf')
            crowding_distance[sorted_indices[-1]] = float('inf')
            
            # 计算中间点的拥挤距离
            obj_range = objectives_values[sorted_indices[-1], m] - objectives_values[sorted_indices[0], m]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distance = (objectives_values[sorted_indices[i + 1], m] - 
                              objectives_values[sorted_indices[i - 1], m]) / obj_range
                    crowding_distance[sorted_indices[i]] += distance
        
        return crowding_distance
    
    def _genetic_operations(self, population: np.ndarray, objectives_values: np.ndarray,
                          fronts: List[np.ndarray], crowding_distances: np.ndarray) -> np.ndarray:
        """遗传操作"""
        offspring = np.zeros_like(population)
        
        # 锦标赛选择
        for i in range(self.population_size):
            parent1 = self._tournament_selection(population, objectives_values, 
                                               fronts, crowding_distances)
            parent2 = self._tournament_selection(population, objectives_values, 
                                               fronts, crowding_distances)
            
            # 交叉
            if np.random.random() < self.crossover_probability:
                child1, child2 = self._sbx_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异
            child1 = self._polynomial_mutation(child1)
            child2 = self._polynomial_mutation(child2)
            
            offspring[i] = child1 if i % 2 == 0 else child2
        
        return offspring
    
    def _tournament_selection(self, population: np.ndarray, objectives_values: np.ndarray,
                            fronts: List[np.ndarray], crowding_distances: np.ndarray) -> np.ndarray:
        """锦标赛选择"""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        
        best_idx = tournament_indices[0]
        best_front_rank = self._get_front_rank(best_idx, fronts)
        best_crowding = crowding_distances[best_idx]
        
        for idx in tournament_indices[1:]:
            current_front_rank = self._get_front_rank(idx, fronts)
            current_crowding = crowding_distances[idx]
            
            if (current_front_rank < best_front_rank or 
                (current_front_rank == best_front_rank and current_crowding > best_crowding)):
                best_idx = idx
                best_front_rank = current_front_rank
                best_crowding = current_crowding
        
        return population[best_idx]
    
    def _get_front_rank(self, index: int, fronts: List[np.ndarray]) -> int:
        """获取解所在的前沿等级"""
        for rank, front in enumerate(fronts):
            if index in front:
                return rank
        return len(fronts)  # 如果不在任何前沿中
    
    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """模拟二进制交叉"""
        n_vars = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(n_vars):
            if np.random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    
                    if self.bounds:
                        lower, upper = self.bounds[i]
                        yl, yu = lower, upper
                    else:
                        yl, yu = -np.inf, np.inf
                    
                    beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(self.eta_crossover + 1.0))
                    
                    if np.random.random() <= (1.0 / alpha):
                        betaq = (np.random.random() * alpha) ** (1.0 / (self.eta_crossover + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - np.random.random() * alpha)) ** (1.0 / (self.eta_crossover + 1.0))
                    
                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(self.eta_crossover + 1.0))
                    
                    if np.random.random() <= (1.0 / alpha):
                        betaq = (np.random.random() * alpha) ** (1.0 / (self.eta_crossover + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - np.random.random() * alpha)) ** (1.0 / (self.eta_crossover + 1.0))
                    
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                    
                    c1 = np.clip(c1, yl, yu)
                    c2 = np.clip(c2, yl, yu)
                    
                    if np.random.random() <= 0.5:
                        child1[i], child2[i] = c2, c1
                    else:
                        child1[i], child2[i] = c1, c2
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """多项式变异"""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() <= self.mutation_probability:
                if self.bounds:
                    lower, upper = self.bounds[i]
                    y = individual[i]
                    yl, yu = lower, upper
                else:
                    y = individual[i]
                    yl, yu = -np.inf, np.inf
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rnd = np.random.random()
                mut_pow = 1.0 / (self.eta_mutation + 1.0)
                
                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.eta_mutation + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.eta_mutation + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (yu - yl)
                y = np.clip(y, yl, yu)
                mutated[i] = y
        
        return mutated
    
    def _environmental_selection(self, population: np.ndarray, 
                               objectives_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """环境选择"""
        fronts = self._non_dominated_sort(objectives_values)
        selected_population = []
        selected_objectives = []
        
        for front in fronts:
            if len(selected_population) + len(front) <= self.population_size:
                # 整个前沿都加入
                selected_population.extend(population[front])
                selected_objectives.extend(objectives_values[front])
            else:
                # 需要从当前前沿中选择部分个体
                remaining = self.population_size - len(selected_population)
                crowding_distances = self._calculate_crowding_distance(objectives_values[front])
                
                # 按拥挤距离排序
                sorted_indices = np.argsort(crowding_distances[front])[::-1]
                selected_indices = front[sorted_indices[:remaining]]
                
                selected_population.extend(population[selected_indices])
                selected_objectives.extend(objectives_values[selected_indices])
                break
        
        return np.array(selected_population), np.array(selected_objectives)
    
    def _calculate_hypervolume(self, objectives_values: np.ndarray) -> float:
        """计算超体积（简化版本）"""
        if len(objectives_values) == 0:
            return 0.0
        
        # 简化的超体积计算（假设最小化问题）
        ref_point = np.max(objectives_values, axis=0) + 1.0
        hypervolume = 0.0
        
        for obj in objectives_values:
            volume = np.prod(ref_point - obj)
            hypervolume += volume
        
        return hypervolume


class SPEA2Optimizer(MultiObjectiveOptimizer):
    """SPEA2多目标优化器"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 1000, population_size: int = 100,
                 archive_size: int = 100, crossover_probability: float = 0.9,
                 mutation_probability: float = 0.1, verbose: bool = True):
        super().__init__(bounds, max_iterations, population_size, verbose)
        self.archive_size = archive_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        
    @timer
    def optimize(self, objectives: List[Callable], x0: np.ndarray = None) -> MultiObjectiveResult:
        """SPEA2优化"""
        n_objectives = len(objectives)
        
        # 初始化种群和档案
        population = self._initialize_population()
        archive = self._initialize_population()
        
        convergence_history = []
        
        try:
            for iteration in range(self.max_iterations):
                # 评估目标函数
                population_objectives = self._evaluate_objectives(objectives, population)
                archive_objectives = self._evaluate_objectives(objectives, archive)
                
                # 合并种群和档案
                combined_population = np.vstack([population, archive])
                combined_objectives = np.vstack([population_objectives, archive_objectives])
                
                # 非支配排序
                fronts = self._non_dominated_sort(combined_objectives)
                
                # 计算适应度
                fitness = self._calculate_fitness(combined_objectives, fronts)
                
                # 环境选择
                archive = self._environmental_selection(combined_population, combined_objectives, fitness)
                
                # 生成新种群
                population = self._genetic_operations(archive, fitness)
                
                # 计算超体积
                archive_objectives = self._evaluate_objectives(objectives, archive)
                hypervolume = self._calculate_hypervolume(archive_objectives)
                convergence_history.append(hypervolume)
                
                self.log_iteration(iteration, hypervolume)
                
        except Exception as e:
            logger.error(f"SPEA2优化失败: {e}")
            archive_objectives = self._evaluate_objectives(objectives, archive)
        
        # 提取Pareto前沿
        fronts = self._non_dominated_sort(archive_objectives)
        pareto_front_indices = fronts[0]
        
        pareto_front = archive_objectives[pareto_front_indices]
        pareto_solutions = archive[pareto_front_indices]
        
        return MultiObjectiveResult(
            pareto_front=pareto_front,
            pareto_values=archive_objectives[pareto_front_indices],
            pareto_solutions=pareto_solutions,
            convergence_history=convergence_history,
            hypervolume=self._calculate_hypervolume(pareto_front),
            execution_time=0.0
        )
    
    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        if self.bounds is None:
            return np.random.randn(self.population_size, 1)
        
        population = np.zeros((self.population_size, len(self.bounds)))
        for i, (lower, upper) in enumerate(self.bounds):
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        
        return population
    
    def _evaluate_objectives(self, objectives: List[Callable], 
                           population: np.ndarray) -> np.ndarray:
        """评估目标函数"""
        n_objectives = len(objectives)
        objectives_values = np.zeros((len(population), n_objectives))
        
        for i, individual in enumerate(population):
            for j, objective in enumerate(objectives):
                try:
                    objectives_values[i, j] = objective(individual)
                except Exception as e:
                    logger.warning(f"目标函数 {j} 计算失败 for individual {i}: {e}")
                    objectives_values[i, j] = 1e10
        
        return objectives_values
    
    def _non_dominated_sort(self, objectives_values: np.ndarray) -> List[np.ndarray]:
        """非支配排序"""
        n = len(objectives_values)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives_values[i], objectives_values[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives_values[j], objectives_values[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return [np.array(front) for front in fronts if front]
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断解1是否支配解2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _calculate_fitness(self, objectives_values: np.ndarray, 
                         fronts: List[np.ndarray]) -> np.ndarray:
        """计算适应度"""
        n = len(objectives_values)
        fitness = np.zeros(n)
        
        # 计算强度
        strength = np.zeros(n)
        for front in fronts:
            for i in range(n):
                if i in front:
                    # 计算被i支配的解的数量
                    strength[i] = sum(1 for j in range(n) 
                                    if j not in front and self._dominates(objectives_values[i], objectives_values[j]))
        
        # 计算原始适应度
        raw_fitness = np.zeros(n)
        for i in range(n):
            dominated_count = 0
            for j in range(n):
                if i != j and self._dominates(objectives_values[j], objectives_values[i]):
                    dominated_count += 1
            raw_fitness[i] = dominated_count
        
        # 计算密度估计
        k = int(np.sqrt(n))  # k近邻参数
        density = np.zeros(n)
        
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(objectives_values[i] - objectives_values[j])
                    distances.append(dist)
            
            distances.sort()
            if len(distances) > k:
                density[i] = 1.0 / (distances[k-1] + 2.0)
            else:
                density[i] = 0.0
        
        # 适应度 = 原始适应度 + 密度
        fitness = raw_fitness + density
        
        return fitness
    
    def _environmental_selection(self, population: np.ndarray, 
                               objectives_values: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """环境选择"""
        # 选择非支配解
        non_dominated_indices = np.where(fitness < 1.0)[0]
        
        if len(non_dominated_indices) <= self.archive_size:
            # 如果非支配解数量少于等于档案大小，全部保留
            archive = population[non_dominated_indices]
        else:
            # 如果非支配解数量超过档案大小，使用截断策略
            archive = population[non_dominated_indices]
            # 简化的截断：随机选择
            selected_indices = np.random.choice(len(archive), self.archive_size, replace=False)
            archive = archive[selected_indices]
        
        return archive
    
    def _genetic_operations(self, archive: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """遗传操作"""
        population = np.zeros((self.population_size, archive.shape[1]))
        
        for i in range(self.population_size):
            # 随机选择父代
            parent1 = archive[np.random.randint(len(archive))]
            parent2 = archive[np.random.randint(len(archive))]
            
            # 交叉
            if np.random.random() < self.crossover_probability:
                child1, child2 = self._uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异
            child1 = self._gaussian_mutation(child1)
            child2 = self._gaussian_mutation(child2)
            
            population[i] = child1 if i % 2 == 0 else child2
        
        return population
    
    def _uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """均匀交叉"""
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        
        return child1, child2
    
    def _gaussian_mutation(self, individual: np.ndarray) -> np.ndarray:
        """高斯变异"""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_probability:
                if self.bounds:
                    lower, upper = self.bounds[i]
                    range_size = upper - lower
                    mutated[i] += np.random.normal(0, range_size * 0.1)
                    mutated[i] = np.clip(mutated[i], lower, upper)
                else:
                    mutated[i] += np.random.normal(0, 0.1)
        
        return mutated
    
    def _calculate_hypervolume(self, objectives_values: np.ndarray) -> float:
        """计算超体积"""
        if len(objectives_values) == 0:
            return 0.0
        
        ref_point = np.max(objectives_values, axis=0) + 1.0
        hypervolume = 0.0
        
        for obj in objectives_values:
            volume = np.prod(ref_point - obj)
            hypervolume += volume
        
        return hypervolume


class ParetoFrontAnalyzer:
    """Pareto前沿分析工具"""
    
    def __init__(self):
        pass
    
    def extract_pareto_front(self, objectives_values: np.ndarray) -> np.ndarray:
        """提取Pareto前沿"""
        n = len(objectives_values)
        is_pareto = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if np.all(objectives_values[j] <= objectives_values[i]) and np.any(objectives_values[j] < objectives_values[i]):
                        is_pareto[i] = False
                        break
        
        return objectives_values[is_pareto]
    
    def calculate_hypervolume(self, objectives_values: np.ndarray, 
                            reference_point: np.ndarray) -> float:
        """计算超体积"""
        if len(objectives_values) == 0:
            return 0.0
        
        hypervolume = 0.0
        for obj in objectives_values:
            volume = np.prod(reference_point - obj)
            hypervolume += volume
        
        return hypervolume
    
    def calculate_spread(self, objectives_values: np.ndarray) -> float:
        """计算分布性指标"""
        if len(objectives_values) < 2:
            return 0.0
        
        # 计算相邻解之间的距离
        distances = []
        for i in range(len(objectives_values)):
            min_dist = float('inf')
            for j in range(len(objectives_values)):
                if i != j:
                    dist = np.linalg.norm(objectives_values[i] - objectives_values[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        return std_dist / mean_dist if mean_dist > 0 else 0.0
    
    def calculate_convergence_metric(self, pareto_front: np.ndarray, 
                                  ideal_point: np.ndarray) -> float:
        """计算收敛性指标"""
        if len(pareto_front) == 0:
            return float('inf')
        
        distances = [np.linalg.norm(obj - ideal_point) for obj in pareto_front]
        return np.mean(distances)


class BayesianOptimizer(Optimizer):
    """贝叶斯优化基类"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 100, tolerance: float = 1e-6,
                 verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.X_observed = None
        self.y_observed = None
        
    @abstractmethod
    def _acquisition_function(self, x: np.ndarray) -> float:
        """获取函数"""
        pass
        
    @abstractmethod
    def _update_model(self):
        """更新代理模型"""
        pass


class GaussianProcessOptimizer(BayesianOptimizer):
    """高斯过程贝叶斯优化器"""
    
    def __init__(self, bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 100, tolerance: float = 1e-6,
                 length_scale: float = 1.0, noise_level: float = 1e-6,
                 acquisition_function: str = "ei", verbose: bool = True):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.acquisition_function = acquisition_function
        
    @timer
    def optimize(self, func: Callable, x0: np.ndarray = None) -> OptimizationResult:
        """高斯过程贝叶斯优化"""
        # 初始化观测数据
        if x0 is not None:
            self.X_observed = x0.reshape(1, -1)
            self.y_observed = np.array([func(x0)])
        else:
            # 随机初始化几个点
            n_init = min(5, self.max_iterations // 4)
            self.X_observed = self._initialize_points(n_init)
            self.y_observed = np.array([func(x) for x in self.X_observed])
        
        convergence_history = []
        best_y = np.min(self.y_observed)
        best_x = self.X_observed[np.argmin(self.y_observed)]
        
        try:
            for iteration in range(self.max_iterations):
                # 更新高斯过程模型
                self._update_model()
                
                # 寻找下一个评估点
                next_x = self._optimize_acquisition()
                
                # 评估目标函数
                next_y = func(next_x)
                
                # 更新观测数据
                self.X_observed = np.vstack([self.X_observed, next_x])
                self.y_observed = np.append(self.y_observed, next_y)
                
                # 更新最佳解
                if next_y < best_y:
                    best_y = next_y
                    best_x = next_x.copy()
                
                convergence_history.append(best_y)
                
                # 检查收敛
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < self.tolerance:
                    self.log_iteration(iteration, best_y, best_x)
                    return OptimizationResult(
                        x=best_x, fval=best_y, success=True,
                        iterations=iteration,
                        message=f"收敛于第 {iteration} 次迭代",
                        convergence_history=convergence_history,
                        execution_time=0.0
                    )
                
                self.log_iteration(iteration, best_y, best_x)
                
        except Exception as e:
            logger.error(f"高斯过程优化失败: {e}")
            return OptimizationResult(
                x=best_x, fval=best_y,
                success=False, iterations=iteration,
                message=f"优化失败: {str(e)}",
                convergence_history=convergence_history
            )
        
        return OptimizationResult(
            x=best_x, fval=best_y, success=False,
            iterations=self.max_iterations,
            message="达到最大迭代次数",
            convergence_history=convergence_history
        )
    
    def _initialize_points(self, n_points: int) -> np.ndarray:
        """初始化评估点"""
        if self.bounds is None:
            return np.random.randn(n_points, 1)
        
        points = np.zeros((n_points, len(self.bounds)))
        for i, (lower, upper) in enumerate(self.bounds):
            points[:, i] = np.random.uniform(lower, upper, n_points)
        
        return points
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """径向基函数核"""
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        
        sq_dist = np.sum(x1**2, axis=-1, keepdims=True) + np.sum(x2**2, axis=-1, keepdims=True).T
        sq_dist -= 2 * np.dot(x1, x2.T)
        
        return np.exp(-0.5 * sq_dist / self.length_scale**2)
    
    def _update_model(self):
        """更新高斯过程模型"""
        # 计算核矩阵
        self.K = self._rbf_kernel(self.X_observed, self.X_observed)
        self.K += self.noise_level**2 * np.eye(len(self.X_observed))
        
        # 计算协方差矩阵的逆
        try:
            self.L = np.linalg.cholesky(self.K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_observed))
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            self.K += 1e-8 * np.eye(len(self.X_observed))
            self.L = np.linalg.cholesky(self.K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_observed))
    
    def _predict(self, x: np.ndarray) -> Tuple[float, float]:
        """预测均值和方差"""
        k_x_x = self._rbf_kernel(self.X_observed, x.reshape(1, -1))
        k_x_x = k_x_x.flatten()
        
        # 预测均值
        mu = np.dot(k_x_x, self.alpha)
        
        # 预测方差
        try:
            v = np.linalg.solve(self.L, k_x_x)
            var = self._rbf_kernel(x.reshape(1, -1), x.reshape(1, -1))[0, 0] - np.dot(v, v)
            var = max(var, 1e-8)  # 确保方差为正
        except np.linalg.LinAlgError:
            var = 1.0
        
        return mu, var
    
    def _acquisition_function(self, x: np.ndarray) -> float:
        """获取函数"""
        mu, var = self._predict(x)
        
        if self.acquisition_function == "ei":
            # 期望改进
            f_best = np.min(self.y_observed)
            z = (f_best - mu) / np.sqrt(var)
            ei = (f_best - mu) * self._norm_cdf(z) + np.sqrt(var) * self._norm_pdf(z)
            return -ei  # 最小化问题
        elif self.acquisition_function == "ucb":
            # 上置信界
            beta = 2.0
            return -(mu + np.sqrt(beta * var))
        elif self.acquisition_function == "pi":
            # 改进概率
            f_best = np.min(self.y_observed)
            z = (f_best - mu) / np.sqrt(var)
            pi = self._norm_cdf(z)
            return -pi
        else:
            raise ValueError(f"未知的获取函数: {self.acquisition_function}")
    
    def _norm_cdf(self, x: float) -> float:
        """标准正态分布累积分布函数"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _norm_pdf(self, x: float) -> float:
        """标准正态分布概率密度函数"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _optimize_acquisition(self) -> np.ndarray:
        """优化获取函数"""
        # 随机采样候选点
        n_candidates = 1000
        if self.bounds:
            candidates = np.random.uniform(
                [lower for lower, _ in self.bounds],
                [upper for _, upper in self.bounds],
                (n_candidates, len(self.bounds))
            )
        else:
            candidates = np.random.randn(n_candidates, self.X_observed.shape[1])
        
        # 评估获取函数
        acquisition_values = np.array([self._acquisition_function(x) for x in candidates])
        
        # 选择最佳候选点
        best_idx = np.argmin(acquisition_values)
        return candidates[best_idx]


class ParallelOptimizer:
    """并行优化器"""
    
    def __init__(self, n_workers: int = 4, backend: str = "thread"):
        """
        初始化并行优化器
        
        Args:
            n_workers: 工作进程/线程数量
            backend: 后端类型 ("thread" 或 "process")
        """
        self.n_workers = n_workers
        self.backend = backend
        
    def parallel_optimize(self, optimizer_class: type, 
                         objective_funcs: List[Callable],
                         x0_list: List[np.ndarray],
                         **optimizer_kwargs) -> List[OptimizationResult]:
        """并行优化多个目标函数"""
        results = []
        
        if self.backend == "thread":
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for obj_func, x0 in zip(objective_funcs, x0_list):
                    optimizer = optimizer_class(**optimizer_kwargs)
                    future = executor.submit(optimizer.optimize, obj_func, x0)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"并行优化任务失败: {e}")
                        results.append(None)
        
        elif self.backend == "process":
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for obj_func, x0 in zip(objective_funcs, x0_list):
                    future = executor.submit(self._optimize_worker, 
                                           optimizer_class, obj_func, x0, optimizer_kwargs)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"并行优化任务失败: {e}")
                        results.append(None)
        
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")
        
        return results
    
    def _optimize_worker(self, optimizer_class: type, 
                        objective_func: Callable, x0: np.ndarray,
                        optimizer_kwargs: dict) -> OptimizationResult:
        """工作进程函数"""
        optimizer = optimizer_class(**optimizer_kwargs)
        return optimizer.optimize(objective_func, x0)
    
    async def async_parallel_optimize(self, optimizer_class: type,
                                    objective_funcs: List[Callable],
                                    x0_list: List[np.ndarray],
                                    **optimizer_kwargs) -> List[OptimizationResult]:
        """异步并行优化"""
        loop = asyncio.get_event_loop()
        
        tasks = []
        for obj_func, x0 in zip(objective_funcs, x0_list):
            optimizer = optimizer_class(**optimizer_kwargs)
            task = loop.run_in_executor(None, optimizer.optimize, obj_func, x0)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"异步优化任务失败: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results


class DistributedOptimizer:
    """分布式优化器"""
    
    def __init__(self, node_addresses: List[str]):
        """
        初始化分布式优化器
        
        Args:
            node_addresses: 计算节点地址列表
        """
        self.node_addresses = node_addresses
        self.node_loads = {addr: 0 for addr in node_addresses}
        
    def distributed_optimize(self, optimizer_class: type,
                           objective_funcs: List[Callable],
                           x0_list: List[np.ndarray],
                           **optimizer_kwargs) -> List[OptimizationResult]:
        """分布式优化"""
        results = {}
        
        # 简单的负载均衡分配
        assignments = self._assign_tasks(len(objective_funcs))
        
        for node_addr, task_indices in assignments.items():
            node_results = self._optimize_on_node(
                optimizer_class, objective_funcs, x0_list, task_indices, 
                optimizer_kwargs, node_addr
            )
            results.update(node_results)
        
        # 按原始顺序返回结果
        ordered_results = [results.get(i) for i in range(len(objective_funcs))]
        return ordered_results
    
    def _assign_tasks(self, n_tasks: int) -> Dict[str, List[int]]:
        """分配任务到节点"""
        assignments = {addr: [] for addr in self.node_addresses}
        
        for i in range(n_tasks):
            # 选择负载最轻的节点
            lightest_node = min(self.node_addresses, key=lambda x: self.node_loads[x])
            assignments[lightest_node].append(i)
            self.node_loads[lightest_node] += 1
        
        return assignments
    
    def _optimize_on_node(self, optimizer_class: type,
                         objective_funcs: List[Callable],
                         x0_list: List[np.ndarray],
                         task_indices: List[int],
                         optimizer_kwargs: dict,
                         node_addr: str) -> Dict[int, OptimizationResult]:
        """在指定节点上执行优化"""
        results = {}
        
        for idx in task_indices:
            try:
                optimizer = optimizer_class(**optimizer_kwargs)
                result = optimizer.optimize(objective_funcs[idx], x0_list[idx])
                results[idx] = result
            except Exception as e:
                logger.error(f"节点 {node_addr} 优化任务 {idx} 失败: {e}")
                results[idx] = None
        
        return results


class OptimizationTools:
    """优化工具集合类"""
    
    def __init__(self):
        self.optimizers = {
            'gradient_descent': GradientDescentOptimizer,
            'newton': NewtonOptimizer,
            'conjugate_gradient': ConjugateGradientOptimizer,
            'genetic_algorithm': GeneticAlgorithm,
            'particle_swarm': ParticleSwarmOptimizer,
            'simulated_annealing': SimulatedAnnealingOptimizer,
            'lagrange_multiplier': LagrangeMultiplierMethod,
            'penalty_function': PenaltyFunctionMethod,
            'barrier_function': BarrierFunctionMethod,
            'nsga2': NSGA2Optimizer,
            'spea2': SPEA2Optimizer,
            'gaussian_process': GaussianProcessOptimizer,
        }
        
        self.parallel_optimizer = ParallelOptimizer()
        self.distributed_optimizer = None
        self.pareto_analyzer = ParetoFrontAnalyzer()
    
    def get_optimizer(self, name: str, **kwargs) -> Optimizer:
        """获取优化器实例"""
        if name not in self.optimizers:
            raise ValueError(f"未知的优化器: {name}")
        
        optimizer_class = self.optimizers[name]
        return optimizer_class(**kwargs)
    
    def single_objective_optimize(self, objective_func: Callable,
                                optimizer_name: str,
                                x0: np.ndarray,
                                **optimizer_kwargs) -> OptimizationResult:
        """单目标优化"""
        optimizer = self.get_optimizer(optimizer_name, **optimizer_kwargs)
        return optimizer.optimize(objective_func, x0)
    
    def multi_objective_optimize(self, objective_funcs: List[Callable],
                               optimizer_name: str,
                               x0: np.ndarray = None,
                               **optimizer_kwargs) -> MultiObjectiveResult:
        """多目标优化"""
        if optimizer_name not in ['nsga2', 'spea2']:
            raise ValueError(f"多目标优化不支持优化器: {optimizer_name}")
        
        optimizer = self.get_optimizer(optimizer_name, **optimizer_kwargs)
        return optimizer.optimize(objective_funcs, x0)
    
    def constrained_optimize(self, objective_func: Callable,
                           optimizer_name: str,
                           x0: np.ndarray,
                           constraints: List[Callable],
                           **optimizer_kwargs) -> OptimizationResult:
        """约束优化"""
        if optimizer_name not in ['lagrange_multiplier', 'penalty_function', 'barrier_function']:
            raise ValueError(f"约束优化不支持优化器: {optimizer_name}")
        
        optimizer = self.get_optimizer(optimizer_name, constraints=constraints, **optimizer_kwargs)
        return optimizer.optimize(objective_func, x0)
    
    def parallel_optimize(self, objective_funcs: List[Callable],
                         optimizer_name: str,
                         x0_list: List[np.ndarray],
                         n_workers: int = 4,
                         backend: str = "thread",
                         **optimizer_kwargs) -> List[OptimizationResult]:
        """并行优化"""
        return self.parallel_optimizer.parallel_optimize(
            self.optimizers[optimizer_name], objective_funcs, x0_list,
            n_workers=n_workers, backend=backend, **optimizer_kwargs
        )
    
    async def async_parallel_optimize(self, objective_funcs: List[Callable],
                                    optimizer_name: str,
                                    x0_list: List[np.ndarray],
                                    n_workers: int = 4,
                                    **optimizer_kwargs) -> List[OptimizationResult]:
        """异步并行优化"""
        return await self.parallel_optimizer.async_parallel_optimize(
            self.optimizers[optimizer_name], objective_funcs, x0_list,
            n_workers=n_workers, **optimizer_kwargs
        )
    
    def set_distributed_nodes(self, node_addresses: List[str]):
        """设置分布式计算节点"""
        self.distributed_optimizer = DistributedOptimizer(node_addresses)
    
    def distributed_optimize(self, objective_funcs: List[Callable],
                           optimizer_name: str,
                           x0_list: List[np.ndarray],
                           **optimizer_kwargs) -> List[OptimizationResult]:
        """分布式优化"""
        if self.distributed_optimizer is None:
            raise ValueError("请先设置分布式计算节点")
        
        return self.distributed_optimizer.distributed_optimize(
            self.optimizers[optimizer_name], objective_funcs, x0_list,
            **optimizer_kwargs
        )
    
    def analyze_pareto_front(self, objectives_values: np.ndarray,
                           reference_point: np.ndarray = None) -> Dict[str, Any]:
        """分析Pareto前沿"""
        pareto_front = self.pareto_analyzer.extract_pareto_front(objectives_values)
        
        analysis = {
            'pareto_front': pareto_front,
            'n_pareto_solutions': len(pareto_front),
        }
        
        if reference_point is not None:
            hypervolume = self.pareto_analyzer.calculate_hypervolume(pareto_front, reference_point)
            analysis['hypervolume'] = hypervolume
        
        spread = self.pareto_analyzer.calculate_spread(pareto_front)
        analysis['spread'] = spread
        
        return analysis
    
    def compare_optimizers(self, objective_func: Callable,
                         optimizer_names: List[str],
                         x0: np.ndarray,
                         n_runs: int = 5,
                         **common_kwargs) -> Dict[str, Dict[str, float]]:
        """比较不同优化器的性能"""
        results = {}
        
        for optimizer_name in optimizer_names:
            run_results = []
            
            for run in range(n_runs):
                try:
                    result = self.single_objective_optimize(
                        objective_func, optimizer_name, x0, **common_kwargs
                    )
                    run_results.append({
                        'fval': result.fval,
                        'iterations': result.iterations,
                        'success': result.success,
                        'execution_time': result.execution_time
                    })
                except Exception as e:
                    logger.error(f"优化器 {optimizer_name} 第 {run} 次运行失败: {e}")
                    run_results.append({
                        'fval': float('inf'),
                        'iterations': 0,
                        'success': False,
                        'execution_time': 0.0
                    })
            
            # 统计分析
            fvals = [r['fval'] for r in run_results]
            iterations = [r['iterations'] for r in run_results]
            success_rate = sum(1 for r in run_results if r['success']) / n_runs
            exec_times = [r['execution_time'] for r in run_results]
            
            results[optimizer_name] = {
                'mean_fval': np.mean(fvals),
                'std_fval': np.std(fvals),
                'min_fval': np.min(fvals),
                'mean_iterations': np.mean(iterations),
                'success_rate': success_rate,
                'mean_execution_time': np.mean(exec_times),
                'all_results': run_results
            }
        
        return results
    
    def save_results(self, results: Union[OptimizationResult, MultiObjectiveResult],
                    filename: str):
        """保存优化结果"""
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(results, OptimizationResult):
            data = {
                'type': 'single_objective',
                'x': results.x.tolist(),
                'fval': results.fval,
                'success': results.success,
                'iterations': results.iterations,
                'message': results.message,
                'convergence_history': results.convergence_history,
                'execution_time': results.execution_time
            }
        elif isinstance(results, MultiObjectiveResult):
            data = {
                'type': 'multi_objective',
                'pareto_front': results.pareto_front.tolist(),
                'pareto_values': results.pareto_values.tolist(),
                'pareto_solutions': results.pareto_solutions.tolist(),
                'convergence_history': results.convergence_history,
                'hypervolume': results.hypervolume,
                'execution_time': results.execution_time
            }
        else:
            raise ValueError(f"不支持的结果类型: {type(results)}")
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"优化结果已保存到: {filepath}")
    
    def load_results(self, filename: str) -> Union[OptimizationResult, MultiObjectiveResult]:
        """加载优化结果"""
        filepath = Path(filename)
        
        if not filepath.exists():
            raise FileNotFoundError(f"结果文件不存在: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data['type'] == 'single_objective':
            return OptimizationResult(
                x=np.array(data['x']),
                fval=data['fval'],
                success=data['success'],
                iterations=data['iterations'],
                message=data['message'],
                convergence_history=data.get('convergence_history'),
                execution_time=data.get('execution_time', 0.0)
            )
        elif data['type'] == 'multi_objective':
            return MultiObjectiveResult(
                pareto_front=np.array(data['pareto_front']),
                pareto_values=np.array(data['pareto_values']),
                pareto_solutions=np.array(data['pareto_solutions']),
                convergence_history=data.get('convergence_history'),
                hypervolume=data.get('hypervolume', 0.0),
                execution_time=data.get('execution_time', 0.0)
            )
        else:
            raise ValueError(f"未知的结果类型: {data['type']}")


# 使用示例
def example_usage():
    """使用示例"""
    
    # 创建优化工具实例
    opt_tools = OptimizationTools()
    
    # 定义目标函数
    def rosenbrock(x):
        """Rosenbrock函数"""
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
    
    def sphere(x):
        """球面函数"""
        return sum(xi**2 for xi in x)
    
    # 单目标优化示例
    print("=== 单目标优化示例 ===")
    x0 = np.array([0.0, 0.0])
    
    # 梯度下降
    result_gd = opt_tools.single_objective_optimize(
        rosenbrock, 'gradient_descent', x0,
        bounds=[(-2, 2), (-2, 2)],
        max_iterations=100
    )
    print(f"梯度下降结果: f={result_gd.fval:.6f}, x={result_gd.x}")
    
    # 遗传算法
    result_ga = opt_tools.single_objective_optimize(
        sphere, 'genetic_algorithm', x0,
        bounds=[(-5, 5), (-5, 5)],
        max_iterations=50
    )
    print(f"遗传算法结果: f={result_ga.fval:.6f}, x={result_ga.x}")
    
    # 多目标优化示例
    print("\n=== 多目标优化示例 ===")
    def obj1(x):
        return (x[0] - 1)**2 + (x[1] - 1)**2
    
    def obj2(x):
        return (x[0] + 1)**2 + (x[1] + 1)**2
    
    result_mo = opt_tools.multi_objective_optimize(
        [obj1, obj2], 'nsga2',
        bounds=[(-2, 2), (-2, 2)],
        max_iterations=50
    )
    print(f"NSGA-II Pareto前沿解数量: {len(result_mo.pareto_front)}")
    print(f"超体积: {result_mo.hypervolume:.6f}")
    
    # 约束优化示例
    print("\n=== 约束优化示例 ===")
    def constraint1(x):
        return x[0] + x[1] - 1  # x[0] + x[1] <= 1
    
    result_constrained = opt_tools.constrained_optimize(
        sphere, 'penalty_function', x0,
        constraints=[constraint1],
        bounds=[(-2, 2), (-2, 2)]
    )
    print(f"约束优化结果: f={result_constrained.fval:.6f}, x={result_constrained.x}")
    
    # 并行优化示例
    print("\n=== 并行优化示例 ===")
    objectives = [sphere, rosenbrock]
    x0s = [np.array([1.0, 1.0]), np.array([-1.0, -1.0])]
    
    parallel_results = opt_tools.parallel_optimize(
        objectives, 'gradient_descent', x0s,
        bounds=[(-2, 2), (-2, 2)],
        n_workers=2
    )
    
    for i, result in enumerate(parallel_results):
        if result:
            print(f"并行任务 {i}: f={result.fval:.6f}")
    
    # 贝叶斯优化示例
    print("\n=== 贝叶斯优化示例 ===")
    result_bayes = opt_tools.single_objective_optimize(
        sphere, 'gaussian_process', x0,
        bounds=[(-3, 3), (-3, 3)],
        max_iterations=20
    )
    print(f"贝叶斯优化结果: f={result_bayes.fval:.6f}, x={result_bayes.x}")
    
    # 优化器比较示例
    print("\n=== 优化器比较示例 ===")
    optimizers_to_compare = ['gradient_descent', 'newton', 'genetic_algorithm']
    comparison = opt_tools.compare_optimizers(
        sphere, optimizers_to_compare, x0,
        bounds=[(-2, 2), (-2, 2)],
        n_runs=3
    )
    
    for opt_name, stats in comparison.items():
        print(f"{opt_name}: 平均目标值={stats['mean_fval']:.6f}, "
              f"成功率={stats['success_rate']:.2%}, "
              f"平均执行时间={stats['mean_execution_time']:.4f}s")
    
    # 保存结果示例
    opt_tools.save_results(result_gd, "optimization_results/gradient_descent.json")
    opt_tools.save_results(result_mo, "optimization_results/nsga2_results.json")
    
    print("\n优化完成！")


class AdvancedOptimizers:
    """高级优化算法集合"""
    
    @staticmethod
    def differential_evolution(objective_func: Callable,
                             bounds: List[Tuple[float, float]],
                             population_size: int = 50,
                             max_iterations: int = 1000,
                             mutation_factor: float = 0.8,
                             crossover_probability: float = 0.7,
                             strategy: str = "rand/1/bin") -> OptimizationResult:
        """
        差分进化算法
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            population_size: 种群大小
            max_iterations: 最大迭代次数
            mutation_factor: 变异因子
            crossover_probability: 交叉概率
            strategy: 变异策略
        """
        n_vars = len(bounds)
        
        # 初始化种群
        population = np.zeros((population_size, n_vars))
        for i, (lower, upper) in enumerate(bounds):
            population[:, i] = np.random.uniform(lower, upper, population_size)
        
        # 评估初始种群
        fitness = np.array([objective_func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_history = []
        
        try:
            for iteration in range(max_iterations):
                new_population = np.zeros_like(population)
                
                for i in range(population_size):
                    # 选择变异策略
                    if strategy == "rand/1/bin":
                        # 随机选择三个不同的个体
                        indices = np.random.choice(population_size, 3, replace=False)
                        while i in indices:
                            indices = np.random.choice(population_size, 3, replace=False)
                        
                        # 差分变异
                        mutant = population[indices[0]] + mutation_factor * (
                            population[indices[1]] - population[indices[2]]
                        )
                    elif strategy == "best/1/bin":
                        indices = np.random.choice(population_size, 2, replace=False)
                        while i in indices:
                            indices = np.random.choice(population_size, 2, replace=False)
                        
                        mutant = best_solution + mutation_factor * (
                            population[indices[0]] - population[indices[1]]
                        )
                    else:
                        raise ValueError(f"未知的变异策略: {strategy}")
                    
                    # 边界处理
                    for j in range(n_vars):
                        lower, upper = bounds[j]
                        mutant[j] = np.clip(mutant[j], lower, upper)
                    
                    # 二项式交叉
                    trial = population[i].copy()
                    j_rand = np.random.randint(n_vars)
                    
                    for j in range(n_vars):
                        if np.random.random() <= crossover_probability or j == j_rand:
                            trial[j] = mutant[j]
                    
                    # 选择操作
                    trial_fitness = objective_func(trial)
                    if trial_fitness <= fitness[i]:
                        new_population[i] = trial
                        fitness[i] = trial_fitness
                        
                        if trial_fitness < best_fitness:
                            best_fitness = trial_fitness
                            best_solution = trial.copy()
                    else:
                        new_population[i] = population[i]
                
                population = new_population
                convergence_history.append(best_fitness)
                
                # 检查收敛
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < 1e-8:
                    break
                    
        except Exception as e:
            logger.error(f"差分进化算法失败: {e}")
        
        return OptimizationResult(
            x=best_solution, fval=best_fitness, success=True,
            iterations=iteration + 1,
            message=f"差分进化算法完成",
            convergence_history=convergence_history,
            execution_time=0.0
        )
    
    @staticmethod
    def firefly_algorithm(objective_func: Callable,
                         bounds: List[Tuple[float, float]],
                         population_size: int = 40,
                         max_iterations: int = 1000,
                         alpha: float = 0.2,
                         beta0: float = 1.0,
                         gamma: float = 1.0) -> OptimizationResult:
        """
        萤火虫算法
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            population_size: 种群大小
            max_iterations: 最大迭代次数
            alpha: 随机步长参数
            beta0: 吸引度基数
            gamma: 光强衰减系数
        """
        n_vars = len(bounds)
        
        # 初始化萤火虫群
        fireflies = np.zeros((population_size, n_vars))
        for i, (lower, upper) in enumerate(bounds):
            fireflies[:, i] = np.random.uniform(lower, upper, population_size)
        
        # 评估光强（适应度）
        intensity = np.array([1.0 / (1.0 + objective_func(fa)) for fa in fireflies])
        
        # 找到最亮的萤火虫
        best_idx = np.argmax(intensity)
        best_firefly = fireflies[best_idx].copy()
        best_intensity = intensity[best_idx]
        
        convergence_history = []
        
        try:
            for iteration in range(max_iterations):
                new_fireflies = fireflies.copy()
                
                for i in range(population_size):
                    for j in range(population_size):
                        if intensity[j] > intensity[i]:  # j比i更亮
                            # 计算距离
                            r = np.linalg.norm(fireflies[j] - fireflies[i])
                            
                            # 计算吸引度
                            beta = beta0 / (1.0 + gamma * r**2)
                            
                            # 移动
                            step_size = alpha * (np.random.random(n_vars) - 0.5)
                            new_fireflies[i] += beta * (fireflies[j] - fireflies[i]) + step_size
                            
                            # 边界处理
                            for k in range(n_vars):
                                lower, upper = bounds[k]
                                new_fireflies[i, k] = np.clip(new_fireflies[i, k], lower, upper)
                
                # 评估新位置
                new_intensity = np.array([1.0 / (1.0 + objective_func(fa)) for fa in new_fireflies])
                
                # 更新位置
                for i in range(population_size):
                    if new_intensity[i] > intensity[i]:
                        fireflies[i] = new_fireflies[i]
                        intensity[i] = new_intensity[i]
                
                # 更新全局最优
                current_best_idx = np.argmax(intensity)
                if intensity[current_best_idx] > best_intensity:
                    best_intensity = intensity[current_best_idx]
                    best_firefly = fireflies[current_best_idx].copy()
                
                convergence_history.append(1.0 / best_intensity - 1.0)
                
                # 动态调整参数
                alpha *= 0.99
                
                # 检查收敛
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < 1e-8:
                    break
                    
        except Exception as e:
            logger.error(f"萤火虫算法失败: {e}")
        
        best_fitness = objective_func(best_firefly)
        
        return OptimizationResult(
            x=best_firefly, fval=best_fitness, success=True,
            iterations=iteration + 1,
            message=f"萤火虫算法完成",
            convergence_history=convergence_history,
            execution_time=0.0
        )
    
    @staticmethod
    def cuckoo_search(objective_func: Callable,
                     bounds: List[Tuple[float, float]],
                     population_size: int = 25,
                     max_iterations: int = 1000,
                     pa: float = 0.25,
                     beta: float = 1.5) -> OptimizationResult:
        """
        布谷鸟搜索算法
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            population_size: 巢穴数量
            max_iterations: 最大迭代次数
            pa: 发现概率
            beta: 步长控制参数
        """
        n_vars = len(bounds)
        
        # 初始化巢穴
        nests = np.zeros((population_size, n_vars))
        for i, (lower, upper) in enumerate(bounds):
            nests[:, i] = np.random.uniform(lower, upper, population_size)
        
        # 评估巢穴
        fitness = np.array([objective_func(nest) for nest in nests])
        best_idx = np.argmin(fitness)
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_history = []
        
        try:
            for iteration in range(max_iterations):
                # 通过Levy飞行获取新解
                for i in range(population_size):
                    # Levy飞行步长
                    step_size = beta * (np.random.gamma(beta, 1) / 
                                      np.power(np.random.gamma(beta, 1), 1.0 / beta))
                    
                    # 随机方向
                    u = np.random.normal(0, step_size, n_vars)
                    v = np.random.normal(0, 1, n_vars)
                    
                    step = u / (np.power(np.abs(v), 1.0 / beta))
                    
                    # 生成新解
                    new_nest = nests[i] + step * np.random.randn(n_vars)
                    
                    # 边界处理
                    for j in range(n_vars):
                        lower, upper = bounds[j]
                        new_nest[j] = np.clip(new_nest[j], lower, upper)
                    
                    # 评估新解
                    new_fitness = objective_func(new_nest)
                    
                    # 如果新解更好，则替换
                    if new_fitness < fitness[i]:
                        nests[i] = new_nest
                        fitness[i] = new_fitness
                        
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_nest = new_nest.copy()
                
                # 发现和替换最差的巢穴
                worst_indices = np.argsort(fitness)[-int(pa * population_size):]
                for idx in worst_indices:
                    # 随机选择两个巢穴进行交叉
                    parent1, parent2 = np.random.choice(population_size, 2, replace=False)
                    while parent1 == idx or parent2 == idx:
                        parent1, parent2 = np.random.choice(population_size, 2, replace=False)
                    
                    # 简单交叉
                    new_nest = nests[parent1].copy()
                    crossover_point = np.random.randint(1, n_vars)
                    new_nest[crossover_point:] = nests[parent2][crossover_point:]
                    
                    # 边界处理
                    for j in range(n_vars):
                        lower, upper = bounds[j]
                        new_nest[j] = np.clip(new_nest[j], lower, upper)
                    
                    # 评估新解
                    new_fitness = objective_func(new_nest)
                    
                    if new_fitness < fitness[idx]:
                        nests[idx] = new_nest
                        fitness[idx] = new_fitness
                
                convergence_history.append(best_fitness)
                
                # 检查收敛
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < 1e-8:
                    break
                    
        except Exception as e:
            logger.error(f"布谷鸟搜索算法失败: {e}")
        
        return OptimizationResult(
            x=best_nest, fval=best_fitness, success=True,
            iterations=iteration + 1,
            message=f"布谷鸟搜索算法完成",
            convergence_history=convergence_history,
            execution_time=0.0
        )


class OptimizationVisualization:
    """优化结果可视化工具"""
    
    @staticmethod
    def plot_convergence_history(result: OptimizationResult, 
                               save_path: str = None, show: bool = True):
        """绘制收敛历史"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，跳过可视化")
            return
        
        if result.convergence_history is None:
            print("没有收敛历史数据")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(result.convergence_history, 'b-', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title(f'收敛历史 - {result.message}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_pareto_front(result: MultiObjectiveResult,
                        save_path: str = None, show: bool = True):
        """绘制Pareto前沿"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，跳过可视化")
            return
        
        if len(result.pareto_front.shape) == 1:
            print("Pareto前沿是1维的，无法绘制")
            return
        
        plt.figure(figsize=(10, 8))
        
        if result.pareto_front.shape[1] == 2:
            # 2D Pareto前沿
            plt.scatter(result.pareto_front[:, 0], result.pareto_front[:, 1], 
                       c='red', s=50, alpha=0.7, label='Pareto前沿')
            plt.xlabel('目标函数1')
            plt.ylabel('目标函数2')
            plt.title('2D Pareto前沿')
        elif result.pareto_front.shape[1] == 3:
            # 3D Pareto前沿
            ax = plt.axes(projection='3d')
            ax.scatter(result.pareto_front[:, 0], result.pareto_front[:, 1], 
                      result.pareto_front[:, 2], c='red', s=50, alpha=0.7)
            ax.set_xlabel('目标函数1')
            ax.set_ylabel('目标函数2')
            ax.set_zlabel('目标函数3')
            ax.set_title('3D Pareto前沿')
        else:
            # 高维Pareto前沿的平行坐标图
            plt.figure(figsize=(12, 6))
            for i, point in enumerate(result.pareto_front):
                plt.plot(range(len(point)), point, 'o-', alpha=0.7, label=f'解{i+1}' if i < 5 else "")
            
            plt.xlabel('目标函数索引')
            plt.ylabel('目标函数值')
            plt.title('高维Pareto前沿（平行坐标图）')
            if len(result.pareto_front) <= 5:
                plt.legend()
        
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_optimization_landscape(objective_func: Callable,
                                  bounds: List[Tuple[float, float]],
                                  resolution: int = 50,
                                  save_path: str = None, show: bool = True):
        """绘制优化景观"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，跳过可视化")
            return
        
        if len(bounds) != 2:
            print("仅支持2D优化景观绘制")
            return
        
        x1 = np.linspace(bounds[0][0], bounds[0][1], resolution)
        x2 = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        Z = np.zeros_like(X1)
        for i in range(resolution):
            for j in range(resolution):
                try:
                    Z[i, j] = objective_func(np.array([X1[i, j], X2[i, j]]))
                except Exception as e:
                    logger.warning(f"函数值计算失败 at ({i}, {j}): {e}")
                    Z[i, j] = np.nan
        
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(contour)
        plt.contour(X1, X2, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        plt.xlabel('变量1')
        plt.ylabel('变量2')
        plt.title('优化景观')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def compare_optimizer_results(comparison_results: Dict[str, Dict[str, float]],
                                save_path: str = None, show: bool = True):
        """比较优化器结果"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，跳过可视化")
            return
        
        optimizers = list(comparison_results.keys())
        metrics = ['mean_fval', 'success_rate', 'mean_execution_time']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 平均目标函数值
        mean_fvals = [comparison_results[opt]['mean_fval'] for opt in optimizers]
        axes[0].bar(optimizers, mean_fvals, color='skyblue', alpha=0.7)
        axes[0].set_title('平均目标函数值')
        axes[0].set_ylabel('目标函数值')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 成功率
        success_rates = [comparison_results[opt]['success_rate'] for opt in optimizers]
        axes[1].bar(optimizers, success_rates, color='lightgreen', alpha=0.7)
        axes[1].set_title('成功率')
        axes[1].set_ylabel('成功率')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        
        # 平均执行时间
        exec_times = [comparison_results[opt]['mean_execution_time'] for opt in optimizers]
        axes[2].bar(optimizers, exec_times, color='salmon', alpha=0.7)
        axes[2].set_title('平均执行时间')
        axes[2].set_ylabel('执行时间 (秒)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


class OptimizationBenchmark:
    """优化算法基准测试"""
    
    # 标准测试函数
    test_functions = {
        'sphere': {
            'func': lambda x: sum(xi**2 for xi in x),
            'bounds': [(-5, 5)] * 10,
            'global_optimum': 0.0,
            'description': '球面函数 - 简单凸函数'
        },
        'rosenbrock': {
            'func': lambda x: sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1)),
            'bounds': [(-2, 2)] * 10,
            'global_optimum': 0.0,
            'description': 'Rosenbrock函数 - 非凸函数'
        },
        'rastrigin': {
            'func': lambda x: 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x),
            'bounds': [(-5.12, 5.12)] * 10,
            'global_optimum': 0.0,
            'description': 'Rastrigin函数 - 多模态函数'
        },
        'ackley': {
            'func': lambda x: -20 * np.exp(-0.2 * np.sqrt(0.5 * sum(xi**2 for xi in x))) - 
                             np.exp(0.5 * sum(np.cos(2 * np.pi * xi) for xi in x)) + np.e + 20,
            'bounds': [(-32.768, 32.768)] * 10,
            'global_optimum': 0.0,
            'description': 'Ackley函数 - 多模态函数'
        },
        'griewank': {
            'func': lambda x: 1 + sum(xi**2 / 4000 for xi in x) - 
                             np.prod(np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)),
            'bounds': [(-600, 600)] * 10,
            'global_optimum': 0.0,
            'description': 'Griewank函数 - 多模态函数'
        }
    }
    
    @staticmethod
    def run_benchmark(optimizer_names: List[str],
                     test_function_names: List[str] = None,
                     dimensions: List[int] = [10, 30],
                     n_runs: int = 5,
                     save_results: bool = True,
                     results_file: str = "benchmark_results.json") -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            optimizer_names: 要测试的优化器名称列表
            test_function_names: 要测试的函数名称列表
            dimensions: 测试维度列表
            n_runs: 每个测试的运行次数
            save_results: 是否保存结果
            results_file: 结果保存文件路径
        """
        if test_function_names is None:
            test_function_names = list(OptimizationBenchmark.test_functions.keys())
        
        opt_tools = OptimizationTools()
        results = {}
        
        for func_name in test_function_names:
            if func_name not in OptimizationBenchmark.test_functions:
                continue
                
            test_func_info = OptimizationBenchmark.test_functions[func_name]
            objective_func = test_func_info['func']
            bounds = test_func_info['bounds']
            global_optimum = test_func_info['global_optimum']
            
            results[func_name] = {}
            
            for dim in dimensions:
                results[func_name][f'dim_{dim}'] = {}
                
                # 调整边界到指定维度
                test_bounds = bounds[:dim]
                
                for optimizer_name in optimizer_names:
                    print(f"测试 {func_name} (维度={dim}) 使用 {optimizer_name}")
                    
                    run_results = []
                    
                    for run in range(n_runs):
                        # 随机初始点
                        x0 = np.random.uniform(
                            [b[0] for b in test_bounds],
                            [b[1] for b in test_bounds]
                        )
                        
                        try:
                            result = opt_tools.single_objective_optimize(
                                objective_func, optimizer_name, x0,
                                bounds=test_bounds, max_iterations=1000
                            )
                            
                            run_results.append({
                                'fval': result.fval,
                                'iterations': result.iterations,
                                'success': result.success,
                                'execution_time': result.execution_time,
                                'error': abs(result.fval - global_optimum)
                            })
                        except Exception as e:
                            run_results.append({
                                'fval': float('inf'),
                                'iterations': 0,
                                'success': False,
                                'execution_time': 0.0,
                                'error': float('inf')
                            })
                    
                    # 统计分析
                    fvals = [r['fval'] for r in run_results]
                    errors = [r['error'] for r in run_results]
                    exec_times = [r['execution_time'] for r in run_results]
                    success_rate = sum(1 for r in run_results if r['success']) / n_runs
                    
                    results[func_name][f'dim_{dim}'][optimizer_name] = {
                        'mean_error': np.mean(errors),
                        'std_error': np.std(errors),
                        'min_error': np.min(errors),
                        'mean_fval': np.mean(fvals),
                        'std_fval': np.std(fvals),
                        'success_rate': success_rate,
                        'mean_execution_time': np.mean(exec_times),
                        'all_results': run_results
                    }
        
        if save_results:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"基准测试结果已保存到: {results_file}")
        
        return results
    
    @staticmethod
    def analyze_benchmark_results(results: Dict[str, Any]) -> Dict[str, Any]:
        """分析基准测试结果"""
        analysis = {
            'summary': {},
            'rankings': {},
            'best_optimizer_per_function': {},
            'best_optimizer_overall': {}
        }
        
        # 总体统计
        all_optimizer_errors = defaultdict(list)
        all_optimizer_success_rates = defaultdict(list)
        all_optimizer_times = defaultdict(list)
        
        for func_name, func_results in results.items():
            for dim_key, dim_results in func_results.items():
                for optimizer_name, stats in dim_results.items():
                    all_optimizer_errors[optimizer_name].append(stats['mean_error'])
                    all_optimizer_success_rates[optimizer_name].append(stats['success_rate'])
                    all_optimizer_times[optimizer_name].append(stats['mean_execution_time'])
        
        # 计算总体性能
        for optimizer_name in all_optimizer_errors:
            analysis['summary'][optimizer_name] = {
                'mean_error': np.mean(all_optimizer_errors[optimizer_name]),
                'mean_success_rate': np.mean(all_optimizer_success_rates[optimizer_name]),
                'mean_execution_time': np.mean(all_optimizer_times[optimizer_name])
            }
        
        # 排名
        error_ranking = sorted(analysis['summary'].items(), 
                             key=lambda x: x[1]['mean_error'])
        success_ranking = sorted(analysis['summary'].items(), 
                               key=lambda x: x[1]['mean_success_rate'], reverse=True)
        time_ranking = sorted(analysis['summary'].items(), 
                            key=lambda x: x[1]['mean_execution_time'])
        
        analysis['rankings'] = {
            'by_error': [name for name, _ in error_ranking],
            'by_success_rate': [name for name, _ in success_ranking],
            'by_execution_time': [name for name, _ in time_ranking]
        }
        
        # 最佳优化器
        analysis['best_optimizer_overall'] = error_ranking[0][0]
        
        return analysis


class OptimizationUtils:
    """优化工具函数集合"""
    
    @staticmethod
    def normalize_objectives(objectives: List[Callable]) -> List[Callable]:
        """标准化目标函数"""
        def normalized_objective_factory(original_func):
            def normalized_func(x):
                try:
                    # 多次评估以估计范围
                    samples = []
                    for _ in range(10):
                        sample_x = np.random.uniform(-5, 5, len(x))
                        samples.append(original_func(sample_x))
                    
                    min_val = min(samples)
                    max_val = max(samples)
                    
                    if abs(max_val - min_val) < 1e-10:
                        return original_func(x)
                    
                    return (original_func(x) - min_val) / (max_val - min_val)
                except Exception as e:
                    logger.warning(f"函数归一化失败: {e}")
                    return original_func(x)
            
            return normalized_func
        
        return [normalized_objective_factory(func) for func in objectives]
    
    @staticmethod
    def scale_variables(x: np.ndarray, bounds: List[Tuple[float, float]], 
                       to_unit: bool = True) -> np.ndarray:
        """变量缩放"""
        if to_unit:
            # 缩放到[0,1]
            scaled = np.zeros_like(x)
            for i, (lower, upper) in enumerate(bounds):
                scaled[i] = (x[i] - lower) / (upper - lower)
            return scaled
        else:
            # 从[0,1]缩放回原范围
            scaled = np.zeros_like(x)
            for i, (lower, upper) in enumerate(bounds):
                scaled[i] = x[i] * (upper - lower) + lower
            return scaled
    
    @staticmethod
    def calculate_constraint_violation(x: np.ndarray, 
                                     constraints: List[Callable]) -> float:
        """计算约束违反程度"""
        violation = 0.0
        for constraint in constraints:
            viol = constraint(x)
            if viol > 0:
                violation += viol
        return violation
    
    @staticmethod
    def adaptive_parameter_tuning(optimizer_name: str, 
                                problem_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """自适应参数调整"""
        default_params = {
            'gradient_descent': {'learning_rate': 0.01},
            'newton': {},
            'genetic_algorithm': {'population_size': 50, 'mutation_rate': 0.1, 'crossover_rate': 0.8},
            'particle_swarm': {'inertia_weight': 0.9, 'cognitive_coefficient': 2.0, 'social_coefficient': 2.0},
            'simulated_annealing': {'initial_temperature': 1000.0, 'cooling_rate': 0.95},
            'differential_evolution': {'mutation_factor': 0.8, 'crossover_probability': 0.7},
            'firefly_algorithm': {'alpha': 0.2, 'beta0': 1.0, 'gamma': 1.0},
            'cuckoo_search': {'pa': 0.25, 'beta': 1.5},
            'gaussian_process': {'length_scale': 1.0, 'noise_level': 1e-6}
        }
        
        params = default_params.get(optimizer_name, {}).copy()
        
        # 根据问题特征调整参数
        n_variables = problem_characteristics.get('n_variables', 10)
        is_multi_modal = problem_characteristics.get('is_multi_modal', False)
        has_constraints = problem_characteristics.get('has_constraints', False)
        
        if optimizer_name == 'genetic_algorithm':
            if n_variables > 50:
                params['population_size'] = min(100, n_variables * 2)
            if is_multi_modal:
                params['mutation_rate'] = 0.2
                params['crossover_rate'] = 0.9
        
        elif optimizer_name == 'particle_swarm':
            if n_variables > 30:
                params['inertia_weight'] = 0.7
        
        elif optimizer_name == 'simulated_annealing':
            if is_multi_modal:
                params['initial_temperature'] = 2000.0
                params['cooling_rate'] = 0.99
        
        elif optimizer_name == 'gaussian_process':
            if n_variables > 20:
                params['length_scale'] = 2.0
        
        return params
    
    @staticmethod
    def hybrid_optimize(objective_func: Callable,
                       optimizer_sequence: List[Tuple[str, Dict[str, Any]]],
                       x0: np.ndarray) -> OptimizationResult:
        """混合优化 - 按顺序使用多个优化器"""
        current_x = x0.copy()
        
        for optimizer_name, kwargs in optimizer_sequence:
            opt_tools = OptimizationTools()
            optimizer = opt_tools.get_optimizer(optimizer_name, **kwargs)
            result = optimizer.optimize(objective_func, current_x)
            
            if result.success:
                current_x = result.x
                logger.info(f"{optimizer_name} 优化成功，目标值: {result.fval:.6f}")
            else:
                logger.warning(f"{optimizer_name} 优化失败")
                break
        
        # 返回最终结果
        final_fval = objective_func(current_x)
        return OptimizationResult(
            x=current_x, fval=final_fval, success=True,
            iterations=0,  # 混合优化不统计迭代次数
            message="混合优化完成",
            convergence_history=None,
            execution_time=0.0
        )
    
    @staticmethod
    def multi_start_optimize(objective_func: Callable,
                           optimizer_name: str,
                           bounds: List[Tuple[float, float]],
                           n_starts: int = 10,
                           **optimizer_kwargs) -> OptimizationResult:
        """多起点优化"""
        opt_tools = OptimizationTools()
        results = []
        
        for start in range(n_starts):
            # 随机初始点
            x0 = np.random.uniform(
                [b[0] for b in bounds],
                [b[1] for b in bounds]
            )
            
            try:
                result = opt_tools.single_objective_optimize(
                    objective_func, optimizer_name, x0,
                    bounds=bounds, **optimizer_kwargs
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"第 {start+1} 次起点优化失败: {e}")
        
        if not results:
            raise OptimizationError("所有起点优化都失败")
        
        # 选择最佳结果
        best_result = min(results, key=lambda r: r.fval)
        
        return OptimizationResult(
            x=best_result.x, fval=best_result.fval, success=best_result.success,
            iterations=best_result.iterations,
            message=f"多起点优化完成 (尝试了 {n_starts} 个起点)",
            convergence_history=best_result.convergence_history,
            execution_time=best_result.execution_time
        )


class OptimizationProfiler:
    """优化性能分析器"""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_optimizer(self, optimizer_name: str, 
                         objective_func: Callable,
                         bounds: List[Tuple[float, float]],
                         x0: np.ndarray,
                         **optimizer_kwargs) -> Dict[str, Any]:
        """分析优化器性能"""
        try:
            import cProfile
            import pstats
            import io
        except ImportError:
            print("cProfile未安装，跳过性能分析")
            return {}
        
        # 性能分析
        pr = cProfile.Profile()
        pr.enable()
        
        # 执行优化
        opt_tools = OptimizationTools()
        result = opt_tools.single_objective_optimize(
            objective_func, optimizer_name, x0,
            bounds=bounds, **optimizer_kwargs
        )
        
        pr.disable()
        
        # 获取性能统计
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # 显示前20个函数
        
        profile_data = {
            'optimizer': optimizer_name,
            'result': result,
            'performance_stats': s.getvalue(),
            'optimization_time': result.execution_time,
            'iterations': result.iterations,
            'final_fval': result.fval
        }
        
        self.profiles[optimizer_name] = profile_data
        return profile_data
    
    def compare_profiles(self) -> Dict[str, Any]:
        """比较不同优化器的性能"""
        if len(self.profiles) < 2:
            return {"error": "需要至少2个优化器配置文件进行比较"}
        
        comparison = {
            'execution_times': {},
            'iterations': {},
            'final_values': {},
            'profiles': self.profiles
        }
        
        for name, profile in self.profiles.items():
            comparison['execution_times'][name] = profile['optimization_time']
            comparison['iterations'][name] = profile['iterations']
            comparison['final_values'][name] = profile['final_fval']
        
        return comparison
    
    def generate_report(self, save_path: str = "optimization_report.txt"):
        """生成优化报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("优化性能分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            for name, profile in self.profiles.items():
                f.write(f"优化器: {name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"执行时间: {profile['optimization_time']:.4f} 秒\n")
                f.write(f"迭代次数: {profile['iterations']}\n")
                f.write(f"最终目标值: {profile['final_fval']:.6f}\n")
                f.write(f"优化结果: {'成功' if profile['result'].success else '失败'}\n")
                f.write("\n性能统计:\n")
                f.write(profile['performance_stats'])
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"优化报告已保存到: {save_path}")


# 扩展OptimizationTools类
class ExtendedOptimizationTools(OptimizationTools):
    """扩展的优化工具类"""
    
    def __init__(self):
        super().__init__()
        self.advanced_optimizers = AdvancedOptimizers()
        self.visualization = OptimizationVisualization()
        self.benchmark = OptimizationBenchmark()
        self.utils = OptimizationUtils()
        self.profiler = OptimizationProfiler()
    
    def differential_evolution(self, objective_func: Callable,
                             bounds: List[Tuple[float, float]],
                             **kwargs) -> OptimizationResult:
        """差分进化算法"""
        return self.advanced_optimizers.differential_evolution(
            objective_func, bounds, **kwargs
        )
    
    def firefly_algorithm(self, objective_func: Callable,
                        bounds: List[Tuple[float, float]],
                        **kwargs) -> OptimizationResult:
        """萤火虫算法"""
        return self.advanced_optimizers.firefly_algorithm(
            objective_func, bounds, **kwargs
        )
    
    def cuckoo_search(self, objective_func: Callable,
                     bounds: List[Tuple[float, float]],
                     **kwargs) -> OptimizationResult:
        """布谷鸟搜索算法"""
        return self.advanced_optimizers.cuckoo_search(
            objective_func, bounds, **kwargs
        )
    
    def hybrid_optimize(self, objective_func: Callable,
                       optimizer_sequence: List[Tuple[str, Dict[str, Any]]],
                       x0: np.ndarray) -> OptimizationResult:
        """混合优化"""
        return self.utils.hybrid_optimize(objective_func, optimizer_sequence, x0)
    
    def multi_start_optimize(self, objective_func: Callable,
                           optimizer_name: str,
                           bounds: List[Tuple[float, float]],
                           n_starts: int = 10,
                           **optimizer_kwargs) -> OptimizationResult:
        """多起点优化"""
        return self.utils.multi_start_optimize(
            objective_func, optimizer_name, bounds, n_starts, **optimizer_kwargs
        )
    
    def run_benchmark(self, optimizer_names: List[str],
                     test_function_names: List[str] = None,
                     dimensions: List[int] = [10, 30],
                     n_runs: int = 5,
                     save_results: bool = True,
                     results_file: str = "benchmark_results.json") -> Dict[str, Any]:
        """运行基准测试"""
        return self.benchmark.run_benchmark(
            optimizer_names, test_function_names, dimensions, n_runs, save_results, results_file
        )
    
    def analyze_benchmark_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析基准测试结果"""
        return self.benchmark.analyze_benchmark_results(results)
    
    def profile_optimizer(self, optimizer_name: str,
                         objective_func: Callable,
                         bounds: List[Tuple[float, float]],
                         x0: np.ndarray,
                         **optimizer_kwargs) -> Dict[str, Any]:
        """分析优化器性能"""
        return self.profiler.profile_optimizer(
            optimizer_name, objective_func, bounds, x0, **optimizer_kwargs
        )
    
    def compare_profiles(self) -> Dict[str, Any]:
        """比较优化器性能"""
        return self.profiler.compare_profiles()
    
    def generate_report(self, save_path: str = "optimization_report.txt"):
        """生成优化报告"""
        self.profiler.generate_report(save_path)


def comprehensive_example():
    """综合使用示例"""
    print("=== J5优化工具综合示例 ===")
    
    # 创建扩展优化工具
    opt_tools = ExtendedOptimizationTools()
    
    # 定义测试函数
    def rosenbrock(x):
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
    
    def sphere(x):
        return sum(xi**2 for xi in x)
    
    # 1. 经典优化算法对比
    print("\n1. 经典优化算法对比")
    x0 = np.array([0.0, 0.0])
    bounds = [(-2, 2), (-2, 2)]
    
    classic_optimizers = ['gradient_descent', 'newton', 'conjugate_gradient']
    classic_results = opt_tools.compare_optimizers(
        sphere, classic_optimizers, x0, bounds=bounds, n_runs=3
    )
    
    for opt_name, stats in classic_results.items():
        print(f"{opt_name}: 平均目标值={stats['mean_fval']:.6f}, 成功率={stats['success_rate']:.2%}")
    
    # 2. 智能优化算法
    print("\n2. 智能优化算法")
    intelligent_optimizers = ['genetic_algorithm', 'particle_swarm', 'simulated_annealing']
    intelligent_results = opt_tools.compare_optimizers(
        rosenbrock, intelligent_optimizers, x0, bounds=bounds, n_runs=3
    )
    
    for opt_name, stats in intelligent_results.items():
        print(f"{opt_name}: 平均目标值={stats['mean_fval']:.6f}, 成功率={stats['success_rate']:.2%}")
    
    # 3. 高级优化算法
    print("\n3. 高级优化算法")
    
    # 差分进化
    de_result = opt_tools.differential_evolution(sphere, bounds, max_iterations=100)
    print(f"差分进化: f={de_result.fval:.6f}")
    
    # 萤火虫算法
    fa_result = opt_tools.firefly_algorithm(rosenbrock, bounds, max_iterations=100)
    print(f"萤火虫算法: f={fa_result.fval:.6f}")
    
    # 布谷鸟搜索
    cs_result = opt_tools.cuckoo_search(sphere, bounds, max_iterations=100)
    print(f"布谷鸟搜索: f={cs_result.fval:.6f}")
    
    # 4. 多目标优化
    print("\n4. 多目标优化")
    def obj1(x):
        return (x[0] - 1)**2 + (x[1] - 1)**2
    
    def obj2(x):
        return (x[0] + 1)**2 + (x[1] + 1)**2
    
    mo_result = opt_tools.multi_objective_optimize(
        [obj1, obj2], 'nsga2', bounds=bounds, max_iterations=50
    )
    print(f"NSGA-II Pareto前沿解数量: {len(mo_result.pareto_front)}")
    print(f"超体积: {mo_result.hypervolume:.6f}")
    
    # 5. 约束优化
    print("\n5. 约束优化")
    def constraint1(x):
        return x[0] + x[1] - 1
    
    constrained_result = opt_tools.constrained_optimize(
        sphere, 'penalty_function', x0, constraints=[constraint1], bounds=bounds
    )
    print(f"约束优化结果: f={constrained_result.fval:.6f}")
    
    # 6. 贝叶斯优化
    print("\n6. 贝叶斯优化")
    bayes_result = opt_tools.single_objective_optimize(
        sphere, 'gaussian_process', x0, bounds=bounds, max_iterations=20
    )
    print(f"贝叶斯优化结果: f={bayes_result.fval:.6f}")
    
    # 7. 混合优化
    print("\n7. 混合优化")
    hybrid_sequence = [
        ('genetic_algorithm', {'max_iterations': 50, 'population_size': 30}),
        ('gradient_descent', {'max_iterations': 100, 'learning_rate': 0.01})
    ]
    
    hybrid_result = opt_tools.hybrid_optimize(rosenbrock, hybrid_sequence, x0)
    print(f"混合优化结果: f={hybrid_result.fval:.6f}")
    
    # 8. 多起点优化
    print("\n8. 多起点优化")
    multi_start_result = opt_tools.multi_start_optimize(
        rosenbrock, 'gradient_descent', bounds, n_starts=5
    )
    print(f"多起点优化结果: f={multi_start_result.fval:.6f}")
    
    # 9. 并行优化
    print("\n9. 并行优化")
    objectives = [sphere, rosenbrock]
    x0s = [np.array([1.0, 1.0]), np.array([-1.0, -1.0])]
    
    parallel_results = opt_tools.parallel_optimize(
        objectives, 'gradient_descent', x0s, bounds=bounds, n_workers=2
    )
    
    for i, result in enumerate(parallel_results):
        if result:
            print(f"并行任务 {i}: f={result.fval:.6f}")
    
    # 10. 基准测试
    print("\n10. 基准测试")
    benchmark_results = opt_tools.run_benchmark(
        ['gradient_descent', 'genetic_algorithm'], 
        ['sphere', 'rosenbrock'], 
        dimensions=[2, 5], 
        n_runs=2,
        save_results=False
    )
    
    analysis = opt_tools.analyze_benchmark_results(benchmark_results)
    print(f"最佳优化器: {analysis['best_optimizer_overall']}")
    
    # 11. 性能分析
    print("\n11. 性能分析")
    profile_result = opt_tools.profile_optimizer(
        'gradient_descent', sphere, bounds, x0, max_iterations=100
    )
    print(f"梯度下降性能分析完成")
    
    # 保存结果
    print("\n12. 保存结果")
    opt_tools.save_results(de_result, "results/differential_evolution.json")
    opt_tools.save_results(mo_result, "results/nsga2_results.json")
    
    # 生成报告
    print("\n13. 生成报告")
    opt_tools.generate_report("optimization_report.txt")
    
    print("\n=== 综合示例完成 ===")


class MetaheuristicOptimizers:
    """元启发式优化算法集合"""
    
    @staticmethod
    def artificial_bee_colony(objective_func: Callable,
                            bounds: List[Tuple[float, float]],
                            colony_size: int = 50,
                            max_iterations: int = 1000,
                            limit: int = 10) -> OptimizationResult:
        """
        人工蜂群算法
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            colony_size: 蜂群大小
            max_iterations: 最大迭代次数
            limit: 雇佣蜂探索次数限制
        """
        n_vars = len(bounds)
        
        # 初始化食物源
        food_sources = np.zeros((colony_size, n_vars))
        for i, (lower, upper) in enumerate(bounds):
            food_sources[:, i] = np.random.uniform(lower, upper, colony_size)
        
        # 评估适应度
        fitness = np.array([1.0 / (1.0 + objective_func(source)) for source in food_sources])
        
        # 记录探索次数
        trial_counter = np.zeros(colony_size)
        
        # 找到最佳食物源
        best_idx = np.argmax(fitness)
        best_source = food_sources[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_history = []
        
        try:
            for iteration in range(max_iterations):
                # 雇佣蜂阶段
                for i in range(colony_size):
                    # 选择邻居
                    neighbor = np.random.randint(colony_size)
                    while neighbor == i:
                        neighbor = np.random.randint(colony_size)
                    
                    # 生成新解
                    new_source = food_sources[i].copy()
                    dimension = np.random.randint(n_vars)
                    phi = np.random.uniform(-1, 1)
                    
                    new_source[dimension] = food_sources[i][dimension] + phi * (
                        food_sources[i][dimension] - food_sources[neighbor][dimension]
                    )
                    
                    # 边界处理
                    lower, upper = bounds[dimension]
                    new_source[dimension] = np.clip(new_source[dimension], lower, upper)
                    
                    # 评估新解
                    new_fitness = 1.0 / (1.0 + objective_func(new_source))
                    
                    # 选择更好的解
                    if new_fitness > fitness[i]:
                        food_sources[i] = new_source
                        fitness[i] = new_fitness
                        trial_counter[i] = 0
                    else:
                        trial_counter[i] += 1
                
                # 观察蜂阶段
                fitness_sum = np.sum(fitness)
                if fitness_sum > 0:
                    selection_probs = fitness / fitness_sum
                else:
                    selection_probs = np.ones(colony_size) / colony_size
                
                for i in range(colony_size):
                    # 按概率选择食物源
                    selected_source_idx = np.random.choice(colony_size, p=selection_probs)
                    
                    # 邻居选择
                    neighbor = np.random.randint(colony_size)
                    while neighbor == selected_source_idx:
                        neighbor = np.random.randint(colony_size)
                    
                    # 生成新解
                    new_source = food_sources[selected_source_idx].copy()
                    dimension = np.random.randint(n_vars)
                    phi = np.random.uniform(-1, 1)
                    
                    new_source[dimension] = food_sources[selected_source_idx][dimension] + phi * (
                        food_sources[selected_source_idx][dimension] - food_sources[neighbor][dimension]
                    )
                    
                    # 边界处理
                    lower, upper = bounds[dimension]
                    new_source[dimension] = np.clip(new_source[dimension], lower, upper)
                    
                    # 评估新解
                    new_fitness = 1.0 / (1.0 + objective_func(new_source))
                    
                    # 选择更好的解
                    if new_fitness > fitness[selected_source_idx]:
                        food_sources[selected_source_idx] = new_source
                        fitness[selected_source_idx] = new_fitness
                        trial_counter[selected_source_idx] = 0
                    else:
                        trial_counter[selected_source_idx] += 1
                
                # 侦察蜂阶段
                for i in range(colony_size):
                    if trial_counter[i] >= limit:
                        # 放弃该食物源，随机生成新的
                        for j in range(n_vars):
                            lower, upper = bounds[j]
                            food_sources[i, j] = np.random.uniform(lower, upper)
                        
                        # 重新评估
                        fitness[i] = 1.0 / (1.0 + objective_func(food_sources[i]))
                        trial_counter[i] = 0
                
                # 更新全局最优
                current_best_idx = np.argmax(fitness)
                if fitness[current_best_idx] > best_fitness:
                    best_fitness = fitness[current_best_idx]
                    best_source = food_sources[current_best_idx].copy()
                
                convergence_history.append(1.0 / best_fitness - 1.0)
                
                # 检查收敛
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < 1e-8:
                    break
                    
        except Exception as e:
            logger.error(f"人工蜂群算法失败: {e}")
        
        best_value = objective_func(best_source)
        
        return OptimizationResult(
            x=best_source, fval=best_value, success=True,
            iterations=iteration + 1,
            message=f"人工蜂群算法完成",
            convergence_history=convergence_history,
            execution_time=0.0
        )
    
    @staticmethod
    def whale_optimization(objective_func: Callable,
                         bounds: List[Tuple[float, float]],
                         population_size: int = 30,
                         max_iterations: int = 1000,
                         a_decrease: float = 2.0) -> OptimizationResult:
        """
        鲸鱼优化算法
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            population_size: 鲸鱼数量
            max_iterations: 最大迭代次数
            a_decrease: 收敛参数递减系数
        """
        n_vars = len(bounds)
        
        # 初始化鲸鱼群
        whales = np.zeros((population_size, n_vars))
        for i, (lower, upper) in enumerate(bounds):
            whales[:, i] = np.random.uniform(lower, upper, population_size)
        
        # 评估适应度
        fitness = np.array([objective_func(whale) for whale in whales])
        
        # 找到最佳鲸鱼
        best_idx = np.argmin(fitness)
        best_whale = whales[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_history = []
        
        try:
            for iteration in range(max_iterations):
                a = 2 - iteration * (2.0 / max_iterations)  # 线性递减从2到0
                
                for i in range(population_size):
                    r = np.random.random()
                    
                    if r < 0.5:
                        if abs(a) >= 1:
                            # 包围猎物
                            rand_whale = whales[np.random.randint(population_size)]
                            D = np.abs(rand_whale - whales[i])
                            whales[i] = rand_whale - a * D
                        else:
                            # 搜索猎物
                            rand_whale = whales[np.random.randint(population_size)]
                            D = np.abs(rand_whale - whales[i])
                            whales[i] = rand_whale - a * D
                    else:
                        # 螺旋攻击
                        D = np.abs(best_whale - whales[i])
                        whales[i] = D * np.exp(0.4 * np.random.random() * 2 * np.pi) * np.cos(2 * np.pi * np.random.random()) + best_whale
                    
                    # 边界处理
                    for j in range(n_vars):
                        lower, upper = bounds[j]
                        whales[i, j] = np.clip(whales[i, j], lower, upper)
                
                # 评估新位置
                new_fitness = np.array([objective_func(whale) for whale in whales])
                
                # 更新最佳解
                for i in range(population_size):
                    if new_fitness[i] < fitness[i]:
                        fitness[i] = new_fitness[i]
                        
                        if new_fitness[i] < best_fitness:
                            best_fitness = new_fitness[i]
                            best_whale = whales[i].copy()
                
                convergence_history.append(best_fitness)
                
                # 检查收敛
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < 1e-8:
                    break
                    
        except Exception as e:
            logger.error(f"鲸鱼优化算法失败: {e}")
        
        return OptimizationResult(
            x=best_whale, fval=best_fitness, success=True,
            iterations=iteration + 1,
            message=f"鲸鱼优化算法完成",
            convergence_history=convergence_history,
            execution_time=0.0
        )
    
    @staticmethod
    def sine_cosine_algorithm(objective_func: Callable,
                            bounds: List[Tuple[float, float]],
                            population_size: int = 50,
                            max_iterations: int = 1000,
                            a: float = 2.0) -> OptimizationResult:
        """
        正弦余弦算法
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            population_size: 解的数量
            max_iterations: 最大迭代次数
            a: 收敛参数
        """
        n_vars = len(bounds)
        
        # 初始化解
        solutions = np.zeros((population_size, n_vars))
        for i, (lower, upper) in enumerate(bounds):
            solutions[:, i] = np.random.uniform(lower, upper, population_size)
        
        # 评估适应度
        fitness = np.array([objective_func(solution) for solution in solutions])
        
        # 找到最佳解
        best_idx = np.argmin(fitness)
        best_solution = solutions[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_history = []
        
        try:
            for iteration in range(max_iterations):
                r1 = a - iteration * (a / max_iterations)  # 线性递减从a到0
                
                for i in range(population_size):
                    new_solution = solutions[i].copy()
                    
                    for j in range(n_vars):
                        r2 = np.random.random()
                        r3 = np.random.random()
                        r4 = np.random.random()
                        
                        if r4 < 0.5:
                            # 正弦更新
                            new_solution[j] = solutions[i][j] + r1 * np.sin(r2) * abs(
                                r3 * best_solution[j] - solutions[i][j]
                            )
                        else:
                            # 余弦更新
                            new_solution[j] = solutions[i][j] + r1 * np.cos(r2) * abs(
                                r3 * best_solution[j] - solutions[i][j]
                            )
                        
                        # 边界处理
                        lower, upper = bounds[j]
                        new_solution[j] = np.clip(new_solution[j], lower, upper)
                    
                    # 评估新解
                    new_fitness = objective_func(new_solution)
                    
                    # 选择更好的解
                    if new_fitness < fitness[i]:
                        solutions[i] = new_solution
                        fitness[i] = new_fitness
                        
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_solution = new_solution.copy()
                
                convergence_history.append(best_fitness)
                
                # 检查收敛
                if iteration > 0 and abs(convergence_history[-2] - convergence_history[-1]) < 1e-8:
                    break
                    
        except Exception as e:
            logger.error(f"正弦余弦算法失败: {e}")
        
        return OptimizationResult(
            x=best_solution, fval=best_fitness, success=True,
            iterations=iteration + 1,
            message=f"正弦余弦算法完成",
            convergence_history=convergence_history,
            execution_time=0.0
        )


class OptimizationAnalysis:
    """优化分析工具"""
    
    @staticmethod
    def sensitivity_analysis(objective_func: Callable,
                           x0: np.ndarray,
                           bounds: List[Tuple[float, float]],
                           perturbation_factor: float = 0.01,
                           n_samples: int = 100) -> Dict[str, Any]:
        """
        敏感性分析
        
        Args:
            objective_func: 目标函数
            x0: 基准点
            bounds: 变量边界
            perturbation_factor: 扰动因子
            n_samples: 采样数量
        """
        n_vars = len(x0)
        sensitivities = np.zeros(n_vars)
        
        for i in range(n_vars):
            # 生成扰动样本
            perturbations = np.random.normal(0, perturbation_factor * (bounds[i][1] - bounds[i][0]), n_samples)
            original_values = []
            perturbed_values = []
            
            for perturbation in perturbations:
                # 原始值
                original_values.append(objective_func(x0))
                
                # 扰动值
                perturbed_x = x0.copy()
                perturbed_x[i] += perturbation
                
                # 边界处理
                lower, upper = bounds[i]
                perturbed_x[i] = np.clip(perturbed_x[i], lower, upper)
                
                perturbed_values.append(objective_func(perturbed_x))
            
            # 计算敏感性（目标函数变化的标准差）
            changes = np.array(perturbed_values) - np.array(original_values)
            sensitivities[i] = np.std(changes)
        
        return {
            'sensitivities': sensitivities,
            'most_sensitive_var': np.argmax(sensitivities),
            'least_sensitive_var': np.argmin(sensitivities),
            'sensitivity_ratio': np.max(sensitivities) / (np.min(sensitivities) + 1e-10)
        }
    
    @staticmethod
    def landscape_analysis(objective_func: Callable,
                          bounds: List[Tuple[float, float]],
                          resolution: int = 20) -> Dict[str, Any]:
        """
        景观分析
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            resolution: 分析分辨率
        """
        if len(bounds) != 2:
            return {"error": "仅支持2D景观分析"}
        
        # 生成网格
        x1 = np.linspace(bounds[0][0], bounds[0][1], resolution)
        x2 = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        # 评估函数值
        Z = np.zeros_like(X1)
        for i in range(resolution):
            for j in range(resolution):
                try:
                    Z[i, j] = objective_func(np.array([X1[i, j], X2[i, j]]))
                except Exception as e:
                    logger.warning(f"函数评估失败 at ({i}, {j}): {e}")
                    Z[i, j] = np.nan
        
        # 分析景观特征
        valid_values = Z[~np.isnan(Z)]
        
        analysis = {
            'min_value': np.nanmin(Z),
            'max_value': np.nanmax(Z),
            'mean_value': np.nanmean(Z),
            'std_value': np.nanstd(Z),
            'range': np.nanmax(Z) - np.nanmin(Z),
            'coefficient_of_variation': np.nanstd(Z) / (np.nanmean(Z) + 1e-10),
            'n_local_minima': 0,  # 简化版本
            'ruggedness': np.nanstd(Z),  # 粗糙度
            'landscape_type': 'unknown'
        }
        
        # 分类景观类型
        cv = analysis['coefficient_of_variation']
        if cv < 0.1:
            analysis['landscape_type'] = 'smooth'
        elif cv < 0.5:
            analysis['landscape_type'] = 'moderate'
        else:
            analysis['landscape_type'] = 'rugged'
        
        return analysis
    
    @staticmethod
    def convergence_analysis(results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        收敛性分析
        
        Args:
            results: 优化结果列表
        """
        if not results:
            return {"error": "没有提供优化结果"}
        
        successful_results = [r for r in results if r.success and r.convergence_history]
        
        if not successful_results:
            return {"error": "没有成功的优化结果"}
        
        analysis = {
            'n_successful': len(successful_results),
            'n_failed': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'convergence_patterns': {},
            'average_iterations': np.mean([r.iterations for r in successful_results]),
            'average_final_value': np.mean([r.fval for r in successful_results]),
            'best_result': min(successful_results, key=lambda r: r.fval),
            'worst_result': max(successful_results, key=lambda r: r.fval)
        }
        
        # 分析收敛模式
        fast_convergers = [r for r in successful_results if r.iterations < analysis['average_iterations']]
        slow_convergers = [r for r in successful_results if r.iterations >= analysis['average_iterations']]
        
        analysis['convergence_patterns'] = {
            'fast_convergence': len(fast_convergers),
            'slow_convergence': len(slow_convergers),
            'fast_convergence_rate': len(fast_convergers) / len(successful_results)
        }
        
        return analysis


class OptimizationConfig:
    """优化配置管理"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.configs = {}
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def add_optimizer_config(self, optimizer_name: str, config: Dict[str, Any]):
        """添加优化器配置"""
        self.configs[optimizer_name] = config
    
    def get_optimizer_config(self, optimizer_name: str, default_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """获取优化器配置"""
        if default_config is None:
            default_config = {}
        
        return self.configs.get(optimizer_name, default_config)
    
    def save_config(self, config_file: str = None):
        """保存配置"""
        file_path = config_file or self.config_file
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.configs, f, indent=2)
    
    def load_config(self, config_file: str):
        """加载配置"""
        with open(config_file, 'r') as f:
            self.configs = json.load(f)
    
    @staticmethod
    def get_default_configs() -> Dict[str, Dict[str, Any]]:
        """获取默认配置"""
        return {
            'gradient_descent': {
                'learning_rate': 0.01,
                'max_iterations': 1000,
                'tolerance': 1e-6
            },
            'newton': {
                'max_iterations': 1000,
                'tolerance': 1e-6
            },
            'genetic_algorithm': {
                'population_size': 50,
                'max_iterations': 1000,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elite_ratio': 0.1
            },
            'particle_swarm': {
                'population_size': 50,
                'max_iterations': 1000,
                'inertia_weight': 0.9,
                'cognitive_coefficient': 2.0,
                'social_coefficient': 2.0
            },
            'simulated_annealing': {
                'max_iterations': 1000,
                'initial_temperature': 1000.0,
                'cooling_rate': 0.95,
                'min_temperature': 1e-8
            },
            'differential_evolution': {
                'population_size': 50,
                'max_iterations': 1000,
                'mutation_factor': 0.8,
                'crossover_probability': 0.7
            },
            'gaussian_process': {
                'max_iterations': 100,
                'length_scale': 1.0,
                'noise_level': 1e-6,
                'acquisition_function': 'ei'
            }
        }


# 更新ExtendedOptimizationTools类
class FinalOptimizationTools(ExtendedOptimizationTools):
    """最终的优化工具类，包含所有功能"""
    
    def __init__(self):
        super().__init__()
        self.metaheuristic_optimizers = MetaheuristicOptimizers()
        self.analysis = OptimizationAnalysis()
        self.config = OptimizationConfig()
        
        # 加载默认配置
        default_configs = OptimizationConfig.get_default_configs()
        for optimizer_name, config in default_configs.items():
            self.config.add_optimizer_config(optimizer_name, config)
    
    def artificial_bee_colony(self, objective_func: Callable,
                            bounds: List[Tuple[float, float]],
                            **kwargs) -> OptimizationResult:
        """人工蜂群算法"""
        return self.metaheuristic_optimizers.artificial_bee_colony(
            objective_func, bounds, **kwargs
        )
    
    def whale_optimization(self, objective_func: Callable,
                         bounds: List[Tuple[float, float]],
                         **kwargs) -> OptimizationResult:
        """鲸鱼优化算法"""
        return self.metaheuristic_optimizers.whale_optimization(
            objective_func, bounds, **kwargs
        )
    
    def sine_cosine_algorithm(self, objective_func: Callable,
                            bounds: List[Tuple[float, float]],
                            **kwargs) -> OptimizationResult:
        """正弦余弦算法"""
        return self.metaheuristic_optimizers.sine_cosine_algorithm(
            objective_func, bounds, **kwargs
        )
    
    def sensitivity_analysis(self, objective_func: Callable,
                           x0: np.ndarray,
                           bounds: List[Tuple[float, float]],
                           **kwargs) -> Dict[str, Any]:
        """敏感性分析"""
        return self.analysis.sensitivity_analysis(
            objective_func, x0, bounds, **kwargs
        )
    
    def landscape_analysis(self, objective_func: Callable,
                          bounds: List[Tuple[float, float]],
                          **kwargs) -> Dict[str, Any]:
        """景观分析"""
        return self.analysis.landscape_analysis(
            objective_func, bounds, **kwargs
        )
    
    def convergence_analysis(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """收敛性分析"""
        return self.analysis.convergence_analysis(results)
    
    def get_optimizer_with_config(self, optimizer_name: str, 
                                bounds: List[Tuple[float, float]],
                                **override_kwargs) -> Optimizer:
        """根据配置获取优化器"""
        config = self.config.get_optimizer_config(optimizer_name)
        config.update(override_kwargs)
        
        return self.get_optimizer(optimizer_name, bounds=bounds, **config)
    
    def comprehensive_optimization(self, objective_func: Callable,
                                 bounds: List[Tuple[float, float]],
                                 x0: np.ndarray,
                                 strategy: str = 'auto') -> OptimizationResult:
        """
        综合优化策略
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            x0: 初始点
            strategy: 优化策略 ('auto', 'global', 'local', 'hybrid')
        """
        if strategy == 'auto':
            # 自动选择策略
            n_vars = len(bounds)
            if n_vars <= 5:
                strategy = 'global'
            elif n_vars <= 20:
                strategy = 'hybrid'
            else:
                strategy = 'local'
        
        if strategy == 'global':
            # 全局优化
            return self.genetic_algorithm(objective_func, bounds, max_iterations=200)
        
        elif strategy == 'local':
            # 局部优化
            return self.gradient_descent(objective_func, bounds, x0, max_iterations=500)
        
        elif strategy == 'hybrid':
            # 混合优化
            # 第一阶段：全局搜索
            global_result = self.genetic_algorithm(objective_func, bounds, max_iterations=100)
            
            # 第二阶段：局部精化
            local_result = self.gradient_descent(
                objective_func, bounds, global_result.x, max_iterations=200
            )
            
            return local_result
        
        else:
            raise ValueError(f"未知的优化策略: {strategy}")
    
    def batch_optimization(self, problems: List[Dict[str, Any]],
                         optimizer_name: str = 'auto',
                         n_workers: int = 4) -> List[OptimizationResult]:
        """
        批量优化
        
        Args:
            problems: 问题列表，每个问题包含objective_func, bounds, x0等
            optimizer_name: 优化器名称，'auto'表示自动选择
            n_workers: 并行工作数
        """
        results = []
        
        for problem in problems:
            objective_func = problem['objective_func']
            bounds = problem['bounds']
            x0 = problem.get('x0', None)
            
            if optimizer_name == 'auto':
                # 根据问题特征自动选择优化器
                if len(bounds) <= 3:
                    optimizer_name = 'newton'
                elif len(bounds) <= 10:
                    optimizer_name = 'gradient_descent'
                else:
                    optimizer_name = 'genetic_algorithm'
            
            try:
                if optimizer_name in ['genetic_algorithm', 'particle_swarm']:
                    result = self.single_objective_optimize(
                        objective_func, optimizer_name, x0, bounds=bounds
                    )
                else:
                    result = self.single_objective_optimize(
                        objective_func, optimizer_name, x0, bounds=bounds
                    )
                
                results.append(result)
            except Exception as e:
                logger.error(f"批量优化问题失败: {e}")
                results.append(None)
        
        return results


def final_comprehensive_example():
    """最终综合示例"""
    print("=== J5优化工具最终综合示例 ===")
    
    # 创建最终优化工具
    opt_tools = FinalOptimizationTools()
    
    # 定义测试函数
    def rosenbrock(x):
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
    
    def sphere(x):
        return sum(xi**2 for xi in x)
    
    # 1. 元启发式算法测试
    print("\n1. 元启发式算法测试")
    x0 = np.array([0.0, 0.0])
    bounds = [(-2, 2), (-2, 2)]
    
    # 人工蜂群算法
    abc_result = opt_tools.artificial_bee_colony(sphere, bounds, max_iterations=100)
    print(f"人工蜂群算法: f={abc_result.fval:.6f}")
    
    # 鲸鱼优化算法
    wo_result = opt_tools.whale_optimization(rosenbrock, bounds, max_iterations=100)
    print(f"鲸鱼优化算法: f={wo_result.fval:.6f}")
    
    # 正弦余弦算法
    sca_result = opt_tools.sine_cosine_algorithm(sphere, bounds, max_iterations=100)
    print(f"正弦余弦算法: f={sca_result.fval:.6f}")
    
    # 2. 综合优化策略
    print("\n2. 综合优化策略")
    comprehensive_result = opt_tools.comprehensive_optimization(
        rosenbrock, bounds, x0, strategy='hybrid'
    )
    print(f"综合优化结果: f={comprehensive_result.fval:.6f}")
    
    # 3. 批量优化
    print("\n3. 批量优化")
    problems = [
        {'objective_func': sphere, 'bounds': bounds, 'x0': np.array([1.0, 1.0])},
        {'objective_func': rosenbrock, 'bounds': bounds, 'x0': np.array([-1.0, -1.0])}
    ]
    
    batch_results = opt_tools.batch_optimization(problems, optimizer_name='auto')
    for i, result in enumerate(batch_results):
        if result:
            print(f"批量问题 {i}: f={result.fval:.6f}")
    
    # 4. 敏感性分析
    print("\n4. 敏感性分析")
    sensitivity = opt_tools.sensitivity_analysis(sphere, x0, bounds)
    print(f"敏感性分析: 最敏感变量={sensitivity['most_sensitive_var']}")
    print(f"敏感性比率: {sensitivity['sensitivity_ratio']:.2f}")
    
    # 5. 景观分析
    print("\n5. 景观分析")
    landscape = opt_tools.landscape_analysis(sphere, bounds)
    print(f"景观类型: {landscape['landscape_type']}")
    print(f"粗糙度: {landscape['ruggedness']:.6f}")
    
    # 6. 收敛性分析
    print("\n6. 收敛性分析")
    all_results = [abc_result, wo_result, sca_result, comprehensive_result]
    convergence = opt_tools.convergence_analysis(all_results)
    print(f"成功率: {convergence['success_rate']:.2%}")
    print(f"平均迭代次数: {convergence['average_iterations']:.1f}")
    
    # 7. 配置管理
    print("\n7. 配置管理")
    custom_config = {
        'learning_rate': 0.001,
        'max_iterations': 500
    }
    opt_tools.config.add_optimizer_config('custom_gd', custom_config)
    
    custom_optimizer = opt_tools.get_optimizer_with_config(
        'custom_gd', bounds, verbose=False
    )
    print("自定义配置优化器创建成功")
    
    # 8. 性能基准测试
    print("\n8. 性能基准测试")
    benchmark_results = opt_tools.run_benchmark(
        ['gradient_descent', 'genetic_algorithm', 'differential_evolution'],
        ['sphere', 'rosenbrock'],
        dimensions=[2],
        n_runs=2,
        save_results=False
    )
    
    analysis = opt_tools.analyze_benchmark_results(benchmark_results)
    print(f"基准测试最佳优化器: {analysis['best_optimizer_overall']}")
    
    # 9. 保存和加载结果
    print("\n9. 保存和加载结果")
    opt_tools.save_results(comprehensive_result, "results/final_comprehensive.json")
    loaded_result = opt_tools.load_results("results/final_comprehensive.json")
    print(f"加载结果验证: f={loaded_result.fval:.6f}")
    
    # 10. 生成最终报告
    print("\n10. 生成最终报告")
    opt_tools.generate_report("final_optimization_report.txt")
    
    print("\n=== J5优化工具最终综合示例完成 ===")
    print("模块包含以下功能:")
    print("- 经典优化算法: 梯度下降、牛顿法、共轭梯度法")
    print("- 智能优化算法: 遗传算法、粒子群、模拟退火")
    print("- 高级优化算法: 差分进化、萤火虫、布谷鸟搜索")
    print("- 元启发式算法: 人工蜂群、鲸鱼优化、正弦余弦")
    print("- 约束优化: 拉格朗日乘数、惩罚函数、障碍函数")
    print("- 多目标优化: NSGA-II、SPEA2")
    print("- 贝叶斯优化: 高斯过程")
    print("- 并行和分布式优化")
    print("- 优化分析和可视化")
    print("- 基准测试和性能分析")
    print("- 配置管理和批量优化")


if __name__ == "__main__":
    final_comprehensive_example()