"""
优化算法库 - Optimization Algorithm Library

这个模块实现了一系列常用的优化算法，包括：
1. 梯度下降和其变种
2. 遗传算法
3. 粒子群优化(PSO)
4. 模拟退火算法
5. 蚁群算法
6. 差分进化
7. 贝叶斯优化
8. 多目标优化
9. 超参数自动调优

Author: AI Assistant
Date: 2025-11-05
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
import random
import copy
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings('ignore')


class OptimizationProblem:
    """优化问题基类"""
    
    def __init__(self, dimension: int, bounds: List[Tuple[float, float]], 
                 objective_func: Callable[[np.ndarray], float]):
        """
        初始化优化问题
        
        Args:
            dimension: 问题的维度
            bounds: 变量的边界 [(min1, max1), (min2, max2), ...]
            objective_func: 目标函数，接收numpy数组，返回函数值
        """
        self.dimension = dimension
        self.bounds = bounds
        self.objective_func = objective_func
    
    def evaluate(self, x: np.ndarray) -> float:
        """评估目标函数"""
        return self.objective_func(x)
    
    def random_point(self) -> np.ndarray:
        """生成随机点"""
        return np.array([np.random.uniform(bound[0], bound[1]) 
                        for bound in self.bounds])


class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self.history = []
        self.best_solution = None
        self.best_fitness = float('inf')
    
    @abstractmethod
    def optimize(self, max_iterations: int = 1000, **kwargs) -> Tuple[np.ndarray, float]:
        """执行优化"""
        pass
    
    def update_history(self, x: np.ndarray, fitness: float):
        """更新历史记录"""
        self.history.append((x.copy(), fitness))
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = x.copy()


class GradientDescentOptimizer(BaseOptimizer):
    """梯度下降优化器及其变种"""
    
    def __init__(self, problem: OptimizationProblem, method: str = 'adam',
                 learning_rate: float = 0.01, momentum: float = 0.9,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        初始化梯度下降优化器
        
        Args:
            problem: 优化问题
            method: 优化方法 ('gd', 'sgd', 'momentum', 'adam', 'rmsprop')
            learning_rate: 学习率
            momentum: 动量系数
            beta1, beta2: Adam的动量参数
            epsilon: 数值稳定性参数
        """
        super().__init__(problem)
        self.method = method
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    
    def numerical_gradient(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """数值计算梯度"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (self.problem.evaluate(x_plus) - self.problem.evaluate(x_minus)) / (2 * h)
        return grad
    
    def optimize(self, max_iterations: int = 1000, **kwargs) -> Tuple[np.ndarray, float]:
        """执行梯度下降优化"""
        # 初始化
        x = self.problem.random_point()
        v = np.zeros_like(x)  # 动量
        m = np.zeros_like(x)  # Adam动量1
        v_hat = np.zeros_like(x)  # Adam动量2
        t = 0  # 时间步
        
        for iteration in range(max_iterations):
            # 计算梯度
            grad = self.numerical_gradient(x)
            t += 1
            
            # 根据方法更新参数
            if self.method == 'gd':
                # 标准梯度下降
                x = x - self.learning_rate * grad
                
            elif self.method == 'momentum':
                # 动量梯度下降
                v = self.momentum * v - self.learning_rate * grad
                x = x + v
                
            elif self.method == 'adam':
                # Adam优化器
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v_hat + (1 - self.beta2) * (grad ** 2)
                
                # 偏差修正
                m_hat = m / (1 - self.beta1 ** t)
                v_hat = v / (1 - self.beta2 ** t)
                
                x = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
            elif self.method == 'rmsprop':
                # RMSprop
                v = self.momentum * v + (1 - self.momentum) * (grad ** 2)
                x = x - self.learning_rate * grad / (np.sqrt(v) + self.epsilon)
            
            # 边界约束
            x = np.clip(x, [bound[0] for bound in self.problem.bounds], 
                       [bound[1] for bound in self.problem.bounds])
            
            # 评估和记录
            fitness = self.problem.evaluate(x)
            self.update_history(x, fitness)
        
        return self.best_solution, self.best_fitness


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """遗传算法优化器"""
    
    def __init__(self, problem: OptimizationProblem, population_size: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 tournament_size: int = 3, elitism: bool = True):
        """
        初始化遗传算法优化器
        
        Args:
            problem: 优化问题
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            tournament_size: 锦标赛大小
            elitism: 是否精英保留
        """
        super().__init__(problem)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
    
    def create_individual(self) -> np.ndarray:
        """创建个体"""
        return self.problem.random_point()
    
    def create_population(self) -> np.ndarray:
        """创建种群"""
        return np.array([self.create_individual() for _ in range(self.population_size)])
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """交叉操作"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 单点交叉
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                # 高斯变异
                sigma = (self.problem.bounds[i][1] - self.problem.bounds[i][0]) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                # 边界约束
                mutated[i] = np.clip(mutated[i], self.problem.bounds[i][0], self.problem.bounds[i][1])
        
        return mutated
    
    def tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """锦标赛选择"""
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]
    
    def optimize(self, max_iterations: int = 1000, **kwargs) -> Tuple[np.ndarray, float]:
        """执行遗传算法优化"""
        # 初始化种群
        population = self.create_population()
        
        for iteration in range(max_iterations):
            # 评估种群
            fitness = np.array([self.problem.evaluate(ind) for ind in population])
            
            # 更新历史
            best_idx = np.argmin(fitness)
            self.update_history(population[best_idx], fitness[best_idx])
            
            # 创建新一代
            new_population = []
            
            # 精英保留
            if self.elitism:
                elite_size = max(1, self.population_size // 10)
                elite_indices = np.argsort(fitness)[:elite_size]
                new_population.extend([population[i].copy() for i in elite_indices])
            
            # 生成剩余个体
            while len(new_population) < self.population_size:
                # 选择
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 更新种群
            population = np.array(new_population[:self.population_size])
        
        return self.best_solution, self.best_fitness


class ParticleSwarmOptimizer(BaseOptimizer):
    """粒子群优化算法"""
    
    def __init__(self, problem: OptimizationProblem, num_particles: int = 30,
                 inertia_weight: float = 0.9, cognitive_coefficient: float = 2.0,
                 social_coefficient: float = 2.0):
        """
        初始化PSO优化器
        
        Args:
            problem: 优化问题
            num_particles: 粒子数量
            inertia_weight: 惯性权重
            cognitive_coefficient: 认知系数
            social_coefficient: 社会系数
        """
        super().__init__(problem)
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
    
    def optimize(self, max_iterations: int = 1000, **kwargs) -> Tuple[np.ndarray, float]:
        """执行PSO优化"""
        # 初始化粒子
        positions = np.array([self.problem.random_point() for _ in range(self.num_particles)])
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.problem.dimension))
        
        # 个体最优和全局最优
        personal_best_positions = positions.copy()
        personal_best_fitness = np.array([self.problem.evaluate(pos) for pos in positions])
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        for iteration in range(max_iterations):
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = np.random.random(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                               self.cognitive_coefficient * r1 * (personal_best_positions[i] - positions[i]) +
                               self.social_coefficient * r2 * (global_best_position - positions[i]))
                
                # 更新位置
                positions[i] += velocities[i]
                
                # 边界约束
                positions[i] = np.clip(positions[i], 
                                     [bound[0] for bound in self.problem.bounds],
                                     [bound[1] for bound in self.problem.bounds])
                
                # 评估
                fitness = self.problem.evaluate(positions[i])
                
                # 更新个体最优
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = positions[i].copy()
                    
                    # 更新全局最优
                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = positions[i].copy()
            
            # 记录历史
            self.update_history(global_best_position, global_best_fitness)
        
        return self.best_solution, self.best_fitness


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """模拟退火算法"""
    
    def __init__(self, problem: OptimizationProblem, initial_temperature: float = 1000.0,
                 cooling_rate: float = 0.95, min_temperature: float = 1e-8,
                 neighborhood_size: float = 0.1):
        """
        初始化模拟退火优化器
        
        Args:
            problem: 优化问题
            initial_temperature: 初始温度
            cooling_rate: 降温速率
            min_temperature: 最低温度
            neighborhood_size: 邻域大小
        """
        super().__init__(problem)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.neighborhood_size = neighborhood_size
    
    def generate_neighbor(self, x: np.ndarray) -> np.ndarray:
        """生成邻域解"""
        neighbor = x.copy()
        for i in range(len(neighbor)):
            # 在当前解附近随机扰动
            range_size = (self.problem.bounds[i][1] - self.problem.bounds[i][0]) * self.neighborhood_size
            neighbor[i] += np.random.uniform(-range_size, range_size)
            # 边界约束
            neighbor[i] = np.clip(neighbor[i], self.problem.bounds[i][0], self.problem.bounds[i][1])
        
        return neighbor
    
    def optimize(self, max_iterations: int = 1000, **kwargs) -> Tuple[np.ndarray, float]:
        """执行模拟退火优化"""
        # 初始化
        current_solution = self.problem.random_point()
        current_fitness = self.problem.evaluate(current_solution)
        temperature = self.initial_temperature
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        for iteration in range(max_iterations):
            # 生成邻域解
            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness = self.problem.evaluate(neighbor)
            
            # 接受准则
            delta = neighbor_fitness - current_fitness
            if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                # 更新最优解
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_solution = current_solution.copy()
            
            # 降温
            temperature *= self.cooling_rate
            if temperature < self.min_temperature:
                break
            
            # 记录历史
            self.update_history(best_solution, best_fitness)
        
        return self.best_solution, self.best_fitness


class AntColonyOptimizer(BaseOptimizer):
    """蚁群算法"""
    
    def __init__(self, problem: OptimizationProblem, num_ants: int = 30,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
                 Q: float = 100.0):
        """
        初始化蚁群优化器
        
        Args:
            problem: 优化问题
            num_ants: 蚂蚁数量
            alpha: 信息素重要程度
            beta: 启发式信息重要程度
            rho: 信息素挥发系数
            Q: 信息素常数
        """
        super().__init__(problem)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # 初始化信息素矩阵
        self.num_variables = problem.dimension
        self.pheromone = np.ones((self.num_variables, 2))  # 每个变量两个边界的信息素
    
    def construct_solution(self) -> np.ndarray:
        """构建解"""
        solution = np.zeros(self.num_variables)
        
        for i in range(self.num_variables):
            # 计算选择概率
            probabilities = np.zeros(2)
            for j in range(2):
                probabilities[j] = (self.pheromone[i, j] ** self.alpha) * \
                                 ((1.0 / (self.problem.bounds[i][j] - self.problem.bounds[i][0] + 1e-10)) ** self.beta)
            
            # 归一化
            probabilities /= np.sum(probabilities)
            
            # 轮盘赌选择
            choice = np.random.choice(2, p=probabilities)
            solution[i] = np.random.uniform(self.problem.bounds[i][choice], 
                                          self.problem.bounds[i][1-choice])
        
        return solution
    
    def update_pheromone(self, solutions: List[np.ndarray], fitnesses: np.ndarray):
        """更新信息素"""
        # 挥发
        self.pheromone *= (1 - self.rho)
        
        # 增强
        for i, (solution, fitness) in enumerate(zip(solutions, fitnesses)):
            # 找到每个变量对应的边界
            for j in range(self.num_variables):
                if solution[j] < (self.problem.bounds[j][0] + self.problem.bounds[j][1]) / 2:
                    self.pheromone[j, 0] += self.Q / fitness
                else:
                    self.pheromone[j, 1] += self.Q / fitness
    
    def optimize(self, max_iterations: int = 1000, **kwargs) -> Tuple[np.ndarray, float]:
        """执行蚁群优化"""
        for iteration in range(max_iterations):
            # 构建解
            solutions = []
            fitnesses = []
            
            for _ in range(self.num_ants):
                solution = self.construct_solution()
                fitness = self.problem.evaluate(solution)
                solutions.append(solution)
                fitnesses.append(fitness)
            
            fitnesses = np.array(fitnesses)
            
            # 更新历史
            best_idx = np.argmin(fitnesses)
            self.update_history(solutions[best_idx], fitnesses[best_idx])
            
            # 更新信息素
            self.update_pheromone(solutions, fitnesses)
        
        return self.best_solution, self.best_fitness


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """差分进化算法"""
    
    def __init__(self, problem: OptimizationProblem, population_size: int = 50,
                 differential_weight: float = 0.5, crossover_rate: float = 0.7):
        """
        初始化差分进化优化器
        
        Args:
            problem: 优化问题
            population_size: 种群大小
            differential_weight: 差分权重F
            crossover_rate: 交叉率CR
        """
        super().__init__(problem)
        self.population_size = population_size
        self.differential_weight = differential_weight
        self.crossover_rate = crossover_rate
    
    def create_population(self) -> np.ndarray:
        """创建种群"""
        return np.array([self.problem.random_point() for _ in range(self.population_size)])
    
    def mutate(self, population: np.ndarray, index: int) -> np.ndarray:
        """变异操作"""
        # 随机选择三个不同的个体
        indices = np.random.choice([i for i in range(len(population)) if i != index], 3, replace=False)
        a, b, c = population[indices]
        
        # DE/rand/1策略
        mutant = a + self.differential_weight * (b - c)
        
        # 边界约束
        mutant = np.clip(mutant, [bound[0] for bound in self.problem.bounds],
                        [bound[1] for bound in self.problem.bounds])
        
        return mutant
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """交叉操作"""
        trial = target.copy()
        
        # 随机选择交叉维度
        crossover_dim = np.random.randint(0, len(target))
        
        for i in range(len(target)):
            if i == crossover_dim or np.random.random() < self.crossover_rate:
                trial[i] = mutant[i]
        
        return trial
    
    def optimize(self, max_iterations: int = 1000, **kwargs) -> Tuple[np.ndarray, float]:
        """执行差分进化优化"""
        # 初始化种群
        population = self.create_population()
        fitness = np.array([self.problem.evaluate(ind) for ind in population])
        
        for iteration in range(max_iterations):
            for i in range(self.population_size):
                # 变异
                mutant = self.mutate(population, i)
                
                # 交叉
                trial = self.crossover(population[i], mutant)
                
                # 选择
                trial_fitness = self.problem.evaluate(trial)
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # 更新历史
                    if trial_fitness < self.best_fitness:
                        self.update_history(trial, trial_fitness)
            
            # 记录当前代最优
            best_idx = np.argmin(fitness)
            self.update_history(population[best_idx], fitness[best_idx])
        
        return self.best_solution, self.best_fitness


class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化"""
    
    def __init__(self, problem: OptimizationProblem, n_initial_points: int = 5,
                 acquisition_function: str = 'ei'):
        """
        初始化贝叶斯优化器
        
        Args:
            problem: 优化问题
            n_initial_points: 初始采样点数
            acquisition_function: 采集函数 ('ei', 'ucb', 'pi')
        """
        super().__init__(problem)
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        
        # 初始化高斯过程
        kernel = ConstantKernel(1.0) * RBF(1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42)
        
        self.X_sample = []
        self.y_sample = []
    
    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """期望改进采集函数"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        f_best = min(self.y_sample)
        
        with np.errstate(divide='warn'):
            imp = mu - f_best - xi
            Z = imp / sigma
            ei = imp * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def upper_confidence_bound(self, X: np.ndarray, kappa: float = 2.576) -> np.ndarray:
        """上置信界采集函数"""
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + kappa * sigma
    
    def probability_of_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """改进概率采集函数"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        f_best = min(self.y_sample)
        
        with np.errstate(divide='warn'):
            imp = mu - f_best - xi
            Z = imp / sigma
            pi = self._normal_cdf(Z)
            pi[sigma == 0.0] = 0.0
        
        return pi
    
    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """标准正态分布累积分布函数"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """标准正态分布概率密度函数"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def optimize(self, max_iterations: int = 100, **kwargs) -> Tuple[np.ndarray, float]:
        """执行贝叶斯优化"""
        # 初始采样
        for _ in range(self.n_initial_points):
            x_sample = self.problem.random_point()
            y_sample = self.problem.evaluate(x_sample)
            
            self.X_sample.append(x_sample)
            self.y_sample.append(y_sample)
            
            self.update_history(x_sample, y_sample)
        
        # 优化循环
        for iteration in range(max_iterations):
            # 更新高斯过程
            self.gp.fit(np.array(self.X_sample), np.array(self.y_sample))
            
            # 生成候选点
            n_candidates = 1000
            X_candidates = np.array([self.problem.random_point() for _ in range(n_candidates)])
            
            # 计算采集函数值
            if self.acquisition_function == 'ei':
                acq_values = self.expected_improvement(X_candidates)
            elif self.acquisition_function == 'ucb':
                acq_values = self.upper_confidence_bound(X_candidates)
            elif self.acquisition_function == 'pi':
                acq_values = self.probability_of_improvement(X_candidates)
            else:
                raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
            
            # 选择最优候选点
            next_x = X_candidates[np.argmax(acq_values)]
            
            # 评估目标函数
            next_y = self.problem.evaluate(next_x)
            
            # 更新样本
            self.X_sample.append(next_x)
            self.y_sample.append(next_y)
            
            # 更新历史
            self.update_history(next_x, next_y)
        
        return self.best_solution, self.best_fitness


class MultiObjectiveProblem:
    """多目标优化问题"""
    
    def __init__(self, objectives: List[Callable[[np.ndarray], float]], 
                 bounds: List[Tuple[float, float]]):
        """
        初始化多目标优化问题
        
        Args:
            objectives: 目标函数列表
            bounds: 变量边界
        """
        self.objectives = objectives
        self.bounds = bounds
        self.dimension = len(bounds)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """评估所有目标函数"""
        return np.array([obj(x) for obj in self.objectives])
    
    def random_point(self) -> np.ndarray:
        """生成随机点"""
        return np.array([np.random.uniform(bound[0], bound[1]) 
                        for bound in self.bounds])


class NSGA2Optimizer:
    """NSGA-II多目标优化算法"""
    
    def __init__(self, problem: MultiObjectiveProblem, population_size: int = 100,
                 crossover_prob: float = 0.9, mutation_prob: float = 0.1):
        """
        初始化NSGA-II优化器
        
        Args:
            problem: 多目标优化问题
            population_size: 种群大小
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
        """
        self.problem = problem
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        self.history = []
        self.pareto_front = []
    
    def dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断是否支配"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def fast_non_dominated_sort(self, population: np.ndarray, objectives: np.ndarray) -> List[List[int]]:
        """快速非支配排序"""
        n = len(population)
        domination_count = np.zeros(n)  # 支配该个体的个体数
        dominated_solutions = [[] for _ in range(n)]  # 该个体支配的个体集合
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # 移除最后一个空前沿
    
    def crowding_distance(self, objectives: np.ndarray, front: List[int]) -> np.ndarray:
        """拥挤距离计算"""
        if len(front) <= 2:
            return np.full(len(front), np.inf)
        
        distances = np.zeros(len(front))
        
        for m in range(objectives.shape[1]):
            front_objs = objectives[front, m]
            sorted_indices = np.argsort(front_objs)
            
            # 边界点设为无穷大
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # 计算中间点的拥挤距离
            obj_range = front_objs[sorted_indices[-1]] - front_objs[sorted_indices[0]]
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distance = (front_objs[sorted_indices[i + 1]] - front_objs[sorted_indices[i - 1]]) / obj_range
                    distances[sorted_indices[i]] += distance
        
        return distances
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """交叉操作"""
        if np.random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        # 模拟二进制交叉
        eta_c = 20
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() <= 0.5:
                y1, y2 = parent1[i], parent2[i]
                if abs(y1 - y2) > 1e-14:
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    rand = np.random.random()
                    beta = 1.0 + (2.0 * (y1 - self.problem.bounds[i][0]) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta_c + 1.0))
                    
                    if rand <= (1.0 / alpha):
                        betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                    
                    c1 = 0.5 * ((y1 + y2) - betaq * abs(y2 - y1))
                    
                    beta = 1.0 + (2.0 * (self.problem.bounds[i][1] - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta_c + 1.0))
                    
                    if rand <= (1.0 / alpha):
                        betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                    
                    c2 = 0.5 * ((y1 + y2) + betaq * abs(y2 - y1))
                    
                    c1 = np.clip(c1, self.problem.bounds[i][0], self.problem.bounds[i][1])
                    c2 = np.clip(c2, self.problem.bounds[i][0], self.problem.bounds[i][1])
                    
                    offspring1[i] = c1
                    offspring2[i] = c2
        
        return offspring1, offspring2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() <= self.mutation_prob:
                eta_m = 20
                y = individual[i]
                delta1 = (y - self.problem.bounds[i][0]) / (self.problem.bounds[i][1] - self.problem.bounds[i][0])
                delta2 = (self.problem.bounds[i][1] - y) / (self.problem.bounds[i][1] - self.problem.bounds[i][0])
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (self.problem.bounds[i][1] - self.problem.bounds[i][0])
                y = np.clip(y, self.problem.bounds[i][0], self.problem.bounds[i][1])
                mutated[i] = y
        
        return mutated
    
    def optimize(self, max_iterations: int = 100, **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """执行NSGA-II优化"""
        # 初始化种群
        population = np.array([self.problem.random_point() for _ in range(self.population_size)])
        
        for iteration in range(max_iterations):
            # 评估种群
            objectives = np.array([self.problem.evaluate(ind) for ind in population])
            
            # 非支配排序
            fronts = self.fast_non_dominated_sort(population, objectives)
            
            # 计算拥挤距离
            crowding_distances = []
            for front in fronts:
                if len(front) > 0:
                    distances = self.crowding_distance(objectives, front)
                    crowding_distances.extend(distances)
            
            # 选择、交叉、变异生成新种群
            offspring = []
            while len(offspring) < self.population_size:
                # 锦标赛选择
                parent1_idx = np.random.randint(0, len(population))
                parent2_idx = np.random.randint(0, len(population))
                
                # 交叉
                child1, child2 = self.crossover(population[parent1_idx], population[parent2_idx])
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.extend([child1, child2])
            
            offspring = np.array(offspring[:self.population_size])
            
            # 合并父代和子代
            combined_population = np.vstack([population, offspring])
            combined_objectives = np.vstack([objectives, 
                                           np.array([self.problem.evaluate(ind) for ind in offspring])])
            
            # 选择新的种群
            fronts = self.fast_non_dominated_sort(combined_population, combined_objectives)
            
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    # 计算拥挤距离并选择
                    remaining = self.population_size - len(new_population)
                    front_distances = self.crowding_distance(combined_objectives, front)
                    selected_indices = np.argsort(front_distances)[-remaining:]
                    new_population.extend([front[i] for i in selected_indices])
                    break
            
            population = combined_population[new_population]
            objectives = combined_objectives[new_population]
            
            # 记录历史
            self.history.append((population.copy(), objectives.copy()))
            
            # 更新帕累托前沿
            if len(fronts) > 0:
                pareto_indices = fronts[0]
                # 确保索引在有效范围内，并使用合并后的种群
                valid_indices = [i for i in pareto_indices if i < len(combined_population)]
                self.pareto_front = [(combined_population[i], combined_objectives[i]) for i in valid_indices]
        
        # 返回帕累托前沿
        pareto_solutions = [sol for sol, obj in self.pareto_front]
        pareto_objectives = [obj for sol, obj in self.pareto_front]
        
        return pareto_solutions, pareto_objectives


class HyperparameterTuner:
    """超参数自动调优器"""
    
    def __init__(self, objective_func: Callable, param_space: Dict[str, Any]):
        """
        初始化超参数调优器
        
        Args:
            objective_func: 目标函数，接收参数字典，返回性能分数
            param_space: 参数搜索空间
        """
        self.objective_func = objective_func
        self.param_space = param_space
        self.optimization_problem = None
        self.optimizer = None
    
    def _create_optimization_problem(self) -> OptimizationProblem:
        """创建优化问题"""
        # 提取参数边界
        bounds = []
        param_names = []
        
        for param_name, param_config in self.param_space.items():
            param_names.append(param_name)
            if isinstance(param_config, dict):
                if 'min' in param_config and 'max' in param_config:
                    bounds.append((param_config['min'], param_config['max']))
                elif 'values' in param_config:
                    # 离散参数，转换为连续空间
                    values = param_config['values']
                    bounds.append((0, len(values) - 1))
                else:
                    raise ValueError(f"Invalid parameter config for {param_name}")
            else:
                raise ValueError(f"Invalid parameter config for {param_name}")
        
        def objective_wrapper(x):
            # 将连续值转换为参数
            params = {}
            for i, param_name in enumerate(param_names):
                if isinstance(self.param_space[param_name], dict) and 'values' in self.param_space[param_name]:
                    # 离散参数
                    idx = int(round(x[i]))
                    values = self.param_space[param_name]['values']
                    params[param_name] = values[max(0, min(idx, len(values) - 1))]
                else:
                    # 连续参数
                    params[param_name] = x[i]
            
            return self.objective_func(params)
        
        return OptimizationProblem(len(bounds), bounds, objective_wrapper)
    
    def tune(self, algorithm: str = 'bayesian', max_iterations: int = 100, **kwargs) -> Dict[str, Any]:
        """
        执行超参数调优
        
        Args:
            algorithm: 优化算法 ('bayesian', 'random', 'grid', 'genetic', 'pso')
            max_iterations: 最大迭代次数
            **kwargs: 算法特定参数
        
        Returns:
            最佳参数配置
        """
        # 创建优化问题
        self.optimization_problem = self._create_optimization_problem()
        
        # 选择优化器
        if algorithm == 'bayesian':
            self.optimizer = BayesianOptimizer(self.optimization_problem, **kwargs)
        elif algorithm == 'genetic':
            self.optimizer = GeneticAlgorithmOptimizer(self.optimization_problem, **kwargs)
        elif algorithm == 'pso':
            self.optimizer = ParticleSwarmOptimizer(self.optimization_problem, **kwargs)
        elif algorithm == 'random':
            # 随机搜索
            best_params = None
            best_score = float('inf')
            
            for _ in range(max_iterations):
                params = {}
                for param_name, param_config in self.param_space.items():
                    if isinstance(param_config, dict) and 'values' in param_config:
                        params[param_name] = random.choice(param_config['values'])
                    elif isinstance(param_config, dict) and 'min' in param_config and 'max' in param_config:
                        params[param_name] = random.uniform(param_config['min'], param_config['max'])
                
                score = self.objective_func(params)
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
            
            return best_params
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # 执行优化
        best_solution, best_fitness = self.optimizer.optimize(max_iterations, **kwargs)
        
        # 转换回参数
        best_params = {}
        param_names = list(self.param_space.keys())
        
        for i, param_name in enumerate(param_names):
            if isinstance(self.param_space[param_name], dict) and 'values' in self.param_space[param_name]:
                # 离散参数
                idx = int(round(best_solution[i]))
                values = self.param_space[param_name]['values']
                best_params[param_name] = values[max(0, min(idx, len(values) - 1))]
            else:
                # 连续参数
                best_params[param_name] = best_solution[i]
        
        return best_params


class OptimizationAlgorithmLibrary:
    """优化算法库主类"""
    
    def __init__(self):
        """初始化优化算法库"""
        self.optimizers = {
            'gradient_descent': GradientDescentOptimizer,
            'genetic': GeneticAlgorithmOptimizer,
            'pso': ParticleSwarmOptimizer,
            'simulated_annealing': SimulatedAnnealingOptimizer,
            'ant_colony': AntColonyOptimizer,
            'differential_evolution': DifferentialEvolutionOptimizer,
            'bayesian': BayesianOptimizer,
            'nsga2': NSGA2Optimizer
        }
    
    def create_problem(self, objective_func: Callable, bounds: List[Tuple[float, float]], 
                      is_multi_objective: bool = False, 
                      objectives: List[Callable[[np.ndarray], float]] = None) -> Union[OptimizationProblem, MultiObjectiveProblem]:
        """
        创建优化问题
        
        Args:
            objective_func: 目标函数
            bounds: 变量边界
            is_multi_objective: 是否多目标
            objectives: 多目标函数列表
        
        Returns:
            优化问题对象
        """
        if is_multi_objective:
            if objectives is None:
                raise ValueError("Multi-objective problems require objectives list")
            return MultiObjectiveProblem(objectives, bounds)
        else:
            return OptimizationProblem(len(bounds), bounds, objective_func)
    
    def optimize(self, problem: Union[OptimizationProblem, MultiObjectiveProblem], 
                algorithm: str, max_iterations: int = 1000, **kwargs) -> Any:
        """
        执行优化
        
        Args:
            problem: 优化问题
            algorithm: 算法名称
            max_iterations: 最大迭代次数
            **kwargs: 算法特定参数
        
        Returns:
            优化结果
        """
        if algorithm not in self.optimizers:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if isinstance(problem, MultiObjectiveProblem):
            if algorithm != 'nsga2':
                raise ValueError(f"Multi-objective problems require nsga2 algorithm")
            optimizer = self.optimizers[algorithm](problem, **kwargs)
            return optimizer.optimize(max_iterations, **kwargs)
        else:
            optimizer = self.optimizers[algorithm](problem, **kwargs)
            return optimizer.optimize(max_iterations, **kwargs)
    
    def hyperparameter_tune(self, objective_func: Callable, param_space: Dict[str, Any],
                          algorithm: str = 'bayesian', max_iterations: int = 100, **kwargs) -> Dict[str, Any]:
        """
        超参数调优
        
        Args:
            objective_func: 目标函数
            param_space: 参数搜索空间
            algorithm: 优化算法
            max_iterations: 最大迭代次数
            **kwargs: 算法特定参数
        
        Returns:
            最佳参数配置
        """
        tuner = HyperparameterTuner(objective_func, param_space)
        return tuner.tune(algorithm, max_iterations, **kwargs)
    
    def plot_convergence(self, optimizer: BaseOptimizer, save_path: str = None):
        """
        绘制收敛曲线
        
        Args:
            optimizer: 优化器实例
            save_path: 保存路径
        """
        if not optimizer.history:
            print("No history to plot")
            return
        
        fitness_history = [fitness for _, fitness in optimizer.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title('优化收敛曲线')
        plt.grid(True)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_pareto_front(self, pareto_solutions: List[np.ndarray], 
                         pareto_objectives: List[np.ndarray], save_path: str = None):
        """
        绘制帕累托前沿
        
        Args:
            pareto_solutions: 帕累托解
            pareto_objectives: 帕累托目标值
            save_path: 保存路径
        """
        if len(pareto_objectives[0]) == 2:
            # 2D帕累托前沿
            plt.figure(figsize=(10, 6))
            objectives_array = np.array(pareto_objectives)
            plt.scatter(objectives_array[:, 0], objectives_array[:, 1], c='red', s=50, alpha=0.7)
            plt.xlabel('目标函数1')
            plt.ylabel('目标函数2')
            plt.title('帕累托前沿')
            plt.grid(True)
        elif len(pareto_objectives[0]) == 3:
            # 3D帕累托前沿
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            objectives_array = np.array(pareto_objectives)
            ax.scatter(objectives_array[:, 0], objectives_array[:, 1], objectives_array[:, 2], 
                      c='red', s=50, alpha=0.7)
            ax.set_xlabel('目标函数1')
            ax.set_ylabel('目标函数2')
            ax.set_zlabel('目标函数3')
            ax.set_title('帕累托前沿')
        else:
            print("Can only plot 2D or 3D Pareto fronts")
            return
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def test_optimization_algorithms():
    """测试各种优化算法"""
    
    # 定义测试函数
    def sphere_function(x):
        """球面函数 f(x) = sum(x_i^2)"""
        return np.sum(x**2)
    
    def rosenbrock_function(x):
        """Rosenbrock函数"""
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
    
    def rastrigin_function(x):
        """Rastrigin函数"""
        A = 10
        n = len(x)
        return A * n + sum(x[i]**2 - A * np.cos(2 * np.pi * x[i]) for i in range(n))
    
    # 创建优化算法库
    library = OptimizationAlgorithmLibrary()
    
    print("=== 优化算法库测试 ===\n")
    
    # 测试单目标优化
    print("1. 单目标优化测试")
    print("-" * 30)
    
    # 创建问题
    bounds = [(-5, 5), (-5, 5)]
    problem = library.create_problem(sphere_function, bounds)
    
    # 测试不同算法
    algorithms = ['gradient_descent', 'genetic', 'pso', 'simulated_annealing', 
                  'differential_evolution', 'bayesian']
    
    results = {}
    
    for algo in algorithms:
        print(f"测试 {algo} 算法...")
        try:
            if algo == 'gradient_descent':
                result = library.optimize(problem, algo, max_iterations=500, 
                                        method='adam', learning_rate=0.01)
            elif algo == 'bayesian':
                result = library.optimize(problem, algo, max_iterations=10, 
                                        n_initial_points=3)
            else:
                result = library.optimize(problem, algo, max_iterations=200)
            
            best_solution, best_fitness = result
            results[algo] = (best_solution, best_fitness)
            print(f"  最优解: {best_solution}")
            print(f"  最优值: {best_fitness:.6f}")
            print()
            
        except Exception as e:
            print(f"  错误: {e}")
            print()
    
    # 测试多目标优化
    print("2. 多目标优化测试")
    print("-" * 30)
    
    # 定义多目标函数
    def objective1(x):
        return (x[0] - 1)**2 + (x[1] - 1)**2
    
    def objective2(x):
        return (x[0] + 1)**2 + (x[1] + 1)**2
    
    multi_problem = library.create_problem(None, bounds, is_multi_objective=True,
                                         objectives=[objective1, objective2])
    
    print("测试 NSGA-II 算法...")
    pareto_solutions, pareto_objectives = library.optimize(multi_problem, 'nsga2', 
                                                         max_iterations=50, population_size=50)
    print(f"找到 {len(pareto_solutions)} 个帕累托最优解")
    print()
    
    # 测试超参数调优
    print("3. 超参数调优测试")
    print("-" * 30)
    
    def simple_model_accuracy(params):
        """模拟模型准确率函数"""
        # 模拟一个简单的机器学习模型调参
        learning_rate = params.get('learning_rate', 0.01)
        batch_size = params.get('batch_size', 32)
        hidden_size = params.get('hidden_size', 64)
        
        # 模拟准确率计算（实际应用中这里应该是真实模型训练）
        accuracy = 0.5 + 0.4 * np.exp(-learning_rate * 0.1) * np.exp(-batch_size * 0.01)
        accuracy += 0.1 * np.tanh(hidden_size / 100)
        accuracy += np.random.normal(0, 0.02)  # 添加噪声
        
        return -accuracy  # 最小化负准确率
    
    param_space = {
        'learning_rate': {'min': 0.001, 'max': 0.1},
        'batch_size': {'values': [16, 32, 64, 128]},
        'hidden_size': {'min': 32, 'max': 256}
    }
    
    print("测试贝叶斯优化调参...")
    best_params = library.hyperparameter_tune(simple_model_accuracy, param_space, 
                                            algorithm='bayesian', max_iterations=20)
    print(f"最佳参数: {best_params}")
    
    print("\n测试随机搜索调参...")
    best_params_random = library.hyperparameter_tune(simple_model_accuracy, param_space, 
                                                   algorithm='random', max_iterations=20)
    print(f"最佳参数: {best_params_random}")
    print()
    
    # 绘制结果
    print("4. 结果可视化")
    print("-" * 30)
    
    # 选择一个优化器绘制收敛曲线
    if 'genetic' in results:
        problem_ga = library.create_problem(sphere_function, bounds)
        optimizer_ga = GeneticAlgorithmOptimizer(problem_ga, population_size=50)
        optimizer_ga.optimize(max_iterations=100)
        
        print("绘制遗传算法收敛曲线...")
        library.plot_convergence(optimizer_ga)
    
    # 绘制帕累托前沿
    if pareto_solutions:
        print("绘制帕累托前沿...")
        library.plot_pareto_front(pareto_solutions, pareto_objectives)
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    # 运行测试
    test_optimization_algorithms()