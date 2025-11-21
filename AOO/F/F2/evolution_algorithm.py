"""
进化算法模块

实现多种进化算法：
- 遗传算法 (Genetic Algorithm)
- 进化策略 (Evolution Strategies)
- 差分进化 (Differential Evolution)
- 粒子群优化 (Particle Swarm Optimization)
- 多目标进化算法 (NSGA-II)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import random
import logging
from copy import deepcopy
from datetime import datetime
from .StrategyLearner import StrategyType, BaseStrategy, LearningContext, StrategyPerformance

logger = logging.getLogger(__name__)

class Individual:
    """个体类"""
    
    def __init__(self, genes: np.ndarray, fitness: float = 0.0):
        self.genes = genes
        self.fitness = fitness
        self.age = 0
        self.generation_born = 0
    
    def copy(self):
        """创建副本"""
        new_individual = Individual(self.genes.copy(), self.fitness)
        new_individual.age = self.age
        new_individual.generation_born = self.generation_born
        return new_individual
    
    def mutate(self, mutation_rate: float, mutation_strength: float):
        """变异操作"""
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] += np.random.normal(0, mutation_strength)
    
    def crossover(self, other: 'Individual') -> Tuple['Individual', 'Individual']:
        """交叉操作"""
        if len(self.genes) != len(other.genes):
            raise ValueError("个体基因长度不匹配")
        
        # 单点交叉
        crossover_point = random.randint(1, len(self.genes) - 1)
        
        child1_genes = np.concatenate([self.genes[:crossover_point], other.genes[crossover_point:]])
        child2_genes = np.concatenate([other.genes[:crossover_point], self.genes[crossover_point:]])
        
        child1 = Individual(child1_genes, 0.0)
        child2 = Individual(child2_genes, 0.0)
        
        return child1, child2

class GeneticAlgorithm(BaseStrategy):
    """遗传算法"""
    
    def __init__(self, strategy_id: str, gene_length: int, 
                 population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, elite_size: int = 5):
        super().__init__(strategy_id, StrategyType.EVOLUTION)
        
        self.gene_length = gene_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        self.population = []
        self.generation = 0
        self.fitness_history = []
        self.fitness_function = None
        
        self._initialize_population()
    
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """遗传算法学习过程"""
        try:
            if not self.fitness_function:
                # 创建默认适应度函数
                self.fitness_function = self._create_default_fitness_function(context)
            
            # 评估种群适应度
            for individual in self.population:
                individual.fitness = self.fitness_function(individual.genes, context)
                individual.age += 1
            
            # 记录适应度历史
            fitnesses = [ind.fitness for ind in self.population]
            self.fitness_history.append({
                'generation': self.generation,
                'best_fitness': max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'worst_fitness': min(fitnesses),
                'diversity': self._calculate_diversity()
            })
            
            # 选择、交叉、变异
            new_population = []
            
            # 精英保留
            sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            for i in range(self.elite_size):
                new_population.append(sorted_population[i].copy())
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 锦标赛选择
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = parent1.crossover(parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 变异
                child1.mutate(self.mutation_rate, 0.1)
                child2.mutate(self.mutation_rate, 0.1)
                
                child1.generation_born = self.generation + 1
                child2.generation_born = self.generation + 1
                
                new_population.extend([child1, child2])
            
            # 更新种群
            self.population = new_population[:self.population_size]
            self.generation += 1
            
            # 返回学习结果
            best_individual = sorted(self.population, key=lambda x: x.fitness, reverse=True)[0]
            
            return {
                'generation': self.generation,
                'best_fitness': best_individual.fitness,
                'best_genes': best_individual.genes.tolist(),
                'population_diversity': self._calculate_diversity(),
                'convergence_rate': self._calculate_convergence(),
                'fitness_history': self.fitness_history[-5:]  # 最近5代
            }
            
        except Exception as e:
            logger.error(f"遗传算法学习出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """基于最优个体进行预测"""
        try:
            if not self.population:
                return {'action': 'hold', 'confidence': 0.0}
            
            # 选择最优个体
            best_individual = max(self.population, key=lambda x: x.fitness)
            
            # 基于基因解码动作
            action = self._decode_genes(best_individual.genes, state)
            confidence = min(1.0, best_individual.fitness)
            
            return {
                'action': action,
                'confidence': confidence,
                'genes': best_individual.genes.tolist(),
                'fitness': best_individual.fitness,
                'generation': self.generation
            }
            
        except Exception as e:
            logger.error(f"遗传算法预测出错: {e}")
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
            
            # 进化算法关注长期表现和多样性
            if performance.return_rate > 0:
                self.state.success_rate = (self.state.success_rate * 0.9 + 0.1) if self.state.usage_count > 0 else 1.0
            else:
                self.state.success_rate = (self.state.success_rate * 0.9) if self.state.usage_count > 0 else 0.0
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
            # 基于性能调整进化参数
            if self.state.success_rate > 0.7:
                self.mutation_rate = max(0.01, self.mutation_rate * 0.95)  # 减少变异
            elif self.state.success_rate < 0.3:
                self.mutation_rate = min(0.2, self.mutation_rate * 1.05)   # 增加变异
                
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def set_fitness_function(self, fitness_func: Callable):
        """设置适应度函数"""
        self.fitness_function = fitness_func
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            genes = np.random.uniform(-1, 1, self.gene_length)
            individual = Individual(genes, 0.0)
            individual.generation_born = 0
            self.population.append(individual)
    
    def _create_default_fitness_function(self, context: LearningContext) -> Callable:
        """创建默认适应度函数"""
        def fitness_function(genes: np.ndarray, context: LearningContext) -> float:
            try:
                # 基于基因计算策略参数
                buy_threshold = (genes[0] + 1) / 2  # 映射到[0,1]
                sell_threshold = (genes[1] + 1) / 2
                stop_loss = abs(genes[2]) * 0.2
                take_profit = abs(genes[3]) * 0.5
                
                # 基于历史性能评估
                if context.historical_performance:
                    returns = context.historical_performance[-20:]
                    avg_return = np.mean(returns)
                    return_stability = 1.0 / (1.0 + np.std(returns))
                    
                    # 风险调整收益
                    risk_adjusted_return = avg_return * return_stability
                    
                    # 参数合理性奖励
                    parameter_bonus = 0.0
                    if 0.1 <= buy_threshold <= 0.9:
                        parameter_bonus += 0.1
                    if 0.1 <= sell_threshold <= 0.9:
                        parameter_bonus += 0.1
                    if stop_loss < take_profit:
                        parameter_bonus += 0.1
                    
                    fitness = risk_adjusted_return + parameter_bonus
                    return max(0.0, fitness)
                else:
                    return 0.5
                    
            except Exception as e:
                logger.error(f"适应度计算出错: {e}")
                return 0.0
        
        return fitness_function
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) < 2:
            return 0.0
        
        diversity_scores = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.linalg.norm(self.population[i].genes - self.population[j].genes)
                diversity_scores.append(distance)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_convergence(self) -> float:
        """计算收敛率"""
        if len(self.fitness_history) < 5:
            return 0.0
        
        recent_fitness = [h['best_fitness'] for h in self.fitness_history[-5:]]
        if len(recent_fitness) < 2:
            return 0.0
        
        # 计算最近几代最佳适应度的变化率
        changes = [abs(recent_fitness[i] - recent_fitness[i-1]) for i in range(1, len(recent_fitness))]
        avg_change = np.mean(changes)
        
        # 收敛率 = 1 - 平均变化率
        convergence_rate = max(0.0, 1.0 - avg_change)
        return convergence_rate
    
    def _decode_genes(self, genes: np.ndarray, state: Dict[str, Any]) -> str:
        """解码基因为动作"""
        try:
            buy_threshold = (genes[0] + 1) / 2
            sell_threshold = (genes[1] + 1) / 2
            
            market_signal = state.get('market_signal', 0.5)
            
            if market_signal > buy_threshold:
                return 'buy'
            elif market_signal < sell_threshold:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            logger.error(f"基因解码出错: {e}")
            return 'hold'

class EvolutionStrategies(BaseStrategy):
    """进化策略"""
    
    def __init__(self, strategy_id: str, gene_length: int,
                 population_size: int = 50, sigma: float = 0.1):
        super().__init__(strategy_id, StrategyType.EVOLUTION)
        
        self.gene_length = gene_length
        self.population_size = population_size
        self.sigma = sigma
        
        self.population = []
        self.generation = 0
        self.fitness_history = []
        self.fitness_function = None
        
        self._initialize_population()
    
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """进化策略学习过程"""
        try:
            if not self.fitness_function:
                self.fitness_function = self._create_default_fitness_function(context)
            
            # 生成变异个体
            offspring = []
            for _ in range(self.population_size):
                # 选择父代
                parent = random.choice(self.population)
                
                # 变异
                mutated_genes = parent.genes + np.random.normal(0, self.sigma, self.gene_length)
                child = Individual(mutated_genes, 0.0)
                child.generation_born = self.generation + 1
                offspring.append(child)
            
            # 评估适应度
            for individual in offspring:
                individual.fitness = self.fitness_function(individual.genes, context)
            
            # 合并父代和子代
            combined_population = self.population + offspring
            
            # 选择下一代
            combined_population.sort(key=lambda x: x.fitness, reverse=True)
            self.population = combined_population[:self.population_size]
            
            # 记录历史
            fitnesses = [ind.fitness for ind in self.population]
            self.fitness_history.append({
                'generation': self.generation,
                'best_fitness': max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'sigma': self.sigma
            })
            
            # 自适应调整sigma
            self._adapt_sigma()
            
            self.generation += 1
            
            # 返回结果
            best_individual = self.population[0]
            
            return {
                'generation': self.generation,
                'best_fitness': best_individual.fitness,
                'best_genes': best_individual.genes.tolist(),
                'sigma': self.sigma,
                'population_size': len(self.population)
            }
            
        except Exception as e:
            logger.error(f"进化策略学习出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """进化策略预测"""
        try:
            if not self.population:
                return {'action': 'hold', 'confidence': 0.0}
            
            best_individual = self.population[0]
            action = self._decode_genes(best_individual.genes, state)
            confidence = min(1.0, best_individual.fitness)
            
            return {
                'action': action,
                'confidence': confidence,
                'genes': best_individual.genes.tolist(),
                'fitness': best_individual.fitness,
                'sigma': self.sigma
            }
            
        except Exception as e:
            logger.error(f"进化策略预测出错: {e}")
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
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def set_fitness_function(self, fitness_func: Callable):
        """设置适应度函数"""
        self.fitness_function = fitness_func
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            genes = np.random.normal(0, 1, self.gene_length)
            individual = Individual(genes, 0.0)
            individual.generation_born = 0
            self.population.append(individual)
    
    def _create_default_fitness_function(self, context: LearningContext) -> Callable:
        """创建默认适应度函数"""
        def fitness_function(genes: np.ndarray, context: LearningContext) -> float:
            # 简化的适应度函数
            if context.historical_performance:
                returns = context.historical_performance[-10:]
                return np.mean(returns)
            return 0.0
        
        return fitness_function
    
    def _adapt_sigma(self):
        """自适应调整变异强度"""
        if len(self.fitness_history) >= 5:
            recent_fitness = [h['best_fitness'] for h in self.fitness_history[-5:]]
            if len(set(recent_fitness)) < 3:  # 适应度变化很小
                self.sigma *= 1.2  # 增加变异
            else:
                self.sigma *= 0.9  # 减少变异
            
            self.sigma = np.clip(self.sigma, 0.01, 1.0)
    
    def _decode_genes(self, genes: np.ndarray, state: Dict[str, Any]) -> str:
        """解码基因为动作"""
        # 简化实现
        signal = np.mean(genes)
        if signal > 0:
            return 'buy'
        elif signal < 0:
            return 'sell'
        else:
            return 'hold'

class DifferentialEvolution(BaseStrategy):
    """差分进化算法"""
    
    def __init__(self, strategy_id: str, gene_length: int,
                 population_size: int = 30, F: float = 0.5, CR: float = 0.7):
        super().__init__(strategy_id, StrategyType.EVOLUTION)
        
        self.gene_length = gene_length
        self.population_size = population_size
        self.F = F  # 缩放因子
        self.CR = CR  # 交叉概率
        
        self.population = []
        self.generation = 0
        self.fitness_history = []
        self.fitness_function = None
        
        self._initialize_population()
    
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """差分进化学习过程"""
        try:
            if not self.fitness_function:
                self.fitness_function = self._create_default_fitness_function(context)
            
            # 评估当前种群
            for individual in self.population:
                individual.fitness = self.fitness_function(individual.genes, context)
            
            # 生成试验个体
            trial_population = []
            for i in range(self.population_size):
                # 选择三个不同的个体
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = random.sample(indices, 3)
                
                # 差分变异
                mutant_genes = self.population[a].genes + self.F * (self.population[b].genes - self.population[c].genes)
                
                # 交叉
                trial_genes = self._crossover(self.population[i].genes, mutant_genes)
                
                trial_individual = Individual(trial_genes, 0.0)
                trial_individual.generation_born = self.generation + 1
                trial_population.append(trial_individual)
            
            # 评估试验个体
            for individual in trial_population:
                individual.fitness = self.fitness_function(individual.genes, context)
            
            # 选择下一代
            new_population = []
            for i in range(self.population_size):
                if trial_population[i].fitness > self.population[i].fitness:
                    new_population.append(trial_population[i])
                else:
                    new_population.append(self.population[i])
            
            self.population = new_population
            
            # 记录历史
            fitnesses = [ind.fitness for ind in self.population]
            self.fitness_history.append({
                'generation': self.generation,
                'best_fitness': max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'F': self.F,
                'CR': self.CR
            })
            
            self.generation += 1
            
            # 返回结果
            best_individual = max(self.population, key=lambda x: x.fitness)
            
            return {
                'generation': self.generation,
                'best_fitness': best_individual.fitness,
                'best_genes': best_individual.genes.tolist(),
                'F': self.F,
                'CR': self.CR
            }
            
        except Exception as e:
            logger.error(f"差分进化学习出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """差分进化预测"""
        try:
            if not self.population:
                return {'action': 'hold', 'confidence': 0.0}
            
            best_individual = max(self.population, key=lambda x: x.fitness)
            action = self._decode_genes(best_individual.genes, state)
            confidence = min(1.0, best_individual.fitness)
            
            return {
                'action': action,
                'confidence': confidence,
                'genes': best_individual.genes.tolist(),
                'fitness': best_individual.fitness
            }
            
        except Exception as e:
            logger.error(f"差分进化预测出错: {e}")
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
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def set_fitness_function(self, fitness_func: Callable):
        """设置适应度函数"""
        self.fitness_function = fitness_func
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            genes = np.random.uniform(-2, 2, self.gene_length)
            individual = Individual(genes, 0.0)
            individual.generation_born = 0
            self.population.append(individual)
    
    def _create_default_fitness_function(self, context: LearningContext) -> Callable:
        """创建默认适应度函数"""
        def fitness_function(genes: np.ndarray, context: LearningContext) -> float:
            # 简化的适应度函数
            if context.historical_performance:
                returns = context.historical_performance[-10:]
                return np.mean(returns)
            return 0.0
        
        return fitness_function
    
    def _crossover(self, target_genes: np.ndarray, mutant_genes: np.ndarray) -> np.ndarray:
        """交叉操作"""
        trial_genes = target_genes.copy()
        
        # 确保至少有一个维度来自变异向量
        j_rand = random.randint(0, len(target_genes) - 1)
        
        for j in range(len(target_genes)):
            if random.random() < self.CR or j == j_rand:
                trial_genes[j] = mutant_genes[j]
        
        return trial_genes
    
    def _decode_genes(self, genes: np.ndarray, state: Dict[str, Any]) -> str:
        """解码基因为动作"""
        # 简化实现
        signal = np.mean(genes)
        if signal > 0:
            return 'buy'
        elif signal < 0:
            return 'sell'
        else:
            return 'hold'

# 工厂函数
def create_evolution_algorithm(algorithm: str, strategy_id: str, gene_length: int,
                              **kwargs) -> BaseStrategy:
    """创建进化算法实例"""
    algorithms = {
        'genetic_algorithm': GeneticAlgorithm,
        'evolution_strategies': EvolutionStrategies,
        'differential_evolution': DifferentialEvolution
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    return algorithms[algorithm](strategy_id, gene_length, **kwargs)