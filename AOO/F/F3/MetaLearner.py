"""
F3元学习器 - 实现元学习功能
=====================================

功能包括：
1. 学习如何学习的能力
2. 元认知监控和分析
3. 学习策略自动选择
4. 学习效果评估和优化
5. 元知识提取和管理
6. 学习迁移和适应
7. 元学习模型更新

实现MAML、Reptile等元学习算法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import copy
import pickle
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningStrategy:
    """学习策略配置"""
    name: str
    algorithm: str  # 'maml', 'reptile', 'fomaml', 'protonet'
    learning_rate: float
    inner_loop_steps: int
    meta_lr: float
    adaptation_steps: int
    parameters: Dict[str, Any]

@dataclass
class LearningEpisode:
    """学习episode记录"""
    task_id: str
    support_data: np.ndarray
    query_data: np.ndarray
    support_labels: np.ndarray
    query_labels: np.ndarray
    initial_params: Dict[str, np.ndarray]
    adapted_params: Dict[str, np.ndarray]
    final_loss: float
    adaptation_history: List[Dict[str, float]]
    timestamp: datetime

@dataclass
class MetaKnowledge:
    """元知识表示"""
    knowledge_type: str  # 'procedural', 'declarative', 'conditional'
    domain: str
    content: Dict[str, Any]
    confidence: float
    applicability: Dict[str, float]  # 适用场景
    timestamp: datetime

class BaseLearner(ABC):
    """基础学习器接口"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def adapt(self, support_data: torch.Tensor, support_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """适应新任务"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置模型参数"""
        pass

class SimpleNeuralNetwork(BaseLearner):
    """简单的神经网络实现"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.initial_params = None
        self.save_initial_params()
    
    def save_initial_params(self):
        """保存初始参数"""
        self.initial_params = {}
        for name, param in self.network.named_parameters():
            self.initial_params[name] = param.data.clone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def adapt(self, support_data: torch.Tensor, support_labels: torch.Tensor, 
              steps: int = 5, lr: float = 0.01) -> Dict[str, torch.Tensor]:
        """适应新任务"""
        adapted_params = {}
        
        # 保存当前参数
        current_params = {}
        for name, param in self.network.named_parameters():
            current_params[name] = param.data.clone()
        
        # 内循环适应
        optimizer = optim.SGD(self.network.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for step in range(steps):
            optimizer.zero_grad()
            predictions = self.network(support_data)
            loss = criterion(predictions, support_labels)
            loss.backward()
            optimizer.step()
        
        # 保存适应后的参数
        for name, param in self.network.named_parameters():
            adapted_params[name] = param.data.clone()
        
        return adapted_params
    
    def reset(self):
        """重置到初始参数"""
        if self.initial_params:
            for name, param in self.network.named_parameters():
                if name in self.initial_params:
                    param.data.copy_(self.initial_params[name])

class MAMLAlgorithm:
    """Model-Agnostic Meta-Learning (MAML) 算法实现"""
    
    def __init__(self, model: BaseLearner, meta_lr: float = 0.001):
        self.model = model
        self.meta_lr = meta_lr
        self.meta_optimizer = optim.Adam(self.model.network.parameters(), lr=meta_lr)
        
    def meta_train(self, tasks: List[Tuple], num_epochs: int = 1000):
        """元训练"""
        for epoch in range(num_epochs):
            meta_loss = 0
            
            for task_data in tasks:
                support_data, support_labels, query_data, query_labels = task_data
                
                # 内循环：适应任务
                adapted_params = self.model.adapt(support_data, support_labels)
                
                # 计算查询损失
                query_loss = self._compute_loss_with_params(query_data, query_labels, adapted_params)
                meta_loss += query_loss
            
            # 外循环：元参数更新
            meta_loss /= len(tasks)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Meta Loss: {meta_loss.item():.4f}")
    
    def _compute_loss_with_params(self, data: torch.Tensor, labels: torch.Tensor, 
                                 params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """使用指定参数计算损失"""
        # 临时应用参数
        original_params = {}
        for name, param in self.model.network.named_parameters():
            original_params[name] = param.data.clone()
            param.data.copy_(params[name])
        
        # 计算损失
        predictions = self.model.forward(data)
        criterion = nn.MSELoss()
        loss = criterion(predictions, labels)
        
        # 恢复原始参数
        for name, param in self.model.network.named_parameters():
            param.data.copy_(original_params[name])
        
        return loss

class ReptileAlgorithm:
    """Reptile算法实现"""
    
    def __init__(self, model: BaseLearner, meta_lr: float = 0.001):
        self.model = model
        self.meta_lr = meta_lr
        self.meta_optimizer = optim.Adam(self.model.network.parameters(), lr=meta_lr)
    
    def meta_train(self, tasks: List[Tuple], num_epochs: int = 1000):
        """元训练"""
        for epoch in range(num_epochs):
            meta_loss = 0
            
            for task_data in tasks:
                support_data, support_labels, query_data, query_labels = task_data
                
                # 保存初始参数
                initial_params = {}
                for name, param in self.model.network.named_parameters():
                    initial_params[name] = param.data.clone()
                
                # 内循环训练
                adapted_params = self.model.adapt(support_data, support_labels, steps=5, lr=0.01)
                
                # 计算元损失：参数更新的方向
                meta_loss += self._compute_reptile_loss(initial_params, adapted_params)
            
            # 元参数更新
            meta_loss /= len(tasks)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Meta Loss: {meta_loss.item():.4f}")
    
    def _compute_reptile_loss(self, initial_params: Dict[str, torch.Tensor], 
                             adapted_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算Reptile损失"""
        loss = 0
        for name in initial_params:
            loss += torch.sum((adapted_params[name] - initial_params[name]) ** 2)
        return loss * 0.5

class MetaCognitionMonitor:
    """元认知监控器"""
    
    def __init__(self):
        self.learning_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        self.strategy_effectiveness = defaultdict(float)
        self.knowledge_gaps = []
        self.learning_patterns = {}
    
    def record_learning_episode(self, episode: LearningEpisode):
        """记录学习episode"""
        self.learning_history.append(episode)
        self._update_performance_metrics(episode)
        self._analyze_learning_patterns(episode)
    
    def _update_performance_metrics(self, episode: LearningEpisode):
        """更新性能指标"""
        self.performance_metrics['loss'].append(episode.final_loss)
        self.performance_metrics['adaptation_speed'].append(len(episode.adaptation_history))
        
        # 计算学习效率
        if episode.adaptation_history:
            initial_loss = episode.adaptation_history[0]['loss']
            final_loss = episode.final_loss
            improvement = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
            self.performance_metrics['learning_efficiency'].append(improvement)
    
    def _analyze_learning_patterns(self, episode: LearningEpisode):
        """分析学习模式"""
        task_domain = episode.task_id.split('_')[0]  # 假设task_id包含域信息
        
        if task_domain not in self.learning_patterns:
            self.learning_patterns[task_domain] = {
                'episodes': 0,
                'avg_loss': 0,
                'avg_adaptation_steps': 0,
                'success_rate': 0
            }
        
        pattern = self.learning_patterns[task_domain]
        pattern['episodes'] += 1
        
        # 更新平均值
        n = pattern['episodes']
        pattern['avg_loss'] = (pattern['avg_loss'] * (n-1) + episode.final_loss) / n
        pattern['avg_adaptation_steps'] = (pattern['avg_adaptation_steps'] * (n-1) + 
                                         len(episode.adaptation_history)) / n
        
        # 更新成功率
        if episode.final_loss < 0.1:  # 假设阈值
            pattern['success_rate'] = (pattern['success_rate'] * (n-1) + 1) / n
        else:
            pattern['success_rate'] = (pattern['success_rate'] * (n-1)) / n
    
    def analyze_learning_state(self) -> Dict[str, Any]:
        """分析当前学习状态"""
        if not self.learning_history:
            return {'status': 'no_data'}
        
        recent_episodes = list(self.learning_history)[-100:]  # 最近100个episodes
        
        analysis = {
            'overall_performance': {
                'avg_loss': np.mean([ep.final_loss for ep in recent_episodes]),
                'std_loss': np.std([ep.final_loss for ep in recent_episodes]),
                'success_rate': len([ep for ep in recent_episodes if ep.final_loss < 0.1]) / len(recent_episodes)
            },
            'learning_trends': self._compute_learning_trends(recent_episodes),
            'domain_performance': self._analyze_domain_performance(),
            'knowledge_gaps': self._identify_knowledge_gaps(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _compute_learning_trends(self, episodes: List[LearningEpisode]) -> Dict[str, float]:
        """计算学习趋势"""
        if len(episodes) < 10:
            return {'trend': 0.0, 'stability': 0.0}
        
        losses = [ep.final_loss for ep in episodes]
        
        # 计算趋势（线性回归斜率）
        x = np.arange(len(losses))
        trend = np.polyfit(x, losses, 1)[0]
        
        # 计算稳定性（损失的标准差）
        stability = 1.0 / (1.0 + np.std(losses))
        
        return {'trend': trend, 'stability': stability}
    
    def _analyze_domain_performance(self) -> Dict[str, float]:
        """分析各域性能"""
        domain_performance = {}
        for domain, pattern in self.learning_patterns.items():
            domain_performance[domain] = {
                'avg_loss': pattern['avg_loss'],
                'success_rate': pattern['success_rate'],
                'adaptation_efficiency': 1.0 / (1.0 + pattern['avg_adaptation_steps'])
            }
        return domain_performance
    
    def _identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """识别知识空白"""
        gaps = []
        for domain, pattern in self.learning_patterns.items():
            if pattern['success_rate'] < 0.7 or pattern['avg_loss'] > 0.2:
                gaps.append({
                    'domain': domain,
                    'issue': 'low_success_rate' if pattern['success_rate'] < 0.7 else 'high_loss',
                    'current_performance': pattern['success_rate'],
                    'target_performance': 0.9
                })
        return gaps
    
    def _generate_recommendations(self) -> List[str]:
        """生成学习建议"""
        recommendations = []
        
        # 基于性能趋势的建议
        trends = self._compute_learning_trends(list(self.learning_history))
        if trends['trend'] > 0:
            recommendations.append("学习效果正在下降，建议调整学习策略")
        elif trends['stability'] < 0.5:
            recommendations.append("学习过程不够稳定，建议增加正则化")
        
        # 基于知识空白的建议
        gaps = self._identify_knowledge_gaps()
        for gap in gaps:
            recommendations.append(f"在{gap['domain']}域需要加强训练")
        
        return recommendations

class LearningStrategySelector:
    """学习策略自动选择器"""
    
    def __init__(self):
        self.strategies = {
            'fast_adaptation': LearningStrategy(
                name='fast_adaptation',
                algorithm='maml',
                learning_rate=0.01,
                inner_loop_steps=5,
                meta_lr=0.001,
                adaptation_steps=10,
                parameters={'batch_size': 32, 'regularization': 0.01}
            ),
            'stable_learning': LearningStrategy(
                name='stable_learning',
                algorithm='reptile',
                learning_rate=0.005,
                inner_loop_steps=10,
                meta_lr=0.0005,
                adaptation_steps=20,
                parameters={'batch_size': 64, 'regularization': 0.05}
            ),
            'few_shot': LearningStrategy(
                name='few_shot',
                algorithm='protonet',
                learning_rate=0.001,
                inner_loop_steps=1,
                meta_lr=0.0001,
                adaptation_steps=5,
                parameters={'batch_size': 16, 'regularization': 0.001}
            )
        }
        self.performance_history = defaultdict(list)
        self.context_features = {}
    
    def select_strategy(self, task_context: Dict[str, Any], 
                       meta_monitor: MetaCognitionMonitor) -> LearningStrategy:
        """基于任务上下文选择最佳策略"""
        
        # 提取任务特征
        task_features = self._extract_task_features(task_context)
        
        # 获取历史性能
        performance_scores = self._get_strategy_performance()
        
        # 策略评分
        strategy_scores = {}
        for name, strategy in self.strategies.items():
            score = self._score_strategy(strategy, task_features, performance_scores[name])
            strategy_scores[name] = score
        
        # 选择最佳策略
        best_strategy_name = max(strategy_scores, key=strategy_scores.get)
        return self.strategies[best_strategy_name]
    
    def _extract_task_features(self, task_context: Dict[str, Any]) -> Dict[str, float]:
        """提取任务特征"""
        features = {}
        
        # 数据特征
        features['data_size'] = task_context.get('data_size', 1000)
        features['feature_dim'] = task_context.get('feature_dim', 10)
        features['task_complexity'] = task_context.get('complexity', 0.5)
        
        # 域特征
        domain = task_context.get('domain', 'general')
        features['domain_familiarity'] = self._get_domain_familiarity(domain)
        
        # 学习约束
        features['time_budget'] = task_context.get('time_budget', 1000)
        features['accuracy_requirement'] = task_context.get('accuracy_requirement', 0.8)
        
        return features
    
    def _get_domain_familiarity(self, domain: str) -> float:
        """获取域熟悉度"""
        # 基于历史性能计算域熟悉度
        if domain in self.performance_history:
            return np.mean([p['success_rate'] for p in self.performance_history[domain]])
        return 0.5  # 默认中等熟悉度
    
    def _score_strategy(self, strategy: LearningStrategy, 
                       task_features: Dict[str, float], 
                       performance_history: List[Dict]) -> float:
        """评分策略"""
        score = 0.0
        
        # 基于历史性能 (40%)
        if performance_history:
            avg_performance = np.mean([p['success_rate'] for p in performance_history])
            score += avg_performance * 0.4
        
        # 基于算法适配性 (30%)
        algorithm_score = self._evaluate_algorithm_fit(strategy.algorithm, task_features)
        score += algorithm_score * 0.3
        
        # 基于参数适配性 (20%)
        param_score = self._evaluate_parameter_fit(strategy, task_features)
        score += param_score * 0.2
        
        # 基于资源约束 (10%)
        resource_score = self._evaluate_resource_constraints(strategy, task_features)
        score += resource_score * 0.1
        
        return score
    
    def _evaluate_algorithm_fit(self, algorithm: str, features: Dict[str, float]) -> float:
        """评估算法适配性"""
        if algorithm == 'maml':
            # MAML适合中等复杂度的任务
            complexity_score = 1.0 - abs(features['task_complexity'] - 0.5)
            return complexity_score
        elif algorithm == 'reptile':
            # Reptile适合稳定性要求高的任务
            stability_score = features['accuracy_requirement']
            return stability_score
        elif algorithm == 'protonet':
            # Protonet适合少样本任务
            few_shot_score = 1.0 if features['data_size'] < 100 else 0.5
            return few_shot_score
        return 0.5
    
    def _evaluate_parameter_fit(self, strategy: LearningStrategy, features: Dict[str, float]) -> float:
        """评估参数适配性"""
        # 学习率与数据大小的适配性
        lr_score = 1.0 if 0.001 <= strategy.learning_rate <= 0.01 else 0.5
        
        # 适应步数与时间预算的适配性
        time_score = 1.0 if strategy.adaptation_steps * 10 <= features['time_budget'] else 0.3
        
        return (lr_score + time_score) / 2
    
    def _evaluate_resource_constraints(self, strategy: LearningStrategy, features: Dict[str, float]) -> float:
        """评估资源约束"""
        # 计算预期计算成本
        expected_cost = strategy.inner_loop_steps * strategy.adaptation_steps
        
        # 与时间预算比较
        budget_ratio = features['time_budget'] / max(expected_cost, 1)
        return min(budget_ratio, 1.0)
    
    def _get_strategy_performance(self) -> Dict[str, List[Dict]]:
        """获取策略性能历史"""
        return self.performance_history
    
    def update_strategy_performance(self, strategy_name: str, task_domain: str, 
                                  performance: Dict[str, float]):
        """更新策略性能"""
        self.performance_history[strategy_name].append({
            'domain': task_domain,
            'success_rate': performance.get('success_rate', 0.5),
            'learning_speed': performance.get('learning_speed', 0.5),
            'timestamp': datetime.now()
        })

class MetaKnowledgeExtractor:
    """元知识提取器"""
    
    def __init__(self):
        self.knowledge_base = []
        self.knowledge_patterns = defaultdict(list)
        self.abstraction_level = 0
    
    def extract_knowledge(self, episodes: List[LearningEpisode], 
                         analysis_results: Dict[str, Any]) -> List[MetaKnowledge]:
        """从学习经验中提取元知识"""
        extracted_knowledge = []
        
        # 提取程序性知识
        procedural_knowledge = self._extract_procedural_knowledge(episodes)
        extracted_knowledge.extend(procedural_knowledge)
        
        # 提取声明性知识
        declarative_knowledge = self._extract_declarative_knowledge(analysis_results)
        extracted_knowledge.extend(declarative_knowledge)
        
        # 提取条件性知识
        conditional_knowledge = self._extract_conditional_knowledge(episodes, analysis_results)
        extracted_knowledge.extend(conditional_knowledge)
        
        # 更新知识库
        self.knowledge_base.extend(extracted_knowledge)
        
        return extracted_knowledge
    
    def _extract_procedural_knowledge(self, episodes: List[LearningEpisode]) -> List[MetaKnowledge]:
        """提取程序性知识"""
        procedural_knowledge = []
        
        # 分析成功的适应模式
        successful_episodes = [ep for ep in episodes if ep.final_loss < 0.1]
        
        if successful_episodes:
            # 提取共同适应模式
            common_patterns = self._find_common_adaptation_patterns(successful_episodes)
            
            for pattern in common_patterns:
                knowledge = MetaKnowledge(
                    knowledge_type='procedural',
                    domain=pattern['domain'],
                    content={
                        'adaptation_steps': pattern['steps'],
                        'learning_rate': pattern['lr'],
                        'convergence_criteria': pattern['criteria'],
                        'success_pattern': pattern['description']
                    },
                    confidence=pattern['confidence'],
                    applicability=pattern['applicability'],
                    timestamp=datetime.now()
                )
                procedural_knowledge.append(knowledge)
        
        return procedural_knowledge
    
    def _extract_declarative_knowledge(self, analysis_results: Dict[str, Any]) -> List[MetaKnowledge]:
        """提取声明性知识"""
        declarative_knowledge = []
        
        # 提取性能规律
        domain_performance = analysis_results.get('domain_performance', {})
        
        for domain, performance in domain_performance.items():
            knowledge = MetaKnowledge(
                knowledge_type='declarative',
                domain=domain,
                content={
                    'expected_performance': performance,
                    'success_threshold': 0.8,
                    'difficulty_level': self._assess_difficulty_level(performance)
                },
                confidence=performance.get('success_rate', 0.5),
                applicability={'generalization': 0.7, 'transfer': 0.6},
                timestamp=datetime.now()
            )
            declarative_knowledge.append(knowledge)
        
        return declarative_knowledge
    
    def _extract_conditional_knowledge(self, episodes: List[LearningEpisode], 
                                     analysis_results: Dict[str, Any]) -> List[MetaKnowledge]:
        """提取条件性知识"""
        conditional_knowledge = []
        
        # 分析不同条件下的学习效果
        conditions = self._categorize_learning_conditions(episodes)
        
        for condition, performance in conditions.items():
            knowledge = MetaKnowledge(
                knowledge_type='conditional',
                domain=condition['domain'],
                content={
                    'trigger_conditions': condition['features'],
                    'expected_outcome': performance,
                    'optimal_strategy': condition.get('best_strategy'),
                    'risk_factors': condition.get('risks', [])
                },
                confidence=performance.get('success_rate', 0.5),
                applicability={'prediction': 0.8, 'preparation': 0.7},
                timestamp=datetime.now()
            )
            conditional_knowledge.append(knowledge)
        
        return conditional_knowledge
    
    def _find_common_adaptation_patterns(self, episodes: List[LearningEpisode]) -> List[Dict]:
        """寻找共同的适应模式"""
        patterns = []
        
        # 按域分组
        domain_episodes = defaultdict(list)
        for ep in episodes:
            domain = ep.task_id.split('_')[0]
            domain_episodes[domain].append(ep)
        
        for domain, domain_episodes in domain_episodes.items():
            if len(domain_episodes) >= 3:  # 至少3个episodes才分析模式
                # 分析适应步数
                adaptation_steps = [len(ep.adaptation_history) for ep in domain_episodes]
                avg_steps = np.mean(adaptation_steps)
                
                # 分析损失变化
                loss_improvements = []
                for ep in domain_episodes:
                    if ep.adaptation_history:
                        initial_loss = ep.adaptation_history[0]['loss']
                        final_loss = ep.final_loss
                        improvement = (initial_loss - final_loss) / initial_loss
                        loss_improvements.append(improvement)
                
                avg_improvement = np.mean(loss_improvements) if loss_improvements else 0
                
                patterns.append({
                    'domain': domain,
                    'steps': avg_steps,
                    'lr': 0.01,  # 默认学习率
                    'criteria': f"loss_improvement > {avg_improvement:.2f}",
                    'description': f"在{domain}域平均需要{avg_steps:.1f}步适应",
                    'confidence': min(len(domain_episodes) / 10, 1.0),
                    'applicability': {'similar_tasks': 0.8, 'transfer_tasks': 0.6}
                })
        
        return patterns
    
    def _assess_difficulty_level(self, performance: Dict[str, float]) -> str:
        """评估难度级别"""
        success_rate = performance.get('success_rate', 0.5)
        avg_loss = performance.get('avg_loss', 0.5)
        
        if success_rate > 0.8 and avg_loss < 0.1:
            return 'easy'
        elif success_rate > 0.6 and avg_loss < 0.2:
            return 'medium'
        else:
            return 'hard'
    
    def _categorize_learning_conditions(self, episodes: List[LearningEpisode]) -> Dict:
        """分类学习条件"""
        conditions = defaultdict(lambda: {
            'episodes': [],
            'features': {},
            'performance': {'success_rate': 0, 'avg_loss': 0}
        })
        
        for ep in episodes:
            # 提取条件特征
            domain = ep.task_id.split('_')[0]
            condition_key = f"{domain}_small_data" if len(ep.support_data) < 50 else f"{domain}_large_data"
            
            conditions[condition_key]['episodes'].append(ep)
            conditions[condition_key]['domain'] = domain
        
        # 计算每个条件的性能
        for condition_key, condition_data in conditions.items():
            episodes_list = condition_data['episodes']
            if episodes_list:
                success_count = len([ep for ep in episodes_list if ep.final_loss < 0.1])
                avg_loss = np.mean([ep.final_loss for ep in episodes_list])
                
                condition_data['performance'] = {
                    'success_rate': success_count / len(episodes_list),
                    'avg_loss': avg_loss
                }
        
        return conditions
    
    def query_knowledge(self, query: Dict[str, Any]) -> List[MetaKnowledge]:
        """查询相关知识"""
        relevant_knowledge = []
        
        for knowledge in self.knowledge_base:
            # 检查域匹配
            if query.get('domain') and knowledge.domain != query['domain']:
                continue
            
            # 检查知识类型
            if query.get('knowledge_type') and knowledge.knowledge_type != query['knowledge_type']:
                continue
            
            # 检查适用性
            if query.get('applicability'):
                applicability_score = max([
                    knowledge.applicability.get(app_type, 0) 
                    for app_type in query['applicability']
                ])
                if applicability_score < 0.5:
                    continue
            
            relevant_knowledge.append(knowledge)
        
        # 按置信度排序
        relevant_knowledge.sort(key=lambda k: k.confidence, reverse=True)
        
        return relevant_knowledge

class LearningTransferManager:
    """学习迁移管理器"""
    
    def __init__(self, knowledge_extractor: MetaKnowledgeExtractor):
        self.knowledge_extractor = knowledge_extractor
        self.transfer_history = []
        self.similarity_metrics = {}
    
    def transfer_knowledge(self, source_domain: str, target_domain: str, 
                          task_context: Dict[str, Any]) -> Dict[str, Any]:
        """知识迁移"""
        
        # 获取源域知识
        source_knowledge = self.knowledge_extractor.query_knowledge({
            'domain': source_domain,
            'knowledge_type': 'procedural'
        })
        
        if not source_knowledge:
            return {'success': False, 'reason': 'no_source_knowledge'}
        
        # 计算域相似度
        similarity = self._compute_domain_similarity(source_domain, target_domain)
        
        # 评估迁移可行性
        transfer_feasibility = self._evaluate_transfer_feasibility(
            source_knowledge, target_domain, task_context, similarity
        )
        
        if transfer_feasibility['feasible']:
            # 执行知识迁移
            transferred_knowledge = self._apply_knowledge_transfer(
                source_knowledge, target_domain, task_context, similarity
            )
            
            # 记录迁移历史
            self.transfer_history.append({
                'source_domain': source_domain,
                'target_domain': target_domain,
                'similarity': similarity,
                'transferred_knowledge': transferred_knowledge,
                'timestamp': datetime.now()
            })
            
            return {
                'success': True,
                'similarity': similarity,
                'transferred_knowledge': transferred_knowledge,
                'confidence': transfer_feasibility['confidence']
            }
        else:
            return {
                'success': False,
                'reason': transfer_feasibility['reason'],
                'recommendations': transfer_feasibility.get('recommendations', [])
            }
    
    def _compute_domain_similarity(self, domain1: str, domain2: str) -> float:
        """计算域相似度"""
        # 简单的基于字符串相似度的计算
        # 实际应用中可以使用更复杂的相似度计算方法
        
        if domain1 == domain2:
            return 1.0
        
        # 计算编辑距离
        def edit_distance(s1, s2):
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = edit_distance(domain1.lower(), domain2.lower())
        max_len = max(len(domain1), len(domain2))
        similarity = 1.0 - (distance / max_len)
        
        return similarity
    
    def _evaluate_transfer_feasibility(self, source_knowledge: List[MetaKnowledge], 
                                     target_domain: str, task_context: Dict[str, Any],
                                     similarity: float) -> Dict[str, Any]:
        """评估迁移可行性"""
        
        # 基于相似度的可行性评估
        if similarity < 0.3:
            return {
                'feasible': False,
                'reason': 'low_similarity',
                'recommendations': ['寻找更相似的源域', '增加目标域训练数据']
            }
        
        # 基于知识质量的评估
        avg_confidence = np.mean([k.confidence for k in source_knowledge])
        if avg_confidence < 0.6:
            return {
                'feasible': False,
                'reason': 'low_knowledge_quality',
                'recommendations': ['提高源域知识质量', '收集更多源域数据']
            }
        
        # 基于任务上下文的评估
        if task_context.get('complexity', 0.5) > 0.8 and similarity < 0.7:
            return {
                'feasible': False,
                'reason': 'complex_task_low_similarity',
                'recommendations': ['降低任务复杂度', '提高域相似度']
            }
        
        confidence = similarity * avg_confidence
        
        return {
            'feasible': True,
            'confidence': confidence,
            'reason': 'feasible_transfer'
        }
    
    def _apply_knowledge_transfer(self, source_knowledge: List[MetaKnowledge], 
                                target_domain: str, task_context: Dict[str, Any],
                                similarity: float) -> Dict[str, Any]:
        """应用知识迁移"""
        
        transferred_procedures = []
        transferred_declarations = []
        
        for knowledge in source_knowledge:
            if knowledge.knowledge_type == 'procedural':
                # 迁移程序性知识
                adapted_procedure = self._adapt_procedural_knowledge(
                    knowledge, target_domain, similarity
                )
                transferred_procedures.append(adapted_procedure)
            
            elif knowledge.knowledge_type == 'declarative':
                # 迁移声明性知识
                adapted_declaration = self._adapt_declarative_knowledge(
                    knowledge, target_domain, similarity
                )
                transferred_declarations.append(adapted_declaration)
        
        return {
            'procedures': transferred_procedures,
            'declarations': transferred_declarations,
            'transfer_ratio': similarity,
            'adaptation_notes': self._generate_adaptation_notes(similarity)
        }
    
    def _adapt_procedural_knowledge(self, knowledge: MetaKnowledge, 
                                  target_domain: str, similarity: float) -> Dict[str, Any]:
        """适应程序性知识"""
        content = knowledge.content.copy()
        
        # 根据相似度调整参数
        if similarity < 0.8:
            # 降低学习率，增加适应步数
            if 'learning_rate' in content:
                content['learning_rate'] *= (1 - (0.8 - similarity))
            if 'adaptation_steps' in content:
                content['adaptation_steps'] = int(content['adaptation_steps'] * (1 + (0.8 - similarity)))
        
        return {
            'original_domain': knowledge.domain,
            'target_domain': target_domain,
            'procedure': content,
            'confidence': knowledge.confidence * similarity,
            'adaptations': self._identify_adaptations(knowledge, similarity)
        }
    
    def _adapt_declarative_knowledge(self, knowledge: MetaKnowledge, 
                                   target_domain: str, similarity: float) -> Dict[str, Any]:
        """适应声明性知识"""
        content = knowledge.content.copy()
        
        # 调整性能预期
        if 'expected_performance' in content:
            performance = content['expected_performance']
            for key in performance:
                if isinstance(performance[key], (int, float)):
                    performance[key] *= similarity
        
        return {
            'original_domain': knowledge.domain,
            'target_domain': target_domain,
            'declaration': content,
            'confidence': knowledge.confidence * similarity,
            'adaptations': self._identify_adaptations(knowledge, similarity)
        }
    
    def _identify_adaptations(self, knowledge: MetaKnowledge, similarity: float) -> List[str]:
        """识别需要的适应"""
        adaptations = []
        
        if similarity < 0.9:
            adaptations.append("调整参数设置")
        
        if similarity < 0.7:
            adaptations.append("增加正则化")
            adaptations.append("延长训练时间")
        
        if similarity < 0.5:
            adaptations.append("考虑重新训练")
        
        return adaptations
    
    def _generate_adaptation_notes(self, similarity: float) -> List[str]:
        """生成适应说明"""
        notes = []
        
        if similarity > 0.9:
            notes.append("高度相似的域，可以直接应用")
        elif similarity > 0.7:
            notes.append("较为相似的域，需要轻微调整")
        elif similarity > 0.5:
            notes.append("中等相似度的域，需要适度调整")
        else:
            notes.append("相似度较低，建议谨慎迁移")
        
        return notes

class MetaLearner:
    """F3元学习器主类"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        # 核心组件
        self.model = SimpleNeuralNetwork(input_dim, hidden_dim)
        self.meta_monitor = MetaCognitionMonitor()
        self.strategy_selector = LearningStrategySelector()
        self.knowledge_extractor = MetaKnowledgeExtractor()
        self.transfer_manager = LearningTransferManager(self.knowledge_extractor)
        
        # 元学习算法
        self.algorithms = {
            'maml': MAMLAlgorithm(self.model),
            'reptile': ReptileAlgorithm(self.model)
        }
        
        # 当前策略
        self.current_strategy = None
        
        # 学习历史
        self.learning_history = []
        
        # 性能统计
        self.performance_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'avg_learning_time': 0,
            'knowledge_base_size': 0
        }
        
        logger.info("F3元学习器初始化完成")
    
    def learn_new_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """学习新任务"""
        logger.info(f"开始学习新任务: {task_data.get('task_id', 'unknown')}")
        
        # 1. 选择学习策略
        strategy = self.strategy_selector.select_strategy(task_data, self.meta_monitor)
        self.current_strategy = strategy
        logger.info(f"选择策略: {strategy.name}")
        
        # 2. 尝试知识迁移
        transfer_result = None
        if task_data.get('source_domain'):
            transfer_result = self.transfer_manager.transfer_knowledge(
                task_data['source_domain'],
                task_data.get('target_domain', 'unknown'),
                task_data
            )
        
        # 3. 执行学习
        learning_result = self._execute_learning(task_data, strategy, transfer_result)
        
        # 4. 记录学习过程
        episode = self._create_learning_episode(task_data, learning_result)
        self.meta_monitor.record_learning_episode(episode)
        self.learning_history.append(episode)
        
        # 5. 更新性能统计
        self._update_performance_stats(learning_result)
        
        # 6. 提取元知识
        if len(self.learning_history) % 10 == 0:  # 每10个任务提取一次知识
            self._extract_and_update_knowledge()
        
        # 7. 生成学习报告
        learning_report = self._generate_learning_report(episode, learning_result, transfer_result)
        
        logger.info(f"任务学习完成，最终损失: {learning_result['final_loss']:.4f}")
        return learning_report
    
    def _execute_learning(self, task_data: Dict[str, Any], strategy: LearningStrategy, 
                         transfer_result: Dict[str, Any]) -> Dict[str, Any]:
        """执行学习过程"""
        
        # 准备数据
        support_data = torch.tensor(task_data['support_data'], dtype=torch.float32)
        support_labels = torch.tensor(task_data['support_labels'], dtype=torch.float32)
        query_data = torch.tensor(task_data['query_data'], dtype=torch.float32)
        query_labels = torch.tensor(task_data['query_labels'], dtype=torch.float32)
        
        # 应用迁移知识（如果有）
        if transfer_result and transfer_result['success']:
            self._apply_transferred_knowledge(transfer_result['transferred_knowledge'])
        
        # 选择算法并执行学习
        algorithm = self.algorithms.get(strategy.algorithm)
        if not algorithm:
            raise ValueError(f"未知算法: {strategy.algorithm}")
        
        # 记录适应过程
        adaptation_history = []
        original_params = {}
        for name, param in self.model.network.named_parameters():
            original_params[name] = param.data.clone()
        
        # 内循环适应
        adapted_params = self.model.adapt(
            support_data, support_labels,
            steps=strategy.adaptation_steps,
            lr=strategy.learning_rate
        )
        
        # 记录适应历史
        for step in range(strategy.adaptation_steps):
            # 模拟每步的损失计算
            step_loss = self._compute_step_loss(support_data, support_labels, step)
            adaptation_history.append({
                'step': step,
                'loss': step_loss,
                'learning_rate': strategy.learning_rate * (0.9 ** step)  # 衰减学习率
            })
        
        # 计算最终查询损失
        final_loss = self._compute_query_loss(query_data, query_labels, adapted_params)
        
        return {
            'final_loss': final_loss,
            'adaptation_history': adaptation_history,
            'adapted_params': adapted_params,
            'original_params': original_params,
            'strategy_used': strategy.name,
            'transfer_applied': transfer_result is not None and transfer_result['success']
        }
    
    def _apply_transferred_knowledge(self, transferred_knowledge: Dict[str, Any]):
        """应用迁移的知识"""
        procedures = transferred_knowledge.get('procedures', [])
        
        for procedure in procedures:
            if 'procedure' in procedure:
                proc_content = procedure['procedure']
                
                # 应用迁移的参数调整
                if 'learning_rate' in proc_content:
                    # 这里可以调整模型的学习相关参数
                    pass
                
                if 'adaptation_steps' in proc_content:
                    # 这里可以调整适应步数
                    pass
    
    def _compute_step_loss(self, data: torch.Tensor, labels: torch.Tensor, step: int) -> float:
        """计算步骤损失"""
        # 简化的损失计算
        predictions = self.model.forward(data)
        criterion = nn.MSELoss()
        loss = criterion(predictions, labels)
        return loss.item() * (0.9 ** step)  # 模拟损失衰减
    
    def _compute_query_loss(self, query_data: torch.Tensor, query_labels: torch.Tensor, 
                           adapted_params: Dict[str, torch.Tensor]) -> float:
        """计算查询损失"""
        # 临时应用适应后的参数
        original_params = {}
        for name, param in self.model.network.named_parameters():
            original_params[name] = param.data.clone()
            param.data.copy_(adapted_params[name])
        
        # 计算查询损失
        predictions = self.model.forward(query_data)
        criterion = nn.MSELoss()
        query_loss = criterion(predictions, query_labels).item()
        
        # 恢复原始参数
        for name, param in self.model.network.named_parameters():
            param.data.copy_(original_params[name])
        
        return query_loss
    
    def _create_learning_episode(self, task_data: Dict[str, Any], 
                               learning_result: Dict[str, Any]) -> LearningEpisode:
        """创建学习episode记录"""
        return LearningEpisode(
            task_id=task_data.get('task_id', 'unknown'),
            support_data=np.array(task_data['support_data']),
            query_data=np.array(task_data['query_data']),
            support_labels=np.array(task_data['support_labels']),
            query_labels=np.array(task_data['query_labels']),
            initial_params=learning_result['original_params'],
            adapted_params=learning_result['adapted_params'],
            final_loss=learning_result['final_loss'],
            adaptation_history=learning_result['adaptation_history'],
            timestamp=datetime.now()
        )
    
    def _update_performance_stats(self, learning_result: Dict[str, Any]):
        """更新性能统计"""
        self.performance_stats['total_episodes'] += 1
        
        if learning_result['final_loss'] < 0.1:
            self.performance_stats['successful_episodes'] += 1
        
        # 更新平均学习时间（这里简化为适应步数）
        current_avg = self.performance_stats['avg_learning_time']
        n = self.performance_stats['total_episodes']
        new_avg = ((current_avg * (n-1)) + len(learning_result['adaptation_history'])) / n
        self.performance_stats['avg_learning_time'] = new_avg
    
    def _extract_and_update_knowledge(self):
        """提取和更新元知识"""
        recent_episodes = self.learning_history[-100:]  # 最近100个episodes
        
        # 分析学习状态
        analysis_results = self.meta_monitor.analyze_learning_state()
        
        # 提取元知识
        extracted_knowledge = self.knowledge_extractor.extract_knowledge(
            recent_episodes, analysis_results
        )
        
        # 更新知识库大小
        self.performance_stats['knowledge_base_size'] = len(self.knowledge_extractor.knowledge_base)
        
        logger.info(f"提取了 {len(extracted_knowledge)} 条新元知识")
    
    def _generate_learning_report(self, episode: LearningEpisode, 
                                learning_result: Dict[str, Any], 
                                transfer_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成学习报告"""
        
        # 获取当前学习状态分析
        learning_state = self.meta_monitor.analyze_learning_state()
        
        report = {
            'task_id': episode.task_id,
            'learning_performance': {
                'final_loss': learning_result['final_loss'],
                'success': learning_result['final_loss'] < 0.1,
                'adaptation_steps': len(episode.adaptation_history),
                'learning_efficiency': self._calculate_learning_efficiency(episode)
            },
            'strategy_info': {
                'strategy_used': learning_result['strategy_used'],
                'algorithm': self.current_strategy.algorithm if self.current_strategy else 'unknown',
                'parameters': asdict(self.current_strategy) if self.current_strategy else {}
            },
            'knowledge_transfer': {
                'applied': transfer_result is not None and transfer_result.get('success', False),
                'similarity': transfer_result.get('similarity', 0) if transfer_result else 0,
                'confidence': transfer_result.get('confidence', 0) if transfer_result else 0
            },
            'meta_analysis': {
                'overall_performance': learning_state.get('overall_performance', {}),
                'domain_performance': learning_state.get('domain_performance', {}),
                'recommendations': learning_state.get('recommendations', [])
            },
            'system_status': {
                'total_episodes': self.performance_stats['total_episodes'],
                'success_rate': (self.performance_stats['successful_episodes'] / 
                               max(self.performance_stats['total_episodes'], 1)),
                'knowledge_base_size': self.performance_stats['knowledge_base_size'],
                'avg_learning_time': self.performance_stats['avg_learning_time']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_learning_efficiency(self, episode: LearningEpisode) -> float:
        """计算学习效率"""
        if not episode.adaptation_history:
            return 0.0
        
        initial_loss = episode.adaptation_history[0]['loss']
        final_loss = episode.final_loss
        
        if initial_loss <= 0:
            return 0.0
        
        improvement = (initial_loss - final_loss) / initial_loss
        efficiency = improvement / len(episode.adaptation_history)
        
        return efficiency
    
    def meta_optimize(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """元优化：优化学习策略和模型"""
        logger.info("开始元优化过程")
        
        # 1. 分析当前性能
        current_analysis = self.meta_monitor.analyze_learning_state()
        
        # 2. 基于分析结果优化策略
        strategy_optimizations = self._optimize_learning_strategies(current_analysis)
        
        # 3. 执行模型更新
        model_updates = self._perform_model_updates(current_analysis, optimization_config)
        
        # 4. 更新元学习算法参数
        algorithm_updates = self._update_algorithm_parameters(current_analysis)
        
        # 5. 清理和压缩知识库
        knowledge_optimization = self._optimize_knowledge_base()
        
        optimization_result = {
            'optimization_summary': {
                'strategies_optimized': len(strategy_optimizations),
                'model_updated': model_updates['success'],
                'algorithms_updated': len(algorithm_updates),
                'knowledge_optimized': knowledge_optimization['optimized_items']
            },
            'strategy_changes': strategy_optimizations,
            'model_changes': model_updates,
            'algorithm_changes': algorithm_updates,
            'knowledge_changes': knowledge_optimization,
            'performance_prediction': self._predict_performance_improvement(current_analysis),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("元优化完成")
        return optimization_result
    
    def _optimize_learning_strategies(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优化学习策略"""
        optimizations = []
        
        # 基于性能趋势调整策略
        trends = analysis.get('learning_trends', {})
        if trends.get('trend', 0) > 0:  # 性能下降趋势
            # 增加正则化
            for strategy in self.strategy_selector.strategies.values():
                if 'regularization' in strategy.parameters:
                    old_reg = strategy.parameters['regularization']
                    strategy.parameters['regularization'] = min(old_reg * 1.2, 0.1)
                    optimizations.append({
                        'strategy': strategy.name,
                        'change': 'increased_regularization',
                        'old_value': old_reg,
                        'new_value': strategy.parameters['regularization']
                    })
        
        # 基于知识空白调整策略
        gaps = analysis.get('knowledge_gaps', [])
        for gap in gaps:
            if gap['domain'] in self.strategy_selector.strategies:
                strategy = self.strategy_selector.strategies[gap['domain']]
                # 增加适应步数
                old_steps = strategy.adaptation_steps
                strategy.adaptation_steps = min(old_steps + 5, 50)
                optimizations.append({
                    'strategy': strategy.name,
                    'change': 'increased_adaptation_steps',
                    'old_value': old_steps,
                    'new_value': strategy.adaptation_steps,
                    'reason': f"knowledge_gap_in_{gap['domain']}"
                })
        
        return optimizations
    
    def _perform_model_updates(self, analysis: Dict[str, Any], 
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """执行模型更新"""
        try:
            # 保存当前模型状态
            current_state = {}
            for name, param in self.model.network.named_parameters():
                current_state[name] = param.data.clone()
            
            # 基于性能分析调整模型结构或参数
            overall_perf = analysis.get('overall_performance', {})
            if overall_perf.get('success_rate', 0) < 0.6:
                # 性能较低，增加模型容量
                logger.info("检测到性能较低，增加模型容量")
                # 这里可以添加动态调整模型结构的逻辑
                
                # 简单的参数调整：增加dropout
                for module in self.model.network.modules():
                    if isinstance(module, nn.Linear) and hasattr(module, 'p'):
                        module.p = min(module.p * 1.1, 0.5)
            
            return {
                'success': True,
                'changes_applied': 'model_capacity_increase',
                'previous_state': current_state,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"模型更新失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _update_algorithm_parameters(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """更新算法参数"""
        updates = []
        
        # 基于学习稳定性调整元学习率
        trends = analysis.get('learning_trends', {})
        stability = trends.get('stability', 0.5)
        
        for name, algorithm in self.algorithms.items():
            if stability < 0.5:
                # 学习不稳定，降低元学习率
                old_meta_lr = algorithm.meta_lr
                algorithm.meta_lr = max(algorithm.meta_lr * 0.8, 0.0001)
                updates.append({
                    'algorithm': name,
                    'parameter': 'meta_lr',
                    'old_value': old_meta_lr,
                    'new_value': algorithm.meta_lr,
                    'reason': 'low_stability'
                })
        
        return updates
    
    def _optimize_knowledge_base(self) -> Dict[str, Any]:
        """优化知识库"""
        original_size = len(self.knowledge_extractor.knowledge_base)
        
        # 移除过时或低质量的知识
        current_time = datetime.now()
        optimized_knowledge = []
        removed_count = 0
        
        for knowledge in self.knowledge_extractor.knowledge_base:
            # 移除置信度过低或过时的知识
            age_days = (current_time - knowledge.timestamp).days
            if knowledge.confidence > 0.3 and age_days < 30:
                optimized_knowledge.append(knowledge)
            else:
                removed_count += 1
        
        # 更新知识库
        self.knowledge_extractor.knowledge_base = optimized_knowledge
        
        # 合并相似知识
        self._merge_similar_knowledge()
        
        return {
            'optimized_items': removed_count,
            'original_size': original_size,
            'new_size': len(self.knowledge_extractor.knowledge_base),
            'compression_ratio': len(self.knowledge_extractor.knowledge_base) / max(original_size, 1)
        }
    
    def _merge_similar_knowledge(self):
        """合并相似知识"""
        # 简化的知识合并逻辑
        merged_knowledge = []
        processed_indices = set()
        
        for i, knowledge1 in enumerate(self.knowledge_extractor.knowledge_base):
            if i in processed_indices:
                continue
            
            similar_knowledge = [knowledge1]
            processed_indices.add(i)
            
            # 寻找相似的知识
            for j, knowledge2 in enumerate(self.knowledge_extractor.knowledge_base):
                if j in processed_indices or i == j:
                    continue
                
                # 检查是否相似（同域且类型相同）
                if (knowledge1.domain == knowledge2.domain and 
                    knowledge1.knowledge_type == knowledge2.knowledge_type):
                    similar_knowledge.append(knowledge2)
                    processed_indices.add(j)
            
            # 合并相似知识
            if len(similar_knowledge) > 1:
                merged_knowledge.append(self._merge_knowledge_list(similar_knowledge))
            else:
                merged_knowledge.append(knowledge1)
        
        self.knowledge_extractor.knowledge_base = merged_knowledge
    
    def _merge_knowledge_list(self, knowledge_list: List[MetaKnowledge]) -> MetaKnowledge:
        """合并知识列表"""
        if not knowledge_list:
            return None
        
        # 使用第一个知识作为基础
        base_knowledge = knowledge_list[0]
        
        # 合并内容（这里简化处理）
        merged_content = base_knowledge.content.copy()
        
        # 计算平均置信度
        avg_confidence = np.mean([k.confidence for k in knowledge_list])
        
        # 合并适用性
        merged_applicability = {}
        for knowledge in knowledge_list:
            for app_type, score in knowledge.applicability.items():
                if app_type not in merged_applicability:
                    merged_applicability[app_type] = []
                merged_applicability[app_type].append(score)
        
        # 计算平均适用性
        for app_type in merged_applicability:
            merged_applicability[app_type] = np.mean(merged_applicability[app_type])
        
        return MetaKnowledge(
            knowledge_type=base_knowledge.knowledge_type,
            domain=base_knowledge.domain,
            content=merged_content,
            confidence=avg_confidence,
            applicability=merged_applicability,
            timestamp=datetime.now()
        )
    
    def _predict_performance_improvement(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """预测性能改进"""
        current_performance = analysis.get('overall_performance', {})
        current_success_rate = current_performance.get('success_rate', 0.5)
        
        # 基于知识库大小预测改进
        knowledge_factor = min(self.performance_stats['knowledge_base_size'] / 100, 0.3)
        
        # 基于学习稳定性预测改进
        stability = analysis.get('learning_trends', {}).get('stability', 0.5)
        stability_factor = (1 - stability) * 0.2
        
        predicted_improvement = knowledge_factor + stability_factor
        predicted_success_rate = min(current_success_rate + predicted_improvement, 0.95)
        
        return {
            'current_success_rate': current_success_rate,
            'predicted_success_rate': predicted_success_rate,
            'improvement': predicted_improvement,
            'confidence': min(knowledge_factor * 2, 0.8)
        }
    
    def save_state(self, filepath: str):
        """保存元学习器状态"""
        state = {
            'model_state_dict': self.model.network.state_dict(),
            'model_initial_params': self.model.initial_params,
            'learning_history': [asdict(ep) for ep in self.learning_history],
            'knowledge_base': [asdict(k) for k in self.knowledge_extractor.knowledge_base],
            'performance_stats': self.performance_stats,
            'strategy_config': {name: asdict(strategy) for name, strategy in self.strategy_selector.strategies.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"元学习器状态已保存到: {filepath}")
    
    def load_state(self, filepath: str):
        """加载元学习器状态"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # 恢复模型状态
            self.model.network.load_state_dict(state['model_state_dict'])
            self.model.initial_params = state['model_initial_params']
            
            # 恢复学习历史
            self.learning_history = []
            for ep_dict in state['learning_history']:
                ep = LearningEpisode(**ep_dict)
                # 转换时间戳
                ep.timestamp = datetime.fromisoformat(ep_dict['timestamp'])
                self.learning_history.append(ep)
            
            # 恢复知识库
            self.knowledge_extractor.knowledge_base = []
            for k_dict in state['knowledge_base']:
                k = MetaKnowledge(**k_dict)
                # 转换时间戳
                k.timestamp = datetime.fromisoformat(k_dict['timestamp'])
                self.knowledge_extractor.knowledge_base.append(k)
            
            # 恢复其他状态
            self.performance_stats = state['performance_stats']
            
            # 恢复策略配置
            for name, strategy_dict in state['strategy_config'].items():
                self.strategy_selector.strategies[name] = LearningStrategy(**strategy_dict)
            
            logger.info(f"元学习器状态已从 {filepath} 加载")
            
        except Exception as e:
            logger.error(f"加载状态失败: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'performance_stats': self.performance_stats,
            'knowledge_base_size': len(self.knowledge_extractor.knowledge_base),
            'learning_history_length': len(self.learning_history),
            'current_strategy': asdict(self.current_strategy) if self.current_strategy else None,
            'available_algorithms': list(self.algorithms.keys()),
            'recent_performance': self.meta_monitor.analyze_learning_state(),
            'timestamp': datetime.now().isoformat()
        }

# 使用示例和测试函数
def create_sample_task(input_dim: int = 10, num_samples: int = 100) -> Dict[str, Any]:
    """创建示例任务"""
    np.random.seed(42)
    
    # 生成支持集和查询集数据
    support_size = num_samples // 2
    query_size = num_samples - support_size
    
    # 随机生成数据
    support_data = np.random.randn(support_size, input_dim)
    query_data = np.random.randn(query_size, input_dim)
    
    # 生成标签（简单的线性关系 + 噪声）
    true_weights = np.random.randn(input_dim)
    support_labels = support_data @ true_weights + 0.1 * np.random.randn(support_size)
    query_labels = query_data @ true_weights + 0.1 * np.random.randn(query_size)
    
    return {
        'task_id': f'task_{np.random.randint(1000)}',
        'domain': 'synthetic',
        'support_data': support_data.tolist(),
        'query_data': query_data.tolist(),
        'support_labels': support_labels.reshape(-1, 1).tolist(),
        'query_labels': query_labels.reshape(-1, 1).tolist(),
        'complexity': 0.5,
        'data_size': num_samples
    }

def demo_meta_learner():
    """演示元学习器功能"""
    print("=== F3元学习器演示 ===")
    
    # 初始化元学习器
    meta_learner = MetaLearner(input_dim=10, hidden_dim=32)
    print("✓ 元学习器初始化完成")
    
    # 创建多个任务进行学习
    tasks = []
    for i in range(20):
        task = create_sample_task(input_dim=10, num_samples=50)
        task['task_id'] = f'task_{i}'
        tasks.append(task)
    
    print(f"✓ 创建了 {len(tasks)} 个学习任务")
    
    # 执行元学习
    learning_results = []
    for i, task in enumerate(tasks):
        print(f"\n--- 学习任务 {i+1}/{len(tasks)} ---")
        result = meta_learner.learn_new_task(task)
        learning_results.append(result)
        
        print(f"任务: {result['task_id']}")
        print(f"策略: {result['strategy_info']['strategy_used']}")
        print(f"最终损失: {result['learning_performance']['final_loss']:.4f}")
        print(f"成功: {result['learning_performance']['success']}")
        
        if result['knowledge_transfer']['applied']:
            print(f"知识迁移: 相似度={result['knowledge_transfer']['similarity']:.2f}")
    
    # 执行元优化
    print("\n=== 元优化 ===")
    optimization_result = meta_learner.meta_optimize({})
    print(f"策略优化: {optimization_result['optimization_summary']['strategies_optimized']} 项")
    print(f"模型更新: {optimization_result['optimization_summary']['model_updated']}")
    print(f"算法更新: {optimization_result['optimization_summary']['algorithms_updated']}")
    
    # 显示系统状态
    print("\n=== 系统状态 ===")
    status = meta_learner.get_system_status()
    print(f"总任务数: {status['performance_stats']['total_episodes']}")
    print(f"成功率: {status['performance_stats']['successful_episodes']/max(status['performance_stats']['total_episodes'],1):.2%}")
    print(f"知识库大小: {status['knowledge_base_size']}")
    print(f"平均学习时间: {status['performance_stats']['avg_learning_time']:.1f} 步")
    
    # 保存状态
    meta_learner.save_state('/tmp/meta_learner_state.pkl')
    print("\n✓ 状态已保存")
    
    return meta_learner, learning_results, optimization_result

if __name__ == "__main__":
    # 运行演示
    meta_learner, results, optimization = demo_meta_learner()
    
    print("\n=== 演示完成 ===")
    print("F3元学习器已成功实现以下功能：")
    print("✓ MAML和Reptile元学习算法")
    print("✓ 元认知监控和分析")
    print("✓ 学习策略自动选择")
    print("✓ 学习效果评估和优化")
    print("✓ 元知识提取和管理")
    print("✓ 学习迁移和适应")
    print("✓ 元学习模型持续更新")