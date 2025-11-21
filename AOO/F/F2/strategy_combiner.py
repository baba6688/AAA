"""
策略组合器模块

实现多种策略组合和融合算法：
- 加权平均组合
- 投票机制组合
- 堆叠组合（元学习）
- 自适应组合
- 动态权重调整
- 风险平价组合
- 最优组合构建
- 组合优化算法
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from .StrategyLearner import BaseStrategy, StrategyType, LearningContext

logger = logging.getLogger(__name__)

@dataclass
class CombinationResult:
    """组合结果数据类"""
    combined_action: Any
    combined_confidence: float
    individual_predictions: Dict[str, Any]
    combination_weights: Dict[str, float]
    method: str
    metadata: Dict[str, Any] = None

@dataclass
class PortfolioConstraints:
    """组合约束条件"""
    max_weight: float = 1.0
    min_weight: float = 0.0
    max_risk: float = 0.2
    turnover_limit: float = 0.5
    sector_limits: Dict[str, float] = None
    concentration_limit: float = 0.3

class BaseCombiner(ABC):
    """组合器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.feature_importance = {}
        self.combination_history = []
    
    @abstractmethod
    def combine(self, predictions: Dict[str, Any], context: Optional[LearningContext] = None) -> CombinationResult:
        """组合预测结果"""
        pass
    
    @abstractmethod
    def fit(self, historical_data: List[Dict[str, Any]]):
        """训练组合器"""
        pass
    
    def update_history(self, result: CombinationResult):
        """更新组合历史"""
        self.combination_history.append({
            'timestamp': datetime.now(),
            'result': result,
            'success': result.combined_confidence > 0.5
        })

class WeightedAverageCombiner(BaseCombiner):
    """加权平均组合器"""
    
    def __init__(self, name: str = "weighted_average", 
                 weight_method: str = 'performance_based',
                 rebalance_frequency: int = 10):
        super().__init__(name)
        self.weight_method = weight_method
        self.rebalance_frequency = rebalance_frequency
        self.weights = {}
        self.performance_history = defaultdict(list)
        self.last_rebalance = 0
        
    def combine(self, predictions: Dict[str, Any], 
               context: Optional[LearningContext] = None) -> CombinationResult:
        """加权平均组合"""
        try:
            if not predictions:
                return CombinationResult(
                    combined_action='hold',
                    combined_confidence=0.0,
                    individual_predictions={},
                    combination_weights={},
                    method=self.name,
                    metadata={'error': '没有预测结果'}
                )
            
            # 获取权重
            if not self.weights:
                self._initialize_weights(predictions)
            
            # 计算加权组合
            weighted_actions = []
            weighted_confidences = []
            
            for strategy_id, prediction in predictions.items():
                if strategy_id in self.weights:
                    weight = self.weights[strategy_id]
                    action = prediction.get('action', 'hold')
                    confidence = prediction.get('confidence', 0.5)
                    
                    weighted_actions.append((strategy_id, action, weight))
                    weighted_confidences.append(confidence * weight)
            
            # 决定最终动作
            if weighted_actions:
                # 简单投票 + 权重
                action_votes = defaultdict(float)
                total_weight = sum(self.weights.values())
                
                for strategy_id, action, weight in weighted_actions:
                    action_votes[action] += weight
                
                if action_votes:
                    best_action = max(action_votes, key=action_votes.get)
                    combined_confidence = action_votes[best_action] / total_weight if total_weight > 0 else 0.0
                else:
                    best_action = 'hold'
                    combined_confidence = 0.0
            else:
                best_action = 'hold'
                combined_confidence = 0.0
            
            result = CombinationResult(
                combined_action=best_action,
                combined_confidence=combined_confidence,
                individual_predictions=predictions,
                combination_weights=self.weights.copy(),
                method=self.name,
                metadata={
                    'weight_method': self.weight_method,
                    'total_weight': sum(self.weights.values()),
                    'active_strategies': len([w for w in self.weights.values() if w > 0])
                }
            )
            
            self.update_history(result)
            return result
            
        except Exception as e:
            logger.error(f"加权平均组合出错: {e}")
            return CombinationResult(
                combined_action=None,
                combined_confidence=0.0,
                individual_predictions=predictions,
                combination_weights={},
                method=self.name,
                metadata={'error': str(e)}
            )
    
    def fit(self, historical_data: List[Dict[str, Any]]):
        """训练权重"""
        try:
            if not historical_data:
                return
            
            # 收集性能数据
            for data in historical_data:
                strategy_id = data.get('strategy_id')
                performance = data.get('performance', 0.0)
                if strategy_id:
                    self.performance_history[strategy_id].append(performance)
            
            # 计算权重
            self._calculate_weights()
            self.is_fitted = True
            
            logger.info(f"加权平均组合器训练完成，权重: {self.weights}")
            
        except Exception as e:
            logger.error(f"加权平均组合器训练出错: {e}")
    
    def update_weights(self, new_performance: Dict[str, float]):
        """更新权重"""
        try:
            for strategy_id, performance in new_performance.items():
                self.performance_history[strategy_id].append(performance)
            
            # 定期重新计算权重
            self.last_rebalance += 1
            if self.last_rebalance >= self.rebalance_frequency:
                self._calculate_weights()
                self.last_rebalance = 0
                
        except Exception as e:
            logger.error(f"更新权重出错: {e}")
    
    def _initialize_weights(self, predictions: Dict[str, Any]):
        """初始化权重"""
        if self.weight_method == 'equal':
            weight = 1.0 / len(predictions)
            self.weights = {strategy_id: weight for strategy_id in predictions.keys()}
        elif self.weight_method == 'confidence_based':
            total_confidence = sum(pred.get('confidence', 0.5) for pred in predictions.values())
            if total_confidence > 0:
                self.weights = {
                    strategy_id: pred.get('confidence', 0.5) / total_confidence
                    for strategy_id, pred in predictions.items()
                }
            else:
                weight = 1.0 / len(predictions)
                self.weights = {strategy_id: weight for strategy_id in predictions.keys()}
        else:  # performance_based
            # 使用默认权重，后续通过训练更新
            weight = 1.0 / len(predictions)
            self.weights = {strategy_id: weight for strategy_id in predictions.keys()}
    
    def _calculate_weights(self):
        """计算权重"""
        if self.weight_method == 'equal':
            weight = 1.0 / len(self.performance_history) if self.performance_history else 1.0
            self.weights = {strategy_id: weight for strategy_id in self.performance_history.keys()}
            
        elif self.weight_method == 'performance_based':
            # 基于历史性能计算权重
            performance_scores = {}
            for strategy_id, performances in self.performance_history.items():
                if performances:
                    # 使用指数加权平均
                    weights = np.exp(-np.arange(len(performances)) * 0.1)
                    weighted_avg = np.average(performances, weights=weights)
                    performance_scores[strategy_id] = weighted_avg
                else:
                    performance_scores[strategy_id] = 0.0
            
            # 归一化权重
            total_score = sum(max(0, score) for score in performance_scores.values())
            if total_score > 0:
                self.weights = {
                    strategy_id: max(0, score) / total_score
                    for strategy_id, score in performance_scores.items()
                }
            else:
                # 平均分配
                weight = 1.0 / len(performance_scores)
                self.weights = {strategy_id: weight for strategy_id in performance_scores.keys()}
            
        elif self.weight_method == 'confidence_based':
            # 基于置信度计算权重（需要历史置信度数据）
            # 简化实现：使用性能数据作为代理
            self._calculate_weights()  # 回退到性能基础权重

class VotingCombiner(BaseCombiner):
    """投票机制组合器"""
    
    def __init__(self, name: str = "voting", 
                 voting_method: str = 'weighted_voting',
                 confidence_threshold: float = 0.5):
        super().__init__(name)
        self.voting_method = voting_method
        self.confidence_threshold = confidence_threshold
        self.vote_weights = {}
        self.action_history = defaultdict(list)
    
    def combine(self, predictions: Dict[str, Any], 
               context: Optional[LearningContext] = None) -> CombinationResult:
        """投票组合"""
        try:
            if not predictions:
                return CombinationResult(
                    combined_action='hold',
                    combined_confidence=0.0,
                    individual_predictions={},
                    combination_weights={},
                    method=self.name,
                    metadata={'error': '没有预测结果'}
                )
            
            # 收集投票
            votes = defaultdict(float)
            total_weight = 0
            
            for strategy_id, prediction in predictions.items():
                action = prediction.get('action', 'hold')
                confidence = prediction.get('confidence', 0.5)
                weight = self.vote_weights.get(strategy_id, 1.0)
                
                # 根据投票方法计算票数
                if self.voting_method == 'simple_voting':
                    votes[action] += 1
                elif self.voting_method == 'weighted_voting':
                    votes[action] += confidence * weight
                elif self.voting_method == 'threshold_voting':
                    if confidence >= self.confidence_threshold:
                        votes[action] += confidence * weight
                
                total_weight += weight
            
            # 决定最终动作
            if votes:
                best_action = max(votes, key=votes.get)
                max_votes = votes[best_action]
                combined_confidence = max_votes / total_weight if total_weight > 0 else 0.0
            else:
                best_action = 'hold'
                combined_confidence = 0.0
            
            # 记录投票结果
            self._record_voting_results(predictions, best_action, combined_confidence)
            
            result = CombinationResult(
                combined_action=best_action,
                combined_confidence=combined_confidence,
                individual_predictions=predictions,
                combination_weights=self.vote_weights.copy(),
                method=self.name,
                metadata={
                    'voting_method': self.voting_method,
                    'votes': dict(votes),
                    'confidence_threshold': self.confidence_threshold,
                    'total_votes': total_weight
                }
            )
            
            self.update_history(result)
            return result
            
        except Exception as e:
            logger.error(f"投票组合出错: {e}")
            return CombinationResult(
                combined_action=None,
                combined_confidence=0.0,
                individual_predictions=predictions,
                combination_weights={},
                method=self.name,
                metadata={'error': str(e)}
            )
    
    def fit(self, historical_data: List[Dict[str, Any]]):
        """训练投票权重"""
        try:
            # 分析历史投票表现
            for data in historical_data:
                strategy_id = data.get('strategy_id')
                action = data.get('action')
                success = data.get('success', False)
                
                if strategy_id and action:
                    self.action_history[strategy_id].append({
                        'action': action,
                        'success': success,
                        'confidence': data.get('confidence', 0.5)
                    })
            
            # 计算投票权重
            self._calculate_vote_weights()
            self.is_fitted = True
            
            logger.info(f"投票组合器训练完成，权重: {self.vote_weights}")
            
        except Exception as e:
            logger.error(f"投票组合器训练出错: {e}")
    
    def _calculate_vote_weights(self):
        """计算投票权重"""
        vote_weights = {}
        
        for strategy_id, history in self.action_history.items():
            if history:
                # 计算策略成功率
                successes = sum(1 for h in history if h['success'])
                total_votes = len(history)
                success_rate = successes / total_votes if total_votes > 0 else 0.5
                
                # 计算置信度准确性
                high_conf_predictions = [h for h in history if h['confidence'] > 0.7]
                if high_conf_predictions:
                    high_conf_success = sum(1 for h in high_conf_predictions if h['success'])
                    confidence_accuracy = high_conf_success / len(high_conf_predictions)
                else:
                    confidence_accuracy = success_rate
                
                # 综合权重
                vote_weights[strategy_id] = (success_rate * 0.7 + confidence_accuracy * 0.3)
            else:
                vote_weights[strategy_id] = 0.5  # 默认权重
        
        # 归一化权重
        total_weight = sum(vote_weights.values())
        if total_weight > 0:
            self.vote_weights = {
                strategy_id: weight / total_weight
                for strategy_id, weight in vote_weights.items()
            }
        else:
            # 平均分配
            weight = 1.0 / len(vote_weights) if vote_weights else 1.0
            self.vote_weights = {strategy_id: weight for strategy_id in vote_weights.keys()}
    
    def _record_voting_results(self, predictions: Dict[str, Any], 
                             final_action: str, confidence: float):
        """记录投票结果"""
        # 这里可以记录详细的投票信息用于后续分析
        pass

class StackingCombiner(BaseCombiner):
    """堆叠组合器（元学习）"""
    
    def __init__(self, name: str = "stacking", 
                 meta_learner: str = 'ridge',
                 use_features: bool = True):
        super().__init__(name)
        self.meta_learner_name = meta_learner
        self.use_features = use_features
        self.meta_learner = None
        self.feature_extractors = {}
        self.training_data = []
        
        # 初始化元学习器
        if meta_learner == 'ridge':
            self.meta_learner = Ridge(alpha=1.0)
        elif meta_learner == 'lasso':
            self.meta_learner = Lasso(alpha=0.1)
        elif meta_learner == 'random_forest':
            self.meta_learner = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.meta_learner = LinearRegression()
    
    def combine(self, predictions: Dict[str, Any], 
               context: Optional[LearningContext] = None) -> CombinationResult:
        """堆叠组合"""
        try:
            if not predictions:
                return CombinationResult(
                    combined_action='hold',
                    combined_confidence=0.0,
                    individual_predictions={},
                    combination_weights={},
                    method=self.name,
                    metadata={'error': '没有预测结果'}
                )
            
            # 提取特征
            features = self._extract_features(predictions, context)
            
            if self.is_fitted and self.meta_learner is not None:
                # 使用训练好的元学习器进行预测
                try:
                    feature_vector = np.array(list(features.values())).reshape(1, -1)
                    predicted_action_prob = self.meta_learner.predict(feature_vector)[0]
                    
                    # 转换为动作
                    if predicted_action_prob > 0.5:
                        combined_action = 'buy'
                        combined_confidence = min(1.0, predicted_action_prob)
                    else:
                        combined_action = 'sell' if predicted_action_prob < -0.5 else 'hold'
                        combined_confidence = min(1.0, abs(predicted_action_prob))
                    
                except Exception as e:
                    logger.warning(f"元学习器预测失败: {e}，回退到简单组合")
                    combined_action, combined_confidence = self._fallback_combination(predictions)
            else:
                # 回退到简单组合
                combined_action, combined_confidence = self._fallback_combination(predictions)
            
            result = CombinationResult(
                combined_action=combined_action,
                combined_confidence=combined_confidence,
                individual_predictions=predictions,
                combination_weights={},
                method=self.name,
                metadata={
                    'meta_learner': self.meta_learner_name,
                    'features': features,
                    'predicted_probability': predicted_action_prob if self.is_fitted else None
                }
            )
            
            self.update_history(result)
            return result
            
        except Exception as e:
            logger.error(f"堆叠组合出错: {e}")
            return CombinationResult(
                combined_action=None,
                combined_confidence=0.0,
                individual_predictions=predictions,
                combination_weights={},
                method=self.name,
                metadata={'error': str(e)}
            )
    
    def fit(self, historical_data: List[Dict[str, Any]]):
        """训练元学习器"""
        try:
            if not historical_data:
                return
            
            # 准备训练数据
            X, y = [], []
            
            for data in historical_data:
                predictions = data.get('predictions', {})
                context = data.get('context')
                true_action = data.get('true_action')
                
                if predictions and true_action is not None:
                    features = self._extract_features(predictions, context)
                    
                    # 转换真实动作为数值
                    if true_action == 'buy':
                        y_val = 1
                    elif true_action == 'sell':
                        y_val = -1
                    else:
                        y_val = 0
                    
                    X.append(list(features.values()))
                    y.append(y_val)
            
            if len(X) < 10:
                logger.warning("训练数据不足，跳过元学习器训练")
                return
            
            # 训练元学习器
            X_array = np.array(X)
            y_array = np.array(y)
            
            self.meta_learner.fit(X_array, y_array)
            
            # 评估模型性能
            try:
                scores = cross_val_score(self.meta_learner, X_array, y_array, cv=5)
                self.feature_importance = {
                    'cv_score': np.mean(scores),
                    'feature_count': X_array.shape[1]
                }
            except:
                self.feature_importance = {'cv_score': 0.0, 'feature_count': X_array.shape[1]}
            
            self.is_fitted = True
            logger.info(f"堆叠组合器训练完成，CV得分: {self.feature_importance.get('cv_score', 0.0)}")
            
        except Exception as e:
            logger.error(f"堆叠组合器训练出错: {e}")
    
    def _extract_features(self, predictions: Dict[str, Any], 
                         context: Optional[LearningContext]) -> Dict[str, float]:
        """提取组合特征"""
        features = {}
        
        # 基础统计特征
        confidences = [pred.get('confidence', 0.5) for pred in predictions.values()]
        if confidences:
            features['avg_confidence'] = np.mean(confidences)
            features['max_confidence'] = max(confidences)
            features['min_confidence'] = min(confidences)
            features['confidence_std'] = np.std(confidences)
            features['confidence_range'] = max(confidences) - min(confidences)
        
        # 动作分布特征
        actions = [pred.get('action', 'hold') for pred in predictions.values()]
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action] += 1
        
        features['buy_votes'] = action_counts.get('buy', 0)
        features['sell_votes'] = action_counts.get('sell', 0)
        features['hold_votes'] = action_counts.get('hold', 0)
        features['action_diversity'] = len(set(actions))
        
        # 策略类型特征
        strategy_types = [pred.get('strategy_type', 'unknown') for pred in predictions.values()]
        type_counts = defaultdict(int)
        for stype in strategy_types:
            type_counts[stype] += 1
        features['strategy_type_count'] = len(type_counts)
        
        # 上下文特征
        if context:
            features['risk_tolerance'] = context.risk_tolerance
            features['time_horizon'] = context.time_horizon
            features['market_volatility'] = context.environment_state.get('volatility', 0.1)
            features['trend_strength'] = context.environment_state.get('trend_strength', 0.5)
        
        return features
    
    def _fallback_combination(self, predictions: Dict[str, Any]) -> Tuple[str, float]:
        """回退组合方法"""
        # 简单的加权平均
        action_scores = defaultdict(float)
        total_confidence = 0
        
        for prediction in predictions.values():
            action = prediction.get('action', 'hold')
            confidence = prediction.get('confidence', 0.5)
            action_scores[action] += confidence
            total_confidence += confidence
        
        if action_scores and total_confidence > 0:
            best_action = max(action_scores, key=action_scores.get)
            confidence = action_scores[best_action] / total_confidence
            return best_action, confidence
        else:
            return 'hold', 0.0

class AdaptiveCombiner(BaseCombiner):
    """自适应组合器"""
    
    def __init__(self, name: str = "adaptive", 
                 adaptation_method: str = 'performance_based',
                 adaptation_rate: float = 0.1):
        super().__init__(name)
        self.adaptation_method = adaptation_method
        self.adaptation_rate = adaptation_rate
        self.sub_combiners = {
            'weighted': WeightedAverageCombiner("adaptive_weighted"),
            'voting': VotingCombiner("adaptive_voting"),
            'stacking': StackingCombiner("adaptive_stacking")
        }
        self.combiner_weights = {'weighted': 0.4, 'voting': 0.3, 'stacking': 0.3}
        self.performance_tracker = defaultdict(list)
        self.current_best_combiner = 'weighted'
        
    def combine(self, predictions: Dict[str, Any], 
               context: Optional[LearningContext] = None) -> CombinationResult:
        """自适应组合"""
        try:
            if not predictions:
                return CombinationResult(
                    combined_action='hold',
                    combined_confidence=0.0,
                    individual_predictions={},
                    combination_weights={},
                    method=self.name,
                    metadata={'error': '没有预测结果'}
                )
            
            # 获取所有子组合器的结果
            sub_results = {}
            for combiner_name, combiner in self.sub_combiners.items():
                try:
                    result = combiner.combine(predictions, context)
                    sub_results[combiner_name] = result
                except Exception as e:
                    logger.warning(f"子组合器 {combiner_name} 失败: {e}")
                    continue
            
            if not sub_results:
                return CombinationResult(
                    combined_action='hold',
                    combined_confidence=0.0,
                    individual_predictions=predictions,
                    combination_weights={},
                    method=self.name,
                    metadata={'error': '所有子组合器都失败'}
                )
            
            # 自适应权重组合
            combined_action, combined_confidence = self._adaptive_weighted_voting(sub_results, context)
            
            # 更新组合器权重
            self._update_combiner_weights(sub_results, context)
            
            result = CombinationResult(
                combined_action=combined_action,
                combined_confidence=combined_confidence,
                individual_predictions=predictions,
                combination_weights=self.combiner_weights.copy(),
                method=self.name,
                metadata={
                    'sub_results': {name: {
                        'action': r.combined_action,
                        'confidence': r.combined_confidence
                    } for name, r in sub_results.items()},
                    'current_best': self.current_best_combiner,
                    'adaptation_method': self.adaptation_method
                }
            )
            
            self.update_history(result)
            return result
            
        except Exception as e:
            logger.error(f"自适应组合出错: {e}")
            return CombinationResult(
                combined_action=None,
                combined_confidence=0.0,
                individual_predictions=predictions,
                combination_weights={},
                method=self.name,
                metadata={'error': str(e)}
            )
    
    def fit(self, historical_data: List[Dict[str, Any]]):
        """训练自适应组合器"""
        try:
            # 训练所有子组合器
            for combiner in self.sub_combiners.values():
                combiner.fit(historical_data)
            
            # 初始化组合器权重
            self._initialize_combiner_weights()
            self.is_fitted = True
            
            logger.info(f"自适应组合器训练完成，权重: {self.combiner_weights}")
            
        except Exception as e:
            logger.error(f"自适应组合器训练出错: {e}")
    
    def _adaptive_weighted_voting(self, sub_results: Dict[str, CombinationResult],
                                context: Optional[LearningContext]) -> Tuple[str, float]:
        """自适应加权投票"""
        action_scores = defaultdict(float)
        total_weight = 0
        
        for combiner_name, result in sub_results.items():
            weight = self.combiner_weights.get(combiner_name, 0.1)
            action = result.combined_action
            confidence = result.combined_confidence
            
            action_scores[action] += confidence * weight
            total_weight += weight
        
        if action_scores and total_weight > 0:
            best_action = max(action_scores, key=action_scores.get)
            combined_confidence = action_scores[best_action] / total_weight
            return best_action, combined_confidence
        else:
            return 'hold', 0.0
    
    def _update_combiner_weights(self, sub_results: Dict[str, CombinationResult],
                               context: Optional[LearningContext]):
        """更新组合器权重"""
        try:
            # 基于性能更新权重
            for combiner_name, result in sub_results.items():
                performance_score = result.combined_confidence
                self.performance_tracker[combiner_name].append(performance_score)
                
                # 保持历史长度
                if len(self.performance_tracker[combiner_name]) > 50:
                    self.performance_tracker[combiner_name] = self.performance_tracker[combiner_name][-50:]
            
            # 计算新的权重
            recent_performance = {}
            for combiner_name, history in self.performance_tracker.items():
                if history:
                    # 指数加权平均
                    weights = np.exp(-np.arange(len(history)) * 0.1)
                    recent_performance[combiner_name] = np.average(history, weights=weights)
                else:
                    recent_performance[combiner_name] = 0.0
            
            # 更新权重
            if recent_performance:
                total_performance = sum(recent_performance.values())
                if total_performance > 0:
                    new_weights = {
                        combiner: perf / total_performance
                        for combiner, perf in recent_performance.items()
                    }
                    
                    # 混合更新
                    for combiner in self.combiner_weights:
                        old_weight = self.combiner_weights[combiner]
                        new_weight = new_weights.get(combiner, old_weight)
                        self.combiner_weights[combiner] = (
                            self.adaptation_rate * new_weight + 
                            (1 - self.adaptation_rate) * old_weight
                        )
                
                # 更新当前最佳组合器
                self.current_best_combiner = max(recent_performance, key=recent_performance.get)
                
        except Exception as e:
            logger.error(f"更新组合器权重出错: {e}")
    
    def _initialize_combiner_weights(self):
        """初始化组合器权重"""
        # 基于经验的初始权重
        self.combiner_weights = {
            'weighted': 0.4,  # 加权平均通常比较稳定
            'voting': 0.3,    # 投票机制简单有效
            'stacking': 0.3   # 堆叠方法在数据充足时表现好
        }

class RiskParityCombiner(BaseCombiner):
    """风险平价组合器"""
    
    def __init__(self, name: str = "risk_parity",
                 risk_measure: str = 'volatility',
                 target_risk: float = 0.15):
        super().__init__(name)
        self.risk_measure = risk_measure
        self.target_risk = target_risk
        self.risk_contributions = {}
        self.correlation_matrix = None
        self.covariance_matrix = None
        
    def combine(self, predictions: Dict[str, Any], 
               context: Optional[LearningContext] = None) -> CombinationResult:
        """风险平价组合"""
        try:
            if not predictions:
                return CombinationResult(
                    combined_action='hold',
                    combined_confidence=0.0,
                    individual_predictions={},
                    combination_weights={},
                    method=self.name,
                    metadata={'error': '没有预测结果'}
                )
            
            # 计算风险贡献
            self._calculate_risk_contributions(predictions)
            
            # 风险平价权重
            weights = self._calculate_risk_parity_weights()
            
            # 加权组合
            action_scores = defaultdict(float)
            total_weight = 0
            
            for strategy_id, prediction in predictions.items():
                if strategy_id in weights:
                    weight = weights[strategy_id]
                    action = prediction.get('action', 'hold')
                    confidence = prediction.get('confidence', 0.5)
                    
                    action_scores[action] += confidence * weight
                    total_weight += weight
            
            if action_scores and total_weight > 0:
                best_action = max(action_scores, key=action_scores.get)
                combined_confidence = action_scores[best_action] / total_weight
            else:
                best_action = 'hold'
                combined_confidence = 0.0
            
            result = CombinationResult(
                combined_action=best_action,
                combined_confidence=combined_confidence,
                individual_predictions=predictions,
                combination_weights=weights,
                method=self.name,
                metadata={
                    'risk_measure': self.risk_measure,
                    'target_risk': self.target_risk,
                    'risk_contributions': self.risk_contributions
                }
            )
            
            self.update_history(result)
            return result
            
        except Exception as e:
            logger.error(f"风险平价组合出错: {e}")
            return CombinationResult(
                combined_action=None,
                combined_confidence=0.0,
                individual_predictions=predictions,
                combination_weights={},
                method=self.name,
                metadata={'error': str(e)}
            )
    
    def fit(self, historical_data: List[Dict[str, Any]]):
        """训练风险平价组合器"""
        try:
            # 收集历史收益数据
            returns_data = defaultdict(list)
            
            for data in historical_data:
                strategy_id = data.get('strategy_id')
                returns = data.get('returns', [])
                if strategy_id and returns:
                    returns_data[strategy_id].extend(returns)
            
            # 计算协方差矩阵
            if len(returns_data) > 1:
                returns_matrix = []
                min_length = min(len(returns) for returns in returns_data.values())
                
                for strategy_id in returns_data:
                    if len(returns_data[strategy_id]) >= min_length:
                        returns_matrix.append(returns_data[strategy_id][:min_length])
                
                if returns_matrix:
                    returns_array = np.array(returns_matrix).T
                    self.covariance_matrix = np.cov(returns_array.T)
                    self.correlation_matrix = np.corrcoef(returns_array.T)
            
            self.is_fitted = True
            logger.info("风险平价组合器训练完成")
            
        except Exception as e:
            logger.error(f"风险平价组合器训练出错: {e}")
    
    def _calculate_risk_contributions(self, predictions: Dict[str, Any]):
        """计算风险贡献"""
        # 简化的风险贡献计算
        for strategy_id in predictions.keys():
            # 使用置信度作为风险的代理
            confidence = predictions[strategy_id].get('confidence', 0.5)
            risk = 1.0 - confidence  # 置信度越低，风险越高
            self.risk_contributions[strategy_id] = risk
    
    def _calculate_risk_parity_weights(self) -> Dict[str, float]:
        """计算风险平价权重"""
        if not self.risk_contributions:
            # 平均权重
            n_strategies = len(self.risk_contributions)
            return {sid: 1.0/n_strategies for sid in self.risk_contributions.keys()}
        
        # 风险平价：每个策略贡献相等风险
        total_risk = sum(self.risk_contributions.values())
        if total_risk == 0:
            n_strategies = len(self.risk_contributions)
            return {sid: 1.0/n_strategies for sid in self.risk_contributions.keys()}
        
        weights = {}
        for strategy_id, risk in self.risk_contributions.items():
            # 风险平价权重 = 1 / (n * 个体风险)
            weights[strategy_id] = 1.0 / (len(self.risk_contributions) * risk)
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {sid: w / total_weight for sid, w in weights.items()}
        
        return weights

class StrategyCombiner:
    """策略组合器主类"""
    
    def __init__(self, default_method: str = 'adaptive'):
        self.default_method = default_method
        self.combiners = {
            'weighted_average': WeightedAverageCombiner(),
            'voting': VotingCombiner(),
            'stacking': StackingCombiner(),
            'adaptive': AdaptiveCombiner(),
            'risk_parity': RiskParityCombiner()
        }
        self.current_combiner = self.combiners.get(default_method, self.combiners['adaptive'])
        self.combination_history = []
        self.performance_tracker = {}
        
    def combine_strategies(self, strategies: List[BaseStrategy], 
                          method: Optional[str] = None,
                          context: Optional[LearningContext] = None) -> CombinationResult:
        """组合策略"""
        try:
            if len(strategies) < 2:
                return CombinationResult(
                    combined_action='hold',
                    combined_confidence=0.0,
                    individual_predictions={},
                    combination_weights={},
                    method='single_strategy',
                    metadata={'error': '至少需要2个策略进行组合'}
                )
            
            # 选择组合方法
            combiner = self._select_combiner(method, context)
            
            # 获取策略预测
            predictions = {}
            for strategy in strategies:
                if strategy.is_active:
                    try:
                        if context:
                            prediction = strategy.predict(context.environment_state)
                        else:
                            prediction = strategy.predict({})
                        
                        predictions[strategy.strategy_id] = {
                            **prediction,
                            'strategy_type': strategy.strategy_type.value
                        }
                    except Exception as e:
                        logger.warning(f"策略 {strategy.strategy_id} 预测失败: {e}")
                        continue
            
            if not predictions:
                return CombinationResult(
                    combined_action='hold',
                    combined_confidence=0.0,
                    individual_predictions={},
                    combination_weights={},
                    method=combiner.name,
                    metadata={'error': '没有可用的策略预测'}
                )
            
            # 执行组合
            result = combiner.combine(predictions, context)
            
            # 记录组合历史
            self.combination_history.append({
                'timestamp': datetime.now(),
                'method': combiner.name,
                'n_strategies': len(strategies),
                'result': result
            })
            
            # 更新性能跟踪
            self._update_performance_tracking(result)
            
            return result
            
        except Exception as e:
            logger.error(f"策略组合出错: {e}")
            return CombinationResult(
                combined_action=None,
                combined_confidence=0.0,
                individual_predictions={},
                combination_weights={},
                method='error',
                metadata={'error': str(e)}
            )
    
    def optimize_combination(self, historical_data: List[Dict[str, Any]],
                           optimization_objective: str = 'sharpe_ratio') -> Dict[str, Any]:
        """优化组合方法"""
        try:
            optimization_results = {}
            
            # 测试不同的组合方法
            for method_name, combiner in self.combiners.items():
                try:
                    # 训练组合器
                    combiner.fit(historical_data)
                    
                    # 评估性能
                    performance_score = self._evaluate_combiner_performance(combiner, historical_data)
                    
                    optimization_results[method_name] = {
                        'performance_score': performance_score,
                        'is_fitted': combiner.is_fitted,
                        'method': method_name
                    }
                    
                except Exception as e:
                    logger.warning(f"组合方法 {method_name} 优化失败: {e}")
                    optimization_results[method_name] = {
                        'performance_score': 0.0,
                        'error': str(e)
                    }
            
            # 选择最佳方法
            best_method = max(
                optimization_results.keys(),
                key=lambda x: optimization_results[x].get('performance_score', 0.0)
            )
            
            # 更新当前组合器
            self.current_combiner = self.combiners[best_method]
            
            return {
                'optimization_results': optimization_results,
                'best_method': best_method,
                'best_score': optimization_results[best_method]['performance_score'],
                'optimization_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"组合优化出错: {e}")
            return {'error': str(e)}
    
    def _select_combiner(self, method: Optional[str], 
                        context: Optional[LearningContext]) -> BaseCombiner:
        """选择组合器"""
        if method and method in self.combiners:
            return self.combiners[method]
        
        # 自适应选择
        if context:
            market_volatility = context.environment_state.get('volatility', 0.1)
            trend_strength = context.environment_state.get('trend_strength', 0.5)
            
            # 基于市场条件选择
            if market_volatility > 0.3:  # 高波动
                return self.combiners['risk_parity']
            elif trend_strength > 0.7:  # 强趋势
                return self.combiners['stacking']
            else:
                return self.combiners['adaptive']
        
        return self.current_combiner
    
    def _evaluate_combiner_performance(self, combiner: BaseCombiner, 
                                     historical_data: List[Dict[str, Any]]) -> float:
        """评估组合器性能"""
        try:
            # 简化的性能评估
            if not hasattr(combiner, 'feature_importance'):
                return 0.5  # 默认分数
            
            cv_score = combiner.feature_importance.get('cv_score', 0.0)
            return max(0.0, min(1.0, cv_score))
            
        except Exception as e:
            logger.error(f"评估组合器性能出错: {e}")
            return 0.0
    
    def _update_performance_tracking(self, result: CombinationResult):
        """更新性能跟踪"""
        method = result.method
        confidence = result.combined_confidence
        
        if method not in self.performance_tracker:
            self.performance_tracker[method] = []
        
        self.performance_tracker[method].append({
            'timestamp': datetime.now(),
            'confidence': confidence,
            'success': confidence > 0.5
        })
        
        # 保持历史长度
        if len(self.performance_tracker[method]) > 100:
            self.performance_tracker[method] = self.performance_tracker[method][-100:]
    
    def get_combination_statistics(self) -> Dict[str, Any]:
        """获取组合统计信息"""
        stats = {
            'total_combinations': len(self.combination_history),
            'methods_used': {},
            'average_confidence': {},
            'success_rates': {}
        }
        
        for method, history in self.performance_tracker.items():
            confidences = [h['confidence'] for h in history]
            successes = [h['success'] for h in history]
            
            stats['methods_used'][method] = len(history)
            stats['average_confidence'][method] = np.mean(confidences) if confidences else 0.0
            stats['success_rates'][method] = np.mean(successes) if successes else 0.0
        
        return stats
    
    def export_combination_history(self, filepath: str) -> bool:
        """导出组合历史"""
        try:
            export_data = {
                'combination_history': [
                    {
                        'timestamp': h['timestamp'].isoformat(),
                        'method': h['method'],
                        'n_strategies': h['n_strategies'],
                        'combined_action': h['result'].combined_action,
                        'combined_confidence': h['result'].combined_confidence
                    }
                    for h in self.combination_history
                ],
                'performance_tracker': {
                    method: [
                        {
                            'timestamp': h['timestamp'].isoformat(),
                            'confidence': h['confidence'],
                            'success': h['success']
                        }
                        for h in history
                    ]
                    for method, history in self.performance_tracker.items()
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"导出组合历史出错: {e}")
            return False