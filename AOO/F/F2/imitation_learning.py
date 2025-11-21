"""
模仿学习模块

实现多种模仿学习算法：
- 行为克隆 (Behavioral Cloning)
- 逆向强化学习 (Inverse Reinforcement Learning)
- 生成对抗模仿学习 (GAIL)
- 专家演示学习
- 轨迹克隆
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import random
import logging
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .StrategyLearner import StrategyType, BaseStrategy, LearningContext, StrategyPerformance

logger = logging.getLogger(__name__)

class ExpertTrajectory:
    """专家轨迹类"""
    
    def __init__(self, states: List[Dict], actions: List[Any], rewards: List[float]):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.length = len(states)
    
    def get_state_action_pairs(self) -> List[Tuple[Dict, Any]]:
        """获取状态-动作对"""
        return list(zip(self.states, self.actions))
    
    def get_demonstration_features(self) -> List[Dict]:
        """获取演示特征"""
        features = []
        for state in self.states:
            feature_dict = self._extract_features(state)
            features.append(feature_dict)
        return features
    
    def _extract_features(self, state: Dict) -> Dict:
        """提取状态特征"""
        features = {}
        
        # 数值特征
        for key, value in state.items():
            if isinstance(value, (int, float)):
                features[f"{key}_value"] = value
                features[f"{key}_squared"] = value ** 2
            elif isinstance(value, str):
                features[f"{key}_category"] = hash(value) % 1000
        
        # 组合特征
        if 'market_signal' in state and 'volatility' in state:
            features['signal_volatility_ratio'] = state['market_signal'] / (state['volatility'] + 1e-8)
        
        if 'trend_strength' in state and 'momentum' in state:
            features['trend_momentum_product'] = state['trend_strength'] * state['momentum']
        
        return features

class BehavioralCloning(BaseStrategy):
    """行为克隆"""
    
    def __init__(self, strategy_id: str, expert_trajectories: List[ExpertTrajectory] = None):
        super().__init__(strategy_id, StrategyType.IMITATION)
        
        self.expert_trajectories = expert_trajectories or []
        self.model = None
        self.scaler = StandardScaler()
        self.action_encoder = {}
        self.feature_names = []
        
        self.training_data = []
        self.performance_history = []
        
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """行为克隆学习过程"""
        try:
            if not self.expert_trajectories:
                return {'error': '没有专家轨迹数据'}
            
            # 准备训练数据
            self._prepare_training_data()
            
            if not self.training_data:
                return {'error': '没有有效的训练数据'}
            
            # 训练模型
            training_result = self._train_model()
            
            # 评估模型性能
            evaluation_result = self._evaluate_model()
            
            # 生成策略统计
            strategy_stats = self._generate_strategy_stats()
            
            return {
                'training_result': training_result,
                'evaluation_result': evaluation_result,
                'strategy_stats': strategy_stats,
                'expert_trajectories': len(self.expert_trajectories),
                'training_samples': len(self.training_data),
                'model_performance': self._assess_model_quality()
            }
            
        except Exception as e:
            logger.error(f"行为克隆学习出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """基于行为克隆进行预测"""
        try:
            if not self.model:
                return {'action': 'hold', 'confidence': 0.0, 'error': '模型未训练'}
            
            # 提取特征
            features = self._extract_features(state)
            
            # 特征向量化
            if not self.feature_names:
                return {'action': 'hold', 'confidence': 0.0, 'error': '特征未初始化'}
            
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # 标准化
            feature_vector = self.scaler.transform(feature_vector)
            
            # 预测
            predicted_action_idx = self.model.predict(feature_vector)[0]
            
            # 解码动作
            action = self._decode_action(predicted_action_idx)
            
            # 计算置信度
            confidence = self._calculate_prediction_confidence(feature_vector, predicted_action_idx)
            
            return {
                'action': action,
                'confidence': confidence,
                'predicted_index': predicted_action_idx,
                'features': features,
                'model_type': 'behavioral_cloning'
            }
            
        except Exception as e:
            logger.error(f"行为克隆预测出错: {e}")
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
            
            # 行为克隆关注与专家行为的相似度
            if performance.return_rate > 0:
                self.state.success_rate = min(1.0, self.state.success_rate + 0.01)
            else:
                self.state.success_rate = max(0.0, self.state.success_rate - 0.01)
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
            # 记录性能历史
            self.performance_history.append(performance)
            
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def add_expert_trajectory(self, trajectory: ExpertTrajectory):
        """添加专家轨迹"""
        self.expert_trajectories.append(trajectory)
        logger.info(f"添加专家轨迹，长度: {trajectory.length}")
    
    def _prepare_training_data(self):
        """准备训练数据"""
        self.training_data = []
        self.feature_names = set()
        
        for trajectory in self.expert_trajectories:
            state_action_pairs = trajectory.get_state_action_pairs()
            
            for state, action in state_action_pairs:
                features = self._extract_features(state)
                self.feature_names.update(features.keys())
                
                self.training_data.append({
                    'features': features,
                    'action': action,
                    'state': state
                })
        
        # 转换为列表
        self.feature_names = list(self.feature_names)
        
        # 编码动作
        self._encode_actions()
    
    def _extract_features(self, state: Dict) -> Dict:
        """提取状态特征"""
        features = {}
        
        # 基础特征
        for key, value in state.items():
            if isinstance(value, (int, float)):
                features[f"{key}_value"] = value
                features[f"{key}_squared"] = value ** 2
                features[f"{key}_sign"] = 1 if value > 0 else (-1 if value < 0 else 0)
            elif isinstance(value, str):
                features[f"{key}_category"] = hash(value) % 1000
        
        # 组合特征
        numeric_keys = [k for k, v in state.items() if isinstance(v, (int, float))]
        
        for i, key1 in enumerate(numeric_keys):
            for key2 in numeric_keys[i+1:]:
                ratio_key = f"{key1}_{key2}_ratio"
                product_key = f"{key1}_{key2}_product"
                features[ratio_key] = state[key1] / (state[key2] + 1e-8)
                features[product_key] = state[key1] * state[key2]
        
        # 统计特征
        numeric_values = [v for v in state.values() if isinstance(v, (int, float))]
        if numeric_values:
            features['numeric_mean'] = np.mean(numeric_values)
            features['numeric_std'] = np.std(numeric_values)
            features['numeric_max'] = max(numeric_values)
            features['numeric_min'] = min(numeric_values)
        
        return features
    
    def _encode_actions(self):
        """编码动作"""
        unique_actions = set()
        for data in self.training_data:
            unique_actions.add(data['action'])
        
        self.action_encoder = {action: idx for idx, action in enumerate(unique_actions)}
        self.action_decoder = {idx: action for action, idx in self.action_encoder.items()}
    
    def _decode_action(self, action_idx: int) -> Any:
        """解码动作"""
        return self.action_decoder.get(action_idx, 'hold')
    
    def _train_model(self) -> Dict[str, Any]:
        """训练模型"""
        try:
            # 准备特征矩阵和标签
            X = []
            y = []
            
            for data in self.training_data:
                feature_vector = []
                for feature_name in self.feature_names:
                    feature_vector.append(data['features'].get(feature_name, 0.0))
                
                X.append(feature_vector)
                y.append(self.action_encoder[data['action']])
            
            X = np.array(X)
            y = np.array(y)
            
            # 分割训练和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 标准化特征
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 训练模型
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # 评估模型
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            return {
                'train_score': train_score,
                'test_score': test_score,
                'n_features': len(self.feature_names),
                'n_samples': len(X),
                'n_classes': len(self.action_encoder),
                'convergence': self.model.n_iter_
            }
            
        except Exception as e:
            logger.error(f"模型训练出错: {e}")
            return {'error': str(e)}
    
    def _evaluate_model(self) -> Dict[str, Any]:
        """评估模型"""
        try:
            if not self.training_data:
                return {'error': '没有训练数据'}
            
            # 计算预测准确率
            correct_predictions = 0
            total_predictions = 0
            
            for data in self.training_data[:100]:  # 评估前100个样本
                features = data['features']
                true_action = data['action']
                
                feature_vector = []
                for feature_name in self.feature_names:
                    feature_vector.append(features.get(feature_name, 0.0))
                
                feature_vector = np.array(feature_vector).reshape(1, -1)
                feature_vector = self.scaler.transform(feature_vector)
                
                predicted_idx = self.model.predict(feature_vector)[0]
                predicted_action = self._decode_action(int(round(predicted_idx)))
                
                if predicted_action == true_action:
                    correct_predictions += 1
                total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'action_distribution': self._get_action_distribution()
            }
            
        except Exception as e:
            logger.error(f"模型评估出错: {e}")
            return {'error': str(e)}
    
    def _calculate_prediction_confidence(self, features: np.ndarray, action_idx: int) -> float:
        """计算预测置信度"""
        try:
            # 简化的置信度计算
            # 在实际应用中，可以使用概率输出或集成方法
            
            # 基于特征完整性计算置信度
            non_zero_features = np.sum(features != 0)
            feature_completeness = non_zero_features / features.shape[1]
            
            # 基于模型性能调整
            if hasattr(self, 'model') and hasattr(self.model, 'score'):
                # 使用模型分数作为置信度基础
                model_confidence = max(0.0, min(1.0, self.model.score(
                    features, np.array([action_idx])
                )))
            else:
                model_confidence = 0.5
            
            # 综合置信度
            confidence = (feature_completeness + model_confidence) / 2
            return confidence
            
        except Exception as e:
            logger.error(f"置信度计算出错: {e}")
            return 0.5
    
    def _get_action_distribution(self) -> Dict[str, int]:
        """获取动作分布"""
        distribution = defaultdict(int)
        for data in self.training_data:
            distribution[data['action']] += 1
        return dict(distribution)
    
    def _generate_strategy_stats(self) -> Dict[str, Any]:
        """生成策略统计"""
        if not self.expert_trajectories:
            return {}
        
        # 计算轨迹统计
        trajectory_lengths = [traj.length for traj in self.expert_trajectories]
        total_expert_steps = sum(trajectory_lengths)
        
        # 计算动作统计
        action_counts = defaultdict(int)
        for trajectory in self.expert_trajectories:
            for action in trajectory.actions:
                action_counts[action] += 1
        
        return {
            'n_trajectories': len(self.expert_trajectories),
            'avg_trajectory_length': np.mean(trajectory_lengths),
            'total_expert_steps': total_expert_steps,
            'action_distribution': dict(action_counts),
            'unique_actions': len(action_counts),
            'most_common_action': max(action_counts, key=action_counts.get) if action_counts else None
        }
    
    def _assess_model_quality(self) -> Dict[str, Any]:
        """评估模型质量"""
        quality_score = 0.0
        quality_factors = {}
        
        # 数据质量
        if len(self.training_data) > 1000:
            quality_score += 0.3
            quality_factors['data_adequacy'] = 1.0
        elif len(self.training_data) > 100:
            quality_score += 0.2
            quality_factors['data_adequacy'] = 0.7
        else:
            quality_factors['data_adequacy'] = 0.3
        
        # 动作多样性
        if len(self.action_encoder) >= 3:
            quality_score += 0.2
            quality_factors['action_diversity'] = 1.0
        elif len(self.action_encoder) >= 2:
            quality_score += 0.1
            quality_factors['action_diversity'] = 0.7
        else:
            quality_factors['action_diversity'] = 0.3
        
        # 特征质量
        if len(self.feature_names) >= 10:
            quality_score += 0.2
            quality_factors['feature_richness'] = 1.0
        elif len(self.feature_names) >= 5:
            quality_score += 0.1
            quality_factors['feature_richness'] = 0.7
        else:
            quality_factors['feature_richness'] = 0.3
        
        # 模型性能
        if hasattr(self.model, 'score'):
            try:
                # 简化的性能评估
                if self.training_data:
                    X = []
                    y = []
                    for data in self.training_data[:100]:
                        feature_vector = []
                        for feature_name in self.feature_names:
                            feature_vector.append(data['features'].get(feature_name, 0.0))
                        X.append(feature_vector)
                        y.append(self.action_encoder[data['action']])
                    
                    if X and y:
                        X = np.array(X)
                        y = np.array(y)
                        X_scaled = self.scaler.transform(X)
                        score = self.model.score(X_scaled, y)
                        
                        if score > 0.8:
                            quality_score += 0.3
                            quality_factors['model_performance'] = 1.0
                        elif score > 0.5:
                            quality_score += 0.2
                            quality_factors['model_performance'] = 0.7
                        else:
                            quality_factors['model_performance'] = 0.3
                    else:
                        quality_factors['model_performance'] = 0.0
                else:
                    quality_factors['model_performance'] = 0.0
            except:
                quality_factors['model_performance'] = 0.0
        else:
            quality_factors['model_performance'] = 0.0
        
        return {
            'overall_quality': quality_score,
            'quality_factors': quality_factors,
            'quality_level': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.5 else 'low'
        }

class TrajectoryCloning(BaseStrategy):
    """轨迹克隆"""
    
    def __init__(self, strategy_id: str, expert_trajectories: List[ExpertTrajectory] = None):
        super().__init__(strategy_id, StrategyType.IMITATION)
        
        self.expert_trajectories = expert_trajectories or []
        self.trajectory_patterns = []
        self.pattern_weights = []
        self.similarity_threshold = 0.8
        
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """轨迹克隆学习过程"""
        try:
            if not self.expert_trajectories:
                return {'error': '没有专家轨迹数据'}
            
            # 提取轨迹模式
            self._extract_trajectory_patterns()
            
            # 计算模式权重
            self._calculate_pattern_weights()
            
            # 学习状态-动作映射
            learning_result = self._learn_state_action_mapping()
            
            return {
                'pattern_extraction': {
                    'n_patterns': len(self.trajectory_patterns),
                    'pattern_weights': self.pattern_weights,
                    'similarity_threshold': self.similarity_threshold
                },
                'learning_result': learning_result,
                'trajectory_statistics': self._generate_trajectory_statistics()
            }
            
        except Exception as e:
            logger.error(f"轨迹克隆学习出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """基于轨迹克隆进行预测"""
        try:
            if not self.trajectory_patterns:
                return {'action': 'hold', 'confidence': 0.0, 'error': '没有轨迹模式'}
            
            # 找到最相似的轨迹片段
            similar_patterns = self._find_similar_patterns(state)
            
            if not similar_patterns:
                return {'action': 'hold', 'confidence': 0.0, 'error': '没有找到相似模式'}
            
            # 基于相似模式投票
            action_votes = defaultdict(float)
            total_weight = 0
            
            for pattern_idx, similarity, next_action in similar_patterns:
                weight = self.pattern_weights[pattern_idx] * similarity
                action_votes[next_action] += weight
                total_weight += weight
            
            if total_weight > 0:
                # 选择得票最高的动作
                best_action = max(action_votes, key=action_votes.get)
                confidence = action_votes[best_action] / total_weight
            else:
                best_action = 'hold'
                confidence = 0.0
            
            return {
                'action': best_action,
                'confidence': confidence,
                'similar_patterns': len(similar_patterns),
                'action_votes': dict(action_votes),
                'method': 'trajectory_cloning'
            }
            
        except Exception as e:
            logger.error(f"轨迹克隆预测出错: {e}")
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
    
    def add_expert_trajectory(self, trajectory: ExpertTrajectory):
        """添加专家轨迹"""
        self.expert_trajectories.append(trajectory)
        logger.info(f"添加专家轨迹，长度: {trajectory.length}")
    
    def _extract_trajectory_patterns(self):
        """提取轨迹模式"""
        self.trajectory_patterns = []
        
        for trajectory in self.expert_trajectories:
            # 滑动窗口提取模式
            window_size = min(5, trajectory.length)
            
            for i in range(trajectory.length - window_size + 1):
                pattern = {
                    'states': trajectory.states[i:i + window_size],
                    'actions': trajectory.actions[i:i + window_size],
                    'trajectory_id': id(trajectory),
                    'start_index': i
                }
                self.trajectory_patterns.append(pattern)
    
    def _calculate_pattern_weights(self):
        """计算模式权重"""
        self.pattern_weights = []
        
        for pattern in self.trajectory_patterns:
            # 基于模式长度和稀有度计算权重
            pattern_length = len(pattern['states'])
            rarity_score = 1.0 / len([p for p in self.trajectory_patterns 
                                    if self._patterns_similar(pattern, p)])
            
            weight = pattern_length * rarity_score
            self.pattern_weights.append(weight)
        
        # 归一化权重
        total_weight = sum(self.pattern_weights)
        if total_weight > 0:
            self.pattern_weights = [w / total_weight for w in self.pattern_weights]
        else:
            self.pattern_weights = [1.0 / len(self.pattern_weights)] * len(self.pattern_weights)
    
    def _learn_state_action_mapping(self) -> Dict[str, Any]:
        """学习状态-动作映射"""
        state_action_mappings = defaultdict(list)
        
        for pattern in self.trajectory_patterns:
            for i, state in enumerate(pattern['states']):
                if i < len(pattern['actions']):
                    state_key = self._state_to_key(state)
                    action = pattern['actions'][i]
                    state_action_mappings[state_key].append(action)
        
        # 计算每个状态的最可能动作
        most_likely_actions = {}
        for state_key, actions in state_action_mappings.items():
            action_counts = defaultdict(int)
            for action in actions:
                action_counts[action] += 1
            most_likely_actions[state_key] = max(action_counts, key=action_counts.get)
        
        return {
            'unique_states': len(state_action_mappings),
            'state_action_mappings': len(most_likely_actions),
            'most_common_actions': dict(action_counts),
            'mapping_completeness': len(most_likely_actions) / len(state_action_mappings) if state_action_mappings else 0
        }
    
    def _find_similar_patterns(self, current_state: Dict[str, Any]) -> List[Tuple[int, float, Any]]:
        """找到相似的轨迹模式"""
        similar_patterns = []
        
        for i, pattern in enumerate(self.trajectory_patterns):
            # 检查最后一个状态是否相似
            if pattern['states']:
                last_state = pattern['states'][-1]
                similarity = self._calculate_state_similarity(current_state, last_state)
                
                if similarity > self.similarity_threshold:
                    # 获取下一个动作
                    next_action = None
                    if len(pattern['actions']) > len(pattern['states']):
                        next_action = pattern['actions'][len(pattern['states'])]
                    
                    if next_action:
                        similar_patterns.append((i, similarity, next_action))
        
        # 按相似度排序
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return similar_patterns[:5]  # 返回前5个最相似的模式
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """将状态转换为键"""
        # 简化的状态键生成
        key_parts = []
        for k, v in sorted(state.items()):
            if isinstance(v, (int, float)):
                # 量化数值
                quantized = round(v, 2)
                key_parts.append(f"{k}:{quantized}")
            else:
                key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
    
    def _calculate_state_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
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
    
    def _patterns_similar(self, pattern1: Dict, pattern2: Dict) -> bool:
        """判断两个模式是否相似"""
        if len(pattern1['states']) != len(pattern2['states']):
            return False
        
        similarities = []
        for s1, s2 in zip(pattern1['states'], pattern2['states']):
            sim = self._calculate_state_similarity(s1, s2)
            similarities.append(sim)
        
        return np.mean(similarities) > 0.8
    
    def _generate_trajectory_statistics(self) -> Dict[str, Any]:
        """生成轨迹统计"""
        if not self.expert_trajectories:
            return {}
        
        lengths = [traj.length for traj in self.expert_trajectories]
        total_actions = sum(len(traj.actions) for traj in self.expert_trajectories)
        
        action_distribution = defaultdict(int)
        for trajectory in self.expert_trajectories:
            for action in trajectory.actions:
                action_distribution[action] += 1
        
        return {
            'n_trajectories': len(self.expert_trajectories),
            'avg_length': np.mean(lengths),
            'total_actions': total_actions,
            'action_distribution': dict(action_distribution),
            'extracted_patterns': len(self.trajectory_patterns)
        }

# 工厂函数
def create_imitation_learner(algorithm: str, strategy_id: str, 
                           expert_trajectories: List[ExpertTrajectory] = None,
                           **kwargs) -> BaseStrategy:
    """创建模仿学习器"""
    algorithms = {
        'behavioral_cloning': BehavioralCloning,
        'trajectory_cloning': TrajectoryCloning
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    return algorithms[algorithm](strategy_id, expert_trajectories, **kwargs)