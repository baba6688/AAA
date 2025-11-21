"""
D1自我认知引擎
实现AI系统的自我认知、监控、学习和优化能力
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import threading
import logging
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CognitiveState:
    """认知状态数据类"""
    timestamp: float
    confidence_level: float
    accuracy_rate: float
    response_time: float
    error_rate: float
    learning_rate: float
    memory_utilization: float
    processing_load: float
    attention_focus: float
    creativity_index: float
    reasoning_quality: float

@dataclass
class CognitiveBias:
    """认知偏差数据类"""
    bias_type: str
    severity: float
    confidence: float
    detected_at: float
    correction_applied: bool
    correction_effectiveness: float

@dataclass
class LearningEvent:
    """学习事件数据类"""
    timestamp: float
    event_type: str
    context: Dict[str, Any]
    outcome: str
    learning_gain: float
    adaptation_applied: bool

@dataclass
class CapabilityModel:
    """能力模型数据类"""
    domain: str
    current_level: float
    potential_level: float
    learning_velocity: float
    last_updated: float
    confidence_interval: Tuple[float, float]
    dependencies: List[str]

class SelfCognitionEngine:
    """D1自我认知引擎主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化自我认知引擎"""
        self.config = config or self._default_config()
        
        # 状态监控
        self.cognitive_states = deque(maxlen=1000)
        self.current_state = None
        self.state_history = deque(maxlen=100)
        
        # 能力建模
        self.capability_models = {}
        self.performance_history = defaultdict(deque)
        self.learning_curves = defaultdict(list)
        
        # 认知偏差检测
        self.bias_detector = BiasDetector()
        self.detected_biases = []
        self.correction_history = deque(maxlen=500)
        
        # 自我反思和学习
        self.reflection_engine = ReflectionEngine()
        self.learning_events = deque(maxlen=1000)
        self.adaptation_history = deque(maxlen=100)
        
        # 评估系统
        self.evaluation_metrics = {}
        self.assessment_history = deque(maxlen=100)
        
        # 实时监控
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # 秒
        
        # 机器学习模型
        self.anomaly_detector = None
        self.performance_predictor = None
        self._initialize_ml_models()
        
        # 线程安全
        self.lock = threading.RLock()
        
        logger.info("D1自我认知引擎初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "monitoring": {
                "enabled": True,
                "interval": 1.0,
                "metrics": [
                    "confidence_level", "accuracy_rate", "response_time",
                    "error_rate", "learning_rate", "memory_utilization"
                ]
            },
            "bias_detection": {
                "enabled": True,
                "sensitivity": 0.8,
                "correction_enabled": True
            },
            "learning": {
                "enabled": True,
                "adaptation_rate": 0.1,
                "memory_decay": 0.95
            },
            "evaluation": {
                "dimensions": [
                    "reasoning", "memory", "attention", "creativity",
                    "learning", "problem_solving", "decision_making"
                ],
                "frequency": 60  # 秒
            }
        }
    
    def _initialize_ml_models(self):
        """初始化机器学习模型"""
        try:
            # 异常检测模型
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # 性能预测模型
            from sklearn.ensemble import RandomForestRegressor
            self.performance_predictor = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            
            logger.info("机器学习模型初始化完成")
        except Exception as e:
            logger.warning(f"机器学习模型初始化失败: {e}")
    
    def start_monitoring(self):
        """启动实时监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("实时监控已启动")
    
    def stop_monitoring(self):
        """停止实时监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("实时监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集当前状态
                current_state = self._collect_cognitive_state()
                
                with self.lock:
                    self.current_state = current_state
                    self.cognitive_states.append(current_state)
                    
                    # 实时分析
                    self._analyze_current_state(current_state)
                    
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_cognitive_state(self) -> CognitiveState:
        """收集当前认知状态"""
        # 模拟状态收集（实际应用中需要从系统获取真实指标）
        timestamp = time.time()
        
        # 生成模拟数据
        state = CognitiveState(
            timestamp=timestamp,
            confidence_level=np.random.normal(0.75, 0.15),
            accuracy_rate=np.random.normal(0.85, 0.1),
            response_time=np.random.normal(0.5, 0.2),
            error_rate=np.random.exponential(0.1),
            learning_rate=np.random.normal(0.1, 0.05),
            memory_utilization=np.random.normal(0.6, 0.2),
            processing_load=np.random.normal(0.4, 0.15),
            attention_focus=np.random.normal(0.8, 0.1),
            creativity_index=np.random.normal(0.7, 0.2),
            reasoning_quality=np.random.normal(0.75, 0.15)
        )
        
        # 确保值在合理范围内
        for field in state.__dataclass_fields__:
            if field != 'timestamp':
                value = getattr(state, field)
                if value < 0:
                    setattr(state, field, 0)
                elif field == 'confidence_level' and value > 1:
                    setattr(state, field, 1)
                elif field == 'accuracy_rate' and value > 1:
                    setattr(state, field, 1)
                elif field == 'response_time' and value > 10:
                    setattr(state, field, 10)
        
        return state
    
    def _analyze_current_state(self, state: CognitiveState):
        """分析当前状态"""
        # 异常检测
        if len(self.cognitive_states) > 10:
            self._detect_anomalies(state)
        
        # 认知偏差检测
        if self.config["bias_detection"]["enabled"]:
            self._detect_cognitive_biases(state)
        
        # 性能评估
        self._evaluate_performance(state)
        
        # 更新能力模型
        self._update_capability_models(state)
    
    def _detect_anomalies(self, current_state: CognitiveState):
        """检测异常状态"""
        try:
            if len(self.cognitive_states) < 20:
                return
            
            # 准备数据
            recent_states = list(self.cognitive_states)[-20:]
            features = []
            
            for state in recent_states:
                feature_vector = [
                    state.confidence_level, state.accuracy_rate, state.response_time,
                    state.error_rate, state.learning_rate, state.memory_utilization,
                    state.processing_load, state.attention_focus, state.creativity_index,
                    state.reasoning_quality
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # 训练异常检测模型
            if self.anomaly_detector is not None:
                self.anomaly_detector.fit(features[:-1])
                
                # 预测当前状态
                current_features = features[-1].reshape(1, -1)
                anomaly_score = self.anomaly_detector.decision_function(current_features)[0]
                is_anomaly = self.anomaly_detector.predict(current_features)[0] == -1
                
                if is_anomaly:
                    logger.warning(f"检测到异常状态: 分数={anomaly_score:.3f}")
                    
                    # 记录异常事件
                    anomaly_event = {
                        "timestamp": current_state.timestamp,
                        "type": "anomaly",
                        "score": anomaly_score,
                        "state": asdict(current_state)
                    }
                    self.learning_events.append(LearningEvent(
                        timestamp=current_state.timestamp,
                        event_type="anomaly_detection",
                        context=anomaly_event,
                        outcome="detected",
                        learning_gain=0.0,
                        adaptation_applied=False
                    ))
        
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
    
    def _detect_cognitive_biases(self, state: CognitiveState):
        """检测认知偏差"""
        try:
            biases = self.bias_detector.detect_biases(state, self.cognitive_states)
            
            for bias in biases:
                self.detected_biases.append(bias)
                
                # 应用纠正
                if self.config["bias_detection"]["correction_enabled"]:
                    correction = self._apply_bias_correction(bias)
                    bias.correction_applied = True
                    bias.correction_effectiveness = correction["effectiveness"]
                    
                    # 记录纠正历史
                    self.correction_history.append(correction)
                    
                    logger.info(f"检测到认知偏差: {bias.bias_type}, 严重程度: {bias.severity:.3f}")
        
        except Exception as e:
            logger.error(f"认知偏差检测失败: {e}")
    
    def _apply_bias_correction(self, bias: CognitiveBias) -> Dict[str, Any]:
        """应用认知偏差纠正"""
        correction_strategies = {
            "confirmation_bias": self._correct_confirmation_bias,
            "anchoring_bias": self._correct_anchoring_bias,
            "availability_bias": self._correct_availability_bias,
            "overconfidence_bias": self._correct_overconfidence_bias,
            "recency_bias": self._correct_recency_bias
        }
        
        strategy = correction_strategies.get(bias.bias_type, self._default_correction)
        correction_result = strategy(bias)
        
        return {
            "bias_type": bias.bias_type,
            "strategy": strategy.__name__,
            "effectiveness": correction_result["effectiveness"],
            "timestamp": time.time(),
            "details": correction_result
        }
    
    def _correct_confirmation_bias(self, bias: CognitiveBias) -> Dict[str, Any]:
        """纠正确认偏差"""
        return {
            "effectiveness": 0.8,
            "action": "主动寻找对立证据",
            "improvement": "提高决策客观性"
        }
    
    def _correct_anchoring_bias(self, bias: CognitiveBias) -> Dict[str, Any]:
        """纠正锚定偏差"""
        return {
            "effectiveness": 0.75,
            "action": "重新评估初始参考点",
            "improvement": "减少初始信息过度影响"
        }
    
    def _correct_availability_bias(self, bias: CognitiveBias) -> Dict[str, Any]:
        """纠正可得性偏差"""
        return {
            "effectiveness": 0.7,
            "action": "平衡考虑所有可用信息",
            "improvement": "提高信息处理全面性"
        }
    
    def _correct_overconfidence_bias(self, bias: CognitiveBias) -> Dict[str, Any]:
        """纠正过度自信偏差"""
        return {
            "effectiveness": 0.85,
            "action": "校准置信度评估",
            "improvement": "提高自我认知准确性"
        }
    
    def _correct_recency_bias(self, bias: CognitiveBias) -> Dict[str, Any]:
        """纠正近因偏差"""
        return {
            "effectiveness": 0.8,
            "action": "考虑历史数据权重",
            "improvement": "平衡时间因素影响"
        }
    
    def _default_correction(self, bias: CognitiveBias) -> Dict[str, Any]:
        """默认纠正策略"""
        return {
            "effectiveness": 0.6,
            "action": "应用通用纠正策略",
            "improvement": "提高认知质量"
        }
    
    def _evaluate_performance(self, state: CognitiveState):
        """评估性能"""
        try:
            # 计算综合性能得分
            performance_score = (
                state.confidence_level * 0.2 +
                state.accuracy_rate * 0.25 +
                (1 - state.error_rate) * 0.2 +
                state.learning_rate * 0.15 +
                state.creativity_index * 0.1 +
                state.reasoning_quality * 0.1
            )
            
            # 记录性能历史
            self.performance_history["overall"].append(performance_score)
            
            # 更新评估指标
            self.evaluation_metrics["current_performance"] = performance_score
            self.evaluation_metrics["trend"] = self._calculate_trend("overall")
            self.evaluation_metrics["stability"] = self._calculate_stability("overall")
            
        except Exception as e:
            logger.error(f"性能评估失败: {e}")
    
    def _calculate_trend(self, metric: str) -> float:
        """计算趋势"""
        try:
            if len(self.performance_history[metric]) < 10:
                return 0.0
            
            recent_values = list(self.performance_history[metric])[-10:]
            
            # 简单线性回归
            x = np.arange(len(recent_values))
            y = np.array(recent_values)
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            return 0.0
        
        except Exception:
            return 0.0
    
    def _calculate_stability(self, metric: str) -> float:
        """计算稳定性"""
        try:
            if len(self.performance_history[metric]) < 5:
                return 1.0
            
            recent_values = list(self.performance_history[metric])[-10:]
            variance = np.var(recent_values)
            stability = 1.0 / (1.0 + variance)
            return stability
        
        except Exception:
            return 1.0
    
    def _update_capability_models(self, state: CognitiveState):
        """更新能力模型"""
        domains = [
            "reasoning", "memory", "attention", "creativity",
            "learning", "problem_solving", "decision_making"
        ]
        
        for domain in domains:
            if domain not in self.capability_models:
                self.capability_models[domain] = CapabilityModel(
                    domain=domain,
                    current_level=0.5,
                    potential_level=1.0,
                    learning_velocity=0.0,
                    last_updated=time.time(),
                    confidence_interval=(0.3, 0.7),
                    dependencies=[]
                )
            
            # 更新能力水平
            model = self.capability_models[domain]
            domain_score = self._get_domain_score(domain, state)
            
            # 应用学习算法
            learning_rate = state.learning_rate * self.config["learning"]["adaptation_rate"]
            model.current_level = model.current_level + learning_rate * (domain_score - model.current_level)
            
            # 更新学习速度
            if model.last_updated > 0:
                time_delta = time.time() - model.last_updated
                if time_delta > 0:
                    velocity = (domain_score - model.current_level) / time_delta
                    model.learning_velocity = 0.9 * model.learning_velocity + 0.1 * velocity
            
            model.last_updated = time.time()
            
            # 更新置信区间
            self._update_confidence_interval(model)
    
    def _get_domain_score(self, domain: str, state: CognitiveState) -> float:
        """获取领域得分"""
        domain_mappings = {
            "reasoning": state.reasoning_quality,
            "memory": state.memory_utilization,
            "attention": state.attention_focus,
            "creativity": state.creativity_index,
            "learning": state.learning_rate,
            "problem_solving": (state.accuracy_rate + state.creativity_index) / 2,
            "decision_making": state.confidence_level
        }
        
        return domain_mappings.get(domain, 0.5)
    
    def _update_confidence_interval(self, model: CapabilityModel):
        """更新置信区间"""
        # 基于历史数据计算置信区间
        if len(self.learning_curves[model.domain]) > 5:
            recent_values = self.learning_curves[model.domain][-10:]
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values)
            
            lower_bound = max(0, mean_val - 1.96 * std_val)
            upper_bound = min(1, mean_val + 1.96 * std_val)
            
            model.confidence_interval = (lower_bound, upper_bound)
    
    def perform_self_reflection(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行自我反思"""
        try:
            reflection_result = self.reflection_engine.reflect(
                self.cognitive_states,
                self.detected_biases,
                self.learning_events,
                self.capability_models,
                context
            )
            
            # 记录反思事件
            self.learning_events.append(LearningEvent(
                timestamp=time.time(),
                event_type="self_reflection",
                context=reflection_result,
                outcome=reflection_result.get("conclusion", "completed"),
                learning_gain=reflection_result.get("insight_value", 0.0),
                adaptation_applied=reflection_result.get("adaptation_applied", False)
            ))
            
            # 应用适应性改进
            if reflection_result.get("adaptation_applied"):
                self._apply_adaptations(reflection_result.get("adaptations", []))
            
            logger.info("自我反思完成")
            return reflection_result
        
        except Exception as e:
            logger.error(f"自我反思失败: {e}")
            return {"error": str(e)}
    
    def _apply_adaptations(self, adaptations: List[Dict[str, Any]]):
        """应用适应性改进"""
        for adaptation in adaptations:
            try:
                # 应用配置调整
                if "config_adjustment" in adaptation:
                    self._apply_config_adjustment(adaptation["config_adjustment"])
                
                # 应用参数优化
                if "parameter_optimization" in adaptation:
                    self._apply_parameter_optimization(adaptation["parameter_optimization"])
                
                # 应用策略调整
                if "strategy_adjustment" in adaptation:
                    self._apply_strategy_adjustment(adaptation["strategy_adjustment"])
                
                # 记录适应历史
                self.adaptation_history.append({
                    "timestamp": time.time(),
                    "adaptation": adaptation,
                    "status": "applied"
                })
            
            except Exception as e:
                logger.error(f"应用适应性改进失败: {e}")
    
    def _apply_config_adjustment(self, adjustment: Dict[str, Any]):
        """应用配置调整"""
        config_path = adjustment.get("path")
        new_value = adjustment.get("value")
        
        if config_path and new_value:
            # 递归设置配置
            keys = config_path.split(".")
            config = self.config
            
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            config[keys[-1]] = new_value
            logger.info(f"配置调整已应用: {config_path} = {new_value}")
    
    def _apply_parameter_optimization(self, optimization: Dict[str, Any]):
        """应用参数优化"""
        parameter = optimization.get("parameter")
        new_value = optimization.get("new_value")
        
        if parameter and new_value:
            setattr(self, parameter, new_value)
            logger.info(f"参数优化已应用: {parameter} = {new_value}")
    
    def _apply_strategy_adjustment(self, adjustment: Dict[str, Any]):
        """应用策略调整"""
        strategy = adjustment.get("strategy")
        parameters = adjustment.get("parameters", {})
        
        if strategy:
            # 这里可以实现具体的策略调整逻辑
            logger.info(f"策略调整已应用: {strategy}")
    
    def assess_cognitive_capabilities(self) -> Dict[str, Any]:
        """评估认知能力"""
        try:
            assessment_result = {
                "timestamp": time.time(),
                "overall_score": 0.0,
                "domain_scores": {},
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "improvement_potential": 0.0
            }
            
            # 评估各个维度
            total_score = 0
            domain_count = 0
            
            for domain, model in self.capability_models.items():
                score = model.current_level
                assessment_result["domain_scores"][domain] = {
                    "current": score,
                    "potential": model.potential_level,
                    "confidence_interval": model.confidence_interval,
                    "learning_velocity": model.learning_velocity
                }
                
                total_score += score
                domain_count += 1
            
            # 计算总体得分
            if domain_count > 0:
                assessment_result["overall_score"] = total_score / domain_count
            
            # 识别优势和劣势
            for domain, scores in assessment_result["domain_scores"].items():
                if scores["current"] > 0.8:
                    assessment_result["strengths"].append(domain)
                elif scores["current"] < 0.5:
                    assessment_result["weaknesses"].append(domain)
            
            # 生成改进建议
            assessment_result["recommendations"] = self._generate_improvement_recommendations()
            
            # 计算改进潜力
            assessment_result["improvement_potential"] = self._calculate_improvement_potential()
            
            # 记录评估历史
            self.assessment_history.append(assessment_result)
            
            logger.info(f"认知能力评估完成，总体得分: {assessment_result['overall_score']:.3f}")
            return assessment_result
        
        except Exception as e:
            logger.error(f"认知能力评估失败: {e}")
            return {"error": str(e)}
    
    def _generate_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """生成改进建议"""
        recommendations = []
        
        # 基于检测到的偏差生成建议
        recent_biases = [b for b in self.detected_biases if time.time() - b.detected_at < 3600]
        
        bias_recommendations = {
            "confirmation_bias": "多角度思考，主动寻求不同观点",
            "anchoring_bias": "重新评估初始假设，考虑多种参考框架",
            "availability_bias": "系统性地收集和分析所有相关信息",
            "overconfidence_bias": "校准自信度，寻求外部反馈",
            "recency_bias": "平衡考虑历史数据和当前信息"
        }
        
        bias_counts = defaultdict(int)
        for bias in recent_biases:
            bias_counts[bias.bias_type] += 1
        
        for bias_type, count in bias_counts.items():
            if count > 2:  # 如果某种偏差频繁出现
                recommendation = {
                    "type": "bias_correction",
                    "bias_type": bias_type,
                    "priority": "high" if count > 5 else "medium",
                    "action": bias_recommendations.get(bias_type, "提高认知质量"),
                    "frequency": count
                }
                recommendations.append(recommendation)
        
        # 基于性能趋势生成建议
        if self.evaluation_metrics.get("trend", 0) < -0.01:
            recommendations.append({
                "type": "performance_improvement",
                "priority": "high",
                "action": "分析性能下降原因，调整学习策略",
                "trend": self.evaluation_metrics["trend"]
            })
        
        # 基于能力模型生成建议
        for domain, model in self.capability_models.items():
            if model.current_level < 0.5 and model.learning_velocity < 0.01:
                recommendations.append({
                    "type": "skill_development",
                    "domain": domain,
                    "priority": "medium",
                    "action": f"加强{domain}领域的学习和练习",
                    "current_level": model.current_level
                })
        
        return recommendations
    
    def _calculate_improvement_potential(self) -> float:
        """计算改进潜力"""
        if not self.capability_models:
            return 0.0
        
        potentials = []
        for model in self.capability_models.values():
            potential = model.potential_level - model.current_level
            potentials.append(max(0, potential))
        
        return np.mean(potentials) if potentials else 0.0
    
    def generate_improvement_suggestions(self, focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """生成自我改进建议"""
        try:
            suggestions = {
                "timestamp": time.time(),
                "focus_areas": focus_areas or [],
                "short_term": [],
                "medium_term": [],
                "long_term": [],
                "priority_actions": []
            }
            
            # 分析当前状态
            current_analysis = self._analyze_current_state_for_improvement()
            
            # 生成短期建议（1-7天）
            suggestions["short_term"] = self._generate_short_term_suggestions(current_analysis)
            
            # 生成中期建议（1-4周）
            suggestions["medium_term"] = self._generate_medium_term_suggestions(current_analysis)
            
            # 生成长期建议（1-6个月）
            suggestions["long_term"] = self._generate_long_term_suggestions(current_analysis)
            
            # 确定优先级行动
            suggestions["priority_actions"] = self._determine_priority_actions(current_analysis)
            
            logger.info("自我改进建议生成完成")
            return suggestions
        
        except Exception as e:
            logger.error(f"生成改进建议失败: {e}")
            return {"error": str(e)}
    
    def _analyze_current_state_for_improvement(self) -> Dict[str, Any]:
        """分析当前状态以生成改进建议"""
        analysis = {
            "performance_trend": self.evaluation_metrics.get("trend", 0),
            "stability": self.evaluation_metrics.get("stability", 1.0),
            "bias_frequency": len([b for b in self.detected_biases if time.time() - b.detected_at < 3600]),
            "learning_velocity": np.mean([m.learning_velocity for m in self.capability_models.values()]) if self.capability_models else 0,
            "low_performing_domains": [],
            "high_potential_domains": []
        }
        
        # 识别低性能领域
        for domain, model in self.capability_models.items():
            if model.current_level < 0.5:
                analysis["low_performing_domains"].append({
                    "domain": domain,
                    "level": model.current_level,
                    "potential": model.potential_level
                })
            elif model.current_level < model.potential_level * 0.8:
                analysis["high_potential_domains"].append({
                    "domain": domain,
                    "level": model.current_level,
                    "potential": model.potential_level
                })
        
        return analysis
    
    def _generate_short_term_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成短期建议"""
        suggestions = []
        
        # 基于性能趋势
        if analysis["performance_trend"] < -0.01:
            suggestions.append({
                "category": "performance",
                "action": "立即分析性能下降原因",
                "description": "检查最近的错误模式和学习效果",
                "timeline": "1-2天",
                "expected_impact": "中等"
            })
        
        # 基于偏差频率
        if analysis["bias_frequency"] > 5:
            suggestions.append({
                "category": "bias_correction",
                "action": "实施偏差纠正训练",
                "description": "针对频繁出现的认知偏差进行专项训练",
                "timeline": "3-5天",
                "expected_impact": "高"
            })
        
        # 基于低性能领域
        for domain_info in analysis["low_performing_domains"][:2]:  # 最多2个
            suggestions.append({
                "category": "skill_building",
                "action": f"专注提升{domain_info['domain']}能力",
                "description": f"针对{domain_info['domain']}领域进行强化训练",
                "timeline": "1周",
                "expected_impact": "高"
            })
        
        return suggestions
    
    def _generate_medium_term_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成中期建议"""
        suggestions = []
        
        # 基于学习速度
        if analysis["learning_velocity"] < 0.01:
            suggestions.append({
                "category": "learning_optimization",
                "action": "优化学习策略",
                "description": "调整学习参数和方法，提高学习效率",
                "timeline": "2-3周",
                "expected_impact": "高"
            })
        
        # 基于稳定性
        if analysis["stability"] < 0.7:
            suggestions.append({
                "category": "stability_improvement",
                "action": "提高认知稳定性",
                "description": "通过冥想、练习等方式提高认知稳定性",
                "timeline": "3-4周",
                "expected_impact": "中等"
            })
        
        # 基于高潜力领域
        for domain_info in analysis["high_potential_domains"][:1]:  # 最多1个
            suggestions.append({
                "category": "potential_development",
                "action": f"开发{domain_info['domain']}潜力",
                "description": f"通过系统性训练开发{domain_info['domain']}领域的高潜力",
                "timeline": "4周",
                "expected_impact": "高"
            })
        
        return suggestions
    
    def _generate_long_term_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成长期建议"""
        suggestions = []
        
        # 整体能力提升
        suggestions.append({
            "category": "comprehensive_development",
            "action": "制定长期能力发展计划",
            "description": "建立系统性的长期学习和改进计划",
            "timeline": "3-6个月",
            "expected_impact": "很高"
        })
        
        # 认知架构优化
        suggestions.append({
            "category": "architecture_optimization",
            "action": "优化认知架构",
            "description": "重新设计和优化认知处理流程",
            "timeline": "6个月",
            "expected_impact": "很高"
        })
        
        # 创新和突破
        suggestions.append({
            "category": "innovation",
            "action": "探索认知创新",
            "description": "探索新的认知方法和处理模式",
            "timeline": "6个月",
            "expected_impact": "很高"
        })
        
        return suggestions
    
    def _determine_priority_actions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """确定优先级行动"""
        priority_actions = []
        
        # 基于紧急性和重要性矩阵
        current_time = time.time()
        
        # 高紧急性行动
        if analysis["performance_trend"] < -0.05:
            priority_actions.append({
                "action": "紧急性能修复",
                "priority": "critical",
                "urgency": "high",
                "impact": "high",
                "deadline": current_time + 86400,  # 24小时
                "description": "立即修复严重的性能下降问题"
            })
        
        # 中等优先级行动
        if analysis["bias_frequency"] > 3:
            priority_actions.append({
                "action": "认知偏差纠正",
                "priority": "high",
                "urgency": "medium",
                "impact": "high",
                "deadline": current_time + 604800,  # 1周
                "description": "系统性纠正频繁出现的认知偏差"
            })
        
        # 长期优先级行动
        if analysis["low_performing_domains"]:
            domain = analysis["low_performing_domains"][0]
            priority_actions.append({
                "action": f"提升{domain['domain']}能力",
                "priority": "medium",
                "urgency": "low",
                "impact": "medium",
                "deadline": current_time + 2592000,  # 1个月
                "description": f"系统性提升{domain['domain']}领域的认知能力"
            })
        
        return priority_actions
    
    def generate_cognitive_report(self, time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """生成认知状态报告"""
        try:
            if time_range is None:
                # 默认最近24小时
                end_time = time.time()
                start_time = end_time - 86400
            else:
                start_time, end_time = time_range
            
            # 筛选时间范围内的数据
            relevant_states = [
                state for state in self.cognitive_states
                if start_time <= state.timestamp <= end_time
            ]
            
            relevant_events = [
                event for event in self.learning_events
                if start_time <= event.timestamp <= end_time
            ]
            
            relevant_biases = [
                bias for bias in self.detected_biases
                if start_time <= bias.detected_at <= end_time
            ]
            
            # 生成报告
            report = {
                "report_id": f"cognitive_report_{int(time.time())}",
                "generated_at": time.time(),
                "time_range": {
                    "start": start_time,
                    "end": end_time,
                    "duration_hours": (end_time - start_time) / 3600
                },
                "summary": {
                    "total_states": len(relevant_states),
                    "total_events": len(relevant_events),
                    "total_biases": len(relevant_biases),
                    "monitoring_uptime": len(relevant_states) / ((end_time - start_time) / self.monitor_interval) if (end_time - start_time) > 0 else 0
                },
                "performance_analysis": self._analyze_performance(relevant_states),
                "bias_analysis": self._analyze_biases(relevant_biases),
                "learning_analysis": self._analyze_learning(relevant_events),
                "capability_assessment": self._assess_capabilities(),
                "recommendations": self.generate_improvement_suggestions(),
                "trends": self._analyze_trends(relevant_states),
                "insights": self._generate_insights(relevant_states, relevant_events, relevant_biases)
            }
            
            logger.info(f"认知状态报告生成完成，时间范围: {start_time:.0f} - {end_time:.0f}")
            return report
        
        except Exception as e:
            logger.error(f"生成认知状态报告失败: {e}")
            return {"error": str(e)}
    
    def _analyze_performance(self, states: List[CognitiveState]) -> Dict[str, Any]:
        """分析性能"""
        if not states:
            return {"error": "没有足够的状态数据"}
        
        # 计算统计指标
        metrics = {}
        for field in ["confidence_level", "accuracy_rate", "response_time", "error_rate", "learning_rate"]:
            values = [getattr(state, field) for state in states]
            metrics[field] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "trend": self._calculate_trend_from_values(values)
            }
        
        # 性能等级评估
        overall_performance = np.mean([metrics[field]["mean"] for field in ["confidence_level", "accuracy_rate", "learning_rate"]])
        performance_grade = self._get_performance_grade(overall_performance)
        
        return {
            "metrics": metrics,
            "overall_performance": overall_performance,
            "performance_grade": performance_grade,
            "stability": 1.0 - np.mean([metrics[field]["std"] for field in metrics])
        }
    
    def _calculate_trend_from_values(self, values: List[float]) -> float:
        """从值列表计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _get_performance_grade(self, performance: float) -> str:
        """获取性能等级"""
        if performance >= 0.9:
            return "A+"
        elif performance >= 0.8:
            return "A"
        elif performance >= 0.7:
            return "B"
        elif performance >= 0.6:
            return "C"
        else:
            return "D"
    
    def _analyze_biases(self, biases: List[CognitiveBias]) -> Dict[str, Any]:
        """分析认知偏差"""
        if not biases:
            return {"message": "未检测到认知偏差"}
        
        # 偏差统计
        bias_counts = defaultdict(int)
        bias_severities = defaultdict(list)
        
        for bias in biases:
            bias_counts[bias.bias_type] += 1
            bias_severities[bias.bias_type].append(bias.severity)
        
        # 分析结果
        analysis = {
            "total_biases": len(biases),
            "bias_types": dict(bias_counts),
            "severity_distribution": {},
            "correction_effectiveness": {},
            "most_frequent_bias": max(bias_counts.items(), key=lambda x: x[1])[0] if bias_counts else None,
            "highest_severity_bias": None
        }
        
        # 严重程度分布
        for bias_type, severities in bias_severities.items():
            analysis["severity_distribution"][bias_type] = {
                "mean": np.mean(severities),
                "max": np.max(severities),
                "count": len(severities)
            }
        
        # 纠正效果
        corrected_biases = [b for b in biases if b.correction_applied]
        if corrected_biases:
            effectiveness_values = [b.correction_effectiveness for b in corrected_biases]
            analysis["correction_effectiveness"] = {
                "mean": np.mean(effectiveness_values),
                "success_rate": len([e for e in effectiveness_values if e > 0.7]) / len(effectiveness_values)
            }
        
        # 最高严重程度偏差
        if biases:
            highest_severity_bias = max(biases, key=lambda x: x.severity)
            analysis["highest_severity_bias"] = {
                "type": highest_severity_bias.bias_type,
                "severity": highest_severity_bias.severity,
                "detected_at": highest_severity_bias.detected_at
            }
        
        return analysis
    
    def _analyze_learning(self, events: List[LearningEvent]) -> Dict[str, Any]:
        """分析学习情况"""
        if not events:
            return {"message": "没有学习事件记录"}
        
        # 事件类型统计
        event_counts = defaultdict(int)
        learning_gains = []
        
        for event in events:
            event_counts[event.event_type] += 1
            if event.learning_gain > 0:
                learning_gains.append(event.learning_gain)
        
        # 分析结果
        analysis = {
            "total_events": len(events),
            "event_types": dict(event_counts),
            "total_learning_gain": sum(learning_gains),
            "average_learning_gain": np.mean(learning_gains) if learning_gains else 0,
            "adaptation_rate": len([e for e in events if e.adaptation_applied]) / len(events),
            "most_common_event": max(event_counts.items(), key=lambda x: x[1])[0] if event_counts else None
        }
        
        # 学习效率
        if events:
            time_span = max(event.timestamp for event in events) - min(event.timestamp for event in events)
            if time_span > 0:
                analysis["learning_efficiency"] = analysis["total_learning_gain"] / (time_span / 3600)  # 每小时学习收益
        
        return analysis
    
    def _assess_capabilities(self) -> Dict[str, Any]:
        """评估能力"""
        if not self.capability_models:
            return {"message": "没有能力模型数据"}
        
        assessment = {
            "domains": {},
            "overall_level": 0.0,
            "development_potential": 0.0,
            "strongest_domain": None,
            "weakest_domain": None
        }
        
        levels = []
        potentials = []
        
        for domain, model in self.capability_models.items():
            assessment["domains"][domain] = {
                "current_level": model.current_level,
                "potential_level": model.potential_level,
                "learning_velocity": model.learning_velocity,
                "confidence_interval": model.confidence_interval,
                "development_gap": model.potential_level - model.current_level
            }
            
            levels.append(model.current_level)
            potentials.append(model.potential_level)
        
        # 总体评估
        assessment["overall_level"] = np.mean(levels) if levels else 0
        assessment["development_potential"] = np.mean([p - l for p, l in zip(potentials, levels)]) if potentials and levels else 0
        
        # 最强和最弱领域
        if self.capability_models:
            strongest = max(self.capability_models.items(), key=lambda x: x[1].current_level)
            weakest = min(self.capability_models.items(), key=lambda x: x[1].current_level)
            
            assessment["strongest_domain"] = {
                "name": strongest[0],
                "level": strongest[1].current_level
            }
            assessment["weakest_domain"] = {
                "name": weakest[0],
                "level": weakest[1].current_level
            }
        
        return assessment
    
    def _analyze_trends(self, states: List[CognitiveState]) -> Dict[str, Any]:
        """分析趋势"""
        if len(states) < 2:
            return {"message": "状态数据不足，无法分析趋势"}
        
        trends = {}
        
        # 分析各指标趋势
        for field in ["confidence_level", "accuracy_rate", "learning_rate", "creativity_index"]:
            values = [getattr(state, field) for state in states]
            trends[field] = self._calculate_trend_from_values(values)
        
        # 总体趋势
        trend_values = list(trends.values())
        overall_trend = np.mean(trend_values) if trend_values else 0
        
        # 趋势分类
        trend_direction = "improving" if overall_trend > 0.01 else "declining" if overall_trend < -0.01 else "stable"
        
        return {
            "metric_trends": trends,
            "overall_trend": overall_trend,
            "trend_direction": trend_direction,
            "trend_strength": abs(overall_trend)
        }
    
    def _generate_insights(self, states: List[CognitiveState], events: List[LearningEvent], biases: List[CognitiveBias]) -> List[str]:
        """生成洞察"""
        insights = []
        
        # 基于性能洞察
        if states:
            recent_states = states[-10:] if len(states) >= 10 else states
            avg_confidence = np.mean([s.confidence_level for s in recent_states])
            avg_accuracy = np.mean([s.accuracy_rate for s in recent_states])
            
            if avg_confidence > 0.9 and avg_accuracy < 0.7:
                insights.append("存在过度自信偏差：信心水平过高但准确率不匹配")
            elif avg_confidence < 0.5 and avg_accuracy > 0.9:
                insights.append("存在信心不足：实际表现比自我认知更好")
        
        # 基于学习洞察
        if events:
            recent_events = [e for e in events if time.time() - e.timestamp < 3600]
            if len(recent_events) > 10:
                insights.append("学习活动频繁，可能存在过度学习或学习效率问题")
        
        # 基于偏差洞察
        bias_types = [b.bias_type for b in biases[-10:]] if biases else []
        if bias_types:
            bias_counts = defaultdict(int)
            for bias_type in bias_types:
                bias_counts[bias_type] += 1
            
            most_common = max(bias_counts.items(), key=lambda x: x[1])
            if most_common[1] > 3:
                insights.append(f"频繁出现{most_common[0]}，建议重点关注该偏差的纠正")
        
        # 基于趋势洞察
        if len(states) >= 5:
            recent_trend = self._analyze_trends(states[-5:])["trend_direction"]
            if recent_trend == "declining":
                insights.append("性能呈下降趋势，需要及时调整策略")
            elif recent_trend == "improving":
                insights.append("性能持续改善，当前策略有效")
        
        return insights
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        with self.lock:
            status = {
                "timestamp": time.time(),
                "monitoring_active": self.monitoring_active,
                "current_state": asdict(self.current_state) if self.current_state else None,
                "statistics": {
                    "total_states": len(self.cognitive_states),
                    "total_biases": len(self.detected_biases),
                    "total_learning_events": len(self.learning_events),
                    "capability_domains": len(self.capability_models)
                },
                "recent_performance": {
                    "trend": self.evaluation_metrics.get("trend", 0),
                    "stability": self.evaluation_metrics.get("stability", 1.0),
                    "current_score": self.evaluation_metrics.get("current_performance", 0)
                }
            }
            
            return status
    
    def export_data(self, filepath: str, format: str = "json"):
        """导出数据"""
        try:
            data = {
                "export_timestamp": time.time(),
                "cognitive_states": [asdict(state) for state in list(self.cognitive_states)],
                "capability_models": {domain: asdict(model) for domain, model in self.capability_models.items()},
                "detected_biases": [asdict(bias) for bias in self.detected_biases],
                "learning_events": [asdict(event) for event in list(self.learning_events)],
                "evaluation_metrics": self.evaluation_metrics,
                "assessment_history": [assessment if isinstance(assessment, dict) else asdict(assessment) for assessment in list(self.assessment_history)]
            }
            
            if format.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            logger.info(f"数据已导出到: {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"数据导出失败: {e}")
            return False
    
    def __del__(self):
        """析构函数"""
        self.stop_monitoring()


class BiasDetector:
    """认知偏差检测器"""
    
    def __init__(self):
        self.bias_patterns = {
            "confirmation_bias": self._detect_confirmation_bias,
            "anchoring_bias": self._detect_anchoring_bias,
            "availability_bias": self._detect_availability_bias,
            "overconfidence_bias": self._detect_overconfidence_bias,
            "recency_bias": self._detect_recency_bias
        }
    
    def detect_biases(self, current_state: CognitiveState, state_history: deque) -> List[CognitiveBias]:
        """检测认知偏差"""
        detected_biases = []
        
        for bias_type, detection_method in self.bias_patterns.items():
            try:
                bias = detection_method(current_state, state_history)
                if bias and bias.severity > 0.3:  # 阈值过滤
                    detected_biases.append(bias)
            except Exception as e:
                logger.error(f"检测{bias_type}时出错: {e}")
        
        return detected_biases
    
    def _detect_confirmation_bias(self, current_state: CognitiveState, history: deque) -> Optional[CognitiveBias]:
        """检测确认偏差"""
        if len(history) < 5:
            return None
        
        # 检查是否过度依赖初始信息
        recent_states = list(history)[-5:]
        confidence_variance = np.var([s.confidence_level for s in recent_states])
        
        if confidence_variance < 0.01:  # 置信度变化很小
            severity = min(1.0, (0.05 - confidence_variance) * 20)
            return CognitiveBias(
                bias_type="confirmation_bias",
                severity=severity,
                confidence=0.8,
                detected_at=time.time(),
                correction_applied=False,
                correction_effectiveness=0.0
            )
        
        return None
    
    def _detect_anchoring_bias(self, current_state: CognitiveState, history: deque) -> Optional[CognitiveBias]:
        """检测锚定偏差"""
        if len(history) < 10:
            return None
        
        # 检查是否过度依赖初始参考点
        first_state = list(history)[0]
        current_response_time = current_state.response_time
        first_response_time = first_state.response_time
        
        time_diff = abs(current_response_time - first_response_time)
        if time_diff < 0.1:  # 响应时间变化很小
            severity = min(1.0, (0.2 - time_diff) * 5)
            return CognitiveBias(
                bias_type="anchoring_bias",
                severity=severity,
                confidence=0.7,
                detected_at=time.time(),
                correction_applied=False,
                correction_effectiveness=0.0
            )
        
        return None
    
    def _detect_availability_bias(self, current_state: CognitiveState, history: deque) -> Optional[CognitiveBias]:
        """检测可得性偏差"""
        if len(history) < 5:
            return None
        
        # 检查是否过度依赖最近的信息
        recent_states = list(history)[-3:]
        recent_accuracy = np.mean([s.accuracy_rate for s in recent_states])
        
        if recent_accuracy > 0.9:  # 最近表现异常好
            severity = min(1.0, (recent_accuracy - 0.9) * 10)
            return CognitiveBias(
                bias_type="availability_bias",
                severity=severity,
                confidence=0.6,
                detected_at=time.time(),
                correction_applied=False,
                correction_effectiveness=0.0
            )
        
        return None
    
    def _detect_overconfidence_bias(self, current_state: CognitiveState, history: deque) -> Optional[CognitiveBias]:
        """检测过度自信偏差"""
        if len(history) < 5:
            return None
        
        # 检查信心水平与实际表现的差异
        recent_states = list(history)[-5:]
        avg_confidence = np.mean([s.confidence_level for s in recent_states])
        avg_accuracy = np.mean([s.accuracy_rate for s in recent_states])
        
        confidence_accuracy_gap = avg_confidence - avg_accuracy
        
        if confidence_accuracy_gap > 0.2:  # 信心超过准确率20%以上
            severity = min(1.0, confidence_accuracy_gap * 2)
            return CognitiveBias(
                bias_type="overconfidence_bias",
                severity=severity,
                confidence=0.9,
                detected_at=time.time(),
                correction_applied=False,
                correction_effectiveness=0.0
            )
        
        return None
    
    def _detect_recency_bias(self, current_state: CognitiveState, history: deque) -> Optional[CognitiveBias]:
        """检测近因偏差"""
        if len(history) < 10:
            return None
        
        # 检查是否过度重视最近的信息
        states = list(history)
        mid_point = len(states) // 2
        
        early_states = states[:mid_point]
        recent_states = states[mid_point:]
        
        early_performance = np.mean([s.accuracy_rate for s in early_states])
        recent_performance = np.mean([s.accuracy_rate for s in recent_states])
        
        performance_shift = recent_performance - early_performance
        
        if abs(performance_shift) > 0.2:  # 性能变化超过20%
            severity = min(1.0, abs(performance_shift) * 2)
            return CognitiveBias(
                bias_type="recency_bias",
                severity=severity,
                confidence=0.7,
                detected_at=time.time(),
                correction_applied=False,
                correction_effectiveness=0.0
            )
        
        return None


class ReflectionEngine:
    """自我反思引擎"""
    
    def reflect(self, states: deque, biases: List[CognitiveBias], events: deque, 
                capabilities: Dict[str, CapabilityModel], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行自我反思"""
        reflection = {
            "timestamp": time.time(),
            "context": context or {},
            "analysis": {},
            "insights": [],
            "conclusion": "",
            "adaptations": [],
            "adaptation_applied": False,
            "insight_value": 0.0
        }
        
        try:
            # 分析认知模式
            reflection["analysis"] = self._analyze_cognitive_patterns(states, biases, events, capabilities)
            
            # 生成洞察
            reflection["insights"] = self._generate_reflection_insights(reflection["analysis"])
            
            # 得出结论
            reflection["conclusion"] = self._form_conclusion(reflection["analysis"], reflection["insights"])
            
            # 生成适应建议
            reflection["adaptations"] = self._generate_adaptations(reflection["analysis"])
            
            # 计算洞察价值
            reflection["insight_value"] = self._calculate_insight_value(reflection["insights"])
            
            # 应用适应
            if reflection["adaptations"]:
                reflection["adaptation_applied"] = True
            
        except Exception as e:
            logger.error(f"自我反思失败: {e}")
            reflection["error"] = str(e)
        
        return reflection
    
    def _analyze_cognitive_patterns(self, states: deque, biases: List[CognitiveBias], 
                                   events: deque, capabilities: Dict[str, CapabilityModel]) -> Dict[str, Any]:
        """分析认知模式"""
        analysis = {
            "performance_patterns": {},
            "bias_patterns": {},
            "learning_patterns": {},
            "capability_patterns": {}
        }
        
        # 分析性能模式
        if states:
            recent_states = list(states)[-20:] if len(states) >= 20 else list(states)
            
            analysis["performance_patterns"] = {
                "consistency": 1.0 - np.std([s.confidence_level for s in recent_states]),
                "improvement_trend": self._calculate_improvement_trend(recent_states),
                "stability": 1.0 - np.std([s.accuracy_rate for s in recent_states]),
                "adaptability": np.std([s.learning_rate for s in recent_states])
            }
        
        # 分析偏差模式
        if biases:
            recent_biases = [b for b in biases if time.time() - b.detected_at < 3600]
            bias_counts = defaultdict(int)
            
            for bias in recent_biases:
                bias_counts[bias.bias_type] += 1
            
            analysis["bias_patterns"] = {
                "frequency": len(recent_biases),
                "most_common": max(bias_counts.items(), key=lambda x: x[1])[0] if bias_counts else None,
                "severity_trend": "increasing" if len(recent_biases) > 5 else "stable"
            }
        
        # 分析学习模式
        if events:
            recent_events = [e for e in events if time.time() - e.timestamp < 3600]
            event_types = defaultdict(int)
            
            for event in recent_events:
                event_types[event.event_type] += 1
            
            analysis["learning_patterns"] = {
                "activity_level": len(recent_events),
                "most_common_event": max(event_types.items(), key=lambda x: x[1])[0] if event_types else None,
                "adaptation_rate": len([e for e in recent_events if e.adaptation_applied]) / len(recent_events) if recent_events else 0
            }
        
        # 分析能力模式
        if capabilities:
            levels = [model.current_level for model in capabilities.values()]
            velocities = [model.learning_velocity for model in capabilities.values()]
            
            analysis["capability_patterns"] = {
                "average_level": np.mean(levels),
                "level_variance": np.var(levels),
                "learning_velocity": np.mean(velocities),
                "development_balance": 1.0 - np.var(levels)  # 越平衡方差越小
            }
        
        return analysis
    
    def _calculate_improvement_trend(self, states: List[CognitiveState]) -> float:
        """计算改进趋势"""
        if len(states) < 2:
            return 0.0
        
        # 使用准确率作为改进指标
        accuracies = [state.accuracy_rate for state in states]
        return self._calculate_trend_from_values(accuracies)
    
    def _calculate_trend_from_values(self, values: List[float]) -> float:
        """从值列表计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _generate_reflection_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """生成反思洞察"""
        insights = []
        
        # 基于性能模式洞察
        perf_patterns = analysis.get("performance_patterns", {})
        if perf_patterns.get("consistency", 0) > 0.9:
            insights.append("认知表现高度一致，显示良好的稳定性")
        elif perf_patterns.get("consistency", 0) < 0.5:
            insights.append("认知表现波动较大，需要提高稳定性")
        
        if perf_patterns.get("improvement_trend", 0) > 0.01:
            insights.append("呈现持续改进趋势，当前策略有效")
        elif perf_patterns.get("improvement_trend", 0) < -0.01:
            insights.append("性能呈下降趋势，需要重新评估方法")
        
        # 基于偏差模式洞察
        bias_patterns = analysis.get("bias_patterns", {})
        if bias_patterns.get("frequency", 0) > 10:
            insights.append("认知偏差频繁出现，需要系统性纠正")
        elif bias_patterns.get("frequency", 0) == 0:
            insights.append("未检测到明显认知偏差，认知质量良好")
        
        # 基于学习模式洞察
        learning_patterns = analysis.get("learning_patterns", {})
        if learning_patterns.get("activity_level", 0) > 20:
            insights.append("学习活动非常活跃，可能存在过度学习")
        elif learning_patterns.get("activity_level", 0) < 5:
            insights.append("学习活动较少，建议增加学习投入")
        
        # 基于能力模式洞察
        capability_patterns = analysis.get("capability_patterns", {})
        if capability_patterns.get("development_balance", 0) < 0.5:
            insights.append("能力发展不平衡，某些领域需要更多关注")
        elif capability_patterns.get("development_balance", 0) > 0.8:
            insights.append("能力发展相对均衡，各领域协调良好")
        
        return insights
    
    def _form_conclusion(self, analysis: Dict[str, Any], insights: List[str]) -> str:
        """形成结论"""
        # 基于分析结果形成总体结论
        perf_patterns = analysis.get("performance_patterns", {})
        bias_patterns = analysis.get("bias_patterns", {})
        learning_patterns = analysis.get("learning_patterns", {})
        capability_patterns = analysis.get("capability_patterns", {})
        
        # 评估整体状态
        positive_indicators = 0
        negative_indicators = 0
        
        if perf_patterns.get("improvement_trend", 0) > 0:
            positive_indicators += 1
        else:
            negative_indicators += 1
        
        if bias_patterns.get("frequency", 0) < 5:
            positive_indicators += 1
        else:
            negative_indicators += 1
        
        if learning_patterns.get("adaptation_rate", 0) > 0.5:
            positive_indicators += 1
        else:
            negative_indicators += 1
        
        # 形成结论
        if positive_indicators > negative_indicators:
            return "认知系统整体表现良好，保持当前策略并持续优化"
        elif negative_indicators > positive_indicators:
            return "认知系统需要改进，建议调整策略和方法"
        else:
            return "认知系统表现中等，有改进空间"
    
    def _generate_adaptations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成适应建议"""
        adaptations = []
        
        # 基于性能模式的适应
        perf_patterns = analysis.get("performance_patterns", {})
        if perf_patterns.get("consistency", 0) < 0.5:
            adaptations.append({
                "type": "stability_improvement",
                "description": "提高认知稳定性",
                "config_adjustment": {
                    "path": "monitoring.interval",
                    "value": 0.5
                }
            })
        
        if perf_patterns.get("improvement_trend", 0) < -0.01:
            adaptations.append({
                "type": "strategy_adjustment",
                "description": "调整学习策略",
                "strategy_adjustment": {
                    "strategy": "adaptive_learning",
                    "parameters": {"learning_rate": 0.15}
                }
            })
        
        # 基于偏差模式的适应
        bias_patterns = analysis.get("bias_patterns", {})
        if bias_patterns.get("frequency", 0) > 5:
            adaptations.append({
                "type": "bias_correction",
                "description": "加强偏差纠正机制",
                "config_adjustment": {
                    "path": "bias_detection.sensitivity",
                    "value": 0.9
                }
            })
        
        # 基于学习模式的适应
        learning_patterns = analysis.get("learning_patterns", {})
        if learning_patterns.get("adaptation_rate", 0) < 0.3:
            adaptations.append({
                "type": "adaptation_enhancement",
                "description": "提高适应能力",
                "parameter_optimization": {
                    "parameter": "adaptation_rate",
                    "new_value": 0.15
                }
            })
        
        return adaptations
    
    def _calculate_insight_value(self, insights: List[str]) -> float:
        """计算洞察价值"""
        if not insights:
            return 0.0
        
        # 基于洞察数量和多样性计算价值
        base_value = len(insights) * 0.1
        
        # 考虑洞察的具体性（这里简化为均匀分布）
        diversity_bonus = min(0.5, len(set(insights)) * 0.05)
        
        return min(1.0, base_value + diversity_bonus)


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建自我认知引擎
    engine = SelfCognitionEngine()
    
    # 启动监控
    engine.start_monitoring()
    
    try:
        # 运行一段时间收集数据
        print("开始自我认知引擎测试...")
        time.sleep(5)
        
        # 执行自我反思
        print("\n执行自我反思...")
        reflection_result = engine.perform_self_reflection({"context": "测试反思"})
        print(f"反思结果: {reflection_result['conclusion']}")
        
        # 评估认知能力
        print("\n评估认知能力...")
        assessment = engine.assess_cognitive_capabilities()
        print(f"总体得分: {assessment['overall_score']:.3f}")
        
        # 生成改进建议
        print("\n生成改进建议...")
        suggestions = engine.generate_improvement_suggestions()
        print(f"短期建议数量: {len(suggestions['short_term'])}")
        
        # 生成认知报告
        print("\n生成认知状态报告...")
        report = engine.generate_cognitive_report()
        print(f"报告ID: {report['report_id']}")
        
        # 获取当前状态
        print("\n获取当前状态...")
        status = engine.get_current_status()
        print(f"监控状态: {status['monitoring_active']}")
        print(f"总状态数: {status['statistics']['total_states']}")
        
        print("\n测试完成！")
        
    finally:
        # 停止监控
        engine.stop_monitoring()
        
        # 导出数据
        engine.export_data("cognitive_data_export.json")
        
        print("自我认知引擎测试结束")