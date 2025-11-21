"""
D2元认知处理器（元控制器）
实现高级元认知功能，包括监控、控制、分析、策略选择、知识管理、负荷监控、效果评估和系统优化
"""

import time
import threading
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
from datetime import datetime, timedelta


class CognitiveState(Enum):
    """认知状态枚举"""
    INITIALIZING = "初始化"
    ACTIVE = "活跃"
    ANALYZING = "分析中"
    MONITORING = "监控中"
    OPTIMIZING = "优化中"
    OVERLOADED = "过载"
    IDLE = "空闲"
    ERROR = "错误"


class StrategyType(Enum):
    """策略类型枚举"""
    ANALYTICAL = "分析型"
    INTUITIVE = "直觉型"
    SYSTEMATIC = "系统型"
    CREATIVE = "创造型"
    CRITICAL = "批判型"
    SYNTHETIC = "综合型"


class LoadLevel(Enum):
    """认知负荷等级"""
    LOW = "低"
    MODERATE = "中等"
    HIGH = "高"
    CRITICAL = "临界"


@dataclass
class CognitiveEvent:
    """认知事件"""
    timestamp: float
    event_type: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    outcome: str = "pending"


@dataclass
class CognitiveStrategy:
    """认知策略"""
    name: str
    strategy_type: StrategyType
    effectiveness_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    context_suitability: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaCognitionMetrics:
    """元认知指标"""
    processing_speed: float = 0.0
    accuracy_rate: float = 0.0
    strategy_effectiveness: float = 0.0
    cognitive_load: float = 0.0
    monitoring_accuracy: float = 0.0
    adaptation_speed: float = 0.0
    knowledge_retrieval_time: float = 0.0


class MetaCognitionKnowledgeBase:
    """元认知知识库"""
    
    def __init__(self):
        self.strategies: Dict[str, CognitiveStrategy] = {}
        self.patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_history: deque = deque(maxlen=1000)
        self.context_knowledge: Dict[str, Any] = {}
        self.learned_patterns: Dict[str, Any] = {}
        
    def add_strategy(self, strategy: CognitiveStrategy):
        """添加认知策略"""
        self.strategies[strategy.name] = strategy
        
    def update_strategy_performance(self, strategy_name: str, success: bool, 
                                  context: str, load_level: LoadLevel):
        """更新策略性能"""
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            strategy.usage_count += 1
            
            # 更新成功率
            total_successes = strategy.success_rate * (strategy.usage_count - 1)
            if success:
                total_successes += 1
            strategy.success_rate = total_successes / strategy.usage_count
            
            # 更新上下文适用性
            context_key = f"{context}_{load_level.value}"
            if context_key not in strategy.context_suitability:
                strategy.context_suitability[context_key] = 0.5
            
            # 贝叶斯更新
            alpha = 0.1  # 学习率
            strategy.context_suitability[context_key] += alpha * (1.0 - strategy.context_suitability[context_key]) if success else -alpha * strategy.context_suitability[context_key]
            
    def get_best_strategy(self, context: str, load_level: LoadLevel) -> Optional[CognitiveStrategy]:
        """获取最佳策略"""
        context_key = f"{context}_{load_level.value}"
        best_strategy = None
        best_score = 0.0
        
        for strategy in self.strategies.values():
            if context_key in strategy.context_suitability:
                score = strategy.success_rate * strategy.context_suitability[context_key]
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
                    
        return best_strategy if best_score > 0.3 else None


class CognitiveLoadMonitor:
    """认知负荷监控器"""
    
    def __init__(self, max_capacity: float = 100.0):
        self.max_capacity = max_capacity
        self.current_load: float = 0.0
        self.load_history: deque = deque(maxlen=100)
        self.load_factors: Dict[str, float] = {
            'processing_complexity': 0.0,
            'memory_usage': 0.0,
            'attention_demand': 0.0,
            'decision_difficulty': 0.0
        }
        
    def update_load(self, processing_complexity: float = 0.0, 
                   memory_usage: float = 0.0, 
                   attention_demand: float = 0.0,
                   decision_difficulty: float = 0.0):
        """更新认知负荷"""
        self.load_factors['processing_complexity'] = processing_complexity
        self.load_factors['memory_usage'] = memory_usage
        self.load_factors['attention_demand'] = attention_demand
        self.load_factors['decision_difficulty'] = decision_difficulty
        
        # 计算综合负荷
        weights = {'processing_complexity': 0.3, 'memory_usage': 0.25, 
                  'attention_demand': 0.25, 'decision_difficulty': 0.2}
        
        self.current_load = sum(self.load_factors[key] * weights[key] 
                              for key in weights.keys())
        self.load_history.append(self.current_load)
        
    def get_load_level(self) -> LoadLevel:
        """获取负荷等级"""
        load_percentage = (self.current_load / self.max_capacity) * 100
        
        if load_percentage < 30:
            return LoadLevel.LOW
        elif load_percentage < 60:
            return LoadLevel.MODERATE
        elif load_percentage < 85:
            return LoadLevel.HIGH
        else:
            return LoadLevel.CRITICAL
            
    def predict_future_load(self, steps: int = 5) -> List[float]:
        """预测未来负荷"""
        if len(self.load_history) < 2:
            return [self.current_load] * steps
            
        # 简单的线性趋势预测
        recent_loads = list(self.load_history)[-10:]
        if len(recent_loads) < 2:
            return [self.current_load] * steps
            
        # 计算趋势
        x = np.arange(len(recent_loads))
        y = np.array(recent_loads)
        
        # 线性回归
        coeffs = np.polyfit(x, y, 1)
        
        # 预测
        future_x = np.arange(len(recent_loads), len(recent_loads) + steps)
        future_loads = np.polyval(coeffs, future_x)
        
        return future_loads.tolist()


class MetaCognitionProcessor:
    """D2元认知处理器（元控制器）"""
    
    def __init__(self, max_concurrent_processes: int = 10):
        self.state = CognitiveState.INITIALIZING
        self.max_concurrent_processes = max_concurrent_processes
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        self.event_queue: deque = deque(maxlen=1000)
        self.monitoring_active = False
        self.optimization_active = False
        
        # 核心组件
        self.knowledge_base = MetaCognitionKnowledgeBase()
        self.load_monitor = CognitiveLoadMonitor()
        
        # 指标和统计
        self.metrics = MetaCognitionMetrics()
        self.performance_history: deque = deque(maxlen=500)
        self.adaptation_history: deque = deque(maxlen=200)
        
        # 线程控制
        self.monitoring_thread: Optional[threading.Thread] = None
        self.optimization_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # 初始化默认策略
        self._initialize_default_strategies()
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _initialize_default_strategies(self):
        """初始化默认认知策略"""
        default_strategies = [
            CognitiveStrategy("深度分析", StrategyType.ANALYTICAL),
            CognitiveStrategy("快速直觉", StrategyType.INTUITIVE),
            CognitiveStrategy("系统化处理", StrategyType.SYSTEMATIC),
            CognitiveStrategy("创新思维", StrategyType.CREATIVE),
            CognitiveStrategy("批判评估", StrategyType.CRITICAL),
            CognitiveStrategy("综合整合", StrategyType.SYNTHETIC)
        ]
        
        for strategy in default_strategies:
            self.knowledge_base.add_strategy(strategy)
            
    def start_monitoring(self):
        """启动元认知监控"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.state = CognitiveState.MONITORING
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("元认知监控已启动")
            
    def stop_monitoring(self):
        """停止元认知监控"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        self.logger.info("元认知监控已停止")
        
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._analyze_current_state()
                self._monitor_cognitive_processes()
                self._check_system_health()
                time.sleep(0.1)  # 监控间隔
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(1.0)
                
    def _analyze_current_state(self):
        """分析当前状态"""
        # 分析认知负荷
        load_level = self.load_monitor.get_load_level()
        
        # 分析处理效率
        if len(self.performance_history) > 0:
            recent_performance = list(self.performance_history)[-10:]
            avg_performance = np.mean([p.get('efficiency', 0.0) for p in recent_performance])
            
            # 根据负荷和性能调整状态
            if load_level == LoadLevel.CRITICAL:
                self.state = CognitiveState.OVERLOADED
            elif avg_performance > 0.8:
                self.state = CognitiveState.ACTIVE
            elif avg_performance > 0.5:
                self.state = CognitiveState.ANALYZING
            else:
                self.state = CognitiveState.IDLE
                
    def _monitor_cognitive_processes(self):
        """监控认知过程"""
        current_time = time.time()
        
        # 检查活跃进程
        with self.lock:
            completed_processes = []
            
            for process_id, process_info in self.active_processes.items():
                if current_time - process_info['start_time'] > process_info.get('timeout', 30.0):
                    # 进程超时
                    self._handle_process_timeout(process_id, process_info)
                    completed_processes.append(process_id)
                    
            # 清理完成的进程
            for process_id in completed_processes:
                del self.active_processes[process_id]
                
    def _check_system_health(self):
        """检查系统健康状态"""
        # 检查内存使用
        if len(self.event_queue) > self.event_queue.maxlen * 0.9:
            self.logger.warning("事件队列接近满载")
            
        # 检查负荷趋势
        if len(self.load_monitor.load_history) > 10:
            recent_loads = list(self.load_monitor.load_history)[-5:]
            if all(load > 80 for load in recent_loads):
                self.logger.warning("认知负荷持续过高")
                
    def _handle_process_timeout(self, process_id: str, process_info: Dict[str, Any]):
        """处理进程超时"""
        timeout_event = CognitiveEvent(
            timestamp=time.time(),
            event_type="process_timeout",
            description=f"进程 {process_id} 超时",
            data={"process_id": process_id, "timeout": process_info.get('timeout', 30.0)}
        )
        self.event_queue.append(timeout_event)
        
    def start_optimization(self):
        """启动系统优化"""
        if not self.optimization_active:
            self.optimization_active = True
            self.state = CognitiveState.OPTIMIZING
            self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
            self.optimization_thread.start()
            self.logger.info("系统优化已启动")
            
    def stop_optimization(self):
        """停止系统优化"""
        self.optimization_active = False
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=1.0)
        self.logger.info("系统优化已停止")
        
    def _optimization_loop(self):
        """优化循环"""
        while self.optimization_active:
            try:
                self._optimize_strategies()
                self._adjust_parameters()
                self._clean_knowledge_base()
                time.sleep(1.0)  # 优化间隔
            except Exception as e:
                self.logger.error(f"优化循环错误: {e}")
                time.sleep(2.0)
                
    def _optimize_strategies(self):
        """优化认知策略"""
        # 分析策略性能
        for strategy_name, strategy in self.knowledge_base.strategies.items():
            if strategy.usage_count > 5:  # 至少使用5次才优化
                # 计算综合性能分数
                performance_score = (
                    strategy.success_rate * 0.4 +
                    strategy.effectiveness_score * 0.3 +
                    np.mean(list(strategy.context_suitability.values())) * 0.3
                )
                
                # 记录优化历史
                self.adaptation_history.append({
                    'timestamp': time.time(),
                    'strategy': strategy_name,
                    'performance_score': performance_score,
                    'adaptation_type': 'strategy_optimization'
                })
                
    def _adjust_parameters(self):
        """调整系统参数"""
        # 根据当前负荷调整参数
        load_level = self.load_monitor.get_load_level()
        
        if load_level == LoadLevel.CRITICAL:
            # 降低并发度，增加处理间隔
            self.max_concurrent_processes = max(3, self.max_concurrent_processes // 2)
        elif load_level == LoadLevel.LOW:
            # 可以增加并发度
            self.max_concurrent_processes = min(20, self.max_concurrent_processes + 1)
            
    def _clean_knowledge_base(self):
        """清理知识库"""
        # 清理过期的性能历史
        cutoff_time = time.time() - 3600  # 1小时前
        self.performance_history = deque(
            [p for p in self.performance_history if p.get('timestamp', 0) > cutoff_time],
            maxlen=500
        )
        
    def process_cognitive_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理认知请求"""
        request_id = request.get('id', f"req_{time.time()}")
        context = request.get('context', 'general')
        complexity = request.get('complexity', 0.5)
        
        # 检查并发限制
        with self.lock:
            if len(self.active_processes) >= self.max_concurrent_processes:
                return {
                    'status': 'rejected',
                    'reason': '系统繁忙，请稍后重试',
                    'request_id': request_id
                }
                
        # 更新认知负荷
        self.load_monitor.update_load(
            processing_complexity=complexity,
            attention_demand=request.get('attention_demand', 0.5),
            decision_difficulty=request.get('decision_difficulty', 0.5)
        )
        
        # 选择最佳策略
        load_level = self.load_monitor.get_load_level()
        best_strategy = self.knowledge_base.get_best_strategy(context, load_level)
        
        if not best_strategy:
            # 使用默认策略
            best_strategy = self.knowledge_base.strategies.get("深度分析")
            
        # 创建进程记录
        process_info = {
            'request': request,
            'strategy': best_strategy,
            'start_time': time.time(),
            'context': context,
            'complexity': complexity
        }
        
        with self.lock:
            self.active_processes[request_id] = process_info
            
        # 记录事件
        event = CognitiveEvent(
            timestamp=time.time(),
            event_type="request_processing",
            description=f"开始处理请求 {request_id}",
            data={
                'request_id': request_id,
                'strategy': best_strategy.name,
                'context': context,
                'complexity': complexity
            }
        )
        self.event_queue.append(event)
        
        # 模拟处理过程
        return self._simulate_cognitive_processing(request_id, process_info)
        
    def _simulate_cognitive_processing(self, request_id: str, process_info: Dict[str, Any]) -> Dict[str, Any]:
        """模拟认知处理过程"""
        strategy = process_info['strategy']
        complexity = process_info['complexity']
        
        # 根据策略和复杂度计算处理时间
        base_time = 0.1
        strategy_multiplier = {
            StrategyType.ANALYTICAL: 1.5,
            StrategyType.INTUITIVE: 0.7,
            StrategyType.SYSTEMATIC: 1.2,
            StrategyType.CREATIVE: 1.8,
            StrategyType.CRITICAL: 1.4,
            StrategyType.SYNTHETIC: 1.6
        }
        
        processing_time = base_time * strategy_multiplier.get(strategy.strategy_type, 1.0) * (1 + complexity)
        
        # 模拟处理
        time.sleep(min(processing_time, 2.0))  # 最多等待2秒
        
        # 计算成功率和效果
        load_level = self.load_monitor.get_load_level()
        success_probability = strategy.success_rate
        
        # 负荷影响成功率
        if load_level == LoadLevel.HIGH:
            success_probability *= 0.8
        elif load_level == LoadLevel.CRITICAL:
            success_probability *= 0.6
            
        success = np.random.random() < success_probability
        
        # 更新策略性能
        self.knowledge_base.update_strategy_performance(
            strategy.name, success, process_info['context'], load_level
        )
        
        # 记录结果
        result = {
            'status': 'completed' if success else 'failed',
            'request_id': request_id,
            'strategy_used': strategy.name,
            'processing_time': processing_time,
            'success': success,
            'load_level': load_level.value,
            'result': {
                'output': f"使用{strategy.name}策略处理完成",
                'confidence': strategy.effectiveness_score,
                'processing_details': {
                    'strategy_type': strategy.strategy_type.value,
                    'context': process_info['context'],
                    'complexity': complexity
                }
            }
        }
        
        # 记录完成事件
        completion_event = CognitiveEvent(
            timestamp=time.time(),
            event_type="request_completed",
            description=f"请求 {request_id} 处理完成",
            data={
                'request_id': request_id,
                'success': success,
                'strategy': strategy.name,
                'processing_time': processing_time
            }
        )
        self.event_queue.append(completion_event)
        
        # 记录性能数据
        performance_data = {
            'timestamp': time.time(),
            'efficiency': 1.0 / processing_time if processing_time > 0 else 1.0,
            'success': success,
            'strategy': strategy.name,
            'load_level': load_level.value
        }
        self.performance_history.append(performance_data)
        
        # 清理活跃进程
        with self.lock:
            if request_id in self.active_processes:
                del self.active_processes[request_id]
                
        return result
        
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            return {
                'state': self.state.value,
                'active_processes': len(self.active_processes),
                'max_concurrent_processes': self.max_concurrent_processes,
                'current_load': self.load_monitor.current_load,
                'load_level': self.load_monitor.get_load_level().value,
                'event_queue_size': len(self.event_queue),
                'monitoring_active': self.monitoring_active,
                'optimization_active': self.optimization_active,
                'metrics': {
                    'processing_speed': self.metrics.processing_speed,
                    'accuracy_rate': self.metrics.accuracy_rate,
                    'strategy_effectiveness': self.metrics.strategy_effectiveness,
                    'cognitive_load': self.metrics.cognitive_load,
                    'monitoring_accuracy': self.metrics.monitoring_accuracy,
                    'adaptation_speed': self.metrics.adaptation_speed,
                    'knowledge_retrieval_time': self.metrics.knowledge_retrieval_time
                },
                'strategy_stats': {
                    name: {
                        'usage_count': strategy.usage_count,
                        'success_rate': strategy.success_rate,
                        'effectiveness_score': strategy.effectiveness_score,
                        'context_suitability': strategy.context_suitability
                    }
                    for name, strategy in self.knowledge_base.strategies.items()
                }
            }
            
    def get_recent_events(self, count: int = 50) -> List[Dict[str, Any]]:
        """获取最近事件"""
        recent_events = list(self.event_queue)[-count:]
        return [
            {
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'description': event.description,
                'data': event.data
            }
            for event in recent_events
        ]
        
    def get_performance_analytics(self) -> Dict[str, Any]:
        """获取性能分析"""
        if len(self.performance_history) == 0:
            return {'message': '暂无性能数据'}
            
        recent_performance = list(self.performance_history)[-50:]
        
        # 计算各种指标
        efficiencies = [p.get('efficiency', 0.0) for p in recent_performance]
        success_rate = np.mean([p.get('success', False) for p in recent_performance])
        
        # 策略效果分析
        strategy_performance = defaultdict(list)
        for p in recent_performance:
            strategy = p.get('strategy', 'unknown')
            efficiency = p.get('efficiency', 0.0)
            strategy_performance[strategy].append(efficiency)
            
        strategy_stats = {
            strategy: {
                'avg_efficiency': np.mean(efficiencies),
                'std_efficiency': np.std(efficiencies),
                'count': len(efficiencies)
            }
            for strategy, efficiencies in strategy_performance.items()
        }
        
        # 负荷分析
        load_history = list(self.load_monitor.load_history)
        load_trend = 'stable'
        if len(load_history) > 10:
            recent_loads = load_history[-10:]
            if recent_loads[-1] > recent_loads[0] * 1.2:
                load_trend = 'increasing'
            elif recent_loads[-1] < recent_loads[0] * 0.8:
                load_trend = 'decreasing'
                
        return {
            'overall_success_rate': success_rate,
            'avg_efficiency': np.mean(efficiencies),
            'efficiency_trend': np.polyfit(range(len(efficiencies)), efficiencies, 1)[0] if len(efficiencies) > 1 else 0,
            'strategy_performance': strategy_stats,
            'load_analysis': {
                'current_load': self.load_monitor.current_load,
                'load_trend': load_trend,
                'load_variance': np.var(load_history) if load_history else 0,
                'predicted_future_load': self.load_monitor.predict_future_load()
            },
            'adaptation_analytics': {
                'recent_adaptations': len(self.adaptation_history),
                'adaptation_rate': len(self.adaptation_history) / max(1, len(self.performance_history))
            }
        }
        
    def optimize_for_context(self, context: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """针对特定上下文优化系统"""
        optimization_start = time.time()
        
        # 分析需求
        complexity_requirement = requirements.get('complexity', 0.5)
        speed_requirement = requirements.get('speed', 0.5)
        accuracy_requirement = requirements.get('accuracy', 0.5)
        
        # 选择最适合的策略组合
        load_level = self.load_monitor.get_load_level()
        suitable_strategies = []
        
        for strategy in self.knowledge_base.strategies.values():
            context_key = f"{context}_{load_level.value}"
            if context_key in strategy.context_suitability:
                suitability = strategy.context_suitability[context_key]
                if suitability > 0.3:  # 最低适用性阈值
                    suitable_strategies.append((strategy, suitability))
                    
        # 按适用性排序
        suitable_strategies.sort(key=lambda x: x[1], reverse=True)
        
        # 调整系统参数
        if speed_requirement > 0.8:
            # 高速要求，降低复杂度阈值
            self.load_monitor.max_capacity *= 1.2
        elif accuracy_requirement > 0.8:
            # 高精度要求，增加处理时间
            self.load_monitor.max_capacity *= 0.8
            
        optimization_time = time.time() - optimization_start
        
        return {
            'optimization_status': 'completed',
            'optimization_time': optimization_time,
            'context': context,
            'recommended_strategies': [
                {
                    'name': strategy.name,
                    'strategy_type': strategy.strategy_type.value,
                    'suitability_score': suitability,
                    'success_rate': strategy.success_rate
                }
                for strategy, suitability in suitable_strategies[:3]  # 返回前3个
            ],
            'parameter_adjustments': {
                'max_capacity': self.load_monitor.max_capacity,
                'concurrent_processes': self.max_concurrent_processes
            },
            'expected_performance': {
                'estimated_success_rate': np.mean([s[0].success_rate for s in suitable_strategies[:3]]) if suitable_strategies else 0.5,
                'estimated_processing_time': 1.0 / max(0.1, speed_requirement),
                'load_prediction': self.load_monitor.predict_future_load(3)
            }
        }
        
    def shutdown(self):
        """关闭元认知处理器"""
        self.logger.info("正在关闭元认知处理器...")
        
        # 停止监控和优化
        self.stop_monitoring()
        self.stop_optimization()
        
        # 等待活跃进程完成
        with self.lock:
            if self.active_processes:
                self.logger.info(f"等待 {len(self.active_processes)} 个活跃进程完成...")
                timeout = 5.0
                start_time = time.time()
                
                while self.active_processes and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                    
                if self.active_processes:
                    self.logger.warning(f"强制终止 {len(self.active_processes)} 个未完成的进程")
                    
        self.state = CognitiveState.IDLE
        self.logger.info("元认知处理器已关闭")


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建元认知处理器
    processor = MetaCognitionProcessor(max_concurrent_processes=5)
    
    try:
        # 启动监控和优化
        processor.start_monitoring()
        processor.start_optimization()
        
        # 模拟一些认知请求
        test_requests = [
            {
                'id': 'req_1',
                'context': 'problem_solving',
                'complexity': 0.7,
                'attention_demand': 0.6,
                'decision_difficulty': 0.8
            },
            {
                'id': 'req_2', 
                'context': 'creative_thinking',
                'complexity': 0.9,
                'attention_demand': 0.9,
                'decision_difficulty': 0.7
            },
            {
                'id': 'req_3',
                'context': 'analysis',
                'complexity': 0.5,
                'attention_demand': 0.4,
                'decision_difficulty': 0.6
            }
        ]
        
        print("开始处理认知请求...")
        results = []
        
        for request in test_requests:
            result = processor.process_cognitive_request(request)
            results.append(result)
            print(f"请求 {request['id']}: {result['status']} - 使用策略: {result.get('strategy_used', 'N/A')}")
            
        # 等待处理完成
        time.sleep(1.0)
        
        # 获取系统状态
        status = processor.get_system_status()
        print(f"\n系统状态: {status['state']}")
        print(f"活跃进程: {status['active_processes']}")
        print(f"当前负荷: {status['current_load']:.2f} ({status['load_level']})")
        
        # 获取性能分析
        analytics = processor.get_performance_analytics()
        print(f"\n性能分析:")
        print(f"总体成功率: {analytics.get('overall_success_rate', 0):.2%}")
        print(f"平均效率: {analytics.get('avg_efficiency', 0):.2f}")
        
        # 获取最近事件
        recent_events = processor.get_recent_events(10)
        print(f"\n最近事件 ({len(recent_events)} 个):")
        for event in recent_events[-5:]:
            print(f"  {time.ctime(event['timestamp'])}: {event['description']}")
            
    finally:
        # 关闭处理器
        processor.shutdown()
        print("\n元认知处理器测试完成")