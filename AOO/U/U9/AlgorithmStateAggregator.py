"""
U9算法状态聚合器模块

该模块实现了算法状态聚合器类，用于管理和监控所有算法模块的状态、性能指标、
使用统计、效果评估、资源消耗等各个方面。

主要功能：
1. 所有算法模块状态收集
2. 算法性能指标聚合
3. 算法使用统计
4. 算法效果评估
5. 算法资源消耗监控
6. 算法健康度检查
7. 算法版本管理
8. 算法配置管理
9. 算法状态报告生成

作者：U9系统
版本：1.0.0
创建时间：2025-11-05
"""

import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import psutil
import statistics


class AlgorithmStatus(Enum):
    """算法状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class AlgorithmMetrics:
    """算法性能指标数据类"""
    algorithm_id: str
    algorithm_name: str
    version: str
    status: AlgorithmStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    execution_time: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    request_count: int = 0
    avg_response_time: float = 0.0
    throughput: float = 0.0
    last_execution_time: Optional[datetime] = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AlgorithmConfig:
    """算法配置数据类"""
    algorithm_id: str
    name: str
    version: str
    description: str
    parameters: Dict[str, Any]
    enabled: bool = True
    auto_scaling: bool = False
    max_instances: int = 1
    timeout: int = 300
    retry_count: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AlgorithmUsageStats:
    """算法使用统计数据类"""
    algorithm_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    peak_concurrent_users: int = 0
    daily_usage: Dict[str, int] = field(default_factory=dict)
    hourly_usage: Dict[str, int] = field(default_factory=dict)
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AlgorithmEffectiveness:
    """算法效果评估数据类"""
    algorithm_id: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_results: Dict[str, float] = field(default_factory=dict)
    last_evaluation: Optional[datetime] = None
    evaluation_count: int = 0


class AlgorithmStateAggregator:
    """算法状态聚合器
    
    该类负责收集、聚合、分析和报告所有算法模块的状态信息。
    提供全面的算法监控和管理功能。
    """
    
    def __init__(self, 
                 collection_interval: int = 60,
                 max_history_size: int = 1000,
                 enable_real_time_monitoring: bool = True):
        """初始化算法状态聚合器
        
        Args:
            collection_interval: 数据收集间隔（秒）
            max_history_size: 历史数据最大保存数量
            enable_real_time_monitoring: 是否启用实时监控
        """
        self.collection_interval = collection_interval
        self.max_history_size = max_history_size
        self.enable_real_time_monitoring = enable_real_time_monitoring
        
        # 数据存储
        self._algorithms: Dict[str, AlgorithmMetrics] = {}
        self._configs: Dict[str, AlgorithmConfig] = {}
        self._usage_stats: Dict[str, AlgorithmUsageStats] = {}
        self._effectiveness: Dict[str, AlgorithmEffectiveness] = {}
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 监控线程
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = False
        
        # 回调函数
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 启动监控
        if enable_real_time_monitoring:
            self.start_monitoring()
    
    def register_algorithm(self, 
                          algorithm_id: str,
                          algorithm_name: str,
                          version: str = "1.0.0",
                          config: Optional[AlgorithmConfig] = None) -> bool:
        """注册算法模块
        
        Args:
            algorithm_id: 算法唯一标识
            algorithm_name: 算法名称
            version: 算法版本
            config: 算法配置
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self._lock:
                if algorithm_id in self._algorithms:
                    self.logger.warning(f"算法 {algorithm_id} 已存在，正在更新")
                
                # 创建默认配置
                if config is None:
                    config = AlgorithmConfig(
                        algorithm_id=algorithm_id,
                        name=algorithm_name,
                        version=version,
                        description=f"算法 {algorithm_name} 的默认配置",
                        parameters={}
                    )
                
                # 创建算法指标
                metrics = AlgorithmMetrics(
                    algorithm_id=algorithm_id,
                    algorithm_name=algorithm_name,
                    version=version,
                    status=AlgorithmStatus.IDLE
                )
                
                # 创建使用统计
                usage_stats = AlgorithmUsageStats(algorithm_id=algorithm_id)
                
                # 创建效果评估
                effectiveness = AlgorithmEffectiveness(algorithm_id=algorithm_id)
                
                # 存储数据
                self._algorithms[algorithm_id] = metrics
                self._configs[algorithm_id] = config
                self._usage_stats[algorithm_id] = usage_stats
                self._effectiveness[algorithm_id] = effectiveness
                
                self.logger.info(f"算法 {algorithm_id} 注册成功")
                return True
                
        except Exception as e:
            self.logger.error(f"注册算法 {algorithm_id} 失败: {str(e)}")
            return False
    
    def update_algorithm_status(self, 
                               algorithm_id: str,
                               status: AlgorithmStatus,
                               custom_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """更新算法状态
        
        Args:
            algorithm_id: 算法ID
            status: 新状态
            custom_metrics: 自定义指标
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self._lock:
                if algorithm_id not in self._algorithms:
                    self.logger.error(f"算法 {algorithm_id} 不存在")
                    return False
                
                metrics = self._algorithms[algorithm_id]
                metrics.status = status
                metrics.timestamp = datetime.now()
                
                if custom_metrics:
                    metrics.custom_metrics.update(custom_metrics)
                
                # 检查健康状态
                self._check_algorithm_health(algorithm_id)
                
                # 添加到历史记录
                self._history[algorithm_id].append(metrics)
                
                # 触发回调
                self._trigger_callbacks('status_change', algorithm_id, status)
                
                return True
                
        except Exception as e:
            self.logger.error(f"更新算法 {algorithm_id} 状态失败: {str(e)}")
            return False
    
    def update_algorithm_metrics(self, 
                                algorithm_id: str,
                                cpu_usage: Optional[float] = None,
                                memory_usage: Optional[float] = None,
                                execution_time: Optional[float] = None,
                                success_rate: Optional[float] = None,
                                error_count: Optional[int] = None,
                                request_count: Optional[int] = None,
                                avg_response_time: Optional[float] = None,
                                throughput: Optional[float] = None,
                                custom_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """更新算法性能指标
        
        Args:
            algorithm_id: 算法ID
            cpu_usage: CPU使用率
            memory_usage: 内存使用率
            execution_time: 执行时间
            success_rate: 成功率
            error_count: 错误计数
            request_count: 请求计数
            avg_response_time: 平均响应时间
            throughput: 吞吐量
            custom_metrics: 自定义指标
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self._lock:
                if algorithm_id not in self._algorithms:
                    self.logger.error(f"算法 {algorithm_id} 不存在")
                    return False
                
                metrics = self._algorithms[algorithm_id]
                
                # 更新指标
                if cpu_usage is not None:
                    metrics.cpu_usage = cpu_usage
                if memory_usage is not None:
                    metrics.memory_usage = memory_usage
                if execution_time is not None:
                    metrics.execution_time = execution_time
                if success_rate is not None:
                    metrics.success_rate = success_rate
                if error_count is not None:
                    metrics.error_count = error_count
                if request_count is not None:
                    metrics.request_count = request_count
                if avg_response_time is not None:
                    metrics.avg_response_time = avg_response_time
                if throughput is not None:
                    metrics.throughput = throughput
                
                if custom_metrics:
                    metrics.custom_metrics.update(custom_metrics)
                
                metrics.last_execution_time = datetime.now()
                metrics.timestamp = datetime.now()
                
                # 检查健康状态
                self._check_algorithm_health(algorithm_id)
                
                # 添加到历史记录
                self._history[algorithm_id].append(metrics)
                
                # 触发回调
                self._trigger_callbacks('metrics_update', algorithm_id, metrics)
                
                return True
                
        except Exception as e:
            self.logger.error(f"更新算法 {algorithm_id} 指标失败: {str(e)}")
            return False
    
    def update_usage_statistics(self, 
                               algorithm_id: str,
                               execution_time: Optional[float] = None,
                               success: Optional[bool] = None,
                               user_id: Optional[str] = None) -> bool:
        """更新算法使用统计
        
        Args:
            algorithm_id: 算法ID
            execution_time: 执行时间
            success: 执行是否成功
            user_id: 用户ID
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self._lock:
                if algorithm_id not in self._usage_stats:
                    self.logger.error(f"算法 {algorithm_id} 不存在")
                    return False
                
                stats = self._usage_stats[algorithm_id]
                now = datetime.now()
                
                # 更新基本统计
                stats.total_requests += 1
                if execution_time:
                    stats.total_execution_time += execution_time
                    stats.avg_execution_time = stats.total_execution_time / stats.total_requests
                
                if success is not None:
                    if success:
                        stats.successful_requests += 1
                    else:
                        stats.failed_requests += 1
                
                # 更新日使用统计
                date_key = now.strftime('%Y-%m-%d')
                stats.daily_usage[date_key] = stats.daily_usage.get(date_key, 0) + 1
                
                # 更新小时使用统计
                hour_key = now.strftime('%Y-%m-%d %H:00')
                stats.hourly_usage[hour_key] = stats.hourly_usage.get(hour_key, 0) + 1
                
                stats.last_updated = now
                
                # 触发回调
                self._trigger_callbacks('usage_update', algorithm_id, stats)
                
                return True
                
        except Exception as e:
            self.logger.error(f"更新算法 {algorithm_id} 使用统计失败: {str(e)}")
            return False
    
    def update_effectiveness_evaluation(self, 
                                       algorithm_id: str,
                                       accuracy: Optional[float] = None,
                                       precision: Optional[float] = None,
                                       recall: Optional[float] = None,
                                       f1_score: Optional[float] = None,
                                       auc_score: Optional[float] = None,
                                       custom_metrics: Optional[Dict[str, float]] = None,
                                       benchmark_results: Optional[Dict[str, float]] = None) -> bool:
        """更新算法效果评估
        
        Args:
            algorithm_id: 算法ID
            accuracy: 准确率
            precision: 精确率
            recall: 召回率
            f1_score: F1分数
            auc_score: AUC分数
            custom_metrics: 自定义指标
            benchmark_results: 基准测试结果
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self._lock:
                if algorithm_id not in self._effectiveness:
                    self.logger.error(f"算法 {algorithm_id} 不存在")
                    return False
                
                eval_data = self._effectiveness[algorithm_id]
                now = datetime.now()
                
                # 更新评估指标
                if accuracy is not None:
                    eval_data.accuracy = accuracy
                if precision is not None:
                    eval_data.precision = precision
                if recall is not None:
                    eval_data.recall = recall
                if f1_score is not None:
                    eval_data.f1_score = f1_score
                if auc_score is not None:
                    eval_data.auc_score = auc_score
                
                if custom_metrics:
                    eval_data.custom_metrics.update(custom_metrics)
                
                if benchmark_results:
                    eval_data.benchmark_results.update(benchmark_results)
                
                eval_data.last_evaluation = now
                eval_data.evaluation_count += 1
                
                # 触发回调
                self._trigger_callbacks('effectiveness_update', algorithm_id, eval_data)
                
                return True
                
        except Exception as e:
            self.logger.error(f"更新算法 {algorithm_id} 效果评估失败: {str(e)}")
            return False
    
    def update_algorithm_config(self, 
                               algorithm_id: str,
                               parameters: Optional[Dict[str, Any]] = None,
                               enabled: Optional[bool] = None,
                               auto_scaling: Optional[bool] = None,
                               max_instances: Optional[int] = None,
                               timeout: Optional[int] = None,
                               retry_count: Optional[int] = None) -> bool:
        """更新算法配置
        
        Args:
            algorithm_id: 算法ID
            parameters: 算法参数
            enabled: 是否启用
            auto_scaling: 是否自动扩缩容
            max_instances: 最大实例数
            timeout: 超时时间
            retry_count: 重试次数
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self._lock:
                if algorithm_id not in self._configs:
                    self.logger.error(f"算法 {algorithm_id} 不存在")
                    return False
                
                config = self._configs[algorithm_id]
                
                # 更新配置
                if parameters:
                    config.parameters.update(parameters)
                if enabled is not None:
                    config.enabled = enabled
                if auto_scaling is not None:
                    config.auto_scaling = auto_scaling
                if max_instances is not None:
                    config.max_instances = max_instances
                if timeout is not None:
                    config.timeout = timeout
                if retry_count is not None:
                    config.retry_count = retry_count
                
                config.updated_at = datetime.now()
                
                # 触发回调
                self._trigger_callbacks('config_update', algorithm_id, config)
                
                return True
                
        except Exception as e:
            self.logger.error(f"更新算法 {algorithm_id} 配置失败: {str(e)}")
            return False
    
    def get_algorithm_status(self, algorithm_id: str) -> Optional[AlgorithmMetrics]:
        """获取算法状态
        
        Args:
            algorithm_id: 算法ID
            
        Returns:
            AlgorithmMetrics: 算法状态信息
        """
        with self._lock:
            return self._algorithms.get(algorithm_id)
    
    def get_all_algorithms_status(self) -> Dict[str, AlgorithmMetrics]:
        """获取所有算法状态
        
        Returns:
            Dict[str, AlgorithmMetrics]: 所有算法状态信息
        """
        with self._lock:
            return self._algorithms.copy()
    
    def get_algorithm_config(self, algorithm_id: str) -> Optional[AlgorithmConfig]:
        """获取算法配置
        
        Args:
            algorithm_id: 算法ID
            
        Returns:
            AlgorithmConfig: 算法配置信息
        """
        with self._lock:
            return self._configs.get(algorithm_id)
    
    def get_algorithm_usage_stats(self, algorithm_id: str) -> Optional[AlgorithmUsageStats]:
        """获取算法使用统计
        
        Args:
            algorithm_id: 算法ID
            
        Returns:
            AlgorithmUsageStats: 算法使用统计信息
        """
        with self._lock:
            return self._usage_stats.get(algorithm_id)
    
    def get_algorithm_effectiveness(self, algorithm_id: str) -> Optional[AlgorithmEffectiveness]:
        """获取算法效果评估
        
        Args:
            algorithm_id: 算法ID
            
        Returns:
            AlgorithmEffectiveness: 算法效果评估信息
        """
        with self._lock:
            return self._effectiveness.get(algorithm_id)
    
    def get_algorithm_history(self, 
                             algorithm_id: str, 
                             hours: int = 24) -> List[AlgorithmMetrics]:
        """获取算法历史数据
        
        Args:
            algorithm_id: 算法ID
            hours: 历史数据时间范围（小时）
            
        Returns:
            List[AlgorithmMetrics]: 历史数据列表
        """
        with self._lock:
            if algorithm_id not in self._history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            history = list(self._history[algorithm_id])
            
            return [metrics for metrics in history if metrics.timestamp >= cutoff_time]
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """获取聚合指标
        
        Returns:
            Dict[str, Any]: 聚合后的指标数据
        """
        with self._lock:
            if not self._algorithms:
                return {}
            
            # 计算总体指标
            total_algorithms = len(self._algorithms)
            running_algorithms = sum(1 for m in self._algorithms.values() 
                                   if m.status == AlgorithmStatus.RUNNING)
            error_algorithms = sum(1 for m in self._algorithms.values() 
                                 if m.status == AlgorithmStatus.ERROR)
            
            # 计算平均性能指标
            cpu_usage = statistics.mean([m.cpu_usage for m in self._algorithms.values()]) if self._algorithms else 0
            memory_usage = statistics.mean([m.memory_usage for m in self._algorithms.values()]) if self._algorithms else 0
            avg_success_rate = statistics.mean([m.success_rate for m in self._algorithms.values()]) if self._algorithms else 0
            
            # 计算总使用量
            total_requests = sum(s.total_requests for s in self._usage_stats.values())
            total_execution_time = sum(s.total_execution_time for s in self._usage_stats.values())
            
            # 健康状态统计
            health_stats = defaultdict(int)
            for m in self._algorithms.values():
                health_stats[m.health_status.value] += 1
            
            return {
                'timestamp': datetime.now(),
                'total_algorithms': total_algorithms,
                'running_algorithms': running_algorithms,
                'error_algorithms': error_algorithms,
                'avg_cpu_usage': round(cpu_usage, 2),
                'avg_memory_usage': round(memory_usage, 2),
                'avg_success_rate': round(avg_success_rate, 2),
                'total_requests': total_requests,
                'total_execution_time': round(total_execution_time, 2),
                'health_status_distribution': dict(health_stats),
                'algorithms': {aid: asdict(metrics) for aid, metrics in self._algorithms.items()}
            }
    
    def check_algorithm_health(self, algorithm_id: str) -> HealthStatus:
        """检查算法健康状态
        
        Args:
            algorithm_id: 算法ID
            
        Returns:
            HealthStatus: 健康状态
        """
        return self._check_algorithm_health(algorithm_id)
    
    def _check_algorithm_health(self, algorithm_id: str) -> HealthStatus:
        """内部健康状态检查方法"""
        if algorithm_id not in self._algorithms:
            return HealthStatus.UNKNOWN
        
        metrics = self._algorithms[algorithm_id]
        health_score = 100
        
        # CPU使用率检查
        if metrics.cpu_usage > 90:
            health_score -= 30
        elif metrics.cpu_usage > 70:
            health_score -= 15
        
        # 内存使用率检查
        if metrics.memory_usage > 90:
            health_score -= 30
        elif metrics.memory_usage > 70:
            health_score -= 15
        
        # 成功率检查
        if metrics.success_rate < 50:
            health_score -= 40
        elif metrics.success_rate < 80:
            health_score -= 20
        
        # 错误计数检查
        if metrics.error_count > 100:
            health_score -= 20
        elif metrics.error_count > 50:
            health_score -= 10
        
        # 响应时间检查
        if metrics.avg_response_time > 10:
            health_score -= 20
        elif metrics.avg_response_time > 5:
            health_score -= 10
        
        # 确定健康状态
        if health_score >= 80:
            health_status = HealthStatus.HEALTHY
        elif health_score >= 60:
            health_status = HealthStatus.WARNING
        else:
            health_status = HealthStatus.CRITICAL
        
        metrics.health_status = health_status
        return health_status
    
    def generate_status_report(self, 
                              algorithm_id: Optional[str] = None,
                              format_type: str = 'json') -> Union[str, Dict[str, Any]]:
        """生成算法状态报告
        
        Args:
            algorithm_id: 算法ID，None表示所有算法
            format_type: 报告格式 ('json', 'text')
            
        Returns:
            Union[str, Dict[str, Any]]: 状态报告
        """
        with self._lock:
            if algorithm_id:
                # 单个算法报告
                if algorithm_id not in self._algorithms:
                    return {"error": f"算法 {algorithm_id} 不存在"}
                
                report = {
                    'algorithm_id': algorithm_id,
                    'timestamp': datetime.now(),
                    'status': asdict(self._algorithms[algorithm_id]),
                    'config': asdict(self._configs[algorithm_id]) if algorithm_id in self._configs else None,
                    'usage_stats': asdict(self._usage_stats[algorithm_id]) if algorithm_id in self._usage_stats else None,
                    'effectiveness': asdict(self._effectiveness[algorithm_id]) if algorithm_id in self._effectiveness else None,
                    'health_status': self._check_algorithm_health(algorithm_id).value
                }
            else:
                # 所有算法报告
                report = self.get_aggregated_metrics()
                report['individual_reports'] = {}
                
                for aid in self._algorithms.keys():
                    report['individual_reports'][aid] = {
                        'status': asdict(self._algorithms[aid]),
                        'config': asdict(self._configs[aid]) if aid in self._configs else None,
                        'usage_stats': asdict(self._usage_stats[aid]) if aid in self._usage_stats else None,
                        'effectiveness': asdict(self._effectiveness[aid]) if aid in self._effectiveness else None,
                        'health_status': self._check_algorithm_health(aid).value
                    }
        
        if format_type.lower() == 'json':
            return json.dumps(report, indent=2, ensure_ascii=False, default=str)
        elif format_type.lower() == 'text':
            return self._format_text_report(report)
        else:
            raise ValueError(f"不支持的报告格式: {format_type}")
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """格式化文本报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("U9算法状态聚合器报告")
        lines.append("=" * 60)
        lines.append(f"生成时间: {report.get('timestamp', datetime.now())}")
        lines.append("")
        
        if 'total_algorithms' in report:
            # 总体报告
            lines.append("总体概况:")
            lines.append(f"  算法总数: {report['total_algorithms']}")
            lines.append(f"  运行中: {report['running_algorithms']}")
            lines.append(f"  错误状态: {report['error_algorithms']}")
            lines.append(f"  平均CPU使用率: {report['avg_cpu_usage']}%")
            lines.append(f"  平均内存使用率: {report['avg_memory_usage']}%")
            lines.append(f"  平均成功率: {report['avg_success_rate']}%")
            lines.append(f"  总请求数: {report['total_requests']}")
            lines.append("")
            
            lines.append("健康状态分布:")
            for status, count in report['health_status_distribution'].items():
                lines.append(f"  {status}: {count}")
            lines.append("")
            
            if 'individual_reports' in report:
                lines.append("算法详情:")
                for aid, individual_report in report['individual_reports'].items():
                    lines.append(f"\n  算法ID: {aid}")
                    status = individual_report['status']
                    lines.append(f"    名称: {status.get('algorithm_name', 'N/A')}")
                    lines.append(f"    状态: {status.get('status', 'N/A')}")
                    lines.append(f"    健康状态: {individual_report.get('health_status', 'N/A')}")
                    lines.append(f"    CPU使用率: {status.get('cpu_usage', 0):.1f}%")
                    lines.append(f"    内存使用率: {status.get('memory_usage', 0):.1f}%")
                    lines.append(f"    成功率: {status.get('success_rate', 0):.1f}%")
        else:
            # 单个算法报告
            lines.append(f"算法ID: {report['algorithm_id']}")
            if report['status']:
                status = report['status']
                lines.append(f"名称: {status.get('algorithm_name', 'N/A')}")
                lines.append(f"状态: {status.get('status', 'N/A')}")
                lines.append(f"版本: {status.get('version', 'N/A')}")
                lines.append(f"健康状态: {report.get('health_status', 'N/A')}")
                lines.append(f"CPU使用率: {status.get('cpu_usage', 0):.1f}%")
                lines.append(f"内存使用率: {status.get('memory_usage', 0):.1f}%")
                lines.append(f"执行时间: {status.get('execution_time', 0):.2f}秒")
                lines.append(f"成功率: {status.get('success_rate', 0):.1f}%")
                lines.append(f"错误计数: {status.get('error_count', 0)}")
                lines.append(f"请求计数: {status.get('request_count', 0)}")
                lines.append(f"平均响应时间: {status.get('avg_response_time', 0):.2f}秒")
                lines.append(f"吞吐量: {status.get('throughput', 0):.2f}")
            
            if report.get('usage_stats'):
                usage = report['usage_stats']
                lines.append(f"\n使用统计:")
                lines.append(f"  总请求数: {usage.get('total_requests', 0)}")
                lines.append(f"  成功请求数: {usage.get('successful_requests', 0)}")
                lines.append(f"  失败请求数: {usage.get('failed_requests', 0)}")
                lines.append(f"  总执行时间: {usage.get('total_execution_time', 0):.2f}秒")
                lines.append(f"  平均执行时间: {usage.get('avg_execution_time', 0):.2f}秒")
            
            if report.get('effectiveness'):
                eff = report['effectiveness']
                lines.append(f"\n效果评估:")
                lines.append(f"  准确率: {eff.get('accuracy', 0):.3f}")
                lines.append(f"  精确率: {eff.get('precision', 0):.3f}")
                lines.append(f"  召回率: {eff.get('recall', 0):.3f}")
                lines.append(f"  F1分数: {eff.get('f1_score', 0):.3f}")
                lines.append(f"  AUC分数: {eff.get('auc_score', 0):.3f}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
    
    def register_callback(self, 
                         event_type: str, 
                         callback: Callable[[str, Any], None]) -> bool:
        """注册事件回调函数
        
        Args:
            event_type: 事件类型 ('status_change', 'metrics_update', 'usage_update', 'effectiveness_update', 'config_update')
            callback: 回调函数
            
        Returns:
            bool: 注册是否成功
        """
        try:
            self._callbacks[event_type].append(callback)
            self.logger.info(f"回调函数注册成功: {event_type}")
            return True
        except Exception as e:
            self.logger.error(f"注册回调函数失败: {str(e)}")
            return False
    
    def _trigger_callbacks(self, event_type: str, algorithm_id: str, data: Any):
        """触发回调函数"""
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    callback(algorithm_id, data)
                except Exception as e:
                    self.logger.error(f"执行回调函数失败: {str(e)}")
    
    def start_monitoring(self):
        """启动实时监控"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("监控已在运行中")
            return
        
        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("实时监控已启动")
    
    def stop_monitoring(self):
        """停止实时监控"""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("实时监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while not self._stop_monitoring:
            try:
                # 收集系统资源使用情况
                self._collect_system_metrics()
                
                # 检查所有算法健康状态
                with self._lock:
                    for algorithm_id in self._algorithms:
                        self._check_algorithm_health(algorithm_id)
                
                # 等待下次收集
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {str(e)}")
                time.sleep(5)  # 出错时等待5秒后重试
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # 获取系统CPU和内存使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # 更新所有算法的系统资源使用情况
            with self._lock:
                for metrics in self._algorithms.values():
                    metrics.cpu_usage = cpu_percent
                    metrics.memory_usage = memory.percent
                    
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {str(e)}")
    
    def remove_algorithm(self, algorithm_id: str) -> bool:
        """移除算法
        
        Args:
            algorithm_id: 算法ID
            
        Returns:
            bool: 移除是否成功
        """
        try:
            with self._lock:
                if algorithm_id not in self._algorithms:
                    self.logger.warning(f"算法 {algorithm_id} 不存在")
                    return False
                
                # 移除所有相关数据
                del self._algorithms[algorithm_id]
                if algorithm_id in self._configs:
                    del self._configs[algorithm_id]
                if algorithm_id in self._usage_stats:
                    del self._usage_stats[algorithm_id]
                if algorithm_id in self._effectiveness:
                    del self._effectiveness[algorithm_id]
                if algorithm_id in self._history:
                    del self._history[algorithm_id]
                
                self.logger.info(f"算法 {algorithm_id} 已移除")
                return True
                
        except Exception as e:
            self.logger.error(f"移除算法 {algorithm_id} 失败: {str(e)}")
            return False
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要
        
        Returns:
            Dict[str, Any]: 健康状态摘要
        """
        with self._lock:
            if not self._algorithms:
                return {"status": "no_algorithms", "message": "没有注册的算法"}
            
            health_counts = defaultdict(int)
            for metrics in self._algorithms.values():
                health_counts[metrics.health_status.value] += 1
            
            total = len(self._algorithms)
            healthy_ratio = health_counts['healthy'] / total if total > 0 else 0
            
            # 确定整体健康状态
            if healthy_ratio >= 0.8:
                overall_status = HealthStatus.HEALTHY
            elif healthy_ratio >= 0.6:
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.CRITICAL
            
            return {
                "overall_status": overall_status.value,
                "total_algorithms": total,
                "health_distribution": dict(health_counts),
                "healthy_ratio": round(healthy_ratio, 3),
                "timestamp": datetime.now()
            }
    
    def export_data(self, format_type: str = 'json') -> str:
        """导出所有数据
        
        Args:
            format_type: 导出格式 ('json', 'csv')
            
        Returns:
            str: 导出的数据
        """
        with self._lock:
            export_data = {
                'algorithms': {aid: asdict(metrics) for aid, metrics in self._algorithms.items()},
                'configs': {aid: asdict(config) for aid, config in self._configs.items()},
                'usage_stats': {aid: asdict(stats) for aid, stats in self._usage_stats.items()},
                'effectiveness': {aid: asdict(eff) for aid, eff in self._effectiveness.items()},
                'export_time': datetime.now(),
                'version': '1.0.0'
            }
            
            if format_type.lower() == 'json':
                return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()


# 测试用例
def run_tests():
    """运行测试用例"""
    print("开始运行AlgorithmStateAggregator测试...")
    
    # 创建聚合器实例
    aggregator = AlgorithmStateAggregator(collection_interval=10)
    
    try:
        # 测试1: 注册算法
        print("\n测试1: 注册算法")
        success = aggregator.register_algorithm(
            algorithm_id="test_algorithm_1",
            algorithm_name="测试算法1",
            version="1.0.0"
        )
        print(f"注册算法结果: {success}")
        
        # 测试2: 更新算法状态
        print("\n测试2: 更新算法状态")
        aggregator.update_algorithm_status(
            algorithm_id="test_algorithm_1",
            status=AlgorithmStatus.RUNNING,
            custom_metrics={"temperature": 25.5, "humidity": 60.0}
        )
        
        # 测试3: 更新算法指标
        print("\n测试3: 更新算法指标")
        aggregator.update_algorithm_metrics(
            algorithm_id="test_algorithm_1",
            cpu_usage=45.5,
            memory_usage=62.3,
            execution_time=1.25,
            success_rate=95.5,
            error_count=2,
            request_count=100,
            avg_response_time=0.85,
            throughput=125.5
        )
        
        # 测试4: 更新使用统计
        print("\n测试4: 更新使用统计")
        aggregator.update_usage_statistics(
            algorithm_id="test_algorithm_1",
            execution_time=1.25,
            success=True,
            user_id="user123"
        )
        
        # 测试5: 更新效果评估
        print("\n测试5: 更新效果评估")
        aggregator.update_effectiveness_evaluation(
            algorithm_id="test_algorithm_1",
            accuracy=0.92,
            precision=0.89,
            recall=0.91,
            f1_score=0.90,
            auc_score=0.94,
            custom_metrics={"mse": 0.05, "mae": 0.03},
            benchmark_results={"dataset_a": 0.93, "dataset_b": 0.88}
        )
        
        # 测试6: 获取算法状态
        print("\n测试6: 获取算法状态")
        status = aggregator.get_algorithm_status("test_algorithm_1")
        if status:
            print(f"算法状态: {status.status.value}")
            print(f"CPU使用率: {status.cpu_usage}%")
            print(f"内存使用率: {status.memory_usage}%")
            print(f"成功率: {status.success_rate}%")
        
        # 测试7: 获取聚合指标
        print("\n测试7: 获取聚合指标")
        metrics = aggregator.get_aggregated_metrics()
        print(f"聚合指标: {json.dumps(metrics, indent=2, default=str)}")
        
        # 测试8: 生成状态报告
        print("\n测试8: 生成状态报告")
        report_json = aggregator.generate_status_report(format_type='json')
        print("JSON报告长度:", len(report_json))
        
        report_text = aggregator.generate_status_report(format_type='text')
        print("文本报告预览:")
        print(report_text[:500] + "..." if len(report_text) > 500 else report_text)
        
        # 测试9: 健康状态检查
        print("\n测试9: 健康状态检查")
        health = aggregator.check_algorithm_health("test_algorithm_1")
        print(f"算法健康状态: {health.value}")
        
        health_summary = aggregator.get_health_summary()
        print(f"健康摘要: {health_summary}")
        
        # 测试10: 注册多个算法并测试聚合功能
        print("\n测试10: 注册多个算法")
        for i in range(2, 6):
            aggregator.register_algorithm(
                algorithm_id=f"test_algorithm_{i}",
                algorithm_name=f"测试算法{i}",
                version="1.0.0"
            )
            aggregator.update_algorithm_status(
                algorithm_id=f"test_algorithm_{i}",
                status=AlgorithmStatus.RUNNING
            )
            aggregator.update_algorithm_metrics(
                algorithm_id=f"test_algorithm_{i}",
                cpu_usage=30.0 + i * 5,
                memory_usage=40.0 + i * 8,
                success_rate=80.0 + i * 3,
                request_count=50 + i * 25
            )
        
        # 获取最终聚合指标
        final_metrics = aggregator.get_aggregated_metrics()
        print(f"\n最终聚合指标:")
        print(f"算法总数: {final_metrics['total_algorithms']}")
        print(f"运行中算法: {final_metrics['running_algorithms']}")
        print(f"平均CPU使用率: {final_metrics['avg_cpu_usage']}%")
        print(f"平均内存使用率: {final_metrics['avg_memory_usage']}%")
        print(f"平均成功率: {final_metrics['avg_success_rate']}%")
        
        # 测试11: 导出数据
        print("\n测试11: 导出数据")
        exported_data = aggregator.export_data()
        print(f"导出数据长度: {len(exported_data)}")
        
        # 测试12: 移除算法
        print("\n测试12: 移除算法")
        remove_success = aggregator.remove_algorithm("test_algorithm_1")
        print(f"移除算法结果: {remove_success}")
        
        # 验证移除结果
        remaining_algorithms = aggregator.get_all_algorithms_status()
        print(f"剩余算法数量: {len(remaining_algorithms)}")
        
        print("\n所有测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        aggregator.stop_monitoring()
        print("测试资源已清理")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    run_tests()