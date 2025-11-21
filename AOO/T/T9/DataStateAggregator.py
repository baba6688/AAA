"""
T9数据状态聚合器模块

该模块提供了完整的数据状态聚合和监控功能，包括：
- 数据模块状态收集
- 数据质量指标聚合
- 数据使用统计
- 数据性能监控
- 数据健康度评估
- 数据资源消耗监控
- 数据状态变更监控
- 数据状态报告生成
- 数据状态告警

Author: T9系统
Date: 2025-11-05
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from threading import Lock, RLock
import statistics
import weakref


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStatus(Enum):
    """数据状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class DataQualityLevel(Enum):
    """数据质量等级枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataModuleInfo:
    """数据模块信息"""
    module_id: str
    module_name: str
    module_type: str
    status: DataStatus
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float  # 完整性 (0-1)
    accuracy: float      # 准确性 (0-1)
    consistency: float   # 一致性 (0-1)
    timeliness: float    # 及时性 (0-1)
    validity: float      # 有效性 (0-1)
    overall_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """计算综合质量分数"""
        metrics = [self.completeness, self.accuracy, self.consistency, 
                  self.timeliness, self.validity]
        self.overall_score = statistics.mean(metrics)


@dataclass
class DataUsageStats:
    """数据使用统计"""
    read_operations: int = 0
    write_operations: int = 0
    query_count: int = 0
    unique_users: Set[str] = field(default_factory=set)
    avg_response_time: float = 0.0
    peak_concurrent_users: int = 0
    data_volume_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataPerformanceMetrics:
    """数据性能指标"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    query_latency_p50: float = 0.0
    query_latency_p95: float = 0.0
    query_latency_p99: float = 0.0
    throughput_qps: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceConsumption:
    """资源消耗监控"""
    cpu_cores: float = 0.0
    memory_mb: float = 0.0
    disk_gb: float = 0.0
    network_mbps: float = 0.0
    storage_gb: float = 0.0
    cost_per_hour: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StateChangeEvent:
    """状态变更事件"""
    module_id: str
    old_status: DataStatus
    new_status: DataStatus
    change_reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataHealthScore:
    """数据健康度评分"""
    overall_score: float  # 0-100
    availability: float   # 可用性 (0-100)
    performance: float    # 性能 (0-100)
    quality: float        # 质量 (0-100)
    security: float       # 安全性 (0-100)
    compliance: float     # 合规性 (0-100)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_health_level(self) -> DataQualityLevel:
        """根据健康分数获取健康等级"""
        if self.overall_score >= 90:
            return DataQualityLevel.EXCELLENT
        elif self.overall_score >= 75:
            return DataQualityLevel.GOOD
        elif self.overall_score >= 60:
            return DataQualityLevel.FAIR
        elif self.overall_score >= 40:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNACCEPTABLE


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    alert_level: AlertLevel
    title: str
    description: str
    module_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataModuleCollector(ABC):
    """数据模块收集器抽象基类"""
    
    @abstractmethod
    async def collect_module_info(self) -> List[DataModuleInfo]:
        """收集模块信息"""
        pass
    
    @abstractmethod
    async def collect_quality_metrics(self, module_id: str) -> DataQualityMetrics:
        """收集质量指标"""
        pass
    
    @abstractmethod
    async def collect_usage_stats(self, module_id: str) -> DataUsageStats:
        """收集使用统计"""
        pass
    
    @abstractmethod
    async def collect_performance_metrics(self, module_id: str) -> DataPerformanceMetrics:
        """收集性能指标"""
        pass
    
    @abstractmethod
    async def collect_resource_consumption(self, module_id: str) -> ResourceConsumption:
        """收集资源消耗"""
        pass


class MockDataModuleCollector(DataModuleCollector):
    """模拟数据模块收集器实现"""
    
    def __init__(self):
        self.modules = [
            DataModuleInfo("module_1", "用户数据模块", "user_data", DataStatus.HEALTHY, datetime.now()),
            DataModuleInfo("module_2", "订单数据模块", "order_data", DataStatus.HEALTHY, datetime.now()),
            DataModuleInfo("module_3", "产品数据模块", "product_data", DataStatus.WARNING, datetime.now()),
        ]
    
    async def collect_module_info(self) -> List[DataModuleInfo]:
        """模拟收集模块信息"""
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return self.modules.copy()
    
    async def collect_quality_metrics(self, module_id: str) -> DataQualityMetrics:
        """模拟收集质量指标"""
        await asyncio.sleep(0.05)
        # 根据模块ID返回不同的质量指标
        if module_id == "module_1":
            return DataQualityMetrics(0.95, 0.98, 0.92, 0.89, 0.96)
        elif module_id == "module_2":
            return DataQualityMetrics(0.88, 0.91, 0.87, 0.85, 0.90)
        else:
            return DataQualityMetrics(0.75, 0.80, 0.78, 0.72, 0.82)
    
    async def collect_usage_stats(self, module_id: str) -> DataUsageStats:
        """模拟收集使用统计"""
        await asyncio.sleep(0.03)
        import random
        return DataUsageStats(
            read_operations=random.randint(100, 1000),
            write_operations=random.randint(50, 500),
            query_count=random.randint(200, 2000),
            unique_users=set([f"user_{i}" for i in range(random.randint(5, 20))]),
            avg_response_time=random.uniform(10, 100),
            peak_concurrent_users=random.randint(10, 50),
            data_volume_mb=random.uniform(100, 1000)
        )
    
    async def collect_performance_metrics(self, module_id: str) -> DataPerformanceMetrics:
        """模拟收集性能指标"""
        await asyncio.sleep(0.04)
        import random
        return DataPerformanceMetrics(
            cpu_usage=random.uniform(10, 80),
            memory_usage=random.uniform(20, 90),
            disk_io=random.uniform(5, 50),
            network_io=random.uniform(1, 20),
            query_latency_p50=random.uniform(10, 50),
            query_latency_p95=random.uniform(50, 200),
            query_latency_p99=random.uniform(100, 500),
            throughput_qps=random.uniform(100, 1000),
            error_rate=random.uniform(0, 5)
        )
    
    async def collect_resource_consumption(self, module_id: str) -> ResourceConsumption:
        """模拟收集资源消耗"""
        await asyncio.sleep(0.02)
        import random
        return ResourceConsumption(
            cpu_cores=random.uniform(0.5, 4.0),
            memory_mb=random.uniform(512, 4096),
            disk_gb=random.uniform(10, 100),
            network_mbps=random.uniform(1, 100),
            storage_gb=random.uniform(50, 500),
            cost_per_hour=random.uniform(0.1, 5.0)
        )


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self._lock = Lock()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        with self._lock:
            self.alert_handlers.append(handler)
    
    def create_alert(self, alert_level: AlertLevel, title: str, description: str, 
                    module_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """创建告警"""
        alert_id = f"alert_{int(time.time() * 1000)}"
        alert = Alert(
            alert_id=alert_id,
            alert_level=alert_level,
            title=title,
            description=description,
            module_id=module_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        # 触发告警处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器执行失败: {e}")
        
        logger.warning(f"创建告警: {alert_level.value} - {title}")
        return alert
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"告警已解决: {alert_id}")
                    break
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """根据级别获取告警"""
        with self._lock:
            return [alert for alert in self.alerts if alert.alert_level == level and not alert.resolved]


class DataStateAggregator:
    """
    T9数据状态聚合器主类
    
    负责收集、聚合、分析和报告所有数据模块的状态信息
    """
    
    def __init__(self, collector: Optional[DataModuleCollector] = None, 
                 collection_interval: int = 60):
        """
        初始化数据状态聚合器
        
        Args:
            collector: 数据模块收集器
            collection_interval: 数据收集间隔（秒）
        """
        self.collector = collector or MockDataModuleCollector()
        self.collection_interval = collection_interval
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = RLock()
        
        # 数据存储
        self.modules: Dict[str, DataModuleInfo] = {}
        self.quality_metrics: Dict[str, DataQualityMetrics] = {}
        self.usage_stats: Dict[str, DataUsageStats] = {}
        self.performance_metrics: Dict[str, DataPerformanceMetrics] = {}
        self.resource_consumption: Dict[str, ResourceConsumption] = {}
        self.health_scores: Dict[str, DataHealthScore] = {}
        self.state_changes: deque = deque(maxlen=1000)
        self.alerts_history: deque = deque(maxlen=5000)
        
        # 告警管理器
        self.alert_manager = AlertManager()
        self._setup_default_alert_handlers()
        
        # 统计信息
        self.collection_stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'last_collection_time': None,
            'avg_collection_duration': 0.0
        }
    
    def _setup_default_alert_handlers(self):
        """设置默认告警处理器"""
        def log_alert_handler(alert: Alert):
            logger.error(f"告警: [{alert.alert_level.value.upper()}] {alert.title} - {alert.description}")
        
        def email_alert_handler(alert: Alert):
            # 模拟邮件发送
            logger.info(f"发送邮件告警: {alert.title}")
        
        self.alert_manager.add_alert_handler(log_alert_handler)
        self.alert_manager.add_alert_handler(email_alert_handler)
    
    async def start_collection(self):
        """启动数据收集"""
        if self.is_running:
            logger.warning("数据收集已在运行中")
            return
        
        self.is_running = True
        self._task = asyncio.create_task(self._collection_loop())
        logger.info("数据状态聚合器已启动")
    
    async def stop_collection(self):
        """停止数据收集"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("数据状态聚合器已停止")
    
    async def _collection_loop(self):
        """数据收集循环"""
        while self.is_running:
            try:
                start_time = time.time()
                await self.collect_all_data()
                duration = time.time() - start_time
                
                with self._lock:
                    self.collection_stats['total_collections'] += 1
                    self.collection_stats['successful_collections'] += 1
                    self.collection_stats['last_collection_time'] = datetime.now()
                    
                    # 计算平均收集时长
                    total = self.collection_stats['total_collections']
                    current_avg = self.collection_stats['avg_collection_duration']
                    self.collection_stats['avg_collection_duration'] = (current_avg * (total - 1) + duration) / total
                
                logger.debug(f"数据收集完成，耗时: {duration:.2f}秒")
                
            except Exception as e:
                with self._lock:
                    self.collection_stats['total_collections'] += 1
                    self.collection_stats['failed_collections'] += 1
                logger.error(f"数据收集失败: {e}")
            
            await asyncio.sleep(self.collection_interval)
    
    async def collect_all_data(self):
        """收集所有数据模块的信息"""
        try:
            # 收集模块信息
            modules = await self.collect_module_status()
            
            # 并发收集其他数据
            tasks = []
            for module in modules:
                tasks.extend([
                    self.collect_quality_metrics(module.module_id),
                    self.collect_usage_statistics(module.module_id),
                    self.collect_performance_metrics(module.module_id),
                    self.collect_resource_consumption(module.module_id),
                ])
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 计算健康度评分
            await self.calculate_health_scores()
            
            # 检查告警条件
            await self.check_alert_conditions()
            
        except Exception as e:
            logger.error(f"收集所有数据时发生错误: {e}")
            raise
    
    async def collect_module_status(self) -> List[DataModuleInfo]:
        """收集所有数据模块状态"""
        try:
            modules = await self.collector.collect_module_info()
            
            with self._lock:
                old_modules = self.modules.copy()
                self.modules.clear()
                
                for module in modules:
                    self.modules[module.module_id] = module
                    
                    # 检查状态变更
                    if module.module_id in old_modules:
                        old_status = old_modules[module.module_id].status
                        if old_status != module.status:
                            self._record_state_change(module.module_id, old_status, module.status)
            
            logger.debug(f"收集到 {len(modules)} 个数据模块状态")
            return modules
            
        except Exception as e:
            logger.error(f"收集模块状态失败: {e}")
            raise
    
    async def collect_quality_metrics(self, module_id: str) -> DataQualityMetrics:
        """收集数据质量指标"""
        try:
            metrics = await self.collector.collect_quality_metrics(module_id)
            
            with self._lock:
                self.quality_metrics[module_id] = metrics
            
            logger.debug(f"收集模块 {module_id} 的质量指标")
            return metrics
            
        except Exception as e:
            logger.error(f"收集模块 {module_id} 质量指标失败: {e}")
            raise
    
    async def collect_usage_statistics(self, module_id: str) -> DataUsageStats:
        """收集数据使用统计"""
        try:
            stats = await self.collector.collect_usage_stats(module_id)
            
            with self._lock:
                self.usage_stats[module_id] = stats
            
            logger.debug(f"收集模块 {module_id} 的使用统计")
            return stats
            
        except Exception as e:
            logger.error(f"收集模块 {module_id} 使用统计失败: {e}")
            raise
    
    async def collect_performance_metrics(self, module_id: str) -> DataPerformanceMetrics:
        """收集数据性能监控"""
        try:
            metrics = await self.collector.collect_performance_metrics(module_id)
            
            with self._lock:
                self.performance_metrics[module_id] = metrics
            
            logger.debug(f"收集模块 {module_id} 的性能指标")
            return metrics
            
        except Exception as e:
            logger.error(f"收集模块 {module_id} 性能指标失败: {e}")
            raise
    
    async def collect_resource_consumption(self, module_id: str) -> ResourceConsumption:
        """收集数据资源消耗监控"""
        try:
            consumption = await self.collector.collect_resource_consumption(module_id)
            
            with self._lock:
                self.resource_consumption[module_id] = consumption
            
            logger.debug(f"收集模块 {module_id} 的资源消耗")
            return consumption
            
        except Exception as e:
            logger.error(f"收集模块 {module_id} 资源消耗失败: {e}")
            raise
    
    def _record_state_change(self, module_id: str, old_status: DataStatus, new_status: DataStatus):
        """记录状态变更事件"""
        event = StateChangeEvent(
            module_id=module_id,
            old_status=old_status,
            new_status=new_status,
            change_reason="状态自动检测"
        )
        
        with self._lock:
            self.state_changes.append(event)
        
        logger.info(f"模块 {module_id} 状态变更: {old_status.value} -> {new_status.value}")
    
    async def calculate_health_scores(self):
        """计算数据健康度评估"""
        with self._lock:
            modules = list(self.modules.keys())
        
        for module_id in modules:
            try:
                # 获取各项指标
                quality = self.quality_metrics.get(module_id)
                performance = self.performance_metrics.get(module_id)
                usage = self.usage_stats.get(module_id)
                resource = self.resource_consumption.get(module_id)
                
                # 计算各项健康度分数
                availability = self._calculate_availability_score(module_id)
                performance_score = self._calculate_performance_score(performance)
                quality_score = (quality.overall_score * 100) if quality else 0
                security_score = 85.0  # 模拟安全分数
                compliance_score = 90.0  # 模拟合规分数
                
                # 计算综合健康分数
                overall_score = statistics.mean([
                    availability, performance_score, quality_score, 
                    security_score, compliance_score
                ])
                
                health_score = DataHealthScore(
                    overall_score=overall_score,
                    availability=availability,
                    performance=performance_score,
                    quality=quality_score,
                    security=security_score,
                    compliance=compliance_score
                )
                
                with self._lock:
                    self.health_scores[module_id] = health_score
                
                logger.debug(f"计算模块 {module_id} 健康度分数: {overall_score:.2f}")
                
            except Exception as e:
                logger.error(f"计算模块 {module_id} 健康度分数失败: {e}")
    
    def _calculate_availability_score(self, module_id: str) -> float:
        """计算可用性分数"""
        module = self.modules.get(module_id)
        if not module:
            return 0.0
        
        # 根据状态计算可用性分数
        status_scores = {
            DataStatus.HEALTHY: 100.0,
            DataStatus.WARNING: 75.0,
            DataStatus.CRITICAL: 25.0,
            DataStatus.MAINTENANCE: 50.0,
            DataStatus.UNKNOWN: 60.0
        }
        
        return status_scores.get(module.status, 0.0)
    
    def _calculate_performance_score(self, performance: Optional[DataPerformanceMetrics]) -> float:
        """计算性能分数"""
        if not performance:
            return 0.0
        
        # 综合性能指标计算分数
        cpu_score = max(0, 100 - performance.cpu_usage)
        memory_score = max(0, 100 - performance.memory_usage)
        latency_score = max(0, 100 - (performance.query_latency_p95 / 10))
        error_score = max(0, 100 - (performance.error_rate * 20))
        
        return statistics.mean([cpu_score, memory_score, latency_score, error_score])
    
    async def check_alert_conditions(self):
        """检查数据状态告警条件"""
        try:
            with self._lock:
                modules = list(self.modules.keys())
            
            for module_id in modules:
                await self._check_module_alerts(module_id)
                
        except Exception as e:
            logger.error(f"检查告警条件失败: {e}")
    
    async def _check_module_alerts(self, module_id: str):
        """检查单个模块的告警条件"""
        try:
            # 检查状态告警
            module = self.modules.get(module_id)
            if module:
                if module.status == DataStatus.CRITICAL:
                    self.alert_manager.create_alert(
                        AlertLevel.CRITICAL,
                        f"模块 {module.module_name} 状态严重",
                        f"模块 {module.module_name} 当前状态为严重，需要立即处理",
                        module_id
                    )
                elif module.status == DataStatus.WARNING:
                    self.alert_manager.create_alert(
                        AlertLevel.WARNING,
                        f"模块 {module.module_name} 状态警告",
                        f"模块 {module.module_name} 当前状态为警告，建议关注",
                        module_id
                    )
            
            # 检查性能告警
            performance = self.performance_metrics.get(module_id)
            if performance:
                if performance.cpu_usage > 80:
                    self.alert_manager.create_alert(
                        AlertLevel.WARNING,
                        f"模块 {module_id} CPU使用率过高",
                        f"CPU使用率为 {performance.cpu_usage:.1f}%",
                        module_id
                    )
                
                if performance.memory_usage > 90:
                    self.alert_manager.create_alert(
                        AlertLevel.ERROR,
                        f"模块 {module_id} 内存使用率过高",
                        f"内存使用率为 {performance.memory_usage:.1f}%",
                        module_id
                    )
                
                if performance.error_rate > 5:
                    self.alert_manager.create_alert(
                        AlertLevel.ERROR,
                        f"模块 {module_id} 错误率过高",
                        f"错误率为 {performance.error_rate:.2f}%",
                        module_id
                    )
            
            # 检查质量告警
            quality = self.quality_metrics.get(module_id)
            if quality and quality.overall_score < 0.7:
                self.alert_manager.create_alert(
                    AlertLevel.WARNING,
                    f"模块 {module_id} 数据质量偏低",
                    f"数据质量分数为 {quality.overall_score:.2f}",
                    module_id
                )
                
        except Exception as e:
            logger.error(f"检查模块 {module_id} 告警条件失败: {e}")
    
    def generate_status_report(self, format_type: str = "json") -> str:
        """
        生成数据状态报告
        
        Args:
            format_type: 报告格式类型 ("json", "text", "html")
            
        Returns:
            格式化的报告字符串
        """
        try:
            with self._lock:
                report_data = {
                    "report_timestamp": datetime.now().isoformat(),
                    "summary": self._generate_summary(),
                    "modules": self._generate_modules_report(),
                    "quality_metrics": self._generate_quality_report(),
                    "performance_metrics": self._generate_performance_report(),
                    "resource_consumption": self._generate_resource_report(),
                    "health_scores": self._generate_health_report(),
                    "active_alerts": [alert.__dict__ for alert in self.alert_manager.get_active_alerts()],
                    "recent_state_changes": [event.__dict__ for event in list(self.state_changes)[-10:]],
                    "collection_stats": self.collection_stats.copy()
                }
            
            if format_type.lower() == "json":
                return json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
            elif format_type.lower() == "text":
                return self._format_text_report(report_data)
            elif format_type.lower() == "html":
                return self._format_html_report(report_data)
            else:
                raise ValueError(f"不支持的报告格式: {format_type}")
                
        except Exception as e:
            logger.error(f"生成状态报告失败: {e}")
            raise
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成汇总信息"""
        with self._lock:
            total_modules = len(self.modules)
            healthy_modules = sum(1 for m in self.modules.values() if m.status == DataStatus.HEALTHY)
            warning_modules = sum(1 for m in self.modules.values() if m.status == DataStatus.WARNING)
            critical_modules = sum(1 for m in self.modules.values() if m.status == DataStatus.CRITICAL)
            
            avg_quality = statistics.mean([m.overall_score for m in self.quality_metrics.values()]) if self.quality_metrics else 0
            active_alerts = len(self.alert_manager.get_active_alerts())
            
            return {
                "total_modules": total_modules,
                "healthy_modules": healthy_modules,
                "warning_modules": warning_modules,
                "critical_modules": critical_modules,
                "average_quality_score": round(avg_quality, 3),
                "active_alerts": active_alerts,
                "system_health_level": self._get_overall_health_level()
            }
    
    def _get_overall_health_level(self) -> str:
        """获取整体健康等级"""
        with self._lock:
            if not self.modules:
                return "unknown"
            
            critical_count = sum(1 for m in self.modules.values() if m.status == DataStatus.CRITICAL)
            warning_count = sum(1 for m in self.modules.values() if m.status == DataStatus.WARNING)
            
            if critical_count > 0:
                return "critical"
            elif warning_count > len(self.modules) * 0.5:
                return "warning"
            else:
                return "healthy"
    
    def _generate_modules_report(self) -> Dict[str, Any]:
        """生成模块报告"""
        with self._lock:
            return {
                module_id: {
                    "name": module.module_name,
                    "type": module.module_type,
                    "status": module.status.value,
                    "last_updated": module.last_updated.isoformat(),
                    "metadata": module.metadata
                }
                for module_id, module in self.modules.items()
            }
    
    def _generate_quality_report(self) -> Dict[str, Any]:
        """生成质量报告"""
        with self._lock:
            return {
                module_id: {
                    "completeness": metrics.completeness,
                    "accuracy": metrics.accuracy,
                    "consistency": metrics.consistency,
                    "timeliness": metrics.timeliness,
                    "validity": metrics.validity,
                    "overall_score": metrics.overall_score,
                    "quality_level": metrics.overall_score,
                    "timestamp": metrics.timestamp.isoformat()
                }
                for module_id, metrics in self.quality_metrics.items()
            }
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        with self._lock:
            return {
                module_id: {
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "disk_io": metrics.disk_io,
                    "network_io": metrics.network_io,
                    "query_latency_p50": metrics.query_latency_p50,
                    "query_latency_p95": metrics.query_latency_p95,
                    "query_latency_p99": metrics.query_latency_p99,
                    "throughput_qps": metrics.throughput_qps,
                    "error_rate": metrics.error_rate,
                    "timestamp": metrics.timestamp.isoformat()
                }
                for module_id, metrics in self.performance_metrics.items()
            }
    
    def _generate_resource_report(self) -> Dict[str, Any]:
        """生成资源报告"""
        with self._lock:
            return {
                module_id: {
                    "cpu_cores": consumption.cpu_cores,
                    "memory_mb": consumption.memory_mb,
                    "disk_gb": consumption.disk_gb,
                    "network_mbps": consumption.network_mbps,
                    "storage_gb": consumption.storage_gb,
                    "cost_per_hour": consumption.cost_per_hour,
                    "timestamp": consumption.timestamp.isoformat()
                }
                for module_id, consumption in self.resource_consumption.items()
            }
    
    def _generate_health_report(self) -> Dict[str, Any]:
        """生成健康度报告"""
        with self._lock:
            return {
                module_id: {
                    "overall_score": score.overall_score,
                    "availability": score.availability,
                    "performance": score.performance,
                    "quality": score.quality,
                    "security": score.security,
                    "compliance": score.compliance,
                    "health_level": score.get_health_level().value,
                    "timestamp": score.timestamp.isoformat()
                }
                for module_id, score in self.health_scores.items()
            }
    
    def _format_text_report(self, data: Dict[str, Any]) -> str:
        """格式化文本报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("T9数据状态聚合报告")
        lines.append("=" * 60)
        lines.append(f"报告时间: {data['report_timestamp']}")
        lines.append("")
        
        # 汇总信息
        summary = data['summary']
        lines.append("【系统概览】")
        lines.append(f"总模块数: {summary['total_modules']}")
        lines.append(f"健康模块: {summary['healthy_modules']}")
        lines.append(f"警告模块: {summary['warning_modules']}")
        lines.append(f"严重模块: {summary['critical_modules']}")
        lines.append(f"平均质量分数: {summary['average_quality_score']}")
        lines.append(f"活跃告警: {summary['active_alerts']}")
        lines.append(f"系统健康等级: {summary['system_health_level']}")
        lines.append("")
        
        # 模块状态
        lines.append("【模块状态】")
        for module_id, module_info in data['modules'].items():
            lines.append(f"{module_info['name']} ({module_id}): {module_info['status']}")
        lines.append("")
        
        # 活跃告警
        if data['active_alerts']:
            lines.append("【活跃告警】")
            for alert in data['active_alerts']:
                lines.append(f"[{alert['alert_level']}] {alert['title']}: {alert['description']}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _format_html_report(self, data: Dict[str, Any]) -> str:
        """格式化HTML报告"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>T9数据状态聚合报告</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 3px solid #007cba; }}
        .alert {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 3px solid #ffc107; }}
        .critical {{ background-color: #f8d7da; padding: 10px; margin: 5px 0; border-left: 3px solid #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>T9数据状态聚合报告</h1>
        <p>报告时间: {data['report_timestamp']}</p>
    </div>
    
    <div class="section">
        <h2>系统概览</h2>
        <div class="metric">
            <strong>总模块数:</strong> {data['summary']['total_modules']}<br>
            <strong>健康模块:</strong> {data['summary']['healthy_modules']}<br>
            <strong>警告模块:</strong> {data['summary']['warning_modules']}<br>
            <strong>严重模块:</strong> {data['summary']['critical_modules']}<br>
            <strong>平均质量分数:</strong> {data['summary']['average_quality_score']}<br>
            <strong>活跃告警:</strong> {data['summary']['active_alerts']}<br>
            <strong>系统健康等级:</strong> {data['summary']['system_health_level']}
        </div>
    </div>
    
    <div class="section">
        <h2>模块状态</h2>
"""
        
        for module_id, module_info in data['modules'].items():
            status_class = "critical" if module_info['status'] == 'critical' else "metric"
            html += f"""
        <div class="{status_class}">
            <strong>{module_info['name']}</strong> ({module_id})<br>
            类型: {module_info['type']}<br>
            状态: {module_info['status']}<br>
            最后更新: {module_info['last_updated']}
        </div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>活跃告警</h2>
"""
        
        for alert in data['active_alerts']:
            html += f"""
        <div class="alert">
            <strong>[{alert['alert_level'].upper()}] {alert['title']}</strong><br>
            {alert['description']}<br>
            时间: {alert['timestamp']}
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        return html
    
    def get_module_status(self, module_id: str) -> Optional[Dict[str, Any]]:
        """获取指定模块的详细状态"""
        with self._lock:
            if module_id not in self.modules:
                return None
            
            return {
                "module_info": self.modules[module_id].__dict__,
                "quality_metrics": self.quality_metrics.get(module_id).__dict__ if module_id in self.quality_metrics else None,
                "usage_stats": self.usage_stats.get(module_id).__dict__ if module_id in self.usage_stats else None,
                "performance_metrics": self.performance_metrics.get(module_id).__dict__ if module_id in self.performance_metrics else None,
                "resource_consumption": self.resource_consumption.get(module_id).__dict__ if module_id in self.resource_consumption else None,
                "health_score": self.health_scores.get(module_id).__dict__ if module_id in self.health_scores else None
            }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览信息"""
        with self._lock:
            return {
                "total_modules": len(self.modules),
                "modules_by_status": {
                    status.value: sum(1 for m in self.modules.values() if m.status == status)
                    for status in DataStatus
                },
                "average_quality_score": statistics.mean([m.overall_score for m in self.quality_metrics.values()]) if self.quality_metrics else 0,
                "active_alerts_count": len(self.alert_manager.get_active_alerts()),
                "collection_stats": self.collection_stats.copy(),
                "last_updated": max([m.last_updated for m in self.modules.values()]) if self.modules else None
            }
    
    def export_data(self, format_type: str = "json") -> str:
        """导出所有数据"""
        with self._lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "modules": {mid: minfo.__dict__ for mid, minfo in self.modules.items()},
                "quality_metrics": {mid: qm.__dict__ for mid, qm in self.quality_metrics.items()},
                "usage_stats": {mid: us.__dict__ for mid, us in self.usage_stats.items()},
                "performance_metrics": {mid: pm.__dict__ for mid, pm in self.performance_metrics.items()},
                "resource_consumption": {mid: rc.__dict__ for mid, rc in self.resource_consumption.items()},
                "health_scores": {mid: hs.__dict__ for mid, hs in self.health_scores.items()},
                "state_changes": [sc.__dict__ for sc in self.state_changes],
                "alerts": [alert.__dict__ for alert in self.alert_manager.alerts]
            }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")


# 测试用例
class TestDataStateAggregator:
    """数据状态聚合器测试类"""
    
    def __init__(self):
        self.aggregator = DataStateAggregator(collection_interval=5)
    
    async def test_basic_functionality(self):
        """测试基本功能"""
        print("开始测试基本功能...")
        
        # 启动数据收集
        await self.aggregator.start_collection()
        
        # 等待一段时间让数据收集完成
        await asyncio.sleep(2)
        
        # 测试获取系统概览
        overview = self.aggregator.get_system_overview()
        print(f"系统概览: {overview}")
        
        # 测试生成报告
        json_report = self.aggregator.generate_status_report("json")
        print(f"JSON报告长度: {len(json_report)} 字符")
        
        text_report = self.aggregator.generate_status_report("text")
        print(f"文本报告:\n{text_report}")
        
        # 测试获取模块状态
        if self.aggregator.modules:
            module_id = list(self.aggregator.modules.keys())[0]
            module_status = self.aggregator.get_module_status(module_id)
            print(f"模块 {module_id} 状态: {module_status is not None}")
        
        # 测试导出数据
        export_data = self.aggregator.export_data()
        print(f"导出数据长度: {len(export_data)} 字符")
        
        # 停止数据收集
        await self.aggregator.stop_collection()
        
        print("基本功能测试完成!")
    
    async def test_alert_system(self):
        """测试告警系统"""
        print("开始测试告警系统...")
        
        # 创建测试告警
        alert = self.aggregator.alert_manager.create_alert(
            AlertLevel.WARNING,
            "测试告警",
            "这是一个测试告警",
            "test_module"
        )
        
        print(f"创建告警: {alert.alert_id}")
        
        # 获取活跃告警
        active_alerts = self.aggregator.alert_manager.get_active_alerts()
        print(f"活跃告警数量: {len(active_alerts)}")
        
        # 解决告警
        self.aggregator.alert_manager.resolve_alert(alert.alert_id)
        
        # 再次检查活跃告警
        active_alerts_after = self.aggregator.alert_manager.get_active_alerts()
        print(f"解决后活跃告警数量: {len(active_alerts_after)}")
        
        print("告警系统测试完成!")
    
    async def test_performance(self):
        """测试性能"""
        print("开始性能测试...")
        
        start_time = time.time()
        
        # 启动数据收集
        await self.aggregator.start_collection()
        
        # 运行一段时间
        await asyncio.sleep(3)
        
        # 停止数据收集
        await self.aggregator.stop_collection()
        
        end_time = time.time()
        
        # 检查收集统计
        stats = self.aggregator.collection_stats
        print(f"总收集次数: {stats['total_collections']}")
        print(f"成功收集次数: {stats['successful_collections']}")
        print(f"失败收集次数: {stats['failed_collections']}")
        print(f"平均收集时长: {stats['avg_collection_duration']:.3f}秒")
        print(f"测试总耗时: {end_time - start_time:.3f}秒")
        
        print("性能测试完成!")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 50)
        print("T9数据状态聚合器测试开始")
        print("=" * 50)
        
        try:
            await self.test_basic_functionality()
            print()
            
            await self.test_alert_system()
            print()
            
            await self.test_performance()
            print()
            
            print("=" * 50)
            print("所有测试完成!")
            print("=" * 50)
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数 - 演示数据状态聚合器的使用"""
    print("T9数据状态聚合器演示")
    print("=" * 40)
    
    # 创建聚合器实例
    aggregator = DataStateAggregator(collection_interval=3)
    
    try:
        # 启动数据收集
        print("启动数据收集...")
        await aggregator.start_collection()
        
        # 运行一段时间
        print("运行数据收集（10秒）...")
        await asyncio.sleep(10)
        
        # 生成并显示报告
        print("\n生成系统状态报告...")
        report = aggregator.generate_status_report("text")
        print(report)
        
        # 显示系统概览
        print("\n系统概览信息:")
        overview = aggregator.get_system_overview()
        for key, value in overview.items():
            print(f"  {key}: {value}")
        
        # 显示活跃告警
        active_alerts = aggregator.alert_manager.get_active_alerts()
        print(f"\n活跃告警数量: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  [{alert.alert_level.value}] {alert.title}: {alert.description}")
        
    except KeyboardInterrupt:
        print("\n收到中断信号...")
    except Exception as e:
        print(f"运行过程中发生错误: {e}")
    finally:
        # 停止数据收集
        print("\n停止数据收集...")
        await aggregator.stop_collection()
        print("演示结束!")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())
    
    # 运行测试
    print("\n" + "=" * 50)
    print("运行测试用例")
    print("=" * 50)
    test_instance = TestDataStateAggregator()
    asyncio.run(test_instance.run_all_tests())