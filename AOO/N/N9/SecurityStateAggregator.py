#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N9安全状态聚合器

该模块实现了一个全面的安全状态聚合器，用于收集、聚合、分析和可视化系统安全状态。
主要功能包括：
- 安全模块状态收集
- 安全指标聚合
- 安全威胁聚合
- 安全事件聚合
- 安全状态评估
- 安全趋势分析
- 安全报告生成
- 安全告警管理
- 安全状态可视化


版本: 1.0.0
创建时间: 2025-11-06
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import threading
import queue
import statistics
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """安全等级枚举"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "严重"


class AlertSeverity(Enum):
    """告警严重程度枚举"""
    INFO = "信息"
    WARNING = "警告"
    ERROR = "错误"
    CRITICAL = "严重"


class EventType(Enum):
    """安全事件类型枚举"""
    INTRUSION = "入侵"
    MALWARE = "恶意软件"
    DATA_BREACH = "数据泄露"
    SYSTEM_FAILURE = "系统故障"
    UNAUTHORIZED_ACCESS = "未授权访问"
    POLICY_VIOLATION = "策略违规"
    NETWORK_ATTACK = "网络攻击"


@dataclass
class SecurityMetric:
    """安全指标数据类"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    category: str = "general"
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, critical


@dataclass
class SecurityThreat:
    """安全威胁数据类"""
    id: str
    name: str
    type: str
    severity: SecurityLevel
    source: str
    target: str
    timestamp: datetime
    description: str
    indicators: List[str] = field(default_factory=list)
    mitigation_status: str = "pending"  # pending, in_progress, resolved
    confidence: float = 0.0


@dataclass
class SecurityEvent:
    """安全事件数据类"""
    id: str
    type: EventType
    severity: AlertSeverity
    source: str
    timestamp: datetime
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, resolved, investigating
    affected_systems: List[str] = field(default_factory=list)


@dataclass
class SecurityAlert:
    """安全告警数据类"""
    id: str
    title: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityReport:
    """安全报告数据类"""
    id: str
    title: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    summary: Dict[str, Any]
    metrics: List[SecurityMetric]
    threats: List[SecurityThreat]
    events: List[SecurityEvent]
    alerts: List[SecurityAlert]
    recommendations: List[str] = field(default_factory=list)


class SecurityModuleInterface(ABC):
    """安全模块接口抽象类"""
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> List[SecurityMetric]:
        """获取安全指标"""
        pass
    
    @abstractmethod
    async def get_threats(self) -> List[SecurityThreat]:
        """获取安全威胁"""
        pass
    
    @abstractmethod
    async def get_events(self) -> List[SecurityEvent]:
        """获取安全事件"""
        pass


class MockSecurityModule(SecurityModuleInterface):
    """模拟安全模块实现"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._threat_id_counter = 0
        self._event_id_counter = 0
    
    async def get_status(self) -> Dict[str, Any]:
        """模拟获取模块状态"""
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return {
            "name": self.module_name,
            "status": "online",
            "health": "good",
            "last_update": datetime.now().isoformat(),
            "cpu_usage": np.random.uniform(10, 30),
            "memory_usage": np.random.uniform(20, 50),
            "active_connections": np.random.randint(50, 200)
        }
    
    async def get_metrics(self) -> List[SecurityMetric]:
        """模拟获取安全指标"""
        await asyncio.sleep(0.1)
        metrics = []
        for i in range(np.random.randint(3, 8)):
            metric = SecurityMetric(
                name=f"{self.module_name}_metric_{i}",
                value=np.random.uniform(0, 100),
                unit="%",
                timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
                source=self.module_name,
                category="performance",
                threshold=80.0
            )
            metrics.append(metric)
        return metrics
    
    async def get_threats(self) -> List[SecurityThreat]:
        """模拟获取安全威胁"""
        await asyncio.sleep(0.1)
        threats = []
        threat_count = np.random.randint(0, 3)
        
        for i in range(threat_count):
            self._threat_id_counter += 1
            threat = SecurityThreat(
                id=f"{self.module_name}_threat_{self._threat_id_counter}",
                name=f"威胁_{i}",
                type=np.random.choice(["恶意软件", "入侵", "异常行为"]),
                severity=np.random.choice(list(SecurityLevel)),
                source="external",
                target="system",
                timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 120)),
                description=f"来自{self.module_name}的威胁检测",
                confidence=np.random.uniform(0.3, 0.9)
            )
            threats.append(threat)
        return threats
    
    async def get_events(self) -> List[SecurityEvent]:
        """模拟获取安全事件"""
        await asyncio.sleep(0.1)
        events = []
        event_count = np.random.randint(0, 5)
        
        for i in range(event_count):
            self._event_id_counter += 1
            event = SecurityEvent(
                id=f"{self.module_name}_event_{self._event_id_counter}",
                type=np.random.choice(list(EventType)),
                severity=np.random.choice(list(AlertSeverity)),
                source=self.module_name,
                timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
                description=f"来自{self.module_name}的安全事件",
                affected_systems=["system_1", "system_2"]
            )
            events.append(event)
        return events


class SecurityStateAggregator:
    """
    N9安全状态聚合器
    
    该类负责收集、聚合、分析和可视化系统安全状态。
    支持多个安全模块的数据聚合，提供全面的安全状态监控和报告功能。
    """
    
    def __init__(self, 
                 collection_interval: int = 60,
                 max_history_size: int = 1000,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        初始化安全状态聚合器
        
        Args:
            collection_interval: 数据收集间隔（秒）
            max_history_size: 历史数据最大存储数量
            alert_thresholds: 告警阈值配置
        """
        self.collection_interval = collection_interval
        self.max_history_size = max_history_size
        self.alert_thresholds = alert_thresholds or {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "threat_severity": 0.7,
            "event_frequency": 100
        }
        
        # 安全模块管理
        self.security_modules: Dict[str, SecurityModuleInterface] = {}
        self.module_status: Dict[str, Dict[str, Any]] = {}
        
        # 数据存储
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.threats_history: deque = deque(maxlen=max_history_size)
        self.events_history: deque = deque(maxlen=max_history_size)
        self.alerts_history: deque = deque(maxlen=max_history_size)
        
        # 当前状态
        self.current_metrics: List[SecurityMetric] = []
        self.current_threats: List[SecurityThreat] = []
        self.current_events: List[SecurityEvent] = []
        self.current_alerts: List[SecurityAlert] = []
        
        # 聚合结果
        self.aggregated_metrics: Dict[str, Any] = {}
        self.aggregated_threats: Dict[str, Any] = {}
        self.aggregated_events: Dict[str, Any] = {}
        self.security_assessment: Dict[str, Any] = {}
        self.trend_analysis: Dict[str, Any] = {}
        
        # 控制变量
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # 告警回调函数
        self.alert_callbacks: List[Callable[[SecurityAlert], None]] = []
        
        logger.info("N9安全状态聚合器初始化完成")
    
    def register_security_module(self, module: SecurityModuleInterface) -> None:
        """
        注册安全模块
        
        Args:
            module: 安全模块实例
        """
        module_name = getattr(module, 'module_name', f"module_{len(self.security_modules)}")
        self.security_modules[module_name] = module
        logger.info(f"安全模块已注册: {module_name}")
    
    def register_alert_callback(self, callback: Callable[[SecurityAlert], None]) -> None:
        """
        注册告警回调函数
        
        Args:
            callback: 告警处理回调函数
        """
        self.alert_callbacks.append(callback)
        logger.info("告警回调函数已注册")
    
    async def start_collection(self) -> None:
        """启动数据收集"""
        if self._running:
            logger.warning("数据收集已在运行中")
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("安全状态数据收集已启动")
    
    async def stop_collection(self) -> None:
        """停止数据收集"""
        if not self._running:
            logger.warning("数据收集未在运行")
            return
        
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("安全状态数据收集已停止")
    
    async def _collection_loop(self) -> None:
        """数据收集循环"""
        while self._running:
            try:
                await self.collect_all_states()
                await self.aggregate_security_data()
                await self.evaluate_security_status()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"数据收集循环错误: {e}")
                await asyncio.sleep(5)  # 错误后等待5秒再继续
    
    async def collect_all_states(self) -> None:
        """收集所有安全模块状态"""
        logger.info("开始收集安全模块状态")
        
        # 收集所有模块数据
        tasks = []
        for module_name, module in self.security_modules.items():
            tasks.extend([
                self._collect_module_status(module_name, module),
                self._collect_module_metrics(module_name, module),
                self._collect_module_threats(module_name, module),
                self._collect_module_events(module_name, module)
            ])
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"状态收集完成 - 模块数: {len(self.security_modules)}")
    
    async def _collect_module_status(self, module_name: str, module: SecurityModuleInterface) -> None:
        """收集单个模块状态"""
        try:
            status = await module.get_status()
            self.module_status[module_name] = status
        except Exception as e:
            logger.error(f"收集模块 {module_name} 状态失败: {e}")
            self.module_status[module_name] = {"status": "error", "error": str(e)}
    
    async def _collect_module_metrics(self, module_name: str, module: SecurityModuleInterface) -> None:
        """收集单个模块指标"""
        try:
            metrics = await module.get_metrics()
            self.current_metrics.extend(metrics)
            self.metrics_history.extend(metrics)
        except Exception as e:
            logger.error(f"收集模块 {module_name} 指标失败: {e}")
    
    async def _collect_module_threats(self, module_name: str, module: SecurityModuleInterface) -> None:
        """收集单个模块威胁"""
        try:
            threats = await module.get_threats()
            self.current_threats.extend(threats)
            self.threats_history.extend(threats)
        except Exception as e:
            logger.error(f"收集模块 {module_name} 威胁失败: {e}")
    
    async def _collect_module_events(self, module_name: str, module: SecurityModuleInterface) -> None:
        """收集单个模块事件"""
        try:
            events = await module.get_events()
            self.current_events.extend(events)
            self.events_history.extend(events)
        except Exception as e:
            logger.error(f"收集模块 {module_name} 事件失败: {e}")
    
    async def aggregate_security_data(self) -> None:
        """聚合安全数据"""
        logger.info("开始聚合安全数据")
        
        # 聚合指标数据
        self.aggregated_metrics = await self._aggregate_metrics()
        
        # 聚合威胁数据
        self.aggregated_threats = await self._aggregate_threats()
        
        # 聚合事件数据
        self.aggregated_events = await self._aggregate_events()
        
        logger.info("安全数据聚合完成")
    
    async def _aggregate_metrics(self) -> Dict[str, Any]:
        """聚合安全指标"""
        if not self.current_metrics:
            return {}
        
        # 按类别分组
        metrics_by_category = defaultdict(list)
        for metric in self.current_metrics:
            metrics_by_category[metric.category].append(metric)
        
        aggregated = {}
        for category, metrics in metrics_by_category.items():
            if not metrics:
                continue
            
            # 计算统计值
            values = [m.value for m in metrics]
            aggregated[category] = {
                "count": len(metrics),
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "latest": max(m.timestamp for m in metrics).isoformat(),
                "threshold_breaches": sum(1 for m in metrics if m.threshold and m.value > m.threshold)
            }
        
        return aggregated
    
    async def _aggregate_threats(self) -> Dict[str, Any]:
        """聚合安全威胁"""
        if not self.current_threats:
            return {}
        
        # 按严重程度分组
        threats_by_severity = defaultdict(list)
        for threat in self.current_threats:
            threats_by_severity[threat.severity.value].append(threat)
        
        # 按类型分组
        threats_by_type = defaultdict(list)
        for threat in self.current_threats:
            threats_by_type[threat.type].append(threat)
        
        aggregated = {
            "total_count": len(self.current_threats),
            "by_severity": {k: len(v) for k, v in threats_by_severity.items()},
            "by_type": {k: len(v) for k, v in threats_by_type.items()},
            "average_confidence": statistics.mean([t.confidence for t in self.current_threats]),
            "pending_mitigation": sum(1 for t in self.current_threats if t.mitigation_status == "pending"),
            "high_confidence_threats": sum(1 for t in self.current_threats if t.confidence > 0.8)
        }
        
        return aggregated
    
    async def _aggregate_events(self) -> Dict[str, Any]:
        """聚合安全事件"""
        if not self.current_events:
            return {}
        
        # 按严重程度分组
        events_by_severity = defaultdict(list)
        for event in self.current_events:
            events_by_severity[event.severity.value].append(event)
        
        # 按类型分组
        events_by_type = defaultdict(list)
        for event in self.current_events:
            events_by_type[event.type.value].append(event)
        
        # 按状态分组
        events_by_status = defaultdict(list)
        for event in self.current_events:
            events_by_status[event.status].append(event)
        
        aggregated = {
            "total_count": len(self.current_events),
            "by_severity": {k: len(v) for k, v in events_by_severity.items()},
            "by_type": {k: len(v) for k, v in events_by_type.items()},
            "by_status": {k: len(v) for k, v in events_by_status.items()},
            "active_events": sum(1 for e in self.current_events if e.status == "active"),
            "recent_events": sum(1 for e in self.current_events 
                               if (datetime.now() - e.timestamp).total_seconds() < 3600)  # 最近1小时
        }
        
        return aggregated
    
    async def evaluate_security_status(self) -> Dict[str, Any]:
        """评估安全状态"""
        logger.info("开始评估安全状态")
        
        assessment = {
            "overall_score": 0.0,
            "level": SecurityLevel.LOW.value,
            "risk_factors": [],
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 计算总体安全分数
        score = 100.0
        
        # 威胁影响
        if self.aggregated_threats:
            high_threats = self.aggregated_threats.get("by_severity", {}).get("严重", 0)
            medium_threats = self.aggregated_threats.get("by_severity", {}).get("高", 0)
            score -= high_threats * 20 + medium_threats * 10
        
        # 事件影响
        if self.aggregated_events:
            critical_events = self.aggregated_events.get("by_severity", {}).get("严重", 0)
            error_events = self.aggregated_events.get("by_severity", {}).get("错误", 0)
            score -= critical_events * 15 + error_events * 5
        
        # 指标阈值违反
        if self.aggregated_metrics:
            for category, metrics in self.aggregated_metrics.items():
                threshold_breaches = metrics.get("threshold_breaches", 0)
                score -= threshold_breaches * 2
        
        score = max(0.0, score)
        assessment["overall_score"] = score
        
        # 确定安全等级
        if score >= 90:
            assessment["level"] = SecurityLevel.LOW.value
        elif score >= 70:
            assessment["level"] = SecurityLevel.MEDIUM.value
        elif score >= 50:
            assessment["level"] = SecurityLevel.HIGH.value
        else:
            assessment["level"] = SecurityLevel.CRITICAL.value
        
        # 风险因素分析
        if self.aggregated_threats and self.aggregated_threats.get("total_count", 0) > 0:
            assessment["risk_factors"].append("检测到安全威胁")
        
        if self.aggregated_events and self.aggregated_events.get("active_events", 0) > 0:
            assessment["risk_factors"].append("存在活跃安全事件")
        
        # 生成建议
        if score < 70:
            assessment["recommendations"].append("建议加强安全监控")
        if self.aggregated_threats and self.aggregated_threats.get("pending_mitigation", 0) > 0:
            assessment["recommendations"].append("及时处理待处理的威胁")
        
        self.security_assessment = assessment
        logger.info(f"安全状态评估完成 - 分数: {score:.1f}, 等级: {assessment['level']}")
        
        # 检查是否需要生成告警
        await self._check_alert_conditions(assessment)
        
        return assessment
    
    async def _check_alert_conditions(self, assessment: Dict[str, Any]) -> None:
        """检查告警条件"""
        # 分数低于阈值
        if assessment["overall_score"] < 50:
            await self._create_alert(
                title="安全分数严重下降",
                severity=AlertSeverity.CRITICAL,
                message=f"当前安全分数: {assessment['overall_score']:.1f}",
                source="security_assessment"
            )
        
        # 高严重性威胁
        if self.aggregated_threats:
            critical_threats = self.aggregated_threats.get("by_severity", {}).get("严重", 0)
            if critical_threats > 0:
                await self._create_alert(
                    title="检测到严重威胁",
                    severity=AlertSeverity.CRITICAL,
                    message=f"检测到 {critical_threats} 个严重威胁",
                    source="threat_detector"
                )
    
    async def _create_alert(self, title: str, severity: AlertSeverity, 
                          message: str, source: str) -> None:
        """创建安全告警"""
        alert = SecurityAlert(
            id=f"alert_{int(time.time())}_{len(self.current_alerts)}",
            title=title,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            source=source
        )
        
        self.current_alerts.append(alert)
        self.alerts_history.append(alert)
        
        # 调用告警回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调函数执行失败: {e}")
        
        logger.warning(f"创建告警: {title} - {message}")
    
    async def analyze_security_trends(self, 
                                    hours: int = 24) -> Dict[str, Any]:
        """分析安全趋势"""
        logger.info(f"开始分析最近 {hours} 小时的安全趋势")
        
        # 筛选历史数据
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 过滤历史数据
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        recent_threats = [t for t in self.threats_history if t.timestamp >= cutoff_time]
        recent_events = [e for e in self.events_history if e.timestamp >= cutoff_time]
        recent_alerts = [a for a in self.alerts_history if a.timestamp >= cutoff_time]
        
        trend_analysis = {
            "period": f"最近 {hours} 小时",
            "metrics_trend": await self._analyze_metrics_trend(recent_metrics),
            "threats_trend": await self._analyze_threats_trend(recent_threats),
            "events_trend": await self._analyze_events_trend(recent_events),
            "alerts_trend": await self._analyze_alerts_trend(recent_alerts),
            "timestamp": datetime.now().isoformat()
        }
        
        self.trend_analysis = trend_analysis
        logger.info("安全趋势分析完成")
        
        return trend_analysis
    
    async def _analyze_metrics_trend(self, metrics: List[SecurityMetric]) -> Dict[str, Any]:
        """分析指标趋势"""
        if not metrics:
            return {"trend": "无数据", "change_rate": 0.0}
        
        # 按时间排序
        metrics.sort(key=lambda x: x.timestamp)
        
        # 计算变化率
        if len(metrics) < 2:
            return {"trend": "数据不足", "change_rate": 0.0}
        
        # 按类别分组分析
        trends_by_category = {}
        for category in set(m.category for m in metrics):
            category_metrics = [m for m in metrics if m.category == category]
            if len(category_metrics) >= 2:
                first_value = category_metrics[0].value
                last_value = category_metrics[-1].value
                change_rate = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                
                if change_rate > 5:
                    trend = "上升"
                elif change_rate < -5:
                    trend = "下降"
                else:
                    trend = "稳定"
                
                trends_by_category[category] = {
                    "trend": trend,
                    "change_rate": change_rate,
                    "first_value": first_value,
                    "last_value": last_value
                }
        
        return trends_by_category
    
    async def _analyze_threats_trend(self, threats: List[SecurityThreat]) -> Dict[str, Any]:
        """分析威胁趋势"""
        if not threats:
            return {"trend": "无威胁", "total_count": 0}
        
        # 按小时分组统计
        hourly_counts = defaultdict(int)
        for threat in threats:
            hour_key = threat.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
        
        # 计算趋势
        if len(hourly_counts) < 2:
            trend = "数据不足"
        else:
            counts = list(hourly_counts.values())
            if counts[-1] > counts[0]:
                trend = "上升"
            elif counts[-1] < counts[0]:
                trend = "下降"
            else:
                trend = "稳定"
        
        return {
            "trend": trend,
            "total_count": len(threats),
            "hourly_distribution": dict(hourly_counts),
            "severity_distribution": {
                severity.value: len([t for t in threats if t.severity == severity])
                for severity in SecurityLevel
            }
        }
    
    async def _analyze_events_trend(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """分析事件趋势"""
        if not events:
            return {"trend": "无事件", "total_count": 0}
        
        # 按严重程度统计
        severity_counts = defaultdict(int)
        for event in events:
            severity_counts[event.severity.value] += 1
        
        # 按类型统计
        type_counts = defaultdict(int)
        for event in events:
            type_counts[event.type.value] += 1
        
        return {
            "trend": "稳定",
            "total_count": len(events),
            "severity_distribution": dict(severity_counts),
            "type_distribution": dict(type_counts),
            "active_count": len([e for e in events if e.status == "active"])
        }
    
    async def _analyze_alerts_trend(self, alerts: List[SecurityAlert]) -> Dict[str, Any]:
        """分析告警趋势"""
        if not alerts:
            return {"trend": "无告警", "total_count": 0}
        
        # 按严重程度统计
        severity_counts = defaultdict(int)
        for alert in alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            "trend": "稳定",
            "total_count": len(alerts),
            "severity_distribution": dict(severity_counts),
            "acknowledged_count": len([a for a in alerts if a.acknowledged]),
            "resolved_count": len([a for a in alerts if a.resolved])
        }
    
    async def generate_security_report(self, 
                                     report_type: str = "summary",
                                     period_hours: int = 24) -> SecurityReport:
        """生成安全报告"""
        logger.info(f"开始生成 {report_type} 类型的安全报告")
        
        # 收集报告数据
        await self.collect_all_states()
        await self.aggregate_security_data()
        await self.evaluate_security_status()
        trend_analysis = await self.analyze_security_trends(period_hours)
        
        # 生成报告ID
        report_id = f"security_report_{int(time.time())}"
        
        # 生成建议
        recommendations = []
        if self.security_assessment.get("overall_score", 100) < 70:
            recommendations.append("建议加强系统安全监控")
        if self.aggregated_threats.get("pending_mitigation", 0) > 0:
            recommendations.append("及时处理待处理的安全威胁")
        if self.aggregated_events.get("active_events", 0) > 0:
            recommendations.append("关注活跃的安全事件")
        
        # 创建报告
        report = SecurityReport(
            id=report_id,
            title=f"{report_type.title()}安全报告",
            generated_at=datetime.now(),
            period_start=datetime.now() - timedelta(hours=period_hours),
            period_end=datetime.now(),
            summary={
                "security_score": self.security_assessment.get("overall_score", 0),
                "security_level": self.security_assessment.get("level", "未知"),
                "total_threats": self.aggregated_threats.get("total_count", 0),
                "total_events": self.aggregated_events.get("total_count", 0),
                "total_alerts": len(self.current_alerts),
                "trend_analysis": trend_analysis
            },
            metrics=self.current_metrics.copy(),
            threats=self.current_threats.copy(),
            events=self.current_events.copy(),
            alerts=self.current_alerts.copy(),
            recommendations=recommendations
        )
        
        logger.info(f"安全报告生成完成: {report_id}")
        return report
    
    def visualize_security_status(self, 
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 10)) -> None:
        """可视化安全状态"""
        logger.info("开始生成安全状态可视化图表")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            fig.suptitle('N9安全状态聚合器 - 安全状态概览', fontsize=16, fontweight='bold')
            
            # 1. 安全分数仪表盘
            ax1 = axes[0, 0]
            score = self.security_assessment.get("overall_score", 0)
            colors = ['red' if score < 50 else 'orange' if score < 70 else 'green']
            ax1.pie([score, 100-score], colors=colors + ['lightgray'], startangle=90)
            ax1.set_title(f'安全分数: {score:.1f}')
            
            # 2. 威胁分布
            ax2 = axes[0, 1]
            if self.aggregated_threats.get("by_severity"):
                severities = list(self.aggregated_threats["by_severity"].keys())
                counts = list(self.aggregated_threats["by_severity"].values())
                ax2.bar(severities, counts, color=['red', 'orange', 'yellow', 'green'])
                ax2.set_title('威胁严重程度分布')
                ax2.tick_params(axis='x', rotation=45)
            
            # 3. 事件类型分布
            ax3 = axes[0, 2]
            if self.aggregated_events.get("by_type"):
                types = list(self.aggregated_events["by_type"].keys())
                counts = list(self.aggregated_events["by_type"].values())
                ax3.pie(counts, labels=types, autopct='%1.1f%%')
                ax3.set_title('事件类型分布')
            
            # 4. 指标趋势
            ax4 = axes[1, 0]
            if self.metrics_history:
                # 取最近的指标数据
                recent_metrics = list(self.metrics_history)[-20:]
                timestamps = [m.timestamp for m in recent_metrics]
                values = [m.value for m in recent_metrics]
                ax4.plot(timestamps, values, marker='o')
                ax4.set_title('指标趋势')
                ax4.tick_params(axis='x', rotation=45)
            
            # 5. 告警状态
            ax5 = axes[1, 1]
            if self.current_alerts:
                alert_severities = [a.severity.value for a in self.current_alerts]
                severity_counts = {s: alert_severities.count(s) for s in set(alert_severities)}
                ax5.bar(severity_counts.keys(), severity_counts.values(), 
                       color=['blue', 'yellow', 'orange', 'red'])
                ax5.set_title('告警严重程度分布')
            else:
                ax5.text(0.5, 0.5, '无告警', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('告警状态')
            
            # 6. 模块状态
            ax6 = axes[1, 2]
            if self.module_status:
                module_names = list(self.module_status.keys())
                statuses = [self.module_status[m].get('status', 'unknown') for m in module_names]
                status_colors = {'online': 'green', 'offline': 'red', 'error': 'orange', 'unknown': 'gray'}
                colors = [status_colors.get(s, 'gray') for s in statuses]
                ax6.bar(module_names, [1]*len(module_names), color=colors)
                ax6.set_title('安全模块状态')
                ax6.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"可视化图表已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"安全状态可视化失败: {e}")
    
    def export_data(self, 
                   format_type: str = "json",
                   include_history: bool = True) -> Union[str, Dict[str, Any]]:
        """导出安全数据"""
        logger.info(f"开始导出安全数据 (格式: {format_type})")
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "security_assessment": self.security_assessment,
            "aggregated_metrics": self.aggregated_metrics,
            "aggregated_threats": self.aggregated_threats,
            "aggregated_events": self.aggregated_events,
            "module_status": self.module_status,
            "current_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp.isoformat(),
                    "source": m.source,
                    "category": m.category
                } for m in self.current_metrics
            ],
            "current_threats": [
                {
                    "id": t.id,
                    "name": t.name,
                    "type": t.type,
                    "severity": t.severity.value,
                    "timestamp": t.timestamp.isoformat(),
                    "description": t.description,
                    "confidence": t.confidence
                } for t in self.current_threats
            ],
            "current_events": [
                {
                    "id": e.id,
                    "type": e.type.value,
                    "severity": e.severity.value,
                    "timestamp": e.timestamp.isoformat(),
                    "description": e.description,
                    "status": e.status
                } for e in self.current_events
            ],
            "current_alerts": [
                {
                    "id": a.id,
                    "title": a.title,
                    "severity": a.severity.value,
                    "timestamp": a.timestamp.isoformat(),
                    "message": a.message,
                    "acknowledged": a.acknowledged,
                    "resolved": a.resolved
                } for a in self.current_alerts
            ]
        }
        
        if include_history:
            export_data["history_summary"] = {
                "metrics_count": len(self.metrics_history),
                "threats_count": len(self.threats_history),
                "events_count": len(self.events_history),
                "alerts_count": len(self.alerts_history)
            }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        elif format_type.lower() == "dict":
            return export_data
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前安全状态摘要"""
        return {
            "timestamp": datetime.now().isoformat(),
            "security_assessment": self.security_assessment,
            "module_count": len(self.security_modules),
            "active_modules": len([s for s in self.module_status.values() if s.get('status') == 'online']),
            "current_metrics_count": len(self.current_metrics),
            "current_threats_count": len(self.current_threats),
            "current_events_count": len(self.current_events),
            "current_alerts_count": len(self.current_alerts),
            "collection_running": self._running
        }
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        for alert in self.current_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"告警已确认: {alert_id}")
                return True
        logger.warning(f"未找到要确认的告警: {alert_id}")
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        for alert in self.current_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.acknowledged = True
                logger.info(f"告警已解决: {alert_id}")
                return True
        logger.warning(f"未找到要解决的告警: {alert_id}")
        return False


# 测试用例
async def test_security_state_aggregator():
    """测试安全状态聚合器"""
    print("=== N9安全状态聚合器测试 ===")
    
    # 创建聚合器实例
    aggregator = SecurityStateAggregator(
        collection_interval=10,
        max_history_size=100
    )
    
    # 注册告警回调函数
    def alert_handler(alert: SecurityAlert):
        print(f"🚨 告警: {alert.title} - {alert.message}")
    
    aggregator.register_alert_callback(alert_handler)
    
    # 注册模拟安全模块
    modules = [
        MockSecurityModule("防火墙模块"),
        MockSecurityModule("入侵检测模块"),
        MockSecurityModule("恶意软件检测模块"),
        MockSecurityModule("网络监控模块")
    ]
    
    for module in modules:
        aggregator.register_security_module(module)
    
    print(f"已注册 {len(modules)} 个安全模块")
    
    # 启动数据收集
    await aggregator.start_collection()
    
    try:
        # 等待数据收集
        print("等待数据收集...")
        await asyncio.sleep(15)
        
        # 获取当前状态
        status = aggregator.get_current_status()
        print(f"\n当前状态:")
        print(f"安全分数: {status['security_assessment'].get('overall_score', 0):.1f}")
        print(f"安全等级: {status['security_assessment'].get('level', '未知')}")
        print(f"活跃模块: {status['active_modules']}/{status['module_count']}")
        print(f"当前威胁: {status['current_threats_count']}")
        print(f"当前事件: {status['current_events_count']}")
        print(f"当前告警: {status['current_alerts_count']}")
        
        # 生成安全报告
        print("\n生成安全报告...")
        report = await aggregator.generate_security_report("summary", 1)
        print(f"报告标题: {report.title}")
        print(f"安全分数: {report.summary['security_score']:.1f}")
        print(f"建议数量: {len(report.recommendations)}")
        
        # 分析安全趋势
        print("\n分析安全趋势...")
        trends = await aggregator.analyze_security_trends(1)
        print(f"趋势分析期间: {trends['period']}")
        
        # 导出数据
        print("\n导出数据...")
        export_data = aggregator.export_data("dict")
        print(f"导出数据键: {list(export_data.keys())}")
        
        # 测试告警确认
        if aggregator.current_alerts:
            alert_id = aggregator.current_alerts[0].id
            print(f"\n测试告警确认: {alert_id}")
            await aggregator.acknowledge_alert(alert_id)
        
        print("\n✅ 测试完成")
        
    finally:
        # 停止数据收集
        await aggregator.stop_collection()
    
    return aggregator


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_security_state_aggregator())