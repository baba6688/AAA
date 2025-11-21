#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4访问控制器模块

该模块实现了一个完整的访问控制系统，包括：
- 访问控制列表(ACL)
- 网络访问控制
- 资源访问控制
- API访问控制
- 实时访问监控
- 访问模式分析
- 异常访问检测
- 访问控制策略
- 访问日志和审计


创建时间: 2025-11-06
版本: 1.0.0
"""

import logging
import time
import threading
import json
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import socket
import re


class AccessLevel(Enum):
    """访问级别枚举"""
    DENY = 0      # 拒绝访问
    READ = 1      # 只读访问
    WRITE = 2     # 读写访问
    EXECUTE = 3   # 执行权限
    ADMIN = 4     # 管理员权限


class AccessType(Enum):
    """访问类型枚举"""
    NETWORK = "network"
    RESOURCE = "resource"
    API = "api"
    FILE = "file"
    DATABASE = "database"
    SYSTEM = "system"


class RiskLevel(Enum):
    """风险级别枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AccessRule:
    """访问规则数据类"""
    rule_id: str
    name: str
    resource: str
    subject: str
    access_level: AccessLevel
    access_type: AccessType
    conditions: Dict[str, Any]
    priority: int
    enabled: bool = True
    created_at: datetime = None
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class AccessEvent:
    """访问事件数据类"""
    event_id: str
    timestamp: datetime
    subject: str
    resource: str
    action: str
    access_type: AccessType
    result: str  # "allow" or "deny"
    source_ip: str
    user_agent: str
    risk_score: float
    details: Dict[str, Any]


@dataclass
@dataclass
class NetworkAccessRule:
    """网络访问规则数据类"""
    rule_id: str
    allowed_ips: Set[str]
    denied_ips: Set[str]
    allowed_ports: Set[int]
    denied_ports: Set[int]
    protocols: Set[str]
    time_restrictions: Dict[str, Any] = None

    def __post_init__(self):
        if self.time_restrictions is None:
            self.time_restrictions = {}


class AccessController:
    """
    N4访问控制器
    
    提供全面的访问控制功能，包括：
    - 访问控制列表管理
    - 网络访问控制
    - 资源访问控制
    - API访问控制
    - 实时监控和分析
    - 异常检测和响应
    - 审计日志记录
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化访问控制器
        
        Args:
            config: 配置字典，包含各种参数设置
        """
        self.config = config or {}
        self._lock = threading.RLock()
        
        # 初始化组件
        self._init_logging()
        self._init_access_rules()
        self._init_network_rules()
        self._init_monitoring()
        self._init_audit_log()
        
        # 启动监控线程
        self._start_monitoring_threads()
        
        self.logger.info("N4访问控制器初始化完成")
    
    def _init_logging(self):
        """初始化日志系统"""
        log_level = self.config.get('log_level', logging.INFO)
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler('access_controller.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AccessController')
    
    def _init_access_rules(self):
        """初始化访问控制规则"""
        self.access_rules: Dict[str, AccessRule] = {}
        self.rule_priorities: Dict[int, List[str]] = defaultdict(list)
        
        # 默认规则
        self._create_default_rules()
    
    def _init_network_rules(self):
        """初始化网络访问规则"""
        self.network_rules: Dict[str, NetworkAccessRule] = {}
        self.ip_reputation_cache: Dict[str, float] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def _init_monitoring(self):
        """初始化监控组件"""
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.monitoring_interval = self.config.get('monitoring_interval', 60)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.7)
        
        self.access_patterns: Dict[str, List[AccessEvent]] = defaultdict(list)
        self.suspicious_activities: List[AccessEvent] = []
        self.blocked_ips: Set[str] = set()
        
        # 访问统计
        self.access_stats = {
            'total_requests': 0,
            'allowed_requests': 0,
            'denied_requests': 0,
            'blocked_ips': 0,
            'suspicious_activities': 0
        }
    
    def _init_audit_log(self):
        """初始化审计日志"""
        self.audit_enabled = self.config.get('audit_enabled', True)
        self.audit_retention_days = self.config.get('audit_retention_days', 30)
        self.audit_log: List[AccessEvent] = []
        
        # 审计事件处理器
        self.audit_handlers: List[Callable] = []
    
    def _create_default_rules(self):
        """创建默认访问规则"""
        # 管理员完全访问权限
        admin_rule = AccessRule(
            rule_id="admin_full_access",
            name="管理员完全访问权限",
            resource="*",
            subject="admin",
            access_level=AccessLevel.ADMIN,
            access_type=AccessType.SYSTEM,
            conditions={},
            priority=100
        )
        self.add_access_rule(admin_rule)
        
        # API访问限制
        api_rule = AccessRule(
            rule_id="api_rate_limit",
            name="API访问频率限制",
            resource="api/*",
            subject="*",
            access_level=AccessLevel.READ,
            access_type=AccessType.API,
            conditions={"rate_limit": 1000, "time_window": 3600},
            priority=50
        )
        self.add_access_rule(api_rule)
        
        # 文件访问控制
        file_rule = AccessRule(
            rule_id="file_access_control",
            name="文件访问控制",
            resource="files/*",
            subject="*",
            access_level=AccessLevel.READ,
            access_type=AccessType.FILE,
            conditions={"allowed_extensions": [".txt", ".pdf", ".doc"]},
            priority=30
        )
        self.add_access_rule(file_rule)
    
    def _start_monitoring_threads(self):
        """启动监控线程"""
        if self.monitoring_enabled:
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.analysis_thread.start()
    
    def add_access_rule(self, rule: AccessRule) -> bool:
        """
        添加访问控制规则
        
        Args:
            rule: 访问规则对象
            
        Returns:
            bool: 添加是否成功
        """
        with self._lock:
            try:
                self.access_rules[rule.rule_id] = rule
                self.rule_priorities[rule.priority].append(rule.rule_id)
                self.logger.info(f"添加访问规则: {rule.name}")
                return True
            except Exception as e:
                self.logger.error(f"添加访问规则失败: {e}")
                return False
    
    def remove_access_rule(self, rule_id: str) -> bool:
        """
        删除访问控制规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            bool: 删除是否成功
        """
        with self._lock:
            try:
                if rule_id in self.access_rules:
                    rule = self.access_rules[rule_id]
                    del self.access_rules[rule_id]
                    self.rule_priorities[rule.priority].remove(rule_id)
                    self.logger.info(f"删除访问规则: {rule.name}")
                    return True
                return False
            except Exception as e:
                self.logger.error(f"删除访问规则失败: {e}")
                return False
    
    def check_access(self, subject: str, resource: str, action: str, 
                    context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        检查访问权限
        
        Args:
            subject: 访问主体（用户/进程等）
            resource: 资源标识
            action: 操作类型
            context: 访问上下文信息
            
        Returns:
            Tuple[bool, str]: (是否允许访问, 拒绝原因)
        """
        context = context or {}
        
        with self._lock:
            try:
                # 记录访问统计
                self.access_stats['total_requests'] += 1
                
                # 获取访问类型
                access_type = self._determine_access_type(resource)
                
                # 按优先级检查规则
                for priority in sorted(self.rule_priorities.keys(), reverse=True):
                    for rule_id in self.rule_priorities[priority]:
                        rule = self.access_rules[rule_id]
                        
                        if not rule.enabled:
                            continue
                            
                        # 检查规则适用性
                        if self._rule_matches(rule, subject, resource, action, access_type):
                            # 检查规则条件
                            if self._check_rule_conditions(rule, context):
                                # 评估访问级别
                                allowed = self._evaluate_access_level(rule, action)
                                
                                if allowed:
                                    self.access_stats['allowed_requests'] += 1
                                    return True, ""
                                else:
                                    reason = f"访问级别不足: 需要{rule.access_level.name}, 尝试执行{action}"
                                    self.access_stats['denied_requests'] += 1
                                    return False, reason
                
                # 默认拒绝
                self.access_stats['denied_requests'] += 1
                return False, "没有匹配的访问规则"
                
            except Exception as e:
                self.logger.error(f"访问检查失败: {e}")
                return False, f"访问检查异常: {str(e)}"
    
    def _determine_access_type(self, resource: str) -> AccessType:
        """根据资源确定访问类型"""
        if resource.startswith("api/"):
            return AccessType.API
        elif resource.startswith("files/"):
            return AccessType.FILE
        elif resource.startswith("db/"):
            return AccessType.DATABASE
        elif resource.startswith("net/"):
            return AccessType.NETWORK
        else:
            return AccessType.RESOURCE
    
    def _rule_matches(self, rule: AccessRule, subject: str, resource: str, 
                     action: str, access_type: AccessType) -> bool:
        """检查规则是否匹配"""
        # 检查主体匹配
        if rule.subject != "*" and rule.subject != subject:
            return False
        
        # 检查资源匹配
        if not self._match_pattern(rule.resource, resource):
            return False
        
        # 检查访问类型匹配
        if rule.access_type != access_type:
            return False
        
        # 检查时间限制
        if rule.expires_at and datetime.now() > rule.expires_at:
            return False
        
        return True
    
    def _match_pattern(self, pattern: str, value: str) -> bool:
        """简单的模式匹配（支持*通配符）"""
        if pattern == "*":
            return True
        
        # 转换为正则表达式
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", value))
    
    def _check_rule_conditions(self, rule: AccessRule, context: Dict[str, Any]) -> bool:
        """检查规则条件"""
        for condition, expected_value in rule.conditions.items():
            if condition == "rate_limit":
                if not self._check_rate_limit(context.get("source_ip", ""), expected_value):
                    return False
            elif condition == "time_restriction":
                if not self._check_time_restriction(expected_value):
                    return False
            elif condition == "allowed_extensions":
                if not self._check_file_extensions(context.get("file_path", ""), expected_value):
                    return False
            # 可以添加更多条件检查
        
        return True
    
    def _evaluate_access_level(self, rule: AccessRule, action: str) -> bool:
        """评估访问级别"""
        action_level_map = {
            "read": AccessLevel.READ,
            "write": AccessLevel.WRITE,
            "execute": AccessLevel.EXECUTE,
            "delete": AccessLevel.ADMIN,
            "admin": AccessLevel.ADMIN
        }
        
        required_level = action_level_map.get(action, AccessLevel.READ)
        return rule.access_level.value >= required_level.value
    
    def _check_rate_limit(self, source_ip: str, limit: int) -> bool:
        """检查访问频率限制"""
        if not source_ip:
            return True
        
        current_time = time.time()
        window_start = current_time - 3600  # 1小时窗口
        
        # 获取该IP的请求历史
        if source_ip not in self.request_history:
            self.request_history[source_ip] = deque()
        
        # 清理过期请求
        requests = self.request_history[source_ip]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # 检查是否超过限制
        return len(requests) < limit
    
    def _check_time_restriction(self, time_restriction: Dict[str, Any]) -> bool:
        """检查时间限制"""
        current_time = datetime.now()
        current_hour = current_time.hour
        
        allowed_hours = time_restriction.get("allowed_hours", [])
        if allowed_hours and current_hour not in allowed_hours:
            return False
        
        return True
    
    def _check_file_extensions(self, file_path: str, allowed_extensions: List[str]) -> bool:
        """检查文件扩展名"""
        if not file_path:
            return True
        
        import os
        _, ext = os.path.splitext(file_path)
        return ext.lower() in [e.lower() for e in allowed_extensions]
    
    def check_network_access(self, source_ip: str, target_ip: str, port: int, 
                           protocol: str) -> Tuple[bool, str]:
        """
        检查网络访问权限
        
        Args:
            source_ip: 源IP地址
            target_ip: 目标IP地址
            port: 端口号
            protocol: 协议类型
            
        Returns:
            Tuple[bool, str]: (是否允许访问, 拒绝原因)
        """
        with self._lock:
            try:
                # 检查IP是否被阻止
                if source_ip in self.blocked_ips:
                    return False, "IP地址被阻止"
                
                # 检查网络规则
                for rule in self.network_rules.values():
                    if self._network_rule_matches(rule, source_ip, target_ip, port, protocol):
                        return True, ""
                
                # 默认允许（可根据安全策略调整）
                return True, ""
                
            except Exception as e:
                self.logger.error(f"网络访问检查失败: {e}")
                return False, f"网络访问检查异常: {str(e)}"
    
    def _network_rule_matches(self, rule: NetworkAccessRule, source_ip: str, 
                            target_ip: str, port: int, protocol: str) -> bool:
        """检查网络规则匹配"""
        # 检查IP规则
        if rule.denied_ips and self._ip_in_ranges(source_ip, rule.denied_ips):
            return False
        
        if rule.allowed_ips and not self._ip_in_ranges(source_ip, rule.allowed_ips):
            return False
        
        # 检查端口规则
        if rule.denied_ports and port in rule.denied_ports:
            return False
        
        if rule.allowed_ports and port not in rule.allowed_ports:
            return False
        
        # 检查协议规则
        if rule.protocols and protocol.lower() not in [p.lower() for p in rule.protocols]:
            return False
        
        return True
    
    def _ip_in_ranges(self, ip: str, ip_ranges: Set[str]) -> bool:
        """检查IP是否在指定范围内"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            for ip_range in ip_ranges:
                if ipaddress.ip_network(ip_range, strict=False).overlaps(
                    ipaddress.ip_network(f"{ip}/{ip_obj.max_prefixlen}", strict=False)
                ):
                    return True
        except ValueError:
            # 如果不是有效IP，尝试直接匹配
            return ip in ip_ranges
        return False
    
    def add_network_rule(self, rule: NetworkAccessRule) -> bool:
        """添加网络访问规则"""
        with self._lock:
            try:
                self.network_rules[rule.rule_id] = rule
                self.logger.info(f"添加网络规则: {rule.rule_id}")
                return True
            except Exception as e:
                self.logger.error(f"添加网络规则失败: {e}")
                return False
    
    def block_ip(self, ip: str, duration: int = 3600) -> bool:
        """
        阻止IP地址
        
        Args:
            ip: 要阻止的IP地址
            duration: 阻止时长（秒）
            
        Returns:
            bool: 操作是否成功
        """
        with self._lock:
            try:
                self.blocked_ips.add(ip)
                self.access_stats['blocked_ips'] += 1
                
                # 设置定时解除阻止
                def unblock():
                    time.sleep(duration)
                    with self._lock:
                        self.blocked_ips.discard(ip)
                
                threading.Thread(target=unblock, daemon=True).start()
                
                self.logger.warning(f"阻止IP: {ip}, 时长: {duration}秒")
                return True
            except Exception as e:
                self.logger.error(f"阻止IP失败: {e}")
                return False
    
    def log_access_event(self, event: AccessEvent):
        """记录访问事件"""
        with self._lock:
            try:
                # 添加到审计日志
                if self.audit_enabled:
                    self.audit_log.append(event)
                
                # 添加到访问模式分析
                self.access_patterns[event.subject].append(event)
                
                # 检查异常访问
                if event.risk_score > self.anomaly_threshold:
                    self.suspicious_activities.append(event)
                    self.access_stats['suspicious_activities'] += 1
                    self.logger.warning(f"检测到异常访问: {event.event_id}, 风险评分: {event.risk_score}")
                
                # 调用审计事件处理器
                for handler in self.audit_handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        self.logger.error(f"审计事件处理器执行失败: {e}")
                
            except Exception as e:
                self.logger.error(f"记录访问事件失败: {e}")
    
    def analyze_access_patterns(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        分析访问模式
        
        Args:
            time_window: 分析时间窗口（秒）
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        with self._lock:
            try:
                current_time = time.time()
                window_start = current_time - time_window
                
                analysis_result = {
                    "time_window": time_window,
                    "total_events": 0,
                    "unique_subjects": set(),
                    "unique_resources": set(),
                    "access_frequency": defaultdict(int),
                    "risk_distribution": defaultdict(int),
                    "top_resources": [],
                    "top_subjects": [],
                    "anomaly_indicators": []
                }
                
                # 分析所有访问模式
                for subject, events in self.access_patterns.items():
                    subject_events = [e for e in events if e.timestamp.timestamp() > window_start]
                    
                    if not subject_events:
                        continue
                    
                    analysis_result["total_events"] += len(subject_events)
                    analysis_result["unique_subjects"].add(subject)
                    
                    # 统计访问频率
                    for event in subject_events:
                        analysis_result["access_frequency"][f"{subject}:{event.resource}"] += 1
                        analysis_result["unique_resources"].add(event.resource)
                        analysis_result["risk_distribution"][event.risk_score] += 1
                    
                    # 检测异常模式
                    if len(subject_events) > 100:  # 高频访问
                        analysis_result["anomaly_indicators"].append({
                            "type": "high_frequency",
                            "subject": subject,
                            "count": len(subject_events),
                            "severity": "medium"
                        })
                
                # 转换为可序列化格式
                analysis_result["unique_subjects"] = list(analysis_result["unique_subjects"])
                analysis_result["unique_resources"] = list(analysis_result["unique_resources"])
                
                # 获取Top资源
                sorted_resources = sorted(
                    analysis_result["access_frequency"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                analysis_result["top_resources"] = sorted_resources[:10]
                
                # 获取Top访问主体
                subject_counts = defaultdict(int)
                for key, count in analysis_result["access_frequency"].items():
                    # key格式为 "subject:resource"
                    subject = key.split(':')[0] if ':' in key else key
                    subject_counts[subject] += count
                
                sorted_subjects = sorted(
                    subject_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                analysis_result["top_subjects"] = sorted_subjects[:10]
                
                return analysis_result
                
            except Exception as e:
                self.logger.error(f"访问模式分析失败: {e}")
                return {}
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        检测异常访问
        
        Returns:
            List[Dict[str, Any]]: 异常访问列表
        """
        with self._lock:
            try:
                anomalies = []
                current_time = time.time()
                
                # 检测可疑活动
                for event in self.suspicious_activities:
                    if current_time - event.timestamp.timestamp() < 3600:  # 最近1小时
                        anomalies.append({
                            "type": "suspicious_activity",
                            "event_id": event.event_id,
                            "subject": event.subject,
                            "resource": event.resource,
                            "risk_score": event.risk_score,
                            "timestamp": event.timestamp.isoformat(),
                            "severity": "high" if event.risk_score > 0.8 else "medium"
                        })
                
                # 检测暴力破解模式
                failed_attempts = defaultdict(int)
                for subject, events in self.access_patterns.items():
                    recent_events = [
                        e for e in events 
                        if current_time - e.timestamp.timestamp() < 300  # 最近5分钟
                    ]
                    
                    failed_count = sum(1 for e in recent_events if e.result == "deny")
                    if failed_count > 10:  # 5分钟内超过10次失败
                        anomalies.append({
                            "type": "brute_force",
                            "subject": subject,
                            "failed_attempts": failed_count,
                            "time_window": 300,
                            "severity": "critical"
                        })
                
                # 检测异常时间段访问
                for subject, events in self.access_patterns.items():
                    night_events = [
                        e for e in events
                        if 0 <= e.timestamp.hour <= 6  # 凌晨0-6点
                    ]
                    
                    if len(night_events) > 50:  # 夜间访问过多
                        anomalies.append({
                            "type": "unusual_time_access",
                            "subject": subject,
                            "night_access_count": len(night_events),
                            "severity": "medium"
                        })
                
                return anomalies
                
            except Exception as e:
                self.logger.error(f"异常检测失败: {e}")
                return []
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_enabled:
            try:
                # 清理过期数据
                self._cleanup_expired_data()
                
                # 更新IP信誉度
                self._update_ip_reputation()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(60)  # 异常时等待更长时间
    
    def _analysis_loop(self):
        """分析循环"""
        while self.monitoring_enabled:
            try:
                # 执行访问模式分析
                patterns = self.analyze_access_patterns()
                
                # 检测异常
                anomalies = self.detect_anomalies()
                
                # 自动响应异常
                self._auto_respond_anomalies(anomalies)
                
                time.sleep(self.monitoring_interval * 2)  # 分析频率较低
                
            except Exception as e:
                self.logger.error(f"分析循环异常: {e}")
                time.sleep(120)
    
    def _cleanup_expired_data(self):
        """清理过期数据"""
        current_time = time.time()
        cutoff_time = current_time - (self.audit_retention_days * 24 * 3600)
        
        # 清理审计日志
        self.audit_log = [
            event for event in self.audit_log
            if event.timestamp.timestamp() > cutoff_time
        ]
        
        # 清理访问模式数据
        for subject in list(self.access_patterns.keys()):
            self.access_patterns[subject] = [
                event for event in self.access_patterns[subject]
                if event.timestamp.timestamp() > cutoff_time
            ]
            
            # 如果没有数据了，删除该主体
            if not self.access_patterns[subject]:
                del self.access_patterns[subject]
    
    def _update_ip_reputation(self):
        """更新IP信誉度"""
        for ip, requests in self.request_history.items():
            if not requests:
                continue
            
            # 计算失败率
            recent_time = time.time() - 3600
            recent_requests = [t for t in requests if t > recent_time]
            
            if len(recent_requests) < 5:
                continue
            
            # 简化信誉度计算（实际应用中需要更复杂的算法）
            failure_rate = 0.1  # 假设失败率为10%
            self.ip_reputation_cache[ip] = max(0, 1 - failure_rate)
    
    def _auto_respond_anomalies(self, anomalies: List[Dict[str, Any]]):
        """自动响应异常"""
        for anomaly in anomalies:
            try:
                if anomaly["severity"] == "critical":
                    if anomaly["type"] == "brute_force":
                        # 阻止攻击源IP
                        self.block_ip(anomaly["subject"], duration=7200)  # 阻止2小时
                        self.logger.critical(f"检测到暴力破解，自动阻止IP: {anomaly['subject']}")
                
                elif anomaly["severity"] == "high":
                    if anomaly["type"] == "suspicious_activity":
                        # 发送告警
                        self.logger.warning(f"高风险活动告警: {anomaly}")
                        
            except Exception as e:
                self.logger.error(f"自动响应异常失败: {e}")
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """获取访问统计信息"""
        with self._lock:
            try:
                stats = self.access_stats.copy()
                
                # 计算成功率
                if stats['total_requests'] > 0:
                    stats['success_rate'] = stats['allowed_requests'] / stats['total_requests']
                else:
                    stats['success_rate'] = 0
                
                # 添加实时统计
                stats['blocked_ips_count'] = len(self.blocked_ips)
                stats['active_rules'] = len([r for r in self.access_rules.values() if r.enabled])
                stats['suspicious_activities_count'] = len(self.suspicious_activities)
                
                return stats
                
            except Exception as e:
                self.logger.error(f"获取统计信息失败: {e}")
                return {}
    
    def export_audit_log(self, file_path: str, format: str = "json") -> bool:
        """
        导出审计日志
        
        Args:
            file_path: 导出文件路径
            format: 导出格式（json或csv）
            
        Returns:
            bool: 导出是否成功
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == "json":
                    events_data = [asdict(event) for event in self.audit_log]
                    json.dump(events_data, f, ensure_ascii=False, indent=2, default=str)
                elif format.lower() == "csv":
                    import csv
                    if self.audit_log:
                        writer = csv.DictWriter(f, fieldnames=asdict(self.audit_log[0]).keys())
                        writer.writeheader()
                        for event in self.audit_log:
                            event_dict = asdict(event)
                            event_dict['timestamp'] = event.timestamp.isoformat()
                            writer.writerow(event_dict)
                else:
                    raise ValueError(f"不支持的格式: {format}")
            
            self.logger.info(f"审计日志已导出到: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出审计日志失败: {e}")
            return False
    
    def add_audit_handler(self, handler: Callable[[AccessEvent], None]):
        """添加审计事件处理器"""
        self.audit_handlers.append(handler)
    
    def shutdown(self):
        """关闭访问控制器"""
        self.logger.info("正在关闭N4访问控制器...")
        
        # 停止监控
        self.monitoring_enabled = False
        
        # 等待线程结束
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        if hasattr(self, 'analysis_thread') and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5)
        
        # 导出最终审计日志
        if self.audit_log:
            self.export_audit_log(f"audit_log_final_{int(time.time())}.json")
        
        self.logger.info("N4访问控制器已关闭")


def create_sample_access_controller() -> AccessController:
    """创建示例访问控制器"""
    config = {
        'log_level': logging.INFO,
        'monitoring_enabled': True,
        'monitoring_interval': 30,
        'anomaly_threshold': 0.6,
        'audit_enabled': True,
        'audit_retention_days': 7
    }
    
    controller = AccessController(config)
    
    # 添加示例网络规则
    network_rule = NetworkAccessRule(
        rule_id="internal_network",
        allowed_ips={"192.168.1.0/24", "10.0.0.0/8"},
        denied_ips=set(),
        allowed_ports={80, 443, 22, 8080},
        denied_ports=set(),
        protocols={"tcp", "udp"}
    )
    controller.add_network_rule(network_rule)
    
    return controller


def run_comprehensive_test():
    """运行综合测试"""
    print("=== N4访问控制器综合测试 ===\n")
    
    # 创建访问控制器
    controller = create_sample_access_controller()
    
    try:
        # 测试1: 基本访问控制
        print("测试1: 基本访问控制")
        allowed, reason = controller.check_access("admin", "api/users", "read")
        print(f"管理员访问API: {'允许' if allowed else '拒绝'} - {reason}")
        
        allowed, reason = controller.check_access("user1", "api/admin", "delete")
        print(f"普通用户访问管理API: {'允许' if allowed else '拒绝'} - {reason}")
        
        # 测试2: 网络访问控制
        print("\n测试2: 网络访问控制")
        allowed, reason = controller.check_network_access("192.168.1.100", "192.168.1.1", 80, "tcp")
        print(f"内网访问: {'允许' if allowed else '拒绝'} - {reason}")
        
        allowed, reason = controller.check_network_access("203.0.113.1", "192.168.1.1", 22, "tcp")
        print(f"外网SSH访问: {'允许' if allowed else '拒绝'} - {reason}")
        
        # 测试3: 访问事件记录
        print("\n测试3: 访问事件记录")
        event = AccessEvent(
            event_id="test_001",
            timestamp=datetime.now(),
            subject="test_user",
            resource="api/test",
            action="read",
            access_type=AccessType.API,
            result="allow",
            source_ip="192.168.1.100",
            user_agent="TestAgent/1.0",
            risk_score=0.3,
            details={"test": True}
        )
        controller.log_access_event(event)
        print("访问事件已记录")
        
        # 测试4: 访问模式分析
        print("\n测试4: 访问模式分析")
        # 模拟多个访问事件
        for i in range(10):
            event = AccessEvent(
                event_id=f"pattern_test_{i}",
                timestamp=datetime.now(),
                subject="pattern_user",
                resource=f"api/resource_{i % 3}",
                action="read",
                access_type=AccessType.API,
                result="allow",
                source_ip="192.168.1.100",
                user_agent="PatternTest/1.0",
                risk_score=0.2,
                details={}
            )
            controller.log_access_event(event)
        
        patterns = controller.analyze_access_patterns()
        print(f"访问模式分析完成，发现 {patterns.get('total_events', 0)} 个事件")
        print(f"独特访问主体: {len(patterns.get('unique_subjects', []))}")
        print(f"独特资源: {len(patterns.get('unique_resources', []))}")
        
        # 测试5: 异常检测
        print("\n测试5: 异常检测")
        # 模拟异常访问
        for i in range(15):
            event = AccessEvent(
                event_id=f"anomaly_test_{i}",
                timestamp=datetime.now(),
                subject="anomaly_user",
                resource="api/sensitive",
                action="admin",
                access_type=AccessType.API,
                result="deny",
                source_ip="203.0.113.100",
                user_agent="AnomalyBot/1.0",
                risk_score=0.9,
                details={"suspicious": True}
            )
            controller.log_access_event(event)
        
        anomalies = controller.detect_anomalies()
        print(f"检测到 {len(anomalies)} 个异常")
        for anomaly in anomalies:
            print(f"  - {anomaly['type']}: {anomaly.get('subject', 'N/A')} (严重程度: {anomaly['severity']})")
        
        # 测试6: IP阻止
        print("\n测试6: IP阻止功能")
        controller.block_ip("203.0.113.100", duration=10)
        print("已阻止可疑IP，10秒后自动解除")
        
        # 测试7: 统计信息
        print("\n测试7: 统计信息")
        stats = controller.get_access_statistics()
        print("访问统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 测试8: 审计日志导出
        print("\n测试8: 审计日志导出")
        success = controller.export_audit_log("test_audit_log.json")
        print(f"审计日志导出: {'成功' if success else '失败'}")
        
        print("\n=== 所有测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        controller.logger.error(f"测试异常: {e}")
    
    finally:
        # 关闭控制器
        controller.shutdown()


if __name__ == "__main__":
    run_comprehensive_test()