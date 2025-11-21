"""
N6威胁检测器 - 高级威胁检测和响应系统

该模块实现了一个全面的威胁检测系统，包括：
- 恶意软件检测
- 网络攻击检测
- 内部威胁检测
- 异常行为检测
- 威胁情报集成
- 实时威胁监控
- 威胁响应机制
- 威胁分类和评分
- 威胁报告生成

Author: N6 Security Team
Date: 2025-11-06
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from ipaddress import ip_address, ip_network
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from urllib.parse import urlparse
import socket
import ssl
import os


class ThreatLevel(Enum):
    """威胁等级枚举"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "严重"


class ThreatType(Enum):
    """威胁类型枚举"""
    MALWARE = "恶意软件"
    NETWORK_ATTACK = "网络攻击"
    INSIDER_THREAT = "内部威胁"
    ANOMALY = "异常行为"
    DATA_BREACH = "数据泄露"
    PHISHING = "钓鱼攻击"
    RANSOMWARE = "勒索软件"
    APT = "高级持续威胁"


class ResponseAction(Enum):
    """响应动作枚举"""
    ALERT = "告警"
    BLOCK = "阻断"
    QUARANTINE = "隔离"
    LOG = "记录"
    INVESTIGATE = "调查"


@dataclass
class ThreatIndicator:
    """威胁指标数据类"""
    indicator_type: str
    value: str
    confidence: float
    source: str
    first_seen: datetime
    last_seen: datetime
    tags: List[str]


@dataclass
class NetworkEvent:
    """网络事件数据类"""
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    bytes_sent: int
    bytes_received: int
    duration: float
    status: str


@dataclass
class ThreatEvent:
    """威胁事件数据类"""
    event_id: str
    timestamp: datetime
    threat_type: ThreatType
    threat_level: ThreatLevel
    source_ip: Optional[str]
    target_ip: Optional[str]
    description: str
    indicators: List[ThreatIndicator]
    confidence_score: float
    response_actions: List[ResponseAction]
    status: str
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None


@dataclass
class ThreatReport:
    """威胁报告数据类"""
    report_id: str
    generated_at: datetime
    time_range_start: datetime
    time_range_end: datetime
    total_events: int
    threats_by_type: Dict[ThreatType, int]
    threats_by_level: Dict[ThreatLevel, int]
    top_sources: List[Tuple[str, int]]
    recommendations: List[str]
    executive_summary: str


class MalwareDetector:
    """恶意软件检测器"""
    
    def __init__(self):
        self.known_signatures: Dict[str, str] = {}
        self.behavioral_patterns: List[Dict] = []
        self._load_known_signatures()
    
    def _load_known_signatures(self) -> None:
        """加载已知恶意软件签名"""
        # 模拟加载已知签名
        self.known_signatures = {
            "d41d8cd98f00b204e9800998ecf8427e": "known_malware_1",
            "098f6bcd4621d373cade4e832627b4f6": "known_malware_2"
        }
    
    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logging.error(f"计算文件哈希失败: {e}")
            return ""
    
    def detect_malware_by_signature(self, file_hash: str) -> Tuple[bool, float, str]:
        """基于签名检测恶意软件"""
        if file_hash in self.known_signatures:
            return True, 1.0, self.known_signatures[file_hash]
        return False, 0.0, ""
    
    def detect_malware_by_behavior(self, process_info: Dict) -> Tuple[bool, float, List[str]]:
        """基于行为检测恶意软件"""
        suspicious_behaviors = []
        risk_score = 0.0
        
        # 检查可疑进程行为
        if process_info.get("parent_process", "") in ["cmd.exe", "powershell.exe", "wscript.exe"]:
            suspicious_behaviors.append("可疑父进程")
            risk_score += 0.3
        
        if process_info.get("network_connections", 0) > 10:
            suspicious_behaviors.append("异常网络连接")
            risk_score += 0.4
        
        if process_info.get("file_modifications", 0) > 5:
            suspicious_behaviors.append("频繁文件修改")
            risk_score += 0.3
        
        return risk_score > 0.5, risk_score, suspicious_behaviors


class NetworkAttackDetector:
    """网络攻击检测器"""
    
    def __init__(self):
        self.attack_patterns: Dict[str, re.Pattern] = {}
        self.suspicious_ips: Set[str] = set()
        self._load_attack_patterns()
    
    def _load_attack_patterns(self) -> None:
        """加载攻击模式"""
        self.attack_patterns = {
            "sql_injection": re.compile(r"(union|select|insert|delete|update|drop|create|alter)\s+", re.IGNORECASE),
            "xss": re.compile(r"<script|javascript:|onload=|onerror=", re.IGNORECASE),
            "path_traversal": re.compile(r"\.\./", re.IGNORECASE),
            "command_injection": re.compile(r"[;&|`$()]", re.IGNORECASE)
        }
    
    def detect_ddos_attack(self, network_events: List[NetworkEvent]) -> Tuple[bool, float, str]:
        """检测DDoS攻击"""
        if not network_events:
            return False, 0.0, ""
        
        # 统计目标IP的连接数
        target_connections = defaultdict(int)
        total_requests = 0
        
        for event in network_events:
            target_connections[event.dst_ip] += 1
            total_requests += 1
        
        # 检测异常高的连接数
        max_connections = max(target_connections.values())
        if max_connections > total_requests * 0.8:  # 80%的请求都指向同一个IP
            return True, 0.9, f"DDoS攻击检测：目标IP {max(target_connections, key=target_connections.get)} 接收了 {max_connections} 个连接"
        
        return False, 0.0, ""
    
    def detect_port_scan(self, network_events: List[NetworkEvent]) -> Tuple[bool, float, str]:
        """检测端口扫描"""
        src_connections = defaultdict(set)
        
        for event in network_events:
            src_connections[event.src_ip].add(event.dst_port)
        
        for src_ip, ports in src_connections.items():
            if len(ports) > 10:  # 扫描超过10个端口
                return True, 0.8, f"端口扫描检测：源IP {src_ip} 扫描了 {len(ports)} 个端口"
        
        return False, 0.0, ""
    
    def detect_injection_attack(self, http_request: str) -> Tuple[bool, float, str]:
        """检测注入攻击"""
        for attack_type, pattern in self.attack_patterns.items():
            if pattern.search(http_request):
                return True, 0.7, f"检测到{attack_type}攻击"
        
        return False, 0.0, ""


class InsiderThreatDetector:
    """内部威胁检测器"""
    
    def __init__(self):
        self.user_behavior_profiles: Dict[str, Dict] = {}
        self.privileged_users: Set[str] = set()
    
    def create_user_profile(self, user_id: str, access_patterns: List[Dict]) -> None:
        """创建用户行为画像"""
        if not access_patterns:
            return
        
        # 分析访问模式
        access_times = [pattern.get("access_time") for pattern in access_patterns]
        accessed_resources = [pattern.get("resource") for pattern in access_patterns]
        
        profile = {
            "typical_access_hours": self._analyze_access_hours(access_times),
            "frequent_resources": list(set(accessed_resources)),
            "access_frequency": len(access_patterns),
            "last_updated": datetime.now()
        }
        
        self.user_behavior_profiles[user_id] = profile
    
    def _analyze_access_hours(self, access_times: List[datetime]) -> List[int]:
        """分析访问时间模式"""
        if not access_times:
            return []
        
        hour_counts = defaultdict(int)
        for access_time in access_times:
            if access_time:
                hour_counts[access_time.hour] += 1
        
        # 返回最频繁的访问时间
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, count in sorted_hours[:3]]
    
    def detect_anomalous_access(self, user_id: str, current_access: Dict) -> Tuple[bool, float, str]:
        """检测异常访问"""
        if user_id not in self.user_behavior_profiles:
            return False, 0.0, "用户画像不存在"
        
        profile = self.user_behavior_profiles[user_id]
        risk_score = 0.0
        anomalies = []
        
        # 检查访问时间异常
        current_hour = current_access.get("access_time", datetime.now()).hour
        typical_hours = profile.get("typical_access_hours", [])
        
        if typical_hours and current_hour not in typical_hours:
            risk_score += 0.4
            anomalies.append("异常访问时间")
        
        # 检查资源访问异常
        current_resource = current_access.get("resource", "")
        if current_resource not in profile.get("frequent_resources", []):
            risk_score += 0.3
            anomalies.append("访问非典型资源")
        
        # 检查访问频率异常
        if current_access.get("access_frequency", 0) > profile.get("access_frequency", 0) * 2:
            risk_score += 0.3
            anomalies.append("异常访问频率")
        
        if risk_score > 0.5:
            return True, risk_score, f"内部威胁检测：{', '.join(anomalies)}"
        
        return False, 0.0, ""


class AnomalyDetector:
    """异常行为检测器"""
    
    def __init__(self):
        self.baseline_metrics: Dict[str, float] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
    
    def establish_baseline(self, metrics_data: Dict[str, List[float]]) -> None:
        """建立基线指标"""
        for metric_name, values in metrics_data.items():
            if values:
                # 计算平均值作为基线
                baseline = sum(values) / len(values)
                self.baseline_metrics[metric_name] = baseline
                
                # 设置异常阈值（基线的2倍标准差）
                if len(values) > 1:
                    variance = sum((x - baseline) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5
                    self.anomaly_thresholds[metric_name] = baseline + (2 * std_dev)
    
    def detect_anomaly(self, current_metrics: Dict[str, float]) -> Tuple[bool, float, List[str]]:
        """检测异常"""
        anomalies = []
        max_risk_score = 0.0
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                threshold = self.anomaly_thresholds.get(metric_name, baseline * 2)
                
                if current_value > threshold:
                    risk_score = min((current_value - baseline) / threshold, 1.0)
                    max_risk_score = max(max_risk_score, risk_score)
                    anomalies.append(f"{metric_name}异常：当前值 {current_value:.2f} 超过基线 {baseline:.2f}")
        
        return max_risk_score > 0.3, max_risk_score, anomalies


class ThreatIntelligenceIntegrator:
    """威胁情报集成器"""
    
    def __init__(self):
        self.threat_feeds: List[Dict] = []
        self.ioc_database: Dict[str, ThreatIndicator] = {}
    
    def add_threat_feed(self, feed_url: str, feed_type: str) -> None:
        """添加威胁情报源"""
        feed_info = {
            "url": feed_url,
            "type": feed_type,
            "last_updated": datetime.now(),
            "status": "active"
        }
        self.threat_feeds.append(feed_info)
    
    def lookup_ioc(self, indicator: str) -> Optional[ThreatIndicator]:
        """查询威胁指标"""
        return self.ioc_database.get(indicator.lower())
    
    def add_ioc(self, indicator: ThreatIndicator) -> None:
        """添加威胁指标"""
        self.ioc_database[indicator.value.lower()] = indicator
    
    def check_reputation(self, ip_address: str) -> Tuple[float, str]:
        """检查IP信誉"""
        # 模拟IP信誉检查
        if ip_address in ["192.168.1.100", "10.0.0.50"]:  # 模拟恶意IP
            return 0.9, "已知恶意IP"
        elif ip_address.startswith("192.168.") or ip_address.startswith("10."):
            return 0.1, "内网IP"
        else:
            return 0.0, "未知"


class RealTimeMonitor:
    """实时威胁监控器"""
    
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.event_queue = deque(maxlen=1000)
        self.callbacks: List[Callable] = []
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logging.info("实时威胁监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logging.info("实时威胁监控已停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                # 模拟监控过程
                time.sleep(1)
                # 这里应该包含实际的监控逻辑
            except Exception as e:
                logging.error(f"监控循环错误: {e}")
    
    def add_event(self, event: Dict) -> None:
        """添加事件到队列"""
        self.event_queue.append(event)
        
        # 触发回调
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logging.error(f"回调执行错误: {e}")
    
    def register_callback(self, callback: Callable) -> None:
        """注册事件回调"""
        self.callbacks.append(callback)


class ThreatResponseEngine:
    """威胁响应引擎"""
    
    def __init__(self):
        self.response_rules: Dict[ThreatLevel, List[ResponseAction]] = {
            ThreatLevel.LOW: [ResponseAction.LOG],
            ThreatLevel.MEDIUM: [ResponseAction.LOG, ResponseAction.ALERT],
            ThreatLevel.HIGH: [ResponseAction.ALERT, ResponseAction.INVESTIGATE],
            ThreatLevel.CRITICAL: [ResponseAction.ALERT, ResponseAction.BLOCK, ResponseAction.INVESTIGATE]
        }
    
    def determine_response(self, threat_event: ThreatEvent) -> List[ResponseAction]:
        """确定响应动作"""
        return self.response_rules.get(threat_event.threat_level, [ResponseAction.LOG])
    
    def execute_response(self, threat_event: ThreatEvent, response_actions: List[ResponseAction]) -> bool:
        """执行响应动作"""
        success = True
        
        for action in response_actions:
            try:
                if action == ResponseAction.ALERT:
                    self._send_alert(threat_event)
                elif action == ResponseAction.BLOCK:
                    success &= self._block_source(threat_event)
                elif action == ResponseAction.QUARANTINE:
                    success &= self._quarantine_affected(threat_event)
                elif action == ResponseAction.LOG:
                    self._log_event(threat_event)
                elif action == ResponseAction.INVESTIGATE:
                    success &= self._initiate_investigation(threat_event)
            except Exception as e:
                logging.error(f"执行响应动作 {action} 失败: {e}")
                success = False
        
        return success
    
    def _send_alert(self, threat_event: ThreatEvent) -> None:
        """发送告警"""
        alert_message = f"威胁告警：{threat_event.threat_type.value} - {threat_event.description}"
        logging.warning(alert_message)
        # 这里可以集成邮件、短信、Slack等通知方式
    
    def _block_source(self, threat_event: ThreatEvent) -> bool:
        """阻断威胁源"""
        if threat_event.source_ip:
            logging.info(f"阻断源IP: {threat_event.source_ip}")
            # 这里应该实现实际的防火墙规则
            return True
        return False
    
    def _quarantine_affected(self, threat_event: ThreatEvent) -> bool:
        """隔离受影响系统"""
        logging.info(f"隔离受影响系统: {threat_event.target_ip}")
        # 这里应该实现实际的隔离逻辑
        return True
    
    def _log_event(self, threat_event: ThreatEvent) -> None:
        """记录事件"""
        logging.info(f"记录威胁事件: {threat_event.event_id}")
    
    def _initiate_investigation(self, threat_event: ThreatEvent) -> bool:
        """启动调查"""
        logging.info(f"启动威胁调查: {threat_event.event_id}")
        # 这里应该创建工单或启动调查流程
        return True


class ThreatClassifier:
    """威胁分类器"""
    
    def __init__(self):
        self.classification_rules: Dict[ThreatType, List[str]] = {
            ThreatType.MALWARE: ["malware", "virus", "trojan", "worm"],
            ThreatType.NETWORK_ATTACK: ["ddos", "port_scan", "injection", "xss"],
            ThreatType.INSIDER_THREAT: ["unauthorized_access", "data_exfiltration", "privilege_abuse"],
            ThreatType.PHISHING: ["phishing", "social_engineering", "fake_login"],
            ThreatType.RANSOMWARE: ["ransomware", "encryption", "decryption"]
        }
    
    def classify_threat(self, threat_data: Dict) -> Tuple[ThreatType, float]:
        """分类威胁"""
        description = threat_data.get("description", "").lower()
        indicators = threat_data.get("indicators", [])
        
        max_score = 0.0
        classified_type = ThreatType.ANOMALY
        
        for threat_type, keywords in self.classification_rules.items():
            score = 0.0
            
            # 基于描述匹配
            for keyword in keywords:
                if keyword in description:
                    score += 0.3
            
            # 基于指标匹配
            for indicator in indicators:
                if any(keyword in indicator.lower() for keyword in keywords):
                    score += 0.4
            
            if score > max_score:
                max_score = score
                classified_type = threat_type
        
        return classified_type, min(max_score, 1.0)


class ThreatScorer:
    """威胁评分器"""
    
    def __init__(self):
        self.scoring_weights = {
            "confidence": 0.3,
            "threat_level": 0.25,
            "impact": 0.2,
            "frequency": 0.15,
            "context": 0.1
        }
    
    def calculate_threat_score(self, threat_event: ThreatEvent, context: Dict) -> float:
        """计算威胁评分"""
        score = 0.0
        
        # 置信度评分
        score += threat_event.confidence_score * self.scoring_weights["confidence"]
        
        # 威胁等级评分
        level_scores = {
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.CRITICAL: 1.0
        }
        score += level_scores.get(threat_event.threat_level, 0) * self.scoring_weights["threat_level"]
        
        # 影响评分
        impact_score = context.get("impact_score", 0.5)
        score += impact_score * self.scoring_weights["impact"]
        
        # 频率评分
        frequency_score = context.get("frequency_score", 0.5)
        score += frequency_score * self.scoring_weights["frequency"]
        
        # 上下文评分
        context_score = context.get("context_score", 0.5)
        score += context_score * self.scoring_weights["context"]
        
        return min(score, 1.0)


class ThreatReporter:
    """威胁报告生成器"""
    
    def __init__(self):
        self.report_templates = {
            "executive": self._generate_executive_summary,
            "technical": self._generate_technical_report,
            "incident": self._generate_incident_report
        }
    
    def generate_report(self, threat_events: List[ThreatEvent], report_type: str = "executive") -> ThreatReport:
        """生成威胁报告"""
        if not threat_events:
            return self._generate_empty_report()
        
        # 统计威胁数据
        threats_by_type = defaultdict(int)
        threats_by_level = defaultdict(int)
        source_counts = defaultdict(int)
        
        for event in threat_events:
            threats_by_type[event.threat_type] += 1
            threats_by_level[event.threat_level] += 1
            if event.source_ip:
                source_counts[event.source_ip] += 1
        
        # 生成时间范围
        start_time = min(event.timestamp for event in threat_events)
        end_time = max(event.timestamp for event in threat_events)
        
        # 生成建议
        recommendations = self._generate_recommendations(threat_events)
        
        # 生成执行摘要
        executive_summary = self.report_templates.get(report_type, self._generate_executive_summary)(threat_events)
        
        # 获取主要威胁源
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return ThreatReport(
            report_id=f"THR-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            generated_at=datetime.now(),
            time_range_start=start_time,
            time_range_end=end_time,
            total_events=len(threat_events),
            threats_by_type={k.value: v for k, v in threats_by_type.items()},
            threats_by_level={k.value: v for k, v in threats_by_level.items()},
            top_sources=top_sources,
            recommendations=recommendations,
            executive_summary=executive_summary
        )
    
    def _generate_empty_report(self) -> ThreatReport:
        """生成空报告"""
        now = datetime.now()
        return ThreatReport(
            report_id=f"THR-{now.strftime('%Y%m%d-%H%M%S')}",
            generated_at=now,
            time_range_start=now,
            time_range_end=now,
            total_events=0,
            threats_by_type={},
            threats_by_level={},
            top_sources=[],
            recommendations=["系统运行正常，未检测到威胁事件"],
            executive_summary="在指定时间范围内未检测到威胁事件。"
        )
    
    def _generate_recommendations(self, threat_events: List[ThreatEvent]) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        # 基于威胁类型生成建议
        threat_types = [event.threat_type for event in threat_events]
        
        if ThreatType.MALWARE in threat_types:
            recommendations.append("加强端点防护，更新恶意软件定义库")
        
        if ThreatType.NETWORK_ATTACK in threat_types:
            recommendations.append("强化网络安全边界，部署入侵检测系统")
        
        if ThreatType.INSIDER_THREAT in threat_types:
            recommendations.append("加强内部安全管控，实施最小权限原则")
        
        if ThreatType.ANOMALY in threat_types:
            recommendations.append("优化异常检测算法，调整检测阈值")
        
        if not recommendations:
            recommendations.append("继续保持当前安全态势，定期进行安全评估")
        
        return recommendations
    
    def _generate_executive_summary(self, threat_events: List[ThreatEvent]) -> str:
        """生成执行摘要"""
        if not threat_events:
            return "系统运行正常，未检测到威胁事件。"
        
        total_events = len(threat_events)
        critical_events = sum(1 for event in threat_events if event.threat_level == ThreatLevel.CRITICAL)
        
        summary = f"在监控期间共检测到 {total_events} 个威胁事件"
        if critical_events > 0:
            summary += f"，其中 {critical_events} 个为严重级别威胁"
        summary += "。建议加强安全防护措施。"
        
        return summary
    
    def _generate_technical_report(self, threat_events: List[ThreatEvent]) -> str:
        """生成技术报告"""
        if not threat_events:
            return "技术分析：无威胁事件报告。"
        
        analysis_details = []
        for event in threat_events[:5]:  # 显示前5个事件
            analysis_details.append(
                f"- {event.timestamp}: {event.threat_type.value} ({event.threat_level.value}) - {event.description}"
            )
        
        return "技术分析详情：\n" + "\n".join(analysis_details)
    
    def _generate_incident_report(self, threat_events: List[ThreatEvent]) -> str:
        """生成事件报告"""
        if not threat_events:
            return "事件分析：无安全事件。"
        
        incident_analysis = []
        for event in threat_events:
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                incident_analysis.append(
                    f"事件 {event.event_id}: {event.description} (置信度: {event.confidence_score:.2f})"
                )
        
        return "重大事件分析：\n" + "\n".join(incident_analysis) if incident_analysis else "无重大安全事件。"
    
    def export_report(self, report: ThreatReport, format_type: str = "json", output_path: str = None) -> str:
        """导出报告"""
        if output_path is None:
            output_path = f"threat_report_{report.report_id}.{format_type}"
        
        if format_type.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)
        elif format_type.lower() == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"威胁报告 - {report.report_id}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"生成时间: {report.generated_at}\n")
                f.write(f"时间范围: {report.time_range_start} 至 {report.time_range_end}\n")
                f.write(f"总事件数: {report.total_events}\n\n")
                f.write("执行摘要:\n")
                f.write(report.executive_summary + "\n\n")
                f.write("安全建议:\n")
                for rec in report.recommendations:
                    f.write(f"- {rec}\n")
        
        return output_path


class ThreatDetector:
    """N6威胁检测器主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化威胁检测器
        
        Args:
            config: 配置字典，包含各种检测器的配置参数
        """
        self.config = config or {}
        
        # 初始化各个检测器组件
        self.malware_detector = MalwareDetector()
        self.network_attack_detector = NetworkAttackDetector()
        self.insider_threat_detector = InsiderThreatDetector()
        self.anomaly_detector = AnomalyDetector()
        self.threat_intelligence = ThreatIntelligenceIntegrator()
        self.real_time_monitor = RealTimeMonitor()
        self.threat_response_engine = ThreatResponseEngine()
        self.threat_classifier = ThreatClassifier()
        self.threat_scorer = ThreatScorer()
        self.threat_reporter = ThreatReporter()
        
        # 威胁事件存储
        self.threat_events: List[ThreatEvent] = []
        self.event_database_path = self.config.get("database_path", "threat_events.db")
        self._init_database()
        
        # 事件ID生成器
        self._event_counter = 0
        self._lock = threading.Lock()
        
        # 配置日志
        self._setup_logging()
        
        # 注册实时监控回调
        self.real_time_monitor.register_callback(self._process_real_time_event)
    
    def _setup_logging(self) -> None:
        """设置日志配置"""
        log_level = self.config.get("log_level", logging.INFO)
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(level=log_level, format=log_format)
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self) -> None:
        """初始化威胁事件数据库"""
        try:
            conn = sqlite3.connect(self.event_database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threat_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    threat_type TEXT,
                    threat_level TEXT,
                    source_ip TEXT,
                    target_ip TEXT,
                    description TEXT,
                    confidence_score REAL,
                    response_actions TEXT,
                    status TEXT,
                    assigned_to TEXT,
                    resolution_notes TEXT
                )
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
    
    def _generate_event_id(self) -> str:
        """生成唯一事件ID"""
        with self._lock:
            self._event_counter += 1
            return f"EVT-{datetime.now().strftime('%Y%m%d')}-{self._event_counter:06d}"
    
    def _save_threat_event(self, threat_event: ThreatEvent) -> None:
        """保存威胁事件到数据库"""
        try:
            conn = sqlite3.connect(self.event_database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO threat_events 
                (event_id, timestamp, threat_type, threat_level, source_ip, target_ip, 
                 description, confidence_score, response_actions, status, assigned_to, resolution_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                threat_event.event_id,
                threat_event.timestamp.isoformat(),
                threat_event.threat_type.value,
                threat_event.threat_level.value,
                threat_event.source_ip,
                threat_event.target_ip,
                threat_event.description,
                threat_event.confidence_score,
                json.dumps([action.value for action in threat_event.response_actions]),
                threat_event.status,
                threat_event.assigned_to,
                threat_event.resolution_notes
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"保存威胁事件失败: {e}")
    
    def detect_malware(self, file_path: str, process_info: Optional[Dict] = None) -> Optional[ThreatEvent]:
        """
        检测恶意软件
        
        Args:
            file_path: 文件路径
            process_info: 进程信息（可选）
            
        Returns:
            ThreatEvent: 如果检测到威胁，返回威胁事件；否则返回None
        """
        try:
            # 计算文件哈希
            file_hash = self.malware_detector.calculate_file_hash(file_path)
            if not file_hash:
                return None
            
            # 基于签名检测
            is_malware, confidence, signature = self.malware_detector.detect_malware_by_signature(file_hash)
            
            if is_malware:
                return self._create_threat_event(
                    ThreatType.MALWARE,
                    ThreatLevel.HIGH,
                    f"检测到已知恶意软件: {signature}",
                    confidence,
                    indicators=[ThreatIndicator("hash", file_hash, confidence, "signature_db", datetime.now(), datetime.now(), ["malware", "known"])]
                )
            
            # 基于行为检测（如果提供了进程信息）
            if process_info:
                is_suspicious, behavior_confidence, behaviors = self.malware_detector.detect_malware_by_behavior(process_info)
                if is_suspicious:
                    return self._create_threat_event(
                        ThreatType.MALWARE,
                        ThreatLevel.MEDIUM,
                        f"检测到可疑行为: {', '.join(behaviors)}",
                        behavior_confidence,
                        indicators=[ThreatIndicator("behavior", json.dumps(behaviors), behavior_confidence, "behavioral_analysis", datetime.now(), datetime.now(), ["suspicious", "behavior"])]
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"恶意软件检测失败: {e}")
            return None
    
    def detect_network_attacks(self, network_events: List[NetworkEvent], http_requests: Optional[List[str]] = None) -> List[ThreatEvent]:
        """
        检测网络攻击
        
        Args:
            network_events: 网络事件列表
            http_requests: HTTP请求列表（可选）
            
        Returns:
            List[ThreatEvent]: 检测到的威胁事件列表
        """
        threat_events = []
        
        try:
            # 检测DDoS攻击
            is_ddos, ddos_confidence, ddos_description = self.network_attack_detector.detect_ddos_attack(network_events)
            if is_ddos:
                threat_events.append(self._create_threat_event(
                    ThreatType.NETWORK_ATTACK,
                    ThreatLevel.HIGH,
                    ddos_description,
                    ddos_confidence
                ))
            
            # 检测端口扫描
            is_scan, scan_confidence, scan_description = self.network_attack_detector.detect_port_scan(network_events)
            if is_scan:
                threat_events.append(self._create_threat_event(
                    ThreatType.NETWORK_ATTACK,
                    ThreatLevel.MEDIUM,
                    scan_description,
                    scan_confidence
                ))
            
            # 检测HTTP请求中的注入攻击
            if http_requests:
                for request in http_requests:
                    is_injection, injection_confidence, injection_description = self.network_attack_detector.detect_injection_attack(request)
                    if is_injection:
                        threat_events.append(self._create_threat_event(
                            ThreatType.NETWORK_ATTACK,
                            ThreatLevel.HIGH,
                            injection_description,
                            injection_confidence,
                            indicators=[ThreatIndicator("http_request", request, injection_confidence, "pattern_matching", datetime.now(), datetime.now(), ["injection", "web"])]
                        ))
            
            return threat_events
            
        except Exception as e:
            self.logger.error(f"网络攻击检测失败: {e}")
            return []
    
    def detect_insider_threats(self, user_id: str, access_patterns: List[Dict]) -> Optional[ThreatEvent]:
        """
        检测内部威胁
        
        Args:
            user_id: 用户ID
            access_patterns: 访问模式列表
            
        Returns:
            ThreatEvent: 如果检测到威胁，返回威胁事件；否则返回None
        """
        try:
            # 创建或更新用户画像
            self.insider_threat_detector.create_user_profile(user_id, access_patterns)
            
            # 检测异常访问
            if access_patterns:
                latest_access = access_patterns[-1]
                is_anomalous, confidence, description = self.insider_threat_detector.detect_anomalous_access(user_id, latest_access)
                
                if is_anomalous:
                    return self._create_threat_event(
                        ThreatType.INSIDER_THREAT,
                        ThreatLevel.MEDIUM,
                        description,
                        confidence,
                        indicators=[ThreatIndicator("user_access", user_id, confidence, "behavioral_analysis", datetime.now(), datetime.now(), ["insider", "anomaly"])]
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"内部威胁检测失败: {e}")
            return None
    
    def detect_anomalies(self, current_metrics: Dict[str, float], baseline_data: Optional[Dict[str, List[float]]] = None) -> Optional[ThreatEvent]:
        """
        检测异常行为
        
        Args:
            current_metrics: 当前指标
            baseline_data: 基线数据（可选）
            
        Returns:
            ThreatEvent: 如果检测到异常，返回威胁事件；否则返回None
        """
        try:
            # 如果提供了基线数据，更新基线
            if baseline_data:
                self.anomaly_detector.establish_baseline(baseline_data)
            
            # 检测异常
            is_anomalous, confidence, anomalies = self.anomaly_detector.detect_anomaly(current_metrics)
            
            if is_anomalous:
                return self._create_threat_event(
                    ThreatType.ANOMALY,
                    ThreatLevel.MEDIUM,
                    f"检测到异常行为: {', '.join(anomalies)}",
                    confidence,
                    indicators=[ThreatIndicator("anomaly_metrics", json.dumps(current_metrics), confidence, "statistical_analysis", datetime.now(), datetime.now(), ["anomaly", "behavior"])]
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            return None
    
    def lookup_threat_intelligence(self, indicator: str) -> Optional[ThreatIndicator]:
        """
        查询威胁情报
        
        Args:
            indicator: 威胁指标
            
        Returns:
            ThreatIndicator: 如果找到，返回威胁指标；否则返回None
        """
        return self.threat_intelligence.lookup_ioc(indicator)
    
    def add_threat_intelligence(self, indicator: ThreatIndicator) -> None:
        """
        添加威胁情报
        
        Args:
            indicator: 威胁指标
        """
        self.threat_intelligence.add_ioc(indicator)
    
    def check_ip_reputation(self, ip_address: str) -> Tuple[float, str]:
        """
        检查IP信誉
        
        Args:
            ip_address: IP地址
            
        Returns:
            Tuple[float, str]: (信誉分数, 描述)
        """
        return self.threat_intelligence.check_reputation(ip_address)
    
    def start_real_time_monitoring(self) -> None:
        """启动实时威胁监控"""
        self.real_time_monitor.start_monitoring()
        self.logger.info("实时威胁监控已启动")
    
    def stop_real_time_monitoring(self) -> None:
        """停止实时威胁监控"""
        self.real_time_monitor.stop_monitoring()
        self.logger.info("实时威胁监控已停止")
    
    def _process_real_time_event(self, event: Dict) -> None:
        """处理实时事件"""
        try:
            # 这里可以实现实时威胁检测逻辑
            self.logger.debug(f"处理实时事件: {event}")
        except Exception as e:
            self.logger.error(f"处理实时事件失败: {e}")
    
    def _create_threat_event(self, threat_type: ThreatType, threat_level: ThreatLevel, 
                           description: str, confidence: float, 
                           indicators: Optional[List[ThreatIndicator]] = None,
                           source_ip: Optional[str] = None, target_ip: Optional[str] = None) -> ThreatEvent:
        """创建威胁事件"""
        event_id = self._generate_event_id()
        
        # 确定响应动作
        temp_event = ThreatEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            threat_type=threat_type,
            threat_level=threat_level,
            source_ip=source_ip,
            target_ip=target_ip,
            description=description,
            indicators=indicators or [],
            confidence_score=confidence,
            response_actions=[],
            status="new"
        )
        
        # 确定威胁等级
        if confidence > 0.8:
            if threat_level == ThreatLevel.MEDIUM:
                threat_level = ThreatLevel.HIGH
            elif threat_level == ThreatLevel.HIGH:
                threat_level = ThreatLevel.CRITICAL
        
        # 更新威胁等级
        temp_event.threat_level = threat_level
        
        # 确定响应动作
        response_actions = self.threat_response_engine.determine_response(temp_event)
        temp_event.response_actions = response_actions
        
        # 执行响应
        self.threat_response_engine.execute_response(temp_event, response_actions)
        
        # 保存事件
        self.threat_events.append(temp_event)
        self._save_threat_event(temp_event)
        
        return temp_event
    
    def analyze_threat(self, threat_data: Dict) -> ThreatEvent:
        """
        综合威胁分析
        
        Args:
            threat_data: 威胁数据
            
        Returns:
            ThreatEvent: 分析后的威胁事件
        """
        try:
            # 威胁分类
            threat_type, classification_confidence = self.threat_classifier.classify_threat(threat_data)
            
            # 计算威胁评分
            context = {
                "impact_score": threat_data.get("impact_score", 0.5),
                "frequency_score": threat_data.get("frequency_score", 0.5),
                "context_score": threat_data.get("context_score", 0.5)
            }
            
            # 创建威胁事件
            threat_event = self._create_threat_event(
                threat_type=threat_type,
                threat_level=self._determine_threat_level(classification_confidence),
                description=threat_data.get("description", "未知威胁"),
                confidence=classification_confidence,
                indicators=threat_data.get("indicators", []),
                source_ip=threat_data.get("source_ip"),
                target_ip=threat_data.get("target_ip")
            )
            
            return threat_event
            
        except Exception as e:
            self.logger.error(f"威胁分析失败: {e}")
            return self._create_threat_event(
                ThreatType.ANOMALY,
                ThreatLevel.MEDIUM,
                f"威胁分析失败: {str(e)}",
                0.0
            )
    
    def _determine_threat_level(self, confidence: float) -> ThreatLevel:
        """根据置信度确定威胁等级"""
        if confidence >= 0.9:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.7:
            return ThreatLevel.HIGH
        elif confidence >= 0.5:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def get_threat_events(self, time_range: Optional[Tuple[datetime, datetime]] = None, 
                         threat_type: Optional[ThreatType] = None,
                         threat_level: Optional[ThreatLevel] = None) -> List[ThreatEvent]:
        """
        获取威胁事件
        
        Args:
            time_range: 时间范围
            threat_type: 威胁类型过滤
            threat_level: 威胁等级过滤
            
        Returns:
            List[ThreatEvent]: 过滤后的威胁事件列表
        """
        filtered_events = self.threat_events
        
        # 时间范围过滤
        if time_range:
            start_time, end_time = time_range
            filtered_events = [event for event in filtered_events 
                             if start_time <= event.timestamp <= end_time]
        
        # 威胁类型过滤
        if threat_type:
            filtered_events = [event for event in filtered_events 
                             if event.threat_type == threat_type]
        
        # 威胁等级过滤
        if threat_level:
            filtered_events = [event for event in filtered_events 
                             if event.threat_level == threat_level]
        
        return filtered_events
    
    def generate_threat_report(self, report_type: str = "executive", 
                              time_range: Optional[Tuple[datetime, datetime]] = None) -> ThreatReport:
        """
        生成威胁报告
        
        Args:
            report_type: 报告类型 ("executive", "technical", "incident")
            time_range: 时间范围
            
        Returns:
            ThreatReport: 威胁报告
        """
        # 获取威胁事件
        threat_events = self.get_threat_events(time_range)
        
        # 生成报告
        report = self.threat_reporter.generate_report(threat_events, report_type)
        
        return report
    
    def export_threat_report(self, report: ThreatReport, format_type: str = "json", 
                           output_path: Optional[str] = None) -> str:
        """
        导出威胁报告
        
        Args:
            report: 威胁报告
            format_type: 导出格式 ("json", "txt")
            output_path: 输出路径
            
        Returns:
            str: 导出的文件路径
        """
        return self.threat_reporter.export_report(report, format_type, output_path)
    
    def update_threat_status(self, event_id: str, status: str, notes: Optional[str] = None) -> bool:
        """
        更新威胁状态
        
        Args:
            event_id: 事件ID
            status: 新状态
            notes: 备注
            
        Returns:
            bool: 更新是否成功
        """
        try:
            # 更新内存中的事件
            for event in self.threat_events:
                if event.event_id == event_id:
                    event.status = status
                    if notes:
                        event.resolution_notes = notes
                    break
            
            # 更新数据库
            conn = sqlite3.connect(self.event_database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE threat_events 
                SET status = ?, resolution_notes = ?
                WHERE event_id = ?
            """, (status, notes, event_id))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"威胁事件 {event_id} 状态已更新为: {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"更新威胁状态失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取威胁统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.threat_events:
            return {"total_events": 0}
        
        # 基本统计
        total_events = len(self.threat_events)
        
        # 按威胁类型统计
        threats_by_type = defaultdict(int)
        threats_by_level = defaultdict(int)
        threats_by_status = defaultdict(int)
        
        for event in self.threat_events:
            threats_by_type[event.threat_type.value] += 1
            threats_by_level[event.threat_level.value] += 1
            threats_by_status[event.status] += 1
        
        # 时间分布
        time_distribution = defaultdict(int)
        for event in self.threat_events:
            date_key = event.timestamp.strftime("%Y-%m-%d")
            time_distribution[date_key] += 1
        
        return {
            "total_events": total_events,
            "threats_by_type": dict(threats_by_type),
            "threats_by_level": dict(threats_by_level),
            "threats_by_status": dict(threats_by_status),
            "time_distribution": dict(time_distribution),
            "recent_events": len([e for e in self.threat_events 
                                if e.timestamp > datetime.now() - timedelta(days=7)])
        }


# 测试用例和示例使用
def run_comprehensive_test():
    """运行综合测试"""
    print("=== N6威胁检测器综合测试 ===\n")
    
    # 初始化威胁检测器
    detector = ThreatDetector({
        "log_level": logging.INFO,
        "database_path": "test_threat_events.db"
    })
    
    # 测试1: 恶意软件检测
    print("测试1: 恶意软件检测")
    malware_event = detector.detect_malware("test_file.exe", {
        "parent_process": "cmd.exe",
        "network_connections": 15,
        "file_modifications": 8
    })
    if malware_event:
        print(f"  检测到威胁: {malware_event.description}")
        print(f"  威胁等级: {malware_event.threat_level.value}")
        print(f"  置信度: {malware_event.confidence_score:.2f}")
    else:
        print("  未检测到恶意软件威胁")
    print()
    
    # 测试2: 网络攻击检测
    print("测试2: 网络攻击检测")
    network_events = [
        NetworkEvent(
            timestamp=datetime.now(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            bytes_sent=1024,
            bytes_received=2048,
            duration=1.0,
            status="established"
        )
    ] * 50  # 模拟50个连接
    
    http_requests = [
        "GET /index.php?id=1' UNION SELECT * FROM users-- HTTP/1.1",
        "<script>alert('XSS')</script>"
    ]
    
    attack_events = detector.detect_network_attacks(network_events, http_requests)
    print(f"  检测到 {len(attack_events)} 个网络攻击事件:")
    for event in attack_events:
        print(f"    - {event.description}")
    print()
    
    # 测试3: 内部威胁检测
    print("测试3: 内部威胁检测")
    access_patterns = [
        {"access_time": datetime.now().replace(hour=9), "resource": "/documents", "access_frequency": 1},
        {"access_time": datetime.now().replace(hour=14), "resource": "/financial", "access_frequency": 1},
        {"access_time": datetime.now().replace(hour=23), "resource": "/admin", "access_frequency": 1}  # 异常访问时间
    ]
    
    insider_event = detector.detect_insider_threats("user123", access_patterns)
    if insider_event:
        print(f"  检测到内部威胁: {insider_event.description}")
    else:
        print("  未检测到内部威胁")
    print()
    
    # 测试4: 异常检测
    print("测试4: 异常行为检测")
    baseline_data = {
        "cpu_usage": [20.0, 25.0, 30.0, 22.0, 28.0],
        "memory_usage": [60.0, 65.0, 70.0, 63.0, 68.0],
        "network_traffic": [1000.0, 1200.0, 1100.0, 1150.0, 1050.0]
    }
    
    current_metrics = {
        "cpu_usage": 85.0,  # 异常高
        "memory_usage": 90.0,  # 异常高
        "network_traffic": 5000.0  # 异常高
    }
    
    anomaly_event = detector.detect_anomalies(current_metrics, baseline_data)
    if anomaly_event:
        print(f"  检测到异常: {anomaly_event.description}")
    else:
        print("  未检测到异常")
    print()
    
    # 测试5: 威胁情报查询
    print("测试5: 威胁情报查询")
    # 添加威胁情报
    threat_ioc = ThreatIndicator(
        indicator_type="ip",
        value="192.168.1.100",
        confidence=0.95,
        source="threat_intel_feed",
        first_seen=datetime.now(),
        last_seen=datetime.now(),
        tags=["malicious", "botnet"]
    )
    detector.add_threat_intelligence(threat_ioc)
    
    # 查询威胁情报
    lookup_result = detector.lookup_threat_intelligence("192.168.1.100")
    if lookup_result:
        print(f"  找到威胁情报: {lookup_result.value} (置信度: {lookup_result.confidence})")
    else:
        print("  未找到相关威胁情报")
    
    # 检查IP信誉
    reputation_score, reputation_desc = detector.check_ip_reputation("192.168.1.100")
    print(f"  IP信誉: {reputation_desc} (分数: {reputation_score})")
    print()
    
    # 测试6: 实时监控
    print("测试6: 实时威胁监控")
    detector.start_real_time_monitoring()
    
    # 模拟实时事件
    detector.real_time_monitor.add_event({
        "type": "login_attempt",
        "user": "admin",
        "ip": "192.168.1.200",
        "timestamp": datetime.now().isoformat()
    })
    
    time.sleep(1)  # 等待处理
    detector.stop_real_time_monitoring()
    print("  实时监控测试完成")
    print()
    
    # 测试7: 综合威胁分析
    print("测试7: 综合威胁分析")
    threat_data = {
        "description": "检测到异常网络流量和可疑进程行为",
        "indicators": ["high_network_traffic", "suspicious_process"],
        "source_ip": "10.0.0.100",
        "target_ip": "192.168.1.1",
        "impact_score": 0.8,
        "frequency_score": 0.6,
        "context_score": 0.7
    }
    
    analyzed_event = detector.analyze_threat(threat_data)
    print(f"  分析结果:")
    print(f"    事件ID: {analyzed_event.event_id}")
    print(f"    威胁类型: {analyzed_event.threat_type.value}")
    print(f"    威胁等级: {analyzed_event.threat_level.value}")
    print(f"    描述: {analyzed_event.description}")
    print(f"    置信度: {analyzed_event.confidence_score:.2f}")
    print(f"    响应动作: {[action.value for action in analyzed_event.response_actions]}")
    print()
    
    # 测试8: 威胁报告生成
    print("测试8: 威胁报告生成")
    report = detector.generate_threat_report("executive")
    print(f"  报告ID: {report.report_id}")
    print(f"  生成时间: {report.generated_at}")
    print(f"  总事件数: {report.total_events}")
    print(f"  执行摘要: {report.executive_summary}")
    print(f"  安全建议: {report.recommendations}")
    
    # 导出报告
    report_path = detector.export_threat_report(report, "json", "test_threat_report.json")
    print(f"  报告已导出至: {report_path}")
    print()
    
    # 测试9: 统计信息
    print("测试9: 威胁统计信息")
    stats = detector.get_statistics()
    print(f"  统计信息:")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    print()
    
    # 测试10: 威胁状态更新
    print("测试10: 威胁状态更新")
    if detector.threat_events:
        latest_event = detector.threat_events[-1]
        success = detector.update_threat_status(latest_event.event_id, "resolved", "已通过安全扫描确认安全")
        print(f"  状态更新{'成功' if success else '失败'}")
    print()
    
    print("=== 测试完成 ===")


if __name__ == "__main__":
    # 运行综合测试
    run_comprehensive_test()