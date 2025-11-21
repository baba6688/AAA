#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W5防火墙管理器

这是一个功能完整的防火墙管理器，提供网络安全防护功能。

功能特性：
- 防火墙规则管理（入站、出站规则管理）
- 访问控制（IP白名单、黑名单管理）
- 端口管理（端口开放、关闭管理）
- 协议控制（TCP、UDP、ICMP协议控制）
- 安全防护（DDoS防护、入侵检测）
- 防火墙日志（安全事件日志记录）
- 防火墙统计（安全事件统计和分析）
- 防火墙配置（防火墙配置管理和备份）

作者: W5防火墙开发团队
版本: 1.0.0
"""

import json
import logging
import re
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import ipaddress
import hashlib
import random


@dataclass
class FirewallRule:
    """防火墙规则数据类"""
    rule_id: str
    name: str
    action: str  # 'allow' or 'deny'
    direction: str  # 'inbound' or 'outbound'
    protocol: str  # 'TCP', 'UDP', 'ICMP', 'ALL'
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[str] = None
    destination_port: Optional[str] = None
    enabled: bool = True
    priority: int = 100
    created_time: str = None
    description: str = ""

    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now().isoformat()


@dataclass
class SecurityEvent:
    """安全事件数据类"""
    event_id: str
    timestamp: str
    event_type: str
    source_ip: str
    destination_ip: str
    port: int
    protocol: str
    action: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    rule_id: Optional[str] = None


class FirewallManager:
    """W5防火墙管理器主类"""

    def __init__(self, config_file: str = "firewall_config.json", log_file: str = "firewall.log"):
        """
        初始化防火墙管理器
        
        Args:
            config_file: 配置文件路径
            log_file: 日志文件路径
        """
        self.config_file = config_file
        self.log_file = log_file
        
        # 初始化日志系统
        self._setup_logging()
        
        # 防火墙规则存储
        self.rules: Dict[str, FirewallRule] = {}
        
        # IP访问控制列表
        self.whitelist: set = set()
        self.blacklist: set = set()
        
        # 端口管理
        self.allowed_ports: set = set()
        self.blocked_ports: set = set()
        
        # 安全事件记录
        self.security_events: deque = deque(maxlen=10000)
        
        # DDoS防护
        self.ddos_protection = {
            'enabled': True,
            'max_requests_per_second': 100,
            'blocked_ips': set(),
            'request_counts': defaultdict(int),
            'last_reset': time.time()
        }
        
        # 入侵检测
        self.intrusion_detection = {
            'enabled': True,
            'suspicious_patterns': [
                r'.*sql.*injection.*',
                r'.*script.*alert.*',
                r'.*<script.*',
                r'.*eval\(.*',
                r'.*system\(.*',
                r'.*exec\(.*'
            ],
            'detected_threats': []
        }
        
        # 统计信息
        self.statistics = {
            'total_packets_processed': 0,
            'allowed_packets': 0,
            'denied_packets': 0,
            'blocked_ips_count': 0,
            'security_events_count': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # 加载配置
        self.load_config()
        
        self.logger.info("W5防火墙管理器已启动")

    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("W5Firewall")

    def validate_ip(self, ip: str) -> bool:
        """验证IP地址格式"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def validate_port(self, port: str) -> bool:
        """验证端口号格式"""
        try:
            port_num = int(port)
            return 1 <= port_num <= 65535
        except ValueError:
            return False

    def validate_port_range(self, port_range: str) -> bool:
        """验证端口范围格式"""
        try:
            if '-' in port_range:
                start, end = map(int, port_range.split('-'))
                return 1 <= start <= end <= 65535
            else:
                return self.validate_port(port_range)
        except ValueError:
            return False

    # ==================== 防火墙规则管理 ====================

    def add_rule(self, name: str, action: str, direction: str, protocol: str,
                 source_ip: Optional[str] = None, destination_ip: Optional[str] = None,
                 source_port: Optional[str] = None, destination_port: Optional[str] = None,
                 priority: int = 100, description: str = "") -> bool:
        """
        添加防火墙规则
        
        Args:
            name: 规则名称
            action: 操作类型 ('allow' 或 'deny')
            direction: 方向 ('inbound' 或 'outbound')
            protocol: 协议 ('TCP', 'UDP', 'ICMP', 'ALL')
            source_ip: 源IP地址
            destination_ip: 目标IP地址
            source_port: 源端口
            destination_port: 目标端口
            priority: 优先级
            description: 描述
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 验证参数
            if action not in ['allow', 'deny']:
                self.logger.error(f"无效的操作类型: {action}")
                return False
            
            if direction not in ['inbound', 'outbound']:
                self.logger.error(f"无效的方向: {direction}")
                return False
            
            if protocol not in ['TCP', 'UDP', 'ICMP', 'ALL']:
                self.logger.error(f"无效的协议: {protocol}")
                return False
            
            # 验证IP和端口
            if source_ip and not self.validate_ip(source_ip):
                self.logger.error(f"无效的源IP地址: {source_ip}")
                return False
            
            if destination_ip and not self.validate_ip(destination_ip):
                self.logger.error(f"无效的目标IP地址: {destination_ip}")
                return False
            
            if source_port and not self.validate_port_range(source_port):
                self.logger.error(f"无效的源端口: {source_port}")
                return False
            
            if destination_port and not self.validate_port_range(destination_port):
                self.logger.error(f"无效的目标端口: {destination_port}")
                return False
            
            # 生成规则ID
            rule_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
            
            # 创建规则
            rule = FirewallRule(
                rule_id=rule_id,
                name=name,
                action=action,
                direction=direction,
                protocol=protocol,
                source_ip=source_ip,
                destination_ip=destination_ip,
                source_port=source_port,
                destination_port=destination_port,
                priority=priority,
                description=description
            )
            
            self.rules[rule_id] = rule
            self.logger.info(f"已添加防火墙规则: {name} (ID: {rule_id})")
            self.save_config()
            return True
            
        except Exception as e:
            self.logger.error(f"添加规则失败: {str(e)}")
            return False

    def remove_rule(self, rule_id: str) -> bool:
        """删除防火墙规则"""
        if rule_id in self.rules:
            rule_name = self.rules[rule_id].name
            del self.rules[rule_id]
            self.logger.info(f"已删除防火墙规则: {rule_name} (ID: {rule_id})")
            self.save_config()
            return True
        else:
            self.logger.warning(f"规则不存在: {rule_id}")
            return False

    def update_rule(self, rule_id: str, **kwargs) -> bool:
        """更新防火墙规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"规则不存在: {rule_id}")
            return False
        
        rule = self.rules[rule_id]
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        self.logger.info(f"已更新防火墙规则: {rule.name} (ID: {rule_id})")
        self.save_config()
        return True

    def get_rules(self, direction: Optional[str] = None, protocol: Optional[str] = None) -> List[FirewallRule]:
        """获取防火墙规则列表"""
        rules = list(self.rules.values())
        
        if direction:
            rules = [r for r in rules if r.direction == direction]
        
        if protocol:
            rules = [r for r in rules if r.protocol == protocol]
        
        # 按优先级排序
        rules.sort(key=lambda x: x.priority)
        return rules

    # ==================== 访问控制 ====================

    def add_to_whitelist(self, ip: str) -> bool:
        """添加IP到白名单"""
        if not self.validate_ip(ip):
            self.logger.error(f"无效的IP地址: {ip}")
            return False
        
        self.whitelist.add(ip)
        self.logger.info(f"已添加IP到白名单: {ip}")
        self.save_config()
        return True

    def remove_from_whitelist(self, ip: str) -> bool:
        """从白名单移除IP"""
        if ip in self.whitelist:
            self.whitelist.remove(ip)
            self.logger.info(f"已从白名单移除IP: {ip}")
            self.save_config()
            return True
        return False

    def add_to_blacklist(self, ip: str) -> bool:
        """添加IP到黑名单"""
        if not self.validate_ip(ip):
            self.logger.error(f"无效的IP地址: {ip}")
            return False
        
        self.blacklist.add(ip)
        self.logger.info(f"已添加IP到黑名单: {ip}")
        self.save_config()
        return True

    def remove_from_blacklist(self, ip: str) -> bool:
        """从黑名单移除IP"""
        if ip in self.blacklist:
            self.blacklist.remove(ip)
            self.logger.info(f"已从黑名单移除IP: {ip}")
            self.save_config()
            return True
        return False

    def get_whitelist(self) -> List[str]:
        """获取白名单"""
        return list(self.whitelist)

    def get_blacklist(self) -> List[str]:
        """获取黑名单"""
        return list(self.blacklist)

    # ==================== 端口管理 ====================

    def allow_port(self, port: str) -> bool:
        """开放端口"""
        if not self.validate_port_range(port):
            self.logger.error(f"无效的端口: {port}")
            return False
        
        self.allowed_ports.add(port)
        self.blocked_ports.discard(port)
        self.logger.info(f"已开放端口: {port}")
        self.save_config()
        return True

    def block_port(self, port: str) -> bool:
        """关闭端口"""
        if not self.validate_port_range(port):
            self.logger.error(f"无效的端口: {port}")
            return False
        
        self.blocked_ports.add(port)
        self.allowed_ports.discard(port)
        self.logger.info(f"已关闭端口: {port}")
        self.save_config()
        return True

    def remove_port_rule(self, port: str) -> bool:
        """移除端口规则"""
        removed = False
        if port in self.allowed_ports:
            self.allowed_ports.remove(port)
            removed = True
        if port in self.blocked_ports:
            self.blocked_ports.remove(port)
            removed = True
        
        if removed:
            self.logger.info(f"已移除端口规则: {port}")
            self.save_config()
            return True
        return False

    def get_allowed_ports(self) -> List[str]:
        """获取开放的端口"""
        return list(self.allowed_ports)

    def get_blocked_ports(self) -> List[str]:
        """获取关闭的端口"""
        return list(self.blocked_ports)

    # ==================== 协议控制 ====================

    def allow_protocol(self, protocol: str) -> bool:
        """允许协议"""
        if protocol not in ['TCP', 'UDP', 'ICMP']:
            self.logger.error(f"不支持的协议: {protocol}")
            return False
        
        # 创建协议允许规则
        rule_name = f"允许{protocol}协议"
        return self.add_rule(
            name=rule_name,
            action='allow',
            direction='both',
            protocol=protocol,
            description=f"允许{protocol}协议流量"
        )

    def block_protocol(self, protocol: str) -> bool:
        """阻止协议"""
        if protocol not in ['TCP', 'UDP', 'ICMP']:
            self.logger.error(f"不支持的协议: {protocol}")
            return False
        
        # 创建协议阻止规则
        rule_name = f"阻止{protocol}协议"
        return self.add_rule(
            name=rule_name,
            action='deny',
            direction='both',
            protocol=protocol,
            description=f"阻止{protocol}协议流量"
        )

    # ==================== 安全防护 ====================

    def check_ddos_protection(self, ip: str) -> bool:
        """检查DDoS防护"""
        if not self.ddos_protection['enabled']:
            return True
        
        current_time = time.time()
        
        # 重置计数器（每分钟）
        if current_time - self.ddos_protection['last_reset'] > 60:
            self.ddos_protection['request_counts'].clear()
            self.ddos_protection['last_reset'] = current_time
        
        # 增加请求计数
        self.ddos_protection['request_counts'][ip] += 1
        
        # 检查是否超过阈值
        if self.ddos_protection['request_counts'][ip] > self.ddos_protection['max_requests_per_second']:
            self.ddos_protection['blocked_ips'].add(ip)
            self.logger.warning(f"DDoS攻击检测: IP {ip} 已被临时封禁")
            self._log_security_event(
                event_type="DDoS检测",
                source_ip=ip,
                destination_ip="",
                port=0,
                protocol="ALL",
                action="block",
                severity="high",
                description=f"检测到DDoS攻击，IP {ip} 请求频率过高"
            )
            return False
        
        return True

    def check_intrusion_detection(self, data: str, ip: str) -> bool:
        """检查入侵检测"""
        if not self.intrusion_detection['enabled']:
            return True
        
        for pattern in self.intrusion_detection['suspicious_patterns']:
            if re.search(pattern, data, re.IGNORECASE):
                threat_info = {
                    'timestamp': datetime.now().isoformat(),
                    'source_ip': ip,
                    'threat_type': '可疑模式匹配',
                    'pattern': pattern,
                    'data': data[:100]  # 只记录前100个字符
                }
                self.intrusion_detection['detected_threats'].append(threat_info)
                
                self.logger.warning(f"入侵检测: IP {ip} 检测到可疑活动")
                self._log_security_event(
                    event_type="入侵检测",
                    source_ip=ip,
                    destination_ip="",
                    port=0,
                    protocol="ALL",
                    action="alert",
                    severity="medium",
                    description=f"检测到可疑活动: {pattern}"
                )
                return False
        
        return True

    def enable_ddos_protection(self, max_requests_per_second: int = 100):
        """启用DDoS防护"""
        self.ddos_protection['enabled'] = True
        self.ddos_protection['max_requests_per_second'] = max_requests_per_second
        self.logger.info("已启用DDoS防护")

    def disable_ddos_protection(self):
        """禁用DDoS防护"""
        self.ddos_protection['enabled'] = False
        self.logger.info("已禁用DDoS防护")

    def enable_intrusion_detection(self):
        """启用入侵检测"""
        self.intrusion_detection['enabled'] = True
        self.logger.info("已启用入侵检测")

    def disable_intrusion_detection(self):
        """禁用入侵检测"""
        self.intrusion_detection['enabled'] = False
        self.logger.info("已禁用入侵检测")

    # ==================== 防火墙日志 ====================

    def _log_security_event(self, event_type: str, source_ip: str, destination_ip: str,
                          port: int, protocol: str, action: str, severity: str, description: str):
        """记录安全事件"""
        event = SecurityEvent(
            event_id=hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:8],
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            source_ip=source_ip,
            destination_ip=destination_ip,
            port=port,
            protocol=protocol,
            action=action,
            severity=severity,
            description=description
        )
        
        self.security_events.append(event)
        self.statistics['security_events_count'] += 1
        
        self.logger.info(f"安全事件: {event_type} - {description}")

    def get_security_events(self, hours: int = 24) -> List[SecurityEvent]:
        """获取安全事件日志"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()
        
        return [event for event in self.security_events if event.timestamp > cutoff_str]

    def export_security_log(self, filename: str, hours: int = 24) -> bool:
        """导出安全日志"""
        try:
            events = self.get_security_events(hours)
            log_data = [asdict(event) for event in events]
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"已导出安全日志到: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"导出安全日志失败: {str(e)}")
            return False

    # ==================== 防火墙统计 ====================

    def update_statistics(self, packet_allowed: bool):
        """更新统计信息"""
        self.statistics['total_packets_processed'] += 1
        if packet_allowed:
            self.statistics['allowed_packets'] += 1
        else:
            self.statistics['denied_packets'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.statistics.copy()
        stats['blocked_ips_count'] = len(self.blacklist) + len(self.ddos_protection['blocked_ips'])
        stats['current_rules_count'] = len(self.rules)
        stats['whitelist_count'] = len(self.whitelist)
        stats['blacklist_count'] = len(self.blacklist)
        stats['allowed_ports_count'] = len(self.allowed_ports)
        stats['blocked_ports_count'] = len(self.blocked_ports)
        stats['security_events_last_24h'] = len(self.get_security_events(24))
        return stats

    def generate_statistics_report(self) -> str:
        """生成统计报告"""
        stats = self.get_statistics()
        
        report = f"""
=== W5防火墙统计报告 ===
生成时间: {datetime.now().isoformat()}

=== 基本统计 ===
启动时间: {stats['start_time']}
总处理包数: {stats['total_packets_processed']}
允许包数: {stats['allowed_packets']}
拒绝包数: {stats['denied_packets']}
允许率: {stats['allowed_packets']/max(stats['total_packets_processed'], 1)*100:.2f}%

=== 访问控制 ===
白名单IP数: {stats['whitelist_count']}
黑名单IP数: {stats['blacklist_count']}
封禁IP总数: {stats['blocked_ips_count']}

=== 端口管理 ===
开放端口数: {stats['allowed_ports_count']}
关闭端口数: {stats['blocked_ports_count']}

=== 规则管理 ===
当前规则数: {stats['current_rules_count']}

=== 安全事件 ===
总安全事件数: {stats['security_events_count']}
24小时内安全事件数: {stats['security_events_last_24h']}

=== 安全防护 ===
DDoS防护: {'启用' if self.ddos_protection['enabled'] else '禁用'}
入侵检测: {'启用' if self.intrusion_detection['enabled'] else '禁用'}
        """
        
        return report.strip()

    # ==================== 防火墙配置 ====================

    def save_config(self):
        """保存配置"""
        try:
            config = {
                'rules': {rid: asdict(rule) for rid, rule in self.rules.items()},
                'whitelist': list(self.whitelist),
                'blacklist': list(self.blacklist),
                'allowed_ports': list(self.allowed_ports),
                'blocked_ports': list(self.blocked_ports),
                'ddos_protection': {
                    'enabled': self.ddos_protection['enabled'],
                    'max_requests_per_second': self.ddos_protection['max_requests_per_second']
                },
                'intrusion_detection': {
                    'enabled': self.intrusion_detection['enabled']
                },
                'statistics': self.statistics
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            self.logger.debug("配置已保存")
        except Exception as e:
            self.logger.error(f"保存配置失败: {str(e)}")

    def load_config(self):
        """加载配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 加载规则
            if 'rules' in config:
                for rid, rule_data in config['rules'].items():
                    self.rules[rid] = FirewallRule(**rule_data)
            
            # 加载访问控制列表
            self.whitelist = set(config.get('whitelist', []))
            self.blacklist = set(config.get('blacklist', []))
            
            # 加载端口管理
            self.allowed_ports = set(config.get('allowed_ports', []))
            self.blocked_ports = set(config.get('blocked_ports', []))
            
            # 加载安全防护设置
            if 'ddos_protection' in config:
                self.ddos_protection.update(config['ddos_protection'])
            
            if 'intrusion_detection' in config:
                self.intrusion_detection.update(config['intrusion_detection'])
            
            # 加载统计信息
            if 'statistics' in config:
                self.statistics.update(config['statistics'])
            
            self.logger.info("配置已加载")
        except FileNotFoundError:
            self.logger.info("配置文件不存在，使用默认配置")
        except Exception as e:
            self.logger.error(f"加载配置失败: {str(e)}")

    def backup_config(self, backup_file: str = None) -> bool:
        """备份配置"""
        if backup_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"firewall_backup_{timestamp}.json"
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                config = {
                    'rules': {rid: asdict(rule) for rid, rule in self.rules.items()},
                    'whitelist': list(self.whitelist),
                    'blacklist': list(self.blacklist),
                    'allowed_ports': list(self.allowed_ports),
                    'blocked_ports': list(self.blocked_ports),
                    'ddos_protection': self.ddos_protection,
                    'intrusion_detection': self.intrusion_detection,
                    'statistics': self.statistics,
                    'backup_time': datetime.now().isoformat()
                }
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"配置已备份到: {backup_file}")
            return True
        except Exception as e:
            self.logger.error(f"备份配置失败: {str(e)}")
            return False

    def restore_config(self, backup_file: str) -> bool:
        """恢复配置"""
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 清空当前配置
            self.rules.clear()
            self.whitelist.clear()
            self.blacklist.clear()
            self.allowed_ports.clear()
            self.blocked_ports.clear()
            
            # 恢复规则
            if 'rules' in config:
                for rid, rule_data in config['rules'].items():
                    self.rules[rid] = FirewallRule(**rule_data)
            
            # 恢复访问控制列表
            self.whitelist = set(config.get('whitelist', []))
            self.blacklist = set(config.get('blacklist', []))
            
            # 恢复端口管理
            self.allowed_ports = set(config.get('allowed_ports', []))
            self.blocked_ports = set(config.get('blocked_ports', []))
            
            # 恢复安全防护设置
            if 'ddos_protection' in config:
                self.ddos_protection.update(config['ddos_protection'])
            
            if 'intrusion_detection' in config:
                self.intrusion_detection.update(config['intrusion_detection'])
            
            self.save_config()
            self.logger.info(f"配置已从 {backup_file} 恢复")
            return True
        except Exception as e:
            self.logger.error(f"恢复配置失败: {str(e)}")
            return False

    # ==================== 核心功能 ====================

    def process_packet(self, source_ip: str, destination_ip: str, 
                      port: int, protocol: str, data: str = "") -> Tuple[bool, str]:
        """
        处理网络数据包
        
        Args:
            source_ip: 源IP地址
            destination_ip: 目标IP地址
            port: 端口号
            protocol: 协议
            data: 数据内容（用于入侵检测）
            
        Returns:
            Tuple[bool, str]: (是否允许, 原因)
        """
        self.statistics['total_packets_processed'] += 1
        
        # DDoS防护检查
        if not self.check_ddos_protection(source_ip):
            self.update_statistics(False)
            return False, "DDoS攻击检测"
        
        # 入侵检测检查
        if data and not self.check_intrusion_detection(data, source_ip):
            self.update_statistics(False)
            return False, "入侵检测警报"
        
        # 黑名单检查
        if source_ip in self.blacklist:
            self._log_security_event(
                event_type="黑名单拦截",
                source_ip=source_ip,
                destination_ip=destination_ip,
                port=port,
                protocol=protocol,
                action="deny",
                severity="medium",
                description=f"来自黑名单IP {source_ip} 的连接被拒绝"
            )
            self.update_statistics(False)
            return False, "IP在黑名单中"
        
        # 白名单检查
        if source_ip in self.whitelist:
            self.update_statistics(True)
            return True, "IP在白名单中"
        
        # 端口检查
        port_str = str(port)
        if port_str in self.blocked_ports:
            self._log_security_event(
                event_type="端口拦截",
                source_ip=source_ip,
                destination_ip=destination_ip,
                port=port,
                protocol=protocol,
                action="deny",
                severity="low",
                description=f"端口 {port} 被阻止"
            )
            self.update_statistics(False)
            return False, f"端口 {port} 被阻止"
        
        # 防火墙规则检查
        allowed = self._check_firewall_rules(source_ip, destination_ip, port, protocol)
        
        if allowed:
            self.statistics['allowed_packets'] += 1
            return True, "通过防火墙规则检查"
        else:
            self.statistics['denied_packets'] += 1
            return False, "被防火墙规则拒绝"

    def _check_firewall_rules(self, source_ip: str, destination_ip: str, 
                            port: int, protocol: str) -> bool:
        """检查防火墙规则"""
        rules = self.get_rules()
        
        for rule in rules:
            if not rule.enabled:
                continue
            
            # 协议检查
            if rule.protocol != 'ALL' and rule.protocol != protocol:
                continue
            
            # 方向检查（简化处理）
            if rule.direction not in ['both', 'inbound', 'outbound']:
                continue
            
            # IP检查
            if rule.source_ip and rule.source_ip != source_ip:
                continue
            
            if rule.destination_ip and rule.destination_ip != destination_ip:
                continue
            
            # 端口检查
            if rule.destination_port:
                if '-' in rule.destination_port:
                    start, end = map(int, rule.destination_port.split('-'))
                    if not (start <= port <= end):
                        continue
                elif rule.destination_port != str(port):
                    continue
            
            # 规则匹配，执行操作
            if rule.action == 'allow':
                self.logger.debug(f"规则允许: {rule.name}")
                return True
            else:
                self.logger.debug(f"规则拒绝: {rule.name}")
                self._log_security_event(
                    event_type="规则拦截",
                    source_ip=source_ip,
                    destination_ip=destination_ip,
                    port=port,
                    protocol=protocol,
                    action="deny",
                    severity="low",
                    description=f"规则 {rule.name} 拒绝连接"
                )
                return False
        
        # 没有匹配的规则，默认允许
        return True

    def start_monitoring(self):
        """启动监控（模拟）"""
        self.logger.info("防火墙监控已启动")
        # 在实际应用中，这里会启动实际的监控线程
        return True

    def stop_monitoring(self):
        """停止监控"""
        self.logger.info("防火墙监控已停止")
        # 在实际应用中，这里会停止监控线程
        return True

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'status': 'running',
            'uptime': time.time() - time.mktime(datetime.fromisoformat(self.statistics['start_time']).timetuple()),
            'rules_count': len(self.rules),
            'whitelist_count': len(self.whitelist),
            'blacklist_count': len(self.blacklist),
            'ddos_protection_enabled': self.ddos_protection['enabled'],
            'intrusion_detection_enabled': self.intrusion_detection['enabled'],
            'total_packets_processed': self.statistics['total_packets_processed']
        }


if __name__ == "__main__":
    # 创建防火墙管理器实例
    firewall = FirewallManager()
    
    # 添加示例规则
    firewall.add_rule(
        name="允许HTTP流量",
        action="allow",
        direction="inbound",
        protocol="TCP",
        destination_port="80",
        description="允许HTTP web流量"
    )
    
    firewall.add_rule(
        name="阻止SSH暴力破解",
        action="deny",
        direction="inbound",
        protocol="TCP",
        destination_port="22",
        description="阻止SSH端口的暴力破解攻击"
    )
    
    # 添加访问控制
    firewall.add_to_whitelist("192.168.1.100")
    firewall.add_to_blacklist("10.0.0.1")
    
    # 端口管理
    firewall.allow_port("443")
    firewall.block_port("23")
    
    # 启动监控
    firewall.start_monitoring()
    
    # 模拟数据包处理
    result, reason = firewall.process_packet("192.168.1.100", "192.168.1.1", 80, "TCP")
    print(f"数据包处理结果: 允许={result}, 原因={reason}")
    
    # 生成统计报告
    print(firewall.generate_statistics_report())
    
    # 获取系统状态
    status = firewall.get_system_status()
    print(f"系统状态: {json.dumps(status, ensure_ascii=False, indent=2)}")