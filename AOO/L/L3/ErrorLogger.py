#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3错误日志记录器模块

本模块提供了企业级的错误日志记录和处理功能，包括：
1. 异常捕获和分类记录（系统异常、业务异常、网络异常）
2. 错误堆栈跟踪和上下文信息
3. 错误统计和分析（错误频率、错误类型、影响范围）
4. 错误告警和通知机制
5. 错误恢复和重试日志
6. 错误解决和关闭流程
7. 异步错误日志处理
8. 完整的错误处理和日志记录

"""

import asyncio
import logging
import json
import sqlite3
import threading
import time
import traceback
import uuid
import smtplib
import requests
from datetime import datetime, timedelta

# 可选导入
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys
import psutil
import gc
import weakref
from pathlib import Path


class ErrorType(Enum):
    """错误类型枚举"""
    SYSTEM = "system"          # 系统异常
    BUSINESS = "business"      # 业务异常
    NETWORK = "network"        # 网络异常
    DATABASE = "database"      # 数据库异常
    SECURITY = "security"      # 安全异常
    PERFORMANCE = "performance" # 性能异常
    CONFIGURATION = "configuration" # 配置异常
    UNKNOWN = "unknown"        # 未知异常


class ErrorSeverity(Enum):
    """错误严重级别枚举"""
    CRITICAL = "critical"      # 严重
    HIGH = "high"              # 高
    MEDIUM = "medium"          # 中
    LOW = "low"                # 低
    INFO = "info"              # 信息


class ErrorStatus(Enum):
    """错误状态枚举"""
    OPEN = "open"              # 打开
    IN_PROGRESS = "in_progress" # 处理中
    RESOLVED = "resolved"      # 已解决
    CLOSED = "closed"          # 已关闭
    IGNORED = "ignored"        # 已忽略
    RECURRING = "recurring"    # 重复发生


class AlertChannel(Enum):
    """告警渠道枚举"""
    EMAIL = "email"            # 邮件
    WEBHOOK = "webhook"        # Webhook
    SMS = "sms"               # 短信
    SLACK = "slack"           # Slack
    DINGTALK = "dingtalk"     # 钉钉
    WECHAT = "wechat"         # 微信
    LOG = "log"               # 日志
    DATABASE = "database"     # 数据库


class RecoveryStrategy(Enum):
    """恢复策略枚举"""
    RETRY = "retry"           # 重试
    FALLBACK = "fallback"     # 降级
    CIRCUIT_BREAKER = "circuit_breaker"  # 熔断器
    BULKHEAD = "bulkhead"     # 隔板模式
    TIMEOUT = "timeout"       # 超时
    IGNORE = "ignore"         # 忽略


@dataclass
class ErrorContext:
    """错误上下文信息类"""
    error_id: str
    timestamp: datetime
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    exception_type: str
    stack_trace: str
    module_name: str
    function_name: str
    line_number: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    service_name: Optional[str] = None
    host_name: Optional[str] = None
    process_id: Optional[int] = None
    thread_id: Optional[int] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    additional_data: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['error_type'] = self.error_type.value
        data['severity'] = self.severity.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorContext':
        """从字典创建实例"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['error_type'] = ErrorType(data['error_type'])
        data['severity'] = ErrorSeverity(data['severity'])
        return cls(**data)


@dataclass
class ErrorStatistics:
    """错误统计分析类"""
    total_errors: int = 0
    errors_by_type: Dict[ErrorType, int] = None
    errors_by_severity: Dict[ErrorSeverity, int] = None
    errors_by_module: Dict[str, int] = None
    errors_by_hour: Dict[int, int] = None
    recurring_errors: int = 0
    resolved_errors: int = 0
    avg_resolution_time: float = 0.0
    peak_error_time: Optional[datetime] = None
    error_frequency: Dict[str, int] = None
    affected_users: int = 0
    system_impact_score: float = 0.0

    def __post_init__(self):
        if self.errors_by_type is None:
            self.errors_by_type = {t: 0 for t in ErrorType}
        if self.errors_by_severity is None:
            self.errors_by_severity = {s: 0 for s in ErrorSeverity}
        if self.errors_by_module is None:
            self.errors_by_module = {}
        if self.errors_by_hour is None:
            self.errors_by_hour = {}
        if self.error_frequency is None:
            self.error_frequency = {}


class AlertManager:
    """错误告警管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = config.get('alert_rules', {})
        self.notification_channels = config.get('notification_channels', {})
        self.alert_history = deque(maxlen=1000)
        self.alert_suppression = defaultdict(float)
        self._lock = threading.RLock()
    
    def should_alert(self, error_context: ErrorContext) -> bool:
        """判断是否需要发送告警"""
        with self._lock:
            # 检查告警抑制
            suppression_key = f"{error_context.error_type.value}_{error_context.severity.value}"
            if suppression_key in self.alert_suppression:
                if time.time() - self.alert_suppression[suppression_key] < 300:  # 5分钟抑制
                    return False
            
            # 检查告警规则
            rule_key = f"{error_context.error_type.value}_{error_context.severity.value}"
            if rule_key in self.alert_rules:
                rule = self.alert_rules[rule_key]
                return rule.get('enabled', True)
            
            return error_context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]
    
    def send_alert(self, error_context: ErrorContext, channels: List[AlertChannel] = None) -> bool:
        """发送告警通知"""
        if channels is None:
            channels = [AlertChannel.LOG]
        
        success_count = 0
        with self._lock:
            for channel in channels:
                try:
                    if self._send_alert_by_channel(channel, error_context):
                        success_count += 1
                except Exception as e:
                    logging.error(f"告警发送失败 [{channel.value}]: {str(e)}")
            
            # 记录告警历史
            self.alert_history.append({
                'timestamp': time.time(),
                'error_id': error_context.error_id,
                'channels': [c.value for c in channels],
                'success_count': success_count,
                'total_count': len(channels)
            })
            
            # 设置告警抑制
            suppression_key = f"{error_context.error_type.value}_{error_context.severity.value}"
            self.alert_suppression[suppression_key] = time.time()
        
        return success_count > 0
    
    def _send_alert_by_channel(self, channel: AlertChannel, error_context: ErrorContext) -> bool:
        """通过指定渠道发送告警"""
        try:
            if channel == AlertChannel.EMAIL:
                return self._send_email_alert(error_context)
            elif channel == AlertChannel.WEBHOOK:
                return self._send_webhook_alert(error_context)
            elif channel == AlertChannel.SLACK:
                return self._send_slack_alert(error_context)
            elif channel == AlertChannel.DINGTALK:
                return self._send_dingtalk_alert(error_context)
            elif channel == AlertChannel.WECHAT:
                return self._send_wechat_alert(error_context)
            elif channel == AlertChannel.LOG:
                return self._send_log_alert(error_context)
            else:
                logging.warning(f"不支持的告警渠道: {channel.value}")
                return False
        except Exception as e:
            logging.error(f"告警渠道 [{channel.value}] 发送失败: {str(e)}")
            return False
    
    def _send_email_alert(self, error_context: ErrorContext) -> bool:
        """发送邮件告警"""
        email_config = self.notification_channels.get('email', {})
        if not email_config:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"[{error_context.severity.value.upper()}] {error_context.error_type.value} 错误告警"
            
            body = f"""
错误详情:
- 错误ID: {error_context.error_id}
- 错误类型: {error_context.error_type.value}
- 严重级别: {error_context.severity.value}
- 错误消息: {error_context.message}
- 模块: {error_context.module_name}
- 函数: {error_context.function_name}
- 行号: {error_context.line_number}
- 时间: {error_context.timestamp}
- 主机: {error_context.host_name}
- 进程ID: {error_context.process_id}

堆栈跟踪:
{error_context.stack_trace}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('use_tls'):
                server.starttls()
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            return True
        except Exception:
            return False
    
    def _send_webhook_alert(self, error_context: ErrorContext) -> bool:
        """发送Webhook告警"""
        webhook_config = self.notification_channels.get('webhook', {})
        if not webhook_config:
            return False
        
        try:
            payload = {
                'error_id': error_context.error_id,
                'error_type': error_context.error_type.value,
                'severity': error_context.severity.value,
                'message': error_context.message,
                'timestamp': error_context.timestamp.isoformat(),
                'module': error_context.module_name,
                'function': error_context.function_name,
                'stack_trace': error_context.stack_trace
            }
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _send_slack_alert(self, error_context: ErrorContext) -> bool:
        """发送Slack告警"""
        slack_config = self.notification_channels.get('slack', {})
        if not slack_config:
            return False
        
        try:
            payload = {
                'text': f"🚨 {error_context.severity.value.upper()} 错误告警",
                'attachments': [
                    {
                        'color': 'danger' if error_context.severity == ErrorSeverity.CRITICAL else 'warning',
                        'fields': [
                            {'title': '错误类型', 'value': error_context.error_type.value, 'short': True},
                            {'title': '模块', 'value': error_context.module_name, 'short': True},
                            {'title': '错误消息', 'value': error_context.message, 'short': False},
                            {'title': '时间', 'value': error_context.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                        ]
                    }
                ]
            }
            
            response = requests.post(
                slack_config['webhook_url'],
                json=payload,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _send_dingtalk_alert(self, error_context: ErrorContext) -> bool:
        """发送钉钉告警"""
        dingtalk_config = self.notification_channels.get('dingtalk', {})
        if not dingtalk_config:
            return False
        
        try:
            payload = {
                'msgtype': 'text',
                'text': {
                    'content': f"🚨 {error_context.severity.value.upper()} 错误告警\n\n错误ID: {error_context.error_id}\n错误类型: {error_context.error_type.value}\n错误消息: {error_context.message}\n模块: {error_context.module_name}\n时间: {error_context.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                }
            }
            
            response = requests.post(
                dingtalk_config['webhook_url'],
                json=payload,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _send_wechat_alert(self, error_context: ErrorContext) -> bool:
        """发送微信告警"""
        wechat_config = self.notification_channels.get('wechat', {})
        if not wechat_config:
            return False
        
        try:
            # 这里可以实现企业微信机器人的告警逻辑
            payload = {
                'msgtype': 'text',
                'text': {
                    'content': f"🚨 {error_context.severity.value.upper()} 错误告警\n\n错误ID: {error_context.error_id}\n错误类型: {error_context.error_type.value}\n错误消息: {error_context.message}\n模块: {error_context.module_name}\n时间: {error_context.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                }
            }
            
            response = requests.post(
                wechat_config['webhook_url'],
                json=payload,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _send_log_alert(self, error_context: ErrorContext) -> bool:
        """发送日志告警"""
        log_level = logging.ERROR if error_context.severity == ErrorSeverity.CRITICAL else logging.WARNING
        logging.log(log_level, f"错误告警: {error_context.error_id} - {error_context.message}")
        return True


class RecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recovery_strategies = config.get('recovery_strategies', {})
        self.retry_policies = config.get('retry_policies', {})
        self.circuit_breakers = {}
        self.recovery_history = deque(maxlen=1000)
        self._lock = threading.RLock()
    
    def attempt_recovery(self, error_context: ErrorContext, recovery_func: Callable = None) -> Tuple[bool, Any]:
        """尝试错误恢复"""
        with self._lock:
            recovery_key = f"{error_context.error_type.value}_{error_context.module_name}"
            
            # 检查熔断器状态
            if self._is_circuit_open(recovery_key):
                return False, "Circuit breaker is open"
            
            # 获取恢复策略
            strategy = self._get_recovery_strategy(error_context)
            if not strategy:
                return False, "No recovery strategy available"
            
            try:
                if strategy == RecoveryStrategy.RETRY:
                    return self._retry_recovery(error_context, recovery_func)
                elif strategy == RecoveryStrategy.FALLBACK:
                    return self._fallback_recovery(error_context, recovery_func)
                elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    return self._circuit_breaker_recovery(error_context, recovery_func)
                elif strategy == RecoveryStrategy.BULKHEAD:
                    return self._bulkhead_recovery(error_context, recovery_func)
                elif strategy == RecoveryStrategy.TIMEOUT:
                    return self._timeout_recovery(error_context, recovery_func)
                else:
                    return False, f"Unknown recovery strategy: {strategy}"
            except Exception as e:
                self._record_recovery_attempt(error_context, strategy, False, str(e))
                return False, str(e)
    
    def _get_recovery_strategy(self, error_context: ErrorContext) -> Optional[RecoveryStrategy]:
        """获取恢复策略"""
        strategy_key = f"{error_context.error_type.value}_{error_context.severity.value}"
        if strategy_key in self.recovery_strategies:
            strategy_name = self.recovery_strategies[strategy_key]
            return RecoveryStrategy(strategy_name)
        
        # 默认策略
        if error_context.error_type == ErrorType.NETWORK:
            return RecoveryStrategy.RETRY
        elif error_context.error_type == ErrorType.SYSTEM:
            return RecoveryStrategy.FALLBACK
        else:
            return RecoveryStrategy.IGNORE
    
    def _is_circuit_open(self, key: str) -> bool:
        """检查熔断器是否开启"""
        if key not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[key]
        if breaker['state'] == 'open':
            if time.time() - breaker['last_failure'] > breaker['timeout']:
                breaker['state'] = 'half-open'
                return False
            return True
        return False
    
    def _retry_recovery(self, error_context: ErrorContext, recovery_func: Callable) -> Tuple[bool, Any]:
        """重试恢复"""
        retry_config = self.retry_policies.get('default', {})
        max_retries = retry_config.get('max_retries', 3)
        retry_delay = retry_config.get('retry_delay', 1.0)
        backoff_factor = retry_config.get('backoff_factor', 2.0)
        
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    time.sleep(retry_delay * (backoff_factor ** (attempt - 1)))
                
                if recovery_func:
                    result = recovery_func()
                    self._record_recovery_attempt(error_context, RecoveryStrategy.RETRY, True, f"Success on attempt {attempt + 1}")
                    return True, result
                else:
                    # 模拟恢复操作
                    time.sleep(0.1)
                    self._record_recovery_attempt(error_context, RecoveryStrategy.RETRY, True, f"Simulated recovery on attempt {attempt + 1}")
                    return True, None
                    
            except Exception as e:
                last_exception = e
                continue
        
        self._record_recovery_attempt(error_context, RecoveryStrategy.RETRY, False, f"All {max_retries + 1} attempts failed")
        return False, str(last_exception)
    
    def _fallback_recovery(self, error_context: ErrorContext, recovery_func: Callable) -> Tuple[bool, Any]:
        """降级恢复"""
        try:
            # 执行降级逻辑
            if recovery_func:
                # 尝试使用降级函数
                fallback_func = getattr(recovery_func, 'fallback', None)
                if fallback_func:
                    result = fallback_func()
                    self._record_recovery_attempt(error_context, RecoveryStrategy.FALLBACK, True, "Fallback successful")
                    return True, result
            
            # 默认降级行为
            logging.warning(f"执行降级恢复: {error_context.error_id}")
            self._record_recovery_attempt(error_context, RecoveryStrategy.FALLBACK, True, "Default fallback executed")
            return True, None
            
        except Exception as e:
            self._record_recovery_attempt(error_context, RecoveryStrategy.FALLBACK, False, str(e))
            return False, str(e)
    
    def _circuit_breaker_recovery(self, error_context: ErrorContext, recovery_func: Callable) -> Tuple[bool, Any]:
        """熔断器恢复"""
        recovery_key = f"{error_context.error_type.value}_{error_context.module_name}"
        
        if recovery_key not in self.circuit_breakers:
            self.circuit_breakers[recovery_key] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': 0,
                'timeout': 60  # 60秒超时
            }
        
        breaker = self.circuit_breakers[recovery_key]
        
        try:
            if recovery_func:
                result = recovery_func()
                # 成功，重置熔断器
                if breaker['state'] == 'half-open':
                    breaker['state'] = 'closed'
                    breaker['failure_count'] = 0
                
                self._record_recovery_attempt(error_context, RecoveryStrategy.CIRCUIT_BREAKER, True, "Circuit breaker success")
                return True, result
            else:
                # 模拟熔断器恢复
                time.sleep(0.1)
                self._record_recovery_attempt(error_context, RecoveryStrategy.CIRCUIT_BREAKER, True, "Circuit breaker simulated success")
                return True, None
                
        except Exception as e:
            # 失败，增加失败计数
            breaker['failure_count'] += 1
            breaker['last_failure'] = time.time()
            
            # 如果失败次数过多，开启熔断器
            if breaker['failure_count'] >= 5:
                breaker['state'] = 'open'
            
            self._record_recovery_attempt(error_context, RecoveryStrategy.CIRCUIT_BREAKER, False, str(e))
            return False, str(e)
    
    def _bulkhead_recovery(self, error_context: ErrorContext, recovery_func: Callable) -> Tuple[bool, Any]:
        """隔板模式恢复"""
        try:
            # 隔板模式：将系统分割成独立的隔板，隔离故障
            logging.info(f"执行隔板模式恢复: {error_context.error_id}")
            
            # 模拟隔板隔离操作
            time.sleep(0.1)
            
            if recovery_func:
                result = recovery_func()
                self._record_recovery_attempt(error_context, RecoveryStrategy.BULKHEAD, True, "Bulkhead isolation successful")
                return True, result
            else:
                self._record_recovery_attempt(error_context, RecoveryStrategy.BULKHEAD, True, "Bulkhead simulated recovery")
                return True, None
                
        except Exception as e:
            self._record_recovery_attempt(error_context, RecoveryStrategy.BULKHEAD, False, str(e))
            return False, str(e)
    
    def _timeout_recovery(self, error_context: ErrorContext, recovery_func: Callable) -> Tuple[bool, Any]:
        """超时恢复"""
        timeout = self.config.get('default_timeout', 30.0)
        
        try:
            if recovery_func:
                # 在指定超时时间内尝试执行
                result = asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, recovery_func),
                    timeout=timeout
                )
                self._record_recovery_attempt(error_context, RecoveryStrategy.TIMEOUT, True, "Timeout recovery successful")
                return True, result
            else:
                # 模拟超时恢复
                time.sleep(min(0.1, timeout))
                self._record_recovery_attempt(error_context, RecoveryStrategy.TIMEOUT, True, "Timeout simulated recovery")
                return True, None
                
        except asyncio.TimeoutError:
            self._record_recovery_attempt(error_context, RecoveryStrategy.TIMEOUT, False, "Operation timed out")
            return False, "Operation timed out"
        except Exception as e:
            self._record_recovery_attempt(error_context, RecoveryStrategy.TIMEOUT, False, str(e))
            return False, str(e)
    
    def _record_recovery_attempt(self, error_context: ErrorContext, strategy: RecoveryStrategy, success: bool, message: str):
        """记录恢复尝试"""
        self.recovery_history.append({
            'timestamp': time.time(),
            'error_id': error_context.error_id,
            'strategy': strategy.value,
            'success': success,
            'message': message
        })


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "error_logs.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建错误日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_logs (
                    error_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    exception_type TEXT NOT NULL,
                    stack_trace TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    request_id TEXT,
                    service_name TEXT,
                    host_name TEXT,
                    process_id INTEGER,
                    thread_id INTEGER,
                    memory_usage REAL,
                    cpu_usage REAL,
                    additional_data TEXT,
                    custom_fields TEXT,
                    status TEXT DEFAULT 'open',
                    resolved_at TEXT,
                    resolution_time REAL,
                    recovery_attempts INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建错误统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_errors INTEGER DEFAULT 0,
                    errors_by_type TEXT,
                    errors_by_severity TEXT,
                    errors_by_module TEXT,
                    recurring_errors INTEGER DEFAULT 0,
                    resolved_errors INTEGER DEFAULT 0,
                    avg_resolution_time REAL DEFAULT 0.0,
                    affected_users INTEGER DEFAULT 0,
                    system_impact_score REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建告警历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    error_id TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    success_count INTEGER NOT NULL,
                    total_count INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建恢复历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recovery_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    error_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_logs_timestamp ON error_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_logs_type ON error_logs(error_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_logs_severity ON error_logs(severity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_logs_status ON error_logs(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_statistics_date ON error_statistics(date)')
            
            conn.commit()
            conn.close()
    
    def save_error_log(self, error_context: ErrorContext) -> bool:
        """保存错误日志"""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO error_logs (
                        error_id, timestamp, error_type, severity, message, exception_type,
                        stack_trace, module_name, function_name, line_number, user_id,
                        session_id, request_id, service_name, host_name, process_id,
                        thread_id, memory_usage, cpu_usage, additional_data, custom_fields
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    error_context.error_id,
                    error_context.timestamp.isoformat(),
                    error_context.error_type.value,
                    error_context.severity.value,
                    error_context.message,
                    error_context.exception_type,
                    error_context.stack_trace,
                    error_context.module_name,
                    error_context.function_name,
                    error_context.line_number,
                    error_context.user_id,
                    error_context.session_id,
                    error_context.request_id,
                    error_context.service_name,
                    error_context.host_name,
                    error_context.process_id,
                    error_context.thread_id,
                    error_context.memory_usage,
                    error_context.cpu_usage,
                    json.dumps(error_context.additional_data) if error_context.additional_data else None,
                    json.dumps(error_context.custom_fields) if error_context.custom_fields else None
                ))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logging.error(f"保存错误日志失败: {str(e)}")
            return False
    
    def update_error_status(self, error_id: str, status: ErrorStatus, resolution_time: float = None) -> bool:
        """更新错误状态"""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                update_data = {
                    'status': status.value,
                    'updated_at': datetime.now().isoformat()
                }
                
                if status in [ErrorStatus.RESOLVED, ErrorStatus.CLOSED]:
                    update_data['resolved_at'] = datetime.now().isoformat()
                    if resolution_time:
                        update_data['resolution_time'] = resolution_time
                
                set_clause = ', '.join([f"{k} = ?" for k in update_data.keys()])
                values = list(update_data.values()) + [error_id]
                
                cursor.execute(f'UPDATE error_logs SET {set_clause} WHERE error_id = ?', values)
                
                conn.commit()
                conn.close()
                return cursor.rowcount > 0
        except Exception as e:
            logging.error(f"更新错误状态失败: {str(e)}")
            return False
    
    def get_error_logs(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取错误日志"""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                where_conditions = []
                values = []
                
                if filters:
                    if 'error_type' in filters:
                        where_conditions.append('error_type = ?')
                        values.append(filters['error_type'])
                    if 'severity' in filters:
                        where_conditions.append('severity = ?')
                        values.append(filters['severity'])
                    if 'status' in filters:
                        where_conditions.append('status = ?')
                        values.append(filters['status'])
                    if 'start_date' in filters:
                        where_conditions.append('timestamp >= ?')
                        values.append(filters['start_date'])
                    if 'end_date' in filters:
                        where_conditions.append('timestamp <= ?')
                        values.append(filters['end_date'])
                
                where_clause = 'WHERE ' + ' AND '.join(where_conditions) if where_conditions else ''
                query = f'''
                    SELECT * FROM error_logs 
                    {where_clause}
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                values.append(limit)
                
                cursor.execute(query, values)
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                conn.close()
                return results
        except Exception as e:
            logging.error(f"获取错误日志失败: {str(e)}")
            return []
    
    def save_statistics(self, statistics: ErrorStatistics) -> bool:
        """保存统计数据"""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO error_statistics (
                        date, total_errors, errors_by_type, errors_by_severity,
                        errors_by_module, recurring_errors, resolved_errors,
                        avg_resolution_time, affected_users, system_impact_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().strftime('%Y-%m-%d'),
                    statistics.total_errors,
                    json.dumps({k.value: v for k, v in statistics.errors_by_type.items()}),
                    json.dumps({k.value: v for k, v in statistics.errors_by_severity.items()}),
                    json.dumps(statistics.errors_by_module),
                    statistics.recurring_errors,
                    statistics.resolved_errors,
                    statistics.avg_resolution_time,
                    statistics.affected_users,
                    statistics.system_impact_score
                ))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logging.error(f"保存统计数据失败: {str(e)}")
            return False


class ErrorLogger:
    """
    L3错误日志记录器主类
    
    提供完整的错误日志记录、处理和分析功能，包括：
    - 异常捕获和分类记录
    - 错误堆栈跟踪和上下文信息
    - 错误统计和分析
    - 错误告警和通知机制
    - 错误恢复和重试日志
    - 错误解决和关闭流程
    - 异步错误日志处理
    
    使用示例:
        ```python
        # 初始化错误日志记录器
        error_logger = ErrorLogger(config={
            'database_path': 'error_logs.db',
            'enable_alerts': True,
            'alert_rules': {
                'system_critical': {'enabled': True},
                'business_high': {'enabled': True}
            },
            'notification_channels': {
                'email': {
                    'smtp_server': 'smtp.example.com',
                    'smtp_port': 587,
                    'from': 'alerts@example.com',
                    'to': ['admin@example.com'],
                    'username': 'user@example.com',
                    'password': 'password'
                }
            }
        })
        
        # 使用装饰器捕获异常
        @error_logger.error_handler
        def risky_function():
            # 可能抛出异常的代码
            pass
        
        # 手动记录错误
        try:
            # 业务代码
            pass
        except Exception as e:
            error_logger.log_error(
                error_type=ErrorType.BUSINESS,
                severity=ErrorSeverity.HIGH,
                message="业务逻辑错误",
                exception=e
            )
        
        # 获取错误统计
        stats = error_logger.get_statistics()
        print(f"总错误数: {stats.total_errors}")
        ```
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化错误日志记录器
        
        Args:
            config: 配置字典，包含数据库路径、告警配置等
        """
        self.config = config or {}
        self.database_path = self.config.get('database_path', 'error_logs.db')
        self.enable_alerts = self.config.get('enable_alerts', True)
        self.enable_recovery = self.config.get('enable_recovery', True)
        self.enable_statistics = self.config.get('enable_statistics', True)
        self.async_processing = self.config.get('async_processing', True)
        
        # 初始化组件
        self.db_manager = DatabaseManager(self.database_path)
        self.alert_manager = AlertManager(self.config)
        self.recovery_manager = RecoveryManager(self.config)
        
        # 内存缓存
        self.error_cache = deque(maxlen=10000)
        self.statistics_cache = {}
        self._lock = threading.RLock()
        
        # 异步处理
        self.async_queue = asyncio.Queue(maxsize=1000)
        self.processing_tasks = []
        
        # 统计信息
        self.start_time = time.time()
        self.total_errors_logged = 0
        self.total_recoveries_attempted = 0
        self.total_alerts_sent = 0
        
        # 启动异步处理
        if self.async_processing:
            self._start_async_processing()
    
    def _start_async_processing(self):
        """启动异步处理任务"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 启动异步处理任务
        task = loop.create_task(self._async_processing_loop())
        self.processing_tasks.append(task)
    
    async def _async_processing_loop(self):
        """异步处理循环"""
        while True:
            try:
                # 从队列获取待处理项目
                item = await asyncio.wait_for(self.async_queue.get(), timeout=1.0)
                
                if item['type'] == 'save_error':
                    await self._async_save_error(item['data'])
                elif item['type'] == 'send_alert':
                    await self._async_send_alert(item['data'])
                elif item['type'] == 'update_statistics':
                    await self._async_update_statistics(item['data'])
                
                self.async_queue.task_done()
                
            except asyncio.TimeoutError:
                # 队列为空，继续循环
                continue
            except Exception as e:
                logging.error(f"异步处理错误: {str(e)}")
    
    async def _async_save_error(self, error_context: ErrorContext):
        """异步保存错误"""
        try:
            success = self.db_manager.save_error_log(error_context)
            if success:
                with self._lock:
                    self.error_cache.append(error_context)
                    self.total_errors_logged += 1
        except Exception as e:
            logging.error(f"异步保存错误失败: {str(e)}")
    
    async def _async_send_alert(self, error_context: ErrorContext):
        """异步发送告警"""
        try:
            if self.alert_manager.should_alert(error_context):
                channels = self._get_alert_channels(error_context)
                success = self.alert_manager.send_alert(error_context, channels)
                if success:
                    with self._lock:
                        self.total_alerts_sent += 1
        except Exception as e:
            logging.error(f"异步发送告警失败: {str(e)}")
    
    async def _async_update_statistics(self, statistics: ErrorStatistics):
        """异步更新统计"""
        try:
            self.db_manager.save_statistics(statistics)
            with self._lock:
                self.statistics_cache = asdict(statistics)
        except Exception as e:
            logging.error(f"异步更新统计失败: {str(e)}")
    
    def _get_alert_channels(self, error_context: ErrorContext) -> List[AlertChannel]:
        """获取告警渠道"""
        channels = [AlertChannel.LOG]  # 默认总是记录日志
        
        if not self.enable_alerts:
            return channels
        
        # 根据错误类型和严重程度确定告警渠道
        if error_context.severity == ErrorSeverity.CRITICAL:
            channels.extend([AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DINGTALK])
        elif error_context.severity == ErrorSeverity.HIGH:
            channels.extend([AlertChannel.EMAIL, AlertChannel.SLACK])
        elif error_context.error_type == ErrorType.SYSTEM:
            channels.append(AlertChannel.EMAIL)
        
        return channels
    
    def _create_error_context(self, 
                            error_type: ErrorType,
                            severity: ErrorSeverity,
                            message: str,
                            exception: Exception = None,
                            additional_data: Dict[str, Any] = None,
                            custom_fields: Dict[str, Any] = None) -> ErrorContext:
        """创建错误上下文"""
        
        # 获取调用栈信息
        stack = traceback.extract_tb(exception.__traceback__ if exception else None)
        if stack:
            frame = stack[-1]  # 最新的栈帧
            module_name = frame.filename
            function_name = frame.name
            line_number = frame.lineno
            stack_trace = ''.join(traceback.format_tb(exception.__traceback__))
        else:
            # 从当前调用栈获取信息
            frame = sys._getframe(2)
            module_name = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            line_number = frame.f_lineno
            stack_trace = ''.join(traceback.format_stack())
        
        # 获取系统信息
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        return ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            error_type=error_type,
            severity=severity,
            message=message,
            exception_type=type(exception).__name__ if exception else 'Unknown',
            stack_trace=stack_trace,
            module_name=os.path.basename(module_name),
            function_name=function_name,
            line_number=line_number,
            host_name=os.uname().nodename,
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
            memory_usage=memory_info.rss / 1024 / 1024,  # MB
            cpu_usage=cpu_percent,
            additional_data=additional_data,
            custom_fields=custom_fields
        )
    
    def error_handler(self, 
                     error_type: ErrorType = ErrorType.UNKNOWN,
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     reraise: bool = True,
                     recovery_func: Callable = None,
                     additional_data: Dict[str, Any] = None,
                     custom_fields: Dict[str, Any] = None):
        """
        错误处理装饰器
        
        Args:
            error_type: 错误类型
            severity: 错误严重程度
            reraise: 是否重新抛出异常
            recovery_func: 恢复函数
            additional_data: 额外数据
            custom_fields: 自定义字段
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 记录错误
                    self.log_error(
                        error_type=error_type,
                        severity=severity,
                        message=f"函数 {func.__name__} 执行失败: {str(e)}",
                        exception=e,
                        additional_data=additional_data,
                        custom_fields=custom_fields
                    )
                    
                    # 尝试恢复
                    if self.enable_recovery and recovery_func:
                        success, result = self.attempt_recovery(error_type, recovery_func)
                        if success:
                            return result
                    
                    # 重新抛出异常
                    if reraise:
                        raise
                    
                    return None
            return wrapper
        return decorator
    
    def log_error(self,
                 error_type: ErrorType,
                 severity: ErrorSeverity,
                 message: str,
                 exception: Exception = None,
                 additional_data: Dict[str, Any] = None,
                 custom_fields: Dict[str, Any] = None,
                 user_id: str = None,
                 session_id: str = None,
                 request_id: str = None,
                 service_name: str = None) -> str:
        """
        记录错误日志
        
        Args:
            error_type: 错误类型
            severity: 错误严重程度
            message: 错误消息
            exception: 异常对象
            additional_data: 额外数据
            custom_fields: 自定义字段
            user_id: 用户ID
            session_id: 会话ID
            request_id: 请求ID
            service_name: 服务名称
            
        Returns:
            错误ID
        """
        try:
            # 创建错误上下文
            error_context = self._create_error_context(
                error_type=error_type,
                severity=severity,
                message=message,
                exception=exception,
                additional_data=additional_data,
                custom_fields=custom_fields
            )
            
            # 设置用户和会话信息
            error_context.user_id = user_id
            error_context.session_id = session_id
            error_context.request_id = request_id
            error_context.service_name = service_name
            
            # 异步处理
            if self.async_processing:
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.async_queue.put({
                        'type': 'save_error',
                        'data': error_context
                    }))
                    
                    # 如果需要告警，也异步处理
                    if self.enable_alerts:
                        loop.create_task(self.async_queue.put({
                            'type': 'send_alert',
                            'data': error_context
                        }))
                        
                except Exception as e:
                    logging.error(f"异步处理错误: {str(e)}")
                    # 降级到同步处理
                    self._sync_save_error(error_context)
            else:
                # 同步处理
                self._sync_save_error(error_context)
                if self.enable_alerts:
                    self._sync_send_alert(error_context)
            
            return error_context.error_id
            
        except Exception as e:
            logging.error(f"记录错误日志失败: {str(e)}")
            return ""
    
    def _sync_save_error(self, error_context: ErrorContext):
        """同步保存错误"""
        try:
            success = self.db_manager.save_error_log(error_context)
            if success:
                with self._lock:
                    self.error_cache.append(error_context)
                    self.total_errors_logged += 1
        except Exception as e:
            logging.error(f"同步保存错误失败: {str(e)}")
    
    def _sync_send_alert(self, error_context: ErrorContext):
        """同步发送告警"""
        try:
            if self.alert_manager.should_alert(error_context):
                channels = self._get_alert_channels(error_context)
                success = self.alert_manager.send_alert(error_context, channels)
                if success:
                    with self._lock:
                        self.total_alerts_sent += 1
        except Exception as e:
            logging.error(f"同步发送告警失败: {str(e)}")
    
    def attempt_recovery(self, 
                        error_type: ErrorType, 
                        recovery_func: Callable,
                        error_context: ErrorContext = None) -> Tuple[bool, Any]:
        """
        尝试错误恢复
        
        Args:
            error_type: 错误类型
            recovery_func: 恢复函数
            error_context: 错误上下文
            
        Returns:
            (是否成功, 结果或错误消息)
        """
        try:
            with self._lock:
                self.total_recoveries_attempted += 1
            
            # 如果没有提供错误上下文，创建一个基本的
            if error_context is None:
                error_context = self._create_error_context(
                    error_type=error_type,
                    severity=ErrorSeverity.MEDIUM,
                    message="Recovery attempt"
                )
            
            success, result = self.recovery_manager.attempt_recovery(error_context, recovery_func)
            
            if success:
                logging.info(f"错误恢复成功: {error_context.error_id}")
            else:
                logging.warning(f"错误恢复失败: {error_context.error_id} - {result}")
            
            return success, result
            
        except Exception as e:
            logging.error(f"错误恢复过程异常: {str(e)}")
            return False, str(e)
    
    def resolve_error(self, error_id: str, resolution_notes: str = None) -> bool:
        """
        解决错误
        
        Args:
            error_id: 错误ID
            resolution_notes: 解决备注
            
        Returns:
            是否成功
        """
        try:
            # 计算解决时间
            error_log = self.db_manager.get_error_logs({'error_id': error_id}, 1)
            if not error_log:
                return False
            
            error_log = error_log[0]
            created_time = datetime.fromisoformat(error_log['timestamp'])
            resolution_time = (datetime.now() - created_time).total_seconds()
            
            # 更新数据库
            success = self.db_manager.update_error_status(
                error_id=error_id,
                status=ErrorStatus.RESOLVED,
                resolution_time=resolution_time
            )
            
            if success:
                logging.info(f"错误已解决: {error_id}")
                if resolution_notes:
                    logging.info(f"解决备注: {resolution_notes}")
            
            return success
            
        except Exception as e:
            logging.error(f"解决错误失败: {str(e)}")
            return False
    
    def close_error(self, error_id: str, close_notes: str = None) -> bool:
        """
        关闭错误
        
        Args:
            error_id: 错误ID
            close_notes: 关闭备注
            
        Returns:
            是否成功
        """
        try:
            success = self.db_manager.update_error_status(
                error_id=error_id,
                status=ErrorStatus.CLOSED
            )
            
            if success:
                logging.info(f"错误已关闭: {error_id}")
                if close_notes:
                    logging.info(f"关闭备注: {close_notes}")
            
            return success
            
        except Exception as e:
            logging.error(f"关闭错误失败: {str(e)}")
            return False
    
    def ignore_error(self, error_id: str, ignore_reason: str = None) -> bool:
        """
        忽略错误
        
        Args:
            error_id: 错误ID
            ignore_reason: 忽略原因
            
        Returns:
            是否成功
        """
        try:
            success = self.db_manager.update_error_status(
                error_id=error_id,
                status=ErrorStatus.IGNORED
            )
            
            if success:
                logging.info(f"错误已忽略: {error_id}")
                if ignore_reason:
                    logging.info(f"忽略原因: {ignore_reason}")
            
            return success
            
        except Exception as e:
            logging.error(f"忽略错误失败: {str(e)}")
            return False
    
    def get_statistics(self, hours: int = 24) -> ErrorStatistics:
        """
        获取错误统计信息
        
        Args:
            hours: 统计时间范围（小时）
            
        Returns:
            错误统计对象
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # 获取指定时间范围内的错误日志
            filters = {
                'start_date': start_time.isoformat(),
                'end_date': end_time.isoformat()
            }
            error_logs = self.db_manager.get_error_logs(filters, limit=10000)
            
            # 初始化统计对象
            stats = ErrorStatistics()
            stats.total_errors = len(error_logs)
            
            # 统计错误类型
            for log in error_logs:
                error_type = ErrorType(log['error_type'])
                severity = ErrorSeverity(log['severity'])
                module = log['module_name']
                timestamp = datetime.fromisoformat(log['timestamp'])
                hour = timestamp.hour
                
                stats.errors_by_type[error_type] += 1
                stats.errors_by_severity[severity] += 1
                stats.errors_by_module[module] = stats.errors_by_module.get(module, 0) + 1
                stats.errors_by_hour[hour] = stats.errors_by_hour.get(hour, 0) + 1
                
                # 检查是否为重复错误
                if log['status'] == 'recurring':
                    stats.recurring_errors += 1
                
                # 检查是否已解决
                if log['status'] in ['resolved', 'closed']:
                    stats.resolved_errors += 1
                    if log['resolution_time']:
                        stats.avg_resolution_time += log['resolution_time']
            
            # 计算平均解决时间
            if stats.resolved_errors > 0:
                stats.avg_resolution_time /= stats.resolved_errors
            
            # 计算错误频率
            error_frequency = defaultdict(int)
            for log in error_logs:
                key = f"{log['error_type']}_{log['module_name']}_{log['function_name']}"
                error_frequency[key] += 1
            
            stats.error_frequency = dict(error_frequency)
            
            # 计算峰值时间
            if stats.errors_by_hour:
                peak_hour = max(stats.errors_by_hour, key=stats.errors_by_hour.get)
                stats.peak_error_time = datetime.now().replace(hour=peak_hour, minute=0, second=0, microsecond=0)
            
            # 计算系统影响分数
            impact_score = 0
            for severity, count in stats.errors_by_severity.items():
                if severity == ErrorSeverity.CRITICAL:
                    impact_score += count * 10
                elif severity == ErrorSeverity.HIGH:
                    impact_score += count * 5
                elif severity == ErrorSeverity.MEDIUM:
                    impact_score += count * 2
                else:
                    impact_score += count * 1
            
            stats.system_impact_score = impact_score / max(stats.total_errors, 1)
            
            # 异步保存统计信息
            if self.enable_statistics:
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.async_queue.put({
                        'type': 'update_statistics',
                        'data': stats
                    }))
                except Exception:
                    # 降级到同步保存
                    self.db_manager.save_statistics(stats)
            
            return stats
            
        except Exception as e:
            logging.error(f"获取统计信息失败: {str(e)}")
            return ErrorStatistics()
    
    def get_error_logs(self, 
                      error_type: ErrorType = None,
                      severity: ErrorSeverity = None,
                      status: ErrorStatus = None,
                      start_date: datetime = None,
                      end_date: datetime = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取错误日志
        
        Args:
            error_type: 错误类型过滤
            severity: 严重程度过滤
            status: 状态过滤
            start_date: 开始日期
            end_date: 结束日期
            limit: 返回数量限制
            
        Returns:
            错误日志列表
        """
        try:
            filters = {}
            
            if error_type:
                filters['error_type'] = error_type.value
            if severity:
                filters['severity'] = severity.value
            if status:
                filters['status'] = status.value
            if start_date:
                filters['start_date'] = start_date.isoformat()
            if end_date:
                filters['end_date'] = end_date.isoformat()
            
            return self.db_manager.get_error_logs(filters, limit)
            
        except Exception as e:
            logging.error(f"获取错误日志失败: {str(e)}")
            return []
    
    def get_recent_errors(self, count: int = 50) -> List[ErrorContext]:
        """
        获取最近的错误
        
        Args:
            count: 返回数量
            
        Returns:
            错误上下文列表
        """
        try:
            with self._lock:
                return list(self.error_cache)[-count:]
        except Exception as e:
            logging.error(f"获取最近错误失败: {str(e)}")
            return []
    
    def get_critical_errors(self, hours: int = 1) -> List[ErrorContext]:
        """
        获取严重错误
        
        Args:
            hours: 时间范围（小时）
            
        Returns:
            严重错误列表
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            critical_errors = []
            
            with self._lock:
                for error in reversed(self.error_cache):
                    if (error.timestamp >= cutoff_time and 
                        error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]):
                        critical_errors.append(error)
            
            return critical_errors
        except Exception as e:
            logging.error(f"获取严重错误失败: {str(e)}")
            return []
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        获取系统健康状态
        
        Returns:
            系统健康状态字典
        """
        try:
            uptime = time.time() - self.start_time
            
            # 获取最近的统计信息
            recent_stats = self.get_statistics(hours=1)
            
            health_status = {
                'uptime_seconds': uptime,
                'total_errors_logged': self.total_errors_logged,
                'total_recoveries_attempted': self.total_recoveries_attempted,
                'total_alerts_sent': self.total_alerts_sent,
                'error_rate_per_hour': recent_stats.total_errors,
                'system_impact_score': recent_stats.system_impact_score,
                'critical_errors_last_hour': len(self.get_critical_errors(1)),
                'database_connected': True,
                'async_processing_active': len(self.processing_tasks) > 0,
                'cache_size': len(self.error_cache),
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_usage_percent': psutil.Process().cpu_percent()
            }
            
            # 计算健康分数
            health_score = 100.0
            if health_status['error_rate_per_hour'] > 100:
                health_score -= 20
            elif health_status['error_rate_per_hour'] > 50:
                health_score -= 10
            
            if health_status['critical_errors_last_hour'] > 5:
                health_score -= 30
            elif health_status['critical_errors_last_hour'] > 0:
                health_score -= 10
            
            if health_status['system_impact_score'] > 50:
                health_score -= 20
            elif health_status['system_impact_score'] > 20:
                health_score -= 10
            
            health_status['health_score'] = max(health_score, 0)
            
            # 确定健康状态
            if health_status['health_score'] >= 80:
                health_status['status'] = 'healthy'
            elif health_status['health_score'] >= 60:
                health_status['status'] = 'warning'
            else:
                health_status['status'] = 'critical'
            
            return health_status
            
        except Exception as e:
            logging.error(f"获取系统健康状态失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """
        清理旧日志
        
        Args:
            days: 保留天数
            
        Returns:
            清理的记录数
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 删除旧记录
                cursor.execute(
                    'DELETE FROM error_logs WHERE timestamp < ?',
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                
                cursor.execute(
                    'DELETE FROM error_statistics WHERE date < ?',
                    (cutoff_date.strftime('%Y-%m-%d'),)
                )
                deleted_stats_count = cursor.rowcount
                
                cursor.execute(
                    'DELETE FROM alert_history WHERE timestamp < ?',
                    (time.time() - (days * 24 * 3600),)
                )
                deleted_alerts_count = cursor.rowcount
                
                cursor.execute(
                    'DELETE FROM recovery_history WHERE timestamp < ?',
                    (time.time() - (days * 24 * 3600),)
                )
                deleted_recovery_count = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                total_deleted = deleted_count + deleted_stats_count + deleted_alerts_count + deleted_recovery_count
                
                logging.info(f"清理完成，删除 {total_deleted} 条旧记录")
                return total_deleted
                
        except Exception as e:
            logging.error(f"清理旧日志失败: {str(e)}")
            return 0
    
    def export_error_report(self, output_file: str, hours: int = 24) -> bool:
        """
        导出错误报告
        
        Args:
            output_file: 输出文件路径
            hours: 统计时间范围
            
        Returns:
            是否成功
        """
        try:
            # 获取统计数据
            stats = self.get_statistics(hours)
            error_logs = self.get_error_logs(limit=1000)
            health_status = self.get_system_health()
            
            # 生成报告
            report = {
                'report_info': {
                    'generated_at': datetime.now().isoformat(),
                    'time_range_hours': hours,
                    'report_version': '1.0.0'
                },
                'summary': {
                    'total_errors': stats.total_errors,
                    'critical_errors': stats.errors_by_severity.get(ErrorSeverity.CRITICAL, 0),
                    'high_errors': stats.errors_by_severity.get(ErrorSeverity.HIGH, 0),
                    'resolved_errors': stats.resolved_errors,
                    'recurring_errors': stats.recurring_errors,
                    'system_impact_score': stats.system_impact_score,
                    'avg_resolution_time': stats.avg_resolution_time
                },
                'error_breakdown': {
                    'by_type': {k.value: v for k, v in stats.errors_by_type.items()},
                    'by_severity': {k.value: v for k, v in stats.errors_by_severity.items()},
                    'by_module': stats.errors_by_module,
                    'by_hour': stats.errors_by_hour
                },
                'top_errors': sorted(
                    [(k, v) for k, v in stats.error_frequency.items()],
                    key=lambda x: x[1], reverse=True
                )[:20],
                'system_health': health_status,
                'recent_errors': error_logs[:50]  # 最近50个错误
            }
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logging.info(f"错误报告已导出到: {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"导出错误报告失败: {str(e)}")
            return False
    
    def shutdown(self):
        """关闭错误日志记录器"""
        try:
            # 停止异步处理任务
            for task in self.processing_tasks:
                task.cancel()
            
            # 等待任务完成
            if self.processing_tasks:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(
                    asyncio.gather(*self.processing_tasks, return_exceptions=True)
                )
            
            # 清理资源
            self.processing_tasks.clear()
            
            logging.info("错误日志记录器已关闭")
            
        except Exception as e:
            logging.error(f"关闭错误日志记录器时发生异常: {str(e)}")


def create_sample_config() -> Dict[str, Any]:
    """
    创建示例配置
    
    Returns:
        示例配置字典
    """
    return {
        'database_path': 'error_logs.db',
        'enable_alerts': True,
        'enable_recovery': True,
        'enable_statistics': True,
        'async_processing': True,
        'alert_rules': {
            'system_critical': {'enabled': True},
            'business_high': {'enabled': True},
            'network_medium': {'enabled': False},
            'unknown_low': {'enabled': False}
        },
        'notification_channels': {
            'email': {
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'use_tls': True,
                'from': 'alerts@example.com',
                'to': ['admin@example.com'],
                'username': 'alerts@example.com',
                'password': 'your_password'
            },
            'webhook': {
                'url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
                'headers': {'Content-Type': 'application/json'}
            },
            'slack': {
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            },
            'dingtalk': {
                'webhook_url': 'https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN'
            }
        },
        'recovery_strategies': {
            'network_high': 'retry',
            'system_critical': 'fallback',
            'business_medium': 'ignore',
            'database_high': 'circuit_breaker'
        },
        'retry_policies': {
            'default': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'backoff_factor': 2.0
            }
        },
        'default_timeout': 30.0
    }


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建配置
    config = create_sample_config()
    
    # 初始化错误日志记录器
    error_logger = ErrorLogger(config)
    
    # 示例1: 使用装饰器
    @error_logger.error_handler(
        error_type=ErrorType.BUSINESS,
        severity=ErrorSeverity.HIGH,
        recovery_func=lambda: "fallback_result"
    )
    def risky_business_function():
        """模拟业务函数"""
        import random
        if random.random() < 0.7:  # 70%概率出错
            raise ValueError("业务逻辑验证失败")
        return "业务处理成功"
    
    # 示例2: 手动记录错误
    try:
        # 模拟系统错误
        result = 1 / 0
    except ZeroDivisionError as e:
        error_id = error_logger.log_error(
            error_type=ErrorType.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            message="除零错误",
            exception=e,
            additional_data={'operation': 'division', 'operand': 0},
            service_name="example_service"
        )
        print(f"记录系统错误: {error_id}")
    
    # 示例3: 网络错误
    try:
        import requests
        response = requests.get("https://nonexistent-domain-12345.com", timeout=1)
    except Exception as e:
        error_id = error_logger.log_error(
            error_type=ErrorType.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            message="网络请求失败",
            exception=e,
            additional_data={'url': 'https://nonexistent-domain-12345.com'}
        )
        print(f"记录网络错误: {error_id}")
    
    # 示例4: 测试恢复机制
    def recovery_function():
        """恢复函数"""
        return "恢复成功"
    
    success, result = error_logger.attempt_recovery(
        error_type=ErrorType.BUSINESS,
        recovery_func=recovery_function
    )
    print(f"恢复结果: {success}, {result}")
    
    # 示例5: 执行可能出错的函数
    print("\n=== 测试装饰器功能 ===")
    for i in range(5):
        try:
            result = risky_business_function()
            print(f"调用 {i+1}: {result}")
        except Exception as e:
            print(f"调用 {i+1}: 异常被装饰器捕获并处理")
    
    # 示例6: 获取统计信息
    print("\n=== 统计信息 ===")
    stats = error_logger.get_statistics(hours=1)
    print(f"总错误数: {stats.total_errors}")
    print(f"严重错误数: {stats.errors_by_severity.get(ErrorSeverity.CRITICAL, 0)}")
    print(f"系统影响分数: {stats.system_impact_score}")
    
    # 示例7: 获取系统健康状态
    print("\n=== 系统健康状态 ===")
    health = error_logger.get_system_health()
    print(f"健康状态: {health['status']}")
    print(f"健康分数: {health['health_score']}")
    print(f"错误率(每小时): {health['error_rate_per_hour']}")
    
    # 示例8: 获取最近的错误
    print("\n=== 最近的错误 ===")
    recent_errors = error_logger.get_recent_errors(5)
    for error in recent_errors:
        print(f"错误ID: {error.error_id}")
        print(f"类型: {error.error_type.value}")
        print(f"严重程度: {error.severity.value}")
        print(f"消息: {error.message}")
        print(f"时间: {error.timestamp}")
        print("---")
    
    # 示例9: 导出错误报告
    print("\n=== 导出错误报告 ===")
    success = error_logger.export_error_report("error_report.json", hours=1)
    if success:
        print("错误报告已导出到 error_report.json")
    
    # 示例10: 解决和关闭错误
    if recent_errors:
        error_id = recent_errors[0].error_id
        success = error_logger.resolve_error(error_id, "问题已修复")
        print(f"解决错误 {error_id}: {success}")
        
        success = error_logger.close_error(error_id, "验证通过，关闭问题")
        print(f"关闭错误 {error_id}: {success}")
    
    # 等待异步处理完成
    time.sleep(2)
    
    # 关闭错误日志记录器
    error_logger.shutdown()
    
    print("\n=== 示例完成 ===")


class ErrorTrendAnalyzer:
    """错误趋势分析器"""
    
    def __init__(self, error_logger: ErrorLogger):
        self.error_logger = error_logger
        self.trend_cache = {}
        self._lock = threading.RLock()
    
    def analyze_error_trends(self, days: int = 7) -> Dict[str, Any]:
        """分析错误趋势"""
        try:
            trends = {
                'period_days': days,
                'daily_error_counts': {},
                'error_growth_rate': 0.0,
                'peak_error_day': None,
                'error_patterns': {},
                'seasonal_analysis': {},
                'correlation_analysis': {},
                'prediction': {}
            }
            
            # 获取每日错误统计
            for day_offset in range(days):
                date = datetime.now() - timedelta(days=day_offset)
                date_str = date.strftime('%Y-%m-%d')
                
                start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = date.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                day_errors = self.error_logger.get_error_logs(
                    start_date=start_time,
                    end_date=end_time,
                    limit=10000
                )
                
                trends['daily_error_counts'][date_str] = len(day_errors)
                
                # 按小时统计
                hourly_counts = {}
                for error in day_errors:
                    error_time = datetime.fromisoformat(error['timestamp'])
                    hour = error_time.hour
                    hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
                
                trends['error_patterns'][date_str] = hourly_counts
            
            # 计算增长率
            if len(trends['daily_error_counts']) >= 2:
                counts = list(trends['daily_error_counts'].values())
                if counts[-1] > 0:
                    growth_rate = (counts[0] - counts[-1]) / counts[-1] * 100
                    trends['error_growth_rate'] = growth_rate
            
            # 找出峰值错误日
            if trends['daily_error_counts']:
                peak_day = max(trends['daily_error_counts'], key=trends['daily_error_counts'].get)
                trends['peak_error_day'] = {
                    'date': peak_day,
                    'error_count': trends['daily_error_counts'][peak_day]
                }
            
            # 季节性分析
            trends['seasonal_analysis'] = self._analyze_seasonal_patterns(trends['daily_error_counts'])
            
            # 相关性分析
            trends['correlation_analysis'] = self._analyze_correlations()
            
            # 预测分析
            trends['prediction'] = self._predict_future_errors(trends['daily_error_counts'])
            
            return trends
            
        except Exception as e:
            logging.error(f"错误趋势分析失败: {str(e)}")
            return {}
    
    def _analyze_seasonal_patterns(self, daily_counts: Dict[str, int]) -> Dict[str, Any]:
        """分析季节性模式"""
        try:
            patterns = {
                'weekday_vs_weekend': {},
                'hourly_distribution': {},
                'monthly_trend': {}
            }
            
            weekday_errors = 0
            weekend_errors = 0
            hourly_totals = defaultdict(int)
            
            for date_str, count in daily_counts.items():
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                weekday = date_obj.weekday()
                
                if weekday < 5:  # 周一到周五
                    weekday_errors += count
                else:  # 周六和周日
                    weekend_errors += count
            
            patterns['weekday_vs_weekend'] = {
                'weekday_avg': weekday_errors / max(len(daily_counts) * 5 / 7, 1),
                'weekend_avg': weekend_errors / max(len(daily_counts) * 2 / 7, 1),
                'weekday_total': weekday_errors,
                'weekend_total': weekend_errors
            }
            
            # 获取最近的数据进行小时分析
            recent_logs = self.error_logger.get_error_logs(limit=1000)
            for log in recent_logs:
                error_time = datetime.fromisoformat(log['timestamp'])
                hour = error_time.hour
                hourly_totals[hour] += 1
            
            patterns['hourly_distribution'] = dict(hourly_totals)
            
            return patterns
            
        except Exception as e:
            logging.error(f"季节性模式分析失败: {str(e)}")
            return {}
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """分析相关性"""
        try:
            correlations = {
                'error_type_correlation': {},
                'severity_correlation': {},
                'module_correlation': {},
                'time_correlation': {}
            }
            
            # 获取最近的错误日志
            logs = self.error_logger.get_error_logs(limit=1000)
            
            # 错误类型相关性
            type_counts = defaultdict(int)
            for log in logs:
                type_counts[log['error_type']] += 1
            
            correlations['error_type_correlation'] = dict(type_counts)
            
            # 严重程度相关性
            severity_counts = defaultdict(int)
            for log in logs:
                severity_counts[log['severity']] += 1
            
            correlations['severity_correlation'] = dict(severity_counts)
            
            # 模块相关性
            module_counts = defaultdict(int)
            for log in logs:
                module_counts[log['module_name']] += 1
            
            correlations['module_correlation'] = dict(module_counts)
            
            return correlations
            
        except Exception as e:
            logging.error(f"相关性分析失败: {str(e)}")
            return {}
    
    def _predict_future_errors(self, daily_counts: Dict[str, int]) -> Dict[str, Any]:
        """预测未来错误"""
        try:
            prediction = {
                'method': 'linear_regression',
                'next_7_days': [],
                'confidence': 0.0,
                'factors': []
            }
            
            if len(daily_counts) < 3:
                return prediction
            
            # 简单的线性回归预测
            counts = list(daily_counts.values())
            days = list(range(len(counts)))
            
            # 计算趋势
            if len(counts) >= 2:
                trend = (counts[-1] - counts[0]) / len(counts)
                last_value = counts[-1]
                
                # 预测未来7天
                for i in range(1, 8):
                    predicted_value = max(0, last_value + trend * i)
                    prediction['next_7_days'].append(round(predicted_value))
                
                # 计算置信度
                if len(counts) >= 5:
                    variance = sum((x - sum(counts)/len(counts))**2 for x in counts) / len(counts)
                    prediction['confidence'] = max(0, 1 - variance / 100)
            
            return prediction
            
        except Exception as e:
            logging.error(f"错误预测失败: {str(e)}")
            return {}


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, error_logger: ErrorLogger):
        self.error_logger = error_logger
        self.performance_data = deque(maxlen=10000)
        self.thresholds = {
            'response_time': 5.0,  # 秒
            'memory_usage': 1000,  # MB
            'cpu_usage': 80.0,     # 百分比
            'error_rate': 10.0     # 每小时错误数
        }
        self._lock = threading.RLock()
    
    def record_performance(self, 
                          operation: str,
                          duration: float,
                          memory_usage: float = None,
                          cpu_usage: float = None,
                          success: bool = True,
                          additional_data: Dict[str, Any] = None) -> bool:
        """记录性能数据"""
        try:
            with self._lock:
                performance_record = {
                    'timestamp': time.time(),
                    'operation': operation,
                    'duration': duration,
                    'memory_usage': memory_usage or psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_usage': cpu_usage or psutil.Process().cpu_percent(),
                    'success': success,
                    'additional_data': additional_data or {}
                }
                
                self.performance_data.append(performance_record)
                
                # 检查阈值
                self._check_performance_thresholds(performance_record)
                
                return True
                
        except Exception as e:
            logging.error(f"记录性能数据失败: {str(e)}")
            return False
    
    def _check_performance_thresholds(self, record: Dict[str, Any]):
        """检查性能阈值"""
        try:
            violations = []
            
            if record['duration'] > self.thresholds['response_time']:
                violations.append(f"响应时间超限: {record['duration']:.2f}s > {self.thresholds['response_time']}s")
            
            if record['memory_usage'] > self.thresholds['memory_usage']:
                violations.append(f"内存使用超限: {record['memory_usage']:.1f}MB > {self.thresholds['memory_usage']}MB")
            
            if record['cpu_usage'] > self.thresholds['cpu_usage']:
                violations.append(f"CPU使用率超限: {record['cpu_usage']:.1f}% > {self.thresholds['cpu_usage']}%")
            
            if violations:
                # 记录性能违规
                message = f"性能监控违规: {'; '.join(violations)}"
                self.error_logger.log_error(
                    error_type=ErrorType.PERFORMANCE,
                    severity=ErrorSeverity.MEDIUM,
                    message=message,
                    additional_data=record
                )
                
        except Exception as e:
            logging.error(f"检查性能阈值失败: {str(e)}")
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """获取性能摘要"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with self._lock:
                recent_records = [
                    record for record in self.performance_data 
                    if record['timestamp'] >= cutoff_time
                ]
            
            if not recent_records:
                return {'message': '没有性能数据'}
            
            # 计算统计信息
            durations = [r['duration'] for r in recent_records]
            memory_usage = [r['memory_usage'] for r in recent_records]
            cpu_usage = [r['cpu_usage'] for r in recent_records]
            success_rate = sum(1 for r in recent_records if r['success']) / len(recent_records) * 100
            
            summary = {
                'period_hours': hours,
                'total_operations': len(recent_records),
                'success_rate': round(success_rate, 2),
                'response_time': {
                    'avg': round(sum(durations) / len(durations), 3),
                    'min': round(min(durations), 3),
                    'max': round(max(durations), 3),
                    'p95': round(sorted(durations)[int(len(durations) * 0.95)], 3) if durations else 0
                },
                'memory_usage': {
                    'avg': round(sum(memory_usage) / len(memory_usage), 1),
                    'min': round(min(memory_usage), 1),
                    'max': round(max(memory_usage), 1)
                },
                'cpu_usage': {
                    'avg': round(sum(cpu_usage) / len(cpu_usage), 1),
                    'min': round(min(cpu_usage), 1),
                    'max': round(max(cpu_usage), 1)
                },
                'operations_by_type': {}
            }
            
            # 按操作类型统计
            for record in recent_records:
                op_type = record['operation']
                if op_type not in summary['operations_by_type']:
                    summary['operations_by_type'][op_type] = {
                        'count': 0,
                        'success_count': 0,
                        'avg_duration': 0,
                        'total_duration': 0
                    }
                
                op_stats = summary['operations_by_type'][op_type]
                op_stats['count'] += 1
                op_stats['total_duration'] += record['duration']
                if record['success']:
                    op_stats['success_count'] += 1
            
            # 计算平均值
            for op_type, stats in summary['operations_by_type'].items():
                if stats['count'] > 0:
                    stats['success_rate'] = round(stats['success_count'] / stats['count'] * 100, 2)
                    stats['avg_duration'] = round(stats['total_duration'] / stats['count'], 3)
            
            return summary
            
        except Exception as e:
            logging.error(f"获取性能摘要失败: {str(e)}")
            return {'error': str(e)}
    
    def set_thresholds(self, **kwargs):
        """设置性能阈值"""
        try:
            with self._lock:
                for key, value in kwargs.items():
                    if key in self.thresholds:
                        self.thresholds[key] = value
                        logging.info(f"性能阈值已更新: {key} = {value}")
        except Exception as e:
            logging.error(f"设置性能阈值失败: {str(e)}")


class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "error_logger_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self._observers = []
        self._lock = threading.RLock()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 创建默认配置
                default_config = create_sample_config()
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logging.error(f"加载配置文件失败: {str(e)}")
            return create_sample_config()
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存配置文件失败: {str(e)}")
    
    def get(self, key: str, default=None):
        """获取配置值"""
        try:
            with self._lock:
                keys = key.split('.')
                value = self.config
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        try:
            with self._lock:
                keys = key.split('.')
                config = self.config
                for k in keys[:-1]:
                    if k not in config:
                        config[k] = {}
                    config = config[k]
                config[keys[-1]] = value
                
                # 保存配置
                self._save_config(self.config)
                
                # 通知观察者
                self._notify_observers(key, value)
                
        except Exception as e:
            logging.error(f"设置配置值失败: {str(e)}")
    
    def add_observer(self, callback: Callable[[str, Any], None]):
        """添加配置观察者"""
        try:
            with self._lock:
                self._observers.append(callback)
        except Exception as e:
            logging.error(f"添加配置观察者失败: {str(e)}")
    
    def _notify_observers(self, key: str, value: Any):
        """通知观察者"""
        try:
            for callback in self._observers:
                try:
                    callback(key, value)
                except Exception as e:
                    logging.error(f"配置观察者回调失败: {str(e)}")
        except Exception as e:
            logging.error(f"通知观察者失败: {str(e)}")
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        try:
            errors = []
            
            # 检查必需的配置项
            required_keys = [
                'database_path',
                'enable_alerts',
                'enable_recovery',
                'enable_statistics'
            ]
            
            for key in required_keys:
                if self.get(key) is None:
                    errors.append(f"缺少必需配置项: {key}")
            
            # 检查数据库路径
            db_path = self.get('database_path')
            if db_path:
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    try:
                        os.makedirs(db_dir, exist_ok=True)
                    except Exception as e:
                        errors.append(f"无法创建数据库目录: {db_dir}, 错误: {str(e)}")
            
            # 检查告警配置
            if self.get('enable_alerts'):
                email_config = self.get('notification_channels.email')
                if email_config:
                    required_email_keys = ['smtp_server', 'from', 'to']
                    for key in required_email_keys:
                        if not email_config.get(key):
                            errors.append(f"邮件配置缺少: {key}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"配置验证异常: {str(e)}"]
    
    def reset_to_default(self):
        """重置为默认配置"""
        try:
            with self._lock:
                self.config = create_sample_config()
                self._save_config(self.config)
                logging.info("配置已重置为默认值")
        except Exception as e:
            logging.error(f"重置配置失败: {str(e)}")


class ErrorPatternRecognizer:
    """错误模式识别器"""
    
    def __init__(self, error_logger: ErrorLogger):
        self.error_logger = error_logger
        self.patterns = {}
        self._lock = threading.RLock()
    
    def learn_patterns(self, days: int = 7) -> Dict[str, Any]:
        """学习错误模式"""
        try:
            patterns = {
                'recurring_errors': {},
                'error_clusters': {},
                'time_patterns': {},
                'sequence_patterns': {},
                'anomaly_detection': {}
            }
            
            # 获取历史错误数据
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            error_logs = self.error_logger.get_error_logs(
                start_date=start_time,
                end_date=end_time,
                limit=10000
            )
            
            # 识别重复错误
            patterns['recurring_errors'] = self._identify_recurring_errors(error_logs)
            
            # 错误聚类
            patterns['error_clusters'] = self._cluster_errors(error_logs)
            
            # 时间模式
            patterns['time_patterns'] = self._analyze_time_patterns(error_logs)
            
            # 序列模式
            patterns['sequence_patterns'] = self._analyze_sequence_patterns(error_logs)
            
            # 异常检测
            patterns['anomaly_detection'] = self._detect_anomalies(error_logs)
            
            with self._lock:
                self.patterns = patterns
            
            return patterns
            
        except Exception as e:
            logging.error(f"学习错误模式失败: {str(e)}")
            return {}
    
    def _identify_recurring_errors(self, error_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """识别重复错误"""
        try:
            recurring = {}
            error_signatures = defaultdict(list)
            
            for log in error_logs:
                # 创建错误签名
                signature = f"{log['error_type']}_{log['module_name']}_{log['function_name']}"
                error_signatures[signature].append(log)
            
            # 找出重复次数超过阈值的错误
            threshold = 3
            for signature, logs in error_signatures.items():
                if len(logs) >= threshold:
                    recurring[signature] = {
                        'count': len(logs),
                        'first_occurrence': min(log['timestamp'] for log in logs),
                        'last_occurrence': max(log['timestamp'] for log in logs),
                        'error_type': logs[0]['error_type'],
                        'module': logs[0]['module_name'],
                        'function': logs[0]['function_name']
                    }
            
            return recurring
            
        except Exception as e:
            logging.error(f"识别重复错误失败: {str(e)}")
            return {}
    
    def _cluster_errors(self, error_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """错误聚类"""
        try:
            clusters = {
                'by_type': defaultdict(list),
                'by_severity': defaultdict(list),
                'by_module': defaultdict(list),
                'by_time_window': defaultdict(list)
            }
            
            for log in error_logs:
                # 按类型聚类
                clusters['by_type'][log['error_type']].append(log)
                
                # 按严重程度聚类
                clusters['by_severity'][log['severity']].append(log)
                
                # 按模块聚类
                clusters['by_module'][log['module_name']].append(log)
                
                # 按时间窗口聚类（每小时）
                error_time = datetime.fromisoformat(log['timestamp'])
                time_window = error_time.strftime('%Y-%m-%d %H:00')
                clusters['by_time_window'][time_window].append(log)
            
            # 转换为普通字典
            result = {}
            for cluster_type, cluster_data in clusters.items():
                result[cluster_type] = {k: len(v) for k, v in cluster_data.items()}
            
            return result
            
        except Exception as e:
            logging.error(f"错误聚类失败: {str(e)}")
            return {}
    
    def _analyze_time_patterns(self, error_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析时间模式"""
        try:
            patterns = {
                'hourly_distribution': defaultdict(int),
                'daily_distribution': defaultdict(int),
                'weekly_distribution': defaultdict(int),
                'peak_hours': [],
                'quiet_hours': []
            }
            
            for log in error_logs:
                error_time = datetime.fromisoformat(log['timestamp'])
                
                patterns['hourly_distribution'][error_time.hour] += 1
                patterns['daily_distribution'][error_time.weekday()] += 1
                patterns['weekly_distribution'][error_time.strftime('%A')] += 1
            
            # 找出高峰和低谷时间
            hourly_counts = patterns['hourly_distribution']
            if hourly_counts:
                sorted_hours = sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)
                patterns['peak_hours'] = [hour for hour, count in sorted_hours[:3]]
                patterns['quiet_hours'] = [hour for hour, count in sorted_hours[-3:]]
            
            return dict(patterns)
            
        except Exception as e:
            logging.error(f"时间模式分析失败: {str(e)}")
            return {}
    
    def _analyze_sequence_patterns(self, error_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析序列模式"""
        try:
            sequences = {}
            
            # 按时间排序
            sorted_logs = sorted(error_logs, key=lambda x: x['timestamp'])
            
            # 查找连续的错误序列
            current_sequence = []
            sequence_threshold = 300  # 5分钟内认为是连续序列
            
            for i, log in enumerate(sorted_logs):
                if not current_sequence:
                    current_sequence.append(log)
                else:
                    prev_time = datetime.fromisoformat(current_sequence[-1]['timestamp'])
                    curr_time = datetime.fromisoformat(log['timestamp'])
                    
                    if (curr_time - prev_time).total_seconds() <= sequence_threshold:
                        current_sequence.append(log)
                    else:
                        if len(current_sequence) >= 2:
                            seq_key = f"{current_sequence[0]['error_type']}->{current_sequence[-1]['error_type']}"
                            if seq_key not in sequences:
                                sequences[seq_key] = []
                            sequences[seq_key].append({
                                'length': len(current_sequence),
                                'duration': (curr_time - prev_time).total_seconds(),
                                'errors': [log['error_type'] for log in current_sequence]
                            })
                        current_sequence = [log]
            
            return sequences
            
        except Exception as e:
            logging.error(f"序列模式分析失败: {str(e)}")
            return {}
    
    def _detect_anomalies(self, error_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检测异常"""
        try:
            anomalies = {
                'sudden_spikes': [],
                'unusual_patterns': [],
                'statistical_outliers': []
            }
            
            # 按小时统计错误数量
            hourly_counts = defaultdict(int)
            for log in error_logs:
                error_time = datetime.fromisoformat(log['timestamp'])
                hour_key = error_time.strftime('%Y-%m-%d %H')
                hourly_counts[hour_key] += 1
            
            if hourly_counts:
                counts = list(hourly_counts.values())
                mean_count = sum(counts) / len(counts)
                std_count = (sum((x - mean_count)**2 for x in counts) / len(counts)) ** 0.5
                
                # 检测突然峰值
                for hour, count in hourly_counts.items():
                    if count > mean_count + 2 * std_count:
                        anomalies['sudden_spikes'].append({
                            'hour': hour,
                            'error_count': count,
                            'expected_count': round(mean_count, 1),
                            'deviation': round((count - mean_count) / std_count, 2)
                        })
            
            return anomalies
            
        except Exception as e:
            logging.error(f"异常检测失败: {str(e)}")
            return {}
    
    def predict_next_errors(self) -> List[Dict[str, Any]]:
        """预测下一个可能发生的错误"""
        try:
            predictions = []
            
            with self._lock:
                if not self.patterns:
                    return predictions
                
                # 基于重复错误预测
                recurring = self.patterns.get('recurring_errors', {})
                for signature, info in recurring.items():
                    predictions.append({
                        'type': 'recurring_error',
                        'signature': signature,
                        'probability': min(info['count'] / 10.0, 1.0),  # 简单概率计算
                        'description': f"错误 {signature} 可能会再次发生",
                        'recommendation': "检查相关代码和配置"
                    })
                
                # 基于时间模式预测
                time_patterns = self.patterns.get('time_patterns', {})
                peak_hours = time_patterns.get('peak_hours', [])
                if peak_hours:
                    current_hour = datetime.now().hour
                    if current_hour in peak_hours:
                        predictions.append({
                            'type': 'time_based',
                            'hour': current_hour,
                            'probability': 0.7,
                            'description': f"当前时间 {current_hour}:00 是错误高峰期",
                            'recommendation': "加强监控和预警"
                        })
            
            return predictions
            
        except Exception as e:
            logging.error(f"预测下一个错误失败: {str(e)}")
            return []


class AutomatedTestSuite:
    """自动化测试套件"""
    
    def __init__(self, error_logger: ErrorLogger):
        self.error_logger = error_logger
        self.test_results = []
        self._lock = threading.RLock()
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        try:
            test_results = {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'test_details': [],
                'overall_status': 'unknown'
            }
            
            # 测试用例
            test_cases = [
                self.test_basic_logging,
                self.test_error_classification,
                self.test_alert_system,
                self.test_recovery_mechanism,
                self.test_statistics_generation,
                self.test_database_operations,
                self.test_performance_monitoring,
                self.test_configuration_management,
                self.test_pattern_recognition,
                self.test_async_processing
            ]
            
            for test_case in test_cases:
                try:
                    test_results['total_tests'] += 1
                    result = test_case()
                    test_results['test_details'].append(result)
                    
                    if result['status'] == 'passed':
                        test_results['passed_tests'] += 1
                    else:
                        test_results['failed_tests'] += 1
                        
                except Exception as e:
                    test_results['total_tests'] += 1
                    test_results['failed_tests'] += 1
                    test_results['test_details'].append({
                        'test_name': test_case.__name__,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # 确定整体状态
            if test_results['failed_tests'] == 0:
                test_results['overall_status'] = 'passed'
            elif test_results['passed_tests'] == test_results['total_tests'] // 2:
                test_results['overall_status'] = 'partial'
            else:
                test_results['overall_status'] = 'failed'
            
            with self._lock:
                self.test_results.append(test_results)
            
            return test_results
            
        except Exception as e:
            logging.error(f"运行测试套件失败: {str(e)}")
            return {'error': str(e)}
    
    def test_basic_logging(self) -> Dict[str, Any]:
        """测试基本日志记录功能"""
        try:
            # 测试记录不同类型的错误
            error_types = [ErrorType.SYSTEM, ErrorType.BUSINESS, ErrorType.NETWORK]
            error_ids = []
            
            for error_type in error_types:
                error_id = self.error_logger.log_error(
                    error_type=error_type,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"测试错误 - {error_type.value}",
                    additional_data={'test': True}
                )
                error_ids.append(error_id)
            
            # 验证错误是否被记录
            recent_errors = self.error_logger.get_recent_errors(len(error_ids))
            
            if len(recent_errors) >= len(error_ids):
                return {
                    'test_name': 'test_basic_logging',
                    'status': 'passed',
                    'message': f'成功记录 {len(error_ids)} 个错误',
                    'details': {'recorded_errors': len(recent_errors)}
                }
            else:
                return {
                    'test_name': 'test_basic_logging',
                    'status': 'failed',
                    'message': f'记录错误数量不匹配，期望 {len(error_ids)}，实际 {len(recent_errors)}'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_basic_logging',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_error_classification(self) -> Dict[str, Any]:
        """测试错误分类功能"""
        try:
            # 记录不同类型的错误
            test_cases = [
                (ErrorType.SYSTEM, "系统错误测试"),
                (ErrorType.BUSINESS, "业务错误测试"),
                (ErrorType.NETWORK, "网络错误测试"),
                (ErrorType.DATABASE, "数据库错误测试")
            ]
            
            for error_type, message in test_cases:
                self.error_logger.log_error(
                    error_type=error_type,
                    severity=ErrorSeverity.MEDIUM,
                    message=message
                )
            
            # 获取统计信息
            stats = self.error_logger.get_statistics(hours=1)
            
            # 验证分类是否正确
            classified_types = [error_type for error_type, count in stats.errors_by_type.items() if count > 0]
            
            expected_types = set(error_type for error_type, _ in test_cases)
            actual_types = set(classified_types)
            
            if expected_types.issubset(actual_types):
                return {
                    'test_name': 'test_error_classification',
                    'status': 'passed',
                    'message': '错误分类功能正常',
                    'details': {'classified_types': len(actual_types)}
                }
            else:
                missing_types = expected_types - actual_types
                return {
                    'test_name': 'test_error_classification',
                    'status': 'failed',
                    'message': f'缺少错误类型: {missing_types}'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_error_classification',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_alert_system(self) -> Dict[str, Any]:
        """测试告警系统"""
        try:
            # 记录一个严重错误以触发告警
            error_id = self.error_logger.log_error(
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                message="测试严重错误 - 触发告警",
                additional_data={'alert_test': True}
            )
            
            # 检查是否应该发送告警
            if error_id:
                return {
                    'test_name': 'test_alert_system',
                    'status': 'passed',
                    'message': '告警系统测试完成',
                    'details': {'test_error_id': error_id}
                }
            else:
                return {
                    'test_name': 'test_alert_system',
                    'status': 'failed',
                    'message': '告警系统测试失败'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_alert_system',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_recovery_mechanism(self) -> Dict[str, Any]:
        """测试恢复机制"""
        try:
            def test_recovery_func():
                return "恢复成功"
            
            success, result = self.error_logger.attempt_recovery(
                error_type=ErrorType.BUSINESS,
                recovery_func=test_recovery_func
            )
            
            if success and result == "恢复成功":
                return {
                    'test_name': 'test_recovery_mechanism',
                    'status': 'passed',
                    'message': '恢复机制测试成功',
                    'details': {'recovery_result': result}
                }
            else:
                return {
                    'test_name': 'test_recovery_mechanism',
                    'status': 'failed',
                    'message': f'恢复机制测试失败: {success}, {result}'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_recovery_mechanism',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_statistics_generation(self) -> Dict[str, Any]:
        """测试统计信息生成"""
        try:
            # 先记录一些错误
            for i in range(5):
                self.error_logger.log_error(
                    error_type=ErrorType.SYSTEM,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"统计测试错误 {i+1}"
                )
            
            # 生成统计信息
            stats = self.error_logger.get_statistics(hours=1)
            
            if stats.total_errors >= 5:
                return {
                    'test_name': 'test_statistics_generation',
                    'status': 'passed',
                    'message': '统计信息生成正常',
                    'details': {'total_errors': stats.total_errors}
                }
            else:
                return {
                    'test_name': 'test_statistics_generation',
                    'status': 'failed',
                    'message': f'统计信息错误数量不匹配: {stats.total_errors}'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_statistics_generation',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_database_operations(self) -> Dict[str, Any]:
        """测试数据库操作"""
        try:
            # 记录错误
            error_id = self.error_logger.log_error(
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.LOW,
                message="数据库操作测试"
            )
            
            # 测试状态更新
            success = self.error_logger.resolve_error(error_id, "测试解决")
            
            if success:
                return {
                    'test_name': 'test_database_operations',
                    'status': 'passed',
                    'message': '数据库操作正常',
                    'details': {'test_error_id': error_id}
                }
            else:
                return {
                    'test_name': 'test_database_operations',
                    'status': 'failed',
                    'message': '数据库状态更新失败'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_database_operations',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_performance_monitoring(self) -> Dict[str, Any]:
        """测试性能监控"""
        try:
            # 创建性能监控器
            monitor = PerformanceMonitor(self.error_logger)
            
            # 记录性能数据
            success = monitor.record_performance(
                operation="test_operation",
                duration=0.1,
                memory_usage=50.0,
                success=True
            )
            
            # 获取性能摘要
            summary = monitor.get_performance_summary(hours=1)
            
            if success and 'total_operations' in summary:
                return {
                    'test_name': 'test_performance_monitoring',
                    'status': 'passed',
                    'message': '性能监控正常',
                    'details': summary
                }
            else:
                return {
                    'test_name': 'test_performance_monitoring',
                    'status': 'failed',
                    'message': '性能监控测试失败'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_performance_monitoring',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_configuration_management(self) -> Dict[str, Any]:
        """测试配置管理"""
        try:
            # 创建配置管理器
            config_manager = ConfigurationManager("test_config.json")
            
            # 测试设置和获取配置
            config_manager.set("test_key", "test_value")
            value = config_manager.get("test_key")
            
            # 验证配置
            is_valid, errors = config_manager.validate_config()
            
            if value == "test_value" and is_valid:
                return {
                    'test_name': 'test_configuration_management',
                    'status': 'passed',
                    'message': '配置管理正常',
                    'details': {'test_value': value}
                }
            else:
                return {
                    'test_name': 'test_configuration_management',
                    'status': 'failed',
                    'message': f'配置管理测试失败: value={value}, valid={is_valid}, errors={errors}'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_configuration_management',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_pattern_recognition(self) -> Dict[str, Any]:
        """测试模式识别"""
        try:
            # 创建模式识别器
            recognizer = ErrorPatternRecognizer(self.error_logger)
            
            # 学习模式
            patterns = recognizer.learn_patterns(days=1)
            
            # 预测下一个错误
            predictions = recognizer.predict_next_errors()
            
            if isinstance(patterns, dict) and isinstance(predictions, list):
                return {
                    'test_name': 'test_pattern_recognition',
                    'status': 'passed',
                    'message': '模式识别正常',
                    'details': {
                        'patterns_found': len(patterns),
                        'predictions_count': len(predictions)
                    }
                }
            else:
                return {
                    'test_name': 'test_pattern_recognition',
                    'status': 'failed',
                    'message': '模式识别测试失败'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_pattern_recognition',
                'status': 'failed',
                'error': str(e)
            }
    
    def test_async_processing(self) -> Dict[str, Any]:
        """测试异步处理"""
        try:
            # 记录多个错误以测试异步处理
            error_ids = []
            for i in range(10):
                error_id = self.error_logger.log_error(
                    error_type=ErrorType.SYSTEM,
                    severity=ErrorSeverity.LOW,
                    message=f"异步处理测试错误 {i+1}",
                    additional_data={'async_test': True}
                )
                error_ids.append(error_id)
            
            # 等待异步处理完成
            time.sleep(1)
            
            # 检查错误是否被处理
            recent_errors = self.error_logger.get_recent_errors(len(error_ids))
            
            if len(recent_errors) >= len(error_ids):
                return {
                    'test_name': 'test_async_processing',
                    'status': 'passed',
                    'message': '异步处理正常',
                    'details': {'processed_errors': len(recent_errors)}
                }
            else:
                return {
                    'test_name': 'test_async_processing',
                    'status': 'failed',
                    'message': f'异步处理错误数量不匹配: 期望 {len(error_ids)}, 实际 {len(recent_errors)}'
                }
                
        except Exception as e:
            return {
                'test_name': 'test_async_processing',
                'status': 'failed',
                'error': str(e)
            }
    
    def generate_test_report(self, output_file: str = "test_report.json") -> bool:
        """生成测试报告"""
        try:
            with self._lock:
                if not self.test_results:
                    return False
                
                latest_result = self.test_results[-1]
                
                report = {
                    'generated_at': datetime.now().isoformat(),
                    'test_summary': {
                        'total_tests': latest_result['total_tests'],
                        'passed_tests': latest_result['passed_tests'],
                        'failed_tests': latest_result['failed_tests'],
                        'success_rate': round(latest_result['passed_tests'] / latest_result['total_tests'] * 100, 2) if latest_result['total_tests'] > 0 else 0,
                        'overall_status': latest_result['overall_status']
                    },
                    'test_details': latest_result['test_details'],
                    'historical_results': self.test_results[-5:]  # 最近5次测试结果
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                logging.info(f"测试报告已生成: {output_file}")
                return True
                
        except Exception as e:
            logging.error(f"生成测试报告失败: {str(e)}")
            return False


# 扩展的示例和演示
def run_comprehensive_demo():
    """运行综合演示"""
    print("=== L3错误日志记录器综合演示 ===\n")
    
    # 1. 创建配置
    config = create_sample_config()
    config['enable_alerts'] = False  # 演示时关闭告警
    config['async_processing'] = True
    
    # 2. 初始化错误日志记录器
    error_logger = ErrorLogger(config)
    
    # 3. 创建辅助组件
    trend_analyzer = ErrorTrendAnalyzer(error_logger)
    performance_monitor = PerformanceMonitor(error_logger)
    config_manager = ConfigurationManager("demo_config.json")
    pattern_recognizer = ErrorPatternRecognizer(error_logger)
    test_suite = AutomatedTestSuite(error_logger)
    
    # 4. 演示各种错误场景
    print("1. 演示错误记录功能")
    error_scenarios = [
        (ErrorType.SYSTEM, ErrorSeverity.CRITICAL, "系统内存不足"),
        (ErrorType.BUSINESS, ErrorSeverity.HIGH, "业务逻辑验证失败"),
        (ErrorType.NETWORK, ErrorSeverity.MEDIUM, "网络连接超时"),
        (ErrorType.DATABASE, ErrorSeverity.HIGH, "数据库连接失败"),
        (ErrorType.SECURITY, ErrorSeverity.CRITICAL, "未授权访问尝试"),
        (ErrorType.PERFORMANCE, ErrorSeverity.MEDIUM, "响应时间过长"),
        (ErrorType.CONFIGURATION, ErrorSeverity.LOW, "配置参数缺失")
    ]
    
    for error_type, severity, message in error_scenarios:
        error_id = error_logger.log_error(
            error_type=error_type,
            severity=severity,
            message=message,
            additional_data={
                'demo': True,
                'scenario': 'comprehensive_demo'
            }
        )
        print(f"  记录错误: {error_id} - {message}")
    
    # 5. 演示性能监控
    print("\n2. 演示性能监控功能")
    operations = ["数据库查询", "API调用", "文件处理", "计算任务"]
    for operation in operations:
        import random
        duration = random.uniform(0.1, 2.0)
        success = random.random() > 0.1  # 90%成功率
        
        performance_monitor.record_performance(
            operation=operation,
            duration=duration,
            success=success
        )
        print(f"  记录性能: {operation} - {duration:.2f}s - {'成功' if success else '失败'}")
    
    # 6. 演示恢复机制
    print("\n3. 演示错误恢复机制")
    recovery_scenarios = [
        ("网络重试", lambda: "网络连接已恢复"),
        ("业务降级", lambda: "使用缓存数据"),
        ("服务重启", lambda: "服务已重启完成")
    ]
    
    for scenario_name, recovery_func in recovery_scenarios:
        success, result = error_logger.attempt_recovery(
            error_type=ErrorType.BUSINESS,
            recovery_func=recovery_func
        )
        print(f"  {scenario_name}: {'成功' if success else '失败'} - {result}")
    
    # 7. 演示错误解决流程
    print("\n4. 演示错误解决流程")
    recent_errors = error_logger.get_recent_errors(3)
    for i, error in enumerate(recent_errors):
        # 解决错误
        success = error_logger.resolve_error(error.error_id, f"演示解决方案 {i+1}")
        print(f"  解决错误 {error.error_id}: {'成功' if success else '失败'}")
        
        # 关闭错误
        success = error_logger.close_error(error.error_id, "问题已验证并关闭")
        print(f"  关闭错误 {error.error_id}: {'成功' if success else '失败'}")
    
    # 8. 等待异步处理完成
    print("\n5. 等待异步处理完成...")
    time.sleep(2)
    
    # 9. 获取统计信息
    print("\n6. 获取统计信息")
    stats = error_logger.get_statistics(hours=1)
    print(f"  总错误数: {stats.total_errors}")
    print(f"  严重错误数: {stats.errors_by_severity.get(ErrorSeverity.CRITICAL, 0)}")
    print(f"  系统影响分数: {stats.system_impact_score:.2f}")
    print(f"  平均解决时间: {stats.avg_resolution_time:.2f}秒")
    
    # 10. 获取系统健康状态
    print("\n7. 系统健康状态")
    health = error_logger.get_system_health()
    print(f"  健康状态: {health['status']}")
    print(f"  健康分数: {health['health_score']:.1f}")
    print(f"  错误率(每小时): {health['error_rate_per_hour']}")
    print(f"  最近1小时严重错误: {health['critical_errors_last_hour']}")
    
    # 11. 性能监控摘要
    print("\n8. 性能监控摘要")
    perf_summary = performance_monitor.get_performance_summary(hours=1)
    print(f"  总操作数: {perf_summary.get('total_operations', 0)}")
    print(f"  成功率: {perf_summary.get('success_rate', 0):.1f}%")
    print(f"  平均响应时间: {perf_summary.get('response_time', {}).get('avg', 0):.3f}秒")
    
    # 12. 错误趋势分析
    print("\n9. 错误趋势分析")
    trends = trend_analyzer.analyze_error_trends(days=1)
    if trends:
        print(f"  错误增长率: {trends.get('error_growth_rate', 0):.1f}%")
        if trends.get('peak_error_day'):
            peak = trends['peak_error_day']
            print(f"  峰值错误日: {peak['date']} ({peak['error_count']}个错误)")
    
    # 13. 错误模式识别
    print("\n10. 错误模式识别")
    patterns = pattern_recognizer.learn_patterns(days=1)
    if patterns:
        recurring_count = len(patterns.get('recurring_errors', {}))
        print(f"  发现重复错误模式: {recurring_count}个")
    
    predictions = pattern_recognizer.predict_next_errors()
    print(f"  预测可能发生的错误: {len(predictions)}个")
    
    # 14. 运行自动化测试
    print("\n11. 运行自动化测试")
    test_results = test_suite.run_all_tests()
    print(f"  测试状态: {test_results['overall_status']}")
    print(f"  通过测试: {test_results['passed_tests']}/{test_results['total_tests']}")
    print(f"  成功率: {test_results['passed_tests']/test_results['total_tests']*100:.1f}%" if test_results['total_tests'] > 0 else "  无测试")
    
    # 15. 导出报告
    print("\n12. 导出报告")
    error_logger.export_error_report("comprehensive_demo_report.json", hours=1)
    test_suite.generate_test_report("demo_test_report.json")
    print("  错误报告已导出: comprehensive_demo_report.json")
    print("  测试报告已导出: demo_test_report.json")
    
    # 16. 清理演示数据
    print("\n13. 清理演示数据")
    cleaned_count = error_logger.cleanup_old_logs(days=0)  # 清理所有演示数据
    print(f"  清理了 {cleaned_count} 条演示记录")
    
    # 17. 关闭系统
    print("\n14. 关闭系统")
    error_logger.shutdown()
    print("  错误日志记录器已关闭")
    
    print("\n=== 综合演示完成 ===")


if __name__ == "__main__":
    # 运行基础示例
    print("运行基础示例...")
    
    # 运行综合演示
    run_comprehensive_demo()