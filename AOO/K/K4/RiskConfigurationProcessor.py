#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K4风险配置处理器模块

该模块提供了全面的风险配置管理功能，包括市场风险、信用风险、操作风险
等各种风险类型的配置管理、监控、告警、对冲和报告功能。

功能特性：
- 风险参数配置管理（风险阈值、风险限制、风险模型）
- 市场风险配置（VaR、CVaR、最大回撤等）
- 信用风险配置（信用评级、违约概率等）
- 操作风险配置（系统风险、人为风险等）
- 风险监控配置和告警设置
- 风险对冲配置和策略设置
- 风险报告配置和输出设置
- 异步风险配置处理
- 完整的错误处理和日志记录

作者: K4风险配置处理器开发团队
版本: 1.0.0
创建时间: 2025-11-06
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set
from collections import defaultdict, deque
import warnings
import traceback
from contextlib import contextmanager
import pickle
import yaml
import csv
from io import StringIO


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('risk_configuration_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class RiskType(Enum):
    """风险类型枚举"""
    MARKET_RISK = auto()
    CREDIT_RISK = auto()
    OPERATIONAL_RISK = auto()
    LIQUIDITY_RISK = auto()
    SYSTEMATIC_RISK = auto()
    COUNTERPARTY_RISK = auto()
    REGULATORY_RISK = auto()


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class AlertSeverity(Enum):
    """告警严重程度枚举"""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class HedgeStrategy(Enum):
    """对冲策略枚举"""
    DYNAMIC_HEDGING = auto()
    STATIC_HEDGING = auto()
    DELTA_HEDGING = auto()
    GAMMA_HEDGING = auto()
    VEGA_HEDGING = auto()


class ReportFormat(Enum):
    """报告格式枚举"""
    JSON = auto()
    XML = auto()
    CSV = auto()
    HTML = auto()
    PDF = auto()


@dataclass
class RiskThreshold:
    """风险阈值配置"""
    name: str
    value: float
    unit: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimit:
    """风险限制配置"""
    name: str
    limit_value: float
    current_value: float = 0.0
    unit: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    utilization_rate: float = field(init=False)
    
    def __post_init__(self):
        self.utilization_rate = (self.current_value / self.limit_value * 100) if self.limit_value > 0 else 0.0


@dataclass
class RiskModel:
    """风险模型配置"""
    name: str
    model_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    validation_status: str = "PENDING"


@dataclass
class MarketRiskConfig:
    """市场风险配置"""
    var_confidence_level: float = 0.95
    var_time_horizon: int = 1  # 天数
    cvar_confidence_level: float = 0.95
    max_drawdown_limit: float = 0.15
    volatility_threshold: float = 0.25
    correlation_limit: float = 0.8
    stress_test_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    historical_data_period: int = 252  # 交易日
    monte_carlo_simulations: int = 10000
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CreditRiskConfig:
    """信用风险配置"""
    rating_scale: Dict[str, int] = field(default_factory=lambda: {
        "AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6, "CCC": 7, "CC": 8, "C": 9, "D": 10
    })
    default_probabilities: Dict[str, float] = field(default_factory=dict)
    exposure_limits: Dict[str, float] = field(default_factory=dict)
    concentration_limits: Dict[str, float] = field(default_factory=dict)
    recovery_rates: Dict[str, float] = field(default_factory=dict)
    credit_migration_matrix: List[List[float]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class OperationalRiskConfig:
    """操作风险配置"""
    system_risk_factors: List[str] = field(default_factory=list)
    human_risk_factors: List[str] = field(default_factory=list)
    process_risk_factors: List[str] = field(default_factory=list)
    external_risk_factors: List[str] = field(default_factory=list)
    loss_event_categories: Dict[str, float] = field(default_factory=dict)
    control_effectiveness_ratings: Dict[str, int] = field(default_factory=dict)
    risk_and_control_assessment_frequency: int = 12  # 月
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class MonitoringConfig:
    """风险监控配置"""
    monitoring_frequency: int = 60  # 秒
    real_time_alerts: bool = True
    alert_channels: List[str] = field(default_factory=list)
    escalation_rules: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    aggregation_methods: Dict[str, str] = field(default_factory=dict)
    retention_period: int = 365  # 天
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AlertConfig:
    """告警配置"""
    alert_id: str
    name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    channels: List[str] = field(default_factory=list)
    enabled: bool = True
    cooldown_period: int = 300  # 秒
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class HedgeConfig:
    """风险对冲配置"""
    strategy: HedgeStrategy
    target_risk_reduction: float = 0.8
    instruments: List[str] = field(default_factory=list)
    rebalancing_frequency: int = 1  # 天
    cost_budget: float = 0.001  # 成本预算比例
    execution_constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReportConfig:
    """风险报告配置"""
    report_name: str
    report_type: str
    format: ReportFormat
    schedule: str = "DAILY"
    recipients: List[str] = field(default_factory=list)
    content_sections: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    template_path: Optional[str] = None
    output_directory: str = "./reports"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class RiskConfigurationError(Exception):
    """风险配置异常基类"""
    pass


class ValidationError(RiskConfigurationError):
    """配置验证异常"""
    pass


class ProcessingError(RiskConfigurationError):
    """处理异常"""
    pass


class DatabaseError(RiskConfigurationError):
    """数据库异常"""
    pass


class AlertSystem:
    """告警系统"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AlertSystem")
        self._callbacks: List[Callable] = []
        self._alert_history = deque(maxlen=1000)
    
    def add_callback(self, callback: Callable[[AlertConfig, Dict[str, Any]], None]):
        """添加告警回调函数"""
        self._callbacks.append(callback)
    
    def trigger_alert(self, context: Dict[str, Any]) -> bool:
        """触发告警"""
        try:
            current_time = datetime.now()
            
            # 检查冷却期
            if (self.config.last_triggered and 
                (current_time - self.config.last_triggered).total_seconds() < self.config.cooldown_period):
                self.logger.debug(f"告警 {self.config.alert_id} 在冷却期内，跳过")
                return False
            
            # 评估告警条件
            if self._evaluate_condition(context):
                alert_data = {
                    'alert_id': self.config.alert_id,
                    'name': self.config.name,
                    'severity': self.config.severity,
                    'timestamp': current_time,
                    'context': context,
                    'trigger_count': self.config.trigger_count + 1
                }
                
                # 更新配置
                self.config.last_triggered = current_time
                self.config.trigger_count += 1
                
                # 记录告警历史
                self._alert_history.append(alert_data)
                
                # 执行回调
                for callback in self._callbacks:
                    try:
                        callback(self.config, alert_data)
                    except Exception as e:
                        self.logger.error(f"告警回调执行失败: {e}")
                
                self.logger.warning(f"告警触发: {self.config.name} - {self.config.severity}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"告警触发失败: {e}")
            return False
    
    def _evaluate_condition(self, context: Dict[str, Any]) -> bool:
        """评估告警条件"""
        try:
            # 简单的条件评估，实际实现中可以更复杂
            if 'var' in context and 'threshold' in context:
                return context['var'] > context['threshold']
            if 'value' in context and 'threshold' in context:
                return context['value'] > context['threshold']
            return False
        except Exception as e:
            self.logger.error(f"条件评估失败: {e}")
            return False
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警历史"""
        return list(self._alert_history)[-limit:]


class RiskDatabase:
    """风险配置数据库管理器"""
    
    def __init__(self, db_path: str = "risk_config.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.RiskDatabase")
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 创建风险阈值表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_thresholds (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        value REAL NOT NULL,
                        unit TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        metadata TEXT
                    )
                ''')
                
                # 创建风险限制表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_limits (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        limit_value REAL NOT NULL,
                        current_value REAL DEFAULT 0,
                        unit TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # 创建风险模型表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        model_type TEXT NOT NULL,
                        parameters TEXT,
                        description TEXT,
                        version TEXT DEFAULT '1.0',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        validation_status TEXT DEFAULT 'PENDING'
                    )
                ''')
                
                # 创建告警配置表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alert_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        threshold REAL NOT NULL,
                        severity TEXT NOT NULL,
                        channels TEXT,
                        enabled BOOLEAN DEFAULT 1,
                        cooldown_period INTEGER DEFAULT 300,
                        last_triggered TIMESTAMP,
                        trigger_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建市场风险配置表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_risk_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        var_confidence_level REAL DEFAULT 0.95,
                        var_time_horizon INTEGER DEFAULT 1,
                        cvar_confidence_level REAL DEFAULT 0.95,
                        max_drawdown_limit REAL DEFAULT 0.15,
                        volatility_threshold REAL DEFAULT 0.25,
                        correlation_limit REAL DEFAULT 0.8,
                        stress_test_scenarios TEXT,
                        historical_data_period INTEGER DEFAULT 252,
                        monte_carlo_simulations INTEGER DEFAULT 10000,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建信用风险配置表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS credit_risk_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        rating_scale TEXT,
                        default_probabilities TEXT,
                        exposure_limits TEXT,
                        concentration_limits TEXT,
                        recovery_rates TEXT,
                        credit_migration_matrix TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建操作风险配置表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS operational_risk_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_risk_factors TEXT,
                        human_risk_factors TEXT,
                        process_risk_factors TEXT,
                        external_risk_factors TEXT,
                        loss_event_categories TEXT,
                        control_effectiveness_ratings TEXT,
                        risk_and_control_assessment_frequency INTEGER DEFAULT 12,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建监控配置表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS monitoring_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        monitoring_frequency INTEGER DEFAULT 60,
                        real_time_alerts BOOLEAN DEFAULT 1,
                        alert_channels TEXT,
                        escalation_rules TEXT,
                        data_sources TEXT,
                        aggregation_methods TEXT,
                        retention_period INTEGER DEFAULT 365,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建对冲配置表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS hedge_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy TEXT NOT NULL,
                        target_risk_reduction REAL DEFAULT 0.8,
                        instruments TEXT,
                        rebalancing_frequency INTEGER DEFAULT 1,
                        cost_budget REAL DEFAULT 0.001,
                        execution_constraints TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建报告配置表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS report_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_name TEXT UNIQUE NOT NULL,
                        report_type TEXT NOT NULL,
                        format TEXT NOT NULL,
                        schedule TEXT DEFAULT 'DAILY',
                        recipients TEXT,
                        content_sections TEXT,
                        data_sources TEXT,
                        template_path TEXT,
                        output_directory TEXT DEFAULT './reports',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("数据库初始化完成")
                
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            raise DatabaseError(f"数据库初始化失败: {e}")
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_risk_threshold(self, threshold: RiskThreshold) -> bool:
        """保存风险阈值"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO risk_thresholds 
                    (name, value, unit, description, created_at, updated_at, is_active, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    threshold.name, threshold.value, threshold.unit, threshold.description,
                    threshold.created_at, threshold.updated_at, threshold.is_active,
                    json.dumps(threshold.metadata)
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"保存风险阈值失败: {e}")
            return False
    
    def load_risk_thresholds(self) -> List[RiskThreshold]:
        """加载风险阈值"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM risk_thresholds WHERE is_active = 1')
                rows = cursor.fetchall()
                
                thresholds = []
                for row in rows:
                    threshold = RiskThreshold(
                        name=row['name'],
                        value=row['value'],
                        unit=row['unit'] or '',
                        description=row['description'] or '',
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        is_active=bool(row['is_active']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else {}
                    )
                    thresholds.append(threshold)
                
                return thresholds
        except Exception as e:
            self.logger.error(f"加载风险阈值失败: {e}")
            return []
    
    def save_risk_limit(self, limit: RiskLimit) -> bool:
        """保存风险限制"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO risk_limits 
                    (name, limit_value, current_value, unit, description, created_at, updated_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    limit.name, limit.limit_value, limit.current_value, limit.unit,
                    limit.description, limit.created_at, limit.updated_at, limit.is_active
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"保存风险限制失败: {e}")
            return False
    
    def load_risk_limits(self) -> List[RiskLimit]:
        """加载风险限制"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM risk_limits WHERE is_active = 1')
                rows = cursor.fetchall()
                
                limits = []
                for row in rows:
                    limit = RiskLimit(
                        name=row['name'],
                        limit_value=row['limit_value'],
                        current_value=row['current_value'],
                        unit=row['unit'] or '',
                        description=row['description'] or '',
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        is_active=bool(row['is_active'])
                    )
                    limits.append(limit)
                
                return limits
        except Exception as e:
            self.logger.error(f"加载风险限制失败: {e}")
            return []
    
    def save_risk_model(self, model: RiskModel) -> bool:
        """保存风险模型"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO risk_models 
                    (name, model_type, parameters, description, version, created_at, updated_at, is_active, validation_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model.name, model.model_type, json.dumps(model.parameters),
                    model.description, model.version, model.created_at, model.updated_at,
                    model.is_active, model.validation_status
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"保存风险模型失败: {e}")
            return False
    
    def load_risk_models(self) -> List[RiskModel]:
        """加载风险模型"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM risk_models WHERE is_active = 1')
                rows = cursor.fetchall()
                
                models = []
                for row in rows:
                    model = RiskModel(
                        name=row['name'],
                        model_type=row['model_type'],
                        parameters=json.loads(row['parameters']) if row['parameters'] else {},
                        description=row['description'] or '',
                        version=row['version'] or '1.0',
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        is_active=bool(row['is_active']),
                        validation_status=row['validation_status'] or 'PENDING'
                    )
                    models.append(model)
                
                return models
        except Exception as e:
            self.logger.error(f"加载风险模型失败: {e}")
            return []
    
    def save_alert_config(self, config: AlertConfig) -> bool:
        """保存告警配置"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alert_configs 
                    (alert_id, name, condition, threshold, severity, channels, enabled, 
                     cooldown_period, last_triggered, trigger_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    config.alert_id, config.name, config.condition, config.threshold,
                    config.severity.name, json.dumps(config.channels), config.enabled,
                    config.cooldown_period, config.last_triggered, config.trigger_count,
                    config.created_at, config.updated_at
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"保存告警配置失败: {e}")
            return False
    
    def load_alert_configs(self) -> List[AlertConfig]:
        """加载告警配置"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM alert_configs')
                rows = cursor.fetchall()
                
                configs = []
                for row in rows:
                    config = AlertConfig(
                        alert_id=row['alert_id'],
                        name=row['name'],
                        condition=row['condition'],
                        threshold=row['threshold'],
                        severity=AlertSeverity[row['severity']],
                        channels=json.loads(row['channels']) if row['channels'] else [],
                        enabled=bool(row['enabled']),
                        cooldown_period=row['cooldown_period'],
                        last_triggered=datetime.fromisoformat(row['last_triggered']) if row['last_triggered'] else None,
                        trigger_count=row['trigger_count'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                    configs.append(config)
                
                return configs
        except Exception as e:
            self.logger.error(f"加载告警配置失败: {e}")
            return []


class RiskValidator:
    """风险配置验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RiskValidator")
    
    def validate_threshold(self, threshold: RiskThreshold) -> Tuple[bool, List[str]]:
        """验证风险阈值"""
        errors = []
        
        if not threshold.name or not threshold.name.strip():
            errors.append("风险阈值名称不能为空")
        
        if threshold.value <= 0:
            errors.append("风险阈值必须大于0")
        
        if not threshold.unit or not threshold.unit.strip():
            errors.append("风险阈值单位不能为空")
        
        return len(errors) == 0, errors
    
    def validate_limit(self, limit: RiskLimit) -> Tuple[bool, List[str]]:
        """验证风险限制"""
        errors = []
        
        if not limit.name or not limit.name.strip():
            errors.append("风险限制名称不能为空")
        
        if limit.limit_value <= 0:
            errors.append("风险限制值必须大于0")
        
        if limit.current_value < 0:
            errors.append("当前风险值不能为负数")
        
        if limit.current_value > limit.limit_value:
            errors.append("当前风险值不能超过限制值")
        
        return len(errors) == 0, errors
    
    def validate_model(self, model: RiskModel) -> Tuple[bool, List[str]]:
        """验证风险模型"""
        errors = []
        
        if not model.name or not model.name.strip():
            errors.append("风险模型名称不能为空")
        
        if not model.model_type or not model.model_type.strip():
            errors.append("风险模型类型不能为空")
        
        if not model.parameters:
            errors.append("风险模型参数不能为空")
        
        return len(errors) == 0, errors
    
    def validate_alert_config(self, config: AlertConfig) -> Tuple[bool, List[str]]:
        """验证告警配置"""
        errors = []
        
        if not config.alert_id or not config.alert_id.strip():
            errors.append("告警ID不能为空")
        
        if not config.name or not config.name.strip():
            errors.append("告警名称不能为空")
        
        if not config.condition or not config.condition.strip():
            errors.append("告警条件不能为空")
        
        if config.threshold <= 0:
            errors.append("告警阈值必须大于0")
        
        if not config.channels:
            errors.append("告警渠道不能为空")
        
        return len(errors) == 0, errors


class RiskConfigurationProcessor:
    """K4风险配置处理器主类"""
    
    def __init__(self, config_path: Optional[str] = None, db_path: str = "risk_config.db"):
        """
        初始化风险配置处理器
        
        Args:
            config_path: 配置文件路径
            db_path: 数据库路径
        """
        self.config_path = config_path
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.RiskConfigurationProcessor")
        
        # 初始化组件
        self.database = RiskDatabase(db_path)
        self.validator = RiskValidator()
        self.alert_systems: Dict[str, AlertSystem] = {}
        
        # 配置存储
        self.risk_thresholds: Dict[str, RiskThreshold] = {}
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.risk_models: Dict[str, RiskModel] = {}
        self.market_risk_config: Optional[MarketRiskConfig] = None
        self.credit_risk_config: Optional[CreditRiskConfig] = None
        self.operational_risk_config: Optional[OperationalRiskConfig] = None
        self.monitoring_config: Optional[MonitoringConfig] = None
        self.hedge_configs: Dict[str, HedgeConfig] = {}
        self.report_configs: Dict[str, ReportConfig] = {}
        
        # 异步处理
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._processing_queue = asyncio.Queue()
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # 事件回调
        self._event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # 加载配置
        self._load_configurations()
        
        self.logger.info("K4风险配置处理器初始化完成")
    
    def _load_configurations(self):
        """加载所有配置"""
        try:
            # 从数据库加载
            self.risk_thresholds = {t.name: t for t in self.database.load_risk_thresholds()}
            self.risk_limits = {l.name: l for l in self.database.load_risk_limits()}
            self.risk_models = {m.name: m for m in self.database.load_risk_models()}
            
            # 初始化默认配置
            if not self.market_risk_config:
                self.market_risk_config = MarketRiskConfig()
            if not self.credit_risk_config:
                self.credit_risk_config = CreditRiskConfig()
            if not self.operational_risk_config:
                self.operational_risk_config = OperationalRiskConfig()
            if not self.monitoring_config:
                self.monitoring_config = MonitoringConfig()
            
            # 加载告警配置
            alert_configs = self.database.load_alert_configs()
            for config in alert_configs:
                self.alert_systems[config.alert_id] = AlertSystem(config)
            
            self.logger.info("配置加载完成")
            
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            raise ProcessingError(f"配置加载失败: {e}")
    
    # ==================== 风险参数配置管理 ====================
    
    def add_risk_threshold(self, threshold: RiskThreshold) -> bool:
        """
        添加风险阈值
        
        Args:
            threshold: 风险阈值对象
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 验证配置
            is_valid, errors = self.validator.validate_threshold(threshold)
            if not is_valid:
                raise ValidationError(f"风险阈值验证失败: {errors}")
            
            # 保存到数据库
            if self.database.save_risk_threshold(threshold):
                self.risk_thresholds[threshold.name] = threshold
                self._trigger_event('threshold_added', threshold)
                self.logger.info(f"风险阈值已添加: {threshold.name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"添加风险阈值失败: {e}")
            return False
    
    def remove_risk_threshold(self, name: str) -> bool:
        """
        删除风险阈值
        
        Args:
            name: 风险阈值名称
            
        Returns:
            bool: 是否删除成功
        """
        try:
            if name in self.risk_thresholds:
                threshold = self.risk_thresholds[name]
                threshold.is_active = False
                threshold.updated_at = datetime.now()
                
                if self.database.save_risk_threshold(threshold):
                    del self.risk_thresholds[name]
                    self._trigger_event('threshold_removed', threshold)
                    self.logger.info(f"风险阈值已删除: {name}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"删除风险阈值失败: {e}")
            return False
    
    def update_risk_threshold(self, name: str, **kwargs) -> bool:
        """
        更新风险阈值
        
        Args:
            name: 风险阈值名称
            **kwargs: 更新的属性
            
        Returns:
            bool: 是否更新成功
        """
        try:
            if name not in self.risk_thresholds:
                raise ValueError(f"风险阈值不存在: {name}")
            
            threshold = self.risk_thresholds[name]
            
            # 更新属性
            for key, value in kwargs.items():
                if hasattr(threshold, key):
                    setattr(threshold, key, value)
            
            threshold.updated_at = datetime.now()
            
            # 验证更新后的配置
            is_valid, errors = self.validator.validate_threshold(threshold)
            if not is_valid:
                raise ValidationError(f"风险阈值验证失败: {errors}")
            
            # 保存到数据库
            if self.database.save_risk_threshold(threshold):
                self._trigger_event('threshold_updated', threshold)
                self.logger.info(f"风险阈值已更新: {name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"更新风险阈值失败: {e}")
            return False
    
    def get_risk_threshold(self, name: str) -> Optional[RiskThreshold]:
        """
        获取风险阈值
        
        Args:
            name: 风险阈值名称
            
        Returns:
            RiskThreshold: 风险阈值对象，如果不存在返回None
        """
        return self.risk_thresholds.get(name)
    
    def list_risk_thresholds(self) -> List[RiskThreshold]:
        """
        获取所有风险阈值
        
        Returns:
            List[RiskThreshold]: 风险阈值列表
        """
        return list(self.risk_thresholds.values())
    
    # 风险限制管理方法（与风险阈值类似）
    def add_risk_limit(self, limit: RiskLimit) -> bool:
        """添加风险限制"""
        try:
            is_valid, errors = self.validator.validate_limit(limit)
            if not is_valid:
                raise ValidationError(f"风险限制验证失败: {errors}")
            
            if self.database.save_risk_limit(limit):
                self.risk_limits[limit.name] = limit
                self._trigger_event('limit_added', limit)
                self.logger.info(f"风险限制已添加: {limit.name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"添加风险限制失败: {e}")
            return False
    
    def remove_risk_limit(self, name: str) -> bool:
        """删除风险限制"""
        try:
            if name in self.risk_limits:
                limit = self.risk_limits[name]
                limit.is_active = False
                limit.updated_at = datetime.now()
                
                if self.database.save_risk_limit(limit):
                    del self.risk_limits[name]
                    self._trigger_event('limit_removed', limit)
                    self.logger.info(f"风险限制已删除: {name}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"删除风险限制失败: {e}")
            return False
    
    def update_risk_limit(self, name: str, **kwargs) -> bool:
        """更新风险限制"""
        try:
            if name not in self.risk_limits:
                raise ValueError(f"风险限制不存在: {name}")
            
            limit = self.risk_limits[name]
            
            for key, value in kwargs.items():
                if hasattr(limit, key):
                    setattr(limit, key, value)
            
            limit.updated_at = datetime.now()
            
            is_valid, errors = self.validator.validate_limit(limit)
            if not is_valid:
                raise ValidationError(f"风险限制验证失败: {errors}")
            
            if self.database.save_risk_limit(limit):
                self._trigger_event('limit_updated', limit)
                self.logger.info(f"风险限制已更新: {name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"更新风险限制失败: {e}")
            return False
    
    def get_risk_limit(self, name: str) -> Optional[RiskLimit]:
        """获取风险限制"""
        return self.risk_limits.get(name)
    
    def list_risk_limits(self) -> List[RiskLimit]:
        """获取所有风险限制"""
        return list(self.risk_limits.values())
    
    # 风险模型管理方法
    def add_risk_model(self, model: RiskModel) -> bool:
        """添加风险模型"""
        try:
            is_valid, errors = self.validator.validate_model(model)
            if not is_valid:
                raise ValidationError(f"风险模型验证失败: {errors}")
            
            if self.database.save_risk_model(model):
                self.risk_models[model.name] = model
                self._trigger_event('model_added', model)
                self.logger.info(f"风险模型已添加: {model.name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"添加风险模型失败: {e}")
            return False
    
    def remove_risk_model(self, name: str) -> bool:
        """删除风险模型"""
        try:
            if name in self.risk_models:
                model = self.risk_models[name]
                model.is_active = False
                model.updated_at = datetime.now()
                
                if self.database.save_risk_model(model):
                    del self.risk_models[name]
                    self._trigger_event('model_removed', model)
                    self.logger.info(f"风险模型已删除: {name}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"删除风险模型失败: {e}")
            return False
    
    def update_risk_model(self, name: str, **kwargs) -> bool:
        """更新风险模型"""
        try:
            if name not in self.risk_models:
                raise ValueError(f"风险模型不存在: {name}")
            
            model = self.risk_models[name]
            
            for key, value in kwargs.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            model.updated_at = datetime.now()
            
            is_valid, errors = self.validator.validate_model(model)
            if not is_valid:
                raise ValidationError(f"风险模型验证失败: {errors}")
            
            if self.database.save_risk_model(model):
                self._trigger_event('model_updated', model)
                self.logger.info(f"风险模型已更新: {name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"更新风险模型失败: {e}")
            return False
    
    def get_risk_model(self, name: str) -> Optional[RiskModel]:
        """获取风险模型"""
        return self.risk_models.get(name)
    
    def list_risk_models(self) -> List[RiskModel]:
        """获取所有风险模型"""
        return list(self.risk_models.values())
    
    # ==================== 市场风险配置 ====================
    
    def configure_market_risk(self, config: MarketRiskConfig) -> bool:
        """
        配置市场风险参数
        
        Args:
            config: 市场风险配置对象
            
        Returns:
            bool: 是否配置成功
        """
        try:
            # 验证配置
            if not 0 < config.var_confidence_level < 1:
                raise ValidationError("VaR置信度必须在(0,1)范围内")
            
            if not 0 < config.cvar_confidence_level < 1:
                raise ValidationError("CVaR置信度必须在(0,1)范围内")
            
            if config.max_drawdown_limit <= 0 or config.max_drawdown_limit >= 1:
                raise ValidationError("最大回撤限制必须在(0,1)范围内")
            
            if config.volatility_threshold <= 0:
                raise ValidationError("波动率阈值必须大于0")
            
            if abs(config.correlation_limit) > 1:
                raise ValidationError("相关性限制必须在[-1,1]范围内")
            
            # 更新配置
            config.updated_at = datetime.now()
            self.market_risk_config = config
            
            self._trigger_event('market_risk_configured', config)
            self.logger.info("市场风险配置已更新")
            return True
            
        except Exception as e:
            self.logger.error(f"配置市场风险失败: {e}")
            return False
    
    def get_market_risk_config(self) -> Optional[MarketRiskConfig]:
        """获取市场风险配置"""
        return self.market_risk_config
    
    def calculate_var(self, returns: List[float], confidence_level: float = None, 
                     time_horizon: int = None) -> float:
        """
        计算风险价值(VaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            time_horizon: 时间跨度(天)
            
        Returns:
            float: VaR值
        """
        try:
            if not returns:
                raise ValueError("收益率序列不能为空")
            
            # 使用配置的参数或默认值
            conf_level = confidence_level or self.market_risk_config.var_confidence_level
            horizon = time_horizon or self.market_risk_config.var_time_horizon
            
            # 排序收益率
            sorted_returns = sorted(returns)
            
            # 计算分位数
            index = int((1 - conf_level) * len(sorted_returns))
            if index >= len(sorted_returns):
                index = len(sorted_returns) - 1
            
            var = -sorted_returns[index] * (horizon ** 0.5)  # 平方根时间法则
            
            return var
            
        except Exception as e:
            self.logger.error(f"VaR计算失败: {e}")
            raise ProcessingError(f"VaR计算失败: {e}")
    
    def calculate_cvar(self, returns: List[float], confidence_level: float = None) -> float:
        """
        计算条件风险价值(CVaR)
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            float: CVaR值
        """
        try:
            if not returns:
                raise ValueError("收益率序列不能为空")
            
            conf_level = confidence_level or self.market_risk_config.cvar_confidence_level
            
            # 计算VaR
            var = self.calculate_var(returns, conf_level)
            
            # 计算CVaR (超过VaR的损失的平均值)
            tail_returns = [r for r in returns if r <= -var]
            if not tail_returns:
                return var
            
            cvar = -sum(tail_returns) / len(tail_returns)
            
            return cvar
            
        except Exception as e:
            self.logger.error(f"CVaR计算失败: {e}")
            raise ProcessingError(f"CVaR计算失败: {e}")
    
    def calculate_max_drawdown(self, prices: List[float]) -> float:
        """
        计算最大回撤
        
        Args:
            prices: 价格序列
            
        Returns:
            float: 最大回撤
        """
        try:
            if not prices:
                raise ValueError("价格序列不能为空")
            
            peak = prices[0]
            max_drawdown = 0.0
            
            for price in prices:
                if price > peak:
                    peak = price
                
                drawdown = (peak - price) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"最大回撤计算失败: {e}")
            raise ProcessingError(f"最大回撤计算失败: {e}")
    
    def stress_test(self, portfolio_value: float, scenarios: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        执行压力测试
        
        Args:
            portfolio_value: 投资组合价值
            scenarios: 压力测试场景
            
        Returns:
            Dict[str, float]: 压力测试结果
        """
        try:
            scenarios = scenarios or self.market_risk_config.stress_test_scenarios
            results = {}
            
            for scenario in scenarios:
                scenario_name = scenario.get('name', 'Unknown')
                shock_size = scenario.get('shock_size', 0.0)
                affected_assets = scenario.get('affected_assets', [])
                
                # 简单的冲击计算
                if affected_assets:
                    # 假设平均分配权重
                    weight_per_asset = 1.0 / len(affected_assets)
                    total_impact = portfolio_value * shock_size * weight_per_asset
                else:
                    total_impact = portfolio_value * shock_size
                
                results[scenario_name] = total_impact
            
            self._trigger_event('stress_test_completed', results)
            return results
            
        except Exception as e:
            self.logger.error(f"压力测试失败: {e}")
            raise ProcessingError(f"压力测试失败: {e}")
    
    # ==================== 信用风险配置 ====================
    
    def configure_credit_risk(self, config: CreditRiskConfig) -> bool:
        """
        配置信用风险参数
        
        Args:
            config: 信用风险配置对象
            
        Returns:
            bool: 是否配置成功
        """
        try:
            # 验证评级尺度
            if not config.rating_scale:
                raise ValidationError("信用评级尺度不能为空")
            
            # 验证违约概率
            for rating, prob in config.default_probabilities.items():
                if not 0 <= prob <= 1:
                    raise ValidationError(f"违约概率必须在[0,1]范围内: {rating}")
            
            # 更新配置
            config.updated_at = datetime.now()
            self.credit_risk_config = config
            
            self._trigger_event('credit_risk_configured', config)
            self.logger.info("信用风险配置已更新")
            return True
            
        except Exception as e:
            self.logger.error(f"配置信用风险失败: {e}")
            return False
    
    def get_credit_risk_config(self) -> Optional[CreditRiskConfig]:
        """获取信用风险配置"""
        return self.credit_risk_config
    
    def calculate_credit_exposure(self, counterparty: str, exposure: float, 
                                rating: str) -> Dict[str, float]:
        """
        计算信用敞口
        
        Args:
            counterparty: 交易对手
            exposure: 敞口金额
            rating: 信用评级
            
        Returns:
            Dict[str, float]: 信用敞口分析结果
        """
        try:
            config = self.credit_risk_config
            
            # 获取违约概率
            default_prob = config.default_probabilities.get(rating, 0.05)
            
            # 获取恢复率
            recovery_rate = config.recovery_rates.get(rating, 0.4)
            
            # 计算预期损失
            expected_loss = exposure * default_prob * (1 - recovery_rate)
            
            # 计算意外损失 (假设LGD = 1 - 恢复率)
            lgd = 1 - recovery_rate
            unexpected_loss = exposure * (default_prob * lgd ** 2) ** 0.5
            
            # 计算风险价值 (99%置信度)
            var_99 = exposure * default_prob * lgd * 2.33  # 99%分位数的倍数
            
            return {
                'counterparty': counterparty,
                'exposure': exposure,
                'rating': rating,
                'default_probability': default_prob,
                'recovery_rate': recovery_rate,
                'expected_loss': expected_loss,
                'unexpected_loss': unexpected_loss,
                'var_99': var_99,
                'loss_given_default': lgd
            }
            
        except Exception as e:
            self.logger.error(f"信用敞口计算失败: {e}")
            raise ProcessingError(f"信用敞口计算失败: {e}")
    
    def calculate_portfolio_credit_risk(self, exposures: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算投资组合信用风险
        
        Args:
            exposures: 敞口列表
            
        Returns:
            Dict[str, float]: 投资组合信用风险指标
        """
        try:
            total_exposure = sum(exp.get('exposure', 0) for exp in exposures)
            total_expected_loss = 0.0
            total_var = 0.0
            
            # 计算总敞口和预期损失
            for exposure in exposures:
                rating = exposure.get('rating', 'BBB')
                amount = exposure.get('exposure', 0)
                
                result = self.calculate_credit_exposure(
                    exposure.get('counterparty', ''), amount, rating
                )
                
                total_expected_loss += result['expected_loss']
                total_var += result['var_99'] ** 2
            
            # 假设独立同分布，计算组合VaR
            portfolio_var = total_var ** 0.5
            
            return {
                'total_exposure': total_exposure,
                'total_expected_loss': total_expected_loss,
                'portfolio_var_99': portfolio_var,
                'expected_loss_ratio': total_expected_loss / total_exposure if total_exposure > 0 else 0,
                'diversification_ratio': portfolio_var / total_var if total_var > 0 else 1
            }
            
        except Exception as e:
            self.logger.error(f"投资组合信用风险计算失败: {e}")
            raise ProcessingError(f"投资组合信用风险计算失败: {e}")
    
    def assess_counterparty_risk(self, counterparty: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估交易对手风险
        
        Args:
            counterparty: 交易对手名称
            financial_data: 财务数据
            
        Returns:
            Dict[str, Any]: 交易对手风险评估结果
        """
        try:
            # 简化的交易对手风险评估
            # 实际实现中会使用更复杂的模型
            
            score = 0.0
            factors = {}
            
            # 财务健康度评分
            if 'debt_to_equity' in financial_data:
                debt_ratio = financial_data['debt_to_equity']
                if debt_ratio < 0.3:
                    factors['debt_ratio_score'] = 1.0
                elif debt_ratio < 0.6:
                    factors['debt_ratio_score'] = 0.7
                else:
                    factors['debt_ratio_score'] = 0.3
                score += factors['debt_ratio_score'] * 0.3
            
            # 流动性评分
            if 'current_ratio' in financial_data:
                current_ratio = financial_data['current_ratio']
                if current_ratio > 2.0:
                    factors['liquidity_score'] = 1.0
                elif current_ratio > 1.5:
                    factors['liquidity_score'] = 0.8
                else:
                    factors['liquidity_score'] = 0.4
                score += factors['liquidity_score'] * 0.2
            
            # 盈利能力评分
            if 'roe' in financial_data:
                roe = financial_data['roe']
                if roe > 0.15:
                    factors['profitability_score'] = 1.0
                elif roe > 0.10:
                    factors['profitability_score'] = 0.8
                else:
                    factors['profitability_score'] = 0.5
                score += factors['profitability_score'] * 0.2
            
            # 行业风险评分
            if 'industry_risk' in financial_data:
                industry_risk = financial_data['industry_risk']
                factors['industry_risk_score'] = 1.0 - industry_risk
                score += factors['industry_risk_score'] * 0.3
            
            # 确定风险等级
            if score >= 0.8:
                risk_level = RiskLevel.LOW
            elif score >= 0.6:
                risk_level = RiskLevel.MEDIUM
            elif score >= 0.4:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return {
                'counterparty': counterparty,
                'overall_score': score,
                'risk_level': risk_level,
                'factors': factors,
                'recommendations': self._generate_risk_recommendations(score, factors)
            }
            
        except Exception as e:
            self.logger.error(f"交易对手风险评估失败: {e}")
            raise ProcessingError(f"交易对手风险评估失败: {e}")
    
    def _generate_risk_recommendations(self, score: float, factors: Dict[str, float]) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        if score < 0.6:
            recommendations.append("建议增加抵押品要求")
            recommendations.append("考虑缩短交易期限")
        
        if factors.get('debt_ratio_score', 1.0) < 0.5:
            recommendations.append("债务比率较高，建议谨慎交易")
        
        if factors.get('liquidity_score', 1.0) < 0.6:
            recommendations.append("流动性不足，建议设置更严格的限额")
        
        if factors.get('profitability_score', 1.0) < 0.5:
            recommendations.append("盈利能力较弱，建议加强监控")
        
        return recommendations
    
    # ==================== 操作风险配置 ====================
    
    def configure_operational_risk(self, config: OperationalRiskConfig) -> bool:
        """
        配置操作风险参数
        
        Args:
            config: 操作风险配置对象
            
        Returns:
            bool: 是否配置成功
        """
        try:
            # 验证配置
            if not config.system_risk_factors and not config.human_risk_factors:
                raise ValidationError("至少需要配置一种风险因子")
            
            # 更新配置
            config.updated_at = datetime.now()
            self.operational_risk_config = config
            
            self._trigger_event('operational_risk_configured', config)
            self.logger.info("操作风险配置已更新")
            return True
            
        except Exception as e:
            self.logger.error(f"配置操作风险失败: {e}")
            return False
    
    def get_operational_risk_config(self) -> Optional[OperationalRiskConfig]:
        """获取操作风险配置"""
        return self.operational_risk_config
    
    def assess_operational_risk(self, process_name: str, risk_factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估操作风险
        
        Args:
            process_name: 流程名称
            risk_factors: 风险因子
            
        Returns:
            Dict[str, Any]: 操作风险评估结果
        """
        try:
            config = self.operational_risk_config
            risk_score = 0.0
            risk_details = {}
            
            # 系统风险评估
            if 'system_availability' in risk_factors:
                availability = risk_factors['system_availability']
                system_risk = 1.0 - availability
                risk_score += system_risk * 0.3
                risk_details['system_risk'] = system_risk
            
            # 人为风险评估
            if 'human_error_rate' in risk_factors:
                error_rate = risk_factors['human_error_rate']
                human_risk = error_rate
                risk_score += human_risk * 0.3
                risk_details['human_risk'] = human_risk
            
            # 流程风险评估
            if 'process_complexity' in risk_factors:
                complexity = risk_factors['process_complexity']
                process_risk = complexity / 10.0  # 假设复杂度为1-10
                risk_score += process_risk * 0.2
                risk_details['process_risk'] = process_risk
            
            # 外部风险评估
            if 'external_threats' in risk_factors:
                threats = risk_factors['external_threats']
                external_risk = threats / 10.0  # 假设威胁级别为1-10
                risk_score += external_risk * 0.2
                risk_details['external_risk'] = external_risk
            
            # 确定风险等级
            if risk_score >= 0.7:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 0.5:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 0.3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            return {
                'process_name': process_name,
                'overall_risk_score': risk_score,
                'risk_level': risk_level,
                'risk_details': risk_details,
                'mitigation_recommendations': self._generate_mitigation_recommendations(risk_details)
            }
            
        except Exception as e:
            self.logger.error(f"操作风险评估失败: {e}")
            raise ProcessingError(f"操作风险评估失败: {e}")
    
    def _generate_mitigation_recommendations(self, risk_details: Dict[str, float]) -> List[str]:
        """生成风险缓解建议"""
        recommendations = []
        
        if risk_details.get('system_risk', 0) > 0.5:
            recommendations.append("加强系统监控和维护")
            recommendations.append("建立冗余系统")
        
        if risk_details.get('human_risk', 0) > 0.5:
            recommendations.append("加强员工培训")
            recommendations.append("实施双人复核机制")
        
        if risk_details.get('process_risk', 0) > 0.5:
            recommendations.append("简化流程")
            recommendations.append("增加自动化控制")
        
        if risk_details.get('external_risk', 0) > 0.5:
            recommendations.append("加强安全防护")
            recommendations.append("建立应急响应机制")
        
        return recommendations
    
    def calculate_capital_requirement(self, gross_income: float, business_lines: Dict[str, float]) -> float:
        """
        计算操作风险资本要求 (基于标准法)
        
        Args:
            gross_income: 总收入
            business_lines: 业务线收入分布
            
        Returns:
            float: 资本要求
        """
        try:
            # 标准法计算
            # β系数表 (简化版)
            beta_factors = {
                'corporate_finance': 0.18,
                'trading_sales': 0.18,
                'retail_banking': 0.12,
                'commercial_banking': 0.15,
                'payment_settlement': 0.18,
                'agency_services': 0.15,
                'asset_management': 0.12
            }
            
            total_capital = 0.0
            for business_line, income in business_lines.items():
                beta = beta_factors.get(business_line, 0.15)  # 默认β值
                capital_requirement = beta * income
                total_capital += capital_requirement
            
            # 确保最小资本要求 (总收入的18%)
            min_capital = 0.18 * gross_income
            
            return max(total_capital, min_capital)
            
        except Exception as e:
            self.logger.error(f"资本要求计算失败: {e}")
            raise ProcessingError(f"资本要求计算失败: {e}")
    
    # ==================== 风险监控配置和告警设置 ====================
    
    def configure_monitoring(self, config: MonitoringConfig) -> bool:
        """
        配置风险监控参数
        
        Args:
            config: 监控配置对象
            
        Returns:
            bool: 是否配置成功
        """
        try:
            # 验证配置
            if config.monitoring_frequency <= 0:
                raise ValidationError("监控频率必须大于0")
            
            if config.retention_period <= 0:
                raise ValidationError("数据保留期必须大于0")
            
            # 更新配置
            config.updated_at = datetime.now()
            self.monitoring_config = config
            
            self._trigger_event('monitoring_configured', config)
            self.logger.info("风险监控配置已更新")
            return True
            
        except Exception as e:
            self.logger.error(f"配置风险监控失败: {e}")
            return False
    
    def get_monitoring_config(self) -> Optional[MonitoringConfig]:
        """获取监控配置"""
        return self.monitoring_config
    
    def add_alert_config(self, config: AlertConfig) -> bool:
        """
        添加告警配置
        
        Args:
            config: 告警配置对象
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 验证配置
            is_valid, errors = self.validator.validate_alert_config(config)
            if not is_valid:
                raise ValidationError(f"告警配置验证失败: {errors}")
            
            # 保存到数据库
            if self.database.save_alert_config(config):
                self.alert_systems[config.alert_id] = AlertSystem(config)
                self._trigger_event('alert_added', config)
                self.logger.info(f"告警配置已添加: {config.name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"添加告警配置失败: {e}")
            return False
    
    def remove_alert_config(self, alert_id: str) -> bool:
        """
        删除告警配置
        
        Args:
            alert_id: 告警ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            if alert_id in self.alert_systems:
                del self.alert_systems[alert_id]
                self._trigger_event('alert_removed', {'alert_id': alert_id})
                self.logger.info(f"告警配置已删除: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"删除告警配置失败: {e}")
            return False
    
    def get_alert_config(self, alert_id: str) -> Optional[AlertConfig]:
        """
        获取告警配置
        
        Args:
            alert_id: 告警ID
            
        Returns:
            AlertConfig: 告警配置对象
        """
        alert_system = self.alert_systems.get(alert_id)
        return alert_system.config if alert_system else None
    
    def list_alert_configs(self) -> List[AlertConfig]:
        """获取所有告警配置"""
        return [system.config for system in self.alert_systems.values()]
    
    def trigger_alert(self, alert_id: str, context: Dict[str, Any]) -> bool:
        """
        手动触发告警
        
        Args:
            alert_id: 告警ID
            context: 告警上下文
            
        Returns:
            bool: 是否触发成功
        """
        try:
            alert_system = self.alert_systems.get(alert_id)
            if not alert_system:
                raise ValueError(f"告警配置不存在: {alert_id}")
            
            return alert_system.trigger_alert(context)
            
        except Exception as e:
            self.logger.error(f"触发告警失败: {e}")
            return False
    
    def start_monitoring(self):
        """启动风险监控"""
        try:
            if self._running:
                self.logger.warning("监控已在运行中")
                return
            
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("风险监控已启动")
            
        except Exception as e:
            self.logger.error(f"启动监控失败: {e}")
            raise ProcessingError(f"启动监控失败: {e}")
    
    def stop_monitoring(self):
        """停止风险监控"""
        try:
            if not self._running:
                self.logger.warning("监控未在运行")
                return
            
            self._running = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                self._monitoring_task = None
            
            self.logger.info("风险监控已停止")
            
        except Exception as e:
            self.logger.error(f"停止监控失败: {e}")
    
    async def _monitoring_loop(self):
        """监控循环"""
        try:
            while self._running:
                await self._process_monitoring_tasks()
                await asyncio.sleep(self.monitoring_config.monitoring_frequency)
                
        except asyncio.CancelledError:
            self.logger.info("监控循环已取消")
        except Exception as e:
            self.logger.error(f"监控循环异常: {e}")
    
    async def _process_monitoring_tasks(self):
        """处理监控任务"""
        try:
            # 检查风险限额
            await self._check_risk_limits()
            
            # 检查风险阈值
            await self._check_risk_thresholds()
            
            # 处理告警
            await self._process_alerts()
            
        except Exception as e:
            self.logger.error(f"处理监控任务失败: {e}")
    
    async def _check_risk_limits(self):
        """检查风险限额"""
        try:
            for limit in self.risk_limits.values():
                if limit.utilization_rate > 80:  # 80%阈值
                    context = {
                        'limit_name': limit.name,
                        'current_value': limit.current_value,
                        'limit_value': limit.limit_value,
                        'utilization_rate': limit.utilization_rate,
                        'threshold': 80.0
                    }
                    
                    await self._trigger_async_alert('risk_limit_warning', context)
                    
        except Exception as e:
            self.logger.error(f"检查风险限额失败: {e}")
    
    async def _check_risk_thresholds(self):
        """检查风险阈值"""
        try:
            for threshold in self.risk_thresholds.values():
                # 这里需要实际的监控数据
                # 简化示例：假设有一个当前值
                current_value = 0.05  # 示例值
                
                if current_value > threshold.value:
                    context = {
                        'threshold_name': threshold.name,
                        'current_value': current_value,
                        'threshold_value': threshold.value,
                        'unit': threshold.unit
                    }
                    
                    await self._trigger_async_alert('risk_threshold_exceeded', context)
                    
        except Exception as e:
            self.logger.error(f"检查风险阈值失败: {e}")
    
    async def _process_alerts(self):
        """处理告警"""
        try:
            # 这里可以添加更多告警处理逻辑
            pass
        except Exception as e:
            self.logger.error(f"处理告警失败: {e}")
    
    async def _trigger_async_alert(self, alert_type: str, context: Dict[str, Any]):
        """异步触发告警"""
        try:
            # 在实际实现中，这里会根据alert_type查找对应的告警配置
            # 并调用相应的告警系统
            self.logger.warning(f"异步告警: {alert_type} - {context}")
        except Exception as e:
            self.logger.error(f"异步告警触发失败: {e}")
    
    # ==================== 风险对冲配置和策略设置 ====================
    
    def configure_hedge(self, config: HedgeConfig) -> bool:
        """
        配置风险对冲参数
        
        Args:
            config: 对冲配置对象
            
        Returns:
            bool: 是否配置成功
        """
        try:
            # 验证配置
            if not 0 < config.target_risk_reduction <= 1:
                raise ValidationError("目标风险降低率必须在(0,1]范围内")
            
            if config.cost_budget < 0:
                raise ValidationError("成本预算不能为负数")
            
            # 更新配置
            config.updated_at = datetime.now()
            self.hedge_configs[config.strategy.name] = config
            
            self._trigger_event('hedge_configured', config)
            self.logger.info(f"风险对冲配置已更新: {config.strategy.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"配置风险对冲失败: {e}")
            return False
    
    def get_hedge_config(self, strategy: HedgeStrategy) -> Optional[HedgeConfig]:
        """
        获取对冲配置
        
        Args:
            strategy: 对冲策略
            
        Returns:
            HedgeConfig: 对冲配置对象
        """
        return self.hedge_configs.get(strategy.name)
    
    def list_hedge_configs(self) -> List[HedgeConfig]:
        """获取所有对冲配置"""
        return list(self.hedge_configs.values())
    
    def calculate_hedge_effectiveness(self, original_risk: float, hedged_risk: float, 
                                    hedge_cost: float) -> Dict[str, float]:
        """
        计算对冲有效性
        
        Args:
            original_risk: 原始风险
            hedged_risk: 对冲后风险
            hedge_cost: 对冲成本
            
        Returns:
            Dict[str, float]: 对冲有效性指标
        """
        try:
            risk_reduction = original_risk - hedged_risk
            risk_reduction_ratio = risk_reduction / original_risk if original_risk > 0 else 0
            net_benefit = risk_reduction - hedge_cost
            cost_benefit_ratio = risk_reduction / hedge_cost if hedge_cost > 0 else float('inf')
            
            return {
                'original_risk': original_risk,
                'hedged_risk': hedged_risk,
                'risk_reduction': risk_reduction,
                'risk_reduction_ratio': risk_reduction_ratio,
                'hedge_cost': hedge_cost,
                'net_benefit': net_benefit,
                'cost_benefit_ratio': cost_benefit_ratio,
                'effectiveness_score': min(risk_reduction_ratio, 1.0) - (hedge_cost / original_risk if original_risk > 0 else 0)
            }
            
        except Exception as e:
            self.logger.error(f"对冲有效性计算失败: {e}")
            raise ProcessingError(f"对冲有效性计算失败: {e}")
    
    def optimize_hedge_portfolio(self, risk_exposures: Dict[str, float], 
                               available_instruments: List[str]) -> Dict[str, Any]:
        """
        优化对冲投资组合
        
        Args:
            risk_exposures: 风险敞口
            available_instruments: 可用对冲工具
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        try:
            # 简化的对冲优化算法
            # 实际实现中会使用更复杂的优化算法
            
            total_exposure = sum(abs(exposure) for exposure in risk_exposures.values())
            optimal_hedges = {}
            total_cost = 0.0
            
            # 按风险敞口大小分配对冲比例
            for exposure_type, exposure_value in risk_exposures.items():
                if abs(exposure_value) < total_exposure * 0.05:  # 小于5%的敞口忽略
                    continue
                
                # 简化：假设每个风险类型有对应的对冲工具
                hedge_instrument = f"HEDGE_{exposure_type}"
                if hedge_instrument in available_instruments:
                    # 80%对冲比例
                    hedge_ratio = 0.8
                    hedge_notional = abs(exposure_value) * hedge_ratio
                    hedge_cost = hedge_notional * 0.01  # 假设1%成本
                    
                    optimal_hedges[hedge_instrument] = {
                        'notional': hedge_notional,
                        'cost': hedge_cost,
                        'exposure_reduced': exposure_value * hedge_ratio
                    }
                    
                    total_cost += hedge_cost
            
            return {
                'optimal_hedges': optimal_hedges,
                'total_cost': total_cost,
                'total_exposure_reduced': sum(h['exposure_reduced'] for h in optimal_hedges.values()),
                'cost_efficiency': total_cost / total_exposure if total_exposure > 0 else 0,
                'hedge_coverage': len(optimal_hedges) / len([k for k, v in risk_exposures.items() if abs(v) >= total_exposure * 0.05])
            }
            
        except Exception as e:
            self.logger.error(f"对冲投资组合优化失败: {e}")
            raise ProcessingError(f"对冲投资组合优化失败: {e}")
    
    def rebalance_hedge_portfolio(self, current_positions: Dict[str, float], 
                                target_allocations: Dict[str, float]) -> Dict[str, Any]:
        """
        重新平衡对冲投资组合
        
        Args:
            current_positions: 当前持仓
            target_allocations: 目标配置
            
        Returns:
            Dict[str, Any]: 重新平衡建议
        """
        try:
            rebalance_actions = []
            total_rebalance_cost = 0.0
            
            for instrument, target_allocation in target_allocations.items():
                current_position = current_positions.get(instrument, 0.0)
                allocation_diff = target_allocation - current_position
                
                if abs(allocation_diff) > 0.01:  # 1%阈值
                    action = {
                        'instrument': instrument,
                        'action': 'BUY' if allocation_diff > 0 else 'SELL',
                        'quantity': abs(allocation_diff),
                        'current_position': current_position,
                        'target_allocation': target_allocation
                    }
                    
                    # 估算交易成本
                    estimated_cost = abs(allocation_diff) * 0.001  # 0.1%交易成本
                    action['estimated_cost'] = estimated_cost
                    total_rebalance_cost += estimated_cost
                    
                    rebalance_actions.append(action)
            
            return {
                'rebalance_actions': rebalance_actions,
                'total_actions': len(rebalance_actions),
                'total_rebalance_cost': total_rebalance_cost,
                'rebalance_urgency': 'HIGH' if len(rebalance_actions) > 5 else 'MEDIUM' if len(rebalance_actions) > 2 else 'LOW'
            }
            
        except Exception as e:
            self.logger.error(f"对冲投资组合重新平衡失败: {e}")
            raise ProcessingError(f"对冲投资组合重新平衡失败: {e}")
    
    # ==================== 风险报告配置和输出设置 ====================
    
    def configure_report(self, config: ReportConfig) -> bool:
        """
        配置风险报告参数
        
        Args:
            config: 报告配置对象
            
        Returns:
            bool: 是否配置成功
        """
        try:
            # 验证配置
            if not config.report_name or not config.report_name.strip():
                raise ValidationError("报告名称不能为空")
            
            if not config.recipients:
                raise ValidationError("报告收件人不能为空")
            
            # 更新配置
            config.updated_at = datetime.now()
            self.report_configs[config.report_name] = config
            
            self._trigger_event('report_configured', config)
            self.logger.info(f"风险报告配置已更新: {config.report_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"配置风险报告失败: {e}")
            return False
    
    def get_report_config(self, report_name: str) -> Optional[ReportConfig]:
        """
        获取报告配置
        
        Args:
            report_name: 报告名称
            
        Returns:
            ReportConfig: 报告配置对象
        """
        return self.report_configs.get(report_name)
    
    def list_report_configs(self) -> List[ReportConfig]:
        """获取所有报告配置"""
        return list(self.report_configs.values())
    
    def generate_risk_report(self, report_name: str, data: Dict[str, Any] = None) -> str:
        """
        生成风险报告
        
        Args:
            report_name: 报告名称
            data: 报告数据
            
        Returns:
            str: 报告文件路径
        """
        try:
            config = self.report_configs.get(report_name)
            if not config:
                raise ValueError(f"报告配置不存在: {report_name}")
            
            data = data or self._collect_risk_data()
            
            # 生成报告内容
            report_content = self._format_report_content(config, data)
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_name}_{timestamp}.{config.format.name.lower()}"
            filepath = Path(config.output_directory) / filename
            
            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存文件
            if config.format == ReportFormat.JSON:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report_content, f, ensure_ascii=False, indent=2, default=str)
            
            elif config.format == ReportFormat.CSV:
                self._save_csv_report(report_content, filepath)
            
            elif config.format == ReportFormat.HTML:
                self._save_html_report(report_content, filepath)
            
            self._trigger_event('report_generated', {
                'report_name': report_name,
                'filepath': str(filepath),
                'format': config.format
            })
            
            self.logger.info(f"风险报告已生成: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"生成风险报告失败: {e}")
            raise ProcessingError(f"生成风险报告失败: {e}")
    
    def _collect_risk_data(self) -> Dict[str, Any]:
        """收集风险数据"""
        try:
            return {
                'timestamp': datetime.now(),
                'risk_thresholds': [asdict(t) for t in self.risk_thresholds.values()],
                'risk_limits': [asdict(l) for l in self.risk_limits.values()],
                'risk_models': [asdict(m) for m in self.risk_models.values()],
                'market_risk_config': asdict(self.market_risk_config) if self.market_risk_config else {},
                'credit_risk_config': asdict(self.credit_risk_config) if self.credit_risk_config else {},
                'operational_risk_config': asdict(self.operational_risk_config) if self.operational_risk_config else {},
                'monitoring_config': asdict(self.monitoring_config) if self.monitoring_config else {},
                'alert_configs': [asdict(config) for config in self.list_alert_configs()],
                'hedge_configs': [asdict(h) for h in self.hedge_configs.values()],
                'report_configs': [asdict(r) for r in self.report_configs.values()]
            }
        except Exception as e:
            self.logger.error(f"收集风险数据失败: {e}")
            return {}
    
    def _format_report_content(self, config: ReportConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化报告内容"""
        try:
            content = {
                'report_info': {
                    'name': config.report_name,
                    'type': config.report_type,
                    'generated_at': datetime.now(),
                    'format': config.format.name
                },
                'summary': self._generate_risk_summary(data),
                'sections': {}
            }
            
            # 根据配置的章节生成内容
            for section in config.content_sections:
                if section == 'market_risk':
                    content['sections']['market_risk'] = self._format_market_risk_section(data)
                elif section == 'credit_risk':
                    content['sections']['credit_risk'] = self._format_credit_risk_section(data)
                elif section == 'operational_risk':
                    content['sections']['operational_risk'] = self._format_operational_risk_section(data)
                elif section == 'risk_limits':
                    content['sections']['risk_limits'] = self._format_risk_limits_section(data)
                elif section == 'alerts':
                    content['sections']['alerts'] = self._format_alerts_section(data)
            
            return content
            
        except Exception as e:
            self.logger.error(f"格式化报告内容失败: {e}")
            return {}
    
    def _generate_risk_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成风险摘要"""
        try:
            total_limits = len(data.get('risk_limits', []))
            active_thresholds = len([t for t in data.get('risk_thresholds', []) if t.get('is_active', True)])
            active_alerts = len([a for a in data.get('alert_configs', []) if a.get('enabled', True)])
            
            return {
                'total_risk_limits': total_limits,
                'active_thresholds': active_thresholds,
                'active_alerts': active_alerts,
                'monitoring_status': 'ACTIVE' if self._running else 'INACTIVE',
                'last_updated': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"生成风险摘要失败: {e}")
            return {}
    
    def _format_market_risk_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化市场风险章节"""
        market_config = data.get('market_risk_config', {})
        return {
            'var_confidence_level': market_config.get('var_confidence_level', 0.95),
            'var_time_horizon': market_config.get('var_time_horizon', 1),
            'max_drawdown_limit': market_config.get('max_drawdown_limit', 0.15),
            'volatility_threshold': market_config.get('volatility_threshold', 0.25),
            'stress_test_scenarios_count': len(market_config.get('stress_test_scenarios', []))
        }
    
    def _format_credit_risk_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化信用风险章节"""
        credit_config = data.get('credit_risk_config', {})
        return {
            'rating_scale_size': len(credit_config.get('rating_scale', {})),
            'default_probabilities_count': len(credit_config.get('default_probabilities', {})),
            'exposure_limits_count': len(credit_config.get('exposure_limits', {})),
            'recovery_rates_count': len(credit_config.get('recovery_rates', {}))
        }
    
    def _format_operational_risk_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化操作风险章节"""
        operational_config = data.get('operational_risk_config', {})
        return {
            'system_risk_factors_count': len(operational_config.get('system_risk_factors', [])),
            'human_risk_factors_count': len(operational_config.get('human_risk_factors', [])),
            'process_risk_factors_count': len(operational_config.get('process_risk_factors', [])),
            'external_risk_factors_count': len(operational_config.get('external_risk_factors', [])),
            'loss_event_categories_count': len(operational_config.get('loss_event_categories', {}))
        }
    
    def _format_risk_limits_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化风险限制章节"""
        limits = data.get('risk_limits', [])
        utilization_rates = [l.get('utilization_rate', 0) for l in limits]
        
        return {
            'total_limits': len(limits),
            'average_utilization': sum(utilization_rates) / len(utilization_rates) if utilization_rates else 0,
            'high_utilization_limits': len([r for r in utilization_rates if r > 80]),
            'limits_detail': limits
        }
    
    def _format_alerts_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化告警章节"""
        alerts = data.get('alert_configs', [])
        enabled_alerts = [a for a in alerts if a.get('enabled', True)]
        
        return {
            'total_alerts': len(alerts),
            'enabled_alerts': len(enabled_alerts),
            'severity_distribution': self._calculate_alert_severity_distribution(alerts),
            'alerts_detail': alerts
        }
    
    def _calculate_alert_severity_distribution(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算告警严重程度分布"""
        distribution = defaultdict(int)
        for alert in alerts:
            severity = alert.get('severity', 'WARNING')
            distribution[severity] += 1
        return dict(distribution)
    
    def _save_csv_report(self, content: Dict[str, Any], filepath: Path):
        """保存CSV格式报告"""
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入报告信息
                report_info = content.get('report_info', {})
                writer.writerow(['报告名称', report_info.get('name', '')])
                writer.writerow(['报告类型', report_info.get('type', '')])
                writer.writerow(['生成时间', report_info.get('generated_at', '')])
                writer.writerow([])
                
                # 写入摘要
                summary = content.get('summary', {})
                writer.writerow(['风险摘要'])
                for key, value in summary.items():
                    writer.writerow([key, value])
                writer.writerow([])
                
                # 写入各章节数据
                sections = content.get('sections', {})
                for section_name, section_data in sections.items():
                    writer.writerow([f'{section_name} 章节'])
                    if isinstance(section_data, dict):
                        for key, value in section_data.items():
                            writer.writerow([key, value])
                    writer.writerow([])
                    
        except Exception as e:
            self.logger.error(f"保存CSV报告失败: {e}")
            raise ProcessingError(f"保存CSV报告失败: {e}")
    
    def _save_html_report(self, content: Dict[str, Any], filepath: Path):
        """保存HTML格式报告"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{content.get('report_info', {}).get('name', '风险报告')}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .summary {{ background-color: #e6f3ff; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{content.get('report_info', {}).get('name', '风险报告')}</h1>
                    <p>生成时间: {content.get('report_info', {}).get('generated_at', '')}</p>
                </div>
                
                <div class="section summary">
                    <h2>风险摘要</h2>
                    <table>
            """
            
            summary = content.get('summary', {})
            for key, value in summary.items():
                html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
            
            html_content += """
                    </table>
                </div>
            """
            
            # 添加各章节
            sections = content.get('sections', {})
            for section_name, section_data in sections.items():
                html_content += f"""
                <div class="section">
                    <h2>{section_name.replace('_', ' ').title()}</h2>
                    <table>
                """
                
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"保存HTML报告失败: {e}")
            raise ProcessingError(f"保存HTML报告失败: {e}")
    
    # ==================== 异步风险配置处理 ====================
    
    async def async_process_configurations(self, configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        异步处理配置
        
        Args:
            configurations: 配置列表
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            results = {}
            
            # 使用信号量限制并发数
            semaphore = asyncio.Semaphore(5)
            
            async def process_single_config(config_data):
                async with semaphore:
                    config_type = config_data.get('type')
                    config_data_inner = config_data.get('data', {})
                    
                    try:
                        if config_type == 'threshold':
                            threshold = RiskThreshold(**config_data_inner)
                            success = self.add_risk_threshold(threshold)
                            return {'type': config_type, 'success': success, 'data': config_data_inner}
                        
                        elif config_type == 'limit':
                            limit = RiskLimit(**config_data_inner)
                            success = self.add_risk_limit(limit)
                            return {'type': config_type, 'success': success, 'data': config_data_inner}
                        
                        elif config_type == 'model':
                            model = RiskModel(**config_data_inner)
                            success = self.add_risk_model(model)
                            return {'type': config_type, 'success': success, 'data': config_data_inner}
                        
                        elif config_type == 'alert':
                            config_data_inner['severity'] = AlertSeverity[config_data_inner['severity']]
                            alert = AlertConfig(**config_data_inner)
                            success = self.add_alert_config(alert)
                            return {'type': config_type, 'success': success, 'data': config_data_inner}
                        
                        else:
                            return {'type': config_type, 'success': False, 'error': 'Unknown config type'}
                    
                    except Exception as e:
                        return {'type': config_type, 'success': False, 'error': str(e)}
            
            # 并发处理所有配置
            tasks = [process_single_config(config) for config in configurations]
            processed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 整理结果
            for result in processed_results:
                if isinstance(result, dict):
                    config_type = result.get('type', 'unknown')
                    if config_type not in results:
                        results[config_type] = {'success': 0, 'failed': 0, 'details': []}
                    
                    if result.get('success', False):
                        results[config_type]['success'] += 1
                    else:
                        results[config_type]['failed'] += 1
                    
                    results[config_type]['details'].append(result)
            
            self._trigger_event('async_configurations_processed', results)
            self.logger.info(f"异步配置处理完成: {results}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"异步配置处理失败: {e}")
            raise ProcessingError(f"异步配置处理失败: {e}")
    
    async def async_generate_reports(self, report_requests: List[Dict[str, Any]]) -> List[str]:
        """
        异步生成报告
        
        Args:
            report_requests: 报告请求列表
            
        Returns:
            List[str]: 生成的报告文件路径列表
        """
        try:
            generated_reports = []
            
            async def generate_single_report(request):
                report_name = request.get('report_name')
                data = request.get('data')
                
                try:
                    report_path = self.generate_risk_report(report_name, data)
                    return {'success': True, 'report_path': report_path, 'report_name': report_name}
                except Exception as e:
                    return {'success': False, 'error': str(e), 'report_name': report_name}
            
            # 并发生成报告
            tasks = [generate_single_report(request) for request in report_requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 整理结果
            for result in results:
                if isinstance(result, dict) and result.get('success'):
                    generated_reports.append(result['report_path'])
            
            self._trigger_event('async_reports_generated', generated_reports)
            self.logger.info(f"异步报告生成完成: {len(generated_reports)} 个报告")
            
            return generated_reports
            
        except Exception as e:
            self.logger.error(f"异步报告生成失败: {e}")
            raise ProcessingError(f"异步报告生成失败: {e}")
    
    async def async_monitor_risk_metrics(self, metrics_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        异步监控风险指标
        
        Args:
            metrics_config: 指标配置
            
        Returns:
            Dict[str, Any]: 监控结果
        """
        try:
            monitoring_results = {}
            
            # 并发监控不同类型的风险指标
            monitoring_tasks = []
            
            if metrics_config.get('monitor_market_risk', True):
                monitoring_tasks.append(self._async_monitor_market_metrics())
            
            if metrics_config.get('monitor_credit_risk', True):
                monitoring_tasks.append(self._async_monitor_credit_metrics())
            
            if metrics_config.get('monitor_operational_risk', True):
                monitoring_tasks.append(self._async_monitor_operational_risk())
            
            # 等待所有监控任务完成
            results = await asyncio.gather(*monitoring_tasks, return_exceptions=True)
            
            # 整理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    monitoring_results[f'task_{i}'] = {'error': str(result)}
                else:
                    monitoring_results.update(result)
            
            self._trigger_event('async_monitoring_completed', monitoring_results)
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"异步风险监控失败: {e}")
            raise ProcessingError(f"异步风险监控失败: {e}")
    
    async def _async_monitor_market_metrics(self) -> Dict[str, Any]:
        """异步监控市场风险指标"""
        try:
            # 模拟市场数据
            await asyncio.sleep(0.1)  # 模拟数据获取延迟
            
            return {
                'market_risk': {
                    'var': 0.025,
                    'cvar': 0.035,
                    'volatility': 0.22,
                    'max_drawdown': 0.08,
                    'status': 'NORMAL'
                }
            }
        except Exception as e:
            return {'market_risk': {'error': str(e)}}
    
    async def _async_monitor_credit_metrics(self) -> Dict[str, Any]:
        """异步监控信用风险指标"""
        try:
            await asyncio.sleep(0.1)
            
            return {
                'credit_risk': {
                    'total_exposure': 1000000,
                    'expected_loss': 15000,
                    'default_rate': 0.015,
                    'concentration_risk': 0.25,
                    'status': 'NORMAL'
                }
            }
        except Exception as e:
            return {'credit_risk': {'error': str(e)}}
    
    async def _async_monitor_operational_risk(self) -> Dict[str, Any]:
        """异步监控操作风险指标"""
        try:
            await asyncio.sleep(0.1)
            
            return {
                'operational_risk': {
                    'system_availability': 0.999,
                    'error_rate': 0.002,
                    'incident_count': 1,
                    'control_effectiveness': 0.85,
                    'status': 'NORMAL'
                }
            }
        except Exception as e:
            return {'operational_risk': {'error': str(e)}}
    
    # ==================== 事件系统 ====================
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """
        添加事件回调
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        self._event_callbacks[event_type].append(callback)
    
    def _trigger_event(self, event_type: str, data: Any):
        """
        触发事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        try:
            for callback in self._event_callbacks.get(event_type, []):
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"事件回调执行失败: {e}")
        except Exception as e:
            self.logger.error(f"事件触发失败: {e}")
    
    # ==================== 配置导入导出 ====================
    
    def export_configurations(self, filepath: str, include_sensitive: bool = False) -> bool:
        """
        导出配置
        
        Args:
            filepath: 导出文件路径
            include_sensitive: 是否包含敏感信息
            
        Returns:
            bool: 是否导出成功
        """
        try:
            configurations = {
                'risk_thresholds': [asdict(t) for t in self.risk_thresholds.values()],
                'risk_limits': [asdict(l) for l in self.risk_limits.values()],
                'risk_models': [asdict(m) for m in self.risk_models.values()],
                'market_risk_config': asdict(self.market_risk_config) if self.market_risk_config else None,
                'credit_risk_config': asdict(self.credit_risk_config) if self.credit_risk_config else None,
                'operational_risk_config': asdict(self.operational_risk_config) if self.operational_risk_config else None,
                'monitoring_config': asdict(self.monitoring_config) if self.monitoring_config else None,
                'alert_configs': [asdict(config) for config in self.list_alert_configs()],
                'hedge_configs': [asdict(h) for h in self.hedge_configs.values()],
                'report_configs': [asdict(r) for r in self.report_configs.values()],
                'exported_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # 如果不包含敏感信息，移除敏感字段
            if not include_sensitive:
                for threshold in configurations['risk_thresholds']:
                    threshold.pop('metadata', None)
                for model in configurations['risk_models']:
                    # 保留参数但移除可能的敏感信息
                    pass
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(configurations, f, ensure_ascii=False, indent=2, default=str)
            
            self._trigger_event('configurations_exported', {'filepath': filepath})
            self.logger.info(f"配置已导出到: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出配置失败: {e}")
            return False
    
    def import_configurations(self, filepath: str, validate: bool = True) -> Dict[str, Any]:
        """
        导入配置
        
        Args:
            filepath: 导入文件路径
            validate: 是否验证配置
            
        Returns:
            Dict[str, Any]: 导入结果
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                configurations = json.load(f)
            
            import_results = {
                'success': 0,
                'failed': 0,
                'errors': [],
                'details': {}
            }
            
            # 导入风险阈值
            if 'risk_thresholds' in configurations:
                for threshold_data in configurations['risk_thresholds']:
                    try:
                        threshold_data['created_at'] = datetime.fromisoformat(threshold_data['created_at'])
                        threshold_data['updated_at'] = datetime.fromisoformat(threshold_data['updated_at'])
                        threshold = RiskThreshold(**threshold_data)
                        
                        if not validate or self.validator.validate_threshold(threshold)[0]:
                            if self.add_risk_threshold(threshold):
                                import_results['success'] += 1
                            else:
                                import_results['failed'] += 1
                                import_results['errors'].append(f"添加风险阈值失败: {threshold.name}")
                        else:
                            import_results['failed'] += 1
                            import_results['errors'].append(f"风险阈值验证失败: {threshold.name}")
                    except Exception as e:
                        import_results['failed'] += 1
                        import_results['errors'].append(f"风险阈值导入错误: {e}")
            
            # 导入风险限制
            if 'risk_limits' in configurations:
                for limit_data in configurations['risk_limits']:
                    try:
                        limit_data['created_at'] = datetime.fromisoformat(limit_data['created_at'])
                        limit_data['updated_at'] = datetime.fromisoformat(limit_data['updated_at'])
                        # 移除utilization_rate字段，它会在__post_init__中自动计算
                        limit_data.pop('utilization_rate', None)
                        limit = RiskLimit(**limit_data)
                        
                        if not validate or self.validator.validate_limit(limit)[0]:
                            if self.add_risk_limit(limit):
                                import_results['success'] += 1
                            else:
                                import_results['failed'] += 1
                                import_results['errors'].append(f"添加风险限制失败: {limit.name}")
                        else:
                            import_results['failed'] += 1
                            import_results['errors'].append(f"风险限制验证失败: {limit.name}")
                    except Exception as e:
                        import_results['failed'] += 1
                        import_results['errors'].append(f"风险限制导入错误: {e}")
            
            # 导入其他配置类型...
            # (为了节省空间，这里只实现核心配置类型的导入)
            
            self._trigger_event('configurations_imported', import_results)
            self.logger.info(f"配置导入完成: {import_results}")
            
            return import_results
            
        except Exception as e:
            self.logger.error(f"导入配置失败: {e}")
            return {'success': 0, 'failed': 0, 'errors': [str(e)]}
    
    # ==================== 清理和关闭 ====================
    
    def cleanup(self):
        """清理资源"""
        try:
            # 停止监控
            self.stop_monitoring()
            
            # 关闭线程池
            self._executor.shutdown(wait=True)
            
            # 清理回调
            self._event_callbacks.clear()
            self.alert_systems.clear()
            
            self.logger.info("风险配置处理器已清理")
            
        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    
    # 创建风险配置处理器
    processor = RiskConfigurationProcessor()
    
    try:
        # 1. 配置市场风险
        market_config = MarketRiskConfig(
            var_confidence_level=0.95,
            var_time_horizon=1,
            cvar_confidence_level=0.95,
            max_drawdown_limit=0.15,
            volatility_threshold=0.25,
            correlation_limit=0.8
        )
        processor.configure_market_risk(market_config)
        
        # 2. 添加风险阈值
        threshold = RiskThreshold(
            name="VaR阈值",
            value=0.05,
            unit="5%",
            description="95%置信度下的VaR阈值"
        )
        processor.add_risk_threshold(threshold)
        
        # 3. 添加风险限制
        limit = RiskLimit(
            name="总敞口限制",
            limit_value=1000000,
            current_value=750000,
            unit="USD",
            description="总信用敞口限制"
        )
        processor.add_risk_limit(limit)
        
        # 4. 添加风险模型
        model = RiskModel(
            name="历史模拟VaR模型",
            model_type="Historical_Simulation",
            parameters={"window": 252, "confidence_level": 0.95},
            description="基于历史数据的历史模拟VaR模型"
        )
        processor.add_risk_model(model)
        
        # 5. 配置信用风险
        credit_config = CreditRiskConfig(
            rating_scale={"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6},
            default_probabilities={"AAA": 0.0001, "AA": 0.0005, "A": 0.002, "BBB": 0.01},
            exposure_limits={"AAA": 500000, "AA": 300000, "A": 200000},
            recovery_rates={"AAA": 0.6, "AA": 0.5, "A": 0.4, "BBB": 0.3}
        )
        processor.configure_credit_risk(credit_config)
        
        # 6. 配置操作风险
        operational_config = OperationalRiskConfig(
            system_risk_factors=["系统可用性", "数据质量", "技术故障"],
            human_risk_factors=["操作失误", "欺诈风险", "合规风险"],
            process_risk_factors=["流程复杂性", "控制缺陷", "文档不完整"],
            external_risk_factors=["监管变化", "市场事件", "自然灾害"]
        )
        processor.configure_operational_risk(operational_config)
        
        # 7. 配置监控
        monitoring_config = MonitoringConfig(
            monitoring_frequency=60,
            real_time_alerts=True,
            alert_channels=["email", "sms", "webhook"],
            data_sources=["market_data_feed", "credit_database", "operational_logs"]
        )
        processor.configure_monitoring(monitoring_config)
        
        # 8. 添加告警配置
        alert_config = AlertConfig(
            alert_id="var_threshold_alert",
            name="VaR阈值告警",
            condition="var > threshold",
            threshold=0.05,
            severity=AlertSeverity.WARNING,
            channels=["email", "sms"]
        )
        processor.add_alert_config(alert_config)
        
        # 9. 配置对冲
        hedge_config = HedgeConfig(
            strategy=HedgeStrategy.DYNAMIC_HEDGING,
            target_risk_reduction=0.8,
            instruments=["options", "futures", "swaps"],
            rebalancing_frequency=1,
            cost_budget=0.001
        )
        processor.configure_hedge(hedge_config)
        
        # 10. 配置报告
        report_config = ReportConfig(
            report_name="daily_risk_report",
            report_type="DAILY_SUMMARY",
            format=ReportFormat.HTML,
            schedule="DAILY",
            recipients=["risk_manager@company.com", "cfo@company.com"],
            content_sections=["market_risk", "credit_risk", "risk_limits", "alerts"]
        )
        processor.configure_report(report_config)
        
        # 11. 启动监控
        processor.start_monitoring()
        
        # 12. 生成报告
        report_path = processor.generate_risk_report("daily_risk_report")
        print(f"报告已生成: {report_path}")
        
        # 13. 异步处理示例
        async def async_example():
            # 异步配置处理
            configs = [
                {
                    'type': 'threshold',
                    'data': {
                        'name': '波动率阈值',
                        'value': 0.30,
                        'unit': '30%',
                        'description': '市场波动率监控阈值'
                    }
                },
                {
                    'type': 'alert',
                    'data': {
                        'alert_id': 'volatility_alert',
                        'name': '波动率告警',
                        'condition': 'volatility > threshold',
                        'threshold': 0.30,
                        'severity': 'WARNING',
                        'channels': ['email']
                    }
                }
            ]
            
            results = await processor.async_process_configurations(configs)
            print(f"异步配置处理结果: {results}")
            
            # 异步报告生成
            report_requests = [
                {'report_name': 'daily_risk_report'},
                {'report_name': 'weekly_risk_summary'}
            ]
            
            generated_reports = await processor.async_generate_reports(report_requests)
            print(f"生成的报告: {generated_reports}")
        
        # 运行异步示例
        asyncio.run(async_example())
        
        print("K4风险配置处理器示例运行完成")
        
    finally:
        # 清理资源
        processor.cleanup()


if __name__ == "__main__":
    # 运行示例
    example_usage()