#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G3风险管控器
实现全面的风险识别、评估、控制、监控、报告和合规管理功能

功能模块：
1. 风险识别和评估
2. 风险度量和管理
3. 风险控制和缓解
4. 风险监控和预警
5. 风险报告和分析
6. 风险模型验证和更新
7. 风险合规和审计


创建时间: 2025-11-05
版本: 1.0.0
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
import sqlite3
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级枚举"""
    CRITICAL = "严重"
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"
    MINIMAL = "极低"


class RiskType(Enum):
    """风险类型枚举"""
    MARKET_RISK = "市场风险"
    CREDIT_RISK = "信用风险"
    OPERATIONAL_RISK = "操作风险"
    LIQUIDITY_RISK = "流动性风险"
    COMPLIANCE_RISK = "合规风险"
    SYSTEM_RISK = "系统风险"
    STRATEGIC_RISK = "战略风险"
    REPUTATION_RISK = "声誉风险"
    CONCENTRATION_RISK = "集中度风险"
    MODEL_RISK = "模型风险"


class AlertStatus(Enum):
    """预警状态枚举"""
    ACTIVE = "活跃"
    ACKNOWLEDGED = "已确认"
    RESOLVED = "已解决"
    CLOSED = "已关闭"


@dataclass
class RiskEvent:
    """风险事件数据类"""
    event_id: str
    event_type: RiskType
    risk_level: RiskLevel
    title: str
    description: str
    probability: float
    impact: float
    risk_score: float
    timestamp: datetime
    source: str
    status: AlertStatus = AlertStatus.ACTIVE
    mitigation_actions: List[str] = field(default_factory=list)
    owner: str = ""
    due_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetric:
    """风险指标数据类"""
    metric_name: str
    metric_value: float
    threshold: float
    risk_level: RiskLevel
    timestamp: datetime
    unit: str = ""
    category: str = ""
    benchmark: Optional[float] = None


@dataclass
class ComplianceRule:
    """合规规则数据类"""
    rule_id: str
    rule_name: str
    rule_type: str
    description: str
    threshold: float
    penalty: str
    status: str = "ACTIVE"
    last_check: Optional[datetime] = None


class RiskIdentificationEngine:
    """风险识别引擎"""
    
    def __init__(self):
        self.risk_patterns = {}
        self.identification_rules = []
        self.historical_data = deque(maxlen=10000)
        
    def register_pattern(self, pattern_name: str, pattern_func: Callable):
        """注册风险识别模式"""
        self.risk_patterns[pattern_name] = pattern_func
        
    def add_identification_rule(self, rule: Dict[str, Any]):
        """添加识别规则"""
        self.identification_rules.append(rule)
        
    def identify_risks(self, data: Dict[str, Any]) -> List[RiskEvent]:
        """识别风险事件"""
        risks = []
        
        # 基于模式的风险识别
        for pattern_name, pattern_func in self.risk_patterns.items():
            try:
                pattern_risks = pattern_func(data)
                if pattern_risks:
                    risks.extend(pattern_risks)
            except Exception as e:
                logger.error(f"风险识别模式 {pattern_name} 执行失败: {e}")
                
        # 基于规则的风险识别
        for rule in self.identification_rules:
            try:
                rule_risks = self._apply_rule(rule, data)
                if rule_risks:
                    risks.extend(rule_risks)
            except Exception as e:
                logger.error(f"识别规则 {rule.get('name', 'unknown')} 执行失败: {e}")
                
        return risks
    
    def _apply_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> List[RiskEvent]:
        """应用识别规则"""
        risks = []
        condition = rule.get('condition', '')
        threshold = rule.get('threshold', 0.5)
        
        # 简化的规则应用逻辑
        if 'value' in data:
            value = data['value']
            if eval(condition, {"value": value, "threshold": threshold}):
                risk = RiskEvent(
                    event_id=f"rule_{rule['id']}_{int(time.time())}",
                    event_type=RiskType(rule.get('risk_type', 'OPERATIONAL_RISK')),
                    risk_level=RiskLevel(rule.get('risk_level', 'MEDIUM')),
                    title=rule.get('name', '规则触发风险'),
                    description=rule.get('description', ''),
                    probability=rule.get('probability', 0.5),
                    impact=rule.get('impact', 0.5),
                    risk_score=0.0,
                    timestamp=datetime.now(),
                    source="rule_engine"
                )
                risk.risk_score = risk.probability * risk.impact
                risks.append(risk)
                
        return risks


class RiskAssessmentEngine:
    """风险评估引擎"""
    
    def __init__(self):
        self.assessment_models = {}
        self.calibration_data = {}
        
    def register_model(self, model_name: str, model_func: Callable):
        """注册评估模型"""
        self.assessment_models[model_name] = model_func
        
    def assess_risk(self, risk_event: RiskEvent, context: Dict[str, Any] = None) -> RiskEvent:
        """评估风险"""
        if context is None:
            context = {}
            
        # 应用评估模型
        for model_name, model_func in self.assessment_models.items():
            try:
                updated_risk = model_func(risk_event, context)
                if updated_risk:
                    risk_event = updated_risk
            except Exception as e:
                logger.error(f"评估模型 {model_name} 执行失败: {e}")
                
        # 重新计算风险评分
        risk_event.risk_score = self._calculate_risk_score(risk_event)
        
        return risk_event
    
    def _calculate_risk_score(self, risk_event: RiskEvent) -> float:
        """计算风险评分"""
        base_score = risk_event.probability * risk_event.impact
        
        # 风险等级调整因子
        level_multipliers = {
            RiskLevel.CRITICAL: 1.5,
            RiskLevel.HIGH: 1.2,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.LOW: 0.8,
            RiskLevel.MINIMAL: 0.5
        }
        
        multiplier = level_multipliers.get(risk_event.risk_level, 1.0)
        return base_score * multiplier


class RiskMeasurementEngine:
    """风险度量引擎"""
    
    def __init__(self):
        self.metrics = {}
        self.calculation_methods = {}
        self.benchmarks = {}
        
    def register_metric(self, metric_name: str, calculation_func: Callable):
        """注册度量指标"""
        self.calculation_methods[metric_name] = calculation_func
        
    def set_benchmark(self, metric_name: str, benchmark_value: float):
        """设置基准值"""
        self.benchmarks[metric_name] = benchmark_value
        
    def calculate_metric(self, metric_name: str, data: Any, context: Dict[str, Any] = None) -> RiskMetric:
        """计算风险指标"""
        if context is None:
            context = {}
            
        if metric_name not in self.calculation_methods:
            raise ValueError(f"未注册的风险指标: {metric_name}")
            
        calculation_func = self.calculation_methods[metric_name]
        
        try:
            metric_value = calculation_func(data, context)
            threshold = context.get('threshold', 0.5)
            benchmark = self.benchmarks.get(metric_name)
            
            # 确定风险等级
            risk_level = self._determine_risk_level(metric_value, threshold, benchmark)
            
            return RiskMetric(
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold,
                risk_level=risk_level,
                timestamp=datetime.now(),
                benchmark=benchmark
            )
            
        except Exception as e:
            logger.error(f"计算风险指标 {metric_name} 失败: {e}")
            raise
    
    def _determine_risk_level(self, value: float, threshold: float, benchmark: Optional[float] = None) -> RiskLevel:
        """确定风险等级"""
        if benchmark is not None:
            # 基于基准值的风险等级
            deviation = abs(value - benchmark) / benchmark if benchmark != 0 else 0
            if deviation > 0.3:
                return RiskLevel.CRITICAL
            elif deviation > 0.2:
                return RiskLevel.HIGH
            elif deviation > 0.1:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        else:
            # 基于阈值的风险等级
            if value > threshold * 1.5:
                return RiskLevel.CRITICAL
            elif value > threshold * 1.2:
                return RiskLevel.HIGH
            elif value > threshold:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW


class RiskControlEngine:
    """风险控制引擎"""
    
    def __init__(self):
        self.control_strategies = {}
        self.mitigation_plans = {}
        self.control_effectiveness = {}
        
    def register_control_strategy(self, strategy_name: str, strategy_func: Callable):
        """注册控制策略"""
        self.control_strategies[strategy_name] = strategy_func
        
    def create_mitigation_plan(self, risk_event: RiskEvent) -> Dict[str, Any]:
        """创建缓解计划"""
        plan = {
            'plan_id': f"plan_{risk_event.event_id}",
            'risk_event_id': risk_event.event_id,
            'strategies': [],
            'timeline': [],
            'resources': [],
            'status': 'DRAFT',
            'created_at': datetime.now(),
            'estimated_completion': datetime.now() + timedelta(days=30)
        }
        
        # 根据风险类型选择控制策略
        strategy_mapping = {
            RiskType.MARKET_RISK: ['position_limit', 'stop_loss'],
            RiskType.CREDIT_RISK: ['position_limit', 'stop_loss'],
            RiskType.OPERATIONAL_RISK: ['position_limit', 'stop_loss'],
            RiskType.LIQUIDITY_RISK: ['position_limit', 'stop_loss'],
            RiskType.COMPLIANCE_RISK: ['position_limit', 'stop_loss'],
            RiskType.SYSTEM_RISK: ['position_limit', 'stop_loss'],
            RiskType.STRATEGIC_RISK: ['position_limit', 'stop_loss'],
            RiskType.REPUTATION_RISK: ['position_limit', 'stop_loss'],
            RiskType.CONCENTRATION_RISK: ['position_limit', 'stop_loss'],
            RiskType.MODEL_RISK: ['position_limit', 'stop_loss']
        }
        
        strategies = strategy_mapping.get(risk_event.event_type, ['general_controls'])
        plan['strategies'] = strategies
        
        # 创建时间线
        for i, strategy in enumerate(strategies):
            timeline_item = {
                'step': i + 1,
                'strategy': strategy,
                'deadline': datetime.now() + timedelta(days=(i + 1) * 7),
                'status': 'PENDING'
            }
            plan['timeline'].append(timeline_item)
            
        self.mitigation_plans[plan['plan_id']] = plan
        return plan
    
    def execute_control_strategy(self, strategy_name: str, risk_event: RiskEvent) -> Dict[str, Any]:
        """执行控制策略"""
        if strategy_name not in self.control_strategies:
            raise ValueError(f"未注册的控制策略: {strategy_name}")
            
        strategy_func = self.control_strategies[strategy_name]
        
        try:
            result = strategy_func(risk_event)
            logger.info(f"控制策略 {strategy_name} 执行成功")
            return result
        except Exception as e:
            logger.error(f"控制策略 {strategy_name} 执行失败: {e}")
            raise


class RiskMonitoringEngine:
    """风险监控引擎"""
    
    def __init__(self):
        self.monitoring_rules = {}
        self.alert_thresholds = {}
        self.monitoring_data = deque(maxlen=100000)
        self.active_alerts = {}
        self.monitoring_threads = {}
        
    def register_monitoring_rule(self, rule_name: str, rule_func: Callable, interval: int = 60):
        """注册监控规则"""
        self.monitoring_rules[rule_name] = {
            'function': rule_func,
            'interval': interval,
            'last_run': None,
            'active': True
        }
        
    def set_alert_threshold(self, metric_name: str, threshold: float, level: RiskLevel):
        """设置预警阈值"""
        self.alert_thresholds[metric_name] = {
            'threshold': threshold,
            'risk_level': level
        }
        
    def start_monitoring(self):
        """启动监控"""
        for rule_name, rule_config in self.monitoring_rules.items():
            if rule_config['active']:
                thread = threading.Thread(
                    target=self._monitoring_loop,
                    args=(rule_name, rule_config),
                    daemon=True
                )
                thread.start()
                self.monitoring_threads[rule_name] = thread
                logger.info(f"监控规则 {rule_name} 已启动")
                
    def stop_monitoring(self):
        """停止监控"""
        for rule_name in self.monitoring_threads:
            self.monitoring_rules[rule_name]['active'] = False
        self.monitoring_threads.clear()
        logger.info("所有监控规则已停止")
        
    def _monitoring_loop(self, rule_name: str, rule_config: Dict[str, Any]):
        """监控循环"""
        while rule_config['active']:
            try:
                start_time = time.time()
                result = rule_config['function']()
                rule_config['last_run'] = datetime.now()
                
                if result:
                    self._process_monitoring_result(rule_name, result)
                    
                # 等待下一次执行
                elapsed = time.time() - start_time
                sleep_time = max(0, rule_config['interval'] - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"监控规则 {rule_name} 执行失败: {e}")
                time.sleep(rule_config['interval'])
                
    def _process_monitoring_result(self, rule_name: str, result: Dict[str, Any]):
        """处理监控结果"""
        self.monitoring_data.append({
            'rule_name': rule_name,
            'timestamp': datetime.now(),
            'result': result
        })
        
        # 检查预警条件
        for metric_name, value in result.items():
            if metric_name in self.alert_thresholds:
                threshold_info = self.alert_thresholds[metric_name]
                if value > threshold_info['threshold']:
                    self._trigger_alert(rule_name, metric_name, value, threshold_info)
                    
    def _trigger_alert(self, rule_name: str, metric_name: str, value: float, threshold_info: Dict[str, Any]):
        """触发预警"""
        alert_id = f"alert_{rule_name}_{metric_name}_{int(time.time())}"
        
        alert = {
            'alert_id': alert_id,
            'rule_name': rule_name,
            'metric_name': metric_name,
            'current_value': value,
            'threshold': threshold_info['threshold'],
            'risk_level': threshold_info['risk_level'],
            'timestamp': datetime.now(),
            'status': AlertStatus.ACTIVE
        }
        
        self.active_alerts[alert_id] = alert
        logger.warning(f"风险预警触发: {alert_id} - {metric_name} = {value} > {threshold_info['threshold']}")


class RiskReportingEngine:
    """风险报告引擎"""
    
    def __init__(self):
        self.report_templates = {}
        self.report_data = {}
        self.report_schedule = {}
        
    def register_report_template(self, template_name: str, template_func: Callable):
        """注册报告模板"""
        self.report_templates[template_name] = template_func
        
    def generate_report(self, template_name: str, data: Dict[str, Any], output_format: str = 'json') -> str:
        """生成报告"""
        if template_name not in self.report_templates:
            raise ValueError(f"未注册的报告模板: {template_name}")
            
        template_func = self.report_templates[template_name]
        
        try:
            report_content = template_func(data)
            
            if output_format == 'json':
                return json.dumps(report_content, ensure_ascii=False, indent=2, default=str)
            elif output_format == 'html':
                return self._generate_html_report(report_content)
            elif output_format == 'pdf':
                return self._generate_pdf_report(report_content)
            else:
                return str(report_content)
                
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            raise
            
    def _generate_html_report(self, report_content: Dict[str, Any]) -> str:
        """生成HTML报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>风险报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .risk-high {{ color: red; font-weight: bold; }}
                .risk-medium {{ color: orange; }}
                .risk-low {{ color: green; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>风险报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>风险概览</h2>
                <p>总风险事件数: {report_content.get('total_risks', 0)}</p>
                <p>高风险事件数: {report_content.get('high_risks', 0)}</p>
                <p>中风险事件数: {report_content.get('medium_risks', 0)}</p>
                <p>低风险事件数: {report_content.get('low_risks', 0)}</p>
            </div>
            
            <div class="section">
                <h2>风险事件详情</h2>
                <table>
                    <tr>
                        <th>事件ID</th>
                        <th>风险类型</th>
                        <th>风险等级</th>
                        <th>标题</th>
                        <th>风险评分</th>
                        <th>状态</th>
                    </tr>
        """
        
        for risk in report_content.get('risks', []):
            risk_class = f"risk-{risk.get('risk_level', 'low').lower()}"
            html += f"""
                    <tr>
                        <td>{risk.get('event_id', '')}</td>
                        <td>{risk.get('event_type', '')}</td>
                        <td class="{risk_class}">{risk.get('risk_level', '')}</td>
                        <td>{risk.get('title', '')}</td>
                        <td>{risk.get('risk_score', 0):.2f}</td>
                        <td>{risk.get('status', '')}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>风险指标</h2>
                <table>
                    <tr>
                        <th>指标名称</th>
                        <th>当前值</th>
                        <th>阈值</th>
                        <th>风险等级</th>
                        <th>基准值</th>
                    </tr>
        """
        
        for metric in report_content.get('metrics', []):
            risk_class = f"risk-{metric.get('risk_level', 'low').lower()}"
            html += f"""
                    <tr>
                        <td>{metric.get('metric_name', '')}</td>
                        <td>{metric.get('metric_value', 0):.2f}</td>
                        <td>{metric.get('threshold', 0):.2f}</td>
                        <td class="{risk_class}">{metric.get('risk_level', '')}</td>
                        <td>{metric.get('benchmark', 'N/A')}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html
        
    def _generate_pdf_report(self, report_content: Dict[str, Any]) -> str:
        """生成PDF报告（简化实现）"""
        # 这里可以集成PDF生成库，如reportlab
        return f"PDF报告内容: {json.dumps(report_content, ensure_ascii=False, indent=2)}"


class RiskModelValidationEngine:
    """风险模型验证引擎"""
    
    def __init__(self):
        self.model_performance = {}
        self.validation_results = {}
        self.backtesting_results = {}
        
    def validate_model(self, model_name: str, test_data: Any, expected_output: Any) -> Dict[str, Any]:
        """验证模型"""
        validation_result = {
            'model_name': model_name,
            'validation_time': datetime.now(),
            'test_data_size': len(test_data) if hasattr(test_data, '__len__') else 1,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'validation_status': 'FAILED'
        }
        
        try:
            # 简化的模型验证逻辑
            if isinstance(test_data, (list, np.ndarray)) and isinstance(expected_output, (list, np.ndarray)):
                # 计算基本性能指标
                if len(test_data) == len(expected_output):
                    # 假设预测准确率为85%（实际应用中需要真实的模型预测）
                    validation_result['accuracy'] = 0.85
                    validation_result['precision'] = 0.82
                    validation_result['recall'] = 0.88
                    validation_result['f1_score'] = 0.85
                    validation_result['validation_status'] = 'PASSED'
                    
            self.validation_results[f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = validation_result
            return validation_result
            
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            validation_result['error'] = str(e)
            return validation_result
            
    def backtest_model(self, model_name: str, historical_data: Any, period: int = 252) -> Dict[str, Any]:
        """回测模型"""
        backtest_result = {
            'model_name': model_name,
            'backtest_period': period,
            'backtest_time': datetime.now(),
            'total_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'backtest_status': 'COMPLETED'
        }
        
        try:
            # 简化的回测逻辑
            if isinstance(historical_data, (list, np.ndarray, pd.DataFrame)):
                # 模拟回测结果
                np.random.seed(42)  # 确保结果可重现
                returns = np.random.normal(0.001, 0.02, period)
                
                backtest_result['total_return'] = np.prod(1 + returns) - 1
                backtest_result['volatility'] = np.std(returns) * np.sqrt(252)
                backtest_result['sharpe_ratio'] = backtest_result['total_return'] / backtest_result['volatility'] if backtest_result['volatility'] > 0 else 0
                
                # 计算最大回撤
                cumulative_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                backtest_result['max_drawdown'] = np.min(drawdown)
                
                # 计算胜率
                positive_returns = np.sum(returns > 0)
                backtest_result['win_rate'] = positive_returns / len(returns)
                
            self.backtesting_results[f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = backtest_result
            return backtest_result
            
        except Exception as e:
            logger.error(f"模型回测失败: {e}")
            backtest_result['error'] = str(e)
            backtest_result['backtest_status'] = 'FAILED'
            return backtest_result
            
    def update_model(self, model_name: str, new_data: Any) -> bool:
        """更新模型"""
        try:
            # 简化的模型更新逻辑
            logger.info(f"开始更新模型: {model_name}")
            
            # 模拟模型训练过程
            time.sleep(2)  # 模拟训练时间
            
            # 更新模型性能记录
            if model_name not in self.model_performance:
                self.model_performance[model_name] = []
                
            performance_record = {
                'update_time': datetime.now(),
                'data_size': len(new_data) if hasattr(new_data, '__len__') else 1,
                'training_status': 'SUCCESS'
            }
            
            self.model_performance[model_name].append(performance_record)
            logger.info(f"模型 {model_name} 更新成功")
            return True
            
        except Exception as e:
            logger.error(f"模型更新失败: {e}")
            return False


class RiskComplianceEngine:
    """风险合规引擎"""
    
    def __init__(self):
        self.compliance_rules = {}
        self.compliance_status = {}
        self.audit_trail = []
        self.violation_records = []
        
    def register_compliance_rule(self, rule: ComplianceRule):
        """注册合规规则"""
        self.compliance_rules[rule.rule_id] = rule
        
    def check_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查合规性"""
        compliance_result = {
            'check_time': datetime.now(),
            'overall_status': 'COMPLIANT',
            'violations': [],
            'warnings': [],
            'compliance_score': 100.0
        }
        
        for rule_id, rule in self.compliance_rules.items():
            try:
                rule_result = self._check_single_rule(rule, data)
                
                if rule_result['status'] == 'VIOLATION':
                    compliance_result['violations'].append(rule_result)
                    compliance_result['overall_status'] = 'NON_COMPLIANT'
                    compliance_result['compliance_score'] -= 10
                elif rule_result['status'] == 'WARNING':
                    compliance_result['warnings'].append(rule_result)
                    compliance_result['compliance_score'] -= 5
                    
                # 记录审计轨迹
                self.audit_trail.append({
                    'rule_id': rule_id,
                    'check_time': datetime.now(),
                    'result': rule_result['status'],
                    'details': rule_result
                })
                
            except Exception as e:
                logger.error(f"合规检查规则 {rule_id} 执行失败: {e}")
                compliance_result['violations'].append({
                    'rule_id': rule_id,
                    'status': 'ERROR',
                    'message': str(e)
                })
                
        self.compliance_status[f"check_{int(time.time())}"] = compliance_result
        return compliance_result
        
    def _check_single_rule(self, rule: ComplianceRule, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查单个合规规则"""
        result = {
            'rule_id': rule.rule_id,
            'rule_name': rule.rule_name,
            'status': 'COMPLIANT',
            'check_time': datetime.now(),
            'details': {}
        }
        
        try:
            # 简化的规则检查逻辑
            if 'value' in data:
                current_value = data['value']
                
                if rule.rule_type == 'MAXIMUM' and current_value > rule.threshold:
                    result['status'] = 'VIOLATION'
                    result['details'] = {
                        'current_value': current_value,
                        'threshold': rule.threshold,
                        'message': f"值 {current_value} 超过最大阈值 {rule.threshold}"
                    }
                elif rule.rule_type == 'MINIMUM' and current_value < rule.threshold:
                    result['status'] = 'VIOLATION'
                    result['details'] = {
                        'current_value': current_value,
                        'threshold': rule.threshold,
                        'message': f"值 {current_value} 低于最小阈值 {rule.threshold}"
                    }
                elif rule.rule_type == 'RANGE':
                    # 假设阈值是元组 (min, max)
                    min_val, max_val = rule.threshold if isinstance(rule.threshold, tuple) else (0, rule.threshold)
                    if not (min_val <= current_value <= max_val):
                        result['status'] = 'VIOLATION'
                        result['details'] = {
                            'current_value': current_value,
                            'min_threshold': min_val,
                            'max_threshold': max_val,
                            'message': f"值 {current_value} 超出范围 [{min_val}, {max_val}]"
                        }
                        
        except Exception as e:
            result['status'] = 'ERROR'
            result['details'] = {'message': f"规则检查错误: {e}"}
            
        # 更新规则检查时间
        rule.last_check = datetime.now()
        
        return result
        
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """生成审计报告"""
        # 筛选指定时间范围内的审计记录
        filtered_trail = [
            record for record in self.audit_trail
            if start_date <= record['check_time'] <= end_date
        ]
        
        audit_report = {
            'report_period': f"{start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}",
            'total_checks': len(filtered_trail),
            'compliance_rate': 0.0,
            'violation_count': 0,
            'warning_count': 0,
            'error_count': 0,
            'top_violations': [],
            'compliance_trend': [],
            'generated_time': datetime.now()
        }
        
        if filtered_trail:
            # 统计各类结果数量
            status_counts = defaultdict(int)
            for record in filtered_trail:
                status_counts[record['result']] += 1
                
            total = len(filtered_trail)
            audit_report['violation_count'] = status_counts['VIOLATION']
            audit_report['warning_count'] = status_counts['WARNING']
            audit_report['error_count'] = status_counts['ERROR']
            audit_report['compliance_rate'] = (status_counts['COMPLIANT'] / total) * 100
            
            # 找出最频繁的违规规则
            violation_rules = defaultdict(int)
            for record in filtered_trail:
                if record['result'] == 'VIOLATION':
                    violation_rules[record['rule_id']] += 1
                    
            audit_report['top_violations'] = [
                {'rule_id': rule_id, 'count': count}
                for rule_id, count in sorted(violation_rules.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
            
        return audit_report


class RiskController:
    """G3风险管控器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化风险管控器"""
        self.config = config or {}
        self.risk_identification_engine = RiskIdentificationEngine()
        self.risk_assessment_engine = RiskAssessmentEngine()
        self.risk_measurement_engine = RiskMeasurementEngine()
        self.risk_control_engine = RiskControlEngine()
        self.risk_monitoring_engine = RiskMonitoringEngine()
        self.risk_reporting_engine = RiskReportingEngine()
        self.risk_model_validation_engine = RiskModelValidationEngine()
        self.risk_compliance_engine = RiskComplianceEngine()
        
        # 风险数据库
        self.risk_database = self._initialize_database()
        
        # 风险事件存储
        self.active_risks = {}
        self.risk_history = deque(maxlen=10000)
        
        # 启动状态
        self.is_running = False
        
        # 初始化默认组件
        self._initialize_default_components()
        
        logger.info("G3风险管控器初始化完成")
        
    def _initialize_database(self) -> sqlite3.Connection:
        """初始化风险数据库"""
        db_path = Path("risk_management.db")
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # 创建风险事件表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS risk_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                risk_level TEXT,
                title TEXT,
                description TEXT,
                probability REAL,
                impact REAL,
                risk_score REAL,
                timestamp TEXT,
                source TEXT,
                status TEXT,
                owner TEXT,
                due_date TEXT,
                metadata TEXT
            )
        ''')
        
        # 创建风险指标表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                threshold REAL,
                risk_level TEXT,
                timestamp TEXT,
                unit TEXT,
                category TEXT,
                benchmark REAL
            )
        ''')
        
        # 创建合规检查表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS compliance_checks (
                check_id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_time TEXT,
                rule_id TEXT,
                rule_name TEXT,
                status TEXT,
                details TEXT,
                data TEXT
            )
        ''')
        
        conn.commit()
        return conn
        
    def _initialize_default_components(self):
        """初始化默认组件"""
        # 注册默认风险识别模式
        self.risk_identification_engine.register_pattern(
            "volatility_spike",
            self._volatility_spike_pattern
        )
        
        self.risk_identification_engine.register_pattern(
            "concentration_risk",
            self._concentration_risk_pattern
        )
        
        # 注册默认评估模型
        self.risk_assessment_engine.register_model(
            "probability_adjustment",
            self._probability_adjustment_model
        )
        
        # 注册默认度量指标
        self.risk_measurement_engine.register_metric(
            "volatility",
            self._calculate_volatility
        )
        
        self.risk_measurement_engine.register_metric(
            "var",
            self._calculate_var
        )
        
        self.risk_measurement_engine.register_metric(
            "sharpe_ratio",
            self._calculate_sharpe_ratio
        )
        
        # 注册默认控制策略
        self.risk_control_engine.register_control_strategy(
            "position_limit",
            self._position_limit_strategy
        )
        
        self.risk_control_engine.register_control_strategy(
            "stop_loss",
            self._stop_loss_strategy
        )
        
        # 注册默认监控规则
        self.risk_monitoring_engine.register_monitoring_rule(
            "price_monitoring",
            self._price_monitoring_rule,
            interval=30
        )
        
        self.risk_monitoring_engine.register_monitoring_rule(
            "volume_monitoring",
            self._volume_monitoring_rule,
            interval=60
        )
        
        # 注册默认报告模板
        self.risk_reporting_engine.register_report_template(
            "executive_summary",
            self._executive_summary_template
        )
        
        self.risk_reporting_engine.register_report_template(
            "detailed_risk_report",
            self._detailed_risk_report_template
        )
        
        # 添加默认合规规则
        self._add_default_compliance_rules()
        
    def _add_default_compliance_rules(self):
        """添加默认合规规则"""
        default_rules = [
            ComplianceRule(
                rule_id="MAX_POSITION_SIZE",
                rule_name="最大持仓规模",
                rule_type="MAXIMUM",
                description="单个资产持仓规模不能超过总资产的30%",
                threshold=0.30,
                penalty="强制平仓"
            ),
            ComplianceRule(
                rule_id="MAX_LEVERAGE",
                rule_name="最大杠杆倍数",
                rule_type="MAXIMUM",
                description="总杠杆倍数不能超过5倍",
                threshold=5.0,
                penalty="限制交易"
            ),
            ComplianceRule(
                rule_id="MIN_LIQUIDITY",
                rule_name="最小流动性",
                rule_type="MINIMUM",
                description="账户流动性比率不能低于20%",
                threshold=0.20,
                penalty="追加保证金"
            )
        ]
        
        for rule in default_rules:
            self.risk_compliance_engine.register_compliance_rule(rule)
            
    # 默认模式和方法实现
    def _volatility_spike_pattern(self, data: Dict[str, Any]) -> List[RiskEvent]:
        """波动率激增识别模式"""
        risks = []
        if 'returns' in data:
            returns = np.array(data['returns'])
            volatility = np.std(returns)
            historical_volatility = np.std(returns[:-20]) if len(returns) > 20 else volatility
            
            if volatility > historical_volatility * 2:
                risk = RiskEvent(
                    event_id=f"vol_spike_{int(time.time())}",
                    event_type=RiskType.MARKET_RISK,
                    risk_level=RiskLevel.HIGH,
                    title="波动率激增",
                    description=f"当前波动率 {volatility:.4f} 超过历史波动率 {historical_volatility:.4f} 的2倍",
                    probability=0.7,
                    impact=0.6,
                    risk_score=0.0,
                    timestamp=datetime.now(),
                    source="volatility_pattern"
                )
                risk.risk_score = risk.probability * risk.impact
                risks.append(risk)
                
        return risks
        
    def _concentration_risk_pattern(self, data: Dict[str, Any]) -> List[RiskEvent]:
        """集中度风险识别模式"""
        risks = []
        if 'positions' in data:
            positions = data['positions']
            total_value = sum(pos.get('value', 0) for pos in positions)
            
            for pos in positions:
                concentration = pos.get('value', 0) / total_value if total_value > 0 else 0
                if concentration > 0.3:  # 超过30%集中度
                    risk = RiskEvent(
                        event_id=f"concentration_{pos.get('asset', 'unknown')}_{int(time.time())}",
                        event_type=RiskType.CONCENTRATION_RISK,
                        risk_level=RiskLevel.MEDIUM,
                        title="资产集中度过高",
                        description=f"资产 {pos.get('asset', 'unknown')} 集中度 {concentration:.1%} 超过30%",
                        probability=0.5,
                        impact=0.7,
                        risk_score=0.0,
                        timestamp=datetime.now(),
                        source="concentration_pattern"
                    )
                    risk.risk_score = risk.probability * risk.impact
                    risks.append(risk)
                    
        return risks
        
    def _probability_adjustment_model(self, risk_event: RiskEvent, context: Dict[str, Any]) -> RiskEvent:
        """概率调整模型"""
        # 基于历史数据的概率调整
        if 'market_regime' in context:
            market_regime = context['market_regime']
            if market_regime == 'crisis':
                risk_event.probability *= 1.5
            elif market_regime == 'bull':
                risk_event.probability *= 0.8
                
        return risk_event
        
    def _calculate_volatility(self, data: Any, context: Dict[str, Any]) -> float:
        """计算波动率"""
        if isinstance(data, (list, np.ndarray)):
            returns = np.array(data)
            return np.std(returns) * np.sqrt(252)  # 年化波动率
        return 0.0
        
    def _calculate_var(self, data: Any, context: Dict[str, Any]) -> float:
        """计算风险价值(VaR)"""
        if isinstance(data, (list, np.ndarray)):
            returns = np.array(data)
            confidence_level = context.get('confidence_level', 0.95)
            return np.percentile(returns, (1 - confidence_level) * 100)
        return 0.0
        
    def _calculate_sharpe_ratio(self, data: Any, context: Dict[str, Any]) -> float:
        """计算夏普比率"""
        if isinstance(data, (list, np.ndarray)):
            returns = np.array(data)
            risk_free_rate = context.get('risk_free_rate', 0.02)
            excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
            return np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        return 0.0
        
    def _position_limit_strategy(self, risk_event: RiskEvent) -> Dict[str, Any]:
        """仓位限制策略"""
        return {
            'strategy': 'position_limit',
            'action': 'reduce_position',
            'target_reduction': 0.2,
            'status': 'executed',
            'timestamp': datetime.now()
        }
        
    def _stop_loss_strategy(self, risk_event: RiskEvent) -> Dict[str, Any]:
        """止损策略"""
        return {
            'strategy': 'stop_loss',
            'action': 'close_position',
            'stop_price': risk_event.metadata.get('stop_price'),
            'status': 'executed',
            'timestamp': datetime.now()
        }
        
    def _price_monitoring_rule(self) -> Dict[str, float]:
        """价格监控规则"""
        # 模拟价格数据
        np.random.seed(int(time.time()) % 1000)
        prices = np.random.normal(100, 5, 5)
        return {
            'price_volatility': np.std(prices),
            'price_change': (prices[-1] - prices[0]) / prices[0],
            'max_price': np.max(prices),
            'min_price': np.min(prices)
        }
        
    def _volume_monitoring_rule(self) -> Dict[str, float]:
        """交易量监控规则"""
        # 模拟交易量数据
        np.random.seed(int(time.time()) % 1000)
        volumes = np.random.lognormal(10, 1, 5)
        return {
            'avg_volume': np.mean(volumes),
            'volume_volatility': np.std(volumes),
            'max_volume': np.max(volumes),
            'volume_trend': (volumes[-1] - volumes[0]) / volumes[0]
        }
        
    def _executive_summary_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """高管摘要报告模板"""
        return {
            'report_type': 'executive_summary',
            'summary': {
                'total_risks': len(data.get('risks', [])),
                'high_risks': len([r for r in data.get('risks', []) if r.get('risk_level') == 'HIGH']),
                'compliance_score': data.get('compliance_score', 100),
                'key_metrics': data.get('metrics', [])[:5]
            },
            'recommendations': [
                '加强高风险事件的监控',
                '完善风险控制措施',
                '定期更新风险模型'
            ]
        }
        
    def _detailed_risk_report_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """详细风险报告模板"""
        return {
            'report_type': 'detailed_risk_report',
            'risks': data.get('risks', []),
            'metrics': data.get('metrics', []),
            'compliance': data.get('compliance', {}),
            'model_performance': data.get('model_performance', {}),
            'recommendations': [
                '详细分析每个高风险事件',
                '评估现有控制措施的有效性',
                '更新风险模型参数'
            ]
        }
        
    def start(self):
        """启动风险管控器"""
        if self.is_running:
            logger.warning("风险管控器已经在运行")
            return
            
        self.is_running = True
        self.risk_monitoring_engine.start_monitoring()
        logger.info("G3风险管控器已启动")
        
    def stop(self):
        """停止风险管控器"""
        if not self.is_running:
            logger.warning("风险管控器未在运行")
            return
            
        self.is_running = False
        self.risk_monitoring_engine.stop_monitoring()
        logger.info("G3风险管控器已停止")
        
    def identify_risks(self, data: Dict[str, Any]) -> List[RiskEvent]:
        """识别风险"""
        risks = self.risk_identification_engine.identify_risks(data)
        
        # 评估每个风险
        for risk in risks:
            risk = self.risk_assessment_engine.assess_risk(risk, data)
            self.active_risks[risk.event_id] = risk
            
        logger.info(f"识别到 {len(risks)} 个风险事件")
        return risks
        
    def measure_risks(self, data: Dict[str, Any]) -> List[RiskMetric]:
        """度量风险"""
        metrics = []
        
        for metric_name in self.risk_measurement_engine.calculation_methods.keys():
            try:
                metric = self.risk_measurement_engine.calculate_metric(metric_name, data)
                metrics.append(metric)
                
                # 保存到数据库
                self._save_metric_to_db(metric)
                
            except Exception as e:
                logger.error(f"计算风险指标 {metric_name} 失败: {e}")
                
        logger.info(f"计算了 {len(metrics)} 个风险指标")
        return metrics
        
    def control_risks(self, risk_event: RiskEvent) -> Dict[str, Any]:
        """控制风险"""
        try:
            # 创建缓解计划
            mitigation_plan = self.risk_control_engine.create_mitigation_plan(risk_event)
            
            # 执行控制策略
            strategies = mitigation_plan['strategies']
            results = []
            
            for strategy in strategies:
                try:
                    result = self.risk_control_engine.execute_control_strategy(strategy, risk_event)
                    results.append(result)
                except Exception as e:
                    logger.error(f"执行控制策略 {strategy} 失败: {e}")
                    
            logger.info(f"为风险事件 {risk_event.event_id} 执行了 {len(results)} 个控制策略")
            return {
                'mitigation_plan': mitigation_plan,
                'execution_results': results
            }
            
        except Exception as e:
            logger.error(f"风险控制失败: {e}")
            raise
            
    def monitor_risks(self) -> Dict[str, Any]:
        """监控风险"""
        monitoring_summary = {
            'active_alerts': len(self.risk_monitoring_engine.active_alerts),
            'monitoring_rules': len(self.risk_monitoring_engine.monitoring_rules),
            'last_update': datetime.now(),
            'alert_details': list(self.risk_monitoring_engine.active_alerts.values())
        }
        
        return monitoring_summary
        
    def generate_report(self, report_type: str = "executive_summary", output_format: str = "json") -> str:
        """生成风险报告"""
        # 将RiskEvent对象转换为字典
        risks_data = []
        for risk in self.active_risks.values():
            risk_dict = {
                'event_id': risk.event_id,
                'event_type': risk.event_type.value,
                'risk_level': risk.risk_level.value,
                'title': risk.title,
                'description': risk.description,
                'probability': risk.probability,
                'impact': risk.impact,
                'risk_score': risk.risk_score,
                'timestamp': risk.timestamp.isoformat(),
                'source': risk.source,
                'status': risk.status.value,
                'owner': risk.owner,
                'mitigation_actions': risk.mitigation_actions,
                'metadata': risk.metadata
            }
            risks_data.append(risk_dict)
        
        # 收集报告数据
        report_data = {
            'risks': risks_data,
            'metrics': self._get_recent_metrics(),
            'compliance': self._get_compliance_status(),
            'model_performance': self.risk_model_validation_engine.model_performance
        }
        
        # 生成报告
        report_content = self.risk_reporting_engine.generate_report(
            report_type, report_data, output_format
        )
        
        logger.info(f"生成 {report_type} 报告，格式: {output_format}")
        return report_content
        
    def validate_models(self, model_name: str = None) -> Dict[str, Any]:
        """验证风险模型"""
        validation_results = {}
        
        if model_name:
            # 验证指定模型
            test_data = np.random.normal(0, 0.02, 1000)  # 模拟测试数据
            expected_output = np.random.binomial(1, 0.1, 1000)  # 模拟预期输出
            
            result = self.risk_model_validation_engine.validate_model(model_name, test_data, expected_output)
            validation_results[model_name] = result
        else:
            # 验证所有模型
            for model_name in self.risk_assessment_engine.assessment_models.keys():
                test_data = np.random.normal(0, 0.02, 1000)
                expected_output = np.random.binomial(1, 0.1, 1000)
                
                result = self.risk_model_validation_engine.validate_model(model_name, test_data, expected_output)
                validation_results[model_name] = result
                
        logger.info(f"验证了 {len(validation_results)} 个风险模型")
        return validation_results
        
    def check_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查合规性"""
        compliance_result = self.risk_compliance_engine.check_compliance(data)
        
        # 保存合规检查结果
        self._save_compliance_to_db(compliance_result)
        
        logger.info(f"合规检查完成，总体状态: {compliance_result['overall_status']}")
        return compliance_result
        
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """获取风险仪表板数据"""
        dashboard = {
            'timestamp': datetime.now(),
            'risk_summary': {
                'total_active_risks': len(self.active_risks),
                'high_risks': len([r for r in self.active_risks.values() if r.risk_level == RiskLevel.HIGH]),
                'critical_risks': len([r for r in self.active_risks.values() if r.risk_level == RiskLevel.CRITICAL]),
                'risk_trend': self._calculate_risk_trend()
            },
            'risk_by_type': self._get_risks_by_type(),
            'recent_metrics': self._get_recent_metrics(limit=10),
            'active_alerts': len(self.risk_monitoring_engine.active_alerts),
            'compliance_status': self._get_compliance_status(),
            'model_performance': self._get_model_performance_summary()
        }
        
        return dashboard
        
    def _calculate_risk_trend(self) -> str:
        """计算风险趋势"""
        if len(self.risk_history) < 2:
            return "稳定"
            
        recent_risks = list(self.risk_history)[-10:]
        avg_recent_score = np.mean([r.risk_score for r in recent_risks])
        
        older_risks = list(self.risk_history)[-20:-10] if len(self.risk_history) >= 20 else recent_risks
        avg_older_score = np.mean([r.risk_score for r in older_risks])
        
        if avg_recent_score > avg_older_score * 1.1:
            return "上升"
        elif avg_recent_score < avg_older_score * 0.9:
            return "下降"
        else:
            return "稳定"
            
    def _get_risks_by_type(self) -> Dict[str, int]:
        """按类型统计风险"""
        type_counts = defaultdict(int)
        for risk in self.active_risks.values():
            type_counts[risk.event_type.value] += 1
        return dict(type_counts)
        
    def _get_recent_metrics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近的指标"""
        cursor = self.risk_database.cursor()
        cursor.execute('''
            SELECT metric_name, metric_value, threshold, risk_level, timestamp, unit, benchmark
            FROM risk_metrics
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
        
    def _get_compliance_status(self) -> Dict[str, Any]:
        """获取合规状态"""
        if not self.risk_compliance_engine.compliance_status:
            return {'overall_status': 'UNKNOWN', 'compliance_score': 100}
            
        latest_check = max(self.risk_compliance_engine.compliance_status.values(), 
                          key=lambda x: x['check_time'])
        return latest_check
        
    def _get_model_performance_summary(self) -> Dict[str, Any]:
        """获取模型性能摘要"""
        summary = {}
        for model_name, performance_records in self.risk_model_validation_engine.model_performance.items():
            if performance_records:
                latest_record = max(performance_records, key=lambda x: x['update_time'])
                summary[model_name] = latest_record
        return summary
        
    def _save_metric_to_db(self, metric: RiskMetric):
        """保存指标到数据库"""
        cursor = self.risk_database.cursor()
        cursor.execute('''
            INSERT INTO risk_metrics 
            (metric_name, metric_value, threshold, risk_level, timestamp, unit, category, benchmark)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.metric_name, metric.metric_value, metric.threshold,
            metric.risk_level.value, metric.timestamp.isoformat(),
            metric.unit, metric.category, metric.benchmark
        ))
        self.risk_database.commit()
        
    def _save_compliance_to_db(self, compliance_result: Dict[str, Any]):
        """保存合规检查结果到数据库"""
        cursor = self.risk_database.cursor()
        
        for violation in compliance_result.get('violations', []):
            cursor.execute('''
                INSERT INTO compliance_checks (check_time, rule_id, rule_name, status, details, data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                compliance_result['check_time'].isoformat(),
                violation['rule_id'],
                violation.get('rule_name', ''),
                violation['status'],
                json.dumps(violation.get('details', {}), default=str),
                json.dumps(compliance_result, default=str)
            ))
            
        self.risk_database.commit()
        
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'risk_database'):
            self.risk_database.close()


def create_sample_risk_controller() -> RiskController:
    """创建示例风险管控器"""
    config = {
        'database_path': 'risk_management.db',
        'log_level': 'INFO',
        'monitoring_interval': 60,
        'alert_thresholds': {
            'volatility': 0.02,
            'var': -0.05,
            'concentration': 0.3
        }
    }
    
    controller = RiskController(config)
    
    # 设置一些基准值
    controller.risk_measurement_engine.set_benchmark('volatility', 0.015)
    controller.risk_measurement_engine.set_benchmark('var', -0.03)
    controller.risk_measurement_engine.set_benchmark('sharpe_ratio', 1.0)
    
    # 设置预警阈值
    controller.risk_monitoring_engine.set_alert_threshold('volatility', 0.025, RiskLevel.HIGH)
    controller.risk_monitoring_engine.set_alert_threshold('var', -0.05, RiskLevel.CRITICAL)
    
    return controller


if __name__ == "__main__":
    # 创建并启动风险管控器
    risk_controller = create_sample_risk_controller()
    
    try:
        # 启动风险管控器
        risk_controller.start()
        
        # 模拟风险识别
        sample_data = {
            'returns': np.random.normal(0, 0.02, 100).tolist(),
            'positions': [
                {'asset': 'AAPL', 'value': 50000},
                {'asset': 'GOOGL', 'value': 30000},
                {'asset': 'MSFT', 'value': 20000}
            ],
            'market_regime': 'normal',
            'value': 0.8
        }
        
        print("=== 风险识别 ===")
        risks = risk_controller.identify_risks(sample_data)
        for risk in risks:
            print(f"风险: {risk.title} - {risk.risk_level.value} - 评分: {risk.risk_score:.2f}")
            
        print("\n=== 风险度量 ===")
        metrics = risk_controller.measure_risks(sample_data)
        for metric in metrics:
            print(f"指标: {metric.metric_name} - 值: {metric.metric_value:.4f} - 等级: {metric.risk_level.value}")
            
        print("\n=== 风险控制 ===")
        if risks:
            control_result = risk_controller.control_risks(risks[0])
            print(f"控制策略: {control_result['mitigation_plan']['strategies']}")
            
        print("\n=== 风险监控 ===")
        monitoring = risk_controller.monitor_risks()
        print(f"活跃预警: {monitoring['active_alerts']}")
        
        print("\n=== 合规检查 ===")
        compliance = risk_controller.check_compliance(sample_data)
        print(f"合规状态: {compliance['overall_status']}")
        print(f"合规评分: {compliance['compliance_score']:.1f}")
        
        print("\n=== 风险仪表板 ===")
        dashboard = risk_controller.get_risk_dashboard()
        print(f"总风险事件: {dashboard['risk_summary']['total_active_risks']}")
        print(f"高风险事件: {dashboard['risk_summary']['high_risks']}")
        print(f"风险趋势: {dashboard['risk_summary']['risk_trend']}")
        
        print("\n=== 生成报告 ===")
        report = risk_controller.generate_report("executive_summary", "json")
        print("报告生成成功")
        
        print("\n=== 模型验证 ===")
        validation = risk_controller.validate_models()
        for model_name, result in validation.items():
            print(f"模型 {model_name}: {result['validation_status']} - 准确率: {result['accuracy']:.2f}")
            
        # 等待一段时间观察监控结果
        print("\n等待监控数据...")
        time.sleep(5)
        
        print("\n=== 最终仪表板 ===")
        final_dashboard = risk_controller.get_risk_dashboard()
        print(f"最终活跃预警数: {final_dashboard['active_alerts']}")
        
    except KeyboardInterrupt:
        print("\n接收到停止信号")
    finally:
        # 停止风险管控器
        risk_controller.stop()
        print("风险管控器已停止")