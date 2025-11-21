"""
M7风险监控器模块

该模块实现了一个全面的风险监控系统，包括：
1. 市场风险监控 - 监控价格波动、波动率、相关性等市场指标
2. 信用风险监控 - 监控交易对手信用状况、违约概率等
3. 操作风险监控 - 监控系统故障、人为错误、流程风险等
4. 流动性风险监控 - 监控市场流动性、持仓流动性等
5. 技术风险监控 - 监控系统性能、安全威胁等
6. 合规风险监控 - 监控监管合规性、交易限制等
7. 风险指标计算 - 计算各类风险指标和度量
8. 风险预警机制 - 基于阈值的风险预警系统
9. 风险监控报告 - 生成详细的风险监控报告

Author: AI量化交易系统
Date: 2025-11-05
"""

import logging
import json
import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    CRITICAL = "严重风险"


class RiskType(Enum):
    """风险类型枚举"""
    MARKET = "市场风险"
    CREDIT = "信用风险"
    OPERATIONAL = "操作风险"
    LIQUIDITY = "流动性风险"
    TECHNICAL = "技术风险"
    COMPLIANCE = "合规风险"


@dataclass
class RiskMetrics:
    """风险指标数据类"""
    risk_type: RiskType
    metric_name: str
    current_value: float
    threshold_low: float
    threshold_medium: float
    threshold_high: float
    timestamp: datetime.datetime
    risk_level: RiskLevel
    description: str = ""


@dataclass
class RiskAlert:
    """风险预警数据类"""
    alert_id: str
    risk_type: RiskType
    risk_level: RiskLevel
    message: str
    timestamp: datetime.datetime
    metrics: List[RiskMetrics]
    recommendations: List[str]


@dataclass
class RiskReport:
    """风险监控报告数据类"""
    report_id: str
    timestamp: datetime.datetime
    overall_risk_level: RiskLevel
    risk_summary: Dict[RiskType, Dict[str, Any]]
    alerts: List[RiskAlert]
    recommendations: List[str]
    key_metrics: List[RiskMetrics]


class RiskMonitor:
    """
    M7风险监控器主类
    
    提供全面的风险监控功能，包括各类风险的实时监控、
    指标计算、预警机制和报告生成。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化风险监控器
        
        Args:
            config: 配置字典，包含各种风险监控参数
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 风险指标存储
        self.risk_metrics: Dict[RiskType, List[RiskMetrics]] = {
            risk_type: [] for risk_type in RiskType
        }
        
        # 风险预警存储
        self.alerts: List[RiskAlert] = []
        
        # 风险阈值配置
        self.thresholds = self.config.get('thresholds', {})
        
        # 监控开关
        self.monitoring_enabled = True
        
        # 历史数据存储
        self.historical_data: Dict[str, List[Dict]] = defaultdict(list)
        
        self.logger.info("M7风险监控器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'thresholds': {
                'volatility': {'low': 0.1, 'medium': 0.2, 'high': 0.3},
                'var_95': {'low': 0.02, 'medium': 0.05, 'high': 0.1},
                'sharpe_ratio': {'low': 0.5, 'medium': 1.0, 'high': 2.0},
                'max_drawdown': {'low': 0.05, 'medium': 0.1, 'high': 0.2},
                'credit_score': {'low': 700, 'medium': 600, 'high': 500},
                'liquidity_ratio': {'low': 0.1, 'medium': 0.05, 'high': 0.02},
                'system_uptime': {'low': 0.999, 'medium': 0.995, 'high': 0.99},
                'compliance_score': {'low': 0.95, 'medium': 0.9, 'high': 0.8}
            },
            'alert_cooldown': 300,  # 5分钟预警冷却时间
            'max_alerts': 1000,     # 最大预警数量
            'data_retention_days': 30  # 数据保留天数
        }
    
    def monitor_market_risk(self, price_data: pd.DataFrame, 
                          positions: Dict[str, float]) -> List[RiskMetrics]:
        """
        市场风险监控
        
        Args:
            price_data: 价格数据，包含各种资产的价格时间序列
            positions: 持仓数据，{资产代码: 持仓数量}
            
        Returns:
            List[RiskMetrics]: 市场风险指标列表
        """
        metrics = []
        
        try:
            # 计算波动率
            volatility = self._calculate_volatility(price_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.MARKET,
                metric_name="历史波动率",
                current_value=volatility,
                threshold_low=self.thresholds['volatility']['low'],
                threshold_medium=self.thresholds['volatility']['medium'],
                threshold_high=self.thresholds['volatility']['high'],
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(volatility, self.thresholds['volatility']),
                description="基于历史数据计算的资产价格波动率"
            ))
            
            # 计算VaR (Value at Risk)
            var_95 = self._calculate_var(price_data, confidence_level=0.95)
            metrics.append(RiskMetrics(
                risk_type=RiskType.MARKET,
                metric_name="95% VaR",
                current_value=var_95,
                threshold_low=self.thresholds['var_95']['low'],
                threshold_medium=self.thresholds['var_95']['medium'],
                threshold_high=self.thresholds['var_95']['high'],
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(var_95, self.thresholds['var_95']),
                description="95%置信水平下的风险价值"
            ))
            
            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio(price_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.MARKET,
                metric_name="夏普比率",
                current_value=sharpe_ratio,
                threshold_low=self.thresholds['sharpe_ratio']['low'],
                threshold_medium=self.thresholds['sharpe_ratio']['medium'],
                threshold_high=self.thresholds['sharpe_ratio']['high'],
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(-sharpe_ratio, 
                                                    {k: -v for k, v in self.thresholds['sharpe_ratio'].items()}),
                description="风险调整后的收益指标"
            ))
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown(price_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.MARKET,
                metric_name="最大回撤",
                current_value=max_drawdown,
                threshold_low=self.thresholds['max_drawdown']['low'],
                threshold_medium=self.thresholds['max_drawdown']['medium'],
                threshold_high=self.thresholds['max_drawdown']['high'],
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(max_drawdown, self.thresholds['max_drawdown']),
                description="从峰值到谷值的最大跌幅"
            ))
            
            # 计算投资组合集中度风险
            concentration_risk = self._calculate_concentration_risk(positions)
            metrics.append(RiskMetrics(
                risk_type=RiskType.MARKET,
                metric_name="投资组合集中度",
                current_value=concentration_risk,
                threshold_low=0.1,
                threshold_medium=0.2,
                threshold_high=0.3,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(concentration_risk, 
                                                    {'low': 0.1, 'medium': 0.2, 'high': 0.3}),
                description="投资组合集中度风险指标"
            ))
            
            self.risk_metrics[RiskType.MARKET].extend(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"市场风险监控失败: {e}")
            return []
    
    def monitor_credit_risk(self, counterparty_data: Dict[str, Dict], 
                          exposure_data: Dict[str, float]) -> List[RiskMetrics]:
        """
        信用风险监控
        
        Args:
            counterparty_data: 交易对手数据，包含信用评级等信息
            exposure_data: 敞口数据，{交易对手: 敞口金额}
            
        Returns:
            List[RiskMetrics]: 信用风险指标列表
        """
        metrics = []
        
        try:
            # 计算平均信用评级
            avg_credit_score = self._calculate_average_credit_score(counterparty_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.CREDIT,
                metric_name="平均信用评分",
                current_value=avg_credit_score,
                threshold_low=self.thresholds['credit_score']['low'],
                threshold_medium=self.thresholds['credit_score']['medium'],
                threshold_high=self.thresholds['credit_score']['high'],
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(-avg_credit_score, 
                                                    {k: -v for k, v in self.thresholds['credit_score'].items()}),
                description="交易对手平均信用评分"
            ))
            
            # 计算信用敞口集中度
            credit_concentration = self._calculate_credit_concentration(exposure_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.CREDIT,
                metric_name="信用敞口集中度",
                current_value=credit_concentration,
                threshold_low=0.1,
                threshold_medium=0.2,
                threshold_high=0.3,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(credit_concentration, 
                                                    {'low': 0.1, 'medium': 0.2, 'high': 0.3}),
                description="信用敞口集中度风险"
            ))
            
            # 计算预期违约概率
            expected_default_rate = self._calculate_expected_default_rate(counterparty_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.CREDIT,
                metric_name="预期违约率",
                current_value=expected_default_rate,
                threshold_low=0.01,
                threshold_medium=0.03,
                threshold_high=0.05,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(expected_default_rate, 
                                                    {'low': 0.01, 'medium': 0.03, 'high': 0.05}),
                description="基于交易对手评级的预期违约率"
            ))
            
            # 计算信用风险价值 (Credit VaR)
            credit_var = self._calculate_credit_var(exposure_data, counterparty_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.CREDIT,
                metric_name="信用VaR",
                current_value=credit_var,
                threshold_low=0.02,
                threshold_medium=0.05,
                threshold_high=0.1,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(credit_var, 
                                                    {'low': 0.02, 'medium': 0.05, 'high': 0.1}),
                description="信用风险价值"
            ))
            
            self.risk_metrics[RiskType.CREDIT].extend(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"信用风险监控失败: {e}")
            return []
    
    def monitor_operational_risk(self, system_logs: List[Dict], 
                               process_data: Dict[str, Any]) -> List[RiskMetrics]:
        """
        操作风险监控
        
        Args:
            system_logs: 系统日志数据
            process_data: 业务流程数据
            
        Returns:
            List[RiskMetrics]: 操作风险指标列表
        """
        metrics = []
        
        try:
            # 计算系统故障率
            failure_rate = self._calculate_system_failure_rate(system_logs)
            metrics.append(RiskMetrics(
                risk_type=RiskType.OPERATIONAL,
                metric_name="系统故障率",
                current_value=failure_rate,
                threshold_low=0.001,
                threshold_medium=0.005,
                threshold_high=0.01,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(failure_rate, 
                                                    {'low': 0.001, 'medium': 0.005, 'high': 0.01}),
                description="系统故障发生频率"
            ))
            
            # 计算人为错误率
            human_error_rate = self._calculate_human_error_rate(system_logs)
            metrics.append(RiskMetrics(
                risk_type=RiskType.OPERATIONAL,
                metric_name="人为错误率",
                current_value=human_error_rate,
                threshold_low=0.002,
                threshold_medium=0.005,
                threshold_high=0.01,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(human_error_rate, 
                                                    {'low': 0.002, 'medium': 0.005, 'high': 0.01}),
                description="人为操作错误频率"
            ))
            
            # 计算流程合规率
            process_compliance_rate = self._calculate_process_compliance_rate(process_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.OPERATIONAL,
                metric_name="流程合规率",
                current_value=process_compliance_rate,
                threshold_low=0.95,
                threshold_medium=0.9,
                threshold_high=0.8,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(-process_compliance_rate, 
                                                    {k: -v for k, v in {'low': 0.95, 'medium': 0.9, 'high': 0.8}.items()}),
                description="业务流程合规执行率"
            ))
            
            # 计算操作损失率
            operational_loss_rate = self._calculate_operational_loss_rate(system_logs)
            metrics.append(RiskMetrics(
                risk_type=RiskType.OPERATIONAL,
                metric_name="操作损失率",
                current_value=operational_loss_rate,
                threshold_low=0.001,
                threshold_medium=0.003,
                threshold_high=0.005,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(operational_loss_rate, 
                                                    {'low': 0.001, 'medium': 0.003, 'high': 0.005}),
                description="操作风险导致的损失率"
            ))
            
            self.risk_metrics[RiskType.OPERATIONAL].extend(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"操作风险监控失败: {e}")
            return []
    
    def monitor_liquidity_risk(self, market_data: Dict[str, Any], 
                             position_data: Dict[str, float]) -> List[RiskMetrics]:
        """
        流动性风险监控
        
        Args:
            market_data: 市场数据，包含买卖价差、成交量等信息
            position_data: 持仓数据
            
        Returns:
            List[RiskMetrics]: 流动性风险指标列表
        """
        metrics = []
        
        try:
            # 计算市场流动性指标
            market_liquidity = self._calculate_market_liquidity(market_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.LIQUIDITY,
                metric_name="市场流动性",
                current_value=market_liquidity,
                threshold_low=0.8,
                threshold_medium=0.6,
                threshold_high=0.4,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(-market_liquidity, 
                                                    {k: -v for k, v in {'low': 0.8, 'medium': 0.6, 'high': 0.4}.items()}),
                description="市场流动性充足程度"
            ))
            
            # 计算持仓流动性比率
            position_liquidity_ratio = self._calculate_position_liquidity_ratio(position_data, market_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.LIQUIDITY,
                metric_name="持仓流动性比率",
                current_value=position_liquidity_ratio,
                threshold_low=self.thresholds['liquidity_ratio']['low'],
                threshold_medium=self.thresholds['liquidity_ratio']['medium'],
                threshold_high=self.thresholds['liquidity_ratio']['high'],
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(position_liquidity_ratio, self.thresholds['liquidity_ratio']),
                description="持仓资产流动性充足程度"
            ))
            
            # 计算资金流动性覆盖率
            liquidity_coverage_ratio = self._calculate_liquidity_coverage_ratio(position_data, market_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.LIQUIDITY,
                metric_name="流动性覆盖率",
                current_value=liquidity_coverage_ratio,
                threshold_low=1.2,
                threshold_medium=1.1,
                threshold_high=1.0,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(-liquidity_coverage_ratio, 
                                                    {k: -v for k, v in {'low': 1.2, 'medium': 1.1, 'high': 1.0}.items()}),
                description="流动性覆盖率指标"
            ))
            
            # 计算流动性缺口
            liquidity_gap = self._calculate_liquidity_gap(position_data, market_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.LIQUIDITY,
                metric_name="流动性缺口",
                current_value=liquidity_gap,
                threshold_low=0.05,
                threshold_medium=0.1,
                threshold_high=0.2,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(abs(liquidity_gap), 
                                                    {'low': 0.05, 'medium': 0.1, 'high': 0.2}),
                description="流动性缺口分析"
            ))
            
            self.risk_metrics[RiskType.LIQUIDITY].extend(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"流动性风险监控失败: {e}")
            return []
    
    def monitor_technical_risk(self, system_metrics: Dict[str, Any], 
                             security_data: Dict[str, Any]) -> List[RiskMetrics]:
        """
        技术风险监控
        
        Args:
            system_metrics: 系统性能指标
            security_data: 安全相关数据
            
        Returns:
            List[RiskMetrics]: 技术风险指标列表
        """
        metrics = []
        
        try:
            # 计算系统可用性
            system_uptime = self._calculate_system_uptime(system_metrics)
            metrics.append(RiskMetrics(
                risk_type=RiskType.TECHNICAL,
                metric_name="系统可用性",
                current_value=system_uptime,
                threshold_low=self.thresholds['system_uptime']['low'],
                threshold_medium=self.thresholds['system_uptime']['medium'],
                threshold_high=self.thresholds['system_uptime']['high'],
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(-system_uptime, 
                                                    {k: -v for k, v in self.thresholds['system_uptime'].items()}),
                description="系统运行时间占比"
            ))
            
            # 计算系统响应时间
            response_time = self._calculate_average_response_time(system_metrics)
            metrics.append(RiskMetrics(
                risk_type=RiskType.TECHNICAL,
                metric_name="平均响应时间",
                current_value=response_time,
                threshold_low=100,
                threshold_medium=500,
                threshold_high=1000,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(response_time, 
                                                    {'low': 100, 'medium': 500, 'high': 1000}),
                description="系统平均响应时间(毫秒)"
            ))
            
            # 计算安全威胁等级
            security_threat_level = self._calculate_security_threat_level(security_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.TECHNICAL,
                metric_name="安全威胁等级",
                current_value=security_threat_level,
                threshold_low=0.1,
                threshold_medium=0.3,
                threshold_high=0.5,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(security_threat_level, 
                                                    {'low': 0.1, 'medium': 0.3, 'high': 0.5}),
                description="安全威胁严重程度"
            ))
            
            # 计算数据完整性
            data_integrity = self._calculate_data_integrity(system_metrics)
            metrics.append(RiskMetrics(
                risk_type=RiskType.TECHNICAL,
                metric_name="数据完整性",
                current_value=data_integrity,
                threshold_low=0.999,
                threshold_medium=0.995,
                threshold_high=0.99,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(-data_integrity, 
                                                    {k: -v for k, v in {'low': 0.999, 'medium': 0.995, 'high': 0.99}.items()}),
                description="数据完整性和一致性"
            ))
            
            self.risk_metrics[RiskType.TECHNICAL].extend(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"技术风险监控失败: {e}")
            return []
    
    def monitor_compliance_risk(self, trading_data: Dict[str, Any], 
                              regulatory_data: Dict[str, Any]) -> List[RiskMetrics]:
        """
        合规风险监控
        
        Args:
            trading_data: 交易数据
            regulatory_data: 监管规则数据
            
        Returns:
            List[RiskMetrics]: 合规风险指标列表
        """
        metrics = []
        
        try:
            # 计算交易合规率
            trading_compliance_rate = self._calculate_trading_compliance_rate(trading_data, regulatory_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.COMPLIANCE,
                metric_name="交易合规率",
                current_value=trading_compliance_rate,
                threshold_low=self.thresholds['compliance_score']['low'],
                threshold_medium=self.thresholds['compliance_score']['medium'],
                threshold_high=self.thresholds['compliance_score']['high'],
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(-trading_compliance_rate, 
                                                    {k: -v for k, v in self.thresholds['compliance_score'].items()}),
                description="交易活动合规执行率"
            ))
            
            # 计算监管报告及时性
            reporting_timeliness = self._calculate_reporting_timeliness(trading_data, regulatory_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.COMPLIANCE,
                metric_name="报告及时性",
                current_value=reporting_timeliness,
                threshold_low=0.95,
                threshold_medium=0.9,
                threshold_high=0.8,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(-reporting_timeliness, 
                                                    {k: -v for k, v in {'low': 0.95, 'medium': 0.9, 'high': 0.8}.items()}),
                description="监管报告提交及时性"
            ))
            
            # 计算违规事件频率
            violation_frequency = self._calculate_violation_frequency(trading_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.COMPLIANCE,
                metric_name="违规事件频率",
                current_value=violation_frequency,
                threshold_low=0.001,
                threshold_medium=0.005,
                threshold_high=0.01,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(violation_frequency, 
                                                    {'low': 0.001, 'medium': 0.005, 'high': 0.01}),
                description="违规事件发生频率"
            ))
            
            # 计算合规成本比率
            compliance_cost_ratio = self._calculate_compliance_cost_ratio(trading_data)
            metrics.append(RiskMetrics(
                risk_type=RiskType.COMPLIANCE,
                metric_name="合规成本比率",
                current_value=compliance_cost_ratio,
                threshold_low=0.02,
                threshold_medium=0.05,
                threshold_high=0.1,
                timestamp=datetime.datetime.now(),
                risk_level=self._calculate_risk_level(compliance_cost_ratio, 
                                                    {'low': 0.02, 'medium': 0.05, 'high': 0.1}),
                description="合规成本占总成本比例"
            ))
            
            self.risk_metrics[RiskType.COMPLIANCE].extend(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"合规风险监控失败: {e}")
            return []
    
    def check_risk_alerts(self) -> List[RiskAlert]:
        """
        检查风险预警
        
        Returns:
            List[RiskAlert]: 风险预警列表
        """
        new_alerts = []
        
        try:
            for risk_type, metrics_list in self.risk_metrics.items():
                for metric in metrics_list:
                    if metric.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                        alert = self._create_risk_alert(metric)
                        if alert:
                            new_alerts.append(alert)
            
            # 添加到预警列表
            self.alerts.extend(new_alerts)
            
            # 保持预警列表大小限制
            if len(self.alerts) > self.config['max_alerts']:
                self.alerts = self.alerts[-self.config['max_alerts']:]
            
            return new_alerts
            
        except Exception as e:
            self.logger.error(f"风险预警检查失败: {e}")
            return []
    
    def generate_risk_report(self) -> RiskReport:
        """
        生成风险监控报告
        
        Returns:
            RiskReport: 风险监控报告
        """
        try:
            report_id = f"risk_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 计算整体风险等级
            overall_risk_level = self._calculate_overall_risk_level()
            
            # 生成风险摘要
            risk_summary = self._generate_risk_summary()
            
            # 生成建议
            recommendations = self._generate_recommendations()
            
            # 获取关键指标
            key_metrics = self._get_key_metrics()
            
            report = RiskReport(
                report_id=report_id,
                timestamp=datetime.datetime.now(),
                overall_risk_level=overall_risk_level,
                risk_summary=risk_summary,
                alerts=self.alerts[-10:],  # 最近10条预警
                recommendations=recommendations,
                key_metrics=key_metrics
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"风险报告生成失败: {e}")
            raise
    
    def export_report(self, report: RiskReport, file_path: str, format: str = "json") -> bool:
        """
        导出风险报告
        
        Args:
            report: 风险报告
            file_path: 文件路径
            format: 导出格式 ("json", "csv", "html")
            
        Returns:
            bool: 导出是否成功
        """
        try:
            if format.lower() == "json":
                # 转换RiskType枚举为字符串
                report_dict = asdict(report)
                self._convert_enums_to_strings(report_dict)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
            
            elif format.lower() == "csv":
                # 将报告转换为CSV格式
                self._export_report_to_csv(report, file_path)
            
            elif format.lower() == "html":
                # 生成HTML报告
                self._export_report_to_html(report, file_path)
            
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"风险报告已导出到: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"报告导出失败: {e}")
            return False
    
    def _convert_enums_to_strings(self, obj: Any):
        """
        将枚举对象转换为字符串，用于JSON序列化
        
        Args:
            obj: 要转换的对象
        """
        if isinstance(obj, dict):
            keys_to_convert = []
            for key, value in obj.items():
                if isinstance(key, (RiskType, RiskLevel)):
                    keys_to_convert.append((key, str(key.value)))
                elif isinstance(value, (RiskType, RiskLevel)):
                    obj[key] = str(value.value)
                else:
                    self._convert_enums_to_strings(value)
            
            # 转换键
            for old_key, new_key in keys_to_convert:
                obj[new_key] = obj.pop(old_key)
                
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (RiskType, RiskLevel)):
                    obj[i] = str(item.value)
                else:
                    self._convert_enums_to_strings(item)
    
    # ==================== 私有方法 ====================
    
    def _calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """计算历史波动率"""
        returns = price_data.pct_change().dropna()
        # 处理多列数据的情况
        if returns.empty or len(returns) == 0:
            return 0.0
        volatilities = returns.std()
        # 处理Series和DataFrame的情况
        if isinstance(volatilities, pd.Series):
            return volatilities.mean()
        elif isinstance(volatilities, pd.DataFrame):
            return volatilities.values.mean()
        else:
            return float(volatilities)
    
    def _calculate_var(self, price_data: pd.DataFrame, confidence_level: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        returns = price_data.pct_change().dropna()
        if returns.empty:
            return 0.0
        var_value = abs(returns.quantile(1 - confidence_level))
        # 确保返回标量值
        if hasattr(var_value, 'iloc'):
            return float(var_value.iloc[0]) if len(var_value) > 0 else 0.0
        else:
            return float(var_value)
    
    def _calculate_sharpe_ratio(self, price_data: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        returns = price_data.pct_change().dropna()
        if returns.empty:
            return 0.0
        mean_return = returns.mean()
        std_return = returns.std()
        
        # 确保返回标量值
        if hasattr(mean_return, 'iloc'):
            mean_return = float(mean_return.mean())
        else:
            mean_return = float(mean_return)
            
        if hasattr(std_return, 'iloc'):
            std_return = float(std_return.mean())
        else:
            std_return = float(std_return)
        
        if std_return == 0:
            return 0.0
            
        excess_returns = mean_return * 252 - risk_free_rate
        return excess_returns / (std_return * np.sqrt(252))
    
    def _calculate_max_drawdown(self, price_data: pd.DataFrame) -> float:
        """计算最大回撤"""
        cumulative = (1 + price_data.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        min_drawdown = drawdown.min()
        # 确保返回标量值
        if hasattr(min_drawdown, 'iloc'):
            return abs(float(min_drawdown.min()))
        else:
            return abs(float(min_drawdown))
    
    def _calculate_concentration_risk(self, positions: Dict[str, float]) -> float:
        """计算投资组合集中度风险"""
        total_value = sum(abs(pos) for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        weights = [abs(pos) / total_value for pos in positions.values()]
        hhi = sum(w**2 for w in weights)  # 赫芬达尔指数
        return hhi
    
    def _calculate_average_credit_score(self, counterparty_data: Dict[str, Dict]) -> float:
        """计算平均信用评分"""
        scores = [data.get('credit_score', 600) for data in counterparty_data.values()]
        return sum(scores) / len(scores) if scores else 600
    
    def _calculate_credit_concentration(self, exposure_data: Dict[str, float]) -> float:
        """计算信用敞口集中度"""
        total_exposure = sum(exposure_data.values())
        if total_exposure == 0:
            return 0.0
        
        weights = [exp / total_exposure for exp in exposure_data.values()]
        hhi = sum(w**2 for w in weights)
        return hhi
    
    def _calculate_expected_default_rate(self, counterparty_data: Dict[str, Dict]) -> float:
        """计算预期违约率"""
        default_rates = []
        for data in counterparty_data.values():
            credit_score = data.get('credit_score', 600)
            # 简化的信用评分到违约率映射
            if credit_score >= 800:
                default_rate = 0.001
            elif credit_score >= 700:
                default_rate = 0.005
            elif credit_score >= 600:
                default_rate = 0.02
            else:
                default_rate = 0.05
            default_rates.append(default_rate)
        
        return sum(default_rates) / len(default_rates) if default_rates else 0.02
    
    def _calculate_credit_var(self, exposure_data: Dict[str, float], 
                            counterparty_data: Dict[str, Dict]) -> float:
        """计算信用VaR"""
        credit_var = 0.0
        for counterparty, exposure in exposure_data.items():
            if counterparty in counterparty_data:
                credit_score = counterparty_data[counterparty].get('credit_score', 600)
                # 简化的信用VaR计算
                if credit_score >= 700:
                    loss_rate = 0.1
                elif credit_score >= 600:
                    loss_rate = 0.4
                else:
                    loss_rate = 0.7
                credit_var += exposure * loss_rate
        
        total_exposure = sum(exposure_data.values())
        return credit_var / total_exposure if total_exposure > 0 else 0.0
    
    def _calculate_system_failure_rate(self, system_logs: List[Dict]) -> float:
        """计算系统故障率"""
        if not system_logs:
            return 0.0
        
        failure_count = sum(1 for log in system_logs if log.get('level') == 'ERROR')
        return failure_count / len(system_logs)
    
    def _calculate_human_error_rate(self, system_logs: List[Dict]) -> float:
        """计算人为错误率"""
        if not system_logs:
            return 0.0
        
        error_count = sum(1 for log in system_logs 
                         if 'user_error' in log.get('message', '').lower())
        return error_count / len(system_logs)
    
    def _calculate_process_compliance_rate(self, process_data: Dict[str, Any]) -> float:
        """计算流程合规率"""
        total_processes = process_data.get('total_processes', 0)
        compliant_processes = process_data.get('compliant_processes', 0)
        
        return compliant_processes / total_processes if total_processes > 0 else 1.0
    
    def _calculate_operational_loss_rate(self, system_logs: List[Dict]) -> float:
        """计算操作损失率"""
        if not system_logs:
            return 0.0
        
        total_loss = sum(log.get('loss_amount', 0) for log in system_logs 
                        if log.get('level') == 'ERROR')
        total_volume = sum(log.get('transaction_volume', 1) for log in system_logs)
        
        return total_loss / total_volume if total_volume > 0 else 0.0
    
    def _calculate_market_liquidity(self, market_data: Dict[str, Any]) -> float:
        """计算市场流动性指标"""
        bid_ask_spreads = market_data.get('bid_ask_spreads', [])
        volumes = market_data.get('volumes', [])
        
        if not bid_ask_spreads or not volumes:
            return 0.5  # 默认中等流动性
        
        # 流动性指标：成交量大、价差小表示流动性好
        avg_spread = np.mean(bid_ask_spreads)
        avg_volume = np.mean(volumes)
        
        # 简化的流动性计算
        liquidity_score = min(1.0, avg_volume / (avg_spread * 1000))
        return liquidity_score
    
    def _calculate_position_liquidity_ratio(self, position_data: Dict[str, float], 
                                          market_data: Dict[str, Any]) -> float:
        """计算持仓流动性比率"""
        total_position = sum(abs(pos) for pos in position_data.values())
        if total_position == 0:
            return 1.0
        
        liquid_positions = sum(abs(pos) for symbol, pos in position_data.items() 
                              if symbol in market_data.get('liquid_assets', []))
        
        return liquid_positions / total_position
    
    def _calculate_liquidity_coverage_ratio(self, position_data: Dict[str, float], 
                                          market_data: Dict[str, Any]) -> float:
        """计算流动性覆盖率"""
        liquid_assets = market_data.get('liquid_assets_value', 0)
        total_liabilities = sum(abs(pos) for pos in position_data.values())
        
        return liquid_assets / total_liabilities if total_liabilities > 0 else float('inf')
    
    def _calculate_liquidity_gap(self, position_data: Dict[str, float], 
                               market_data: Dict[str, Any]) -> float:
        """计算流动性缺口"""
        assets = sum(pos for pos in position_data.values() if pos > 0)
        liabilities = sum(abs(pos) for pos in position_data.values() if pos < 0)
        
        return (assets - liabilities) / max(assets, liabilities, 1)
    
    def _calculate_system_uptime(self, system_metrics: Dict[str, Any]) -> float:
        """计算系统可用性"""
        uptime = system_metrics.get('uptime_seconds', 0)
        total_time = system_metrics.get('total_time_seconds', 1)
        return uptime / total_time
    
    def _calculate_average_response_time(self, system_metrics: Dict[str, Any]) -> float:
        """计算平均响应时间"""
        response_times = system_metrics.get('response_times', [])
        return np.mean(response_times) if response_times else 100
    
    def _calculate_security_threat_level(self, security_data: Dict[str, Any]) -> float:
        """计算安全威胁等级"""
        threats = security_data.get('threats', [])
        if not threats:
            return 0.0
        
        threat_scores = [threat.get('severity', 1) for threat in threats]
        return np.mean(threat_scores)
    
    def _calculate_data_integrity(self, system_metrics: Dict[str, Any]) -> float:
        """计算数据完整性"""
        total_records = system_metrics.get('total_records', 0)
        corrupted_records = system_metrics.get('corrupted_records', 0)
        
        return (total_records - corrupted_records) / total_records if total_records > 0 else 1.0
    
    def _calculate_trading_compliance_rate(self, trading_data: Dict[str, Any], 
                                         regulatory_data: Dict[str, Any]) -> float:
        """计算交易合规率"""
        total_trades = trading_data.get('total_trades', 0)
        compliant_trades = trading_data.get('compliant_trades', 0)
        
        return compliant_trades / total_trades if total_trades > 0 else 1.0
    
    def _calculate_reporting_timeliness(self, trading_data: Dict[str, Any], 
                                      regulatory_data: Dict[str, Any]) -> float:
        """计算报告及时性"""
        total_reports = trading_data.get('total_reports', 0)
        timely_reports = trading_data.get('timely_reports', 0)
        
        return timely_reports / total_reports if total_reports > 0 else 1.0
    
    def _calculate_violation_frequency(self, trading_data: Dict[str, Any]) -> float:
        """计算违规事件频率"""
        total_activities = trading_data.get('total_activities', 0)
        violations = trading_data.get('violations', 0)
        
        return violations / total_activities if total_activities > 0 else 0.0
    
    def _calculate_compliance_cost_ratio(self, trading_data: Dict[str, Any]) -> float:
        """计算合规成本比率"""
        total_cost = trading_data.get('total_cost', 0)
        compliance_cost = trading_data.get('compliance_cost', 0)
        
        return compliance_cost / total_cost if total_cost > 0 else 0.0
    
    def _calculate_risk_level(self, value: float, thresholds: Dict[str, float]) -> RiskLevel:
        """根据阈值计算风险等级"""
        if value <= thresholds['low']:
            return RiskLevel.LOW
        elif value <= thresholds['medium']:
            return RiskLevel.MEDIUM
        elif value <= thresholds['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _create_risk_alert(self, metric: RiskMetrics) -> Optional[RiskAlert]:
        """创建风险预警"""
        # 检查预警冷却时间
        recent_alerts = [a for a in self.alerts 
                        if (datetime.datetime.now() - a.timestamp).seconds < self.config['alert_cooldown']
                        and a.risk_type == metric.risk_type]
        
        if recent_alerts:
            return None  # 在冷却期内，不创建新预警
        
        alert_id = f"alert_{metric.risk_type.value}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 生成预警消息和建议
        message = self._generate_alert_message(metric)
        recommendations = self._generate_alert_recommendations(metric)
        
        return RiskAlert(
            alert_id=alert_id,
            risk_type=metric.risk_type,
            risk_level=metric.risk_level,
            message=message,
            timestamp=metric.timestamp,
            metrics=[metric],
            recommendations=recommendations
        )
    
    def _generate_alert_message(self, metric: RiskMetrics) -> str:
        """生成预警消息"""
        return f"{metric.risk_type.value} - {metric.metric_name}当前值为{metric.current_value:.4f}，" \
               f"风险等级: {metric.risk_level.value}"
    
    def _generate_alert_recommendations(self, metric: RiskMetrics) -> List[str]:
        """生成预警建议"""
        recommendations = []
        
        if metric.risk_type == RiskType.MARKET:
            if "波动率" in metric.metric_name:
                recommendations.append("考虑增加对冲策略")
                recommendations.append("降低高波动性资产仓位")
            elif "VaR" in metric.metric_name:
                recommendations.append("减少风险敞口")
                recommendations.append("增加风险缓释措施")
        
        elif metric.risk_type == RiskType.CREDIT:
            recommendations.append("加强交易对手信用监控")
            recommendations.append("考虑降低高风险对手方敞口")
        
        elif metric.risk_type == RiskType.OPERATIONAL:
            recommendations.append("加强系统维护")
            recommendations.append("完善操作流程")
        
        elif metric.risk_type == RiskType.LIQUIDITY:
            recommendations.append("增加流动性资产配置")
            recommendations.append("优化资产负债结构")
        
        elif metric.risk_type == RiskType.TECHNICAL:
            recommendations.append("加强系统监控")
            recommendations.append("完善备份和恢复机制")
        
        elif metric.risk_type == RiskType.COMPLIANCE:
            recommendations.append("加强合规培训")
            recommendations.append("完善内控制度")
        
        return recommendations
    
    def _calculate_overall_risk_level(self) -> RiskLevel:
        """计算整体风险等级"""
        all_metrics = []
        for metrics_list in self.risk_metrics.values():
            all_metrics.extend(metrics_list)
        
        if not all_metrics:
            return RiskLevel.LOW
        
        # 风险等级权重
        level_weights = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        
        weighted_score = sum(level_weights[metric.risk_level] for metric in all_metrics)
        average_score = weighted_score / len(all_metrics)
        
        if average_score <= 1.5:
            return RiskLevel.LOW
        elif average_score <= 2.5:
            return RiskLevel.MEDIUM
        elif average_score <= 3.5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_risk_summary(self) -> Dict[RiskType, Dict[str, Any]]:
        """生成风险摘要"""
        summary = {}
        
        for risk_type, metrics_list in self.risk_metrics.items():
            if not metrics_list:
                continue
            
            latest_metrics = metrics_list[-5:]  # 最近5个指标
            
            summary[risk_type] = {
                'metric_count': len(metrics_list),
                'latest_metrics': [
                    {
                        'name': m.metric_name,
                        'value': m.current_value,
                        'risk_level': m.risk_level.value
                    }
                    for m in latest_metrics
                ],
                'risk_distribution': self._calculate_risk_distribution(metrics_list)
            }
        
        return summary
    
    def _calculate_risk_distribution(self, metrics_list: List[RiskMetrics]) -> Dict[str, int]:
        """计算风险等级分布"""
        distribution = {level.value: 0 for level in RiskLevel}
        
        for metric in metrics_list:
            distribution[metric.risk_level.value] += 1
        
        return distribution
    
    def _generate_recommendations(self) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        # 基于整体风险等级的建议
        overall_level = self._calculate_overall_risk_level()
        
        if overall_level == RiskLevel.CRITICAL:
            recommendations.append("立即采取风险缓释措施")
            recommendations.append("考虑暂停高风险交易活动")
        elif overall_level == RiskLevel.HIGH:
            recommendations.append("加强风险监控频率")
            recommendations.append("准备应急预案")
        elif overall_level == RiskLevel.MEDIUM:
            recommendations.append("定期评估风险状况")
            recommendations.append("优化风险管理策略")
        
        # 基于具体风险类型的建议
        for risk_type, metrics_list in self.risk_metrics.items():
            high_risk_metrics = [m for m in metrics_list if m.risk_level == RiskLevel.HIGH]
            if high_risk_metrics:
                recommendations.append(f"重点关注{risk_type.value}，当前有{len(high_risk_metrics)}项高风险指标")
        
        return recommendations
    
    def _get_key_metrics(self) -> List[RiskMetrics]:
        """获取关键指标"""
        key_metrics = []
        
        # 选择每种风险类型最新的指标
        for risk_type, metrics_list in self.risk_metrics.items():
            if metrics_list:
                latest_metric = max(metrics_list, key=lambda x: x.timestamp)
                key_metrics.append(latest_metric)
        
        return key_metrics
    
    def _export_report_to_csv(self, report: RiskReport, file_path: str):
        """导出报告为CSV格式"""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入报告基本信息
            writer.writerow(['报告ID', report.report_id])
            writer.writerow(['生成时间', report.timestamp])
            writer.writerow(['整体风险等级', report.overall_risk_level.value])
            writer.writerow([])
            
            # 写入关键指标
            writer.writerow(['关键指标'])
            writer.writerow(['风险类型', '指标名称', '当前值', '风险等级', '时间戳'])
            
            for metric in report.key_metrics:
                writer.writerow([
                    metric.risk_type.value,
                    metric.metric_name,
                    metric.current_value,
                    metric.risk_level.value,
                    metric.timestamp
                ])
            
            writer.writerow([])
            
            # 写入预警信息
            writer.writerow(['风险预警'])
            writer.writerow(['预警ID', '风险类型', '风险等级', '消息', '时间戳'])
            
            for alert in report.alerts:
                writer.writerow([
                    alert.alert_id,
                    alert.risk_type.value,
                    alert.risk_level.value,
                    alert.message,
                    alert.timestamp
                ])
    
    def _export_report_to_html(self, report: RiskReport, file_path: str):
        """导出报告为HTML格式"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>M7风险监控报告</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .alert {{ background-color: #ffe8e8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .risk-low {{ border-left: 5px solid green; }}
                .risk-medium {{ border-left: 5px solid orange; }}
                .risk-high {{ border-left: 5px solid red; }}
                .risk-critical {{ border-left: 5px solid darkred; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>M7风险监控报告</h1>
                <p><strong>报告ID:</strong> {report.report_id}</p>
                <p><strong>生成时间:</strong> {report.timestamp}</p>
                <p><strong>整体风险等级:</strong> {report.overall_risk_level.value}</p>
            </div>
            
            <div class="section">
                <h2>关键风险指标</h2>
        """
        
        for metric in report.key_metrics:
            risk_class = f"risk-{metric.risk_level.value.lower().replace('风险', '')}"
            html_content += f"""
                <div class="metric {risk_class}">
                    <strong>{metric.risk_type.value} - {metric.metric_name}</strong><br>
                    当前值: {metric.current_value:.4f}<br>
                    风险等级: {metric.risk_level.value}<br>
                    时间戳: {metric.timestamp}
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>风险预警</h2>
        """
        
        for alert in report.alerts:
            risk_class = f"risk-{alert.risk_level.value.lower().replace('风险', '')}"
            html_content += f"""
                <div class="alert {risk_class}">
                    <strong>{alert.risk_type.value}</strong><br>
                    等级: {alert.risk_level.value}<br>
                    消息: {alert.message}<br>
                    时间: {alert.timestamp}
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>管理建议</h2>
                <ul>
        """
        
        for rec in report.recommendations:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


# ==================== 测试用例 ====================

def test_risk_monitor():
    """风险监控器测试函数"""
    
    # 创建风险监控器实例
    monitor = RiskMonitor()
    
    # 模拟市场数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    price_data = pd.DataFrame({
        'AAPL': np.random.randn(100).cumsum() + 100,
        'GOOGL': np.random.randn(100).cumsum() + 2000,
        'MSFT': np.random.randn(100).cumsum() + 300
    }, index=dates)
    
    positions = {'AAPL': 1000, 'GOOGL': 500, 'MSFT': 800}
    
    # 测试市场风险监控
    print("=== 测试市场风险监控 ===")
    market_metrics = monitor.monitor_market_risk(price_data, positions)
    for metric in market_metrics:
        print(f"{metric.metric_name}: {metric.current_value:.4f} ({metric.risk_level.value})")
    
    # 模拟交易对手数据
    counterparty_data = {
        'counterparty1': {'credit_score': 750},
        'counterparty2': {'credit_score': 680},
        'counterparty3': {'credit_score': 820}
    }
    
    exposure_data = {
        'counterparty1': 1000000,
        'counterparty2': 500000,
        'counterparty3': 800000
    }
    
    # 测试信用风险监控
    print("\n=== 测试信用风险监控 ===")
    credit_metrics = monitor.monitor_credit_risk(counterparty_data, exposure_data)
    for metric in credit_metrics:
        print(f"{metric.metric_name}: {metric.current_value:.4f} ({metric.risk_level.value})")
    
    # 模拟系统日志
    system_logs = [
        {'level': 'INFO', 'message': 'Normal operation', 'timestamp': datetime.datetime.now()},
        {'level': 'ERROR', 'message': 'Database connection failed', 'timestamp': datetime.datetime.now()},
        {'level': 'WARNING', 'message': 'High memory usage', 'timestamp': datetime.datetime.now()},
        {'level': 'ERROR', 'message': 'User error: Invalid input', 'timestamp': datetime.datetime.now()}
    ]
    
    process_data = {
        'total_processes': 100,
        'compliant_processes': 95
    }
    
    # 测试操作风险监控
    print("\n=== 测试操作风险监控 ===")
    operational_metrics = monitor.monitor_operational_risk(system_logs, process_data)
    for metric in operational_metrics:
        print(f"{metric.metric_name}: {metric.current_value:.4f} ({metric.risk_level.value})")
    
    # 模拟市场流动性数据
    market_data = {
        'bid_ask_spreads': [0.01, 0.02, 0.015, 0.03],
        'volumes': [1000000, 2000000, 1500000, 800000],
        'liquid_assets_value': 5000000
    }
    
    # 测试流动性风险监控
    print("\n=== 测试流动性风险监控 ===")
    liquidity_metrics = monitor.monitor_liquidity_risk(market_data, positions)
    for metric in liquidity_metrics:
        print(f"{metric.metric_name}: {metric.current_value:.4f} ({metric.risk_level.value})")
    
    # 模拟系统性能数据
    system_metrics = {
        'uptime_seconds': 86400 * 30 - 300,  # 30天中5分钟 downtime
        'total_time_seconds': 86400 * 30,
        'response_times': [50, 80, 120, 90, 70, 110],
        'total_records': 10000,
        'corrupted_records': 2
    }
    
    security_data = {
        'threats': [
            {'severity': 0.2, 'type': 'malware'},
            {'severity': 0.1, 'type': 'phishing'}
        ]
    }
    
    # 测试技术风险监控
    print("\n=== 测试技术风险监控 ===")
    technical_metrics = monitor.monitor_technical_risk(system_metrics, security_data)
    for metric in technical_metrics:
        print(f"{metric.metric_name}: {metric.current_value:.4f} ({metric.risk_level.value})")
    
    # 模拟交易和监管数据
    trading_data = {
        'total_trades': 1000,
        'compliant_trades': 980,
        'total_reports': 50,
        'timely_reports': 48,
        'violations': 2,
        'total_cost': 1000000,
        'compliance_cost': 50000
    }
    
    regulatory_data = {
        'max_position_limit': 1000000,
        'reporting_deadline': 'T+1'
    }
    
    # 测试合规风险监控
    print("\n=== 测试合规风险监控 ===")
    compliance_metrics = monitor.monitor_compliance_risk(trading_data, regulatory_data)
    for metric in compliance_metrics:
        print(f"{metric.metric_name}: {metric.current_value:.4f} ({metric.risk_level.value})")
    
    # 检查风险预警
    print("\n=== 检查风险预警 ===")
    alerts = monitor.check_risk_alerts()
    for alert in alerts:
        print(f"预警: {alert.message}")
        print(f"建议: {', '.join(alert.recommendations)}")
    
    # 生成风险报告
    print("\n=== 生成风险报告 ===")
    report = monitor.generate_risk_report()
    print(f"整体风险等级: {report.overall_risk_level.value}")
    print(f"关键指标数量: {len(report.key_metrics)}")
    print(f"预警数量: {len(report.alerts)}")
    print(f"建议数量: {len(report.recommendations)}")
    
    # 导出报告
    print("\n=== 导出报告 ===")
    monitor.export_report(report, "/workspace/risk_report.json", "json")
    monitor.export_report(report, "/workspace/risk_report.csv", "csv")
    monitor.export_report(report, "/workspace/risk_report.html", "html")
    
    print("测试完成！")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    test_risk_monitor()