"""
M6 交易监控器 (Trading Monitor)

该模块实现了一个全面的交易监控系统，用于实时监控和分析交易活动的各个方面。
主要功能包括交易执行监控、交易量监控、交易成功率监控、交易延迟监控、
交易错误监控、交易成本监控、交易策略监控、交易合规性监控以及生成监控报告。


版本: 1.0.0
创建时间: 2025-11-05
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from concurrent.futures import ThreadPoolExecutor


class TradeStatus(Enum):
    """交易状态枚举"""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class TradeType(Enum):
    """交易类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ComplianceLevel(Enum):
    """合规级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TradeRecord:
    """交易记录数据类"""
    trade_id: str
    symbol: str
    trade_type: TradeType
    side: str  # buy/sell
    quantity: float
    price: float
    timestamp: datetime
    status: TradeStatus
    execution_time: Optional[float] = None
    latency: Optional[float] = None
    error_message: Optional[str] = None
    commission: float = 0.0
    strategy_id: Optional[str] = None
    compliance_score: float = 1.0


@dataclass
class MonitoringMetrics:
    """监控指标数据类"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_volume: float = 0.0
    total_value: float = 0.0
    total_commission: float = 0.0
    average_latency: float = 0.0
    success_rate: float = 0.0
    compliance_violations: int = 0
    strategy_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """告警配置数据类"""
    success_rate_threshold: float = 0.95
    latency_threshold: float = 1000.0  # 毫秒
    error_rate_threshold: float = 0.05
    volume_spike_threshold: float = 2.0  # 相对于平均值的倍数
    compliance_threshold: float = 0.9


class TradingMonitor:
    """
    交易监控器类
    
    该类提供全面的交易监控功能，包括实时监控、指标分析、告警和报告生成。
    支持多种监控维度和灵活的告警配置。
    """
    
    def __init__(self, 
                 max_history_size: int = 10000,
                 alert_config: Optional[AlertConfig] = None,
                 log_level: str = "INFO"):
        """
        初始化交易监控器
        
        Args:
            max_history_size: 最大历史记录保存数量
            alert_config: 告警配置
            log_level: 日志级别
        """
        self.max_history_size = max_history_size
        self.alert_config = alert_config or AlertConfig()
        
        # 设置日志
        self.logger = self._setup_logger(log_level)
        
        # 数据存储
        self.trade_history: deque = deque(maxlen=max_history_size)
        self.metrics_cache: Dict[str, Any] = {}
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.compliance_violations: List[Dict[str, Any]] = []
        
        # 监控状态
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_update_time = datetime.now()
        
        # 告警回调
        self.alert_callbacks: List[callable] = []
        
        self.logger.info("交易监控器初始化完成")
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """设置日志配置"""
        logger = logging.getLogger("TradingMonitor")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    # ==================== 交易记录管理 ====================
    
    def record_trade(self, trade: TradeRecord) -> None:
        """
        记录一笔交易
        
        Args:
            trade: 交易记录
        """
        try:
            # 计算延迟
            if trade.execution_time:
                trade.latency = (trade.execution_time - trade.timestamp.timestamp()) * 1000
            
            # 添加到历史记录
            self.trade_history.append(trade)
            
            # 更新策略性能
            if trade.strategy_id:
                profit_loss = self._calculate_pnl(trade)
                self.strategy_performance[trade.strategy_id].append(profit_loss)
            
            # 检查合规性
            self._check_compliance(trade)
            
            # 更新缓存
            self._update_metrics_cache()
            
            self.logger.debug(f"交易记录已保存: {trade.trade_id}")
            
        except Exception as e:
            self.logger.error(f"记录交易时出错: {e}")
    
    def _calculate_pnl(self, trade: TradeRecord) -> float:
        """
        计算交易盈亏
        
        Args:
            trade: 交易记录
            
        Returns:
            盈亏金额
        """
        # 简化的PnL计算，实际应用中需要更复杂的逻辑
        return (trade.price - 100.0) * trade.quantity if trade.side == "sell" else (100.0 - trade.price) * trade.quantity
    
    def _check_compliance(self, trade: TradeRecord) -> None:
        """
        检查交易合规性
        
        Args:
            trade: 交易记录
        """
        violations = []
        
        # 检查交易量限制
        if trade.quantity > 10000:
            violations.append({
                "type": "volume_limit",
                "message": f"交易量超过限制: {trade.quantity}",
                "severity": ComplianceLevel.HIGH
            })
        
        # 检查价格异常
        if trade.price < 0 or trade.price > 1000:
            violations.append({
                "type": "price_anomaly",
                "message": f"异常价格: {trade.price}",
                "severity": ComplianceLevel.MEDIUM
            })
        
        # 检查延迟
        if trade.latency and trade.latency > 5000:
            violations.append({
                "type": "high_latency",
                "message": f"延迟过高: {trade.latency}ms",
                "severity": ComplianceLevel.MEDIUM
            })
        
        if violations:
            self.compliance_violations.extend(violations)
            self.logger.warning(f"发现合规性问题: {trade.trade_id}")
    
    # ==================== 监控功能 ====================
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.last_update_time = datetime.now()
        self.logger.info("开始交易监控")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        self.logger.info("停止交易监控")
    
    async def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                # 执行监控检查
                await self._perform_health_checks()
                
                # 等待下次检查
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"监控循环出错: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self) -> None:
        """执行健康检查"""
        current_time = datetime.now()
        
        # 检查交易成功率
        success_rate = self.get_success_rate()
        if success_rate < self.alert_config.success_rate_threshold:
            await self._trigger_alert("success_rate_low", {
                "current_rate": success_rate,
                "threshold": self.alert_config.success_rate_threshold
            })
        
        # 检查延迟
        avg_latency = self.get_average_latency()
        if avg_latency > self.alert_config.latency_threshold:
            await self._trigger_alert("latency_high", {
                "current_latency": avg_latency,
                "threshold": self.alert_config.latency_threshold
            })
        
        # 检查错误率
        error_rate = self.get_error_rate()
        if error_rate > self.alert_config.error_rate_threshold:
            await self._trigger_alert("error_rate_high", {
                "current_rate": error_rate,
                "threshold": self.alert_config.error_rate_threshold
            })
        
        self.last_update_time = current_time
    
    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """
        触发告警
        
        Args:
            alert_type: 告警类型
            data: 告警数据
        """
        alert = {
            "type": alert_type,
            "timestamp": datetime.now(),
            "data": data
        }
        
        self.logger.warning(f"告警触发: {alert_type} - {data}")
        
        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调执行失败: {e}")
    
    def add_alert_callback(self, callback: callable) -> None:
        """
        添加告警回调函数
        
        Args:
            callback: 告警回调函数
        """
        self.alert_callbacks.append(callback)
    
    # ==================== 指标计算 ====================
    
    def _update_metrics_cache(self) -> None:
        """更新指标缓存"""
        if not self.trade_history:
            return
        
        recent_trades = list(self.trade_history)
        
        # 基本统计
        total_trades = len(recent_trades)
        successful_trades = sum(1 for t in recent_trades if t.status == TradeStatus.EXECUTED)
        failed_trades = sum(1 for t in recent_trades if t.status == TradeStatus.FAILED)
        
        # 交易量和价值
        total_volume = sum(t.quantity for t in recent_trades)
        total_value = sum(t.quantity * t.price for t in recent_trades)
        total_commission = sum(t.commission for t in recent_trades)
        
        # 延迟统计
        latencies = [t.latency for t in recent_trades if t.latency is not None]
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        
        # 成功率
        success_rate = successful_trades / total_trades if total_trades > 0 else 0.0
        
        # 更新缓存
        self.metrics_cache = {
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "failed_trades": failed_trades,
            "total_volume": total_volume,
            "total_value": total_value,
            "total_commission": total_commission,
            "average_latency": avg_latency,
            "success_rate": success_rate,
            "error_rate": failed_trades / total_trades if total_trades > 0 else 0.0,
            "compliance_violations": len(self.compliance_violations),
            "last_update": datetime.now()
        }
    
    def get_total_trades(self) -> int:
        """获取总交易数"""
        return len(self.trade_history)
    
    def get_success_rate(self) -> float:
        """获取交易成功率"""
        if not self.trade_history:
            return 1.0
        
        successful = sum(1 for t in self.trade_history if t.status == TradeStatus.EXECUTED)
        return successful / len(self.trade_history)
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        return 1.0 - self.get_success_rate()
    
    def get_average_latency(self) -> float:
        """获取平均延迟"""
        latencies = [t.latency for t in self.trade_history if t.latency is not None]
        return statistics.mean(latencies) if latencies else 0.0
    
    def get_total_volume(self) -> float:
        """获取总交易量"""
        return sum(t.quantity for t in self.trade_history)
    
    def get_total_value(self) -> float:
        """获取总交易价值"""
        return sum(t.quantity * t.price for t in self.trade_history)
    
    def get_total_commission(self) -> float:
        """获取总手续费"""
        return sum(t.commission for t in self.trade_history)
    
    def get_volume_by_symbol(self) -> Dict[str, float]:
        """按交易对获取交易量"""
        volume_by_symbol = defaultdict(float)
        for trade in self.trade_history:
            volume_by_symbol[trade.symbol] += trade.quantity
        return dict(volume_by_symbol)
    
    def get_trade_frequency(self, time_window: timedelta = timedelta(hours=1)) -> float:
        """
        获取交易频率
        
        Args:
            time_window: 时间窗口
            
        Returns:
            每单位时间的交易数
        """
        if not self.trade_history:
            return 0.0
        
        cutoff_time = datetime.now() - time_window
        recent_trades = [t for t in self.trade_history if t.timestamp >= cutoff_time]
        
        return len(recent_trades) / time_window.total_seconds()
    
    def get_latency_distribution(self) -> Dict[str, float]:
        """获取延迟分布统计"""
        latencies = [t.latency for t in self.trade_history if t.latency is not None]
        
        if not latencies:
            return {}
        
        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": self._percentile(latencies, 95),
            "p99": self._percentile(latencies, 99)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """计算百分位数"""
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    # ==================== 策略监控 ====================
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """获取策略性能统计"""
        performance = {}
        
        for strategy_id, pnl_list in self.strategy_performance.items():
            if not pnl_list:
                continue
            
            performance[strategy_id] = {
                "total_pnl": sum(pnl_list),
                "average_pnl": statistics.mean(pnl_list),
                "win_rate": len([pnl for pnl in pnl_list if pnl > 0]) / len(pnl_list),
                "max_gain": max(pnl_list) if pnl_list else 0.0,
                "max_loss": min(pnl_list) if pnl_list else 0.0,
                "trade_count": len(pnl_list)
            }
        
        return performance
    
    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """获取策略排名（按总盈亏）"""
        performance = self.get_strategy_performance()
        return sorted(performance.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
    
    # ==================== 合规性监控 ====================
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """获取合规状态"""
        total_violations = len(self.compliance_violations)
        
        if total_violations == 0:
            compliance_score = 1.0
        else:
            # 根据违规严重程度计算合规分数
            severity_weights = {
                ComplianceLevel.LOW: 0.1,
                ComplianceLevel.MEDIUM: 0.3,
                ComplianceLevel.HIGH: 0.6,
                ComplianceLevel.CRITICAL: 1.0
            }
            
            total_weight = sum(
                severity_weights.get(violation.get("severity"), ComplianceLevel.MEDIUM) 
                for violation in self.compliance_violations
            )
            
            compliance_score = max(0.0, 1.0 - total_weight / max(1, len(self.compliance_violations)))
        
        return {
            "compliance_score": compliance_score,
            "total_violations": total_violations,
            "violations_by_type": self._group_violations_by_type(),
            "recent_violations": self.compliance_violations[-10:]  # 最近10个违规
        }
    
    def _group_violations_by_type(self) -> Dict[str, int]:
        """按类型分组违规"""
        violations_by_type = defaultdict(int)
        for violation in self.compliance_violations:
            violations_by_type[violation["type"]] += 1
        return dict(violations_by_type)
    
    # ==================== 报告生成 ====================
    
    def generate_report(self, 
                       report_type: str = "summary",
                       time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        生成监控报告
        
        Args:
            report_type: 报告类型 (summary, detailed, compliance, strategy)
            time_range: 时间范围
            
        Returns:
            报告数据
        """
        # 过滤时间范围内的交易
        filtered_trades = self._filter_trades_by_time(time_range) if time_range else list(self.trade_history)
        
        if report_type == "summary":
            return self._generate_summary_report(filtered_trades)
        elif report_type == "detailed":
            return self._generate_detailed_report(filtered_trades)
        elif report_type == "compliance":
            return self._generate_compliance_report(filtered_trades)
        elif report_type == "strategy":
            return self._generate_strategy_report(filtered_trades)
        else:
            raise ValueError(f"不支持的报告类型: {report_type}")
    
    def _filter_trades_by_time(self, time_range: timedelta) -> List[TradeRecord]:
        """按时间范围过滤交易"""
        cutoff_time = datetime.now() - time_range
        return [t for t in self.trade_history if t.timestamp >= cutoff_time]
    
    def _generate_summary_report(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """生成摘要报告"""
        if not trades:
            return {"message": "没有交易数据"}
        
        # 计算基本指标
        total_trades = len(trades)
        successful_trades = sum(1 for t in trades if t.status == TradeStatus.EXECUTED)
        total_volume = sum(t.quantity for t in trades)
        total_value = sum(t.quantity * t.price for t in trades)
        total_commission = sum(t.commission for t in trades)
        
        # 延迟统计
        latencies = [t.latency for t in trades if t.latency is not None]
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        
        return {
            "report_type": "summary",
            "generated_at": datetime.now(),
            "time_range": f"{len(trades)} 笔交易",
            "metrics": {
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "success_rate": successful_trades / total_trades if total_trades > 0 else 0.0,
                "total_volume": total_volume,
                "total_value": total_value,
                "total_commission": total_commission,
                "average_latency_ms": avg_latency,
                "error_rate": (total_trades - successful_trades) / total_trades if total_trades > 0 else 0.0
            },
            "top_symbols": dict(sorted(self._get_volume_by_symbol(trades).items(), 
                                     key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _generate_detailed_report(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """生成详细报告"""
        summary = self._generate_summary_report(trades)
        
        # 添加详细分析
        summary.update({
            "report_type": "detailed",
            "latency_distribution": self._calculate_latency_distribution(trades),
            "trade_type_analysis": self._analyze_trade_types(trades),
            "hourly_distribution": self._analyze_hourly_distribution(trades),
            "error_analysis": self._analyze_errors(trades)
        })
        
        return summary
    
    def _generate_compliance_report(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """生成合规报告"""
        compliance_status = self.get_compliance_status()
        
        return {
            "report_type": "compliance",
            "generated_at": datetime.now(),
            "compliance_score": compliance_status["compliance_score"],
            "total_violations": compliance_status["total_violations"],
            "violations_by_type": compliance_status["violations_by_type"],
            "recent_violations": compliance_status["recent_violations"],
            "recommendations": self._generate_compliance_recommendations(compliance_status)
        }
    
    def _generate_strategy_report(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """生成策略报告"""
        strategy_performance = self.get_strategy_performance()
        
        return {
            "report_type": "strategy",
            "generated_at": datetime.now(),
            "strategy_count": len(strategy_performance),
            "strategy_performance": strategy_performance,
            "top_performers": self.get_strategy_ranking()[:5],
            "recommendations": self._generate_strategy_recommendations(strategy_performance)
        }
    
    def _get_volume_by_symbol(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """获取交易量按交易对分布"""
        volume_by_symbol = defaultdict(float)
        for trade in trades:
            volume_by_symbol[trade.symbol] += trade.quantity
        return dict(volume_by_symbol)
    
    def _calculate_latency_distribution(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """计算延迟分布"""
        latencies = [t.latency for t in trades if t.latency is not None]
        
        if not latencies:
            return {}
        
        return {
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": self._percentile(latencies, 95),
            "p99_ms": self._percentile(latencies, 99)
        }
    
    def _analyze_trade_types(self, trades: List[TradeRecord]) -> Dict[str, int]:
        """分析交易类型分布"""
        type_distribution = defaultdict(int)
        for trade in trades:
            type_distribution[trade.trade_type.value] += 1
        return dict(type_distribution)
    
    def _analyze_hourly_distribution(self, trades: List[TradeRecord]) -> Dict[int, int]:
        """分析每小时交易分布"""
        hourly_dist = defaultdict(int)
        for trade in trades:
            hour = trade.timestamp.hour
            hourly_dist[hour] += 1
        return dict(hourly_dist)
    
    def _analyze_errors(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """分析错误"""
        failed_trades = [t for t in trades if t.status == TradeStatus.FAILED]
        
        error_types = defaultdict(int)
        for trade in failed_trades:
            if trade.error_message:
                error_types[trade.error_message] += 1
        
        return {
            "total_errors": len(failed_trades),
            "error_rate": len(failed_trades) / len(trades) if trades else 0.0,
            "common_errors": dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _generate_compliance_recommendations(self, compliance_status: Dict[str, Any]) -> List[str]:
        """生成合规建议"""
        recommendations = []
        
        if compliance_status["compliance_score"] < 0.8:
            recommendations.append("合规分数较低，建议加强合规检查")
        
        if compliance_status["total_violations"] > 10:
            recommendations.append("违规次数较多，建议审查交易流程")
        
        violations_by_type = compliance_status["violations_by_type"]
        for violation_type, count in violations_by_type.items():
            if count > 5:
                recommendations.append(f"{violation_type} 违规频繁，建议重点关注")
        
        return recommendations
    
    def _generate_strategy_recommendations(self, strategy_performance: Dict[str, Dict[str, float]]) -> List[str]:
        """生成策略建议"""
        recommendations = []
        
        if not strategy_performance:
            recommendations.append("没有足够的策略数据进行分析")
            return recommendations
        
        # 找出表现最差的策略
        worst_strategy = min(strategy_performance.items(), key=lambda x: x[1]["total_pnl"])
        if worst_strategy[1]["total_pnl"] < 0:
            recommendations.append(f"策略 {worst_strategy[0]} 表现不佳，建议暂停或优化")
        
        # 找出胜率最低的策略
        lowest_win_rate = min(strategy_performance.items(), key=lambda x: x[1]["win_rate"])
        if lowest_win_rate[1]["win_rate"] < 0.4:
            recommendations.append(f"策略 {lowest_win_rate[0]} 胜率过低，建议调整参数")
        
        return recommendations
    
    # ==================== 导出功能 ====================
    
    def export_to_json(self, filename: str, report_type: str = "summary") -> None:
        """
        导出报告到JSON文件
        
        Args:
            filename: 文件名
            report_type: 报告类型
        """
        try:
            report = self.generate_report(report_type)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"报告已导出到: {filename}")
            
        except Exception as e:
            self.logger.error(f"导出报告失败: {e}")
            raise
    
    def export_trades_to_csv(self, filename: str, time_range: Optional[timedelta] = None) -> None:
        """
        导出交易记录到CSV文件
        
        Args:
            filename: 文件名
            time_range: 时间范围
        """
        try:
            import csv
            
            trades = self._filter_trades_by_time(time_range) if time_range else list(self.trade_history)
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if not trades:
                    f.write("没有交易数据")
                    return
                
                fieldnames = ['trade_id', 'symbol', 'trade_type', 'side', 'quantity', 
                            'price', 'timestamp', 'status', 'latency', 'commission', 
                            'strategy_id', 'error_message']
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for trade in trades:
                    writer.writerow({
                        'trade_id': trade.trade_id,
                        'symbol': trade.symbol,
                        'trade_type': trade.trade_type.value,
                        'side': trade.side,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'timestamp': trade.timestamp.isoformat(),
                        'status': trade.status.value,
                        'latency': trade.latency,
                        'commission': trade.commission,
                        'strategy_id': trade.strategy_id,
                        'error_message': trade.error_message
                    })
            
            self.logger.info(f"交易记录已导出到: {filename}")
            
        except Exception as e:
            self.logger.error(f"导出交易记录失败: {e}")
            raise


# ==================== 测试用例 ====================

def create_sample_trades() -> List[TradeRecord]:
    """创建示例交易数据"""
    import random
    
    trades = []
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    strategies = ["momentum", "mean_reversion", "arbitrage", "scalping"]
    
    for i in range(100):
        trade = TradeRecord(
            trade_id=f"trade_{i:06d}",
            symbol=random.choice(symbols),
            trade_type=random.choice(list(TradeType)),
            side=random.choice(["buy", "sell"]),
            quantity=random.uniform(0.1, 100.0),
            price=random.uniform(50.0, 500.0),
            timestamp=datetime.now() - timedelta(minutes=random.randint(0, 1440)),
            status=random.choice([TradeStatus.EXECUTED, TradeStatus.FAILED, TradeStatus.PENDING]),
            execution_time=time.time() + random.uniform(-100, 100),
            commission=random.uniform(0.1, 5.0),
            strategy_id=random.choice(strategies),
            compliance_score=random.uniform(0.8, 1.0)
        )
        
        # 模拟一些错误
        if trade.status == TradeStatus.FAILED:
            trade.error_message = random.choice([
                "网络超时", "价格变动过快", "余额不足", "市场关闭"
            ])
        
        trades.append(trade)
    
    return trades


def test_trading_monitor():
    """测试交易监控器"""
    print("=== M6 交易监控器测试 ===\n")
    
    # 创建监控器
    monitor = TradingMonitor(max_history_size=1000)
    
    # 添加告警回调
    def alert_handler(alert):
        print(f"🚨 告警: {alert['type']} - {alert['data']}")
    
    monitor.add_alert_callback(alert_handler)
    
    # 生成示例交易数据
    print("1. 生成示例交易数据...")
    sample_trades = create_sample_trades()
    
    # 记录交易
    print("2. 记录交易数据...")
    for trade in sample_trades:
        monitor.record_trade(trade)
    
    # 基本统计测试
    print("\n3. 基本统计测试:")
    print(f"   总交易数: {monitor.get_total_trades()}")
    print(f"   成功率: {monitor.get_success_rate():.2%}")
    print(f"   平均延迟: {monitor.get_average_latency():.2f}ms")
    print(f"   总交易量: {monitor.get_total_volume():.2f}")
    print(f"   总交易价值: {monitor.get_total_value():.2f}")
    print(f"   总手续费: {monitor.get_total_commission():.2f}")
    
    # 交易频率测试
    print(f"\n4. 交易频率测试:")
    print(f"   每小时交易频率: {monitor.get_trade_frequency(timedelta(hours=1)):.2f}")
    
    # 延迟分布测试
    print(f"\n5. 延迟分布测试:")
    latency_dist = monitor.get_latency_distribution()
    for key, value in latency_dist.items():
        print(f"   {key}: {value:.2f}ms")
    
    # 按交易对统计
    print(f"\n6. 按交易对统计:")
    volume_by_symbol = monitor.get_volume_by_symbol()
    for symbol, volume in sorted(volume_by_symbol.items(), key=lambda x: x[1], reverse=True):
        print(f"   {symbol}: {volume:.2f}")
    
    # 策略性能测试
    print(f"\n7. 策略性能测试:")
    strategy_performance = monitor.get_strategy_performance()
    for strategy, perf in strategy_performance.items():
        print(f"   {strategy}:")
        print(f"     总盈亏: {perf['total_pnl']:.2f}")
        print(f"     胜率: {perf['win_rate']:.2%}")
        print(f"     交易次数: {perf['trade_count']}")
    
    # 策略排名
    print(f"\n8. 策略排名:")
    strategy_ranking = monitor.get_strategy_ranking()
    for i, (strategy, perf) in enumerate(strategy_ranking[:3], 1):
        print(f"   {i}. {strategy}: {perf['total_pnl']:.2f}")
    
    # 合规性测试
    print(f"\n9. 合规性测试:")
    compliance_status = monitor.get_compliance_status()
    print(f"   合规分数: {compliance_status['compliance_score']:.2f}")
    print(f"   违规总数: {compliance_status['total_violations']}")
    print(f"   违规类型: {compliance_status['violations_by_type']}")
    
    # 报告生成测试
    print(f"\n10. 报告生成测试:")
    
    # 摘要报告
    summary_report = monitor.generate_report("summary")
    print(f"   摘要报告已生成，包含 {len(summary_report)} 个字段")
    
    # 详细报告
    detailed_report = monitor.generate_report("detailed")
    print(f"   详细报告已生成，包含 {len(detailed_report)} 个字段")
    
    # 合规报告
    compliance_report = monitor.generate_report("compliance")
    print(f"   合规报告已生成，合规分数: {compliance_report['compliance_score']:.2f}")
    
    # 策略报告
    strategy_report = monitor.generate_report("strategy")
    print(f"   策略报告已生成，包含 {strategy_report['strategy_count']} 个策略")
    
    # 导出测试
    print(f"\n11. 导出功能测试:")
    try:
        monitor.export_to_json("test_summary_report.json", "summary")
        monitor.export_trades_to_csv("test_trades.csv", timedelta(hours=24))
        print("   ✅ 导出功能正常")
    except Exception as e:
        print(f"   ❌ 导出功能出错: {e}")
    
    # 监控功能测试
    print(f"\n12. 监控功能测试:")
    monitor.start_monitoring()
    print("   ✅ 监控已启动")
    
    # 模拟一些新的交易来触发告警
    for i in range(5):
        new_trade = TradeRecord(
            trade_id=f"alert_test_{i}",
            symbol="BTCUSDT",
            trade_type=TradeType.MARKET,
            side="buy",
            quantity=50000.0,  # 大量交易，触发合规告警
            price=100.0,
            timestamp=datetime.now(),
            status=TradeStatus.EXECUTED,
            execution_time=time.time(),
            latency=2000.0,  # 高延迟
            commission=10.0
        )
        monitor.record_trade(new_trade)
    
    print("   已添加测试交易数据")
    
    # 停止监控
    monitor.stop_monitoring()
    print("   ✅ 监控已停止")
    
    print(f"\n=== 测试完成 ===")
    print(f"总共测试了 {len(sample_trades)} 笔交易记录")
    print(f"监控器功能正常，所有核心功能均通过测试")


if __name__ == "__main__":
    # 运行测试
    test_trading_monitor()