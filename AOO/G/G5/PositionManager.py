#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G5仓位管理器
实现智能仓位管理和优化的核心功能模块

功能模块：
1. 仓位计算和优化
2. 仓位监控和调整
3. 仓位风险评估
4. 仓位组合优化
5. 仓位历史跟踪
6. 仓位报告和分析
7. 仓位策略优化
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """仓位数据结构"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    weight: float
    timestamp: datetime
    risk_level: str
    sector: str = ""
    
    def __post_init__(self):
        """初始化后处理"""
        if self.current_price > 0:
            self.market_value = self.quantity * self.current_price
            self.unrealized_pnl = (self.current_price - self.avg_price) * self.quantity
            self.weight = 0  # 将在组合计算中设置

@dataclass
class RiskMetrics:
    """风险指标数据结构"""
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    concentration_risk: float
    sector_concentration: Dict[str, float]

@dataclass
class PositionAdjustment:
    """仓位调整建议数据结构"""
    symbol: str
    current_weight: float
    target_weight: float
    adjustment_amount: float
    action: str  # "BUY", "SELL", "HOLD"
    priority: str  # "HIGH", "MEDIUM", "LOW"
    reason: str
    timestamp: datetime

class PositionOptimizer:
    """仓位优化算法类"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 无风险利率
        
    def mean_variance_optimization(self, returns: np.ndarray, target_return: float = None) -> np.ndarray:
        """
        均值方差优化
        
        Args:
            returns: 收益率矩阵 (n_assets, n_periods)
            target_return: 目标收益率
            
        Returns:
            优化后的权重向量
        """
        n_assets = returns.shape[0]
        
        # 计算期望收益率和协方差矩阵
        expected_returns = np.mean(returns, axis=1)
        cov_matrix = np.cov(returns)
        
        # 添加正则化项以确保协方差矩阵可逆
        cov_matrix += np.eye(n_assets) * 1e-8
        
        try:
            if target_return is None:
                # 等权重基准
                weights = np.ones(n_assets) / n_assets
            else:
                # 目标收益率优化
                inv_cov = np.linalg.inv(cov_matrix)
                ones = np.ones(n_assets)
                
                # 计算最优权重
                A = ones.T @ inv_cov @ ones
                B = ones.T @ inv_cov @ expected_returns
                C = expected_returns.T @ inv_cov @ expected_returns
                
                # 拉格朗日乘数法求解
                if A * C - B * B > 1e-10:
                    g = (C - B * target_return) / (A * C - B * B)
                    h = (A * target_return - B) / (A * C - B * B)
                    weights = g * inv_cov @ ones + h * inv_cov @ expected_returns
                else:
                    weights = np.ones(n_assets) / n_assets
                    
        except np.linalg.LinAlgError:
            logger.warning("协方差矩阵奇异，使用等权重")
            weights = np.ones(n_assets) / n_assets
            
        # 确保权重非负且和为1
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        return weights
    
    def risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        风险平价优化
        
        Args:
            cov_matrix: 协方差矩阵
            
        Returns:
            风险平价权重向量
        """
        n_assets = cov_matrix.shape[0]
        
        # 初始化权重
        weights = np.ones(n_assets) / n_assets
        
        # 迭代优化
        for _ in range(100):
            # 计算边际风险贡献
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            
            # 计算风险贡献
            risk_contrib = weights * marginal_contrib
            
            # 计算目标风险贡献
            target_risk = portfolio_vol / n_assets
            
            # 更新权重
            diff = risk_contrib - target_risk
            if np.linalg.norm(diff) < 1e-6:
                break
                
            # 梯度下降更新
            weights = weights - 0.01 * diff
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
        return weights
    
    def black_litterman_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, 
                                   market_weights: np.ndarray, tau: float = 0.05) -> np.ndarray:
        """
        Black-Litterman模型优化
        
        Args:
            expected_returns: 期望收益率
            cov_matrix: 协方差矩阵
            market_weights: 市场权重
            tau: 不确定性参数
            
        Returns:
            Black-Litterman权重向量
        """
        n_assets = len(expected_returns)
        
        # 假设投资者观点
        P = np.eye(n_assets)  # 观点矩阵
        Q = expected_returns  # 观点收益率
        
        # 观点不确定性
        Omega = np.diag(np.diag(P @ cov_matrix @ P.T) * tau)
        
        # Black-Litterman公式
        M1 = np.linalg.inv(tau * cov_matrix)
        M2 = P.T @ np.linalg.inv(Omega) @ P
        M3 = P.T @ np.linalg.inv(Omega) @ Q
        
        # 计算后验估计
        mu_bl = np.linalg.inv(M1 + M2) @ (M1 @ expected_returns + M3)
        cov_bl = np.linalg.inv(M1 + M2)
        
        # 最优权重（假设风险厌恶系数为1）
        risk_aversion = 1.0
        w_bl = risk_aversion * np.linalg.inv(cov_bl) @ mu_bl
        
        return w_bl

class PositionManager:
    """G5仓位管理器主类"""
    
    def __init__(self, initial_capital: float = 1000000.0):
        """
        初始化仓位管理器
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict] = []
        self.optimization_history: List[Dict] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # 优化器
        self.optimizer = PositionOptimizer()
        
        # 配置参数
        self.max_position_weight = 0.1  # 最大单仓位权重
        self.min_position_weight = 0.01  # 最小单仓位权重
        self.max_sector_weight = 0.3  # 最大行业权重
        self.rebalance_threshold = 0.05  # 再平衡阈值
        self.risk_free_rate = 0.02
        
        # 监控参数
        self.var_threshold = 0.05  # VaR阈值
        self.max_drawdown_threshold = 0.2  # 最大回撤阈值
        self.concentration_threshold = 0.4  # 集中度阈值
        
        logger.info(f"仓位管理器初始化完成，初始资金: {initial_capital:,.2f}")
    
    def add_position(self, symbol: str, quantity: float, price: float, 
                    sector: str = "", timestamp: datetime = None) -> Position:
        """
        添加仓位
        
        Args:
            symbol: 标的代码
            quantity: 数量
            price: 价格
            sector: 行业
            timestamp: 时间戳
            
        Returns:
            Position对象
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=price,
            current_price=price,
            market_value=quantity * price,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            weight=0.0,
            timestamp=timestamp,
            risk_level="MEDIUM",
            sector=sector
        )
        
        self.positions[symbol] = position
        self._update_portfolio_weights()
        
        logger.info(f"添加仓位: {symbol}, 数量: {quantity}, 价格: {price}")
        return position
    
    def update_position_price(self, symbol: str, new_price: float) -> bool:
        """
        更新仓位价格
        
        Args:
            symbol: 标的代码
            new_price: 新价格
            
        Returns:
            是否更新成功
        """
        if symbol not in self.positions:
            logger.warning(f"未找到仓位: {symbol}")
            return False
            
        position = self.positions[symbol]
        old_price = position.current_price
        position.current_price = new_price
        position.market_value = position.quantity * new_price
        position.unrealized_pnl = (new_price - position.avg_price) * position.quantity
        
        # 记录历史
        self._record_position_change(symbol, old_price, new_price)
        
        logger.debug(f"更新价格: {symbol}, {old_price:.4f} -> {new_price:.4f}")
        return True
    
    def calculate_position_metrics(self) -> Dict[str, Any]:
        """
        计算仓位指标
        
        Returns:
            仓位指标字典
        """
        if not self.positions:
            return {}
            
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        
        metrics = {
            "total_positions": len(self.positions),
            "total_market_value": total_market_value,
            "total_unrealized_pnl": sum(pos.unrealized_pnl for pos in self.positions.values()),
            "total_realized_pnl": sum(pos.realized_pnl for pos in self.positions.values()),
            "cash_position": self.current_capital - total_market_value,
            "portfolio_return": (total_market_value + sum(pos.realized_pnl for pos in self.positions.values()) - self.initial_capital) / self.initial_capital,
            "position_weights": {symbol: pos.weight for symbol, pos in self.positions.items()},
            "sector_allocation": self._calculate_sector_allocation(),
            "risk_distribution": self._calculate_risk_distribution()
        }
        
        return metrics
    
    def _calculate_sector_allocation(self) -> Dict[str, float]:
        """计算行业配置"""
        sector_values = defaultdict(float)
        total_value = sum(pos.market_value for pos in self.positions.values())
        
        for position in self.positions.values():
            sector = position.sector if position.sector else "未分类"
            sector_values[sector] += position.market_value
            
        if total_value > 0:
            return {sector: value/total_value for sector, value in sector_values.items()}
        else:
            return {}
    
    def _calculate_risk_distribution(self) -> Dict[str, int]:
        """计算风险分布"""
        risk_dist = defaultdict(int)
        for position in self.positions.values():
            risk_dist[position.risk_level] += 1
        return dict(risk_dist)
    
    def _update_portfolio_weights(self):
        """更新组合权重"""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        
        if total_market_value > 0:
            for position in self.positions.values():
                position.weight = position.market_value / total_market_value
    
    def _record_position_change(self, symbol: str, old_price: float, new_price: float):
        """记录仓位变化"""
        record = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "old_price": old_price,
            "new_price": new_price,
            "change_percent": (new_price - old_price) / old_price * 100
        }
        self.position_history.append(record)
        
        # 保持历史记录在合理范围内
        if len(self.position_history) > 10000:
            self.position_history = self.position_history[-5000:]
    
    def optimize_portfolio(self, method: str = "mean_variance", 
                          target_return: float = None) -> Dict[str, float]:
        """
        组合优化
        
        Args:
            method: 优化方法 ("mean_variance", "risk_parity", "black_litterman")
            target_return: 目标收益率
            
        Returns:
            优化后的权重字典
        """
        if not self.positions:
            logger.warning("没有仓位可优化")
            return {}
            
        symbols = list(self.positions.keys())
        n_assets = len(symbols)
        
        # 模拟历史收益率数据（实际应用中应从数据源获取）
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, 252))  # 252个交易日
        
        # 计算协方差矩阵
        cov_matrix = np.cov(returns)
        
        # 根据方法进行优化
        if method == "mean_variance":
            weights = self.optimizer.mean_variance_optimization(returns, target_return)
        elif method == "risk_parity":
            weights = self.optimizer.risk_parity_optimization(cov_matrix)
        elif method == "black_litterman":
            expected_returns = np.mean(returns, axis=1)
            market_weights = np.ones(n_assets) / n_assets
            raw_weights = self.optimizer.black_litterman_optimization(
                expected_returns, cov_matrix, market_weights)
            # 转换为非负权重
            weights = np.maximum(raw_weights, 0)
            weights = weights / np.sum(weights)
        else:
            logger.warning(f"未知优化方法: {method}，使用等权重")
            weights = np.ones(n_assets) / n_assets
        
        # 应用约束条件
        weights = self._apply_constraints(weights)
        
        # 生成优化结果
        optimization_result = {
            "method": method,
            "timestamp": datetime.now(),
            "target_return": target_return,
            "weights": {symbols[i]: weights[i] for i in range(n_assets)},
            "expected_return": np.sum(weights * np.mean(returns, axis=1)),
            "expected_volatility": np.sqrt(weights.T @ cov_matrix @ weights)
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"组合优化完成，方法: {method}")
        return optimization_result["weights"]
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """应用约束条件"""
        # 单资产权重约束
        weights = np.maximum(weights, self.min_position_weight)
        
        # 归一化
        weights = weights / np.sum(weights)
        
        # 最大权重约束
        max_weight_indices = weights > self.max_position_weight
        if np.any(max_weight_indices):
            weights[max_weight_indices] = self.max_position_weight
            remaining_weight = 1.0 - np.sum(weights[max_weight_indices])
            non_max_indices = ~max_weight_indices
            if np.any(non_max_indices) and remaining_weight > 0:
                weights[non_max_indices] *= remaining_weight / np.sum(weights[non_max_indices])
        
        return weights
    
    def calculate_risk_metrics(self, returns: np.ndarray = None) -> RiskMetrics:
        """
        计算风险指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            RiskMetrics对象
        """
        if returns is None:
            # 使用历史数据模拟收益率
            returns = np.random.normal(0, 0.02, 252)
        
        # 计算基本指标
        portfolio_return = np.mean(returns)
        portfolio_volatility = np.std(returns)
        
        # 夏普比率
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # VaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Beta和Alpha（假设市场收益率为0）
        market_returns = np.random.normal(0, 0.015, len(returns))
        beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns) if np.var(market_returns) > 0 else 1
        alpha = portfolio_return - beta * np.mean(market_returns)
        
        # 信息比率
        excess_returns = returns - self.risk_free_rate / 252
        tracking_error = np.std(excess_returns)
        information_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
        
        # 集中度风险
        if self.positions:
            weights = np.array([pos.weight for pos in self.positions.values()])
            concentration_risk = np.sum(weights ** 2)
        else:
            concentration_risk = 0
        
        # 行业集中度
        sector_concentration = self._calculate_sector_allocation()
        
        risk_metrics = RiskMetrics(
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            concentration_risk=concentration_risk,
            sector_concentration=sector_concentration
        )
        
        self.risk_metrics_history.append(risk_metrics)
        
        return risk_metrics
    
    def generate_rebalance_recommendations(self) -> List[PositionAdjustment]:
        """
        生成再平衡建议
        
        Returns:
            仓位调整建议列表
        """
        recommendations = []
        
        if not self.positions:
            return recommendations
        
        # 获取最新优化结果
        if not self.optimization_history:
            logger.info("没有优化历史，使用等权重作为目标权重")
            n_positions = len(self.positions)
            target_weights = {symbol: 1.0/n_positions for symbol in self.positions.keys()}
        else:
            latest_optimization = self.optimization_history[-1]
            target_weights = latest_optimization["weights"]
        
        current_weights = {symbol: pos.weight for symbol, pos in self.positions.items()}
        
        for symbol in self.positions.keys():
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            
            weight_diff = target_weight - current_weight
            
            # 判断是否需要调整
            if abs(weight_diff) > self.rebalance_threshold:
                action = "BUY" if weight_diff > 0 else "SELL"
                priority = "HIGH" if abs(weight_diff) > 0.1 else "MEDIUM"
                
                recommendation = PositionAdjustment(
                    symbol=symbol,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    adjustment_amount=weight_diff,
                    action=action,
                    priority=priority,
                    reason=f"权重偏离目标 {weight_diff:.2%}",
                    timestamp=datetime.now()
                )
                
                recommendations.append(recommendation)
        
        # 按优先级排序
        recommendations.sort(key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[x.priority], reverse=True)
        
        logger.info(f"生成 {len(recommendations)} 条再平衡建议")
        return recommendations
    
    def monitor_position_risks(self) -> Dict[str, Any]:
        """
        监控仓位风险
        
        Returns:
            风险监控报告
        """
        risk_report = {
            "timestamp": datetime.now(),
            "alerts": [],
            "warnings": [],
            "recommendations": []
        }
        
        if not self.positions:
            return risk_report
        
        # 检查单资产集中度
        for symbol, position in self.positions.items():
            if position.weight > self.max_position_weight:
                risk_report["alerts"].append({
                    "type": "CONCENTRATION_RISK",
                    "symbol": symbol,
                    "message": f"单资产权重超限: {position.weight:.2%}",
                    "severity": "HIGH"
                })
        
        # 检查行业集中度
        sector_allocation = self._calculate_sector_allocation()
        for sector, weight in sector_allocation.items():
            if weight > self.max_sector_weight:
                risk_report["warnings"].append({
                    "type": "SECTOR_CONCENTRATION",
                    "sector": sector,
                    "message": f"行业权重超限: {weight:.2%}",
                    "severity": "MEDIUM"
                })
        
        # 检查VaR
        risk_metrics = self.calculate_risk_metrics()
        if risk_metrics.var_95 < -self.var_threshold:
            risk_report["alerts"].append({
                "type": "VAR_RISK",
                "message": f"95% VaR超限: {risk_metrics.var_95:.2%}",
                "severity": "HIGH"
            })
        
        # 检查最大回撤
        if risk_metrics.max_drawdown < -self.max_drawdown_threshold:
            risk_report["alerts"].append({
                "type": "DRAWDOWN_RISK",
                "message": f"最大回撤超限: {risk_metrics.max_drawdown:.2%}",
                "severity": "HIGH"
            })
        
        # 生成建议
        recommendations = self.generate_rebalance_recommendations()
        risk_report["recommendations"] = [asdict(rec) for rec in recommendations[:5]]  # 只返回前5条
        
        logger.info(f"风险监控完成，发现 {len(risk_report['alerts'])} 个警报，{len(risk_report['warnings'])} 个警告")
        return risk_report
    
    def generate_performance_report(self, period_days: int = 30) -> Dict[str, Any]:
        """
        生成业绩报告
        
        Args:
            period_days: 报告期间（天数）
            
        Returns:
            业绩报告字典
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # 过滤历史数据
        period_history = [
            record for record in self.position_history 
            if record["timestamp"] >= start_date
        ]
        
        # 计算期间指标
        if period_history:
            returns = [record["change_percent"] / 100 for record in period_history]
            period_return = np.mean(returns) * len(returns) if returns else 0
            period_volatility = np.std(returns) if len(returns) > 1 else 0
        else:
            period_return = 0
            period_volatility = 0
        
        # 当前仓位指标
        current_metrics = self.calculate_position_metrics()
        
        # 风险指标
        risk_metrics = self.calculate_risk_metrics()
        
        # 生成报告
        report = {
            "report_period": {
                "start_date": start_date,
                "end_date": end_date,
                "days": period_days
            },
            "performance_summary": {
                "period_return": period_return,
                "period_volatility": period_volatility,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "max_drawdown": risk_metrics.max_drawdown,
                "total_return": current_metrics.get("portfolio_return", 0)
            },
            "position_summary": {
                "total_positions": current_metrics.get("total_positions", 0),
                "total_market_value": current_metrics.get("total_market_value", 0),
                "cash_position": current_metrics.get("cash_position", 0),
                "unrealized_pnl": current_metrics.get("total_unrealized_pnl", 0),
                "realized_pnl": current_metrics.get("total_realized_pnl", 0)
            },
            "risk_summary": {
                "portfolio_volatility": risk_metrics.portfolio_volatility,
                "var_95": risk_metrics.var_95,
                "var_99": risk_metrics.var_99,
                "concentration_risk": risk_metrics.concentration_risk,
                "beta": risk_metrics.beta,
                "alpha": risk_metrics.alpha
            },
            "allocation_summary": {
                "sector_allocation": current_metrics.get("sector_allocation", {}),
                "top_holdings": self._get_top_holdings(10),
                "risk_distribution": current_metrics.get("risk_distribution", {})
            },
            "optimization_summary": {
                "last_optimization": self.optimization_history[-1] if self.optimization_history else None,
                "optimization_frequency": len(self.optimization_history),
                "rebalance_recommendations": len(self.generate_rebalance_recommendations())
            }
        }
        
        logger.info(f"生成 {period_days} 天业绩报告")
        return report
    
    def _get_top_holdings(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取前N大持仓"""
        if not self.positions:
            return []
        
        holdings = []
        for symbol, position in self.positions.items():
            holdings.append({
                "symbol": symbol,
                "weight": position.weight,
                "market_value": position.market_value,
                "quantity": position.quantity,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "sector": position.sector
            })
        
        # 按市值排序
        holdings.sort(key=lambda x: x["market_value"], reverse=True)
        return holdings[:n]
    
    def optimize_strategy_parameters(self) -> Dict[str, Any]:
        """
        优化策略参数
        
        Returns:
            优化结果
        """
        # 分析历史优化效果
        if len(self.optimization_history) < 2:
            return {"message": "历史数据不足，无法进行策略优化"}
        
        # 计算优化效果指标
        optimization_performance = []
        for opt in self.optimization_history[-10:]:  # 最近10次优化
            performance = {
                "method": opt["method"],
                "expected_return": opt["expected_return"],
                "expected_volatility": opt["expected_volatility"],
                "sharpe_ratio": opt["expected_return"] / opt["expected_volatility"] if opt["expected_volatility"] > 0 else 0
            }
            optimization_performance.append(performance)
        
        # 找出最佳方法
        best_method = max(optimization_performance, key=lambda x: x["sharpe_ratio"])
        
        # 分析风险指标趋势
        if len(self.risk_metrics_history) >= 5:
            recent_metrics = self.risk_metrics_history[-5:]
            volatility_trend = np.mean([m.portfolio_volatility for m in recent_metrics])
            drawdown_trend = np.mean([m.max_drawdown for m in recent_metrics])
        else:
            volatility_trend = 0
            drawdown_trend = 0
        
        # 生成优化建议
        optimization_suggestions = {
            "recommended_method": best_method["method"],
            "parameter_adjustments": {
                "max_position_weight": self.max_position_weight,
                "rebalance_threshold": self.rebalance_threshold,
                "risk_free_rate": self.risk_free_rate
            },
            "risk_management": {
                "current_volatility": volatility_trend,
                "current_drawdown": drawdown_trend,
                "suggestions": []
            }
        }
        
        # 根据风险趋势给出建议
        if volatility_trend > 0.03:  # 波动率过高
            optimization_suggestions["risk_management"]["suggestions"].append(
                "考虑降低仓位集中度，增加分散化"
            )
        
        if drawdown_trend < -0.15:  # 回撤过大
            optimization_suggestions["risk_management"]["suggestions"].append(
                "建议增加风险控制措施，设置止损线"
            )
        
        logger.info("策略参数优化完成")
        return optimization_suggestions
    
    def export_data(self, format_type: str = "json") -> str:
        """
        导出数据
        
        Args:
            format_type: 导出格式 ("json", "csv")
            
        Returns:
            导出的数据字符串
        """
        data = {
            "positions": {symbol: asdict(pos) for symbol, pos in self.positions.items()},
            "position_history": self.position_history,
            "optimization_history": self.optimization_history,
            "risk_metrics_history": [asdict(metric) for metric in self.risk_metrics_history],
            "current_metrics": self.calculate_position_metrics(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        if format_type.lower() == "json":
            return json.dumps(data, ensure_ascii=False, indent=2, default=str)
        elif format_type.lower() == "csv":
            # 转换为DataFrame并导出CSV
            positions_df = pd.DataFrame([
                asdict(pos) for pos in self.positions.values()
            ])
            return positions_df.to_csv(index=False)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        获取仪表盘数据
        
        Returns:
            仪表盘数据字典
        """
        current_metrics = self.calculate_position_metrics()
        risk_metrics = self.calculate_risk_metrics()
        risk_report = self.monitor_position_risks()
        recommendations = self.generate_rebalance_recommendations()
        
        dashboard_data = {
            "timestamp": datetime.now(),
            "portfolio_summary": {
                "total_value": current_metrics.get("total_market_value", 0) + current_metrics.get("cash_position", 0),
                "total_return": current_metrics.get("portfolio_return", 0),
                "total_pnl": current_metrics.get("total_unrealized_pnl", 0) + current_metrics.get("total_realized_pnl", 0),
                "position_count": current_metrics.get("total_positions", 0)
            },
            "risk_metrics": asdict(risk_metrics),
            "position_allocation": {
                "by_symbol": current_metrics.get("position_weights", {}),
                "by_sector": current_metrics.get("sector_allocation", {})
            },
            "alerts": {
                "high_priority": len([a for a in risk_report["alerts"] if a["severity"] == "HIGH"]),
                "medium_priority": len([a for a in risk_report["warnings"] if a["severity"] == "MEDIUM"]),
                "total_recommendations": len(recommendations)
            },
            "top_holdings": self._get_top_holdings(5),
            "recent_optimizations": self.optimization_history[-3:] if self.optimization_history else [],
            "performance_indicators": {
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "max_drawdown": risk_metrics.max_drawdown,
                "var_95": risk_metrics.var_95,
                "concentration_risk": risk_metrics.concentration_risk
            }
        }
        
        return dashboard_data

def main():
    """主函数 - 演示仓位管理器功能"""
    print("=== G5仓位管理器演示 ===")
    
    # 创建仓位管理器
    pm = PositionManager(initial_capital=1000000)
    
    # 添加测试仓位
    test_positions = [
        ("AAPL", 100, 150.0, "科技"),
        ("GOOGL", 50, 2800.0, "科技"),
        ("MSFT", 75, 300.0, "科技"),
        ("JPM", 200, 120.0, "金融"),
        ("JNJ", 150, 160.0, "医疗")
    ]
    
    for symbol, quantity, price, sector in test_positions:
        pm.add_position(symbol, quantity, price, sector)
    
    # 更新价格
    price_updates = [
        ("AAPL", 155.0),
        ("GOOGL", 2750.0),
        ("MSFT", 310.0),
        ("JPM", 125.0),
        ("JNJ", 158.0)
    ]
    
    for symbol, new_price in price_updates:
        pm.update_position_price(symbol, new_price)
    
    print("\n1. 仓位指标计算:")
    metrics = pm.calculate_position_metrics()
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:,.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n2. 组合优化:")
    weights = pm.optimize_portfolio("mean_variance")
    print("   优化后权重:")
    for symbol, weight in weights.items():
        print(f"   {symbol}: {weight:.2%}")
    
    print("\n3. 风险指标计算:")
    risk_metrics = pm.calculate_risk_metrics()
    print(f"   夏普比率: {risk_metrics.sharpe_ratio:.3f}")
    print(f"   最大回撤: {risk_metrics.max_drawdown:.2%}")
    print(f"   95% VaR: {risk_metrics.var_95:.2%}")
    print(f"   集中度风险: {risk_metrics.concentration_risk:.3f}")
    
    print("\n4. 再平衡建议:")
    recommendations = pm.generate_rebalance_recommendations()
    for rec in recommendations:
        print(f"   {rec.symbol}: {rec.action} {rec.adjustment_amount:.2%} ({rec.reason})")
    
    print("\n5. 风险监控:")
    risk_report = pm.monitor_position_risks()
    print(f"   警报数量: {len(risk_report['alerts'])}")
    print(f"   警告数量: {len(risk_report['warnings'])}")
    
    print("\n6. 业绩报告:")
    report = pm.generate_performance_report(30)
    print(f"   期间收益率: {report['performance_summary']['period_return']:.2%}")
    print(f"   夏普比率: {report['performance_summary']['sharpe_ratio']:.3f}")
    print(f"   最大回撤: {report['performance_summary']['max_drawdown']:.2%}")
    
    print("\n7. 策略优化:")
    strategy_opt = pm.optimize_strategy_parameters()
    print(f"   推荐方法: {strategy_opt.get('recommended_method', 'N/A')}")
    
    print("\n8. 仪表盘数据:")
    dashboard = pm.get_dashboard_data()
    print(f"   总资产: {dashboard['portfolio_summary']['total_value']:,.2f}")
    print(f"   总收益率: {dashboard['portfolio_summary']['total_return']:.2%}")
    print(f"   夏普比率: {dashboard['performance_indicators']['sharpe_ratio']:.3f}")
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    main()