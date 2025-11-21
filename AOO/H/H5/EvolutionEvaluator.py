#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5 进化评估器
实现进化效果评估、质量验证、适应性分析、效率优化、风险控制和历史分析等功能
"""

import json
import time
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationType(Enum):
    """评估类型枚举"""
    EFFECTIVENESS = "effectiveness"  # 效果评估
    QUALITY = "quality"             # 质量评估
    ADAPTABILITY = "adaptability"   # 适应性评估
    EFFICIENCY = "efficiency"       # 效率评估
    RISK = "risk"                   # 风险评估
    HISTORICAL = "historical"       # 历史分析

class EvolutionStage(Enum):
    """进化阶段枚举"""
    INITIALIZATION = "initialization"
    GROWTH = "growth"
    MATURATION = "maturation"
    OPTIMIZATION = "optimization"
    CONVERGENCE = "convergence"

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class EvolutionMetrics:
    """进化指标数据类"""
    timestamp: float
    stage: EvolutionStage
    effectiveness_score: float
    quality_score: float
    adaptability_score: float
    efficiency_score: float
    risk_score: float
    convergence_rate: float
    diversity_index: float
    stability_score: float
    learning_rate: float
    adaptation_speed: float
    resource_utilization: float
    error_rate: float
    success_rate: float
    metadata: Dict[str, Any]

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    evaluation_id: str
    evaluation_type: EvaluationType
    timestamp: float
    scores: Dict[str, float]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    historical_comparison: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class EvolutionReport:
    """进化报告数据类"""
    report_id: str
    generation_time: float
    overall_score: float
    stage_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    recommendations: List[str]
    risk_alerts: List[str]
    performance_metrics: Dict[str, Any]
    historical_insights: Dict[str, Any]
    export_timestamp: float

class EvolutionEvaluator:
    """H5进化评估器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化进化评估器
        
        Args:
            config: 配置参数
        """
        # 合并用户配置和默认配置
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        
        self.metrics_history = deque(maxlen=self.config['history_size'])
        self.evaluation_cache = {}
        self.is_monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # 初始化数据库
        self._init_database()
        
        # 初始化评估器组件
        self._init_evaluators()
        
        logger.info("H5进化评估器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'history_size': 10000,
            'evaluation_interval': 60,  # 秒
            'risk_threshold': 0.7,
            'convergence_threshold': 0.95,
            'diversity_threshold': 0.3,
            'efficiency_threshold': 0.8,
            'quality_threshold': 0.85,
            'adaptability_threshold': 0.75,
            'database_path': 'evolution_evaluator.db',
            'enable_real_time_monitoring': True,
            'enable_predictive_analysis': True,
            'max_workers': 4
        }
    
    def _init_database(self):
        """初始化数据库"""
        try:
            self.db_conn = sqlite3.connect(
                self.config['database_path'], 
                check_same_thread=False
            )
            
            # 创建表
            cursor = self.db_conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evolution_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    stage TEXT,
                    effectiveness_score REAL,
                    quality_score REAL,
                    adaptability_score REAL,
                    efficiency_score REAL,
                    risk_score REAL,
                    convergence_rate REAL,
                    diversity_index REAL,
                    stability_score REAL,
                    learning_rate REAL,
                    adaptation_speed REAL,
                    resource_utilization REAL,
                    error_rate REAL,
                    success_rate REAL,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT,
                    evaluation_type TEXT,
                    timestamp REAL,
                    scores TEXT,
                    recommendations TEXT,
                    risk_assessment TEXT,
                    historical_comparison TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evolution_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT,
                    generation_time REAL,
                    overall_score REAL,
                    stage_analysis TEXT,
                    trend_analysis TEXT,
                    recommendations TEXT,
                    risk_alerts TEXT,
                    performance_metrics TEXT,
                    historical_insights TEXT,
                    export_timestamp REAL
                )
            ''')
            
            self.db_conn.commit()
            logger.info("数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def _init_evaluators(self):
        """初始化评估器组件"""
        self.effectiveness_evaluator = EffectivenessEvaluator()
        self.quality_evaluator = QualityEvaluator()
        self.adaptability_evaluator = AdaptabilityEvaluator()
        self.efficiency_evaluator = EfficiencyEvaluator()
        self.risk_evaluator = RiskEvaluator()
        self.historical_analyzer = HistoricalAnalyzer()
        
        logger.info("评估器组件初始化完成")
    
    def evaluate_evolution(self, 
                          current_state: Dict[str, Any],
                          target_state: Optional[Dict[str, Any]] = None,
                          context: Optional[Dict[str, Any]] = None) -> EvolutionMetrics:
        """评估进化状态
        
        Args:
            current_state: 当前状态
            target_state: 目标状态
            context: 上下文信息
            
        Returns:
            EvolutionMetrics: 进化指标
        """
        try:
            with self.lock:
                # 并行执行各项评估
                with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                    future_effectiveness = executor.submit(
                        self.effectiveness_evaluator.evaluate, 
                        current_state, target_state, context
                    )
                    future_quality = executor.submit(
                        self.quality_evaluator.evaluate, 
                        current_state, target_state, context
                    )
                    future_adaptability = executor.submit(
                        self.adaptability_evaluator.evaluate, 
                        current_state, target_state, context
                    )
                    future_efficiency = executor.submit(
                        self.efficiency_evaluator.evaluate, 
                        current_state, target_state, context
                    )
                    future_risk = executor.submit(
                        self.risk_evaluator.evaluate, 
                        current_state, target_state, context
                    )
                    
                    effectiveness_score = future_effectiveness.result()
                    quality_score = future_quality.result()
                    adaptability_score = future_adaptability.result()
                    efficiency_score = future_efficiency.result()
                    risk_score = future_risk.result()
                
                # 计算综合指标
                convergence_rate = self._calculate_convergence_rate(current_state, target_state)
                diversity_index = self._calculate_diversity_index(current_state)
                stability_score = self._calculate_stability_score()
                learning_rate = self._calculate_learning_rate()
                adaptation_speed = self._calculate_adaptation_speed()
                resource_utilization = self._calculate_resource_utilization(current_state)
                error_rate = self._calculate_error_rate(current_state)
                success_rate = self._calculate_success_rate(current_state)
                
                # 确定当前阶段
                stage = self._determine_evolution_stage(effectiveness_score, convergence_rate)
                
                # 构建指标对象
                metrics = EvolutionMetrics(
                    timestamp=time.time(),
                    stage=stage,
                    effectiveness_score=effectiveness_score,
                    quality_score=quality_score,
                    adaptability_score=adaptability_score,
                    efficiency_score=efficiency_score,
                    risk_score=risk_score,
                    convergence_rate=convergence_rate,
                    diversity_index=diversity_index,
                    stability_score=stability_score,
                    learning_rate=learning_rate,
                    adaptation_speed=adaptation_speed,
                    resource_utilization=resource_utilization,
                    error_rate=error_rate,
                    success_rate=success_rate,
                    metadata={
                        'current_state_hash': hash(str(current_state)),
                        'target_state_hash': hash(str(target_state)) if target_state else None,
                        'context_size': len(str(context)) if context else 0
                    }
                )
                
                # 保存指标到历史记录
                self.metrics_history.append(metrics)
                self._save_metrics_to_db(metrics)
                
                # 实时监控
                if self.is_monitoring:
                    self._check_alerts(metrics)
                
                logger.info(f"进化评估完成 - 阶段: {stage.value}, 效果: {effectiveness_score:.3f}")
                return metrics
                
        except Exception as e:
            logger.error(f"进化评估失败: {e}")
            raise
    
    def generate_evaluation_report(self, 
                                  evaluation_type: EvaluationType,
                                  time_range: Optional[Tuple[float, float]] = None) -> EvaluationResult:
        """生成评估报告
        
        Args:
            evaluation_type: 评估类型
            time_range: 时间范围
            
        Returns:
            EvaluationResult: 评估结果
        """
        try:
            evaluation_id = f"{evaluation_type.value}_{int(time.time())}"
            
            # 获取时间范围内的数据
            if time_range:
                start_time, end_time = time_range
                filtered_metrics = [
                    m for m in self.metrics_history 
                    if start_time <= m.timestamp <= end_time
                ]
            else:
                filtered_metrics = list(self.metrics_history)
            
            if not filtered_metrics:
                raise ValueError("指定时间范围内没有数据")
            
            # 执行特定类型的评估
            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                if evaluation_type == EvaluationType.EFFECTIVENESS:
                    scores = self.effectiveness_evaluator.generate_report(filtered_metrics)
                    recommendations = self._generate_effectiveness_recommendations(scores)
                elif evaluation_type == EvaluationType.QUALITY:
                    scores = self.quality_evaluator.generate_report(filtered_metrics)
                    recommendations = self._generate_quality_recommendations(scores)
                elif evaluation_type == EvaluationType.ADAPTABILITY:
                    scores = self.adaptability_evaluator.generate_report(filtered_metrics)
                    recommendations = self._generate_adaptability_recommendations(scores)
                elif evaluation_type == EvaluationType.EFFICIENCY:
                    scores = self.efficiency_evaluator.generate_report(filtered_metrics)
                    recommendations = self._generate_efficiency_recommendations(scores)
                elif evaluation_type == EvaluationType.RISK:
                    scores = self.risk_evaluator.generate_report(filtered_metrics)
                    recommendations = self._generate_risk_recommendations(scores)
                elif evaluation_type == EvaluationType.HISTORICAL:
                    scores = self.historical_analyzer.generate_report(filtered_metrics)
                    recommendations = self._generate_historical_recommendations(scores)
                else:
                    raise ValueError(f"不支持的评估类型: {evaluation_type}")
            
            # 风险评估
            risk_assessment = self._perform_risk_assessment(scores)
            
            # 历史对比
            historical_comparison = self._compare_with_history(scores)
            
            # 保存评估结果
            result = EvaluationResult(
                evaluation_id=evaluation_id,
                evaluation_type=evaluation_type,
                timestamp=time.time(),
                scores=scores,
                recommendations=recommendations,
                risk_assessment=risk_assessment,
                historical_comparison=historical_comparison,
                metadata={
                    'data_points': len(filtered_metrics),
                    'time_range': time_range,
                    'evaluation_version': '1.0'
                }
            )
            
            self._save_evaluation_result(result)
            
            logger.info(f"评估报告生成完成 - 类型: {evaluation_type.value}")
            return result
            
        except Exception as e:
            logger.error(f"评估报告生成失败: {e}")
            raise
    
    def generate_comprehensive_report(self) -> EvolutionReport:
        """生成综合进化报告
        
        Returns:
            EvolutionReport: 综合进化报告
        """
        try:
            report_id = f"comprehensive_{int(time.time())}"
            
            # 计算整体得分
            if self.metrics_history:
                recent_metrics = list(self.metrics_history)[-100:]  # 最近100个数据点
                overall_score = np.mean([
                    m.effectiveness_score for m in recent_metrics
                ])
            else:
                overall_score = 0.0
            
            # 阶段分析
            stage_analysis = self._analyze_evolution_stages()
            
            # 趋势分析
            trend_analysis = self._analyze_evolution_trends()
            
            # 生成建议
            recommendations = self._generate_comprehensive_recommendations()
            
            # 风险警报
            risk_alerts = self._generate_risk_alerts()
            
            # 性能指标
            performance_metrics = self._calculate_performance_metrics()
            
            # 历史洞察
            historical_insights = self._generate_historical_insights()
            
            # 构建报告
            report = EvolutionReport(
                report_id=report_id,
                generation_time=time.time(),
                overall_score=overall_score,
                stage_analysis=stage_analysis,
                trend_analysis=trend_analysis,
                recommendations=recommendations,
                risk_alerts=risk_alerts,
                performance_metrics=performance_metrics,
                historical_insights=historical_insights,
                export_timestamp=time.time()
            )
            
            # 保存报告
            self._save_evolution_report(report)
            
            logger.info(f"综合进化报告生成完成 - ID: {report_id}")
            return report
            
        except Exception as e:
            logger.error(f"综合进化报告生成失败: {e}")
            raise
    
    def start_real_time_monitoring(self, callback: Optional[callable] = None):
        """启动实时监控
        
        Args:
            callback: 回调函数
        """
        if self.is_monitoring:
            logger.warning("实时监控已在运行")
            return
        
        self.is_monitoring = True
        self.monitor_callback = callback
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    if self.metrics_history:
                        latest_metrics = self.metrics_history[-1]
                        
                        # 检查是否需要触发回调
                        if self.monitor_callback:
                            self.monitor_callback(latest_metrics)
                        
                        # 检查警报条件
                        self._check_monitoring_alerts(latest_metrics)
                    
                    time.sleep(self.config['evaluation_interval'])
                    
                except Exception as e:
                    logger.error(f"实时监控错误: {e}")
                    time.sleep(5)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("实时监控已启动")
    
    def stop_real_time_monitoring(self):
        """停止实时监控"""
        if not self.is_monitoring:
            logger.warning("实时监控未在运行")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("实时监控已停止")
    
    def export_report(self, report: EvolutionReport, format: str = 'json') -> str:
        """导出报告
        
        Args:
            report: 进化报告
            format: 导出格式 ('json', 'csv', 'excel')
            
        Returns:
            str: 导出文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == 'json':
                file_path = f"evolution_report_{timestamp}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)
            
            elif format == 'csv':
                file_path = f"evolution_metrics_{timestamp}.csv"
                # 将指标数据转换为DataFrame并导出
                data = []
                for metrics in self.metrics_history:
                    data.append(asdict(metrics))
                
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            elif format == 'excel':
                file_path = f"evolution_report_{timestamp}.xlsx"
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # 指标数据表
                    metrics_data = [asdict(m) for m in self.metrics_history]
                    df_metrics = pd.DataFrame(metrics_data)
                    df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
                    
                    # 报告数据
                    df_report = pd.DataFrame([asdict(report)])
                    df_report.to_excel(writer, sheet_name='Report', index=False)
            
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"报告已导出到: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"报告导出失败: {e}")
            raise
    
    def _is_numeric(self, value: Any) -> bool:
        """检查值是否为数值"""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def _calculate_convergence_rate(self, current_state: Dict, target_state: Optional[Dict]) -> float:
        """计算收敛率"""
        if not target_state:
            return 0.5  # 默认中等收敛率
        
        try:
            # 安全地计算收敛率
            differences = []
            max_target_values = []
            
            for key in current_state.keys():
                if key in target_state:
                    c_value = current_state[key]
                    t_value = target_state[key]
                    
                    # 计算单个键的收敛率
                    try:
                        key_convergence = self._calculate_key_convergence(c_value, t_value)
                        differences.append(key_convergence['difference'])
                        max_target_values.append(key_convergence['max_target'])
                    except (TypeError, ValueError):
                        # 如果计算失败，使用默认值
                        differences.append(0.5)
                        max_target_values.append(1.0)
            
            if not differences:
                return 0.5
            
            avg_difference = np.mean(differences)
            max_possible_diff = max(max_target_values + [1.0])
            
            convergence_rate = 1.0 - min(avg_difference / max_possible_diff, 1.0)
            return max(0.0, min(1.0, convergence_rate))
            
        except Exception:
            return 0.5
    
    def _calculate_key_convergence(self, current_value: Any, target_value: Any) -> Dict[str, float]:
        """计算单个键的收敛率"""
        # 处理列表或数组类型
        if isinstance(current_value, (list, tuple, np.ndarray)):
            if isinstance(target_value, (list, tuple, np.ndarray)):
                # 如果都是列表，计算平均收敛率
                differences = []
                max_targets = []
                for c, t in zip(current_value, target_value):
                    try:
                        diff = abs(float(c) - float(t))
                        max_target = max(abs(float(t)), 1.0)
                        differences.append(diff)
                        max_targets.append(max_target)
                    except (TypeError, ValueError):
                        differences.append(0.5)
                        max_targets.append(1.0)
                return {
                    'difference': np.mean(differences) if differences else 0.5,
                    'max_target': np.mean(max_targets) if max_targets else 1.0
                }
            else:
                # 列表与单值比较，计算平均
                differences = []
                max_targets = []
                for c in current_value:
                    try:
                        diff = abs(float(c) - float(target_value))
                        max_target = max(abs(float(target_value)), 1.0)
                        differences.append(diff)
                        max_targets.append(max_target)
                    except (TypeError, ValueError):
                        differences.append(0.5)
                        max_targets.append(1.0)
                return {
                    'difference': np.mean(differences) if differences else 0.5,
                    'max_target': np.mean(max_targets) if max_targets else 1.0
                }
        
        # 处理数值类型
        try:
            c = float(current_value)
            t = float(target_value)
            diff = abs(c - t)
            max_target = max(abs(t), 1.0)
            return {
                'difference': diff,
                'max_target': max_target
            }
        except (TypeError, ValueError):
            return {
                'difference': 0.5,
                'max_target': 1.0
            }
    
    def _calculate_diversity_index(self, current_state: Dict) -> float:
        """计算多样性指数"""
        # 安全地提取可哈希的值
        hashable_values = []
        for value in current_state.values():
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，使用其长度和平均值作为特征
                    if len(value) > 0:
                        numeric_list = [float(v) for v in value if self._is_numeric(v)]
                        if numeric_list:
                            # 使用平均值作为代表值
                            hashable_values.append(('list', np.mean(numeric_list)))
                        else:
                            hashable_values.append(('list', 0.0))
                    else:
                        hashable_values.append(('empty_list', 0.0))
                else:
                    # 对于标量值，直接使用
                    hashable_values.append(('scalar', float(value)))
            except (TypeError, ValueError):
                hashable_values.append(('invalid', 0.0))
        
        if len(hashable_values) <= 1:
            return 0.0
        
        # 使用香农多样性指数
        value_counts = defaultdict(float)
        for value in hashable_values:
            value_counts[value] += 1
        
        total = len(hashable_values)
        diversity = 0.0
        for count in value_counts.values():
            if count > 0:
                p = count / total
                diversity -= p * np.log(p)
        
        # 归一化
        max_diversity = np.log(len(value_counts))
        return diversity / max_diversity if max_diversity > 0 else 0.0
    
    def _calculate_stability_score(self) -> float:
        """计算稳定性得分"""
        if len(self.metrics_history) < 10:
            return 0.5
        
        recent_metrics = list(self.metrics_history)[-10:]
        effectiveness_scores = [m.effectiveness_score for m in recent_metrics]
        
        # 计算变异系数
        mean_score = np.mean(effectiveness_scores)
        std_score = np.std(effectiveness_scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = std_score / mean_score
        stability_score = 1.0 / (1.0 + cv)  # 变异系数越小，稳定性越高
        
        return stability_score
    
    def _calculate_learning_rate(self) -> float:
        """计算学习率"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-2:]
        score_diff = recent_metrics[1].effectiveness_score - recent_metrics[0].effectiveness_score
        time_diff = recent_metrics[1].timestamp - recent_metrics[0].timestamp
        
        if time_diff == 0:
            return 0.0
        
        learning_rate = score_diff / time_diff
        return max(0.0, min(1.0, learning_rate))
    
    def _calculate_adaptation_speed(self) -> float:
        """计算适应速度"""
        if len(self.metrics_history) < 5:
            return 0.5
        
        recent_metrics = list(self.metrics_history)[-5:]
        adaptability_scores = [m.adaptability_score for m in recent_metrics]
        
        # 计算适应性变化趋势
        x = np.arange(len(adaptability_scores))
        y = np.array(adaptability_scores)
        
        if len(x) < 2:
            return 0.5
        
        # 简单线性回归
        slope = np.polyfit(x, y, 1)[0]
        
        # 归一化斜率
        adaptation_speed = max(0.0, min(1.0, slope * 10))
        return adaptation_speed
    
    def _calculate_resource_utilization(self, current_state: Dict) -> float:
        """计算资源利用率"""
        total_resources = 0
        used_resources = 0
        
        for value in current_state.values():
            total_resources += 1
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，检查非零元素
                    used_in_list = sum(1 for item in value if self._is_numeric(item) and float(item) != 0)
                    if len(value) > 0:
                        used_ratio = used_in_list / len(value)
                        used_resources += used_ratio
                else:
                    if self._is_numeric(value) and float(value) != 0:
                        used_resources += 1
            except (TypeError, ValueError):
                continue
        
        return used_resources / total_resources if total_resources > 0 else 0.0
    
    def _calculate_error_rate(self, current_state: Dict) -> float:
        """计算错误率"""
        total_operations = 0
        error_operations = 0
        
        for value in current_state.values():
            total_operations += 1
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，检查负元素
                    error_in_list = sum(1 for item in value if self._is_numeric(item) and float(item) < 0)
                    if len(value) > 0:
                        error_ratio = error_in_list / len(value)
                        error_operations += error_ratio
                else:
                    if self._is_numeric(value) and float(value) < 0:
                        error_operations += 1
            except (TypeError, ValueError):
                continue
        
        return error_operations / total_operations if total_operations > 0 else 0.0
    
    def _calculate_success_rate(self, current_state: Dict) -> float:
        """计算成功率"""
        total_operations = 0
        success_operations = 0
        
        for value in current_state.values():
            total_operations += 1
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，检查正元素
                    success_in_list = sum(1 for item in value if self._is_numeric(item) and float(item) > 0)
                    if len(value) > 0:
                        success_ratio = success_in_list / len(value)
                        success_operations += success_ratio
                else:
                    if self._is_numeric(value) and float(value) > 0:
                        success_operations += 1
            except (TypeError, ValueError):
                continue
        
        return success_operations / total_operations if total_operations > 0 else 0.0
    
    def _determine_evolution_stage(self, effectiveness_score: float, convergence_rate: float) -> EvolutionStage:
        """确定进化阶段"""
        if effectiveness_score < 0.3:
            return EvolutionStage.INITIALIZATION
        elif effectiveness_score < 0.6:
            return EvolutionStage.GROWTH
        elif effectiveness_score < 0.8:
            return EvolutionStage.MATURATION
        elif convergence_rate < self.config['convergence_threshold']:
            return EvolutionStage.OPTIMIZATION
        else:
            return EvolutionStage.CONVERGENCE
    
    def _save_metrics_to_db(self, metrics: EvolutionMetrics):
        """保存指标到数据库"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO evolution_metrics (
                    timestamp, stage, effectiveness_score, quality_score,
                    adaptability_score, efficiency_score, risk_score,
                    convergence_rate, diversity_index, stability_score,
                    learning_rate, adaptation_speed, resource_utilization,
                    error_rate, success_rate, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.stage.value, metrics.effectiveness_score,
                metrics.quality_score, metrics.adaptability_score, metrics.efficiency_score,
                metrics.risk_score, metrics.convergence_rate, metrics.diversity_index,
                metrics.stability_score, metrics.learning_rate, metrics.adaptation_speed,
                metrics.resource_utilization, metrics.error_rate, metrics.success_rate,
                json.dumps(metrics.metadata)
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"保存指标到数据库失败: {e}")
    
    def _save_evaluation_result(self, result: EvaluationResult):
        """保存评估结果到数据库"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO evaluation_results (
                    evaluation_id, evaluation_type, timestamp, scores,
                    recommendations, risk_assessment, historical_comparison, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.evaluation_id, result.evaluation_type.value, result.timestamp,
                json.dumps(result.scores), json.dumps(result.recommendations),
                json.dumps(result.risk_assessment), json.dumps(result.historical_comparison),
                json.dumps(result.metadata)
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
    
    def _save_evolution_report(self, report: EvolutionReport):
        """保存进化报告到数据库"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO evolution_reports (
                    report_id, generation_time, overall_score, stage_analysis,
                    trend_analysis, recommendations, risk_alerts, performance_metrics,
                    historical_insights, export_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.report_id, report.generation_time, report.overall_score,
                json.dumps(report.stage_analysis), json.dumps(report.trend_analysis),
                json.dumps(report.recommendations), json.dumps(report.risk_alerts),
                json.dumps(report.performance_metrics), json.dumps(report.historical_insights),
                report.export_timestamp
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"保存进化报告失败: {e}")
    
    def _check_alerts(self, metrics: EvolutionMetrics):
        """检查警报条件"""
        alerts = []
        
        if metrics.risk_score > self.config['risk_threshold']:
            alerts.append(f"高风险警报: 风险得分 {metrics.risk_score:.3f}")
        
        if metrics.efficiency_score < self.config['efficiency_threshold']:
            alerts.append(f"低效率警报: 效率得分 {metrics.efficiency_score:.3f}")
        
        if metrics.quality_score < self.config['quality_threshold']:
            alerts.append(f"低质量警报: 质量得分 {metrics.quality_score:.3f}")
        
        if alerts:
            logger.warning(f"进化警报: {'; '.join(alerts)}")
    
    def _check_monitoring_alerts(self, metrics: EvolutionMetrics):
        """检查监控警报"""
        if hasattr(self, 'monitor_callback') and self.monitor_callback:
            alerts = []
            
            if metrics.risk_score > self.config['risk_threshold']:
                alerts.append({
                    'type': 'risk',
                    'level': 'high',
                    'message': f'风险得分过高: {metrics.risk_score:.3f}',
                    'timestamp': metrics.timestamp
                })
            
            if metrics.effectiveness_score < 0.3:
                alerts.append({
                    'type': 'effectiveness',
                    'level': 'medium',
                    'message': f'效果得分过低: {metrics.effectiveness_score:.3f}',
                    'timestamp': metrics.timestamp
                })
            
            if alerts:
                self.monitor_callback({
                    'type': 'alert',
                    'alerts': alerts,
                    'metrics': asdict(metrics)
                })
    
    # 建议生成方法
    def _generate_effectiveness_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """生成效果评估建议"""
        recommendations = []
        
        if scores.get('overall', 0) < 0.5:
            recommendations.append("建议优化算法参数以提高进化效果")
        
        if scores.get('convergence', 0) < 0.7:
            recommendations.append("建议增加训练迭代次数以提高收敛性")
        
        if scores.get('stability', 0) < 0.6:
            recommendations.append("建议优化学习率调度策略以提高稳定性")
        
        return recommendations
    
    def _generate_quality_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """生成质量评估建议"""
        recommendations = []
        
        if scores.get('accuracy', 0) < 0.8:
            recommendations.append("建议提高模型准确性和预测精度")
        
        if scores.get('consistency', 0) < 0.7:
            recommendations.append("建议增强结果一致性和可重复性")
        
        if scores.get('reliability', 0) < 0.75:
            recommendations.append("建议增强系统可靠性和容错能力")
        
        return recommendations
    
    def _generate_adaptability_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """生成适应性评估建议"""
        recommendations = []
        
        if scores.get('flexibility', 0) < 0.6:
            recommendations.append("建议提高系统灵活性和适应性")
        
        if scores.get('responsiveness', 0) < 0.7:
            recommendations.append("建议增强系统响应速度和适应性")
        
        if scores.get('scalability', 0) < 0.8:
            recommendations.append("建议优化系统扩展性和可扩展性")
        
        return recommendations
    
    def _generate_efficiency_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """生成效率评估建议"""
        recommendations = []
        
        if scores.get('speed', 0) < 0.7:
            recommendations.append("建议优化算法效率以提高执行速度")
        
        if scores.get('resource_usage', 0) > 0.8:
            recommendations.append("建议优化资源使用以降低消耗")
        
        if scores.get('throughput', 0) < 0.75:
            recommendations.append("建议提高系统吞吐量和处理能力")
        
        return recommendations
    
    def _generate_risk_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """生成风险评估建议"""
        recommendations = []
        
        if scores.get('overall_risk', 0) > 0.7:
            recommendations.append("高风险状态，建议立即采取风险控制措施")
        
        if scores.get('stability_risk', 0) > 0.6:
            recommendations.append("稳定性风险较高，建议增强系统稳定性")
        
        if scores.get('performance_risk', 0) > 0.5:
            recommendations.append("性能风险存在，建议优化性能表现")
        
        return recommendations
    
    def _generate_historical_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """生成历史分析建议"""
        recommendations = []
        
        if scores.get('trend_direction', 0) < 0:
            recommendations.append("历史趋势显示下降，建议分析原因并采取改进措施")
        
        if scores.get('volatility', 0) > 0.7:
            recommendations.append("历史波动较大，建议提高系统稳定性")
        
        if scores.get('improvement_rate', 0) < 0.1:
            recommendations.append("改进速度较慢，建议优化改进策略")
        
        return recommendations
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        if not self.metrics_history:
            return ["系统尚未开始进化，建议先进行初始化"]
        
        recent_metrics = list(self.metrics_history)[-10:]
        avg_effectiveness = np.mean([m.effectiveness_score for m in recent_metrics])
        avg_risk = np.mean([m.risk_score for m in recent_metrics])
        
        if avg_effectiveness < 0.5:
            recommendations.append("整体进化效果不佳，建议全面优化算法参数")
        
        if avg_risk > 0.7:
            recommendations.append("系统风险较高，建议加强风险控制机制")
        
        if len(recent_metrics) >= 5:
            recent_trends = [m.effectiveness_score for m in recent_metrics[-5:]]
            if all(recent_trends[i] <= recent_trends[i+1] for i in range(len(recent_trends)-1)):
                recommendations.append("进化效果持续改善，保持当前策略")
            elif all(recent_trends[i] >= recent_trends[i+1] for i in range(len(recent_trends)-1)):
                recommendations.append("进化效果持续下降，建议调整策略")
        
        return recommendations
    
    def _generate_risk_alerts(self) -> List[str]:
        """生成风险警报"""
        alerts = []
        
        if not self.metrics_history:
            return alerts
        
        recent_metrics = list(self.metrics_history)[-5:]
        
        for metrics in recent_metrics:
            if metrics.risk_score > self.config['risk_threshold']:
                alerts.append(f"高风险警报 ({datetime.fromtimestamp(metrics.timestamp)}): 风险得分 {metrics.risk_score:.3f}")
            
            if metrics.effectiveness_score < 0.3:
                alerts.append(f"低效果警报 ({datetime.fromtimestamp(metrics.timestamp)}): 效果得分 {metrics.effectiveness_score:.3f}")
            
            if metrics.error_rate > 0.2:
                alerts.append(f"高错误率警报 ({datetime.fromtimestamp(metrics.timestamp)}): 错误率 {metrics.error_rate:.3f}")
        
        return alerts
    
    def _perform_risk_assessment(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """执行风险评估"""
        overall_risk = scores.get('overall', 0)
        
        if overall_risk > 0.8:
            risk_level = RiskLevel.CRITICAL
        elif overall_risk > 0.6:
            risk_level = RiskLevel.HIGH
        elif overall_risk > 0.4:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            'risk_level': risk_level.value,
            'risk_score': overall_risk,
            'risk_factors': self._identify_risk_factors(scores),
            'mitigation_strategies': self._generate_mitigation_strategies(overall_risk)
        }
    
    def _identify_risk_factors(self, scores: Dict[str, float]) -> List[str]:
        """识别风险因素"""
        risk_factors = []
        
        for metric, score in scores.items():
            if score < 0.3:
                risk_factors.append(f"{metric}: 低分风险 ({score:.3f})")
            elif score > 0.8:
                risk_factors.append(f"{metric}: 过度优化风险 ({score:.3f})")
        
        return risk_factors
    
    def _generate_mitigation_strategies(self, risk_score: float) -> List[str]:
        """生成缓解策略"""
        strategies = []
        
        if risk_score > 0.7:
            strategies.extend([
                "立即实施风险控制措施",
                "增加监控频率",
                "准备回滚方案"
            ])
        elif risk_score > 0.5:
            strategies.extend([
                "加强风险监控",
                "优化参数设置",
                "增强容错机制"
            ])
        else:
            strategies.append("保持当前稳定状态")
        
        return strategies
    
    def _compare_with_history(self, current_scores: Dict[str, float]) -> Dict[str, Any]:
        """与历史数据对比"""
        if len(self.metrics_history) < 10:
            return {'message': '历史数据不足，无法进行对比'}
        
        # 获取历史平均得分
        historical_scores = defaultdict(list)
        for metrics in list(self.metrics_history)[-50:]:  # 最近50个数据点
            historical_scores['effectiveness'].append(metrics.effectiveness_score)
            historical_scores['quality'].append(metrics.quality_score)
            historical_scores['adaptability'].append(metrics.adaptability_score)
            historical_scores['efficiency'].append(metrics.efficiency_score)
            historical_scores['risk'].append(metrics.risk_score)
        
        comparison = {}
        for metric, values in historical_scores.items():
            if metric in current_scores:
                historical_avg = np.mean(values)
                current_value = current_scores[metric]
                change_percent = ((current_value - historical_avg) / historical_avg) * 100 if historical_avg != 0 else 0
                
                comparison[metric] = {
                    'historical_average': historical_avg,
                    'current_value': current_value,
                    'change_percent': change_percent,
                    'trend': 'improving' if change_percent > 5 else 'declining' if change_percent < -5 else 'stable'
                }
        
        return comparison
    
    def _analyze_evolution_stages(self) -> Dict[str, Any]:
        """分析进化阶段"""
        if not self.metrics_history:
            return {'message': '无进化数据'}
        
        stage_counts = defaultdict(int)
        stage_durations = defaultdict(list)
        
        current_stage = None
        stage_start_time = None
        
        for metrics in self.metrics_history:
            stage = metrics.stage
            stage_counts[stage.value] += 1
            
            if current_stage != stage:
                if current_stage is not None and stage_start_time is not None:
                    duration = metrics.timestamp - stage_start_time
                    stage_durations[current_stage.value].append(duration)
                
                current_stage = stage
                stage_start_time = metrics.timestamp
        
        # 处理最后一个阶段
        if current_stage is not None and stage_start_time is not None:
            duration = time.time() - stage_start_time
            stage_durations[current_stage.value].append(duration)
        
        analysis = {
            'stage_distribution': dict(stage_counts),
            'stage_durations': {k: np.mean(v) if v else 0 for k, v in stage_durations.items()},
            'current_stage': current_stage.value if current_stage else 'unknown',
            'dominant_stage': max(stage_counts.items(), key=lambda x: x[1])[0] if stage_counts else 'unknown'
        }
        
        return analysis
    
    def _analyze_evolution_trends(self) -> Dict[str, Any]:
        """分析进化趋势"""
        if len(self.metrics_history) < 5:
            return {'message': '数据点不足，无法分析趋势'}
        
        recent_metrics = list(self.metrics_history)[-20:]  # 最近20个数据点
        
        # 提取时间序列数据
        timestamps = [m.timestamp for m in recent_metrics]
        effectiveness_scores = [m.effectiveness_score for m in recent_metrics]
        quality_scores = [m.quality_score for m in recent_metrics]
        adaptability_scores = [m.adaptability_score for m in recent_metrics]
        efficiency_scores = [m.efficiency_score for m in recent_metrics]
        
        # 计算趋势
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return slope
        
        trends = {
            'effectiveness_trend': calculate_trend(effectiveness_scores),
            'quality_trend': calculate_trend(quality_scores),
            'adaptability_trend': calculate_trend(adaptability_scores),
            'efficiency_trend': calculate_trend(efficiency_scores)
        }
        
        # 趋势解释
        trend_interpretation = {}
        for metric, slope in trends.items():
            if slope > 0.01:
                trend_interpretation[metric] = '上升趋势'
            elif slope < -0.01:
                trend_interpretation[metric] = '下降趋势'
            else:
                trend_interpretation[metric] = '稳定趋势'
        
        return {
            'trends': trends,
            'interpretation': trend_interpretation,
            'overall_direction': 'improving' if sum(trends.values()) > 0 else 'declining'
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """计算性能指标"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-50:]  # 最近50个数据点
        
        metrics = {
            'avg_effectiveness': np.mean([m.effectiveness_score for m in recent_metrics]),
            'avg_quality': np.mean([m.quality_score for m in recent_metrics]),
            'avg_adaptability': np.mean([m.adaptability_score for m in recent_metrics]),
            'avg_efficiency': np.mean([m.efficiency_score for m in recent_metrics]),
            'avg_risk': np.mean([m.risk_score for m in recent_metrics]),
            'stability_index': np.mean([m.stability_score for m in recent_metrics]),
            'learning_efficiency': np.mean([m.learning_rate for m in recent_metrics]),
            'adaptation_speed': np.mean([m.adaptation_speed for m in recent_metrics]),
            'resource_efficiency': np.mean([m.resource_utilization for m in recent_metrics]),
            'success_rate': np.mean([m.success_rate for m in recent_metrics])
        }
        
        return metrics
    
    def _generate_historical_insights(self) -> Dict[str, Any]:
        """生成历史洞察"""
        if len(self.metrics_history) < 10:
            return {'message': '历史数据不足，无法生成洞察'}
        
        # 计算历史统计
        all_effectiveness = [m.effectiveness_score for m in self.metrics_history]
        all_risk = [m.risk_score for m in self.metrics_history]
        
        insights = {
            'best_performance': {
                'effectiveness': max(all_effectiveness),
                'timestamp': self.metrics_history[all_effectiveness.index(max(all_effectiveness))].timestamp
            },
            'worst_performance': {
                'effectiveness': min(all_effectiveness),
                'timestamp': self.metrics_history[all_effectiveness.index(min(all_effectiveness))].timestamp
            },
            'performance_volatility': np.std(all_effectiveness),
            'risk_volatility': np.std(all_risk),
            'total_evaluations': len(self.metrics_history),
            'evaluation_period': {
                'start': min(m.timestamp for m in self.metrics_history),
                'end': max(m.timestamp for m in self.metrics_history)
            }
        }
        
        return insights
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()


class EffectivenessEvaluator:
    """效果评估器"""
    
    def evaluate(self, current_state: Dict, target_state: Optional[Dict], context: Optional[Dict]) -> float:
        """评估进化效果"""
        # 简化的效果评估
        if not target_state:
            return 0.5
        
        # 计算目标达成度
        achievement_score = self._calculate_achievement_score(current_state, target_state)
        
        # 计算效率得分
        efficiency_score = self._calculate_efficiency_score(current_state)
        
        # 综合得分
        effectiveness_score = (achievement_score * 0.7 + efficiency_score * 0.3)
        
        return max(0.0, min(1.0, effectiveness_score))
    
    def generate_report(self, metrics_history: List[EvolutionMetrics]) -> Dict[str, float]:
        """生成效果评估报告"""
        if not metrics_history:
            return {}
        
        effectiveness_scores = [m.effectiveness_score for m in metrics_history]
        convergence_rates = [m.convergence_rate for m in metrics_history]
        
        report = {
            'overall': np.mean(effectiveness_scores),
            'convergence': np.mean(convergence_rates),
            'stability': np.std(effectiveness_scores),
            'trend': self._calculate_trend(effectiveness_scores),
            'best': max(effectiveness_scores),
            'worst': min(effectiveness_scores)
        }
        
        return report
    
    def _calculate_achievement_score(self, current_state: Dict, target_state: Dict) -> float:
        """计算目标达成得分"""
        if not target_state:
            return 0.5
        
        current_values = list(current_state.values())
        target_values = list(target_state.values())
        
        if len(current_values) != len(target_values):
            return 0.5
        
        # 计算达成度
        achievements = []
        for c, t in zip(current_values, target_values):
            try:
                # 安全地处理不同类型的数据
                achievement = self._calculate_single_achievement(c, t)
                achievements.append(achievement)
            except (TypeError, ValueError, ZeroDivisionError):
                # 如果计算失败，使用默认值
                achievements.append(0.5)
        
        return np.mean(achievements) if achievements else 0.5
    
    def _calculate_single_achievement(self, current_value: Any, target_value: Any) -> float:
        """计算单个值的达成度"""
        # 处理列表或数组类型
        if isinstance(current_value, (list, tuple, np.ndarray)):
            if isinstance(target_value, (list, tuple, np.ndarray)):
                # 如果都是列表，计算平均达成度
                achievements = []
                for c, t in zip(current_value, target_value):
                    try:
                        achievement = self._calculate_single_achievement(c, t)
                        achievements.append(achievement)
                    except (TypeError, ValueError, ZeroDivisionError):
                        achievements.append(0.5)
                return np.mean(achievements) if achievements else 0.5
            else:
                # 列表与单值比较，计算平均
                achievements = []
                for c in current_value:
                    try:
                        achievement = self._calculate_single_achievement(c, target_value)
                        achievements.append(achievement)
                    except (TypeError, ValueError, ZeroDivisionError):
                        achievements.append(0.5)
                return np.mean(achievements) if achievements else 0.5
        
        # 处理数值类型
        try:
            c = float(current_value)
            t = float(target_value)
            
            if t != 0:
                achievement = min(c / t, 1.0) if t > 0 else min((c - t) / abs(t), 1.0)
            else:
                achievement = 1.0 if c == 0 else 0.0
            
            return max(0.0, min(1.0, achievement))
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.5
    
    def _calculate_efficiency_score(self, current_state: Dict) -> float:
        """计算效率得分"""
        values = list(current_state.values())
        positive_values = []
        
        for v in values:
            try:
                # 安全地处理不同类型的数据
                if isinstance(v, (list, tuple, np.ndarray)):
                    # 对于列表，计算正值的比例
                    positive_in_list = sum(1 for item in v if self._is_positive_number(item))
                    if len(v) > 0:
                        positive_values.append(positive_in_list / len(v))
                else:
                    if self._is_positive_number(v):
                        positive_values.append(1.0)
            except (TypeError, ValueError):
                continue
        
        if not positive_values:
            return 0.0
        
        # 简化的效率计算
        efficiency = np.mean(positive_values)
        return efficiency
    
    def _is_positive_number(self, value: Any) -> bool:
        """检查值是否为正数"""
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False
    
    def _is_numeric(self, value: Any) -> bool:
        """检查值是否为数值"""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope


class QualityEvaluator:
    """质量评估器"""
    
    def evaluate(self, current_state: Dict, target_state: Optional[Dict], context: Optional[Dict]) -> float:
        """评估进化质量"""
        # 简化的质量评估
        consistency_score = self._calculate_consistency_score(current_state)
        reliability_score = self._calculate_reliability_score(current_state)
        accuracy_score = self._calculate_accuracy_score(current_state, target_state)
        
        quality_score = (consistency_score * 0.3 + reliability_score * 0.4 + accuracy_score * 0.3)
        
        return max(0.0, min(1.0, quality_score))
    
    def generate_report(self, metrics_history: List[EvolutionMetrics]) -> Dict[str, float]:
        """生成质量评估报告"""
        if not metrics_history:
            return {}
        
        quality_scores = [m.quality_score for m in metrics_history]
        
        report = {
            'overall': np.mean(quality_scores),
            'consistency': np.std(quality_scores),
            'trend': self._calculate_trend(quality_scores),
            'stability': 1.0 / (1.0 + np.std(quality_scores)),
            'reliability': np.mean([m.stability_score for m in metrics_history])
        }
        
        return report
    
    def _is_numeric(self, value: Any) -> bool:
        """检查值是否为数值"""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def _calculate_consistency_score(self, current_state: Dict) -> float:
        """计算一致性得分"""
        # 安全地提取数值
        numeric_values = []
        for value in current_state.values():
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，取平均值
                    if len(value) > 0:
                        numeric_values.append(np.mean([float(v) for v in value if self._is_numeric(v)]))
                else:
                    if self._is_numeric(value):
                        numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
        
        if len(numeric_values) <= 1:
            return 1.0
        
        # 计算变异系数
        mean_val = np.mean(numeric_values)
        std_val = np.std(numeric_values)
        
        if mean_val == 0:
            return 1.0 if std_val == 0 else 0.0
        
        cv = std_val / abs(mean_val)
        consistency_score = 1.0 / (1.0 + cv)
        
        return consistency_score
    
    def _calculate_reliability_score(self, current_state: Dict) -> float:
        """计算可靠性得分"""
        total_values = 0
        valid_values = 0
        
        for value in current_state.values():
            total_values += 1
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，检查每个元素
                    valid_in_list = sum(1 for item in value if self._is_numeric(item))
                    if len(value) > 0:
                        valid_ratio = valid_in_list / len(value)
                        valid_values += valid_ratio
                else:
                    if self._is_numeric(value):
                        valid_values += 1
            except (TypeError, ValueError):
                continue
        
        if total_values == 0:
            return 0.0
        
        reliability_score = valid_values / total_values
        return reliability_score
    
    def _calculate_accuracy_score(self, current_state: Dict, target_state: Optional[Dict]) -> float:
        """计算准确性得分"""
        if not target_state:
            return 0.5
        
        accuracies = []
        
        for key in current_state.keys():
            if key in target_state:
                c_value = current_state[key]
                t_value = target_state[key]
                
                try:
                    accuracy = self._calculate_single_accuracy(c_value, t_value)
                    accuracies.append(accuracy)
                except (TypeError, ValueError, ZeroDivisionError):
                    accuracies.append(0.5)
        
        return np.mean(accuracies) if accuracies else 0.5
    
    def _calculate_single_accuracy(self, current_value: Any, target_value: Any) -> float:
        """计算单个值的准确性"""
        # 处理列表或数组类型
        if isinstance(current_value, (list, tuple, np.ndarray)):
            if isinstance(target_value, (list, tuple, np.ndarray)):
                # 如果都是列表，计算平均准确性
                accuracies = []
                for c, t in zip(current_value, target_value):
                    try:
                        acc = self._calculate_single_accuracy(c, t)
                        accuracies.append(acc)
                    except (TypeError, ValueError, ZeroDivisionError):
                        accuracies.append(0.5)
                return np.mean(accuracies) if accuracies else 0.5
            else:
                # 列表与单值比较，计算平均
                accuracies = []
                for c in current_value:
                    try:
                        acc = self._calculate_single_accuracy(c, target_value)
                        accuracies.append(acc)
                    except (TypeError, ValueError, ZeroDivisionError):
                        accuracies.append(0.5)
                return np.mean(accuracies) if accuracies else 0.5
        
        # 处理数值类型
        try:
            c = float(current_value)
            t = float(target_value)
            
            if t != 0:
                accuracy = 1.0 - min(abs(c - t) / abs(t), 1.0)
            else:
                accuracy = 1.0 if c == 0 else 0.0
            
            return max(0.0, min(1.0, accuracy))
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.5
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope


class AdaptabilityEvaluator:
    """适应性评估器"""
    
    def evaluate(self, current_state: Dict, target_state: Optional[Dict], context: Optional[Dict]) -> float:
        """评估进化适应性"""
        flexibility_score = self._calculate_flexibility_score(current_state)
        responsiveness_score = self._calculate_responsiveness_score(current_state)
        scalability_score = self._calculate_scalability_score(current_state)
        
        adaptability_score = (flexibility_score * 0.4 + responsiveness_score * 0.3 + scalability_score * 0.3)
        
        return max(0.0, min(1.0, adaptability_score))
    
    def generate_report(self, metrics_history: List[EvolutionMetrics]) -> Dict[str, float]:
        """生成适应性评估报告"""
        if not metrics_history:
            return {}
        
        adaptability_scores = [m.adaptability_score for m in metrics_history]
        
        report = {
            'overall': np.mean(adaptability_scores),
            'flexibility': np.mean([m.diversity_index for m in metrics_history]),
            'responsiveness': np.mean([m.adaptation_speed for m in metrics_history]),
            'scalability': np.mean([m.resource_utilization for m in metrics_history]),
            'trend': self._calculate_trend(adaptability_scores)
        }
        
        return report
    
    def _is_numeric(self, value: Any) -> bool:
        """检查值是否为数值"""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def _calculate_flexibility_score(self, current_state: Dict) -> float:
        """计算灵活性得分"""
        # 安全地提取数值
        numeric_values = []
        for value in current_state.values():
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，取标准差
                    if len(value) > 0:
                        numeric_list = [float(v) for v in value if self._is_numeric(v)]
                        if len(numeric_list) > 0:
                            numeric_values.append(np.std(numeric_list))
                else:
                    if self._is_numeric(value):
                        numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
        
        if len(numeric_values) <= 1:
            return 0.0
        
        # 使用变异系数衡量灵活性
        mean_val = np.mean(numeric_values)
        std_val = np.std(numeric_values)
        
        if mean_val == 0:
            return 0.0
        
        cv = std_val / abs(mean_val)
        flexibility_score = min(cv, 1.0)
        
        return flexibility_score
    
    def _calculate_responsiveness_score(self, current_state: Dict) -> float:
        """计算响应性得分"""
        total_values = 0
        non_zero_values = 0
        
        for value in current_state.values():
            total_values += 1
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，检查非零元素
                    non_zero_in_list = sum(1 for item in value if self._is_numeric(item) and float(item) != 0)
                    if len(value) > 0:
                        non_zero_ratio = non_zero_in_list / len(value)
                        non_zero_values += non_zero_ratio
                else:
                    if self._is_numeric(value) and float(value) != 0:
                        non_zero_values += 1
            except (TypeError, ValueError):
                continue
        
        if total_values == 0:
            return 0.0
        
        responsiveness_score = non_zero_values / total_values
        return responsiveness_score
    
    def _calculate_scalability_score(self, current_state: Dict) -> float:
        """计算可扩展性得分"""
        total_values = 0
        positive_values = 0
        
        for value in current_state.values():
            total_values += 1
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，检查正元素
                    positive_in_list = sum(1 for item in value if self._is_numeric(item) and float(item) > 0)
                    if len(value) > 0:
                        positive_ratio = positive_in_list / len(value)
                        positive_values += positive_ratio
                else:
                    if self._is_numeric(value) and float(value) > 0:
                        positive_values += 1
            except (TypeError, ValueError):
                continue
        
        if total_values == 0:
            return 0.0
        
        # 简化的可扩展性计算
        scalability_score = positive_values / total_values
        return scalability_score
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope


class EfficiencyEvaluator:
    """效率评估器"""
    
    def evaluate(self, current_state: Dict, target_state: Optional[Dict], context: Optional[Dict]) -> float:
        """评估进化效率"""
        speed_score = self._calculate_speed_score(current_state)
        resource_score = self._calculate_resource_score(current_state)
        throughput_score = self._calculate_throughput_score(current_state)
        
        efficiency_score = (speed_score * 0.4 + resource_score * 0.3 + throughput_score * 0.3)
        
        return max(0.0, min(1.0, efficiency_score))
    
    def generate_report(self, metrics_history: List[EvolutionMetrics]) -> Dict[str, float]:
        """生成效率评估报告"""
        if not metrics_history:
            return {}
        
        efficiency_scores = [m.efficiency_score for m in metrics_history]
        
        report = {
            'overall': np.mean(efficiency_scores),
            'speed': np.mean([m.learning_rate for m in metrics_history]),
            'resource_usage': 1.0 - np.mean([m.resource_utilization for m in metrics_history]),
            'throughput': np.mean([m.success_rate for m in metrics_history]),
            'trend': self._calculate_trend(efficiency_scores)
        }
        
        return report
    
    def _is_numeric(self, value: Any) -> bool:
        """检查值是否为数值"""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def _calculate_speed_score(self, current_state: Dict) -> float:
        """计算速度得分"""
        numeric_values = []
        for value in current_state.values():
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，取绝对值的平均
                    if len(value) > 0:
                        abs_values = [abs(float(v)) for v in value if self._is_numeric(v)]
                        if abs_values:
                            numeric_values.append(np.mean(abs_values))
                else:
                    if self._is_numeric(value):
                        numeric_values.append(abs(float(value)))
            except (TypeError, ValueError):
                continue
        
        if not numeric_values:
            return 0.0
        
        # 简化的速度计算
        avg_value = np.mean(numeric_values)
        speed_score = min(avg_value, 1.0)
        
        return speed_score
    
    def _calculate_resource_score(self, current_state: Dict) -> float:
        """计算资源得分"""
        values = list(current_state.values())
        total_resources = len(values)
        used_resources = sum(1 for v in values if v != 0)
        
        if total_resources == 0:
            return 0.0
        
        # 资源利用率越低，资源得分越高
        utilization = used_resources / total_resources
        resource_score = 1.0 - utilization
        
        return resource_score
    
    def _calculate_throughput_score(self, current_state: Dict) -> float:
        """计算吞吐量得分"""
        total_values = 0
        positive_values = 0
        
        for value in current_state.values():
            total_values += 1
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，检查正元素
                    positive_in_list = sum(1 for item in value if self._is_numeric(item) and float(item) > 0)
                    if len(value) > 0:
                        positive_ratio = positive_in_list / len(value)
                        positive_values += positive_ratio
                else:
                    if self._is_numeric(value) and float(value) > 0:
                        positive_values += 1
            except (TypeError, ValueError):
                continue
        
        if total_values == 0:
            return 0.0
        
        throughput_score = positive_values / total_values
        return throughput_score
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope


class RiskEvaluator:
    """风险评估器"""
    
    def evaluate(self, current_state: Dict, target_state: Optional[Dict], context: Optional[Dict]) -> float:
        """评估进化风险"""
        stability_risk = self._calculate_stability_risk(current_state)
        performance_risk = self._calculate_performance_risk(current_state)
        operational_risk = self._calculate_operational_risk(current_state)
        
        risk_score = (stability_risk * 0.4 + performance_risk * 0.3 + operational_risk * 0.3)
        
        return max(0.0, min(1.0, risk_score))
    
    def generate_report(self, metrics_history: List[EvolutionMetrics]) -> Dict[str, float]:
        """生成风险评估报告"""
        if not metrics_history:
            return {}
        
        risk_scores = [m.risk_score for m in metrics_history]
        
        report = {
            'overall_risk': np.mean(risk_scores),
            'stability_risk': np.std([m.stability_score for m in metrics_history]),
            'performance_risk': np.mean([m.error_rate for m in metrics_history]),
            'operational_risk': 1.0 - np.mean([m.success_rate for m in metrics_history]),
            'trend': self._calculate_trend(risk_scores)
        }
        
        return report
    
    def _is_numeric(self, value: Any) -> bool:
        """检查值是否为数值"""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def _calculate_stability_risk(self, current_state: Dict) -> float:
        """计算稳定性风险"""
        # 安全地提取数值
        numeric_values = []
        for value in current_state.values():
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，取标准差
                    if len(value) > 0:
                        numeric_list = [float(v) for v in value if self._is_numeric(v)]
                        if len(numeric_list) > 0:
                            numeric_values.append(np.std(numeric_list))
                else:
                    if self._is_numeric(value):
                        numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
        
        if len(numeric_values) <= 1:
            return 0.0
        
        # 计算变异系数作为稳定性风险
        mean_val = np.mean(numeric_values)
        std_val = np.std(numeric_values)
        
        if mean_val == 0:
            return 1.0 if std_val > 0 else 0.0
        
        cv = std_val / abs(mean_val)
        stability_risk = min(cv, 1.0)
        
        return stability_risk
    
    def _calculate_performance_risk(self, current_state: Dict) -> float:
        """计算性能风险"""
        total_values = 0
        negative_values = 0
        
        for value in current_state.values():
            total_values += 1
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，检查负元素
                    negative_in_list = sum(1 for item in value if self._is_numeric(item) and float(item) < 0)
                    if len(value) > 0:
                        negative_ratio = negative_in_list / len(value)
                        negative_values += negative_ratio
                else:
                    if self._is_numeric(value) and float(value) < 0:
                        negative_values += 1
            except (TypeError, ValueError):
                continue
        
        if total_values == 0:
            return 1.0
        
        performance_risk = negative_values / total_values
        return performance_risk
    
    def _calculate_operational_risk(self, current_state: Dict) -> float:
        """计算操作风险"""
        total_values = 0
        zero_values = 0
        
        for value in current_state.values():
            total_values += 1
            try:
                if isinstance(value, (list, tuple, np.ndarray)):
                    # 对于列表，检查零元素
                    zero_in_list = sum(1 for item in value if self._is_numeric(item) and float(item) == 0)
                    if len(value) > 0:
                        zero_ratio = zero_in_list / len(value)
                        zero_values += zero_ratio
                else:
                    if self._is_numeric(value) and float(value) == 0:
                        zero_values += 1
            except (TypeError, ValueError):
                continue
        
        if total_values == 0:
            return 1.0
        
        operational_risk = zero_values / total_values
        return operational_risk
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope


class HistoricalAnalyzer:
    """历史分析器"""
    
    def generate_report(self, metrics_history: List[EvolutionMetrics]) -> Dict[str, float]:
        """生成历史分析报告"""
        if not metrics_history:
            return {}
        
        effectiveness_scores = [m.effectiveness_score for m in metrics_history]
        
        report = {
            'trend_direction': self._calculate_trend_direction(effectiveness_scores),
            'volatility': np.std(effectiveness_scores),
            'improvement_rate': self._calculate_improvement_rate(effectiveness_scores),
            'consistency': 1.0 / (1.0 + np.std(effectiveness_scores)),
            'peak_performance': max(effectiveness_scores),
            'average_performance': np.mean(effectiveness_scores)
        }
        
        return report
    
    def _calculate_trend_direction(self, values: List[float]) -> float:
        """计算趋势方向"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # 归一化斜率
        max_possible_slope = max(values) - min(values)
        if max_possible_slope == 0:
            return 0.0
        
        return slope / max_possible_slope
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """计算改进率"""
        if len(values) < 2:
            return 0.0
        
        # 计算改进率
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if not first_half or not second_half:
            return 0.0
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        if first_avg == 0:
            return 0.0
        
        improvement_rate = (second_avg - first_avg) / first_avg
        return max(0.0, improvement_rate)


# 使用示例和测试代码
def demo_evolution_evaluator():
    """演示进化评估器的使用"""
    print("=== H5进化评估器演示 ===")
    
    # 初始化评估器
    config = {
        'history_size': 1000,
        'evaluation_interval': 30,
        'risk_threshold': 0.6,
        'enable_real_time_monitoring': True
    }
    
    evaluator = EvolutionEvaluator(config)
    
    # 模拟进化过程
    print("\n1. 模拟进化过程...")
    for i in range(10):
        current_state = {
            'parameter_1': np.random.normal(0.5, 0.1),
            'parameter_2': np.random.normal(0.7, 0.15),
            'parameter_3': np.random.normal(0.3, 0.05),
            'performance': np.random.normal(0.6, 0.1)
        }
        
        target_state = {
            'parameter_1': 0.8,
            'parameter_2': 0.9,
            'parameter_3': 0.5,
            'performance': 0.85
        }
        
        metrics = evaluator.evaluate_evolution(current_state, target_state)
        print(f"阶段 {i+1}: {metrics.stage.value}, 效果: {metrics.effectiveness_score:.3f}, 质量: {metrics.quality_score:.3f}")
    
    # 生成评估报告
    print("\n2. 生成效果评估报告...")
    effectiveness_report = evaluator.generate_evaluation_report(EvaluationType.EFFECTIVENESS)
    print(f"效果评估报告ID: {effectiveness_report.evaluation_id}")
    print(f"整体得分: {effectiveness_report.scores.get('overall', 0):.3f}")
    print(f"建议: {effectiveness_report.recommendations}")
    
    # 生成质量评估报告
    print("\n3. 生成质量评估报告...")
    quality_report = evaluator.generate_evaluation_report(EvaluationType.QUALITY)
    print(f"质量评估报告ID: {quality_report.evaluation_id}")
    print(f"整体得分: {quality_report.scores.get('overall', 0):.3f}")
    
    # 生成综合进化报告
    print("\n4. 生成综合进化报告...")
    comprehensive_report = evaluator.generate_comprehensive_report()
    print(f"综合报告ID: {comprehensive_report.report_id}")
    print(f"整体得分: {comprehensive_report.overall_score:.3f}")
    print(f"当前阶段: {comprehensive_report.stage_analysis.get('current_stage', 'unknown')}")
    print(f"建议数量: {len(comprehensive_report.recommendations)}")
    
    # 导出报告
    print("\n5. 导出报告...")
    json_file = evaluator.export_report(comprehensive_report, 'json')
    print(f"报告已导出到: {json_file}")
    
    # 启动实时监控
    print("\n6. 启动实时监控...")
    def monitoring_callback(data):
        if data.get('type') == 'alert':
            print(f"监控警报: {data['alerts']}")
    
    evaluator.start_real_time_monitoring(monitoring_callback)
    
    # 运行一段时间的监控
    time.sleep(2)
    
    # 停止监控
    evaluator.stop_real_time_monitoring()
    
    print("\n=== 演示完成 ===")
    return evaluator


if __name__ == "__main__":
    # 运行演示
    evaluator = demo_evolution_evaluator()
    
    # 清理资源
    if hasattr(evaluator, 'db_conn'):
        evaluator.db_conn.close()