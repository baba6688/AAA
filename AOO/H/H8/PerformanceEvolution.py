"""
H8性能进化器
Performance Evolution Engine

实现功能：
1. 性能进化趋势分析
2. 性能进化模式识别
3. 性能进化效果评估
4. 性能进化优化建议
5. 性能进化风险控制
6. 性能进化历史跟踪
7. 性能进化报告生成


日期: 2025-11-05
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvolutionPhase(Enum):
    """进化阶段枚举"""
    INITIAL = "初始阶段"
    GROWTH = "增长阶段"
    MATURE = "成熟阶段"
    OPTIMIZATION = "优化阶段"
    TRANSFORMATION = "转型阶段"
    DECLINE = "衰退阶段"


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "低风险"
    MEDIUM = "中等风险"
    HIGH = "高风险"
    CRITICAL = "严重风险"


class EvolutionPattern(Enum):
    """进化模式枚举"""
    LINEAR = "线性增长"
    EXPONENTIAL = "指数增长"
    CYCLICAL = "周期性波动"
    STEP = "阶梯式增长"
    PLATEAU = "平台期"
    DECLINE = "下降趋势"


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    category: str
    target_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class EvolutionEvent:
    """进化事件数据类"""
    timestamp: datetime
    event_type: str
    description: str
    impact_score: float
    category: str
    metadata: Dict[str, Any]


@dataclass
class RiskAssessment:
    """风险评估数据类"""
    risk_type: str
    level: RiskLevel
    probability: float
    impact: float
    mitigation_strategy: str
    timestamp: datetime


@dataclass
class OptimizationSuggestion:
    """优化建议数据类"""
    category: str
    suggestion: str
    priority: int
    expected_improvement: float
    implementation_cost: float
    timeframe: str


class PerformanceEvolution:
    """性能进化器主类"""
    
    def __init__(self, db_path: str = "performance_evolution.db"):
        """
        初始化性能进化器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_database()
        self.metrics_cache = []
        self.events_cache = []
        self.risk_cache = []
        
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建性能指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                category TEXT,
                target_value REAL,
                threshold REAL
            )
        ''')
        
        # 创建进化事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evolution_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT,
                impact_score REAL,
                category TEXT,
                metadata TEXT
            )
        ''')
        
        # 创建风险评估表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                risk_type TEXT NOT NULL,
                level TEXT NOT NULL,
                probability REAL,
                impact REAL,
                mitigation_strategy TEXT
            )
        ''')
        
        # 创建优化建议表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                category TEXT,
                suggestion TEXT,
                priority INTEGER,
                expected_improvement REAL,
                implementation_cost REAL,
                timeframe TEXT
            )
        ''')
        
        # 创建进化历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evolution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                phase TEXT,
                pattern TEXT,
                metrics_summary TEXT,
                key_events TEXT,
                performance_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("数据库初始化完成")
    
    # ==================== 1. 性能进化趋势分析 ====================
    
    def analyze_evolution_trends(self, 
                                metric_name: str, 
                                time_range: int = 30) -> Dict[str, Any]:
        """
        分析性能进化趋势
        
        Args:
            metric_name: 指标名称
            time_range: 时间范围（天）
            
        Returns:
            趋势分析结果
        """
        try:
            # 获取历史数据
            data = self._get_metric_data(metric_name, time_range)
            if len(data) < 2:
                return {"error": "数据不足，无法进行趋势分析"}
            
            # 计算趋势指标
            values = [item['value'] for item in data]
            timestamps = [datetime.fromisoformat(item['timestamp']) for item in data]
            
            # 线性趋势
            slope = self._calculate_slope(values)
            
            # 增长率
            growth_rate = self._calculate_growth_rate(values)
            
            # 波动性
            volatility = self._calculate_volatility(values)
            
            # 趋势强度
            trend_strength = self._calculate_trend_strength(values)
            
            # 预测
            prediction = self._predict_next_values(values, periods=7)
            
            trend_analysis = {
                "metric_name": metric_name,
                "time_range": time_range,
                "data_points": len(data),
                "trend_analysis": {
                    "slope": slope,
                    "growth_rate": growth_rate,
                    "volatility": volatility,
                    "trend_strength": trend_strength,
                    "direction": self._interpret_slope(slope)
                },
                "statistics": {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                },
                "prediction": {
                    "next_7_days": prediction,
                    "confidence": self._calculate_prediction_confidence(values)
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"完成{metric_name}的进化趋势分析")
            return trend_analysis
            
        except Exception as e:
            logger.error(f"趋势分析失败: {str(e)}")
            return {"error": f"趋势分析失败: {str(e)}"}
    
    def _calculate_slope(self, values: List[float]) -> float:
        """计算线性趋势斜率"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """计算增长率"""
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / values[0] * 100
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """计算波动性"""
        if len(values) < 2:
            return 0.0
        return np.std(values) / np.mean(values) * 100
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """计算趋势强度"""
        if len(values) < 3:
            return 0.0
        # 使用相关系数衡量趋势强度
        x = np.arange(len(values))
        correlation = np.corrcoef(x, values)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _interpret_slope(self, slope: float) -> str:
        """解释斜率含义"""
        if slope > 0.1:
            return "强烈上升"
        elif slope > 0.01:
            return "轻微上升"
        elif slope > -0.01:
            return "平稳"
        elif slope > -0.1:
            return "轻微下降"
        else:
            return "强烈下降"
    
    def _predict_next_values(self, values: List[float], periods: int = 7) -> List[float]:
        """预测未来值"""
        if len(values) < 2:
            return values
        
        # 使用线性回归预测
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        future_x = np.arange(len(values), len(values) + periods)
        predictions = slope * future_x + intercept
        
        return predictions.tolist()
    
    def _calculate_prediction_confidence(self, values: List[float]) -> float:
        """计算预测置信度"""
        if len(values) < 3:
            return 0.5
        
        # 基于数据稳定性和趋势一致性计算置信度
        volatility = self._calculate_volatility(values)
        trend_strength = self._calculate_trend_strength(values)
        
        # 置信度与稳定性和趋势强度正相关
        confidence = min(0.95, max(0.1, (1 - volatility/100) * trend_strength))
        return confidence
    
    def _get_metric_data(self, metric_name: str, time_range: int) -> List[Dict]:
        """获取指标数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=time_range)).isoformat()
        
        cursor.execute('''
            SELECT timestamp, value, unit, category, target_value, threshold
            FROM performance_metrics
            WHERE metric_name = ? AND timestamp >= ?
            ORDER BY timestamp
        ''', (metric_name, start_date))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'timestamp': row[0],
                'value': row[1],
                'unit': row[2],
                'category': row[3],
                'target_value': row[4],
                'threshold': row[5]
            })
        
        conn.close()
        return results
    
    # ==================== 2. 性能进化模式识别 ====================
    
    def identify_evolution_patterns(self, metric_name: str, time_range: int = 30) -> Dict[str, Any]:
        """
        识别性能进化模式
        
        Args:
            metric_name: 指标名称
            time_range: 时间范围（天）
            
        Returns:
            模式识别结果
        """
        try:
            data = self._get_metric_data(metric_name, time_range)
            if len(data) < 5:
                return {"error": "数据不足，无法识别模式"}
            
            values = [item['value'] for item in data]
            timestamps = [datetime.fromisoformat(item['timestamp']) for item in data]
            
            # 识别各种模式
            patterns = {
                "linear": self._detect_linear_pattern(values),
                "exponential": self._detect_exponential_pattern(values),
                "cyclical": self._detect_cyclical_pattern(values),
                "step": self._detect_step_pattern(values),
                "plateau": self._detect_plateau_pattern(values),
                "decline": self._detect_decline_pattern(values)
            }
            
            # 确定主导模式
            dominant_pattern = self._determine_dominant_pattern(patterns)
            
            # 模式特征分析
            pattern_features = self._analyze_pattern_features(values, dominant_pattern)
            
            pattern_analysis = {
                "metric_name": metric_name,
                "time_range": time_range,
                "patterns_detected": patterns,
                "dominant_pattern": dominant_pattern,
                "pattern_features": pattern_features,
                "confidence": self._calculate_pattern_confidence(values, patterns),
                "recommendations": self._generate_pattern_recommendations(dominant_pattern),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"完成{metric_name}的进化模式识别")
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"模式识别失败: {str(e)}")
            return {"error": f"模式识别失败: {str(e)}"}
    
    def _detect_linear_pattern(self, values: List[float]) -> Dict[str, Any]:
        """检测线性模式"""
        if len(values) < 3:
            return {"detected": False, "strength": 0.0}
        
        x = np.arange(len(values))
        correlation = np.corrcoef(x, values)[0, 1]
        strength = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return {
            "detected": strength > 0.7,
            "strength": strength,
            "slope": self._calculate_slope(values)
        }
    
    def _detect_exponential_pattern(self, values: List[float]) -> Dict[str, Any]:
        """检测指数模式"""
        if len(values) < 4 or any(v <= 0 for v in values):
            return {"detected": False, "strength": 0.0}
        
        try:
            log_values = [np.log(v) for v in values]
            x = np.arange(len(log_values))
            correlation = np.corrcoef(x, log_values)[0, 1]
            strength = abs(correlation) if not np.isnan(correlation) else 0.0
            
            return {
                "detected": strength > 0.7,
                "strength": strength,
                "growth_factor": np.exp(self._calculate_slope(log_values))
            }
        except:
            return {"detected": False, "strength": 0.0}
    
    def _detect_cyclical_pattern(self, values: List[float]) -> Dict[str, Any]:
        """检测周期性模式"""
        if len(values) < 8:
            return {"detected": False, "strength": 0.0}
        
        # 使用FFT检测周期性
        try:
            fft = np.fft.fft(values)
            freqs = np.fft.fftfreq(len(values))
            power_spectrum = np.abs(fft) ** 2
            
            # 找到主要频率
            max_freq_idx = np.argmax(power_spectrum[1:len(values)//2]) + 1
            dominant_freq = freqs[max_freq_idx]
            strength = power_spectrum[max_freq_idx] / np.sum(power_spectrum)
            
            return {
                "detected": strength > 0.3,
                "strength": strength,
                "period": 1 / abs(dominant_freq) if dominant_freq != 0 else len(values),
                "frequency": dominant_freq
            }
        except:
            return {"detected": False, "strength": 0.0}
    
    def _detect_step_pattern(self, values: List[float]) -> Dict[str, Any]:
        """检测阶梯式模式"""
        if len(values) < 5:
            return {"detected": False, "strength": 0.0}
        
        # 计算相邻差值的变化
        differences = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        # 寻找显著的变化点
        threshold = np.std(differences) * 2
        step_points = []
        
        for i, diff in enumerate(differences):
            if abs(diff) > threshold:
                step_points.append(i)
        
        strength = len(step_points) / len(differences) if differences else 0
        
        return {
            "detected": len(step_points) >= 2 and strength > 0.2,
            "strength": strength,
            "step_points": step_points
        }
    
    def _detect_plateau_pattern(self, values: List[float]) -> Dict[str, Any]:
        """检测平台期模式"""
        if len(values) < 5:
            return {"detected": False, "strength": 0.0}
        
        # 计算移动平均和标准差
        window_size = min(5, len(values) // 2)
        if window_size < 2:
            return {"detected": False, "strength": 0.0}
        
        moving_avg = []
        moving_std = []
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            moving_avg.append(np.mean(window))
            moving_std.append(np.std(window))
        
        # 平台期特征：移动标准差较小且移动平均变化缓慢
        avg_std = np.mean(moving_std) if moving_std else float('inf')
        avg_change = np.mean([abs(moving_avg[i+1] - moving_avg[i]) 
                             for i in range(len(moving_avg)-1)])
        
        stability_score = 1 / (1 + avg_std)  # 标准差越小，分数越高
        consistency_score = 1 / (1 + avg_change)  # 变化越小，分数越高
        
        strength = (stability_score + consistency_score) / 2
        
        return {
            "detected": strength > 0.6,
            "strength": strength,
            "stability_score": stability_score,
            "consistency_score": consistency_score
        }
    
    def _detect_decline_pattern(self, values: List[float]) -> Dict[str, Any]:
        """检测下降模式"""
        if len(values) < 3:
            return {"detected": False, "strength": 0.0}
        
        slope = self._calculate_slope(values)
        trend_strength = self._calculate_trend_strength(values)
        
        return {
            "detected": slope < -0.01 and trend_strength > 0.5,
            "strength": trend_strength,
            "decline_rate": abs(slope)
        }
    
    def _determine_dominant_pattern(self, patterns: Dict[str, Any]) -> str:
        """确定主导模式"""
        max_strength = 0
        dominant = "unknown"
        
        for pattern_name, pattern_info in patterns.items():
            if pattern_info.get("detected", False):
                strength = pattern_info.get("strength", 0)
                if strength > max_strength:
                    max_strength = strength
                    dominant = pattern_name
        
        return dominant
    
    def _analyze_pattern_features(self, values: List[float], pattern: str) -> Dict[str, Any]:
        """分析模式特征"""
        features = {
            "data_points": len(values),
            "value_range": max(values) - min(values),
            "mean": np.mean(values),
            "trend": self._interpret_slope(self._calculate_slope(values))
        }
        
        if pattern == "cyclical":
            # 周期性模式额外特征
            features["amplitude"] = np.std(values)
            features["frequency"] = "detected"
        elif pattern == "exponential":
            # 指数模式额外特征
            features["growth_acceleration"] = self._calculate_growth_rate(values)
        elif pattern == "step":
            # 阶梯模式额外特征
            features["step_magnitude"] = np.mean([abs(values[i+1] - values[i]) 
                                                 for i in range(len(values)-1)])
        
        return features
    
    def _calculate_pattern_confidence(self, values: List[float], patterns: Dict[str, Any]) -> float:
        """计算模式识别置信度"""
        detected_patterns = sum(1 for p in patterns.values() if p.get("detected", False))
        total_patterns = len(patterns)
        
        if detected_patterns == 0:
            return 0.1  # 未检测到任何模式
        
        # 置信度基于检测到的模式数量和强度
        avg_strength = np.mean([p.get("strength", 0) for p in patterns.values() 
                               if p.get("detected", False)])
        
        confidence = (detected_patterns / total_patterns) * avg_strength
        return min(0.95, max(0.1, confidence))
    
    def _generate_pattern_recommendations(self, pattern: str) -> List[str]:
        """生成基于模式的建议"""
        recommendations = {
            "linear": [
                "继续保持当前的增长节奏",
                "考虑设定更具体的目标",
                "监控外部因素影响"
            ],
            "exponential": [
                "准备应对快速增长带来的挑战",
                "优化资源分配",
                "建立可持续的增长机制"
            ],
            "cyclical": [
                "识别周期性规律的原因",
                "在低潮期做好储备",
                "在高潮期抓住机会"
            ],
            "step": [
                "分析每次跳跃的原因",
                "寻找触发下一次跳跃的条件",
                "确保跳跃的可持续性"
            ],
            "plateau": [
                "寻找突破平台期的新方法",
                "重新评估目标和策略",
                "考虑引入新的增长动力"
            ],
            "decline": [
                "立即分析下降原因",
                "制定扭转趋势的行动计划",
                "加强监控和预警机制"
            ]
        }
        
        return recommendations.get(pattern, ["继续监控和分析"])
    
    # ==================== 3. 性能进化效果评估 ====================
    
    def evaluate_evolution_effectiveness(self, 
                                       metric_names: List[str],
                                       baseline_period: int = 30,
                                       evaluation_period: int = 30) -> Dict[str, Any]:
        """
        评估进化效果
        
        Args:
            metric_names: 指标名称列表
            baseline_period: 基线期（天）
            evaluation_period: 评估期（天）
            
        Returns:
            效果评估结果
        """
        try:
            evaluation_results = {}
            
            for metric_name in metric_names:
                # 获取基线期和评估期数据
                baseline_data = self._get_metric_data(metric_name, baseline_period)
                evaluation_data = self._get_metric_data(metric_name, evaluation_period)
                
                if len(baseline_data) < 2 or len(evaluation_data) < 2:
                    evaluation_results[metric_name] = {
                        "status": "insufficient_data",
                        "message": "数据不足，无法评估"
                    }
                    continue
                
                # 计算各阶段统计指标
                baseline_stats = self._calculate_statistics([d['value'] for d in baseline_data])
                evaluation_stats = self._calculate_statistics([d['value'] for d in evaluation_data])
                
                # 计算改进指标
                improvement_metrics = self._calculate_improvement_metrics(
                    baseline_stats, evaluation_stats)
                
                # 效果评级
                effectiveness_rating = self._rate_effectiveness(improvement_metrics)
                
                evaluation_results[metric_name] = {
                    "baseline_period": baseline_period,
                    "evaluation_period": evaluation_period,
                    "baseline_stats": baseline_stats,
                    "evaluation_stats": evaluation_stats,
                    "improvement_metrics": improvement_metrics,
                    "effectiveness_rating": effectiveness_rating,
                    "key_insights": self._generate_effectiveness_insights(improvement_metrics)
                }
            
            # 综合评估
            overall_assessment = self._calculate_overall_assessment(evaluation_results)
            
            effectiveness_report = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "metrics_evaluated": len(metric_names),
                "individual_results": evaluation_results,
                "overall_assessment": overall_assessment,
                "recommendations": self._generate_effectiveness_recommendations(overall_assessment)
            }
            
            logger.info(f"完成{len(metric_names)}个指标的效果评估")
            return effectiveness_report
            
        except Exception as e:
            logger.error(f"效果评估失败: {str(e)}")
            return {"error": f"效果评估失败: {str(e)}"}
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """计算统计指标"""
        if not values:
            return {}
        
        return {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "range": np.max(values) - np.min(values),
            "cv": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        }
    
    def _calculate_improvement_metrics(self, 
                                     baseline: Dict[str, float], 
                                     evaluation: Dict[str, float]) -> Dict[str, float]:
        """计算改进指标"""
        improvements = {}
        
        for key in baseline:
            if key in evaluation and baseline[key] != 0:
                improvement = (evaluation[key] - baseline[key]) / baseline[key] * 100
                improvements[f"{key}_improvement_pct"] = improvement
        
        # 特殊指标计算
        if "mean" in baseline and "mean" in evaluation:
            improvements["absolute_improvement"] = evaluation["mean"] - baseline["mean"]
        
        if "cv" in baseline and "cv" in evaluation:
            # 变异系数改善（越小越好）
            cv_improvement = (baseline["cv"] - evaluation["cv"]) / baseline["cv"] * 100
            improvements["stability_improvement_pct"] = cv_improvement
        
        return improvements
    
    def _rate_effectiveness(self, improvement_metrics: Dict[str, float]) -> Dict[str, Any]:
        """评级效果"""
        # 综合评分算法
        scores = []
        
        for metric, value in improvement_metrics.items():
            if "improvement_pct" in metric:
                # 对于改进百分比，正值表示改善
                if "stability" in metric:
                    # 稳定性改善，越大越好
                    score = min(100, max(0, 50 + value))
                else:
                    # 一般改进指标
                    score = min(100, max(0, 50 + value/2))
                scores.append(score)
        
        if not scores:
            return {"rating": "无法评级", "score": 0, "reason": "无可用指标"}
        
        overall_score = np.mean(scores)
        
        if overall_score >= 80:
            rating = "优秀"
        elif overall_score >= 65:
            rating = "良好"
        elif overall_score >= 50:
            rating = "一般"
        elif overall_score >= 35:
            rating = "较差"
        else:
            rating = "很差"
        
        return {
            "rating": rating,
            "score": overall_score,
            "grade": self._score_to_grade(overall_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """将分数转换为等级"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"
    
    def _generate_effectiveness_insights(self, improvement_metrics: Dict[str, float]) -> List[str]:
        """生成效果洞察"""
        insights = []
        
        for metric, value in improvement_metrics.items():
            if "improvement_pct" in metric:
                if value > 10:
                    insights.append(f"{metric.replace('_improvement_pct', '')}显著改善 ({value:.1f}%)")
                elif value > 0:
                    insights.append(f"{metric.replace('_improvement_pct', '')}有所改善 ({value:.1f}%)")
                elif value < -10:
                    insights.append(f"{metric.replace('_improvement_pct', '')}显著恶化 ({value:.1f}%)")
                else:
                    insights.append(f"{metric.replace('_improvement_pct', '')}基本稳定")
        
        return insights
    
    def _calculate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算综合评估"""
        scores = []
        ratings = []
        
        for metric, result in results.items():
            if "effectiveness_rating" in result:
                score = result["effectiveness_rating"].get("score", 0)
                rating = result["effectiveness_rating"].get("rating", "无法评级")
                scores.append(score)
                ratings.append(rating)
        
        if not scores:
            return {"status": "no_data"}
        
        overall_score = np.mean(scores)
        
        # 统计各等级数量
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        return {
            "overall_score": overall_score,
            "overall_rating": self._score_to_grade(overall_score),
            "rating_distribution": rating_counts,
            "metrics_improved": sum(1 for s in scores if s > 50),
            "metrics_total": len(scores)
        }
    
    def _generate_effectiveness_recommendations(self, overall: Dict[str, Any]) -> List[str]:
        """生成效果改进建议"""
        recommendations = []
        
        if overall.get("overall_score", 0) < 50:
            recommendations.append("整体效果不佳，需要重新审视策略和方法")
            recommendations.append("加强数据监控和分析，及时调整改进方向")
        
        if overall.get("metrics_improved", 0) < overall.get("metrics_total", 1) / 2:
            recommendations.append("超过半数指标未改善，需要重点关注这些指标")
        
        if overall.get("overall_score", 0) >= 80:
            recommendations.append("效果优秀，可以考虑将成功经验推广到其他领域")
        elif overall.get("overall_score", 0) >= 60:
            recommendations.append("效果良好，可以进一步优化以获得更好结果")
        
        return recommendations
    
    # ==================== 4. 性能进化优化建议 ====================
    
    def generate_optimization_suggestions(self, 
                                        metric_names: List[str],
                                        current_trends: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成优化建议
        
        Args:
            metric_names: 指标名称列表
            current_trends: 当前趋势数据
            
        Returns:
            优化建议
        """
        try:
            suggestions = []
            
            for metric_name in metric_names:
                # 获取当前状态
                current_analysis = self.analyze_evolution_trends(metric_name)
                pattern_analysis = self.identify_evolution_patterns(metric_name)
                
                if "error" in current_analysis or "error" in pattern_analysis:
                    continue
                
                # 基于趋势生成建议
                trend_suggestions = self._generate_trend_based_suggestions(
                    metric_name, current_analysis)
                
                # 基于模式生成建议
                pattern_suggestions = self._generate_pattern_based_suggestions(
                    metric_name, pattern_analysis)
                
                # 基于性能差距生成建议
                gap_suggestions = self._generate_gap_based_suggestions(
                    metric_name, current_analysis)
                
                metric_suggestions = trend_suggestions + pattern_suggestions + gap_suggestions
                suggestions.extend(metric_suggestions)
            
            # 去重和优先级排序
            unique_suggestions = self._deduplicate_suggestions(suggestions)
            prioritized_suggestions = self._prioritize_suggestions(unique_suggestions)
            
            # 分类建议
            categorized_suggestions = self._categorize_suggestions(prioritized_suggestions)
            
            optimization_report = {
                "generation_timestamp": datetime.now().isoformat(),
                "metrics_analyzed": len(metric_names),
                "total_suggestions": len(prioritized_suggestions),
                "suggestions_by_category": categorized_suggestions,
                "priority_suggestions": prioritized_suggestions[:10],  # 前10个高优先级建议
                "implementation_roadmap": self._create_implementation_roadmap(prioritized_suggestions)
            }
            
            logger.info(f"生成了{len(prioritized_suggestions)}条优化建议")
            return optimization_report
            
        except Exception as e:
            logger.error(f"优化建议生成失败: {str(e)}")
            return {"error": f"优化建议生成失败: {str(e)}"}
    
    def _generate_trend_based_suggestions(self, metric_name: str, trend_analysis: Dict[str, Any]) -> List[OptimizationSuggestion]:
        """基于趋势生成建议"""
        suggestions = []
        
        trend_info = trend_analysis.get("trend_analysis", {})
        slope = trend_info.get("slope", 0)
        growth_rate = trend_info.get("growth_rate", 0)
        
        if slope < -0.01:  # 下降趋势
            suggestions.append(OptimizationSuggestion(
                category="趋势扭转",
                suggestion=f"{metric_name}呈现下降趋势，建议立即分析原因并制定扭转策略",
                priority=9,
                expected_improvement=15.0,
                implementation_cost=5.0,
                timeframe="1-2周"
            ))
        elif slope > 0.1:  # 快速增长
            suggestions.append(OptimizationSuggestion(
                category="增长优化",
                suggestion=f"{metric_name}增长较快，建议优化资源配置以支持持续增长",
                priority=7,
                expected_improvement=10.0,
                implementation_cost=3.0,
                timeframe="2-4周"
            ))
        
        if abs(growth_rate) < 5:  # 增长缓慢
            suggestions.append(OptimizationSuggestion(
                category="增长加速",
                suggestion=f"{metric_name}增长缓慢，建议探索新的增长动力",
                priority=6,
                expected_improvement=12.0,
                implementation_cost=4.0,
                timeframe="3-6周"
            ))
        
        return suggestions
    
    def _generate_pattern_based_suggestions(self, metric_name: str, pattern_analysis: Dict[str, Any]) -> List[OptimizationSuggestion]:
        """基于模式生成建议"""
        suggestions = []
        
        dominant_pattern = pattern_analysis.get("dominant_pattern", "unknown")
        recommendations = pattern_analysis.get("recommendations", [])
        
        for rec in recommendations:
            suggestions.append(OptimizationSuggestion(
                category="模式优化",
                suggestion=f"{metric_name}: {rec}",
                priority=5,
                expected_improvement=8.0,
                implementation_cost=2.0,
                timeframe="2-8周"
            ))
        
        return suggestions
    
    def _generate_gap_based_suggestions(self, metric_name: str, trend_analysis: Dict[str, Any]) -> List[OptimizationSuggestion]:
        """基于性能差距生成建议"""
        suggestions = []
        
        stats = trend_analysis.get("statistics", {})
        prediction = trend_analysis.get("prediction", {})
        
        current_value = stats.get("mean", 0)
        predicted_value = prediction.get("next_7_days", [current_value])[-1]
        
        # 如果预测值低于当前值，建议改进
        if predicted_value < current_value * 0.95:
            suggestions.append(OptimizationSuggestion(
                category="预测改进",
                suggestion=f"{metric_name}预测显示可能下降，建议提前采取预防措施",
                priority=8,
                expected_improvement=10.0,
                implementation_cost=3.0,
                timeframe="1-3周"
            ))
        
        return suggestions
    
    def _deduplicate_suggestions(self, suggestions: List[OptimizationSuggestion]) -> List[OptimizationSuggestion]:
        """去重建议"""
        seen = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            key = (suggestion.category, suggestion.suggestion)
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _prioritize_suggestions(self, suggestions: List[OptimizationSuggestion]) -> List[OptimizationSuggestion]:
        """优先级排序"""
        # 计算综合优先级分数
        for suggestion in suggestions:
            # 优先级分数 = 优先级 * (预期改进 / 实现成本) * 时间因子
            time_factor = {"1-2周": 1.2, "2-4周": 1.0, "3-6周": 0.8, "2-8周": 0.6}.get(suggestion.timeframe, 1.0)
            cost_benefit = suggestion.expected_improvement / max(suggestion.implementation_cost, 0.1)
            suggestion.priority_score = suggestion.priority * cost_benefit * time_factor
        
        return sorted(suggestions, key=lambda x: x.priority_score, reverse=True)
    
    def _categorize_suggestions(self, suggestions: List[OptimizationSuggestion]) -> Dict[str, List[OptimizationSuggestion]]:
        """分类建议"""
        categories = {}
        
        for suggestion in suggestions:
            category = suggestion.category
            if category not in categories:
                categories[category] = []
            categories[category].append(suggestion)
        
        return categories
    
    def _create_implementation_roadmap(self, suggestions: List[OptimizationSuggestion]) -> Dict[str, List[str]]:
        """创建实施路线图"""
        roadmap = {
            "immediate": [],  # 立即执行
            "short_term": [],  # 短期执行
            "medium_term": [],  # 中期执行
            "long_term": []  # 长期执行
        }
        
        for suggestion in suggestions:
            if suggestion.timeframe in ["1-2周"]:
                roadmap["immediate"].append(suggestion.suggestion)
            elif suggestion.timeframe in ["2-4周"]:
                roadmap["short_term"].append(suggestion.suggestion)
            elif suggestion.timeframe in ["3-6周"]:
                roadmap["medium_term"].append(suggestion.suggestion)
            else:
                roadmap["long_term"].append(suggestion.suggestion)
        
        return roadmap
    
    # ==================== 5. 性能进化风险控制 ====================
    
    def assess_evolution_risks(self, metric_names: List[str]) -> Dict[str, Any]:
        """
        评估进化风险
        
        Args:
            metric_names: 指标名称列表
            
        Returns:
            风险评估结果
        """
        try:
            risk_assessments = []
            
            for metric_name in metric_names:
                # 获取趋势分析
                trend_analysis = self.analyze_evolution_trends(metric_name)
                
                if "error" in trend_analysis:
                    continue
                
                # 识别各种风险
                trend_risks = self._identify_trend_risks(metric_name, trend_analysis)
                volatility_risks = self._identify_volatility_risks(metric_name, trend_analysis)
                prediction_risks = self._identify_prediction_risks(metric_name, trend_analysis)
                
                metric_risks = trend_risks + volatility_risks + prediction_risks
                risk_assessments.extend(metric_risks)
            
            # 风险聚合和分析
            risk_summary = self._aggregate_risk_analysis(risk_assessments)
            
            # 生成风险控制策略
            control_strategies = self._generate_risk_control_strategies(risk_assessments)
            
            risk_report = {
                "assessment_timestamp": datetime.now().isoformat(),
                "metrics_assessed": len(metric_names),
                "total_risks_identified": len(risk_assessments),
                "risk_summary": risk_summary,
                "detailed_assessments": risk_assessments,
                "control_strategies": control_strategies,
                "risk_monitoring_plan": self._create_risk_monitoring_plan(risk_assessments)
            }
            
            # 保存风险评估到数据库
            self._save_risk_assessments(risk_assessments)
            
            logger.info(f"识别了{len(risk_assessments)}个风险点")
            return risk_report
            
        except Exception as e:
            logger.error(f"风险评估失败: {str(e)}")
            return {"error": f"风险评估失败: {str(e)}"}
    
    def _identify_trend_risks(self, metric_name: str, trend_analysis: Dict[str, Any]) -> List[RiskAssessment]:
        """识别趋势风险"""
        risks = []
        
        trend_info = trend_analysis.get("trend_analysis", {})
        slope = trend_info.get("slope", 0)
        growth_rate = trend_info.get("growth_rate", 0)
        
        # 下降趋势风险
        if slope < -0.05:
            risks.append(RiskAssessment(
                risk_type="趋势逆转风险",
                level=RiskLevel.HIGH,
                probability=0.8,
                impact=0.7,
                mitigation_strategy="立即分析下降原因，制定扭转策略",
                timestamp=datetime.now()
            ))
        
        # 快速增长风险
        elif slope > 0.2:
            risks.append(RiskAssessment(
                risk_type="增长失控风险",
                level=RiskLevel.MEDIUM,
                probability=0.6,
                impact=0.5,
                mitigation_strategy="建立增长监控机制，防止过度增长",
                timestamp=datetime.now()
            ))
        
        return risks
    
    def _identify_volatility_risks(self, metric_name: str, trend_analysis: Dict[str, Any]) -> List[RiskAssessment]:
        """识别波动性风险"""
        risks = []
        
        trend_info = trend_analysis.get("trend_analysis", {})
        volatility = trend_info.get("volatility", 0)
        
        if volatility > 50:  # 高波动性
            risks.append(RiskAssessment(
                risk_type="高波动性风险",
                level=RiskLevel.MEDIUM,
                probability=0.7,
                impact=0.6,
                mitigation_strategy="建立稳定机制，减少波动性",
                timestamp=datetime.now()
            ))
        
        return risks
    
    def _identify_prediction_risks(self, metric_name: str, trend_analysis: Dict[str, Any]) -> List[RiskAssessment]:
        """识别预测风险"""
        risks = []
        
        prediction = trend_analysis.get("prediction", {})
        confidence = prediction.get("confidence", 1.0)
        
        if confidence < 0.5:  # 低预测置信度
            risks.append(RiskAssessment(
                risk_type="预测不确定性风险",
                level=RiskLevel.MEDIUM,
                probability=0.8,
                impact=0.4,
                mitigation_strategy="增加数据收集频率，提高预测准确性",
                timestamp=datetime.now()
            ))
        
        return risks
    
    def _aggregate_risk_analysis(self, risk_assessments: List[RiskAssessment]) -> Dict[str, Any]:
        """聚合风险分析"""
        if not risk_assessments:
            return {"status": "no_risks"}
        
        # 按风险等级统计
        risk_levels = {}
        for risk in risk_assessments:
            level = risk.level.value
            risk_levels[level] = risk_levels.get(level, 0) + 1
        
        # 计算整体风险分数
        total_impact = sum(risk.impact * risk.probability for risk in risk_assessments)
        max_possible_impact = len(risk_assessments)
        risk_score = (total_impact / max_possible_impact) * 100 if max_possible_impact > 0 else 0
        
        # 风险等级评估
        if risk_score >= 70:
            overall_level = RiskLevel.CRITICAL
        elif risk_score >= 50:
            overall_level = RiskLevel.HIGH
        elif risk_score >= 30:
            overall_level = RiskLevel.MEDIUM
        else:
            overall_level = RiskLevel.LOW
        
        return {
            "overall_risk_score": risk_score,
            "overall_risk_level": overall_level.value,
            "risk_level_distribution": risk_levels,
            "total_risks": len(risk_assessments),
            "high_priority_risks": sum(1 for r in risk_assessments if r.level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        }
    
    def _generate_risk_control_strategies(self, risk_assessments: List[RiskAssessment]) -> List[Dict[str, Any]]:
        """生成风险控制策略"""
        strategies = []
        
        # 按风险类型分组
        risk_groups = {}
        for risk in risk_assessments:
            risk_type = risk.risk_type
            if risk_type not in risk_groups:
                risk_groups[risk_type] = []
            risk_groups[risk_type].append(risk)
        
        for risk_type, risks in risk_groups.items():
            # 选择最严重的风险作为代表
            worst_risk = max(risks, key=lambda x: x.impact * x.probability)
            
            strategies.append({
                "risk_type": risk_type,
                "strategy": worst_risk.mitigation_strategy,
                "priority": worst_risk.level.value,
                "expected_effectiveness": self._estimate_strategy_effectiveness(worst_risk),
                "implementation_timeline": self._suggest_implementation_timeline(worst_risk.level)
            })
        
        return strategies
    
    def _estimate_strategy_effectiveness(self, risk: RiskAssessment) -> str:
        """估算策略有效性"""
        risk_score = risk.impact * risk.probability
        
        if risk_score >= 0.7:
            return "高"
        elif risk_score >= 0.4:
            return "中"
        else:
            return "低"
    
    def _suggest_implementation_timeline(self, risk_level: RiskLevel) -> str:
        """建议实施时间线"""
        timelines = {
            RiskLevel.CRITICAL: "立即",
            RiskLevel.HIGH: "1周内",
            RiskLevel.MEDIUM: "2-4周",
            RiskLevel.LOW: "1-3个月"
        }
        return timelines.get(risk_level, "待评估")
    
    def _create_risk_monitoring_plan(self, risk_assessments: List[RiskAssessment]) -> Dict[str, Any]:
        """创建风险监控计划"""
        plan = {
            "monitoring_frequency": {},
            "key_indicators": [],
            "alert_thresholds": {},
            "review_schedule": "每周"
        }
        
        # 为每个风险等级设定监控频率
        for risk in risk_assessments:
            level = risk.level
            if level == RiskLevel.CRITICAL:
                plan["monitoring_frequency"][risk.risk_type] = "每日"
            elif level == RiskLevel.HIGH:
                plan["monitoring_frequency"][risk.risk_type] = "每3日"
            elif level == RiskLevel.MEDIUM:
                plan["monitoring_frequency"][risk.risk_type] = "每周"
            else:
                plan["monitoring_frequency"][risk.risk_type] = "每月"
        
        return plan
    
    def _save_risk_assessments(self, risk_assessments: List[RiskAssessment]):
        """保存风险评估到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for risk in risk_assessments:
            cursor.execute('''
                INSERT INTO risk_assessments 
                (timestamp, risk_type, level, probability, impact, mitigation_strategy)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                risk.timestamp.isoformat(),
                risk.risk_type,
                risk.level.value,
                risk.probability,
                risk.impact,
                risk.mitigation_strategy
            ))
        
        conn.commit()
        conn.close()
    
    # ==================== 6. 性能进化历史跟踪 ====================
    
    def track_evolution_history(self, 
                              metric_names: List[str],
                              time_range: int = 90) -> Dict[str, Any]:
        """
        跟踪进化历史
        
        Args:
            metric_names: 指标名称列表
            time_range: 时间范围（天）
            
        Returns:
            历史跟踪结果
        """
        try:
            historical_data = {}
            
            for metric_name in metric_names:
                # 获取历史数据
                data = self._get_metric_data(metric_name, time_range)
                
                if len(data) < 5:
                    historical_data[metric_name] = {
                        "status": "insufficient_data",
                        "data_points": len(data)
                    }
                    continue
                
                # 分析历史阶段
                phases = self._analyze_evolution_phases(data)
                
                # 识别关键事件
                key_events = self._identify_key_events(data)
                
                # 计算历史性能分数
                performance_score = self._calculate_historical_performance_score(data)
                
                historical_data[metric_name] = {
                    "time_range": time_range,
                    "data_points": len(data),
                    "evolution_phases": phases,
                    "key_events": key_events,
                    "performance_score": performance_score,
                    "trend_summary": self._generate_trend_summary(data)
                }
            
            # 生成综合历史报告
            overall_history = self._generate_overall_history_summary(historical_data)
            
            history_report = {
                "tracking_timestamp": datetime.now().isoformat(),
                "metrics_tracked": len(metric_names),
                "time_range": time_range,
                "individual_histories": historical_data,
                "overall_summary": overall_history,
                "historical_insights": self._generate_historical_insights(historical_data)
            }
            
            # 保存历史记录
            self._save_evolution_history(history_report)
            
            logger.info(f"跟踪了{len(metric_names)}个指标的历史进化")
            return history_report
            
        except Exception as e:
            logger.error(f"历史跟踪失败: {str(e)}")
            return {"error": f"历史跟踪失败: {str(e)}"}
    
    def _analyze_evolution_phases(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """分析进化阶段"""
        if len(data) < 10:
            return [{"phase": "数据不足", "description": "历史数据不足以分析阶段"}]
        
        values = [item['value'] for item in data]
        timestamps = [datetime.fromisoformat(item['timestamp']) for item in data]
        
        # 将数据分为几个时间段进行分析
        segment_size = max(3, len(values) // 5)  # 分为5个阶段
        phases = []
        
        for i in range(0, len(values), segment_size):
            segment = values[i:i+segment_size]
            segment_timestamps = timestamps[i:i+segment_size]
            
            if len(segment) < 2:
                continue
            
            # 分析该阶段的特征
            phase_slope = self._calculate_slope(segment)
            phase_growth = self._calculate_growth_rate(segment)
            phase_volatility = self._calculate_volatility(segment)
            
            # 确定阶段类型
            phase_type = self._classify_evolution_phase(phase_slope, phase_growth, phase_volatility)
            
            phases.append({
                "start_date": segment_timestamps[0].isoformat(),
                "end_date": segment_timestamps[-1].isoformat(),
                "phase_type": phase_type.value,
                "slope": phase_slope,
                "growth_rate": phase_growth,
                "volatility": phase_volatility,
                "description": self._describe_phase(phase_type, phase_slope, phase_growth)
            })
        
        return phases
    
    def _classify_evolution_phase(self, slope: float, growth: float, volatility: float) -> EvolutionPhase:
        """分类进化阶段"""
        if volatility > 30:
            return EvolutionPhase.TRANSFORMATION
        elif slope < -0.05:
            return EvolutionPhase.DECLINE
        elif slope > 0.1 and growth > 20:
            return EvolutionPhase.GROWTH
        elif abs(slope) < 0.01 and growth < 5:
            return EvolutionPhase.MATURE
        elif slope > 0.01:
            return EvolutionPhase.OPTIMIZATION
        else:
            return EvolutionPhase.INITIAL
    
    def _describe_phase(self, phase: EvolutionPhase, slope: float, growth: float) -> str:
        """描述阶段特征"""
        descriptions = {
            EvolutionPhase.INITIAL: "系统初始化阶段，趋势相对平稳",
            EvolutionPhase.GROWTH: f"快速增长阶段，增长率{growth:.1f}%",
            EvolutionPhase.MATURE: "成熟稳定阶段，变化较小",
            EvolutionPhase.OPTIMIZATION: f"优化调整阶段，斜率{slope:.3f}",
            EvolutionPhase.TRANSFORMATION: "转型变化阶段，波动较大",
            EvolutionPhase.DECLINE: f"下降阶段，斜率{slope:.3f}"
        }
        return descriptions.get(phase, "未知阶段")
    
    def _identify_key_events(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """识别关键事件"""
        if len(data) < 5:
            return []
        
        values = [item['value'] for item in data]
        timestamps = [datetime.fromisoformat(item['timestamp']) for item in data]
        
        # 计算变化点
        changes = [values[i+1] - values[i] for i in range(len(values)-1)]
        threshold = np.std(changes) * 2
        
        key_events = []
        
        for i, change in enumerate(changes):
            if abs(change) > threshold:
                event_type = "显著上升" if change > 0 else "显著下降"
                impact_score = min(10, abs(change) / np.std(values) * 5)
                
                key_events.append({
                    "timestamp": timestamps[i+1].isoformat(),
                    "event_type": event_type,
                    "description": f"{event_type}事件，变化幅度{change:.2f}",
                    "impact_score": impact_score,
                    "category": "performance_change"
                })
        
        return key_events
    
    def _calculate_historical_performance_score(self, data: List[Dict]) -> Dict[str, float]:
        """计算历史性能分数"""
        values = [item['value'] for item in data]
        
        if not values:
            return {"overall_score": 0}
        
        # 趋势分数
        slope = self._calculate_slope(values)
        trend_score = max(0, min(100, 50 + slope * 100))
        
        # 稳定性分数
        volatility = self._calculate_volatility(values)
        stability_score = max(0, min(100, 100 - volatility))
        
        # 增长分数
        growth_rate = self._calculate_growth_rate(values)
        growth_score = max(0, min(100, 50 + growth_rate))
        
        # 综合分数
        overall_score = (trend_score + stability_score + growth_score) / 3
        
        return {
            "trend_score": trend_score,
            "stability_score": stability_score,
            "growth_score": growth_score,
            "overall_score": overall_score
        }
    
    def _generate_trend_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """生成趋势摘要"""
        values = [item['value'] for item in data]
        
        if len(values) < 2:
            return {"summary": "数据不足"}
        
        start_value = values[0]
        end_value = values[-1]
        total_change = (end_value - start_value) / start_value * 100
        
        direction = "上升" if total_change > 0 else "下降" if total_change < 0 else "平稳"
        
        return {
            "start_value": start_value,
            "end_value": end_value,
            "total_change_pct": total_change,
            "direction": direction,
            "summary": f"总体{direction}趋势，变化{total_change:.1f}%"
        }
    
    def _generate_overall_history_summary(self, histories: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合历史摘要"""
        if not histories:
            return {"status": "no_data"}
        
        total_metrics = len(histories)
        improving_metrics = 0
        declining_metrics = 0
        stable_metrics = 0
        
        for metric_name, history in histories.items():
            if "trend_summary" in history:
                direction = history["trend_summary"].get("direction", "未知")
                if direction == "上升":
                    improving_metrics += 1
                elif direction == "下降":
                    declining_metrics += 1
                else:
                    stable_metrics += 1
        
        return {
            "total_metrics": total_metrics,
            "improving_metrics": improving_metrics,
            "declining_metrics": declining_metrics,
            "stable_metrics": stable_metrics,
            "improvement_rate": improving_metrics / total_metrics * 100 if total_metrics > 0 else 0,
            "overall_health": "良好" if improving_metrics > declining_metrics else "需要关注"
        }
    
    def _generate_historical_insights(self, histories: Dict[str, Any]) -> List[str]:
        """生成历史洞察"""
        insights = []
        
        overall = self._generate_overall_history_summary(histories)
        
        improvement_rate = overall.get("improvement_rate", 0)
        if improvement_rate > 70:
            insights.append("大多数指标呈现改善趋势，整体表现优秀")
        elif improvement_rate > 50:
            insights.append("超过半数指标有所改善，整体趋势向好")
        elif improvement_rate > 30:
            insights.append("部分指标改善，但整体提升空间较大")
        else:
            insights.append("改善指标较少，需要重点关注和改善")
        
        # 分析阶段分布
        phase_distribution = {}
        for history in histories.values():
            if "evolution_phases" in history:
                for phase in history["evolution_phases"]:
                    phase_type = phase.get("phase_type", "未知")
                    phase_distribution[phase_type] = phase_distribution.get(phase_type, 0) + 1
        
        if phase_distribution:
            dominant_phase = max(phase_distribution, key=phase_distribution.get)
            insights.append(f"历史进化以{dominant_phase}为主")
        
        return insights
    
    def _save_evolution_history(self, history_report: Dict[str, Any]):
        """保存进化历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, history in history_report["individual_histories"].items():
            if "evolution_phases" in history:
                for phase in history["evolution_phases"]:
                    cursor.execute('''
                        INSERT INTO evolution_history 
                        (timestamp, phase, pattern, metrics_summary, key_events, performance_score)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        datetime.now().isoformat(),
                        phase.get("phase_type", ""),
                        "detected",
                        json.dumps({"metric": metric_name, "phase_data": phase}),
                        json.dumps(history.get("key_events", [])),
                        history.get("performance_score", {}).get("overall_score", 0)
                    ))
        
        conn.commit()
        conn.close()
    
    # ==================== 7. 性能进化报告生成 ====================
    
    def generate_evolution_report(self, 
                                metric_names: List[str],
                                report_type: str = "comprehensive",
                                time_range: int = 30) -> Dict[str, Any]:
        """
        生成进化报告
        
        Args:
            metric_names: 指标名称列表
            report_type: 报告类型 (comprehensive, trend, pattern, risk, optimization)
            time_range: 时间范围（天）
            
        Returns:
            进化报告
        """
        try:
            logger.info(f"开始生成{report_type}类型的进化报告")
            
            # 基础分析
            trend_analyses = {}
            pattern_analyses = {}
            
            for metric_name in metric_names:
                trend_analyses[metric_name] = self.analyze_evolution_trends(metric_name, time_range)
                pattern_analyses[metric_name] = self.identify_evolution_patterns(metric_name, time_range)
            
            # 根据报告类型生成特定内容
            if report_type == "comprehensive":
                report_content = self._generate_comprehensive_report(
                    metric_names, trend_analyses, pattern_analyses, time_range)
            elif report_type == "trend":
                report_content = self._generate_trend_report(metric_names, trend_analyses, time_range)
            elif report_type == "pattern":
                report_content = self._generate_pattern_report(metric_names, pattern_analyses, time_range)
            elif report_type == "risk":
                risk_assessment = self.assess_evolution_risks(metric_names)
                report_content = self._generate_risk_report(metric_names, risk_assessment, time_range)
            elif report_type == "optimization":
                optimization_suggestions = self.generate_optimization_suggestions(metric_names)
                report_content = self._generate_optimization_report(metric_names, optimization_suggestions, time_range)
            else:
                report_content = self._generate_comprehensive_report(
                    metric_names, trend_analyses, pattern_analyses, time_range)
            
            # 生成报告结构
            report = {
                "report_metadata": {
                    "report_id": f"PERF_EVOL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "report_type": report_type,
                    "generation_time": datetime.now().isoformat(),
                    "time_range": time_range,
                    "metrics_analyzed": len(metric_names),
                    "analyst": "H8性能进化器"
                },
                "executive_summary": self._generate_executive_summary(report_content),
                "detailed_analysis": report_content,
                "recommendations": self._extract_recommendations(report_content),
                "appendices": {
                    "data_quality": self._assess_data_quality(metric_names, time_range),
                    "methodology": "本报告基于统计学分析和机器学习算法生成",
                    "disclaimer": "报告结果仅供参考，实际应用需要结合具体业务场景"
                }
            }
            
            # 保存报告
            self._save_report(report)
            
            logger.info(f"完成{report_type}类型进化报告生成")
            return report
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            return {"error": f"报告生成失败: {str(e)}"}
    
    def _generate_comprehensive_report(self, 
                                     metric_names: List[str],
                                     trend_analyses: Dict[str, Any],
                                     pattern_analyses: Dict[str, Any],
                                     time_range: int) -> Dict[str, Any]:
        """生成综合报告"""
        # 效果评估
        effectiveness_evaluation = self.evaluate_evolution_effectiveness(metric_names, time_range, time_range)
        
        # 风险评估
        risk_assessment = self.assess_evolution_risks(metric_names)
        
        # 优化建议
        optimization_suggestions = self.generate_optimization_suggestions(metric_names)
        
        # 历史跟踪
        historical_tracking = self.track_evolution_history(metric_names, time_range * 3)
        
        return {
            "trend_analysis": trend_analyses,
            "pattern_analysis": pattern_analyses,
            "effectiveness_evaluation": effectiveness_evaluation,
            "risk_assessment": risk_assessment,
            "optimization_suggestions": optimization_suggestions,
            "historical_tracking": historical_tracking,
            "key_findings": self._extract_key_findings(trend_analyses, pattern_analyses, risk_assessment),
            "action_items": self._generate_action_items(optimization_suggestions, risk_assessment)
        }
    
    def _generate_trend_report(self, 
                             metric_names: List[str],
                             trend_analyses: Dict[str, Any],
                             time_range: int) -> Dict[str, Any]:
        """生成趋势报告"""
        return {
            "trend_analyses": trend_analyses,
            "trend_summary": self._summarize_trends(trend_analyses),
            "trend_predictions": self._extract_predictions(trend_analyses),
            "trend_recommendations": self._generate_trend_recommendations(trend_analyses)
        }
    
    def _generate_pattern_report(self, 
                               metric_names: List[str],
                               pattern_analyses: Dict[str, Any],
                               time_range: int) -> Dict[str, Any]:
        """生成模式报告"""
        return {
            "pattern_analyses": pattern_analyses,
            "pattern_summary": self._summarize_patterns(pattern_analyses),
            "pattern_insights": self._extract_pattern_insights(pattern_analyses),
            "pattern_recommendations": self._generate_pattern_recommendations(pattern_analyses)
        }
    
    def _generate_risk_report(self, 
                            metric_names: List[str],
                            risk_assessment: Dict[str, Any],
                            time_range: int) -> Dict[str, Any]:
        """生成风险报告"""
        return {
            "risk_assessment": risk_assessment,
            "risk_summary": risk_assessment.get("risk_summary", {}),
            "critical_risks": self._identify_critical_risks(risk_assessment),
            "risk_mitigation_plan": self._create_risk_mitigation_plan(risk_assessment)
        }
    
    def _generate_optimization_report(self, 
                                    metric_names: List[str],
                                    optimization_suggestions: Dict[str, Any],
                                    time_range: int) -> Dict[str, Any]:
        """生成优化报告"""
        return {
            "optimization_suggestions": optimization_suggestions,
            "priority_actions": optimization_suggestions.get("priority_suggestions", []),
            "implementation_roadmap": optimization_suggestions.get("implementation_roadmap", {}),
            "expected_outcomes": self._estimate_optimization_outcomes(optimization_suggestions)
        }
    
    def _generate_executive_summary(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行摘要"""
        summary = {
            "key_metrics_analyzed": len(content.get("trend_analysis", {})),
            "overall_performance": "良好",
            "main_findings": [],
            "immediate_actions": [],
            "strategic_recommendations": []
        }
        
        # 提取主要发现
        if "key_findings" in content:
            summary["main_findings"] = content["key_findings"][:3]  # 前3个主要发现
        
        # 提取立即行动
        if "action_items" in content:
            summary["immediate_actions"] = content["action_items"][:3]  # 前3个立即行动
        
        # 战略建议
        if "recommendations" in content:
            summary["strategic_recommendations"] = content["recommendations"][:3]  # 前3个战略建议
        
        return summary
    
    def _extract_key_findings(self, 
                            trend_analyses: Dict[str, Any],
                            pattern_analyses: Dict[str, Any],
                            risk_assessment: Dict[str, Any]) -> List[str]:
        """提取关键发现"""
        findings = []
        
        # 从趋势分析中提取
        for metric_name, analysis in trend_analyses.items():
            if "error" not in analysis:
                trend_info = analysis.get("trend_analysis", {})
                direction = trend_info.get("direction", "未知")
                findings.append(f"{metric_name}呈现{direction}趋势")
        
        # 从风险评估中提取
        risk_summary = risk_assessment.get("risk_summary", {})
        overall_level = risk_summary.get("overall_risk_level", "未知")
        findings.append(f"整体风险等级：{overall_level}")
        
        return findings
    
    def _generate_action_items(self, 
                             optimization_suggestions: Dict[str, Any],
                             risk_assessment: Dict[str, Any]) -> List[str]:
        """生成行动项目"""
        actions = []
        
        # 从优化建议中提取高优先级行动
        priority_suggestions = optimization_suggestions.get("priority_suggestions", [])
        for suggestion in priority_suggestions[:3]:
            actions.append(suggestion.suggestion)
        
        # 从风险评估中提取风险控制行动
        control_strategies = risk_assessment.get("control_strategies", [])
        for strategy in control_strategies[:2]:
            actions.append(f"风险控制：{strategy.get('strategy', '')}")
        
        return actions
    
    def _extract_recommendations(self, content: Dict[str, Any]) -> List[str]:
        """提取建议"""
        recommendations = []
        
        # 收集各种建议
        if "optimization_suggestions" in content:
            suggestions = content["optimization_suggestions"]
            if "suggestions_by_category" in suggestions:
                for category_suggestions in suggestions["suggestions_by_category"].values():
                    for suggestion in category_suggestions[:2]:  # 每个类别取前2个
                        recommendations.append(f"[{suggestion.category}] {suggestion.suggestion}")
        
        return recommendations
    
    def _assess_data_quality(self, metric_names: List[str], time_range: int) -> Dict[str, Any]:
        """评估数据质量"""
        quality_scores = {}
        
        for metric_name in metric_names:
            data = self._get_metric_data(metric_name, time_range)
            
            # 数据完整性
            completeness = len(data) / time_range if time_range > 0 else 0
            
            # 数据一致性（检查异常值）
            if data:
                values = [item['value'] for item in data]
                q75, q25 = np.percentile(values, [75, 25])
                iqr = q75 - q25
                outliers = sum(1 for v in values if v < q25 - 1.5*iqr or v > q75 + 1.5*iqr)
                consistency = 1 - (outliers / len(values)) if values else 0
            else:
                consistency = 0
            
            quality_scores[metric_name] = {
                "completeness": completeness,
                "consistency": consistency,
                "overall_quality": (completeness + consistency) / 2
            }
        
        return quality_scores
    
    def _save_report(self, report: Dict[str, Any]):
        """保存报告"""
        report_id = report["report_metadata"]["report_id"]
        filename = f"performance_evolution_report_{report_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"报告已保存到 {filename}")
    
    def _estimate_optimization_outcomes(self, optimization_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """估算优化结果"""
        priority_suggestions = optimization_suggestions.get("priority_suggestions", [])
        
        total_expected_improvement = sum(s.expected_improvement for s in priority_suggestions)
        total_implementation_cost = sum(s.implementation_cost for s in priority_suggestions)
        
        return {
            "total_expected_improvement": total_expected_improvement,
            "total_implementation_cost": total_implementation_cost,
            "cost_benefit_ratio": total_expected_improvement / max(total_implementation_cost, 1),
            "high_impact_suggestions": len([s for s in priority_suggestions if s.expected_improvement > 10]),
            "quick_wins": len([s for s in priority_suggestions if s.timeframe in ["1-2周", "2-4周"]])
        }
    
    # ==================== 辅助方法 ====================
    
    def add_performance_metric(self, metric: PerformanceMetric):
        """添加性能指标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (timestamp, metric_name, value, unit, category, target_value, threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.timestamp.isoformat(),
            metric.metric_name,
            metric.value,
            metric.unit,
            metric.category,
            metric.target_value,
            metric.threshold
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"添加性能指标: {metric.metric_name} = {metric.value}")
    
    def add_evolution_event(self, event: EvolutionEvent):
        """添加进化事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO evolution_events 
            (timestamp, event_type, description, impact_score, category, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp.isoformat(),
            event.event_type,
            event.description,
            event.impact_score,
            event.category,
            json.dumps(event.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"添加进化事件: {event.event_type} - {event.description}")
    
    def get_evolution_summary(self, time_range: int = 30) -> Dict[str, Any]:
        """获取进化摘要"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取最近的指标统计
            start_date = (datetime.now() - timedelta(days=time_range)).isoformat()
            
            cursor.execute('''
                SELECT metric_name, COUNT(*) as data_points, 
                       AVG(value) as avg_value, MIN(value) as min_value, MAX(value) as max_value
                FROM performance_metrics
                WHERE timestamp >= ?
                GROUP BY metric_name
            ''', (start_date,))
            
            metrics_summary = {}
            for row in cursor.fetchall():
                metrics_summary[row[0]] = {
                    "data_points": row[1],
                    "avg_value": row[2],
                    "min_value": row[3],
                    "max_value": row[4]
                }
            
            # 获取最近的事件
            cursor.execute('''
                SELECT event_type, COUNT(*) as count, AVG(impact_score) as avg_impact
                FROM evolution_events
                WHERE timestamp >= ?
                GROUP BY event_type
            ''', (start_date,))
            
            events_summary = {}
            for row in cursor.fetchall():
                events_summary[row[0]] = {
                    "count": row[1],
                    "avg_impact": row[2]
                }
            
            conn.close()
            
            return {
                "summary_timestamp": datetime.now().isoformat(),
                "time_range": time_range,
                "metrics_summary": metrics_summary,
                "events_summary": events_summary,
                "total_metrics": len(metrics_summary),
                "total_events": sum(event["count"] for event in events_summary.values())
            }
            
        except Exception as e:
            logger.error(f"获取进化摘要失败: {str(e)}")
            return {"error": f"获取进化摘要失败: {str(e)}"}
    
    def export_evolution_data(self, format: str = "json") -> str:
        """导出进化数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if format.lower() == "json":
                # 导出为JSON格式
                data = {}
                
                # 性能指标
                df_metrics = pd.read_sql_query("SELECT * FROM performance_metrics", conn)
                data["performance_metrics"] = df_metrics.to_dict('records')
                
                # 进化事件
                df_events = pd.read_sql_query("SELECT * FROM evolution_events", conn)
                data["evolution_events"] = df_events.to_dict('records')
                
                # 风险评估
                df_risks = pd.read_sql_query("SELECT * FROM risk_assessments", conn)
                data["risk_assessments"] = df_risks.to_dict('records')
                
                conn.close()
                
                filename = f"evolution_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                
                return filename
                
            else:
                conn.close()
                return "不支持的导出格式"
                
        except Exception as e:
            logger.error(f"数据导出失败: {str(e)}")
            return f"数据导出失败: {str(e)}"


# ==================== 使用示例 ====================

def demo_performance_evolution():
    """性能进化器演示"""
    print("=== H8性能进化器演示 ===\n")
    
    # 初始化进化器
    evolution = PerformanceEvolution("demo_evolution.db")
    
    # 生成示例数据
    print("1. 生成示例性能数据...")
    import random
    
    base_values = {
        "系统响应时间": 100,
        "CPU使用率": 50,
        "内存使用率": 60,
        "吞吐量": 1000,
        "错误率": 5
    }
    
    for i in range(30):  # 生成30天的数据
        for metric_name, base_value in base_values.items():
            # 模拟趋势变化
            trend_factor = 1 + (i - 15) * 0.02  # 中期有所改善
            noise = random.gauss(0, base_value * 0.1)
            value = max(0, base_value * trend_factor + noise)
            
            metric = PerformanceMetric(
                timestamp=datetime.now() - timedelta(days=30-i),
                metric_name=metric_name,
                value=value,
                unit="ms" if "时间" in metric_name else "%" if "率" in metric_name else "req/s",
                category="performance"
            )
            evolution.add_performance_metric(metric)
    
    # 分析指标
    metric_names = list(base_values.keys())
    
    print("\n2. 进行趋势分析...")
    for metric_name in metric_names[:2]:  # 分析前2个指标
        trend_analysis = evolution.analyze_evolution_trends(metric_name)
        print(f"{metric_name}趋势: {trend_analysis.get('trend_analysis', {}).get('direction', '未知')}")
    
    print("\n3. 识别进化模式...")
    pattern_analysis = evolution.identify_evolution_patterns(metric_names[0])
    print(f"{metric_names[0]}模式: {pattern_analysis.get('dominant_pattern', '未知')}")
    
    print("\n4. 评估进化效果...")
    effectiveness = evolution.evaluate_evolution_effectiveness(metric_names[:2])
    print(f"效果评级: {effectiveness.get('overall_assessment', {}).get('overall_rating', '未知')}")
    
    print("\n5. 生成优化建议...")
    suggestions = evolution.generate_optimization_suggestions(metric_names[:2])
    print(f"生成建议数量: {suggestions.get('total_suggestions', 0)}")
    
    print("\n6. 评估进化风险...")
    risks = evolution.assess_evolution_risks(metric_names[:2])
    print(f"识别风险数量: {risks.get('total_risks_identified', 0)}")
    
    print("\n7. 跟踪进化历史...")
    history = evolution.track_evolution_history(metric_names[:2])
    print(f"历史阶段数量: {len(history.get('individual_histories', {}).get(metric_names[0], {}).get('evolution_phases', []))}")
    
    print("\n8. 生成综合报告...")
    report = evolution.generate_evolution_report(metric_names[:2], "comprehensive")
    print(f"报告ID: {report['report_metadata']['report_id']}")
    
    print("\n9. 导出数据...")
    export_file = evolution.export_evolution_data()
    print(f"数据已导出到: {export_file}")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    # 运行演示
    demo_performance_evolution()