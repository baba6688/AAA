"""
P7 A/B测试器 - 主要实现文件
包含完整的A/B测试功能：实验设计、用户分组、数据收集、统计分析等
"""

import hashlib
import json
import math
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """实验配置类"""
    name: str
    description: str
    variants: Dict[str, Dict[str, Any]]  # 变体配置
    target_metrics: List[str]  # 目标指标
    traffic_split: Dict[str, float]  # 流量分配比例
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "draft"  # draft, running, paused, completed, stopped
    confidence_level: float = 0.95  # 置信水平
    min_sample_size: int = 1000  # 最小样本量
    max_duration_days: int = 30  # 最大实验天数
    
    def __post_init__(self):
        """验证配置参数"""
        if not self.variants:
            raise ValueError("实验必须至少有一个变体")
        
        # 验证流量分配比例
        total_split = sum(self.traffic_split.values())
        if abs(total_split - 1.0) > 0.001:
            raise ValueError("流量分配比例总和必须为1.0")
        
        # 验证变体名称
        variant_names = set(self.variants.keys())
        split_names = set(self.traffic_split.keys())
        if variant_names != split_names:
            raise ValueError("变体名称与流量分配名称不匹配")


@dataclass
class UserData:
    """用户数据类"""
    user_id: str
    variant: str
    timestamp: datetime
    metrics: Dict[str, float]
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class ExperimentResult:
    """实验结果类"""
    experiment_name: str
    variant_results: Dict[str, Dict[str, Any]]
    statistical_significance: bool
    confidence_interval: Dict[str, Tuple[float, float]]
    p_value: float
    effect_size: float
    recommendation: str
    summary: str


class UserGroup:
    """用户分组管理类"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.user_assignments: Dict[str, str] = {}  # user_id -> variant
        self.user_attributes: Dict[str, Dict[str, Any]] = {}  # user_id -> attributes
        self.group_sizes: Dict[str, int] = defaultdict(int)
        
    def assign_user(self, user_id: str, variant: str, attributes: Dict[str, Any] = None) -> None:
        """分配用户到变体"""
        self.user_assignments[user_id] = variant
        if attributes:
            self.user_attributes[user_id] = attributes
        self.group_sizes[variant] += 1
        
    def get_user_variant(self, user_id: str) -> Optional[str]:
        """获取用户变体"""
        return self.user_assignments.get(user_id)
        
    def get_group_size(self, variant: str) -> int:
        """获取变体组大小"""
        return self.group_sizes[variant]
        
    def get_all_groups(self) -> Dict[str, List[str]]:
        """获取所有变体组"""
        groups = defaultdict(list)
        for user_id, variant in self.user_assignments.items():
            groups[variant].append(user_id)
        return dict(groups)
        
    def stratified_assignment(self, user_ids: List[str], 
                            traffic_split: Dict[str, float],
                            stratification_key: str = None) -> None:
        """分层随机分组"""
        if stratification_key is None:
            # 简单随机分组
            self._simple_random_assignment(user_ids, traffic_split)
        else:
            # 分层抽样
            self._stratified_sampling(user_ids, traffic_split, stratification_key)
    
    def _simple_random_assignment(self, user_ids: List[str], 
                                traffic_split: Dict[str, float]) -> None:
        """简单随机分组"""
        # 创建累积概率列表
        cumulative_probs = []
        cumulative = 0
        for variant, prob in traffic_split.items():
            cumulative += prob
            cumulative_probs.append((variant, cumulative))
        
        for user_id in user_ids:
            rand_val = random.random()
            for variant, cum_prob in cumulative_probs:
                if rand_val <= cum_prob:
                    self.assign_user(user_id, variant)
                    break
    
    def _stratified_sampling(self, user_ids: List[str], 
                           traffic_split: Dict[str, float],
                           stratification_key: str) -> None:
        """分层抽样分组"""
        # 按分层键值分组
        strata = defaultdict(list)
        for user_id in user_ids:
            if user_id in self.user_attributes:
                strat_value = self.user_attributes[user_id].get(stratification_key, "unknown")
            else:
                strat_value = "unknown"
            strata[strat_value].append(user_id)
        
        # 在每个层级内进行随机分组
        for strat_users in strata.values():
            self._simple_random_assignment(strat_users, traffic_split)


class Statistics:
    """统计分析工具类"""
    
    @staticmethod
    def calculate_sample_size(baseline_rate: float, 
                            min_detectable_effect: float,
                            alpha: float = 0.05,
                            power: float = 0.8) -> int:
        """计算所需样本量"""
        # 使用双样本比例检验的样本量计算公式
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + min_detectable_effect)
        
        pooled_p = (p1 + p2) / 2
        
        numerator = (z_alpha * math.sqrt(2 * pooled_p * (1 - pooled_p)) + 
                    z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        
        denominator = (p2 - p1) ** 2
        
        return int(math.ceil(numerator / denominator))
    
    @staticmethod
    def two_proportion_z_test(control_data: List[float], 
                            treatment_data: List[float],
                            alpha: float = 0.05) -> Dict[str, Any]:
        """双比例Z检验"""
        n1, n2 = len(control_data), len(treatment_data)
        p1 = sum(control_data) / n1
        p2 = sum(treatment_data) / n2
        
        # 合并比例
        p_pooled = (sum(control_data) + sum(treatment_data)) / (n1 + n2)
        
        # 标准误差
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Z统计量
        z_stat = (p2 - p1) / se
        
        # p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # 置信区间
        diff = p2 - p1
        margin_error = stats.norm.ppf(1 - alpha/2) * se
        ci_lower = diff - margin_error
        ci_upper = diff + margin_error
        
        # 效应大小 (Cohen's h)
        h = 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))
        
        return {
            'p1': p1,
            'p2': p2,
            'difference': diff,
            'z_statistic': z_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': h,
            'significant': p_value < alpha
        }
    
    @staticmethod
    def continuous_t_test(control_data: List[float], 
                         treatment_data: List[float],
                         alpha: float = 0.05) -> Dict[str, Any]:
        """连续变量t检验"""
        # 使用Welch's t-test (不假设等方差)
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
        
        # 效应大小 (Cohen's d)
        pooled_std = math.sqrt(((len(control_data) - 1) * statistics.variance(control_data) + 
                               (len(treatment_data) - 1) * statistics.variance(treatment_data)) / 
                              (len(control_data) + len(treatment_data) - 2))
        
        cohens_d = (statistics.mean(treatment_data) - statistics.mean(control_data)) / pooled_std
        
        # 置信区间
        se = pooled_std * math.sqrt(1/len(control_data) + 1/len(treatment_data))
        df = len(control_data) + len(treatment_data) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        diff = statistics.mean(treatment_data) - statistics.mean(control_data)
        ci_lower = diff - t_critical * se
        ci_upper = diff + t_critical * se
        
        return {
            'control_mean': statistics.mean(control_data),
            'treatment_mean': statistics.mean(treatment_data),
            'difference': diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': cohens_d,
            'significant': p_value < alpha
        }
    
    @staticmethod
    def calculate_confidence_interval(data: List[float], 
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        mean = statistics.mean(data)
        std_err = statistics.stdev(data) / math.sqrt(len(data))
        
        # 使用t分布
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, len(data) - 1)
        
        margin_error = t_critical * std_err
        return (mean - margin_error, mean + margin_error)
    
    @staticmethod
    def sequential_test_analysis(data_points: List[Tuple[datetime, Dict[str, float]]],
                               alpha: float = 0.05) -> Dict[str, Any]:
        """序贯检验分析"""
        if len(data_points) < 2:
            return {'ready_for_analysis': False}
        
        # 按时间排序
        data_points.sort(key=lambda x: x[0])
        
        # 计算累积统计量
        cumulative_data = defaultdict(list)
        for timestamp, metrics in data_points:
            for metric, value in metrics.items():
                cumulative_data[metric].append(value)
        
        results = {}
        for metric, values in cumulative_data.items():
            if len(values) >= 2:
                # 执行t检验
                mid_point = len(values) // 2
                control = values[:mid_point]
                treatment = values[mid_point:]
                
                if len(control) > 0 and len(treatment) > 0:
                    test_result = Statistics.continuous_t_test(control, treatment, alpha)
                    results[metric] = test_result
        
        return {
            'ready_for_analysis': len(results) > 0,
            'results': results,
            'data_points': len(data_points)
        }


class ExperimentManager:
    """实验管理类"""
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        self.user_groups: Dict[str, UserGroup] = {}
        self.data_collection: Dict[str, List[UserData]] = defaultdict(list)
        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """创建实验"""
        if config.name in self.experiments:
            raise ValueError(f"实验 {config.name} 已存在")
        
        self.experiments[config.name] = config
        self.user_groups[config.name] = UserGroup(config.name)
        
        logger.info(f"创建实验: {config.name}")
        return config.name
    
    def start_experiment(self, experiment_name: str) -> None:
        """启动实验"""
        if experiment_name not in self.experiments:
            raise ValueError(f"实验 {experiment_name} 不存在")
        
        config = self.experiments[experiment_name]
        config.status = "running"
        config.start_time = datetime.now()
        
        logger.info(f"启动实验: {experiment_name}")
    
    def pause_experiment(self, experiment_name: str) -> None:
        """暂停实验"""
        if experiment_name not in self.experiments:
            raise ValueError(f"实验 {experiment_name} 不存在")
        
        self.experiments[experiment_name].status = "paused"
        logger.info(f"暂停实验: {experiment_name}")
    
    def resume_experiment(self, experiment_name: str) -> None:
        """恢复实验"""
        if experiment_name not in self.experiments:
            raise ValueError(f"实验 {experiment_name} 不存在")
        
        self.experiments[experiment_name].status = "running"
        logger.info(f"恢复实验: {experiment_name}")
    
    def stop_experiment(self, experiment_name: str) -> None:
        """停止实验"""
        if experiment_name not in self.experiments:
            raise ValueError(f"实验 {experiment_name} 不存在")
        
        self.experiments[experiment_name].status = "stopped"
        self.experiments[experiment_name].end_time = datetime.now()
        
        logger.info(f"停止实验: {experiment_name}")
    
    def complete_experiment(self, experiment_name: str) -> None:
        """完成实验"""
        if experiment_name not in self.experiments:
            raise ValueError(f"实验 {experiment_name} 不存在")
        
        self.experiments[experiment_name].status = "completed"
        self.experiments[experiment_name].end_time = datetime.now()
        
        logger.info(f"完成实验: {experiment_name}")
    
    def get_experiment_status(self, experiment_name: str) -> Dict[str, Any]:
        """获取实验状态"""
        if experiment_name not in self.experiments:
            raise ValueError(f"实验 {experiment_name} 不存在")
        
        config = self.experiments[experiment_name]
        user_group = self.user_groups[experiment_name]
        
        return {
            'name': config.name,
            'status': config.status,
            'start_time': config.start_time,
            'end_time': config.end_time,
            'group_sizes': dict(user_group.group_sizes),
            'total_users': len(user_group.user_assignments),
            'data_points': len(self.data_collection[experiment_name])
        }
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """列出所有实验"""
        return [self.get_experiment_status(name) for name in self.experiments.keys()]


class ABTester:
    """主要的A/B测试器类"""
    
    def __init__(self):
        self.manager = ExperimentManager()
        self.data_processor = DataProcessor()
        
    def create_experiment(self, 
                         name: str,
                         description: str,
                         variants: Dict[str, Dict[str, Any]],
                         target_metrics: List[str],
                         traffic_split: Dict[str, float],
                         confidence_level: float = 0.95,
                         min_sample_size: int = 1000,
                         max_duration_days: int = 30) -> str:
        """创建A/B测试实验"""
        config = ExperimentConfig(
            name=name,
            description=description,
            variants=variants,
            target_metrics=target_metrics,
            traffic_split=traffic_split,
            start_time=datetime.now(),
            confidence_level=confidence_level,
            min_sample_size=min_sample_size,
            max_duration_days=max_duration_days
        )
        
        return self.manager.create_experiment(config)
    
    def assign_users(self, 
                    experiment_name: str,
                    user_ids: List[str],
                    user_attributes: Dict[str, Dict[str, Any]] = None,
                    stratification_key: str = None) -> None:
        """分配用户到实验组"""
        user_group = self.manager.user_groups[experiment_name]
        
        # 设置用户属性
        if user_attributes:
            for user_id, attributes in user_attributes.items():
                user_group.user_attributes[user_id] = attributes
        
        # 执行分组
        config = self.manager.experiments[experiment_name]
        user_group.stratified_assignment(
            user_ids, 
            config.traffic_split, 
            stratification_key
        )
        
        logger.info(f"为实验 {experiment_name} 分配了 {len(user_ids)} 个用户")
    
    def collect_data(self, 
                    experiment_name: str,
                    user_id: str,
                    metrics: Dict[str, float],
                    attributes: Dict[str, Any] = None) -> None:
        """收集用户数据"""
        user_group = self.manager.user_groups[experiment_name]
        variant = user_group.get_user_variant(user_id)
        
        if variant is None:
            raise ValueError(f"用户 {user_id} 未分配到实验 {experiment_name}")
        
        user_data = UserData(
            user_id=user_id,
            variant=variant,
            timestamp=datetime.now(),
            metrics=metrics,
            attributes=attributes
        )
        
        self.manager.data_collection[experiment_name].append(user_data)
    
    def batch_collect_data(self, 
                          experiment_name: str,
                          data_batch: List[Tuple[str, Dict[str, float], Dict[str, Any]]]) -> None:
        """批量收集数据"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for user_id, metrics, attributes in data_batch:
                future = executor.submit(
                    self.collect_data, experiment_name, user_id, metrics, attributes
                )
                futures.append(future)
            
            # 等待所有任务完成
            for future in futures:
                future.result()
    
    def analyze_experiment(self, experiment_name: str) -> ExperimentResult:
        """分析实验结果"""
        if experiment_name not in self.manager.experiments:
            raise ValueError(f"实验 {experiment_name} 不存在")
        
        config = self.manager.experiments[experiment_name]
        data_collection = self.manager.data_collection[experiment_name]
        
        # 按变体分组数据
        variant_data = defaultdict(lambda: defaultdict(list))
        for user_data in data_collection:
            for metric, value in user_data.metrics.items():
                variant_data[user_data.variant][metric].append(value)
        
        # 执行统计分析
        variant_results = {}
        statistical_significance = False
        overall_p_value = 1.0
        effect_size = 0.0
        
        # 获取对照组和实验组
        variants = list(config.variants.keys())
        if len(variants) >= 2:
            control_variant = variants[0]
            treatment_variants = variants[1:]
            
            for target_metric in config.target_metrics:
                if target_metric in variant_data[control_variant]:
                    control_data = variant_data[control_variant][target_metric]
                    
                    for treatment_variant in treatment_variants:
                        if target_metric in variant_data[treatment_variant]:
                            treatment_data = variant_data[treatment_variant][target_metric]
                            
                            # 判断数据类型并执行相应检验
                            if all(isinstance(x, (int, float)) and x in [0, 1] for x in control_data + treatment_data):
                                # 二元数据
                                test_result = Statistics.two_proportion_z_test(
                                    control_data, treatment_data, 1 - config.confidence_level
                                )
                            else:
                                # 连续数据
                                test_result = Statistics.continuous_t_test(
                                    control_data, treatment_data, 1 - config.confidence_level
                                )
                            
                            variant_results[f"{control_variant}_vs_{treatment_variant}"] = test_result
                            
                            # 更新整体显著性
                            if test_result['significant']:
                                statistical_significance = True
                                overall_p_value = min(overall_p_value, test_result['p_value'])
                                effect_size = test_result['effect_size']
        
        # 生成建议
        if statistical_significance:
            if effect_size > 0.5:
                recommendation = "强烈推荐采用实验组方案"
            elif effect_size > 0.2:
                recommendation = "推荐采用实验组方案"
            else:
                recommendation = "可以考虑采用实验组方案，但效果较小"
        else:
            recommendation = "建议继续实验或调整实验设计"
        
        # 计算置信区间
        confidence_interval = {}
        for variant_pair, result in variant_results.items():
            confidence_interval[variant_pair] = result['confidence_interval']
        
        # 生成总结
        summary = f"""
实验 {experiment_name} 分析总结:
- 参与用户数: {len(self.manager.user_groups[experiment_name].user_assignments)}
- 数据点数量: {len(data_collection)}
- 统计显著性: {'是' if statistical_significance else '否'}
- 最小p值: {overall_p_value:.4f}
- 效应大小: {effect_size:.4f}
- 建议: {recommendation}
        """.strip()
        
        result = ExperimentResult(
            experiment_name=experiment_name,
            variant_results=variant_results,
            statistical_significance=statistical_significance,
            confidence_interval=confidence_interval,
            p_value=overall_p_value,
            effect_size=effect_size,
            recommendation=recommendation,
            summary=summary
        )
        
        # 缓存结果
        self.manager.results[experiment_name] = result
        
        return result
    
    def get_experiment_progress(self, experiment_name: str) -> Dict[str, Any]:
        """获取实验进度"""
        status = self.manager.get_experiment_status(experiment_name)
        config = self.manager.experiments[experiment_name]
        
        # 计算实验持续时间
        if status['start_time']:
            duration = datetime.now() - status['start_time']
            days_running = duration.days + duration.seconds / (24 * 3600)
        else:
            days_running = 0
        
        # 计算样本量进度
        min_sample_needed = config.min_sample_size * len(config.variants)
        current_sample = status['total_users']
        sample_progress = min(current_sample / min_sample_needed, 1.0) if min_sample_needed > 0 else 0
        
        # 计算时间进度
        time_progress = min(days_running / config.max_duration_days, 1.0) if config.max_duration_days > 0 else 0
        
        # 建议决策
        if sample_progress >= 1.0 and time_progress >= 0.5:
            decision_ready = True
        elif sample_progress >= 0.8 and time_progress >= 0.7:
            decision_ready = True
        else:
            decision_ready = False
        
        return {
            **status,
            'days_running': days_running,
            'sample_progress': sample_progress,
            'time_progress': time_progress,
            'decision_ready': decision_ready,
            'recommendation': '可以进行分析' if decision_ready else '继续收集数据'
        }
    
    def visualize_results(self, experiment_name: str, save_path: str = None) -> None:
        """可视化实验结果"""
        if experiment_name not in self.manager.results:
            # 如果没有缓存结果，先进行分析
            self.analyze_experiment(experiment_name)
        
        result = self.manager.results[experiment_name]
        config = self.manager.experiments[experiment_name]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'实验 {experiment_name} 结果分析', fontsize=16, fontweight='bold')
        
        # 1. 变体样本量分布
        ax1 = axes[0, 0]
        user_group = self.manager.user_groups[experiment_name]
        variants = list(config.variants.keys())
        sizes = [user_group.get_group_size(v) for v in variants]
        
        ax1.bar(variants, sizes, color=['skyblue', 'lightcoral', 'lightgreen'][:len(variants)])
        ax1.set_title('变体样本量分布')
        ax1.set_ylabel('用户数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 效应大小对比
        ax2 = axes[0, 1]
        effect_sizes = []
        labels = []
        for variant_pair, variant_result in result.variant_results.items():
            effect_sizes.append(abs(variant_result['effect_size']))
            labels.append(variant_pair.replace('_vs_', ' vs '))
        
        if effect_sizes:
            bars = ax2.bar(range(len(effect_sizes)), effect_sizes, 
                          color=['green' if x > 0 else 'red' for x in effect_sizes])
            ax2.set_title('效应大小对比')
            ax2.set_ylabel('Cohen\'s d / h')
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='小效应')
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='中效应')
            ax2.legend()
        
        # 3. p值分布
        ax3 = axes[1, 0]
        p_values = [variant_result['p_value'] for variant_result in result.variant_results.values()]
        if p_values:
            ax3.hist(p_values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=0.05, color='red', linestyle='--', label='显著性阈值 (0.05)')
            ax3.set_title('p值分布')
            ax3.set_xlabel('p值')
            ax3.set_ylabel('频次')
            ax3.legend()
        
        # 4. 实验进度
        ax4 = axes[1, 1]
        progress = self.get_experiment_progress(experiment_name)
        
        categories = ['样本量进度', '时间进度']
        values = [progress['sample_progress'], progress['time_progress']]
        colors = ['lightblue', 'lightgreen']
        
        bars = ax4.bar(categories, values, color=colors)
        ax4.set_title('实验进度')
        ax4.set_ylabel('进度比例')
        ax4.set_ylim(0, 1)
        
        # 添加进度标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def export_results(self, experiment_name: str, file_path: str) -> None:
        """导出实验结果"""
        if experiment_name not in self.manager.results:
            self.analyze_experiment(experiment_name)
        
        result = self.manager.results[experiment_name]
        
        # 准备导出数据
        export_data = {
            'experiment_info': asdict(self.manager.experiments[experiment_name]),
            'experiment_status': self.manager.get_experiment_status(experiment_name),
            'experiment_progress': self.get_experiment_progress(experiment_name),
            'analysis_result': asdict(result)
        }
        
        # 导出为JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"实验结果已导出到: {file_path}")
    
    def get_recommendation(self, experiment_name: str) -> str:
        """获取实验建议"""
        if experiment_name not in self.manager.results:
            self.analyze_experiment(experiment_name)
        
        return self.manager.results[experiment_name].recommendation
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> None:
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleaned_count = 0
        for experiment_name, data_list in self.manager.data_collection.items():
            original_count = len(data_list)
            self.manager.data_collection[experiment_name] = [
                data for data in data_list if data.timestamp > cutoff_date
            ]
            cleaned_count += original_count - len(self.manager.data_collection[experiment_name])
        
        logger.info(f"清理了 {cleaned_count} 条过期数据记录")


class DataProcessor:
    """数据处理工具类"""
    
    @staticmethod
    def calculate_conversion_rate(data: List[UserData], variant: str) -> float:
        """计算转化率"""
        variant_data = [d for d in data if d.variant == variant]
        if not variant_data:
            return 0.0
        
        # 假设转化指标为 'converted'
        conversions = sum(1 for d in variant_data if d.metrics.get('converted', 0) == 1)
        return conversions / len(variant_data)
    
    @staticmethod
    def calculate_retention_rate(data: List[UserData], variant: str, days: int = 7) -> float:
        """计算留存率"""
        variant_data = [d for d in data if d.variant == variant]
        if not variant_data:
            return 0.0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        retained_users = sum(1 for d in variant_data if d.timestamp > cutoff_date)
        return retained_users / len(variant_data)
    
    @staticmethod
    def generate_sample_data(experiment_name: str, 
                           num_users: int = 1000,
                           num_days: int = 7) -> List[Tuple[str, Dict[str, float], Dict[str, Any]]]:
        """生成示例数据"""
        sample_data = []
        
        for i in range(num_users):
            user_id = f"user_{i:06d}"
            
            # 随机生成用户属性
            attributes = {
                'age_group': random.choice(['18-25', '26-35', '36-45', '46+']),
                'device_type': random.choice(['mobile', 'desktop', 'tablet']),
                'region': random.choice(['北京', '上海', '广州', '深圳', '其他'])
            }
            
            # 模拟多天的数据
            for day in range(num_days):
                timestamp = datetime.now() - timedelta(days=day)
                
                # 模拟用户行为指标
                metrics = {
                    'page_views': random.randint(1, 20),
                    'session_duration': random.uniform(30, 600),  # 秒
                    'bounce_rate': random.uniform(0.1, 0.8),
                    'converted': random.choice([0, 1]),
                    'revenue': random.uniform(0, 100) if random.random() < 0.1 else 0
                }
                
                sample_data.append((user_id, metrics, attributes))
        
        return sample_data