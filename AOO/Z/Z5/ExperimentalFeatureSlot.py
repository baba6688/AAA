"""
Z5实验性功能插槽系统 - 主要实现

该模块实现了完整的实验性功能管理解决方案，包括实验功能管理、
特性开关、A/B测试、实验监控等功能。
"""

import json
import time
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import threading
import random


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """实验状态枚举"""
    DRAFT = "draft"           # 草稿
    ACTIVE = "active"         # 活跃
    PAUSED = "paused"         # 暂停
    COMPLETED = "completed"   # 完成
    CANCELLED = "cancelled"   # 取消


class FeatureFlagStatus(Enum):
    """特性开关状态枚举"""
    OFF = "off"               # 关闭
    ON = "on"                 # 开启
    CONDITIONAL = "conditional"  # 条件开启


class ABTestStatus(Enum):
    """A/B测试状态枚举"""
    PLANNING = "planning"     # 规划中
    RUNNING = "running"       # 运行中
    PAUSED = "paused"         # 暂停
    FINISHED = "finished"     # 已完成
    ARCHIVED = "archived"     # 已归档


@dataclass
class FeatureFlag:
    """特性开关数据类"""
    name: str
    status: FeatureFlagStatus
    description: str = ""
    conditions: Dict[str, Any] = None
    rollout_percentage: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.conditions is None:
            self.conditions = {}


@dataclass
class ABTestGroup:
    """A/B测试组数据类"""
    name: str
    percentage: float
    description: str = ""
    variant_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.variant_config is None:
            self.variant_config = {}


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    experiment_id: str
    user_id: str
    group_name: str
    timestamp: datetime
    metrics: Dict[str, Any] = None
    success: bool = True
    error_message: str = ""
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class ExperimentMetrics:
    """实验指标数据类"""
    total_users: int = 0
    conversion_rate: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


class ExperimentSecurity:
    """实验安全控制器"""
    
    def __init__(self):
        self.security_rules = {}
        self.user_permissions = {}
        self.risk_thresholds = {}
    
    def add_security_rule(self, experiment_id: str, rule: Dict[str, Any]):
        """添加安全规则"""
        self.security_rules[experiment_id] = rule
        logger.info(f"为实验 {experiment_id} 添加安全规则")
    
    def validate_experiment_access(self, experiment_id: str, user_id: str, 
                                 action: str) -> bool:
        """验证实验访问权限"""
        if experiment_id not in self.security_rules:
            # 如果没有安全规则，默认拒绝访问
            return False
        
        rule = self.security_rules[experiment_id]
        
        # 检查用户权限
        if user_id not in self.user_permissions:
            return False
        
        user_perms = self.user_permissions[user_id]
        if action not in user_perms.get('actions', []):
            return False
        
        # 检查实验权限
        if 'experiments' in user_perms:
            if experiment_id not in user_perms['experiments']:
                return False
        
        # 检查风险阈值
        if 'risk_threshold' in rule:
            current_risk = self._calculate_current_risk(experiment_id)
            if current_risk > rule['risk_threshold']:
                logger.warning(f"实验 {experiment_id} 当前风险 {current_risk} 超过阈值 {rule['risk_threshold']}")
                return False
        
        return True
    
    def _calculate_current_risk(self, experiment_id: str) -> float:
        """计算当前风险值"""
        # 简化实现，实际应该基于更复杂的算法
        return random.uniform(0, 1)
    
    def register_user_permission(self, user_id: str, permissions: Dict[str, Any]):
        """注册用户权限"""
        self.user_permissions[user_id] = permissions
        logger.info(f"为用户 {user_id} 注册权限")


class ExperimentConfig:
    """实验配置管理"""
    
    def __init__(self):
        self.configs = {}
        self.default_config = {
            'max_users_per_experiment': 10000,
            'default_duration_days': 7,
            'min_sample_size': 100,
            'confidence_level': 0.95,
            'auto_stop_enabled': True
        }
    
    def get_config(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验配置"""
        return self.configs.get(experiment_id, self.default_config.copy())
    
    def set_config(self, experiment_id: str, config: Dict[str, Any]):
        """设置实验配置"""
        self.configs[experiment_id] = {**self.default_config, **config}
        logger.info(f"为实验 {experiment_id} 设置配置: {config}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置有效性"""
        required_keys = ['max_users_per_experiment', 'default_duration_days']
        return all(key in config for key in required_keys)


class FeatureToggle:
    """特性开关管理"""
    
    def __init__(self):
        self.flags = {}
        self.conditions = {}
        self.user_assignments = {}
        self.lock = threading.Lock()
    
    def create_flag(self, name: str, status: FeatureFlagStatus = FeatureFlagStatus.OFF,
                   description: str = "", rollout_percentage: float = 0.0, 
                   conditions: Dict[str, Any] = None) -> FeatureFlag:
        """创建特性开关"""
        with self.lock:
            if conditions is None:
                conditions = {}
            flag = FeatureFlag(
                name=name,
                status=status,
                description=description,
                rollout_percentage=rollout_percentage,
                conditions=conditions
            )
            self.flags[name] = flag
            logger.info(f"创建特性开关: {name}")
            return flag
    
    def update_flag(self, name: str, **kwargs) -> bool:
        """更新特性开关"""
        with self.lock:
            if name not in self.flags:
                return False
            
            flag = self.flags[name]
            for key, value in kwargs.items():
                if hasattr(flag, key):
                    setattr(flag, key, value)
            
            flag.updated_at = datetime.now()
            logger.info(f"更新特性开关: {name}")
            return True
    
    def delete_flag(self, name: str) -> bool:
        """删除特性开关"""
        with self.lock:
            if name in self.flags:
                del self.flags[name]
                logger.info(f"删除特性开关: {name}")
                return True
            return False
    
    def is_enabled(self, flag_name: str, user_id: str = None, 
                  context: Dict[str, Any] = None) -> bool:
        """检查特性开关是否启用"""
        if flag_name not in self.flags:
            return False
        
        flag = self.flags[flag_name]
        
        if flag.status == FeatureFlagStatus.OFF:
            return False
        elif flag.status == FeatureFlagStatus.ON:
            return True
        elif flag.status == FeatureFlagStatus.CONDITIONAL:
            return self._evaluate_conditions(flag, user_id, context)
        
        return False
    
    def _evaluate_conditions(self, flag: FeatureFlag, user_id: str, 
                           context: Dict[str, Any] = None) -> bool:
        """评估条件"""
        # 检查用户白名单
        if 'whitelist' in flag.conditions and user_id in flag.conditions['whitelist']:
            return True
        
        # 检查用户黑名单
        if 'blacklist' in flag.conditions and user_id in flag.conditions['blacklist']:
            return False
        
        # 检查上下文条件
        if context:
            for condition_key, condition_value in flag.conditions.items():
                if condition_key in ['whitelist', 'blacklist']:
                    continue  # 跳过已经处理的白名单和黑名单
                
                if condition_key in context:
                    context_value = context[condition_key]
                    if isinstance(condition_value, list):
                        if context_value not in condition_value:
                            return False
                    else:
                        if context_value != condition_value:
                            return False
                else:
                    # 上下文缺少必需的条件
                    return False
        
        # 如果有上下文条件但没有百分比要求，匹配条件即可
        if context and any(key not in ['whitelist', 'blacklist'] for key in flag.conditions.keys()):
            # 如果有非白名单/黑名单的条件，并且上下文提供了这些条件，则检查是否匹配
            if context:
                for condition_key, condition_value in flag.conditions.items():
                    if condition_key in ['whitelist', 'blacklist']:
                        continue
                    if condition_key in context:
                        context_value = context[condition_key]
                        if isinstance(condition_value, list):
                            if context_value in condition_value:
                                return True
                        else:
                            if context_value == condition_value:
                                return True
                return False
        
        # 检查百分比
        if flag.rollout_percentage > 0:
            user_hash = hashlib.md5(f"{user_id}_{flag.name}".encode()).hexdigest()
            hash_int = int(user_hash[:8], 16)
            percentage = (hash_int % 10000) / 100.0
            return percentage < flag.rollout_percentage
        
        # 如果没有其他条件匹配，返回False
        return False
    
    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """获取特性开关"""
        return self.flags.get(name)
    
    def list_flags(self) -> List[FeatureFlag]:
        """列出所有特性开关"""
        return list(self.flags.values())


class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self):
        self.experiments = {}
        self.user_assignments = {}
        self.results = defaultdict(list)
        self.lock = threading.Lock()
    
    def create_experiment(self, name: str, description: str = "", 
                         groups: List[ABTestGroup] = None) -> str:
        """创建A/B测试实验"""
        experiment_id = str(uuid.uuid4())
        
        if groups is None:
            groups = [
                ABTestGroup(name="control", percentage=50.0),
                ABTestGroup(name="treatment", percentage=50.0)
            ]
        
        experiment = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'groups': groups,
            'status': ABTestStatus.PLANNING,
            'created_at': datetime.now(),
            'start_time': None,
            'end_time': None,
            'metrics': ExperimentMetrics()
        }
        
        with self.lock:
            self.experiments[experiment_id] = experiment
            logger.info(f"创建A/B测试实验: {name} (ID: {experiment_id})")
        
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """开始实验"""
        with self.lock:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            experiment['status'] = ABTestStatus.RUNNING
            experiment['start_time'] = datetime.now()
            logger.info(f"开始实验: {experiment['name']}")
            return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """停止实验"""
        with self.lock:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            experiment['status'] = ABTestStatus.FINISHED
            experiment['end_time'] = datetime.now()
            logger.info(f"停止实验: {experiment['name']}")
            return True
    
    def assign_user_to_group(self, experiment_id: str, user_id: str) -> Optional[str]:
        """为用户分配实验组"""
        if experiment_id not in self.experiments:
            return None
        
        # 检查是否已经分配
        if experiment_id in self.user_assignments and user_id in self.user_assignments[experiment_id]:
            return self.user_assignments[experiment_id][user_id]
        
        experiment = self.experiments[experiment_id]
        
        # 基于哈希的用户分配，确保一致性
        user_hash = hashlib.md5(f"{user_id}_{experiment_id}".encode()).hexdigest()
        hash_int = int(user_hash[:8], 16)
        percentage = (hash_int % 10000) / 100.0
        
        cumulative_percentage = 0.0
        assigned_group = None
        
        for group in experiment['groups']:
            cumulative_percentage += group.percentage
            if percentage < cumulative_percentage:
                assigned_group = group.name
                break
        
        if assigned_group is None and experiment['groups']:
            assigned_group = experiment['groups'][-1].name
        
        with self.lock:
            if experiment_id not in self.user_assignments:
                self.user_assignments[experiment_id] = {}
            self.user_assignments[experiment_id][user_id] = assigned_group
        
        logger.info(f"用户 {user_id} 分配到实验组 {assigned_group}")
        return assigned_group
    
    def record_result(self, experiment_id: str, user_id: str, 
                     group_name: str, success: bool = True, 
                     metrics: Dict[str, Any] = None):
        """记录实验结果"""
        result = ExperimentResult(
            experiment_id=experiment_id,
            user_id=user_id,
            group_name=group_name,
            timestamp=datetime.now(),
            success=success,
            metrics=metrics or {}
        )
        
        with self.lock:
            self.results[experiment_id].append(result)
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """获取实验信息"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """列出所有实验"""
        return list(self.experiments.values())


class ExperimentMonitor:
    """实验监控器"""
    
    def __init__(self):
        self.metrics = defaultdict(ExperimentMetrics)
        self.alerts = []
        self.thresholds = {}
        self.lock = threading.Lock()
    
    def update_metrics(self, experiment_id: str, group_name: str, 
                      success: bool, response_time: float = 0.0,
                      custom_metrics: Dict[str, Any] = None):
        """更新实验指标"""
        with self.lock:
            metrics = self.metrics[f"{experiment_id}_{group_name}"]
            metrics.total_users += 1
            
            if success:
                metrics.success_rate = (metrics.success_rate * (metrics.total_users - 1) + 1) / metrics.total_users
            else:
                metrics.success_rate = (metrics.success_rate * (metrics.total_users - 1)) / metrics.total_users
            
            metrics.error_rate = 1.0 - metrics.success_rate
            
            if response_time > 0:
                metrics.avg_response_time = (metrics.avg_response_time * (metrics.total_users - 1) + response_time) / metrics.total_users
            
            if custom_metrics:
                for key, value in custom_metrics.items():
                    if key not in metrics.custom_metrics:
                        metrics.custom_metrics[key] = []
                    if isinstance(value, list):
                        metrics.custom_metrics[key].extend(value)
                    else:
                        metrics.custom_metrics[key].append(value)
            
            # 检查告警阈值
            self._check_alerts(experiment_id, group_name, metrics)
    
    def _check_alerts(self, experiment_id: str, group_name: str, metrics: ExperimentMetrics):
        """检查告警"""
        if experiment_id not in self.thresholds:
            return
        
        threshold = self.thresholds[experiment_id]
        
        alerts = []
        
        if 'error_rate' in threshold and metrics.error_rate > threshold['error_rate']:
            alerts.append(f"错误率超过阈值: {metrics.error_rate:.2%} > {threshold['error_rate']:.2%}")
        
        if 'success_rate' in threshold and metrics.success_rate < threshold['success_rate']:
            alerts.append(f"成功率低于阈值: {metrics.success_rate:.2%} < {threshold['success_rate']:.2%}")
        
        if 'response_time' in threshold and metrics.avg_response_time > threshold['response_time']:
            alerts.append(f"响应时间超过阈值: {metrics.avg_response_time:.2f}s > {threshold['response_time']}s")
        
        if alerts:
            self.alerts.extend([{
                'experiment_id': experiment_id,
                'group_name': group_name,
                'message': alert,
                'timestamp': datetime.now()
            } for alert in alerts])
            
            logger.warning(f"实验 {experiment_id} 组 {group_name} 触发告警: {alerts}")
    
    def set_threshold(self, experiment_id: str, threshold: Dict[str, float]):
        """设置告警阈值"""
        self.thresholds[experiment_id] = threshold
        logger.info(f"为实验 {experiment_id} 设置告警阈值: {threshold}")
    
    def get_metrics(self, experiment_id: str, group_name: str) -> Optional[ExperimentMetrics]:
        """获取实验指标"""
        return self.metrics.get(f"{experiment_id}_{group_name}")
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取告警列表"""
        return self.alerts.copy()


class ExperimentReporter:
    """实验报告生成器"""
    
    def __init__(self, monitor: ExperimentMonitor):
        self.monitor = monitor
    
    def generate_report(self, experiment_id: str, ab_manager: ABTestManager) -> Dict[str, Any]:
        """生成实验报告"""
        experiment = ab_manager.get_experiment(experiment_id)
        if not experiment:
            return {}
        
        report = {
            'experiment_info': {
                'id': experiment['id'],
                'name': experiment['name'],
                'description': experiment['description'],
                'status': experiment['status'].value,
                'created_at': experiment['created_at'].isoformat(),
                'start_time': experiment['start_time'].isoformat() if experiment['start_time'] else None,
                'end_time': experiment['end_time'].isoformat() if experiment['end_time'] else None
            },
            'group_metrics': {},
            'comparison': {},
            'recommendations': []
        }
        
        # 收集各组指标
        for group in experiment['groups']:
            metrics = self.monitor.get_metrics(experiment_id, group.name)
            if metrics:
                report['group_metrics'][group.name] = {
                    'total_users': metrics.total_users,
                    'success_rate': f"{metrics.success_rate:.2%}",
                    'error_rate': f"{metrics.error_rate:.2%}",
                    'avg_response_time': f"{metrics.avg_response_time:.2f}s",
                    'custom_metrics': {k: f"{sum([x for x in v if isinstance(x, (int, float))])/len([x for x in v if isinstance(x, (int, float))]):.2f}" if v and any(isinstance(x, (int, float)) for x in v) else "0" 
                                     for k, v in metrics.custom_metrics.items()}
                }
        
        # 生成比较分析
        if len(report['group_metrics']) >= 2:
            groups = list(report['group_metrics'].keys())
            control_group = groups[0]
            treatment_group = groups[1]
            
            control_metrics = report['group_metrics'][control_group]
            treatment_metrics = report['group_metrics'][treatment_group]
            
            # 成功率比较
            control_success = float(control_metrics['success_rate'].rstrip('%')) / 100
            treatment_success = float(treatment_metrics['success_rate'].rstrip('%')) / 100
            success_improvement = (treatment_success - control_success) / control_success if control_success > 0 else 0
            
            report['comparison'] = {
                'success_rate_improvement': f"{success_improvement:.2%}",
                'winner': treatment_group if success_improvement > 0 else control_group
            }
            
            # 生成建议
            if success_improvement > 0.05:  # 5%提升
                report['recommendations'].append("建议推广治疗组功能")
            elif success_improvement < -0.05:  # 5%下降
                report['recommendations'].append("建议停止治疗组实验")
            else:
                report['recommendations'].append("需要更多数据来判断效果")
        
        return report
    
    def export_report(self, experiment_id: str, ab_manager: ABTestManager, 
                     format_type: str = 'json') -> str:
        """导出实验报告"""
        report = self.generate_report(experiment_id, ab_manager)
        
        if format_type == 'json':
            return json.dumps(report, indent=2, ensure_ascii=False)
        elif format_type == 'markdown':
            return self._format_as_markdown(report)
        else:
            raise ValueError(f"不支持的报告格式: {format_type}")
    
    def _format_as_markdown(self, report: Dict[str, Any]) -> str:
        """格式化为Markdown"""
        md = []
        md.append(f"# 实验报告: {report['experiment_info']['name']}")
        md.append("")
        md.append(f"**实验ID:** {report['experiment_info']['id']}")
        md.append(f"**状态:** {report['experiment_info']['status']}")
        md.append(f"**创建时间:** {report['experiment_info']['created_at']}")
        md.append("")
        
        md.append("## 实验组指标")
        md.append("")
        for group_name, metrics in report['group_metrics'].items():
            md.append(f"### {group_name}")
            md.append(f"- 用户数: {metrics['total_users']}")
            md.append(f"- 成功率: {metrics['success_rate']}")
            md.append(f"- 错误率: {metrics['error_rate']}")
            md.append(f"- 平均响应时间: {metrics['avg_response_time']}")
            md.append("")
        
        if report['comparison']:
            md.append("## 比较分析")
            md.append("")
            md.append(f"**成功率提升:** {report['comparison']['success_rate_improvement']}")
            md.append(f"**优胜组:** {report['comparison']['winner']}")
            md.append("")
        
        if report['recommendations']:
            md.append("## 建议")
            md.append("")
            for rec in report['recommendations']:
                md.append(f"- {rec}")
        
        return "\n".join(md)


class ExperimentalFeatureSlot:
    """实验性功能插槽主类"""
    
    def __init__(self):
        self.feature_toggle = FeatureToggle()
        self.ab_test_manager = ABTestManager()
        self.experiment_config = ExperimentConfig()
        self.experiment_monitor = ExperimentMonitor()
        self.experiment_reporter = ExperimentReporter(self.experiment_monitor)
        self.experiment_security = ExperimentSecurity()
        self.active_experiments = {}
        self.lock = threading.Lock()
    
    # 特性开关相关方法
    def create_feature_flag(self, name: str, status: FeatureFlagStatus = FeatureFlagStatus.OFF,
                          description: str = "", rollout_percentage: float = 0.0,
                          conditions: Dict[str, Any] = None) -> FeatureFlag:
        """创建特性开关"""
        return self.feature_toggle.create_flag(name, status, description, rollout_percentage, conditions)
    
    def enable_feature(self, name: str) -> bool:
        """启用特性"""
        return self.feature_toggle.update_flag(name, status=FeatureFlagStatus.ON)
    
    def disable_feature(self, name: str) -> bool:
        """禁用特性"""
        return self.feature_toggle.update_flag(name, status=FeatureFlagStatus.OFF)
    
    def is_feature_enabled(self, name: str, user_id: str = None, 
                          context: Dict[str, Any] = None) -> bool:
        """检查特性是否启用"""
        return self.feature_toggle.is_enabled(name, user_id, context)
    
    # A/B测试相关方法
    def create_ab_test(self, name: str, description: str = "", 
                      groups: List[ABTestGroup] = None) -> str:
        """创建A/B测试"""
        return self.ab_test_manager.create_experiment(name, description, groups)
    
    def start_ab_test(self, experiment_id: str) -> bool:
        """开始A/B测试"""
        return self.ab_test_manager.start_experiment(experiment_id)
    
    def stop_ab_test(self, experiment_id: str) -> bool:
        """停止A/B测试"""
        return self.ab_test_manager.stop_experiment(experiment_id)
    
    def get_user_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """获取用户实验变体"""
        return self.ab_test_manager.assign_user_to_group(experiment_id, user_id)
    
    def record_ab_test_result(self, experiment_id: str, user_id: str, 
                             success: bool = True, metrics: Dict[str, Any] = None):
        """记录A/B测试结果"""
        group = self.get_user_variant(experiment_id, user_id)
        if group:
            self.ab_test_manager.record_result(experiment_id, user_id, group, success, metrics)
            # 更新监控指标
            self.experiment_monitor.update_metrics(experiment_id, group, success, 0.0, metrics)
    
    # 实验配置相关方法
    def configure_experiment(self, experiment_id: str, config: Dict[str, Any]):
        """配置实验"""
        self.experiment_config.set_config(experiment_id, config)
    
    def get_experiment_config(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验配置"""
        return self.experiment_config.get_config(experiment_id)
    
    # 实验监控相关方法
    def set_monitoring_threshold(self, experiment_id: str, threshold: Dict[str, float]):
        """设置监控阈值"""
        self.experiment_monitor.set_threshold(experiment_id, threshold)
    
    def get_experiment_metrics(self, experiment_id: str, group_name: str) -> Optional[ExperimentMetrics]:
        """获取实验指标"""
        return self.experiment_monitor.get_metrics(experiment_id, group_name)
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取告警"""
        return self.experiment_monitor.get_alerts()
    
    # 实验报告相关方法
    def generate_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """生成实验报告"""
        return self.experiment_reporter.generate_report(experiment_id, self.ab_test_manager)
    
    def export_experiment_report(self, experiment_id: str, format_type: str = 'json') -> str:
        """导出实验报告"""
        return self.experiment_reporter.export_report(experiment_id, self.ab_test_manager, format_type)
    
    # 实验安全相关方法
    def add_security_rule(self, experiment_id: str, rule: Dict[str, Any]):
        """添加安全规则"""
        self.experiment_security.add_security_rule(experiment_id, rule)
    
    def validate_experiment_access(self, experiment_id: str, user_id: str, action: str) -> bool:
        """验证实验访问权限"""
        return self.experiment_security.validate_experiment_access(experiment_id, user_id, action)
    
    def register_user_permissions(self, user_id: str, permissions: Dict[str, Any]):
        """注册用户权限"""
        self.experiment_security.register_user_permission(user_id, permissions)
    
    # 高级功能
    def run_feature_with_ab_test(self, feature_name: str, experiment_id: str, 
                                user_id: str, feature_func: Callable, 
                                context: Dict[str, Any] = None) -> Any:
        """运行带A/B测试的功能"""
        # 检查特性开关
        if not self.is_feature_enabled(feature_name, user_id, context):
            return None
        
        # 获取用户变体
        variant = self.get_user_variant(experiment_id, user_id)
        if not variant:
            return None
        
        # 准备变体上下文
        variant_context = context or {}
        variant_context['ab_variant'] = variant
        
        try:
            # 执行功能
            result = feature_func(user_id, variant_context)
            # 记录成功结果
            self.record_ab_test_result(experiment_id, user_id, success=True)
            return result
        except Exception as e:
            # 记录失败结果
            self.record_ab_test_result(experiment_id, user_id, success=False, 
                                     metrics={'error': str(e)})
            logger.error(f"功能执行失败: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'feature_flags_count': len(self.feature_toggle.flags),
            'active_experiments_count': len([exp for exp in self.ab_test_manager.experiments.values() 
                                           if exp['status'] == ABTestStatus.RUNNING]),
            'total_experiments_count': len(self.ab_test_manager.experiments),
            'alerts_count': len(self.experiment_monitor.alerts),
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup_expired_experiments(self):
        """清理过期实验"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        expired_experiments = []
        for exp_id, experiment in self.ab_test_manager.experiments.items():
            if (experiment['status'] == ABTestStatus.FINISHED and 
                experiment['end_time'] and experiment['end_time'] < cutoff_date):
                expired_experiments.append(exp_id)
        
        for exp_id in expired_experiments:
            with self.lock:
                del self.ab_test_manager.experiments[exp_id]
                if exp_id in self.ab_test_manager.user_assignments:
                    del self.ab_test_manager.user_assignments[exp_id]
                if exp_id in self.ab_test_manager.results:
                    del self.ab_test_manager.results[exp_id]
        
        logger.info(f"清理了 {len(expired_experiments)} 个过期实验")
        return len(expired_experiments)