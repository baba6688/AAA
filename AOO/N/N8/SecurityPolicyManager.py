#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N8安全策略管理器 (Security Policy Manager)

该模块实现了一个全面的安全策略管理系统，包括：
1. 安全策略定义 - 定义各种安全策略规则
2. 安全策略执行 - 执行安全策略检查和验证
3. 安全策略更新 - 动态更新和修改安全策略
4. 安全策略冲突检测 - 检测策略间的冲突
5. 安全策略版本管理 - 管理策略版本历史
6. 安全策略评估 - 评估策略有效性
7. 安全策略优化 - 优化策略性能
8. 安全策略审计 - 记录和审计策略操作
9. 安全策略报告 - 生成策略报告


创建时间: 2025-11-06
版本: 1.0.0
"""

import json
import hashlib
import uuid
import datetime
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import copy


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """安全策略类型枚举"""
    ACCESS_CONTROL = "access_control"  # 访问控制策略
    DATA_PROTECTION = "data_protection"  # 数据保护策略
    SYSTEM_SECURITY = "system_security"  # 系统安全策略
    COMPLIANCE = "compliance"  # 合规性策略
    AUDIT = "audit"  # 审计策略
    INCIDENT_RESPONSE = "incident_response"  # 事件响应策略


class PolicyStatus(Enum):
    """策略状态枚举"""
    ACTIVE = "active"  # 激活状态
    INACTIVE = "inactive"  # 非激活状态
    PENDING = "pending"  # 待审核状态
    DEPRECATED = "deprecated"  # 已废弃状态


class PolicyPriority(Enum):
    """策略优先级枚举"""
    CRITICAL = 1  # 关键级别
    HIGH = 2  # 高优先级
    MEDIUM = 3  # 中等优先级
    LOW = 4  # 低优先级


@dataclass
class PolicyCondition:
    """策略条件定义"""
    field: str  # 字段名
    operator: str  # 操作符 (==, !=, >, <, >=, <=, in, not in, contains)
    value: Any  # 条件值
    description: str = ""  # 条件描述


@dataclass
class PolicyAction:
    """策略动作定义"""
    action_type: str  # 动作类型 (allow, deny, log, alert, quarantine)
    parameters: Dict[str, Any]  # 动作参数
    description: str = ""  # 动作描述


@dataclass
class SecurityPolicy:
    """安全策略数据类"""
    id: str  # 策略ID
    name: str  # 策略名称
    type: PolicyType  # 策略类型
    description: str  # 策略描述
    conditions: List[PolicyCondition]  # 策略条件
    actions: List[PolicyAction]  # 策略动作
    priority: PolicyPriority  # 策略优先级
    status: PolicyStatus  # 策略状态
    version: str  # 策略版本
    created_at: datetime.datetime  # 创建时间
    updated_at: datetime.datetime  # 更新时间
    created_by: str  # 创建者
    updated_by: str = ""  # 更新者
    tags: List[str] = None  # 策略标签
    metadata: Dict[str, Any] = None  # 策略元数据

    def __post_init__(self):
        """初始化后处理"""
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PolicyEvaluationResult:
    """策略评估结果"""
    policy_id: str  # 策略ID
    matched: bool  # 是否匹配
    actions_executed: List[PolicyAction]  # 执行的动作
    evaluation_time: datetime.datetime  # 评估时间
    context: Dict[str, Any]  # 评估上下文
    result: bool  # 评估结果 (True=允许, False=拒绝)
    reason: str  # 结果原因


@dataclass
class PolicyConflict:
    """策略冲突信息"""
    policy1_id: str  # 冲突策略1 ID
    policy2_id: str  # 冲突策略2 ID
    conflict_type: str  # 冲突类型
    severity: str  # 严重程度
    description: str  # 冲突描述
    resolution_suggestion: str = ""  # 解决建议


@dataclass
class PolicyAuditLog:
    """策略审计日志"""
    id: str  # 日志ID
    timestamp: datetime.datetime  # 时间戳
    operation: str  # 操作类型 (create, update, delete, execute)
    policy_id: str = ""  # 策略ID
    user_id: str = ""  # 用户ID
    details: Dict[str, Any] = None  # 详细信息
    result: str = ""  # 操作结果

    def __post_init__(self):
        """初始化后处理"""
        if self.details is None:
            self.details = {}


class PolicyEngine(ABC):
    """策略引擎抽象基类"""
    
    @abstractmethod
    def evaluate(self, policy: SecurityPolicy, context: Dict[str, Any]) -> PolicyEvaluationResult:
        """评估策略"""
        pass


class BasicPolicyEngine(PolicyEngine):
    """基础策略引擎实现"""
    
    def evaluate(self, policy: SecurityPolicy, context: Dict[str, Any]) -> PolicyEvaluationResult:
        """评估策略"""
        start_time = datetime.datetime.now()
        
        # 检查策略状态
        if policy.status != PolicyStatus.ACTIVE:
            return PolicyEvaluationResult(
                policy_id=policy.id,
                matched=False,
                actions_executed=[],
                evaluation_time=datetime.datetime.now(),
                context=context,
                result=True,
                reason=f"策略状态为{policy.status.value}，未激活"
            )
        
        # 评估条件
        conditions_met = self._evaluate_conditions(policy.conditions, context)
        
        if not conditions_met:
            return PolicyEvaluationResult(
                policy_id=policy.id,
                matched=False,
                actions_executed=[],
                evaluation_time=datetime.datetime.now(),
                context=context,
                result=True,
                reason="策略条件不匹配"
            )
        
        # 执行动作
        executed_actions = self._execute_actions(policy.actions, context)
        
        # 确定最终结果
        result = self._determine_result(policy.actions)
        
        return PolicyEvaluationResult(
            policy_id=policy.id,
            matched=True,
            actions_executed=executed_actions,
            evaluation_time=datetime.datetime.now(),
            context=context,
            result=result,
            reason="策略条件匹配并执行相应动作"
        )
    
    def _evaluate_conditions(self, conditions: List[PolicyCondition], context: Dict[str, Any]) -> bool:
        """评估条件"""
        for condition in conditions:
            if not self._evaluate_single_condition(condition, context):
                return False
        return True
    
    def _evaluate_single_condition(self, condition: PolicyCondition, context: Dict[str, Any]) -> bool:
        """评估单个条件"""
        field_value = self._get_nested_value(context, condition.field)
        
        if field_value is None:
            return False
        
        try:
            if condition.operator == "==":
                return field_value == condition.value
            elif condition.operator == "!=":
                return field_value != condition.value
            elif condition.operator == ">":
                return field_value > condition.value
            elif condition.operator == "<":
                return field_value < condition.value
            elif condition.operator == ">=":
                return field_value >= condition.value
            elif condition.operator == "<=":
                return field_value <= condition.value
            elif condition.operator == "in":
                return field_value in condition.value
            elif condition.operator == "not in":
                return field_value not in condition.value
            elif condition.operator == "contains":
                return condition.value in str(field_value)
            else:
                logger.warning(f"未知的操作符: {condition.operator}")
                return False
        except Exception as e:
            logger.error(f"条件评估错误: {e}")
            return False
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """获取嵌套字典值"""
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _execute_actions(self, actions: List[PolicyAction], context: Dict[str, Any]) -> List[PolicyAction]:
        """执行动作"""
        executed = []
        for action in actions:
            try:
                # 这里可以添加具体的动作执行逻辑
                logger.info(f"执行动作: {action.action_type} - {action.description}")
                executed.append(action)
            except Exception as e:
                logger.error(f"动作执行失败: {e}")
        return executed
    
    def _determine_result(self, actions: List[PolicyAction]) -> bool:
        """确定最终结果"""
        # 如果有任何deny动作，则拒绝
        for action in actions:
            if action.action_type == "deny":
                return False
        # 如果有allow动作，则允许
        for action in actions:
            if action.action_type == "allow":
                return True
        # 默认允许
        return True


class SecurityPolicyManager:
    """N8安全策略管理器主类"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化安全策略管理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.policies: Dict[str, SecurityPolicy] = {}
        self.policy_versions: Dict[str, List[SecurityPolicy]] = defaultdict(list)
        self.audit_logs: List[PolicyAuditLog] = []
        self.policy_engine = BasicPolicyEngine()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.RLock()
        
        # 初始化默认策略
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """初始化默认安全策略"""
        default_policies = [
            # 访问控制策略
            SecurityPolicy(
                id=str(uuid.uuid4()),
                name="管理员访问控制",
                type=PolicyType.ACCESS_CONTROL,
                description="限制管理员权限访问",
                conditions=[
                    PolicyCondition("user.role", "==", "admin", "用户角色为管理员"),
                    PolicyCondition("resource.type", "in", ["system", "config"], "资源类型为系统或配置")
                ],
                actions=[
                    PolicyAction("allow", {"scope": "read"}, "允许读取"),
                    PolicyAction("log", {"level": "info"}, "记录访问日志")
                ],
                priority=PolicyPriority.HIGH,
                status=PolicyStatus.ACTIVE,
                version="1.0.0",
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                created_by="system"
            ),
            # 数据保护策略
            SecurityPolicy(
                id=str(uuid.uuid4()),
                name="敏感数据保护",
                type=PolicyType.DATA_PROTECTION,
                description="保护敏感数据不被未授权访问",
                conditions=[
                    PolicyCondition("data.classification", "==", "sensitive", "数据分类为敏感"),
                    PolicyCondition("user.clearance_level", "<", 3, "用户 clearance 级别低于3")
                ],
                actions=[
                    PolicyAction("deny", {"reason": "insufficient_clearance"}, "拒绝访问"),
                    PolicyAction("alert", {"level": "high"}, "发送高级别警报")
                ],
                priority=PolicyPriority.CRITICAL,
                status=PolicyStatus.ACTIVE,
                version="1.0.0",
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                created_by="system"
            )
        ]
        
        for policy in default_policies:
            self.create_policy(policy)
    
    # ==================== 1. 安全策略定义 ====================
    
    def create_policy(self, policy: SecurityPolicy) -> bool:
        """
        创建新的安全策略
        
        Args:
            policy: 安全策略对象
            
        Returns:
            bool: 创建是否成功
        """
        try:
            with self.lock:
                # 检查策略ID是否已存在
                if policy.id in self.policies:
                    logger.error(f"策略ID {policy.id} 已存在")
                    return False
                
                # 验证策略
                if not self._validate_policy(policy):
                    logger.error(f"策略验证失败: {policy.name}")
                    return False
                
                # 保存策略
                self.policies[policy.id] = policy
                
                # 保存版本历史
                self.policy_versions[policy.id].append(copy.deepcopy(policy))
                
                # 记录审计日志
                self._log_audit("create", policy.id, "system", {"policy_name": policy.name}, "success")
                
                logger.info(f"成功创建策略: {policy.name} ({policy.id})")
                return True
                
        except Exception as e:
            logger.error(f"创建策略失败: {e}")
            self._log_audit("create", policy.id, "system", {"error": str(e)}, "failed")
            return False
    
    def _validate_policy(self, policy: SecurityPolicy) -> bool:
        """验证策略有效性"""
        # 检查必要字段
        if not all([policy.id, policy.name, policy.type, policy.description]):
            logger.error("策略缺少必要字段")
            return False
        
        # 检查条件格式
        for condition in policy.conditions:
            if not all([condition.field, condition.operator]):
                logger.error(f"条件格式错误: {condition}")
                return False
        
        # 检查动作格式
        for action in policy.actions:
            if not action.action_type:
                logger.error(f"动作格式错误: {action}")
                return False
        
        return True
    
    def define_policy_template(self, template_name: str, template_data: Dict[str, Any]) -> str:
        """
        定义策略模板
        
        Args:
            template_name: 模板名称
            template_data: 模板数据
            
        Returns:
            str: 生成的策略ID
        """
        policy_id = str(uuid.uuid4())
        
        # 根据模板数据创建策略
        policy = SecurityPolicy(
            id=policy_id,
            name=template_data.get("name", f"模板策略-{template_name}"),
            type=PolicyType(template_data.get("type", "access_control")),
            description=template_data.get("description", ""),
            conditions=[
                PolicyCondition(**cond) for cond in template_data.get("conditions", [])
            ],
            actions=[
                PolicyAction(**action) for action in template_data.get("actions", [])
            ],
            priority=PolicyPriority(template_data.get("priority", "medium")),
            status=PolicyStatus(template_data.get("status", "pending")),
            version=template_data.get("version", "1.0.0"),
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now(),
            created_by=template_data.get("created_by", "template"),
            tags=template_data.get("tags", []),
            metadata=template_data.get("metadata", {})
        )
        
        if self.create_policy(policy):
            logger.info(f"根据模板创建策略成功: {template_name}")
            return policy_id
        else:
            logger.error(f"根据模板创建策略失败: {template_name}")
            return ""
    
    # ==================== 2. 安全策略执行 ====================
    
    def evaluate_policies(self, context: Dict[str, Any]) -> List[PolicyEvaluationResult]:
        """
        评估所有适用策略
        
        Args:
            context: 评估上下文
            
        Returns:
            List[PolicyEvaluationResult]: 评估结果列表
        """
        results = []
        
        # 并行评估策略
        futures = []
        for policy in self.policies.values():
            future = self.executor.submit(self._evaluate_single_policy, policy, context)
            futures.append(future)
        
        # 收集结果
        for future in futures:
            try:
                result = future.result(timeout=5.0)  # 5秒超时
                results.append(result)
            except Exception as e:
                logger.error(f"策略评估失败: {e}")
        
        return results
    
    def _evaluate_single_policy(self, policy: SecurityPolicy, context: Dict[str, Any]) -> PolicyEvaluationResult:
        """评估单个策略"""
        try:
            return self.policy_engine.evaluate(policy, context)
        except Exception as e:
            logger.error(f"策略 {policy.id} 评估错误: {e}")
            return PolicyEvaluationResult(
                policy_id=policy.id,
                matched=False,
                actions_executed=[],
                evaluation_time=datetime.datetime.now(),
                context=context,
                result=True,
                reason=f"评估错误: {str(e)}"
            )
    
    def execute_policy_action(self, action: PolicyAction, context: Dict[str, Any]) -> bool:
        """
        执行策略动作
        
        Args:
            action: 策略动作
            context: 执行上下文
            
        Returns:
            bool: 执行是否成功
        """
        try:
            if action.action_type == "allow":
                logger.info(f"允许操作: {action.description}")
                return True
            elif action.action_type == "deny":
                logger.warning(f"拒绝操作: {action.description}")
                return False
            elif action.action_type == "log":
                logger.info(f"记录日志: {action.description} - {context}")
                return True
            elif action.action_type == "alert":
                logger.warning(f"发送警报: {action.description} - {context}")
                return True
            elif action.action_type == "quarantine":
                logger.error(f"隔离资源: {action.description}")
                return True
            else:
                logger.warning(f"未知动作类型: {action.action_type}")
                return False
        except Exception as e:
            logger.error(f"动作执行失败: {e}")
            return False
    
    # ==================== 3. 安全策略更新 ====================
    
    def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新安全策略
        
        Args:
            policy_id: 策略ID
            updates: 更新内容
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self.lock:
                if policy_id not in self.policies:
                    logger.error(f"策略不存在: {policy_id}")
                    return False
                
                policy = self.policies[policy_id]
                old_policy = copy.deepcopy(policy)
                
                # 应用更新
                for key, value in updates.items():
                    if hasattr(policy, key):
                        setattr(policy, key, value)
                
                policy.updated_at = datetime.datetime.now()
                
                # 验证更新后的策略
                if not self._validate_policy(policy):
                    # 回滚更新
                    self.policies[policy_id] = old_policy
                    logger.error(f"更新后策略验证失败，已回滚: {policy_id}")
                    return False
                
                # 保存版本历史
                self.policy_versions[policy_id].append(copy.deepcopy(policy))
                
                # 记录审计日志
                self._log_audit("update", policy_id, "system", {
                    "old_policy": asdict(old_policy),
                    "new_policy": asdict(policy)
                }, "success")
                
                logger.info(f"成功更新策略: {policy.name} ({policy_id})")
                return True
                
        except Exception as e:
            logger.error(f"更新策略失败: {e}")
            self._log_audit("update", policy_id, "system", {"error": str(e)}, "failed")
            return False
    
    def deactivate_policy(self, policy_id: str) -> bool:
        """停用策略"""
        return self.update_policy(policy_id, {"status": PolicyStatus.INACTIVE})
    
    def activate_policy(self, policy_id: str) -> bool:
        """激活策略"""
        return self.update_policy(policy_id, {"status": PolicyStatus.ACTIVE})
    
    def delete_policy(self, policy_id: str) -> bool:
        """
        删除策略
        
        Args:
            policy_id: 策略ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            with self.lock:
                if policy_id not in self.policies:
                    logger.error(f"策略不存在: {policy_id}")
                    return False
                
                policy = self.policies[policy_id]
                
                # 删除策略
                del self.policies[policy_id]
                
                # 记录审计日志
                self._log_audit("delete", policy_id, "system", {"policy_name": policy.name}, "success")
                
                logger.info(f"成功删除策略: {policy.name} ({policy_id})")
                return True
                
        except Exception as e:
            logger.error(f"删除策略失败: {e}")
            self._log_audit("delete", policy_id, "system", {"error": str(e)}, "failed")
            return False
    
    # ==================== 4. 安全策略冲突检测 ====================
    
    def detect_conflicts(self) -> List[PolicyConflict]:
        """
        检测策略冲突
        
        Returns:
            List[PolicyConflict]: 冲突列表
        """
        conflicts = []
        
        try:
            policy_list = list(self.policies.values())
            
            # 比较所有策略对
            for i, policy1 in enumerate(policy_list):
                for policy2 in policy_list[i+1:]:
                    conflict = self._check_policy_conflict(policy1, policy2)
                    if conflict:
                        conflicts.append(conflict)
            
            logger.info(f"检测到 {len(conflicts)} 个策略冲突")
            return conflicts
            
        except Exception as e:
            logger.error(f"冲突检测失败: {e}")
            return []
    
    def _check_policy_conflict(self, policy1: SecurityPolicy, policy2: SecurityPolicy) -> Optional[PolicyConflict]:
        """检查两个策略是否冲突"""
        # 检查条件重叠
        if self._conditions_overlap(policy1.conditions, policy2.conditions):
            # 检查动作冲突
            if self._actions_conflict(policy1.actions, policy2.actions):
                return PolicyConflict(
                    policy1_id=policy1.id,
                    policy2_id=policy2.id,
                    conflict_type="condition_action_conflict",
                    severity="high",
                    description=f"策略 {policy1.name} 和 {policy2.name} 存在条件重叠和动作冲突",
                    resolution_suggestion="调整策略优先级或修改条件/动作"
                )
        
        # 检查优先级冲突
        if policy1.priority == policy2.priority and policy1.status == policy2.status:
            if self._similar_conditions(policy1.conditions, policy2.conditions):
                return PolicyConflict(
                    policy1_id=policy1.id,
                    policy2_id=policy2.id,
                    conflict_type="priority_conflict",
                    severity="medium",
                    description=f"策略 {policy1.name} 和 {policy2.name} 优先级相同且条件相似",
                    resolution_suggestion="调整策略优先级或区分条件"
                )
        
        return None
    
    def _conditions_overlap(self, conditions1: List[PolicyCondition], conditions2: List[PolicyCondition]) -> bool:
        """检查条件是否重叠"""
        for cond1 in conditions1:
            for cond2 in conditions2:
                if cond1.field == cond2.field and cond1.operator == cond2.operator:
                    # 检查值是否有重叠
                    if self._values_overlap(cond1.value, cond2.value):
                        return True
        return False
    
    def _values_overlap(self, value1: Any, value2: Any) -> bool:
        """检查值是否重叠"""
        try:
            if value1 == value2:
                return True
            if isinstance(value1, list) and isinstance(value2, list):
                return bool(set(value1) & set(value2))
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                return abs(value1 - value2) < 0.001  # 浮点数容差
            return False
        except:
            return False
    
    def _actions_conflict(self, actions1: List[PolicyAction], actions2: List[PolicyAction]) -> bool:
        """检查动作是否冲突"""
        allow_actions1 = any(action.action_type == "allow" for action in actions1)
        deny_actions2 = any(action.action_type == "deny" for action in actions2)
        deny_actions1 = any(action.action_type == "deny" for action in actions1)
        allow_actions2 = any(action.action_type == "allow" for action in actions2)
        
        return (allow_actions1 and deny_actions2) or (deny_actions1 and allow_actions2)
    
    def _similar_conditions(self, conditions1: List[PolicyCondition], conditions2: List[PolicyCondition]) -> bool:
        """检查条件是否相似"""
        if len(conditions1) != len(conditions2):
            return False
        
        common_fields = 0
        for cond1 in conditions1:
            for cond2 in conditions2:
                if cond1.field == cond2.field and cond1.operator == cond2.operator:
                    common_fields += 1
                    break
        
        return common_fields >= len(conditions1) * 0.5  # 50%以上字段相同
    
    # ==================== 5. 安全策略版本管理 ====================
    
    def get_policy_versions(self, policy_id: str) -> List[SecurityPolicy]:
        """
        获取策略版本历史
        
        Args:
            policy_id: 策略ID
            
        Returns:
            List[SecurityPolicy]: 版本历史列表
        """
        return self.policy_versions.get(policy_id, [])
    
    def rollback_policy(self, policy_id: str, version: str) -> bool:
        """
        回滚策略到指定版本
        
        Args:
            policy_id: 策略ID
            version: 目标版本
            
        Returns:
            bool: 回滚是否成功
        """
        try:
            versions = self.policy_versions.get(policy_id, [])
            target_policy = None
            
            for v in versions:
                if v.version == version:
                    target_policy = copy.deepcopy(v)
                    break
            
            if not target_policy:
                logger.error(f"未找到策略 {policy_id} 的版本 {version}")
                return False
            
            # 更新当前策略
            target_policy.updated_at = datetime.datetime.now()
            target_policy.updated_by = "rollback"
            
            self.policies[policy_id] = target_policy
            
            # 记录审计日志
            self._log_audit("rollback", policy_id, "system", {
                "target_version": version,
                "current_version": target_policy.version
            }, "success")
            
            logger.info(f"成功回滚策略 {policy_id} 到版本 {version}")
            return True
            
        except Exception as e:
            logger.error(f"策略回滚失败: {e}")
            self._log_audit("rollback", policy_id, "system", {"error": str(e)}, "failed")
            return False
    
    def compare_policy_versions(self, policy_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        比较策略版本差异
        
        Args:
            policy_id: 策略ID
            version1: 版本1
            version2: 版本2
            
        Returns:
            Dict[str, Any]: 版本差异信息
        """
        try:
            versions = self.policy_versions.get(policy_id, [])
            policy1 = None
            policy2 = None
            
            for v in versions:
                if v.version == version1:
                    policy1 = v
                elif v.version == version2:
                    policy2 = v
            
            if not policy1 or not policy2:
                return {"error": "未找到指定版本"}
            
            diff = {
                "policy_id": policy_id,
                "version1": version1,
                "version2": version2,
                "differences": []
            }
            
            # 比较字段差异
            fields_to_compare = ["name", "description", "priority", "status", "conditions", "actions"]
            
            for field in fields_to_compare:
                value1 = getattr(policy1, field)
                value2 = getattr(policy2, field)
                
                if value1 != value2:
                    diff["differences"].append({
                        "field": field,
                        "old_value": value1,
                        "new_value": value2
                    })
            
            return diff
            
        except Exception as e:
            logger.error(f"版本比较失败: {e}")
            return {"error": str(e)}
    
    # ==================== 6. 安全策略评估 ====================
    
    def evaluate_policy_effectiveness(self, policy_id: str, time_range: Tuple[datetime.datetime, datetime.datetime]) -> Dict[str, Any]:
        """
        评估策略有效性
        
        Args:
            policy_id: 策略ID
            time_range: 评估时间范围
            
        Returns:
            Dict[str, Any]: 有效性评估结果
        """
        try:
            # 获取审计日志
            relevant_logs = [
                log for log in self.audit_logs
                if log.policy_id == policy_id and log.timestamp >= time_range[0] and log.timestamp <= time_range[1]
            ]
            
            # 计算统计信息
            total_executions = len([log for log in relevant_logs if log.operation == "execute"])
            successful_executions = len([log for log in relevant_logs if log.result == "success"])
            failed_executions = len([log for log in relevant_logs if log.result == "failed"])
            
            # 计算性能指标
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            
            # 评估结果
            effectiveness_score = self._calculate_effectiveness_score(
                success_rate, total_executions, failed_executions
            )
            
            return {
                "policy_id": policy_id,
                "time_range": {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat()
                },
                "statistics": {
                    "total_executions": total_executions,
                    "successful_executions": successful_executions,
                    "failed_executions": failed_executions,
                    "success_rate": success_rate
                },
                "effectiveness_score": effectiveness_score,
                "recommendation": self._generate_effectiveness_recommendation(effectiveness_score)
            }
            
        except Exception as e:
            logger.error(f"策略有效性评估失败: {e}")
            return {"error": str(e)}
    
    def _calculate_effectiveness_score(self, success_rate: float, total_executions: int, failed_executions: int) -> float:
        """计算有效性分数"""
        # 基础分数来自成功率
        base_score = success_rate
        
        # 根据执行次数调整分数
        if total_executions == 0:
            execution_factor = 0.5  # 没有执行记录，给中等分数
        elif total_executions < 10:
            execution_factor = 0.8  # 执行次数较少
        elif total_executions < 100:
            execution_factor = 1.0  # 执行次数适中
        else:
            execution_factor = 1.2  # 执行次数较多，有足够数据
        
        # 根据失败次数调整分数
        failure_rate = (failed_executions / total_executions * 100) if total_executions > 0 else 0
        failure_factor = max(0.5, 1.0 - failure_rate / 100)
        
        return min(100.0, base_score * execution_factor * failure_factor)
    
    def _generate_effectiveness_recommendation(self, score: float) -> str:
        """生成有效性建议"""
        if score >= 90:
            return "策略运行良好，建议保持当前配置"
        elif score >= 70:
            return "策略运行正常，可考虑微调优化"
        elif score >= 50:
            return "策略效果一般，建议检查条件和动作配置"
        else:
            return "策略效果不佳，建议重新评估策略设计"
    
    # ==================== 7. 安全策略优化 ====================
    
    def optimize_policies(self) -> Dict[str, Any]:
        """
        优化策略性能
        
        Returns:
            Dict[str, Any]: 优化结果
        """
        try:
            optimization_results = {
                "optimized_policies": [],
                "removed_duplicates": 0,
                "merged_similar": 0,
                "performance_improvements": {}
            }
            
            # 1. 移除重复策略
            duplicates_removed = self._remove_duplicate_policies()
            optimization_results["removed_duplicates"] = duplicates_removed
            
            # 2. 合并相似策略
            similar_merged = self._merge_similar_policies()
            optimization_results["merged_similar"] = similar_merged
            
            # 3. 优化条件顺序
            optimized_conditions = self._optimize_condition_order()
            optimization_results["optimized_policies"].extend(optimized_conditions)
            
            # 4. 性能分析
            performance_metrics = self._analyze_performance_metrics()
            optimization_results["performance_improvements"] = performance_metrics
            
            logger.info(f"策略优化完成: 移除重复策略 {duplicates_removed} 个，合并相似策略 {similar_merged} 个")
            return optimization_results
            
        except Exception as e:
            logger.error(f"策略优化失败: {e}")
            return {"error": str(e)}
    
    def _remove_duplicate_policies(self) -> int:
        """移除重复策略"""
        removed_count = 0
        policy_signatures = {}
        
        with self.lock:
            policies_to_remove = []
            
            for policy_id, policy in self.policies.items():
                # 生成策略签名
                signature = self._generate_policy_signature(policy)
                
                if signature in policy_signatures:
                    # 发现重复策略，保留优先级高的
                    existing_id = policy_signatures[signature]
                    existing_policy = self.policies[existing_id]
                    
                    if policy.priority.value < existing_policy.priority.value:
                        # 新策略优先级更高，删除旧策略
                        policies_to_remove.append(existing_id)
                        policy_signatures[signature] = policy_id
                    else:
                        # 旧策略优先级更高，删除新策略
                        policies_to_remove.append(policy_id)
                else:
                    policy_signatures[signature] = policy_id
            
            # 删除重复策略
            for policy_id in policies_to_remove:
                del self.policies[policy_id]
                removed_count += 1
        
        return removed_count
    
    def _generate_policy_signature(self, policy: SecurityPolicy) -> str:
        """生成策略签名"""
        signature_data = {
            "name": policy.name,
            "type": policy.type.value,
            "conditions": [(c.field, c.operator, str(c.value)) for c in policy.conditions],
            "actions": [(a.action_type, str(a.parameters)) for a in policy.actions]
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def _merge_similar_policies(self) -> int:
        """合并相似策略"""
        merged_count = 0
        similar_groups = self._find_similar_policy_groups()
        
        with self.lock:
            for group in similar_groups:
                if len(group) > 1:
                    # 合并组内的策略
                    merged_policy = self._merge_policy_group(group)
                    if merged_policy:
                        # 删除原策略
                        for policy_id in group:
                            if policy_id in self.policies:
                                del self.policies[policy_id]
                        
                        # 添加合并后的策略
                        self.policies[merged_policy.id] = merged_policy
                        merged_count += 1
        
        return merged_count
    
    def _find_similar_policy_groups(self) -> List[List[str]]:
        """查找相似策略组"""
        groups = []
        processed = set()
        
        policy_list = list(self.policies.keys())
        
        for i, policy_id1 in enumerate(policy_list):
            if policy_id1 in processed:
                continue
            
            group = [policy_id1]
            processed.add(policy_id1)
            
            for policy_id2 in policy_list[i+1:]:
                if policy_id2 in processed:
                    continue
                
                policy1 = self.policies[policy_id1]
                policy2 = self.policies[policy_id2]
                
                if self._are_policies_similar(policy1, policy2):
                    group.append(policy_id2)
                    processed.add(policy_id2)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _are_policies_similar(self, policy1: SecurityPolicy, policy2: SecurityPolicy) -> bool:
        """判断策略是否相似"""
        # 类型相同
        if policy1.type != policy2.type:
            return False
        
        # 优先级相同
        if policy1.priority != policy2.priority:
            return False
        
        # 条件相似度超过阈值
        similarity = self._calculate_policy_similarity(policy1, policy2)
        return similarity >= 0.7  # 70%相似度阈值
    
    def _calculate_policy_similarity(self, policy1: SecurityPolicy, policy2: SecurityPolicy) -> float:
        """计算策略相似度"""
        # 条件相似度
        condition_similarity = self._calculate_conditions_similarity(policy1.conditions, policy2.conditions)
        
        # 动作相似度
        action_similarity = self._calculate_actions_similarity(policy1.actions, policy2.actions)
        
        # 综合相似度
        return (condition_similarity + action_similarity) / 2
    
    def _calculate_conditions_similarity(self, conditions1: List[PolicyCondition], conditions2: List[PolicyCondition]) -> float:
        """计算条件相似度"""
        if not conditions1 or not conditions2:
            return 0.0
        
        common_conditions = 0
        for cond1 in conditions1:
            for cond2 in conditions2:
                if cond1.field == cond2.field and cond1.operator == cond2.operator:
                    if self._values_similar(cond1.value, cond2.value):
                        common_conditions += 1
                        break
        
        return common_conditions / max(len(conditions1), len(conditions2))
    
    def _calculate_actions_similarity(self, actions1: List[PolicyAction], actions2: List[PolicyAction]) -> float:
        """计算动作相似度"""
        if not actions1 or not actions2:
            return 0.0
        
        common_actions = 0
        for action1 in actions1:
            for action2 in actions2:
                if action1.action_type == action2.action_type:
                    common_actions += 1
                    break
        
        return common_actions / max(len(actions1), len(actions2))
    
    def _values_similar(self, value1: Any, value2: Any) -> bool:
        """判断值是否相似"""
        try:
            if value1 == value2:
                return True
            if isinstance(value1, str) and isinstance(value2, str):
                # 字符串相似度（简单实现）
                return value1.lower() in value2.lower() or value2.lower() in value1.lower()
            return False
        except:
            return False
    
    def _merge_policy_group(self, policy_ids: List[str]) -> Optional[SecurityPolicy]:
        """合并策略组"""
        if not policy_ids:
            return None
        
        policies = [self.policies[pid] for pid in policy_ids if pid in self.policies]
        if not policies:
            return None
        
        # 使用第一个策略作为基础
        base_policy = copy.deepcopy(policies[0])
        base_policy.id = str(uuid.uuid4())
        base_policy.name = f"合并策略-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        base_policy.updated_at = datetime.datetime.now()
        
        # 合并条件
        all_conditions = []
        for policy in policies:
            all_conditions.extend(policy.conditions)
        
        # 去重条件
        unique_conditions = []
        seen_signatures = set()
        for condition in all_conditions:
            signature = f"{condition.field}-{condition.operator}-{str(condition.value)}"
            if signature not in seen_signatures:
                unique_conditions.append(condition)
                seen_signatures.add(signature)
        
        base_policy.conditions = unique_conditions
        
        # 合并动作
        all_actions = []
        for policy in policies:
            all_actions.extend(policy.actions)
        
        # 去重动作
        unique_actions = []
        seen_action_types = set()
        for action in all_actions:
            if action.action_type not in seen_action_types:
                unique_actions.append(action)
                seen_action_types.add(action.action_type)
        
        base_policy.actions = unique_actions
        
        return base_policy
    
    def _optimize_condition_order(self) -> List[str]:
        """优化条件顺序"""
        optimized_policies = []
        
        for policy_id, policy in self.policies.items():
            if len(policy.conditions) > 1:
                # 按选择性排序条件（选择性强的条件在前）
                sorted_conditions = sorted(policy.conditions, key=self._calculate_condition_selectivity)
                
                if sorted_conditions != policy.conditions:
                    policy.conditions = sorted_conditions
                    optimized_policies.append(policy_id)
        
        return optimized_policies
    
    def _calculate_condition_selectivity(self, condition: PolicyCondition) -> float:
        """计算条件选择性（选择性越强，数值越小，排序越靠前）"""
        # 这里可以根据不同操作符和值类型计算选择性
        # 简单实现：根据操作符类型估算选择性
        selectivity_map = {
            "==": 1.0,  # 完全匹配，选择性最强
            "in": 0.8,  # 在集合中
            ">": 0.6,  # 大于
            "<": 0.6,  # 小于
            ">=": 0.7,  # 大于等于
            "<=": 0.7,  # 小于等于
            "!=": 0.9,  # 不等于，选择性较强
            "contains": 0.5  # 包含，选择性较弱
        }
        
        return selectivity_map.get(condition.operator, 0.5)
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """分析性能指标"""
        metrics = {
            "total_policies": len(self.policies),
            "active_policies": len([p for p in self.policies.values() if p.status == PolicyStatus.ACTIVE]),
            "policy_types": {},
            "priority_distribution": {},
            "average_conditions_per_policy": 0,
            "average_actions_per_policy": 0
        }
        
        total_conditions = 0
        total_actions = 0
        
        for policy in self.policies.values():
            # 策略类型分布
            policy_type = policy.type.value
            metrics["policy_types"][policy_type] = metrics["policy_types"].get(policy_type, 0) + 1
            
            # 优先级分布
            priority = policy.priority.name
            metrics["priority_distribution"][priority] = metrics["priority_distribution"].get(priority, 0) + 1
            
            # 条件和动作计数
            total_conditions += len(policy.conditions)
            total_actions += len(policy.actions)
        
        # 计算平均值
        if len(self.policies) > 0:
            metrics["average_conditions_per_policy"] = total_conditions / len(self.policies)
            metrics["average_actions_per_policy"] = total_actions / len(self.policies)
        
        return metrics
    
    # ==================== 8. 安全策略审计 ====================
    
    def _log_audit(self, operation: str, policy_id: str, user_id: str, details: Dict[str, Any], result: str):
        """记录审计日志"""
        audit_log = PolicyAuditLog(
            id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(),
            operation=operation,
            policy_id=policy_id,
            user_id=user_id,
            details=details,
            result=result
        )
        
        self.audit_logs.append(audit_log)
        
        # 保持日志数量在合理范围内（避免内存溢出）
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-5000:]  # 保留最近5000条日志
    
    def get_audit_logs(self, filters: Optional[Dict[str, Any]] = None) -> List[PolicyAuditLog]:
        """
        获取审计日志
        
        Args:
            filters: 过滤条件
            
        Returns:
            List[PolicyAuditLog]: 审计日志列表
        """
        logs = self.audit_logs
        
        if filters:
            if "operation" in filters:
                logs = [log for log in logs if log.operation == filters["operation"]]
            if "policy_id" in filters:
                logs = [log for log in logs if log.policy_id == filters["policy_id"]]
            if "user_id" in filters:
                logs = [log for log in logs if log.user_id == filters["user_id"]]
            if "start_time" in filters:
                logs = [log for log in logs if log.timestamp >= filters["start_time"]]
            if "end_time" in filters:
                logs = [log for log in logs if log.timestamp <= filters["end_time"]]
        
        return logs
    
    def generate_audit_report(self, time_range: Tuple[datetime.datetime, datetime.datetime]) -> Dict[str, Any]:
        """
        生成审计报告
        
        Args:
            time_range: 报告时间范围
            
        Returns:
            Dict[str, Any]: 审计报告
        """
        try:
            logs = [
                log for log in self.audit_logs
                if log.timestamp >= time_range[0] and log.timestamp <= time_range[1]
            ]
            
            # 统计信息
            operation_counts = defaultdict(int)
            user_activity = defaultdict(int)
            policy_activity = defaultdict(int)
            result_counts = defaultdict(int)
            
            for log in logs:
                operation_counts[log.operation] += 1
                user_activity[log.user_id] += 1
                if log.policy_id:
                    policy_activity[log.policy_id] += 1
                result_counts[log.result] += 1
            
            return {
                "report_period": {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat()
                },
                "summary": {
                    "total_operations": len(logs),
                    "unique_users": len(user_activity),
                    "unique_policies": len(policy_activity)
                },
                "operation_breakdown": dict(operation_counts),
                "user_activity": dict(user_activity),
                "policy_activity": dict(policy_activity),
                "result_summary": dict(result_counts),
                "generated_at": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"生成审计报告失败: {e}")
            return {"error": str(e)}
    
    # ==================== 9. 安全策略报告 ====================
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        生成综合策略报告
        
        Returns:
            Dict[str, Any]: 综合报告
        """
        try:
            report = {
                "report_metadata": {
                    "generated_at": datetime.datetime.now().isoformat(),
                    "report_version": "1.0.0",
                    "total_policies": len(self.policies)
                },
                "policy_overview": self._generate_policy_overview(),
                "conflict_analysis": self._generate_conflict_analysis(),
                "performance_metrics": self._analyze_performance_metrics(),
                "audit_summary": self._generate_audit_summary(),
                "recommendations": self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            return {"error": str(e)}
    
    def _generate_policy_overview(self) -> Dict[str, Any]:
        """生成策略概览"""
        overview = {
            "total_policies": len(self.policies),
            "active_policies": 0,
            "inactive_policies": 0,
            "pending_policies": 0,
            "deprecated_policies": 0,
            "policies_by_type": {},
            "policies_by_priority": {}
        }
        
        for policy in self.policies.values():
            # 按状态统计
            if policy.status == PolicyStatus.ACTIVE:
                overview["active_policies"] += 1
            elif policy.status == PolicyStatus.INACTIVE:
                overview["inactive_policies"] += 1
            elif policy.status == PolicyStatus.PENDING:
                overview["pending_policies"] += 1
            elif policy.status == PolicyStatus.DEPRECATED:
                overview["deprecated_policies"] += 1
            
            # 按类型统计
            policy_type = policy.type.value
            overview["policies_by_type"][policy_type] = overview["policies_by_type"].get(policy_type, 0) + 1
            
            # 按优先级统计
            priority = policy.priority.name
            overview["policies_by_priority"][priority] = overview["policies_by_priority"].get(priority, 0) + 1
        
        return overview
    
    def _generate_conflict_analysis(self) -> Dict[str, Any]:
        """生成冲突分析"""
        conflicts = self.detect_conflicts()
        
        analysis = {
            "total_conflicts": len(conflicts),
            "conflicts_by_severity": defaultdict(int),
            "conflicts_by_type": defaultdict(int),
            "conflict_details": []
        }
        
        for conflict in conflicts:
            analysis["conflicts_by_severity"][conflict.severity] += 1
            analysis["conflicts_by_type"][conflict.conflict_type] += 1
            analysis["conflict_details"].append(asdict(conflict))
        
        return dict(analysis)
    
    def _generate_audit_summary(self) -> Dict[str, Any]:
        """生成审计摘要"""
        if not self.audit_logs:
            return {"message": "暂无审计日志"}
        
        # 最近24小时的日志
        recent_time = datetime.datetime.now() - datetime.timedelta(days=1)
        recent_logs = [log for log in self.audit_logs if log.timestamp >= recent_time]
        
        operation_counts = defaultdict(int)
        for log in recent_logs:
            operation_counts[log.operation] += 1
        
        return {
            "recent_activity": {
                "time_period": "24小时",
                "total_operations": len(recent_logs),
                "operation_breakdown": dict(operation_counts)
            },
            "most_active_users": self._get_most_active_users(5),
            "most_active_policies": self._get_most_active_policies(5)
        }
    
    def _get_most_active_users(self, limit: int) -> List[Tuple[str, int]]:
        """获取最活跃用户"""
        user_counts = defaultdict(int)
        for log in self.audit_logs:
            if log.user_id:
                user_counts[log.user_id] += 1
        
        return sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _get_most_active_policies(self, limit: int) -> List[Tuple[str, int]]:
        """获取最活跃策略"""
        policy_counts = defaultdict(int)
        for log in self.audit_logs:
            if log.policy_id:
                policy_counts[log.policy_id] += 1
        
        return sorted(policy_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于策略数量的建议
        if len(self.policies) > 100:
            recommendations.append("策略数量较多，建议进行优化和清理")
        
        # 基于冲突的建议
        conflicts = self.detect_conflicts()
        if conflicts:
            recommendations.append(f"检测到 {len(conflicts)} 个策略冲突，建议及时解决")
        
        # 基于审计活动的建议
        recent_logs = [log for log in self.audit_logs if log.timestamp >= datetime.datetime.now() - datetime.timedelta(days=7)]
        if len(recent_logs) < 10:
            recommendations.append("策略使用频率较低，建议检查策略配置的有效性")
        
        # 基于策略状态的建议
        inactive_policies = [p for p in self.policies.values() if p.status == PolicyStatus.INACTIVE]
        if inactive_policies:
            recommendations.append(f"发现 {len(inactive_policies)} 个非激活策略，建议清理或激活")
        
        if not recommendations:
            recommendations.append("系统运行良好，策略配置合理")
        
        return recommendations
    
    # ==================== 公共接口方法 ====================
    
    def get_policy(self, policy_id: str) -> Optional[SecurityPolicy]:
        """获取策略"""
        return self.policies.get(policy_id)
    
    def list_policies(self, filters: Optional[Dict[str, Any]] = None) -> List[SecurityPolicy]:
        """列出策略"""
        policies = list(self.policies.values())
        
        if filters:
            if "type" in filters:
                policies = [p for p in policies if p.type.value == filters["type"]]
            if "status" in filters:
                policies = [p for p in policies if p.status.value == filters["status"]]
            if "priority" in filters:
                policies = [p for p in policies if p.priority.name == filters["priority"]]
        
        return policies
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        return {
            "total_policies": len(self.policies),
            "active_policies": len([p for p in self.policies.values() if p.status == PolicyStatus.ACTIVE]),
            "inactive_policies": len([p for p in self.policies.values() if p.status == PolicyStatus.INACTIVE]),
            "pending_policies": len([p for p in self.policies.values() if p.status == PolicyStatus.PENDING]),
            "deprecated_policies": len([p for p in self.policies.values() if p.status == PolicyStatus.DEPRECATED]),
            "total_audit_logs": len(self.audit_logs)
        }
    
    def export_policies(self, file_path: str) -> bool:
        """
        导出策略到文件
        
        Args:
            file_path: 导出文件路径
            
        Returns:
            bool: 导出是否成功
        """
        try:
            # 自定义序列化函数
            def serialize_policy(policy):
                data = asdict(policy)
                # 转换枚举值为字符串
                if isinstance(data["type"], PolicyType):
                    data["type"] = data["type"].value
                if isinstance(data["status"], PolicyStatus):
                    data["status"] = data["status"].value
                if isinstance(data["priority"], PolicyPriority):
                    data["priority"] = data["priority"].name
                # 转换datetime对象为字符串
                if isinstance(data["created_at"], datetime.datetime):
                    data["created_at"] = data["created_at"].isoformat()
                if isinstance(data["updated_at"], datetime.datetime):
                    data["updated_at"] = data["updated_at"].isoformat()
                return data
            
            def serialize_audit_log(log):
                data = asdict(log)
                # 转换datetime对象为字符串
                if isinstance(data["timestamp"], datetime.datetime):
                    data["timestamp"] = data["timestamp"].isoformat()
                return data
            
            export_data = {
                "export_timestamp": datetime.datetime.now().isoformat(),
                "policies": [serialize_policy(policy) for policy in self.policies.values()],
                "audit_logs": [serialize_audit_log(log) for log in self.audit_logs[-1000:]]  # 只导出最近1000条日志
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"策略导出成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"策略导出失败: {e}")
            return False
    
    def import_policies(self, file_path: str, merge_mode: bool = False) -> bool:
        """
        从文件导入策略
        
        Args:
            file_path: 导入文件路径
            merge_mode: 是否为合并模式（否则为替换模式）
            
        Returns:
            bool: 导入是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if not merge_mode:
                # 替换模式：清空现有策略
                self.policies.clear()
                self.audit_logs.clear()
            
            # 导入策略
            imported_count = 0
            for policy_data in import_data.get("policies", []):
                try:
                    # 转换数据格式
                    policy_data["type"] = PolicyType(policy_data["type"])
                    policy_data["status"] = PolicyStatus(policy_data["status"])
                    
                    # 转换priority：从字符串转换为枚举值
                    priority_str = policy_data["priority"]
                    if isinstance(priority_str, str):
                        policy_data["priority"] = PolicyPriority[priority_str]
                    else:
                        policy_data["priority"] = PolicyPriority(priority_str)
                    
                    # 转换条件
                    conditions = [PolicyCondition(**cond) for cond in policy_data.get("conditions", [])]
                    policy_data["conditions"] = conditions
                    
                    # 转换动作
                    actions = [PolicyAction(**action) for action in policy_data.get("actions", [])]
                    policy_data["actions"] = actions
                    
                    # 创建策略对象
                    policy = SecurityPolicy(**policy_data)
                    
                    if self.create_policy(policy):
                        imported_count += 1
                        
                except Exception as e:
                    logger.warning(f"导入策略失败: {e}")
                    continue
            
            logger.info(f"成功导入 {imported_count} 个策略")
            return True
            
        except Exception as e:
            logger.error(f"策略导入失败: {e}")
            return False
    
    def shutdown(self):
        """关闭管理器"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            logger.info("安全策略管理器已关闭")
        except Exception as e:
            logger.error(f"关闭管理器时出错: {e}")


# ==================== 测试用例 ====================

class SecurityPolicyManagerTest:
    """安全策略管理器测试类"""
    
    def __init__(self):
        """初始化测试"""
        self.manager = SecurityPolicyManager()
        self.test_results = []
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始运行N8安全策略管理器测试...")
        print("=" * 60)
        
        test_methods = [
            self.test_policy_creation,
            self.test_policy_evaluation,
            self.test_policy_update,
            self.test_conflict_detection,
            self.test_version_management,
            self.test_audit_logging,
            self.test_policy_optimization,
            self.test_report_generation,
            self.test_import_export
        ]
        
        for test_method in test_methods:
            try:
                print(f"\n运行测试: {test_method.__name__}")
                result = test_method()
                self.test_results.append((test_method.__name__, result))
                print(f"测试结果: {'通过' if result else '失败'}")
            except Exception as e:
                print(f"测试异常: {e}")
                self.test_results.append((test_method.__name__, False))
        
        # 输出测试总结
        self._print_test_summary()
    
    def test_policy_creation(self) -> bool:
        """测试策略创建"""
        try:
            # 创建测试策略
            policy = SecurityPolicy(
                id=str(uuid.uuid4()),
                name="测试访问控制策略",
                type=PolicyType.ACCESS_CONTROL,
                description="用于测试的访问控制策略",
                conditions=[
                    PolicyCondition("user.role", "==", "tester", "用户角色为测试员"),
                    PolicyCondition("action.type", "in", ["read", "write"], "操作类型为读或写")
                ],
                actions=[
                    PolicyAction("allow", {"scope": "test"}, "允许测试操作"),
                    PolicyAction("log", {"level": "debug"}, "记录调试日志")
                ],
                priority=PolicyPriority.MEDIUM,
                status=PolicyStatus.ACTIVE,
                version="1.0.0",
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                created_by="test_user"
            )
            
            result = self.manager.create_policy(policy)
            
            if result:
                # 验证策略是否正确创建
                retrieved_policy = self.manager.get_policy(policy.id)
                if retrieved_policy and retrieved_policy.name == policy.name:
                    print("✓ 策略创建测试通过")
                    return True
            
            print("✗ 策略创建测试失败")
            return False
            
        except Exception as e:
            print(f"✗ 策略创建测试异常: {e}")
            return False
    
    def test_policy_evaluation(self) -> bool:
        """测试策略评估"""
        try:
            # 创建测试上下文
            test_context = {
                "user": {"role": "tester", "clearance_level": 2},
                "action": {"type": "read"},
                "resource": {"type": "test_data", "classification": "normal"}
            }
            
            # 评估策略
            results = self.manager.evaluate_policies(test_context)
            
            if results:
                print(f"✓ 策略评估测试通过，评估了 {len(results)} 个策略")
                return True
            else:
                print("✗ 策略评估测试失败：没有评估结果")
                return False
                
        except Exception as e:
            print(f"✗ 策略评估测试异常: {e}")
            return False
    
    def test_policy_update(self) -> bool:
        """测试策略更新"""
        try:
            # 获取第一个策略进行测试
            policies = list(self.manager.policies.values())
            if not policies:
                print("✗ 策略更新测试失败：没有可用的策略")
                return False
            
            test_policy = policies[0]
            original_name = test_policy.name
            
            # 更新策略
            updates = {
                "name": f"{original_name}_更新版",
                "description": "更新后的策略描述"
            }
            
            result = self.manager.update_policy(test_policy.id, updates)
            
            if result:
                # 验证更新
                updated_policy = self.manager.get_policy(test_policy.id)
                if updated_policy and updated_policy.name != original_name:
                    print("✓ 策略更新测试通过")
                    return True
            
            print("✗ 策略更新测试失败")
            return False
            
        except Exception as e:
            print(f"✗ 策略更新测试异常: {e}")
            return False
    
    def test_conflict_detection(self) -> bool:
        """测试冲突检测"""
        try:
            # 创建可能冲突的策略
            policy1 = SecurityPolicy(
                id=str(uuid.uuid4()),
                name="冲突测试策略1",
                type=PolicyType.ACCESS_CONTROL,
                description="测试冲突的策略1",
                conditions=[
                    PolicyCondition("user.role", "==", "admin", "用户角色为管理员")
                ],
                actions=[
                    PolicyAction("allow", {"scope": "all"}, "允许所有操作")
                ],
                priority=PolicyPriority.HIGH,
                status=PolicyStatus.ACTIVE,
                version="1.0.0",
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                created_by="test_user"
            )
            
            policy2 = SecurityPolicy(
                id=str(uuid.uuid4()),
                name="冲突测试策略2",
                type=PolicyType.ACCESS_CONTROL,
                description="测试冲突的策略2",
                conditions=[
                    PolicyCondition("user.role", "==", "admin", "用户角色为管理员")
                ],
                actions=[
                    PolicyAction("deny", {"reason": "security"}, "拒绝所有操作")
                ],
                priority=PolicyPriority.HIGH,
                status=PolicyStatus.ACTIVE,
                version="1.0.0",
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                created_by="test_user"
            )
            
            # 创建策略
            self.manager.create_policy(policy1)
            self.manager.create_policy(policy2)
            
            # 检测冲突
            conflicts = self.manager.detect_conflicts()
            
            print(f"✓ 冲突检测测试完成，检测到 {len(conflicts)} 个冲突")
            return True
            
        except Exception as e:
            print(f"✗ 冲突检测测试异常: {e}")
            return False
    
    def test_version_management(self) -> bool:
        """测试版本管理"""
        try:
            # 获取第一个策略进行测试
            policies = list(self.manager.policies.values())
            if not policies:
                print("✗ 版本管理测试失败：没有可用的策略")
                return False
            
            test_policy = policies[0]
            original_version = test_policy.version
            
            # 更新策略版本
            updates = {
                "version": "2.0.0",
                "description": "版本更新测试"
            }
            
            result = self.manager.update_policy(test_policy.id, updates)
            
            if result:
                # 检查版本历史
                versions = self.manager.get_policy_versions(test_policy.id)
                if len(versions) >= 2:
                    print("✓ 版本管理测试通过")
                    return True
            
            print("✗ 版本管理测试失败")
            return False
            
        except Exception as e:
            print(f"✗ 版本管理测试异常: {e}")
            return False
    
    def test_audit_logging(self) -> bool:
        """测试审计日志"""
        try:
            # 获取审计日志
            logs = self.manager.get_audit_logs()
            
            if logs:
                print(f"✓ 审计日志测试通过，记录了 {len(logs)} 条日志")
                return True
            else:
                print("✗ 审计日志测试失败：没有审计日志")
                return False
                
        except Exception as e:
            print(f"✗ 审计日志测试异常: {e}")
            return False
    
    def test_policy_optimization(self) -> bool:
        """测试策略优化"""
        try:
            # 执行策略优化
            results = self.manager.optimize_policies()
            
            if "error" not in results:
                print(f"✓ 策略优化测试通过")
                print(f"  - 移除重复策略: {results.get('removed_duplicates', 0)}")
                print(f"  - 合并相似策略: {results.get('merged_similar', 0)}")
                return True
            else:
                print(f"✗ 策略优化测试失败: {results['error']}")
                return False
                
        except Exception as e:
            print(f"✗ 策略优化测试异常: {e}")
            return False
    
    def test_report_generation(self) -> bool:
        """测试报告生成"""
        try:
            # 生成综合报告
            report = self.manager.generate_comprehensive_report()
            
            if "error" not in report:
                print("✓ 报告生成测试通过")
                print(f"  - 策略总数: {report['report_metadata']['total_policies']}")
                print(f"  - 建议数量: {len(report['recommendations'])}")
                return True
            else:
                print(f"✗ 报告生成测试失败: {report['error']}")
                return False
                
        except Exception as e:
            print(f"✗ 报告生成测试异常: {e}")
            return False
    
    def test_import_export(self) -> bool:
        """测试导入导出"""
        try:
            export_file = "/tmp/test_policies_export.json"
            
            # 导出策略
            export_result = self.manager.export_policies(export_file)
            
            if export_result:
                # 导入策略
                import_result = self.manager.import_policies(export_file, merge_mode=True)
                
                if import_result:
                    print("✓ 导入导出测试通过")
                    return True
            
            print("✗ 导入导出测试失败")
            return False
            
        except Exception as e:
            print(f"✗ 导入导出测试异常: {e}")
            return False
    
    def _print_test_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        
        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        
        print(f"总测试数: {total}")
        print(f"通过测试: {passed}")
        print(f"失败测试: {total - passed}")
        print(f"通过率: {passed/total*100:.1f}%")
        
        if passed == total:
            print("\n🎉 所有测试通过！")
        else:
            print("\n⚠️  部分测试失败，请检查相关功能")
        
        # 显示失败的测试
        failed_tests = [name for name, result in self.test_results if not result]
        if failed_tests:
            print(f"\n失败的测试: {', '.join(failed_tests)}")


def main():
    """主函数 - 演示安全策略管理器功能"""
    print("N8安全策略管理器演示")
    print("=" * 50)
    
    # 创建管理器实例
    manager = SecurityPolicyManager()
    
    try:
        # 1. 显示策略统计
        print("\n1. 策略统计信息:")
        stats = manager.get_policy_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 2. 评估策略
        print("\n2. 策略评估演示:")
        test_context = {
            "user": {"role": "admin", "clearance_level": 3},
            "action": {"type": "read"},
            "resource": {"type": "config", "classification": "sensitive"},
            "data": {"classification": "sensitive"}
        }
        
        results = manager.evaluate_policies(test_context)
        print(f"评估了 {len(results)} 个策略:")
        for result in results[:3]:  # 只显示前3个结果
            print(f"  策略 {result.policy_id}: 匹配={result.matched}, 结果={result.result}")
        
        # 3. 检测冲突
        print("\n3. 策略冲突检测:")
        conflicts = manager.detect_conflicts()
        print(f"检测到 {len(conflicts)} 个策略冲突")
        
        # 4. 生成报告
        print("\n4. 生成综合报告:")
        report = manager.generate_comprehensive_report()
        if "error" not in report:
            print("报告生成成功，包含以下部分:")
            for section in report.keys():
                print(f"  - {section}")
        
        # 5. 运行测试
        print("\n5. 运行功能测试:")
        test_runner = SecurityPolicyManagerTest()
        test_runner.run_all_tests()
        
    finally:
        # 清理资源
        manager.shutdown()
        print("\n安全策略管理器演示完成")


if __name__ == "__main__":
    main()