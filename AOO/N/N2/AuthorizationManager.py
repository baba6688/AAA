"""
N2授权管理器模块
================

实现完整的授权管理系统，包括RBAC角色权限管理、ABAC属性访问控制、
权限继承和委派、动态权限评估、权限缓存机制、权限变更审计、
权限冲突检测、权限策略管理和授权决策日志。

Author: N2 Authorization System
Date: 2025-11-06
Version: 1.0.0
"""

import json
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable, Tuple, Union
from enum import Enum, auto
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import logging
import copy


# ==================== 核心数据结构和枚举 ====================

class PermissionEffect(Enum):
    """权限效果枚举"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class ResourceType(Enum):
    """资源类型枚举"""
    USER = "user"
    ROLE = "role"
    DATA = "data"
    API = "api"
    FILE = "file"
    SYSTEM = "system"


class ActionType(Enum):
    """操作类型枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    CREATE = "create"
    UPDATE = "update"
    APPROVE = "approve"
    DENY = "deny"


@dataclass
class Subject:
    """主体类 - 表示请求访问的实体"""
    id: str
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    roles: Set[str] = field(default_factory=set)
    groups: Set[str] = field(default_factory=set)
    
    def has_role(self, role: str) -> bool:
        """检查是否拥有指定角色"""
        return role in self.roles
    
    def has_attribute(self, key: str, value: Any = None) -> bool:
        """检查是否拥有指定属性"""
        if key not in self.attributes:
            return False
        if value is None:
            return True
        return self.attributes[key] == value


@dataclass
class Resource:
    """资源类 - 表示被访问的对象"""
    id: str
    type: ResourceType
    attributes: Dict[str, Any] = field(default_factory=dict)
    owner: Optional[str] = None
    sensitivity_level: int = 1  # 敏感度级别 1-5
    
    def has_attribute(self, key: str, value: Any = None) -> bool:
        """检查是否拥有指定属性"""
        if key not in self.attributes:
            return False
        if value is None:
            return True
        return self.attributes[key] == value


@dataclass
class Action:
    """操作类 - 表示要执行的操作"""
    type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def has_parameter(self, key: str, value: Any = None) -> bool:
        """检查是否拥有指定参数"""
        if key not in self.parameters:
            return False
        if value is None:
            return True
        return self.parameters[key] == value


@dataclass
class Policy:
    """策略类 - 定义授权规则"""
    id: str
    name: str
    effect: PermissionEffect
    subject_pattern: Dict[str, Any]
    resource_pattern: Dict[str, Any]
    action_pattern: Dict[str, Any]
    conditions: Dict[str, Any]
    priority: int = 100
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def matches(self, subject: Subject, resource: Resource, action: Action) -> Tuple[bool, Dict[str, Any]]:
        """
        检查策略是否匹配给定的subject、resource和action
        
        Returns:
            Tuple[bool, Dict[str, Any]]: (是否匹配, 匹配的上下文)
        """
        context = {}
        
        # 检查主体匹配
        if not self._matches_pattern(subject, self.subject_pattern, context):
            return False, {}
        
        # 检查资源匹配
        if not self._matches_pattern(resource, self.resource_pattern, context):
            return False, {}
        
        # 检查操作匹配
        if not self._matches_pattern(action, self.action_pattern, context):
            return False, {}
        
        # 检查条件
        if not self._evaluate_conditions(context):
            return False, {}
        
        return True, context
    
    def _matches_pattern(self, obj: Any, pattern: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """检查对象是否匹配模式"""
        for key, value in pattern.items():
            if hasattr(obj, key):
                attr_value = getattr(obj, key)
                
                # 处理枚举类型
                if hasattr(attr_value, 'value'):
                    attr_value = attr_value.value
                
                if isinstance(value, dict):
                    # 复杂匹配规则
                    if 'equals' in value and attr_value != value['equals']:
                        return False
                    if 'in' in value:
                        # 处理集合与列表的匹配
                        if isinstance(attr_value, set) and isinstance(value['in'], list):
                            if not attr_value.intersection(set(value['in'])):
                                return False
                        elif attr_value not in value['in']:
                            return False
                    if 'regex' in value and not isinstance(attr_value, str):
                        return False
                    if 'regex' in value and not value['regex'].match(attr_value):
                        return False
                    if 'greater_than' in value and not (attr_value > value['greater_than']):
                        return False
                    if 'less_than' in value and not (attr_value < value['less_than']):
                        return False
                elif isinstance(value, list):
                    # 列表匹配 - 处理集合类型
                    if isinstance(attr_value, set):
                        # 如果属性是集合，检查是否有交集
                        if not attr_value.intersection(set(value)):
                            return False
                    elif attr_value not in value:
                        return False
                else:
                    # 简单匹配
                    if attr_value != value:
                        return False
            else:
                return False
        
        return True
    
    def _evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """评估策略条件"""
        for key, condition in self.conditions.items():
            if isinstance(condition, dict):
                if 'time_range' in condition:
                    now = datetime.now()
                    start_time = datetime.strptime(condition['time_range']['start'], '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.strptime(condition['time_range']['end'], '%Y-%m-%d %H:%M:%S')
                    if not (start_time <= now <= end_time):
                        return False
                
                if 'context_key' in condition:
                    if condition['context_key'] not in context:
                        return False
                    if 'equals' in condition and context[condition['context_key']] != condition['equals']:
                        return False
        
        return True


@dataclass
class Role:
    """角色类"""
    id: str
    name: str
    description: str = ""
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)  # 继承的角色
    delegated_permissions: Dict[str, datetime] = field(default_factory=dict)  # 委派的权限
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_permission(self, permission: str) -> None:
        """添加权限"""
        self.permissions.add(permission)
        self.updated_at = datetime.now()
    
    def remove_permission(self, permission: str) -> None:
        """移除权限"""
        self.permissions.discard(permission)
        self.updated_at = datetime.now()
    
    def delegate_permission(self, permission: str, to_role: str, expires_at: datetime) -> None:
        """委派权限给其他角色"""
        self.delegated_permissions[permission] = expires_at
        self.updated_at = datetime.now()
    
    def get_all_permissions(self, role_manager: 'RoleManager') -> Set[str]:
        """获取所有权限（包括继承的）"""
        all_permissions = set(self.permissions)
        
        # 添加继承的权限
        for parent_role_id in self.parent_roles:
            parent_role = role_manager.get_role(parent_role_id)
            if parent_role:
                all_permissions.update(parent_role.get_all_permissions(role_manager))
        
        # 添加委派的权限（检查是否过期）
        now = datetime.now()
        expired_permissions = []
        for permission, expires_at in self.delegated_permissions.items():
            if now <= expires_at:
                all_permissions.add(permission)
            else:
                expired_permissions.append(permission)
        
        # 清理过期的委派权限
        for permission in expired_permissions:
            del self.delegated_permissions[permission]
        
        return all_permissions


@dataclass
class AuditLog:
    """审计日志类"""
    id: str
    timestamp: datetime
    subject_id: str
    action: str
    resource_id: str
    result: str  # allow, deny, error
    policy_id: Optional[str]
    context: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'subject_id': self.subject_id,
            'action': self.action,
            'resource_id': self.resource_id,
            'result': self.result,
            'policy_id': self.policy_id,
            'context': self.context,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent
        }


# ==================== 缓存管理 ====================

class PermissionCache:
    """权限缓存管理器"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # 缓存生存时间（秒）
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_order = deque()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                    # 更新访问顺序
                    self._access_order.remove(key)
                    self._access_order.append(key)
                    return value
                else:
                    # 缓存过期，删除
                    del self._cache[key]
                    self._access_order.remove(key)
            return None
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            # 检查缓存大小
            if len(self._cache) >= self.max_size and key not in self._cache:
                # 删除最久未访问的项
                oldest_key = self._access_order.popleft()
                del self._cache[oldest_key]
            
            if key in self._cache:
                self._access_order.remove(key)
            
            self._cache[key] = (value, datetime.now())
            self._access_order.append(key)
    
    def invalidate(self, pattern: str = None) -> None:
        """失效缓存"""
        with self._lock:
            if pattern is None:
                self._cache.clear()
                self._access_order.clear()
            else:
                # 模式匹配删除
                keys_to_remove = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
    
    def clear_expired(self) -> int:
        """清理过期缓存"""
        with self._lock:
            now = datetime.now()
            expired_keys = []
            
            for key, (value, timestamp) in self._cache.items():
                if now - timestamp >= timedelta(seconds=self.ttl):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                self._access_order.remove(key)
            
            return len(expired_keys)


# ==================== 角色管理器 ====================

class RoleManager:
    """角色管理器"""
    
    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._role_hierarchy: Dict[str, Set[str]] = defaultdict(set)  # 角色层级关系
        self._lock = threading.RLock()
    
    def create_role(self, role: Role) -> bool:
        """创建角色"""
        with self._lock:
            if role.id in self._roles:
                return False
            self._roles[role.id] = role
            return True
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """获取角色"""
        return self._roles.get(role_id)
    
    def update_role(self, role: Role) -> bool:
        """更新角色"""
        with self._lock:
            if role.id not in self._roles:
                return False
            self._roles[role.id] = role
            return True
    
    def delete_role(self, role_id: str) -> bool:
        """删除角色"""
        with self._lock:
            if role_id not in self._roles:
                return False
            
            # 检查是否有其他角色依赖此角色
            for other_role_id, dependencies in self._role_hierarchy.items():
                if role_id in dependencies:
                    return False  # 不能删除被依赖的角色
            
            del self._roles[role_id]
            del self._role_hierarchy[role_id]
            
            # 从其他角色的依赖中移除
            for dependencies in self._role_hierarchy.values():
                dependencies.discard(role_id)
            
            return True
    
    def add_role_hierarchy(self, parent_role: str, child_role: str) -> bool:
        """添加角色层级关系"""
        with self._lock:
            if parent_role not in self._roles or child_role not in self._roles:
                return False
            
            self._role_hierarchy[child_role].add(parent_role)
            return True
    
    def get_inherited_roles(self, role_id: str) -> Set[str]:
        """获取继承的角色"""
        inherited = set()
        to_process = {role_id}
        processed = set()
        
        while to_process:
            current = to_process.pop()
            if current in processed:
                continue
            
            processed.add(current)
            parents = self._role_hierarchy.get(current, set())
            
            for parent in parents:
                if parent not in inherited:
                    inherited.add(parent)
                    to_process.add(parent)
        
        return inherited
    
    def check_permission_conflict(self, role_id: str, permission: str) -> List[str]:
        """检查权限冲突"""
        conflicts = []
        
        # 检查角色是否有冲突的权限设置
        role = self.get_role(role_id)
        if not role:
            return conflicts
        
        # 检查权限继承链中的冲突
        inherited_roles = self.get_inherited_roles(role_id)
        for inherited_role_id in inherited_roles:
            inherited_role = self.get_role(inherited_role_id)
            if inherited_role and permission in inherited_role.permissions:
                conflicts.append(f"Permission '{permission}' conflicts with inherited role '{inherited_role_id}'")
        
        return conflicts


# ==================== 策略管理器 ====================

class PolicyManager:
    """策略管理器"""
    
    def __init__(self):
        self._policies: Dict[str, Policy] = {}
        self._policy_index: Dict[str, Set[str]] = defaultdict(set)  # 索引：resource_type -> policy_ids
        self._lock = threading.RLock()
    
    def create_policy(self, policy: Policy) -> bool:
        """创建策略"""
        with self._lock:
            if policy.id in self._policies:
                return False
            
            self._policies[policy.id] = policy
            
            # 更新索引
            if 'type' in policy.resource_pattern:
                resource_type = policy.resource_pattern['type']
                self._policy_index[resource_type].add(policy.id)
            
            return True
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """获取策略"""
        return self._policies.get(policy_id)
    
    def update_policy(self, policy: Policy) -> bool:
        """更新策略"""
        with self._lock:
            if policy.id not in self._policies:
                return False
            
            self._policies[policy.id] = policy
            return True
    
    def delete_policy(self, policy_id: str) -> bool:
        """删除策略"""
        with self._lock:
            if policy_id not in self._policies:
                return False
            
            policy = self._policies[policy_id]
            del self._policies[policy_id]
            
            # 从索引中移除
            if 'type' in policy.resource_pattern:
                resource_type = policy.resource_pattern['type']
                self._policy_index[resource_type].discard(policy_id)
            
            return True
    
    def get_policies_for_resource(self, resource_type: str) -> List[Policy]:
        """获取适用于特定资源类型的策略"""
        with self._lock:
            policy_ids = self._policy_index.get(resource_type, set())
            return [self._policies[pid] for pid in policy_ids if pid in self._policies]
    
    def check_policy_conflicts(self) -> List[Dict[str, Any]]:
        """检查策略冲突"""
        conflicts = []
        
        # 按优先级分组策略
        priority_groups = defaultdict(list)
        for policy in self._policies.values():
            if policy.enabled:
                priority_groups[policy.priority].append(policy)
        
        # 检查同优先级策略的冲突
        for priority, policies in priority_groups.items():
            if len(policies) > 1:
                for i, policy1 in enumerate(policies):
                    for policy2 in policies[i+1:]:
                        if self._policies_conflict(policy1, policy2):
                            conflicts.append({
                                'type': 'policy_conflict',
                                'priority': priority,
                                'policy1': policy1.id,
                                'policy2': policy2.id,
                                'description': f"Policies {policy1.id} and {policy2.id} conflict at priority {priority}"
                            })
        
        return conflicts
    
    def _policies_conflict(self, policy1: Policy, policy2: Policy) -> bool:
        """检查两个策略是否冲突"""
        # 简化的冲突检测逻辑
        # 在实际应用中，这会更复杂
        
        # 如果两个策略有相同的效果和重叠的条件，则可能冲突
        if policy1.effect == policy2.effect:
            # 检查条件重叠
            common_conditions = set(policy1.conditions.keys()) & set(policy2.conditions.keys())
            if common_conditions:
                for condition_key in common_conditions:
                    if policy1.conditions[condition_key] == policy2.conditions[condition_key]:
                        return True
        
        return False


# ==================== 授权决策引擎 ====================

class AuthorizationEngine:
    """授权决策引擎"""
    
    def __init__(self, role_manager: RoleManager, policy_manager: PolicyManager):
        self.role_manager = role_manager
        self.policy_manager = policy_manager
        self.logger = logging.getLogger(__name__)
    
    def evaluate_access(self, subject: Subject, resource: Resource, action: Action) -> Tuple[str, Dict[str, Any]]:
        """
        评估访问权限
        
        Returns:
            Tuple[str, Dict[str, Any]]: (决策结果, 决策上下文)
        """
        context = {
            'subject_id': subject.id,
            'resource_id': resource.id,
            'action_type': action.type.value,
            'timestamp': datetime.now().isoformat(),
            'matched_policies': [],
            'evaluation_steps': []
        }
        
        try:
            # 步骤1: 获取适用的策略
            applicable_policies = self._get_applicable_policies(subject, resource, action)
            context['evaluation_steps'].append(f"Found {len(applicable_policies)} applicable policies")
            
            if not applicable_policies:
                context['evaluation_steps'].append("No applicable policies found")
                return "deny", context
            
            # 步骤2: 按优先级排序策略
            sorted_policies = sorted(applicable_policies, key=lambda p: p.priority)
            context['evaluation_steps'].append(f"Evaluated {len(sorted_policies)} policies in priority order")
            
            # 步骤3: 评估策略
            for policy in sorted_policies:
                matches, policy_context = policy.matches(subject, resource, action)
                if matches:
                    context['matched_policies'].append(policy.id)
                    context['evaluation_steps'].append(f"Policy {policy.id} matched")
                    
                    if policy.effect == PermissionEffect.ALLOW:
                        context['evaluation_steps'].append(f"Access allowed by policy {policy.id}")
                        return "allow", context
                    elif policy.effect == PermissionEffect.DENY:
                        context['evaluation_steps'].append(f"Access denied by policy {policy.id}")
                        return "deny", context
                    elif policy.effect == PermissionEffect.CONDITIONAL:
                        # 条件性允许，需要进一步评估
                        if self._evaluate_conditional_allow(policy, subject, resource, action, policy_context):
                            context['evaluation_steps'].append(f"Conditional access allowed by policy {policy.id}")
                            return "allow", context
                        else:
                            context['evaluation_steps'].append(f"Conditional access denied by policy {policy.id}")
                            return "deny", context
            
            # 步骤4: 如果没有明确的允许或拒绝，默认拒绝
            context['evaluation_steps'].append("No explicit allow/deny decision found, defaulting to deny")
            return "deny", context
            
        except Exception as e:
            context['evaluation_steps'].append(f"Error during evaluation: {str(e)}")
            self.logger.error(f"Authorization evaluation error: {str(e)}")
            return "error", context
    
    def _get_applicable_policies(self, subject: Subject, resource: Resource, action: Action) -> List[Policy]:
        """获取适用的策略"""
        policies = []
        
        # 获取所有启用的策略
        all_policies = [p for p in self.policy_manager._policies.values() if p.enabled]
        
        # 过滤出可能适用的策略
        for policy in all_policies:
            # 快速预检查：资源类型匹配
            if 'type' in policy.resource_pattern:
                if policy.resource_pattern['type'] == resource.type.value or policy.resource_pattern['type'] == '*':
                    policies.append(policy)
            else:
                # 如果策略没有指定资源类型，也考虑它
                policies.append(policy)
        
        return policies
    
    def _evaluate_conditional_allow(self, policy: Policy, subject: Subject, 
                                  resource: Resource, action: Action, context: Dict[str, Any]) -> bool:
        """评估条件性允许"""
        # 这里可以实现复杂的条件评估逻辑
        # 例如：时间限制、地理位置限制、设备类型限制等
        
        if 'max_requests_per_hour' in policy.conditions:
            # 检查请求频率限制
            # 这里需要实现请求频率检查逻辑
            pass
        
        if 'require_approval' in policy.conditions:
            # 检查是否需要审批
            # 这里需要实现审批流程检查逻辑
            pass
        
        return True  # 默认允许


# ==================== 审计管理器 ====================

class AuditManager:
    """审计管理器"""
    
    def __init__(self, max_logs: int = 100000):
        self._logs: deque = deque(maxlen=max_logs)
        self._lock = threading.RLock()
        self._log_file: Optional[str] = None
        self.logger = logging.getLogger(__name__)
    
    def set_log_file(self, file_path: str) -> None:
        """设置日志文件路径"""
        self._log_file = file_path
    
    def log_decision(self, subject_id: str, action: str, resource_id: str, 
                    result: str, policy_id: Optional[str], context: Dict[str, Any],
                    ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> None:
        """记录授权决策"""
        audit_log = AuditLog(
            id=self._generate_log_id(),
            timestamp=datetime.now(),
            subject_id=subject_id,
            action=action,
            resource_id=resource_id,
            result=result,
            policy_id=policy_id,
            context=context,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        with self._lock:
            self._logs.append(audit_log)
            
            # 写入文件（如果设置了日志文件）
            if self._log_file:
                self._write_to_file(audit_log)
    
    def get_logs(self, subject_id: Optional[str] = None, 
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None,
                result: Optional[str] = None) -> List[AuditLog]:
        """查询审计日志"""
        with self._lock:
            filtered_logs = list(self._logs)
            
            if subject_id:
                filtered_logs = [log for log in filtered_logs if log.subject_id == subject_id]
            
            if start_time:
                filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
            
            if end_time:
                filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
            
            if result:
                filtered_logs = [log for log in filtered_logs if log.result == result]
            
            return filtered_logs
    
    def generate_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """生成审计报告"""
        logs = self.get_logs(start_time=start_time, end_time=end_time)
        
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_requests': len(logs),
            'allow_count': len([log for log in logs if log.result == 'allow']),
            'deny_count': len([log for log in logs if log.result == 'deny']),
            'error_count': len([log for log in logs if log.result == 'error']),
            'top_subjects': {},
            'top_resources': {},
            'top_actions': {},
            'policy_usage': {}
        }
        
        # 统计主体
        subject_counts = defaultdict(int)
        for log in logs:
            subject_counts[log.subject_id] += 1
        
        report['top_subjects'] = dict(sorted(subject_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:10])
        
        # 统计资源
        resource_counts = defaultdict(int)
        for log in logs:
            resource_counts[log.resource_id] += 1
        
        report['top_resources'] = dict(sorted(resource_counts.items(), 
                                            key=lambda x: x[1], reverse=True)[:10])
        
        # 统计操作
        action_counts = defaultdict(int)
        for log in logs:
            action_counts[log.action] += 1
        
        report['top_actions'] = dict(sorted(action_counts.items(), 
                                          key=lambda x: x[1], reverse=True)[:10])
        
        # 统计策略使用
        policy_counts = defaultdict(int)
        for log in logs:
            if log.policy_id:
                policy_counts[log.policy_id] += 1
        
        report['policy_usage'] = dict(sorted(policy_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:10])
        
        return report
    
    def _generate_log_id(self) -> str:
        """生成日志ID"""
        timestamp = str(int(time.time() * 1000000))
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"audit_{timestamp}_{random_part}"
    
    def _write_to_file(self, audit_log: AuditLog) -> None:
        """写入日志文件"""
        try:
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_log.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write audit log to file: {str(e)}")


# ==================== N2授权管理器主类 ====================

class AuthorizationManager:
    """
    N2授权管理器
    
    集成RBAC和ABAC的完整授权管理系统，支持：
    - 角色基础访问控制 (RBAC)
    - 属性基础访问控制 (ABAC)
    - 权限继承和委派
    - 动态权限评估
    - 权限缓存机制
    - 权限变更审计
    - 权限冲突检测
    - 权限策略管理
    - 授权决策日志
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化授权管理器
        
        Args:
            config: 配置字典
        """
        # 配置
        self.config = config or {}
        cache_size = self.config.get('cache_size', 10000)
        cache_ttl = self.config.get('cache_ttl', 3600)
        max_audit_logs = self.config.get('max_audit_logs', 100000)
        
        # 初始化组件
        self.role_manager = RoleManager()
        self.policy_manager = PolicyManager()
        self.authorization_engine = AuthorizationEngine(self.role_manager, self.policy_manager)
        self.audit_manager = AuditManager(max_audit_logs)
        self.permission_cache = PermissionCache(cache_size, cache_ttl)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'policy_evaluations': 0
        }
    
    # ==================== 公共接口方法 ====================
    
    def authorize(self, subject: Subject, resource: Resource, action: Action,
                 ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        执行授权检查
        
        Args:
            subject: 主体
            resource: 资源
            action: 操作
            ip_address: IP地址
            user_agent: 用户代理
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (是否允许访问, 决策上下文)
        """
        with self._lock:
            self._stats['total_requests'] += 1
            
            # 生成缓存键
            cache_key = self._generate_cache_key(subject, resource, action)
            
            # 检查缓存
            cached_result = self.permission_cache.get(cache_key)
            if cached_result is not None:
                self._stats['cache_hits'] += 1
                return cached_result
            
            self._stats['cache_misses'] += 1
            
            # 执行授权决策
            decision, context = self.authorization_engine.evaluate_access(subject, resource, action)
            
            # 记录审计日志
            matched_policies = context.get('matched_policies', [])
            policy_id = matched_policies[0] if matched_policies else None
            
            self.audit_manager.log_decision(
                subject_id=subject.id,
                action=action.type.value,
                resource_id=resource.id,
                result=decision,
                policy_id=policy_id,
                context=context,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # 缓存结果
            result = (decision == "allow", context)
            self.permission_cache.set(cache_key, result)
            
            return result
    
    def add_role(self, role: Role) -> bool:
        """添加角色"""
        return self.role_manager.create_role(role)
    
    def add_policy(self, policy: Policy) -> bool:
        """添加策略"""
        return self.policy_manager.create_policy(policy)
    
    def update_role(self, role: Role) -> bool:
        """更新角色"""
        result = self.role_manager.update_role(role)
        if result:
            self._invalidate_related_cache(role.id)
        return result
    
    def update_policy(self, policy: Policy) -> bool:
        """更新策略"""
        result = self.policy_manager.update_policy(policy)
        if result:
            self._invalidate_related_cache(policy.id)
        return result
    
    def delete_role(self, role_id: str) -> bool:
        """删除角色"""
        result = self.role_manager.delete_role(role_id)
        if result:
            self._invalidate_related_cache(f"role:{role_id}")
        return result
    
    def delete_policy(self, policy_id: str) -> bool:
        """删除策略"""
        result = self.policy_manager.delete_policy(policy_id)
        if result:
            self._invalidate_related_cache(f"policy:{policy_id}")
        return result
    
    def get_user_permissions(self, user_id: str, resource_type: Optional[str] = None) -> Set[str]:
        """获取用户的所有权限"""
        # 这里需要根据用户ID查找用户对象
        # 简化实现，假设用户对象已存在
        return set()  # 实际实现需要查询用户数据库
    
    def check_permission_conflicts(self) -> List[Dict[str, Any]]:
        """检查权限冲突"""
        conflicts = []
        
        # 检查角色权限冲突
        for role_id in self.role_manager._roles.keys():
            role_conflicts = self.role_manager.check_permission_conflict(role_id, "")
            if role_conflicts:
                conflicts.extend([{
                    'type': 'role_conflict',
                    'role_id': role_id,
                    'conflicts': role_conflicts
                }])
        
        # 检查策略冲突
        policy_conflicts = self.policy_manager.check_policy_conflicts()
        conflicts.extend(policy_conflicts)
        
        return conflicts
    
    def get_audit_logs(self, subject_id: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      result: Optional[str] = None) -> List[AuditLog]:
        """获取审计日志"""
        return self.audit_manager.get_logs(subject_id, start_time, end_time, result)
    
    def generate_audit_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """生成审计报告"""
        return self.audit_manager.generate_report(start_time, end_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            cache_hit_rate = 0
            if self._stats['cache_hits'] + self._stats['cache_misses'] > 0:
                cache_hit_rate = self._stats['cache_hits'] / (self._stats['cache_hits'] + self._stats['cache_misses'])
            
            return {
                'total_requests': self._stats['total_requests'],
                'cache_hit_rate': cache_hit_rate,
                'cache_hits': self._stats['cache_hits'],
                'cache_misses': self._stats['cache_misses'],
                'policy_evaluations': self._stats['policy_evaluations'],
                'total_roles': len(self.role_manager._roles),
                'total_policies': len(self.policy_manager._policies),
                'total_audit_logs': len(self.audit_manager._logs)
            }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.permission_cache.invalidate()
    
    def export_configuration(self) -> Dict[str, Any]:
        """导出配置"""
        return {
            'roles': {role_id: {
                'id': role.id,
                'name': role.name,
                'description': role.description,
                'permissions': list(role.permissions),
                'parent_roles': list(role.parent_roles)
            } for role_id, role in self.role_manager._roles.items()},
            'policies': {policy_id: {
                'id': policy.id,
                'name': policy.name,
                'effect': policy.effect.value,
                'subject_pattern': policy.subject_pattern,
                'resource_pattern': policy.resource_pattern,
                'action_pattern': policy.action_pattern,
                'conditions': policy.conditions,
                'priority': policy.priority,
                'enabled': policy.enabled
            } for policy_id, policy in self.policy_manager._policies.items()},
            'statistics': self.get_statistics()
        }
    
    def import_configuration(self, config: Dict[str, Any]) -> bool:
        """导入配置"""
        try:
            # 导入角色
            for role_data in config.get('roles', {}).values():
                role = Role(
                    id=role_data['id'],
                    name=role_data['name'],
                    description=role_data['description'],
                    permissions=set(role_data['permissions']),
                    parent_roles=set(role_data['parent_roles'])
                )
                self.role_manager.create_role(role)
            
            # 导入策略
            for policy_data in config.get('policies', {}).values():
                policy = Policy(
                    id=policy_data['id'],
                    name=policy_data['name'],
                    effect=PermissionEffect(policy_data['effect']),
                    subject_pattern=policy_data['subject_pattern'],
                    resource_pattern=policy_data['resource_pattern'],
                    action_pattern=policy_data['action_pattern'],
                    conditions=policy_data['conditions'],
                    priority=policy_data['priority'],
                    enabled=policy_data['enabled']
                )
                self.policy_manager.create_policy(policy)
            
            # 清空缓存
            self.clear_cache()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {str(e)}")
            return False
    
    # ==================== 私有方法 ====================
    
    def _generate_cache_key(self, subject: Subject, resource: Resource, action: Action) -> str:
        """生成缓存键"""
        key_data = {
            'subject_id': subject.id,
            'subject_type': subject.type,
            'subject_roles': sorted(list(subject.roles)),
            'subject_attributes': subject.attributes,
            'resource_id': resource.id,
            'resource_type': resource.type.value,
            'resource_attributes': resource.attributes,
            'action_type': action.type.value,
            'action_parameters': action.parameters
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _invalidate_related_cache(self, identifier: str) -> None:
        """失效相关缓存"""
        self.permission_cache.invalidate(identifier)


# ==================== 测试用例 ====================

def create_test_data():
    """创建测试数据"""
    # 创建测试角色
    admin_role = Role(
        id="admin",
        name="管理员",
        description="系统管理员角色",
        permissions={"user:read", "user:write", "user:delete", "system:admin"}
    )
    
    user_role = Role(
        id="user",
        name="普通用户",
        description="普通用户角色",
        permissions={"user:read", "data:read", "data:write"}
    )
    
    guest_role = Role(
        id="guest",
        name="访客",
        description="访客角色",
        permissions={"user:read"}
    )
    
    # 创建测试策略
    admin_policy = Policy(
        id="admin_full_access",
        name="管理员完全访问策略",
        effect=PermissionEffect.ALLOW,
        subject_pattern={"roles": ["admin"]},
        resource_pattern={"type": "system"},
        action_pattern={"type": "admin"},
        conditions={},
        priority=1
    )
    
    user_data_policy = Policy(
        id="user_data_access",
        name="用户数据访问策略",
        effect=PermissionEffect.ALLOW,
        subject_pattern={"type": "user"},
        resource_pattern={"type": "data"},
        action_pattern={"type": "read"},
        conditions={"time_range": {
            "start": "2025-01-01 00:00:00",
            "end": "2025-12-31 23:59:59"
        }},
        priority=100
    )
    
    sensitive_data_policy = Policy(
        id="sensitive_data_protection",
        name="敏感数据保护策略",
        effect=PermissionEffect.DENY,
        subject_pattern={"type": "user"},
        resource_pattern={"sensitivity_level": {"greater_than": 3}},
        action_pattern={"type": "read"},
        conditions={},
        priority=50
    )
    
    return [admin_role, user_role, guest_role], [admin_policy, user_data_policy, sensitive_data_policy]


def run_basic_tests():
    """运行基本测试"""
    print("=== N2授权管理器基本测试 ===\n")
    
    # 创建授权管理器
    auth_manager = AuthorizationManager()
    
    # 创建测试数据
    roles, policies = create_test_data()
    
    # 添加角色
    print("1. 添加角色:")
    for role in roles:
        success = auth_manager.add_role(role)
        print(f"   - 添加角色 {role.name}: {'成功' if success else '失败'}")
    
    # 添加策略
    print("\n2. 添加策略:")
    for policy in policies:
        success = auth_manager.add_policy(policy)
        print(f"   - 添加策略 {policy.name}: {'成功' if success else '失败'}")
    
    # 创建测试主体和资源
    admin_subject = Subject(
        id="admin_user",
        type="user",
        roles={"admin"},
        attributes={"department": "IT", "level": "senior"}
    )
    
    regular_subject = Subject(
        id="regular_user",
        type="user",
        roles={"user"},
        attributes={"department": "Sales", "level": "junior"}
    )
    
    guest_subject = Subject(
        id="guest_user",
        type="user",
        roles={"guest"},
        attributes={"department": "Marketing", "level": "intern"}
    )
    
    normal_resource = Resource(
        id="normal_data",
        type=ResourceType.DATA,
        attributes={"category": "public", "owner": "user123"},
        sensitivity_level=1
    )
    
    sensitive_resource = Resource(
        id="sensitive_data",
        type=ResourceType.DATA,
        attributes={"category": "confidential", "owner": "user456"},
        sensitivity_level=4
    )
    
    system_resource = Resource(
        id="system_config",
        type=ResourceType.SYSTEM,
        attributes={"component": "database"},
        sensitivity_level=5
    )
    
    # 测试授权
    print("\n3. 授权测试:")
    
    test_cases = [
        (admin_subject, system_resource, Action(ActionType.ADMIN), "管理员访问系统配置"),
        (regular_subject, normal_resource, Action(ActionType.READ), "普通用户读取普通数据"),
        (regular_subject, sensitive_resource, Action(ActionType.READ), "普通用户读取敏感数据"),
        (guest_subject, normal_resource, Action(ActionType.READ), "访客读取普通数据"),
        (guest_subject, system_resource, Action(ActionType.ADMIN), "访客访问系统管理")
    ]
    
    for subject, resource, action, description in test_cases:
        allowed, context = auth_manager.authorize(subject, resource, action)
        result = "允许" if allowed else "拒绝"
        print(f"   - {description}: {result}")
        if not allowed:
            print(f"     原因: {context.get('evaluation_steps', ['未知'])[-1]}")
    
    # 测试统计信息
    print("\n4. 统计信息:")
    stats = auth_manager.get_statistics()
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # 测试审计日志
    print("\n5. 审计日志:")
    audit_logs = auth_manager.get_audit_logs()
    print(f"   总共 {len(audit_logs)} 条审计记录")
    
    # 生成审计报告
    print("\n6. 审计报告:")
    now = datetime.now()
    start_time = now - timedelta(hours=1)
    report = auth_manager.generate_audit_report(start_time, now)
    print(f"   时间范围: {start_time} 到 {now}")
    print(f"   总请求数: {report['total_requests']}")
    print(f"   允许访问: {report['allow_count']}")
    print(f"   拒绝访问: {report['deny_count']}")
    print(f"   错误: {report['error_count']}")
    
    # 测试权限冲突检测
    print("\n7. 权限冲突检测:")
    conflicts = auth_manager.check_permission_conflicts()
    if conflicts:
        print("   发现冲突:")
        for conflict in conflicts:
            print(f"   - {conflict}")
    else:
        print("   未发现权限冲突")
    
    # 测试缓存
    print("\n8. 缓存测试:")
    # 重复一次相同的授权请求
    allowed, _ = auth_manager.authorize(admin_subject, system_resource, Action(ActionType.ADMIN))
    stats = auth_manager.get_statistics()
    print(f"   缓存命中率: {stats['cache_hit_rate']:.2%}")
    
    # 测试配置导入导出
    print("\n9. 配置导入导出:")
    config = auth_manager.export_configuration()
    print(f"   导出了 {len(config['roles'])} 个角色和 {len(config['policies'])} 个策略")
    
    print("\n=== 测试完成 ===")


def run_advanced_tests():
    """运行高级测试"""
    print("\n=== N2授权管理器高级测试 ===\n")
    
    auth_manager = AuthorizationManager()
    
    # 测试权限继承
    print("1. 权限继承测试:")
    
    # 创建角色层级
    super_admin = Role(id="super_admin", name="超级管理员", permissions={"system:all"})
    admin = Role(id="admin", name="管理员", permissions={"user:admin"}, parent_roles={"super_admin"})
    manager = Role(id="manager", name="经理", permissions={"user:manage"}, parent_roles={"admin"})
    employee = Role(id="employee", name="员工", permissions={"user:read"}, parent_roles={"manager"})
    
    for role in [super_admin, admin, manager, employee]:
        auth_manager.add_role(role)
    
    # 添加层级关系
    auth_manager.role_manager.add_role_hierarchy("super_admin", "admin")
    auth_manager.role_manager.add_role_hierarchy("admin", "manager")
    auth_manager.role_manager.add_role_hierarchy("manager", "employee")
    
    # 测试继承的权限
    employee_role = auth_manager.role_manager.get_role("employee")
    inherited_permissions = employee_role.get_all_permissions(auth_manager.role_manager)
    print(f"   员工角色继承的权限: {inherited_permissions}")
    
    # 测试权限委派
    print("\n2. 权限委派测试:")
    
    # 临时委派权限
    admin_role = auth_manager.role_manager.get_role("admin")
    expires_at = datetime.now() + timedelta(hours=24)
    admin_role.delegate_permission("user:delete", "temp_admin", expires_at)
    
    temp_admin_role = Role(id="temp_admin", name="临时管理员", permissions=set())
    auth_manager.add_role(temp_admin_role)
    
    temp_permissions = temp_admin_role.get_all_permissions(auth_manager.role_manager)
    print(f"   临时管理员权限: {temp_permissions}")
    
    # 测试动态策略
    print("\n3. 动态策略测试:")
    
    # 创建基于时间的策略
    time_based_policy = Policy(
        id="business_hours_policy",
        name="工作时间访问策略",
        effect=PermissionEffect.ALLOW,
        subject_pattern={"type": "user"},
        resource_pattern={"type": "api"},
        action_pattern={"type": "read"},
        conditions={"context_key": "is_business_hours", "equals": True},
        priority=90
    )
    
    auth_manager.add_policy(time_based_policy)
    
    # 测试条件性访问
    business_hours_subject = Subject(
        id="business_user",
        type="user",
        roles={"employee"},
        attributes={"department": "IT"}
    )
    
    api_resource = Resource(
        id="internal_api",
        type=ResourceType.API,
        attributes={"endpoint": "/internal/data"}
    )
    
    read_action = Action(ActionType.READ, context={"is_business_hours": True})
    
    allowed, context = auth_manager.authorize(business_hours_subject, api_resource, read_action)
    print(f"   工作时间内API访问: {'允许' if allowed else '拒绝'}")
    
    # 测试权限缓存失效
    print("\n4. 缓存失效测试:")
    
    # 第一次请求
    allowed1, _ = auth_manager.authorize(business_hours_subject, api_resource, read_action)
    stats1 = auth_manager.get_statistics()
    
    # 更新策略
    updated_policy = Policy(
        id="business_hours_policy",
        name="工作时间访问策略(已更新)",
        effect=PermissionEffect.DENY,
        subject_pattern={"type": "user"},
        resource_pattern={"type": "api"},
        action_pattern={"type": "read"},
        conditions={"context_key": "is_business_hours", "equals": True},
        priority=90
    )
    auth_manager.update_policy(updated_policy)
    
    # 第二次请求（应该使用新策略）
    allowed2, _ = auth_manager.authorize(business_hours_subject, api_resource, read_action)
    stats2 = auth_manager.get_statistics()
    
    print(f"   策略更新前: {'允许' if allowed1 else '拒绝'}")
    print(f"   策略更新后: {'允许' if allowed2 else '拒绝'}")
    print(f"   缓存失效正常工作: {'是' if allowed1 != allowed2 else '否'}")
    
    print("\n=== 高级测试完成 ===")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    run_basic_tests()
    run_advanced_tests()