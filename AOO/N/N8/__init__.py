#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N8安全策略管理器包

一个完整的企业级安全策略管理系统，提供安全策略定义、执行、管理等核心功能。

主要模块:
- SecurityPolicyManager: 核心安全策略管理器类
- SecurityPolicy: 安全策略类
- PolicyEngine: 策略引擎类
- PolicyType: 策略类型枚举
- PolicyStatus: 策略状态枚举
- PolicyCondition: 策略条件类
- PolicyAction: 策略动作类

使用示例:
    from N8 import SecurityPolicyManager, SecurityPolicy, PolicyType
    
    policy_manager = SecurityPolicyManager()
    policy = SecurityPolicy(
        name="data_access_policy",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="数据访问控制策略"
    )
    policy_manager.add_policy(policy)

版本: 1.0.0
日期: 2025-11-06
"""

from .SecurityPolicyManager import (
    PolicyType,
    PolicyStatus,
    PolicyPriority,
    PolicyCondition,
    PolicyAction,
    SecurityPolicy,
    PolicyEvaluationResult,
    PolicyConflict,
    PolicyAuditLog,
    PolicyEngine,
    BasicPolicyEngine,
    SecurityPolicyManager
)

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "security@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # 核心类
    "SecurityPolicyManager",
    
    # 枚举类型
    "PolicyType",
    "PolicyStatus",
    "PolicyPriority",
    
    # 策略模型
    "SecurityPolicy",
    "PolicyCondition",
    "PolicyAction",
    
    # 引擎和结果
    "PolicyEngine",
    "BasicPolicyEngine",
    "PolicyEvaluationResult",
    
    # 管理和审计
    "PolicyConflict",
    "PolicyAuditLog",
]

# 包初始化信息
def get_version():
    """获取版本信息"""
    return __version__

def get_author():
    """获取作者信息"""
    return __author__

def get_license():
    """获取许可证信息"""
    return __license__

# 便捷函数
def create_policy_manager(config_file=None):
    """创建安全策略管理器实例的便捷函数
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        SecurityPolicyManager实例
    """
    return SecurityPolicyManager(config_file)

def create_policy(name, policy_type, description, **kwargs):
    """创建安全策略实例的便捷函数
    
    Args:
        name: 策略名称
        policy_type: 策略类型
        description: 策略描述
        **kwargs: 其他策略属性
        
    Returns:
        SecurityPolicy实例
    """
    return SecurityPolicy(
        name=name,
        policy_type=policy_type,
        description=description,
        **kwargs
    )

def create_policy_condition(condition_type, parameters):
    """创建策略条件实例的便捷函数
    
    Args:
        condition_type: 条件类型
        parameters: 条件参数
        
    Returns:
        PolicyCondition实例
    """
    return PolicyCondition(
        condition_type=condition_type,
        parameters=parameters
    )

def create_policy_action(action_type, parameters):
    """创建策略动作实例的便捷函数
    
    Args:
        action_type: 动作类型
        parameters: 动作参数
        
    Returns:
        PolicyAction实例
    """
    return PolicyAction(
        action_type=action_type,
        parameters=parameters
    )

# 使用示例
EXAMPLE_USAGE = """
使用示例:

1. 基本策略管理:
   from N8 import SecurityPolicyManager, SecurityPolicy, PolicyType, PolicyPriority
   
   policy_manager = SecurityPolicyManager()
   policy = SecurityPolicy(
       name="user_access_policy",
       policy_type=PolicyType.ACCESS_CONTROL,
       description="用户访问控制策略",
       priority=PolicyPriority.HIGH,
       enabled=True
   )
   policy_manager.add_policy(policy)

2. 策略条件设置:
   from N8 import PolicyCondition, PolicyAction
   
   condition = PolicyCondition(
       condition_type="time_based",
       parameters={"start_time": "09:00", "end_time": "18:00"}
   )
   action = PolicyAction(
       action_type="allow_access",
       parameters={"resource": "/work_area/*"}
   )

3. 策略冲突处理:
   from N8 import SecurityPolicyManager
   
   policy_manager = SecurityPolicyManager()
   conflicts = policy_manager.detect_conflicts()
   for conflict in conflicts:
       print(f"发现策略冲突: {conflict.conflict_type}")
       resolution = policy_manager.resolve_conflict(conflict)
       print(f"冲突解决: {resolution}")

4. 策略评估:
   from N8 import SecurityPolicyManager
   
   policy_manager = SecurityPolicyManager()
   result = policy_manager.evaluate_policy(
       policy_name="user_access_policy",
       context={
           "user_id": "user123",
           "resource": "/work_area/data",
           "time": "10:30"
       }
   )
   if result.allowed:
       print("策略评估通过")
   else:
       print(f"策略评估失败: {result.reason}")

5. 策略审计日志:
   from N8 import PolicyAuditLog
   
   audit_log = PolicyAuditLog(
       policy_name="data_access_policy",
       evaluation_result="ALLOW",
       timestamp=datetime.now(),
       context={"user_id": "user123"}
   )
   policy_manager.log_policy_evaluation(audit_log)
"""

if __name__ == "__main__":
    print("N8安全策略管理器包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)