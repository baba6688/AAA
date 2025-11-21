#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N区安全模块包
============

这是一个完整的企业级安全系统，包含以下核心模块：
- N1: 身份验证器 (Authenticator) - 用户认证和身份管理
- N2: 授权管理器 (AuthorizationManager) - 权限控制和访问管理  
- N3: 加密处理器 (EncryptionProcessor) - 数据加密和解密
- N4: 访问控制器 (AccessController) - 访问控制和监控
- N5: 安全审计器 (SecurityAuditor) - 安全审计和合规检查
- N6: 威胁检测器 (ThreatDetector) - 威胁检测和响应
- N7: 数据脱敏器 (DataMasker) - 数据脱敏和隐私保护
- N8: 安全策略管理器 (SecurityPolicyManager) - 安全策略管理
- N9: 安全状态聚合器 (SecurityStateAggregator) - 安全状态监控

版本: 1.0.0
创建时间: 2025-11-06
"""

# 导入必要的类型注解
from typing import List, Any

# 从各子模块导入核心类和功能
from .N1 import (
    Authenticator,
    AuthLevel,
    SecurityEventType,
    User,
    SecurityEvent,
    Session,
    PasswordPolicy,
    BiometricAuthenticator,
    OAuthProvider
)

from .N2 import (
    AuthorizationManager,
    PermissionEffect,
    ResourceType,
    ActionType,
    Subject,
    Resource,
    Action,
    Policy,
    Role,
    AuditLog,
    PermissionCache,
    RoleManager,
    PolicyManager,
    AuthorizationEngine,
    AuditManager
)

from .N3 import (
    EncryptionProcessor,
    EncryptionConfig,
    KeyInfo,
    AuditLog as EncryptionAuditLog,
    ComplianceChecker,
    PerformanceOptimizer
)

from .N4 import (
    AccessController,
    AccessLevel,
    AccessType,
    RiskLevel,
    AccessRule,
    AccessEvent,
    NetworkAccessRule
)

from .N5 import (
    SecurityAuditor,
    SecurityEventType as SecurityEventTypeAuditor,
    ThreatLevel,
    ComplianceStatus,
    SecurityEvent,
    SecurityMetric,
    ComplianceCheck
)

from .N6 import (
    ThreatDetector,
    ThreatLevel as ThreatLevelDetector,
    ThreatType,
    ResponseAction,
    ThreatIndicator,
    NetworkEvent,
    ThreatEvent,
    ThreatReport,
    MalwareDetector,
    NetworkAttackDetector,
    InsiderThreatDetector,
    AnomalyDetector,
    ThreatIntelligenceIntegrator,
    RealTimeMonitor,
    ThreatResponseEngine,
    ThreatClassifier,
    ThreatScorer,
    ThreatReporter
)

from .N7 import (
    DataMasker,
    SensitiveDataIdentifier,
    MaskingAlgorithm,
    SensitiveDataType,
    MaskingStrategy,
    ComplianceStandard,
    SensitiveDataPattern,
    MaskingRule,
    MaskingResult,
    MaskingStatistics,
    HashMaskingAlgorithm,
    ReplaceMaskingAlgorithm,
    PartialMaskingAlgorithm,
    ShuffleMaskingAlgorithm,
    TokenizationMaskingAlgorithm,
    DateShiftMaskingAlgorithm,
    EncryptionMaskingAlgorithm,
    mask_sensitive_data
)

from .N8 import (
    SecurityPolicyManager,
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
    BasicPolicyEngine
)

from .N9 import (
    SecurityStateAggregator,
    SecurityLevel,
    AlertSeverity,
    EventType,
    SecurityMetric,
    SecurityThreat,
    SecurityEvent,
    SecurityAlert,
    SecurityReport,
    SecurityModuleInterface,
    MockSecurityModule
)

# 版本信息
__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "security@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # N1 身份验证器
    "Authenticator",
    "AuthLevel",
    "SecurityEventType", 
    "User",
    "SecurityEvent",
    "Session",
    "PasswordPolicy",
    "BiometricAuthenticator",
    "OAuthProvider",
    
    # N2 授权管理器
    "AuthorizationManager",
    "PermissionEffect",
    "ResourceType",
    "ActionType",
    "Subject",
    "Resource",
    "Action",
    "Policy",
    "Role",
    "AuditLog",
    "PermissionCache",
    "RoleManager",
    "PolicyManager",
    "AuthorizationEngine",
    "AuditManager",
    
    # N3 加密处理器
    "EncryptionProcessor",
    "EncryptionConfig",
    "KeyInfo",
    "EncryptionAuditLog",
    "ComplianceChecker",
    "PerformanceOptimizer",
    
    # N4 访问控制器
    "AccessController",
    "AccessLevel",
    "AccessType",
    "RiskLevel",
    "AccessRule",
    "AccessEvent",
    "NetworkAccessRule",
    
    # N5 安全审计器
    "SecurityAuditor",
    "SecurityEventTypeAuditor",
    "ThreatLevel",
    "ComplianceStatus",
    "SecurityEvent",
    "SecurityMetric",
    "ComplianceCheck",
    
    # N6 威胁检测器
    "ThreatDetector",
    "ThreatLevelDetector",
    "ThreatType",
    "ResponseAction",
    "ThreatIndicator",
    "NetworkEvent",
    "ThreatEvent",
    "ThreatReport",
    "MalwareDetector",
    "NetworkAttackDetector",
    "InsiderThreatDetector",
    "AnomalyDetector",
    "ThreatIntelligenceIntegrator",
    "RealTimeMonitor",
    "ThreatResponseEngine",
    "ThreatClassifier",
    "ThreatScorer",
    "ThreatReporter",
    
    # N7 数据脱敏器
    "DataMasker",
    "SensitiveDataIdentifier",
    "MaskingAlgorithm",
    "SensitiveDataType",
    "MaskingStrategy",
    "ComplianceStandard",
    "SensitiveDataPattern",
    "MaskingRule",
    "MaskingResult",
    "MaskingStatistics",
    "HashMaskingAlgorithm",
    "ReplaceMaskingAlgorithm",
    "PartialMaskingAlgorithm",
    "ShuffleMaskingAlgorithm",
    "TokenizationMaskingAlgorithm",
    "DateShiftMaskingAlgorithm",
    "EncryptionMaskingAlgorithm",
    "mask_sensitive_data",
    
    # N8 安全策略管理器
    "SecurityPolicyManager",
    "PolicyType",
    "PolicyStatus",
    "PolicyPriority",
    "PolicyCondition",
    "PolicyAction",
    "SecurityPolicy",
    "PolicyEvaluationResult",
    "PolicyConflict",
    "PolicyAuditLog",
    "PolicyEngine",
    "BasicPolicyEngine",
    
    # N9 安全状态聚合器
    "SecurityStateAggregator",
    "SecurityLevel",
    "AlertSeverity",
    "EventType",
    "SecurityMetric",
    "SecurityThreat",
    "SecurityEvent",
    "SecurityAlert",
    "SecurityReport",
    "SecurityModuleInterface",
    "MockSecurityModule",
]

# 模块信息映射
MODULE_INFO = {
    "N1": {
        "name": "身份验证器",
        "description": "提供用户认证、多因子认证、JWT令牌等身份验证功能",
        "main_class": "Authenticator"
    },
    "N2": {
        "name": "授权管理器", 
        "description": "提供RBAC/ABAC权限控制、策略管理等授权功能",
        "main_class": "AuthorizationManager"
    },
    "N3": {
        "name": "加密处理器",
        "description": "提供数据加密解密、密钥管理、数字签名等加密功能",
        "main_class": "EncryptionProcessor"
    },
    "N4": {
        "name": "访问控制器",
        "description": "提供访问控制、网络监控、异常检测等访问控制功能",
        "main_class": "AccessController"
    },
    "N5": {
        "name": "安全审计器",
        "description": "提供安全审计、合规检查、威胁评估等审计功能",
        "main_class": "SecurityAuditor"
    },
    "N6": {
        "name": "威胁检测器",
        "description": "提供恶意软件检测、网络攻击检测、内部威胁检测等威胁检测功能",
        "main_class": "ThreatDetector"
    },
    "N7": {
        "name": "数据脱敏器",
        "description": "提供数据脱敏、隐私保护、合规性脱敏等数据脱敏功能",
        "main_class": "DataMasker"
    },
    "N8": {
        "name": "安全策略管理器",
        "description": "提供安全策略定义、执行、管理等策略管理功能",
        "main_class": "SecurityPolicyManager"
    },
    "N9": {
        "name": "安全状态聚合器",
        "description": "提供安全状态收集、聚合、分析等安全监控功能",
        "main_class": "SecurityStateAggregator"
    }
}

def get_module_info(module_name: str) -> dict:
    """
    获取模块信息
    
    Args:
        module_name: 模块名称 (如 "N1", "N2", etc.)
        
    Returns:
        dict: 模块信息字典
    """
    return MODULE_INFO.get(module_name, {})

def list_available_modules() -> List[str]:
    """
    列出所有可用模块
    
    Returns:
        List[str]: 模块名称列表
    """
    return list(MODULE_INFO.keys())

def create_security_component(component_type: str, **kwargs) -> Any:
    """
    创建安全组件实例的工厂方法
    
    Args:
        component_type: 组件类型 ("authenticator", "authorization", "encryption", etc.)
        **kwargs: 组件初始化参数
        
    Returns:
        Any: 组件实例
    """
    component_map = {
        "authenticator": Authenticator,
        "authorization": AuthorizationManager,
        "encryption": EncryptionProcessor,
        "access_control": AccessController,
        "security_audit": SecurityAuditor,
        "threat_detection": ThreatDetector,
        "data_masking": DataMasker,
        "policy_management": SecurityPolicyManager,
        "state_aggregation": SecurityStateAggregator
    }
    
    component_class = component_map.get(component_type.lower())
    if component_class:
        return component_class(**kwargs)
    else:
        raise ValueError(f"不支持的组件类型: {component_type}")

def get_version():
    """获取版本信息"""
    return __version__

def get_author():
    """获取作者信息"""
    return __author__

def get_license():
    """获取许可证信息"""
    return __license__

# 使用示例
EXAMPLE_USAGE = """
N区安全模块包使用示例:

1. 身份验证:
   from N import Authenticator
   auth = Authenticator()
   result = auth.authenticate_user("username", "password")

2. 授权管理:
   from N import AuthorizationManager, Subject, Resource, Action
   auth_manager = AuthorizationManager()
   subject = Subject("user1", "user")
   resource = Resource("data1", ResourceType.DATA)
   action = Action(ActionType.READ)
   allowed, context = auth_manager.authorize(subject, resource, action)

3. 数据加密:
   from N import EncryptionProcessor
   processor = EncryptionProcessor()
   key_id = processor.generate_symmetric_key("AES-256-GCM")
   encrypted = processor.encrypt_symmetric("sensitive_data", key_id)

4. 威胁检测:
   from N import ThreatDetector
   detector = ThreatDetector()
   threat_event = detector.detect_malware("file_path")

5. 安全状态聚合:
   from N import SecurityStateAggregator
   aggregator = SecurityStateAggregator()
   await aggregator.start_collection()

详细文档请参考各子模块的docstring。
"""

if __name__ == "__main__":
    print("N区安全模块包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n可用模块:")
    for module_id, info in MODULE_INFO.items():
        print(f"  {module_id}: {info['name']} - {info['description']}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)