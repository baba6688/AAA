#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3加密处理器包

一个完整的企业级加密系统，提供数据加密解密、密钥管理、数字签名等核心功能。

主要模块:
- EncryptionProcessor: 核心加密处理器类
- EncryptionConfig: 加密配置类
- KeyInfo: 密钥信息类
- AuditLog: 审计日志类
- ComplianceChecker: 合规性检查器
- PerformanceOptimizer: 性能优化器

使用示例:
    from N3 import EncryptionProcessor, EncryptionConfig
    
    processor = EncryptionProcessor()
    config = EncryptionConfig(
        algorithm="AES-256-GCM",
        key_length=256,
        mode="GCM"
    )
    encrypted_data = processor.encrypt(data, config)

版本: 1.0.0
日期: 2025-11-06
"""

from .EncryptionProcessor import (
    EncryptionConfig,
    KeyInfo,
    AuditLog as EncryptionAuditLog,
    ComplianceChecker,
    PerformanceOptimizer,
    EncryptionProcessor
)

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "security@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # 核心类
    "EncryptionProcessor",
    
    # 配置和管理
    "EncryptionConfig",
    "KeyInfo",
    "PerformanceOptimizer",
    
    # 审计和合规
    "EncryptionAuditLog",
    "ComplianceChecker",
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
def create_processor(config_file=None):
    """创建加密处理器实例的便捷函数
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        EncryptionProcessor实例
    """
    return EncryptionProcessor(config_file)

def create_encryption_config(algorithm, key_length=None, mode=None, **kwargs):
    """创建加密配置实例的便捷函数
    
    Args:
        algorithm: 加密算法（如"AES-256-GCM"）
        key_length: 密钥长度，可选
        mode: 加密模式，可选
        **kwargs: 其他配置参数
        
    Returns:
        EncryptionConfig实例
    """
    return EncryptionConfig(
        algorithm=algorithm,
        key_length=key_length,
        mode=mode,
        **kwargs
    )

def create_key_info(key_id, algorithm, key_data, **kwargs):
    """创建密钥信息实例的便捷函数
    
    Args:
        key_id: 密钥ID
        algorithm: 算法名称
        key_data: 密钥数据
        **kwargs: 其他密钥属性
        
    Returns:
        KeyInfo实例
    """
    return KeyInfo(
        key_id=key_id,
        algorithm=algorithm,
        key_data=key_data,
        **kwargs
    )

# 使用示例
EXAMPLE_USAGE = """
使用示例:

1. 数据加密:
   from N3 import EncryptionProcessor, EncryptionConfig
   
   processor = EncryptionProcessor()
   config = EncryptionConfig(algorithm="AES-256-GCM")
   encrypted = processor.encrypt("sensitive_data", config)
   print(f"加密数据: {encrypted}")

2. 对称密钥管理:
   from N3 import EncryptionProcessor
   
   processor = EncryptionProcessor()
   key_id = processor.generate_symmetric_key("AES-256-GCM")
   key_info = processor.get_key_info(key_id)
   print(f"密钥ID: {key_info.key_id}")

3. 数字签名:
   from N3 import EncryptionProcessor
   
   processor = EncryptionProcessor()
   key_pair = processor.generate_asymmetric_keypair("RSA-2048")
   signature = processor.sign_data("data_to_sign", key_pair.private_key)
   is_valid = processor.verify_signature("data_to_sign", signature, key_pair.public_key)

4. 性能优化:
   from N3 import EncryptionProcessor, PerformanceOptimizer
   
   processor = EncryptionProcessor()
   optimizer = PerformanceOptimizer(processor)
   optimizer.set_threading_enabled(True)
   encrypted = optimizer.encrypt_batch(data_list)

5. 合规性检查:
   from N3 import EncryptionProcessor
   
   processor = EncryptionProcessor()
   compliance = processor.check_compliance("AES-256", "FIPS-140-2")
   if compliance['compliant']:
       print("符合合规性要求")
"""

if __name__ == "__main__":
    print("N3加密处理器包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)