#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N7数据脱敏器包

一个功能完整的企业级数据脱敏系统，支持多种脱敏算法和合规性标准。

主要模块:
- DataMasker: 核心脱敏器类
- SensitiveDataIdentifier: 敏感数据识别器
- MaskingAlgorithm: 脱敏算法基类
- 各种具体脱敏算法实现

使用示例:
    from DataMasker import DataMasker, SensitiveDataType
    
    masker = DataMasker()
    result = masker.mask_data("13800138000", SensitiveDataType.PHONE)
    print(result.masked_data)  # 输出: 138****8000


日期: 2025-11-06
版本: 1.0.0
"""

from .DataMasker import (
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
    # 脱敏算法实现
    HashMaskingAlgorithm,
    ReplaceMaskingAlgorithm,
    PartialMaskingAlgorithm,
    ShuffleMaskingAlgorithm,
    TokenizationMaskingAlgorithm,
    DateShiftMaskingAlgorithm,
    EncryptionMaskingAlgorithm,
    # 装饰器
    mask_sensitive_data,
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # 核心类
    "DataMasker",
    "SensitiveDataIdentifier",
    "MaskingAlgorithm",
    
    # 枚举类型
    "SensitiveDataType",
    "MaskingStrategy", 
    "ComplianceStandard",
    
    # 数据结构
    "SensitiveDataPattern",
    "MaskingRule",
    "MaskingResult",
    "MaskingStatistics",
    
    # 脱敏算法
    "HashMaskingAlgorithm",
    "ReplaceMaskingAlgorithm",
    "PartialMaskingAlgorithm",
    "ShuffleMaskingAlgorithm",
    "TokenizationMaskingAlgorithm",
    "DateShiftMaskingAlgorithm",
    "EncryptionMaskingAlgorithm",
    
    # 装饰器
    "mask_sensitive_data",
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
def create_masker(config_file=None):
    """创建数据脱敏器实例的便捷函数
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        DataMasker实例
    """
    return DataMasker(config_file)

def quick_mask(data, data_type=None, strategy=None):
    """快速脱敏单个数据的便捷函数
    
    Args:
        data: 待脱敏数据
        data_type: 数据类型，可选
        strategy: 脱敏策略，可选
        
    Returns:
        脱敏后的数据字符串
    """
    masker = DataMasker()
    if data_type and strategy:
        rule = MaskingRule(data_type=data_type, strategy=strategy, parameters={})
        masker.add_masking_rule(rule)
    
    result = masker.mask_data(data, data_type)
    return result.masked_data

# 使用示例
EXAMPLE_USAGE = """
使用示例:

1. 基本使用:
   from DataMasker import DataMasker, SensitiveDataType
   
   masker = DataMasker()
   result = masker.mask_data("13800138000", SensitiveDataType.PHONE)
   print(result.masked_data)  # 138****8000

2. 批量处理:
   data_list = ["13800138000", "user@example.com"]
   results = masker.mask_batch(data_list)
   for r in results:
       print(r.masked_data)

3. 使用装饰器:
   from DataMasker import mask_sensitive_data, SensitiveDataType, MaskingStrategy
   
   @mask_sensitive_data(SensitiveDataType.PHONE, MaskingStrategy.PARTIAL_MASK)
   def get_phone():
       return "13800138000"
   
   phone = get_phone()  # 返回: 138****8000

4. 合规性检查:
   from DataMasker import ComplianceStandard
   
   compliance = masker.check_compliance(
       SensitiveDataType.PHONE, 
       ComplianceStandard.GDPR
   )
   print(compliance['compliant'])

5. 脱敏验证:
   validation = masker.validate_masking(
       "13800138000", 
       "138****8000", 
       SensitiveDataType.PHONE
   )
   print(validation['score'])  # 验证得分
"""

if __name__ == "__main__":
    print("N7数据脱敏器包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)