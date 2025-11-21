#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C8知识验证器模块
提供知识一致性检验、完整性检查、准确性验证、时效性评估、冲突检测、质量评分和修复优化功能

模块组件:
- KnowledgeValidator: 主要的知识验证器类
- ValidationResult: 验证结果数据类
- KnowledgeQualityReport: 知识质量报告数据类
- ValidationLevel: 验证级别枚举
- QualityScore: 质量评分等级枚举

使用示例:
    from C.C8 import KnowledgeValidator, ValidationLevel
    
    validator = KnowledgeValidator(ValidationLevel.COMPREHENSIVE)
    report = validator.validate_knowledge_base(knowledge_items)
    
    # 生成可视化报告
    viz_path = validator.visualize_validation_results(report)
    
    # 生成详细报告
    detailed_path = validator.generate_detailed_report(report)
    
    # 优化知识库
    optimized_items = validator.optimize_knowledge_base(knowledge_items)


版本: 1.0.0
创建时间: 2025-11-05
"""

from .KnowledgeValidator import (
    KnowledgeValidator,
    ValidationResult,
    KnowledgeQualityReport,
    ValidationLevel,
    QualityScore,
    create_sample_knowledge_data
)

__version__ = "1.0.0"
__author__ = "AI系统"
__email__ = "ai@example.com"

__all__ = [
    'KnowledgeValidator',
    'ValidationResult', 
    'KnowledgeQualityReport',
    'ValidationLevel',
    'QualityScore',
    'create_sample_knowledge_data'
]

# 模块信息
__module_info__ = {
    'name': 'C8知识验证器',
    'description': '实现知识一致性检验、完整性检查、准确性验证、时效性评估、冲突检测、质量评分和修复优化',
    'version': __version__,
    'author': __author__,
    'created': '2025-11-05'
}