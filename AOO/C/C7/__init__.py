"""
C7 概念映射器模块
实现概念间的映射、转换、关联等功能

主要类:
- Concept: 概念类
- MappingResult: 映射结果
- ConceptMapper: 概念映射器

版本: 1.0.0
作者: AI量化系统
"""

from .ConceptMapper import (
    Concept,
    MappingResult,
    ConceptMapper
)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    'Concept',
    'MappingResult',
    'ConceptMapper'
]