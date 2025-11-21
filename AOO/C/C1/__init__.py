"""
C1 知识图谱构建器模块
实现知识图谱的构建、实体识别、关系抽取等功能

主要类:
- Entity: 实体类
- Relation: 关系类  
- KnowledgeTriple: 知识三元组
- EntityExtractor: 实体提取器
- RelationExtractor: 关系提取器
- KnowledgeGraphBuilder: 知识图谱构建器

版本: 1.0.0
作者: AI量化系统
"""

from .KnowledgeGraphBuilder import (
    Entity,
    Relation,
    KnowledgeTriple,
    EntityExtractor,
    RelationExtractor,
    KnowledgeGraphBuilder
)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    'Entity',
    'Relation', 
    'KnowledgeTriple',
    'EntityExtractor',
    'RelationExtractor',
    'KnowledgeGraphBuilder'
]