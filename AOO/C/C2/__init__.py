"""
D/AO/AOO/C/C2 - C2因果推理引擎模块

提供完整的因果推理功能，包括因果发现、效应估计、反事实推理等。

主要类:
- CausalReasoningEngine: 主要的因果推理引擎类
- CausalEffect: 因果效应数据结构
- CausalRelation: 因果关系数据结构
- CausalAlgorithm: 支持的算法枚举


日期: 2025-11-05
"""

from .CausalReasoningEngine import (
    CausalReasoningEngine,
    CausalEffect,
    CausalRelation,
    CausalAlgorithm
)

__version__ = "1.0.0"
__author__ = "AI系统"

__all__ = [
    'CausalReasoningEngine',
    'CausalEffect', 
    'CausalRelation',
    'CausalAlgorithm'
]