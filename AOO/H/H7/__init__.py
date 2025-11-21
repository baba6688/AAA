#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H7知识更新器模块

实现知识更新系统的核心功能，包括：
1. 更新状态、冲突类型、更新策略枚举
2. 知识项目、更新需求、更新过程数据类
3. 更新验证、冲突解决机制
4. 更新历史管理和报告生成
5. 知识更新器主类和策略管理
"""

from .KnowledgeUpdater import (
    # 枚举类
    UpdateStatus,
    ConflictType,
    UpdateStrategy,
    # 核心数据类
    KnowledgeItem,
    UpdateRequirement,
    UpdateProcess,
    UpdateValidation,
    KnowledgeConflict,
    UpdateHistory,
    UpdateReport,
    # 主更新器
    KnowledgeUpdater
)

__version__ = "1.0.0"
__author__ = "H7 Team"

__all__ = [
    # 枚举类
    "UpdateStatus",
    "ConflictType",
    "UpdateStrategy",
    # 核心数据类
    "KnowledgeItem",
    "UpdateRequirement",
    "UpdateProcess",
    "UpdateValidation",
    "KnowledgeConflict",
    "UpdateHistory",
    "UpdateReport",
    # 主更新器
    "KnowledgeUpdater"
]