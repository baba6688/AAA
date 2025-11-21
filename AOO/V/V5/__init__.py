"""
V5模块
======

V5模块包含模型版本控制器等核心组件。

主要组件:
- ModelVersionController: 模型版本控制器
"""

from .ModelVersionController import (
    ModelVersionController,
    ModelMetadata,
    ChangeRecord,
    VersionComparison,
    VersionStatus,
    MergeStrategy
)

__version__ = "5.0.0"
__all__ = [
    "ModelVersionController",
    "ModelMetadata", 
    "ChangeRecord",
    "VersionComparison",
    "VersionStatus",
    "MergeStrategy"
]