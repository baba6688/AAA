"""
Q2用户手册生成器包

这个包提供了完整的用户手册生成解决方案，包括：
- 内容结构化组织
- 多种模板系统
- 多媒体支持
- 分步指导生成
- FAQ管理
- 用户反馈处理
- 多语言支持
- 版本控制
"""

from .UserManualGenerator import UserManualGenerator
from .UserManualGenerator import (
    ContentStructure,
    TemplateManager,
    MultimediaHandler,
    StepByStepGuide,
    FAQManager,
    FeedbackHandler,
    MultiLanguageSupport,
    VersionControl
)

__version__ = "1.0.0"
__author__ = "Q2开发团队"
__description__ = "专业的用户手册生成器"

__all__ = [
    "UserManualGenerator",
    "ContentStructure",
    "TemplateManager", 
    "MultimediaHandler",
    "StepByStepGuide",
    "FAQManager",
    "FeedbackHandler",
    "MultiLanguageSupport",
    "VersionControl"
]