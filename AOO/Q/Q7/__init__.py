#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q7教程生成器包
一个功能完整的教程生成工具，支持内容结构化、步骤指导、代码示例、互动元素等功能

主要组件:
- TutorialGenerator: 主要的教程生成器类
- DifficultyLevel: 难度级别枚举
- ContentType: 内容类型枚举
- create_sample_tutorial: 创建示例教程的函数

使用示例:
    from Q7 import TutorialGenerator, DifficultyLevel
    
    tutorial = TutorialGenerator("my_tutorial", "我的教程", "描述")
    chapter = tutorial.create_chapter("第一章", "描述", 30, DifficultyLevel.BEGINNER)
    tutorial.create_text_content(chapter.id, "标题", "内容")
    html = tutorial.generate_tutorial_html("user_001")
"""

__version__ = "1.0.0"
__author__ = "Q7教程生成器团队"
__description__ = "功能完整的教程生成工具"

# 导入主要类和函数
from .TutorialGenerator import (
    TutorialGenerator,
    DifficultyLevel,
    ContentType,
    CodeExample,
    QuizQuestion,
    Exercise,
    ContentBlock,
    Chapter,
    ProgressRecord,
    create_sample_tutorial
)

# 公开API
__all__ = [
    "TutorialGenerator",
    "DifficultyLevel", 
    "ContentType",
    "CodeExample",
    "QuizQuestion",
    "Exercise",
    "ContentBlock",
    "Chapter",
    "ProgressRecord",
    "create_sample_tutorial"
]