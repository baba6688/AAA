"""
Q1 API文档生成器包

这是一个功能完整的API文档生成器，支持：
- 代码解析（解析Python代码中的函数、类、参数）
- 文档字符串解析（提取和分析docstring）
- API端点识别（识别REST API端点）
- 参数文档化（文档化函数参数和返回值）
- 示例代码生成（自动生成使用示例）
- 多格式输出（HTML、Markdown、PDF等）
- 交互式文档（可搜索的在线文档）
- 版本管理（支持API版本管理）

版本: 1.0.0
作者: Q1团队
"""

from .APIDocGenerator import (
    APIDocGenerator,
    CodeParser,
    DocstringParser,
    EndpointDetector,
    ParameterDocumenter,
    ExampleGenerator,
    OutputFormatter,
    InteractiveDocumentation,
    VersionManager
)

__version__ = "1.0.0"
__author__ = "Q1团队"

__all__ = [
    "APIDocGenerator",
    "CodeParser", 
    "DocstringParser",
    "EndpointDetector",
    "ParameterDocumenter",
    "ExampleGenerator",
    "OutputFormatter",
    "InteractiveDocumentation",
    "VersionManager"
]