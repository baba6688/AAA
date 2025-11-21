#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q8代码文档生成器包

一个功能完整的代码文档生成工具，支持多种编程语言
主要功能包括：
- 代码解析（解析多种编程语言代码）
- 注释生成（自动生成代码注释）
- 函数文档（函数和类的文档化）
- 依赖分析（代码依赖关系分析）
- 流程图生成（代码流程图和架构图）
- API文档（自动生成API文档）
- 代码质量分析（代码质量评估）
- 文档模板（多种文档模板支持）

版本: 1.0.0
作者: Q8团队
"""

__version__ = "1.0.0"
__author__ = "Q8团队"
__email__ = "q8@example.com"
__description__ = "Q8代码文档生成器 - 智能代码文档生成工具"

# 导入主要类和函数
from .CodeDocGenerator import (
    # 主要类
    CodeDocGenerator,
    PythonCodeParser,
    CodeCommentGenerator,
    DependencyAnalyzer,
    FlowchartGenerator,
    QualityAnalyzer,
    DocumentTemplate,
    
    # 数据类
    CodeElement,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
)

# 导出公共接口
__all__ = [
    # 主要类
    'CodeDocGenerator',
    'PythonCodeParser', 
    'CodeCommentGenerator',
    'DependencyAnalyzer',
    'FlowchartGenerator',
    'QualityAnalyzer',
    'DocumentTemplate',
    
    # 数据类
    'CodeElement',
    'FunctionInfo', 
    'ClassInfo',
    'ImportInfo',
    
    # 版本信息
    '__version__',
    '__author__',
    '__email__',
    '__description__',
]

# 包级别的方法
def create_generator():
    """
    创建代码文档生成器实例
    
    Returns:
        CodeDocGenerator: 代码文档生成器实例
    """
    return CodeDocGenerator()

def quick_generate(source_path, output_path="docs", template_type="markdown"):
    """
    快速生成文档
    
    Args:
        source_path (str): 源代码路径
        output_path (str, optional): 输出目录. 默认为 "docs"
        template_type (str, optional): 文档模板类型. 默认为 "markdown"
    
    Returns:
        dict: 生成的文档内容
    """
    generator = CodeDocGenerator()
    return generator.generate_documentation(
        source_path=source_path,
        output_path=output_path,
        template_type=template_type
    )

# 支持的文档模板类型
SUPPORTED_TEMPLATES = {
    'markdown': 'Markdown格式文档',
    'html': 'HTML格式文档', 
    'rst': 'reStructuredText格式文档',
    'api': 'API文档格式'
}

# 支持的编程语言
SUPPORTED_LANGUAGES = {
    'python': 'Python',
    # 未来可扩展其他语言
    # 'javascript': 'JavaScript',
    # 'java': 'Java',
    # 'cpp': 'C++',
}

# 默认配置
DEFAULT_CONFIG = {
    'template_type': 'markdown',
    'include_flowcharts': True,
    'include_quality_analysis': True,
    'include_dependency_analysis': True,
    'output_encoding': 'utf-8',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
}

def get_version():
    """获取版本信息"""
    return __version__

def get_supported_templates():
    """获取支持的文档模板"""
    return SUPPORTED_TEMPLATES.copy()

def get_supported_languages():
    """获取支持的编程语言"""
    return SUPPORTED_LANGUAGES.copy()

def get_default_config():
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()

# 示例用法
def example_usage():
    """使用示例"""
    print("Q8代码文档生成器使用示例:")
    print("=" * 40)
    print()
    print("1. 基本使用:")
    print("   from Q8 import CodeDocGenerator")
    print("   generator = CodeDocGenerator()")
    print("   generator.generate_documentation('your_project/', 'docs/')")
    print()
    print("2. 快速生成:")
    print("   from Q8 import quick_generate")
    print("   quick_generate('your_project/', 'docs/', 'html')")
    print()
    print("3. 命令行使用:")
    print("   python -m Q8 your_project/ -o docs/ -t markdown")
    print()
    print("支持的模板类型:", list(SUPPORTED_TEMPLATES.keys()))
    print("支持的编程语言:", list(SUPPORTED_LANGUAGES.keys()))

if __name__ == '__main__':
    example_usage()