#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5变更日志生成器包

这是一个功能完整的变更日志生成工具，支持：
- 版本管理和标记
- 变更分类和识别
- Git集成和提交解析
- 多格式输出（Markdown、HTML、JSON）
- 自动化生成和CI/CD集成
- 发布说明生成
- 贡献者信息统计
- 语义化版本规范支持

主要组件：
- ChangelogGenerator: 主要的变更日志生成器类
- VersionInfo: 版本信息数据类
- ChangeEntry: 变更条目数据类
- ReleaseInfo: 发布信息数据类

使用方法：
    from Q5 import ChangelogGenerator
    
    generator = ChangelogGenerator()
    changelog = generator.generate_changelog()
    print(changelog)

或者直接使用命令行：
    python ChangelogGenerator.py changelog
"""

from .ChangelogGenerator import (
    ChangelogGenerator,
    VersionInfo,
    ChangeEntry,
    ReleaseInfo
)

__version__ = "1.0.0"
__author__ = "Q5 Development Team"
__email__ = "dev@q5.example.com"
__license__ = "MIT"

__all__ = [
    "ChangelogGenerator",
    "VersionInfo", 
    "ChangeEntry",
    "ReleaseInfo"
]

# 包级别的便捷函数
def create_generator(repo_path=".", config=None):
    """
    创建变更日志生成器实例的便捷函数
    
    Args:
        repo_path: Git仓库路径，默认为当前目录
        config: 配置字典，可选
        
    Returns:
        ChangelogGenerator实例
    """
    return ChangelogGenerator(repo_path=repo_path, config=config)

def quick_changelog(repo_path=".", format="markdown", output_file=None):
    """
    快速生成变更日志的便捷函数
    
    Args:
        repo_path: Git仓库路径，默认为当前目录
        format: 输出格式，默认为markdown
        output_file: 输出文件路径，可选
        
    Returns:
        生成的变更日志内容
    """
    generator = ChangelogGenerator(repo_path=repo_path)
    return generator.generate_changelog(format=format, output_file=output_file)

def quick_release(version, repo_path=".", format="markdown", output_file=None):
    """
    快速生成发布信息的便捷函数
    
    Args:
        version: 版本号
        repo_path: Git仓库路径，默认为当前目录
        format: 输出格式，默认为markdown
        output_file: 输出文件路径，可选
        
    Returns:
        生成的发布信息内容
    """
    generator = ChangelogGenerator(repo_path=repo_path)
    return generator.generate_release(version=version, format=format, output_file=output_file)

# 版本信息
VERSION_INFO = {
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "description": "Q5变更日志生成器 - 智能变更日志生成工具",
    "features": [
        "版本管理和标记",
        "变更分类和识别", 
        "Git集成和提交解析",
        "多格式输出支持",
        "自动化生成能力",
        "发布说明生成",
        "贡献者信息统计",
        "语义化版本规范"
    ],
    "supported_formats": ["markdown", "html", "json"],
    "supported_commit_types": [
        "feat", "fix", "docs", "style", "refactor", 
        "perf", "test", "chore", "ci", "build"
    ],
    "python_version": ">=3.7",
    "git_version": ">=2.0"
}

def get_version_info():
    """
    获取版本信息
    
    Returns:
        包含版本详细信息的字典
    """
    return VERSION_INFO.copy()

def print_version_info():
    """打印版本信息"""
    print(f"Q5变更日志生成器 v{__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print()
    print("主要特性:")
    for feature in VERSION_INFO["features"]:
        print(f"  ✓ {feature}")
    print()
    print("支持格式:", ", ".join(VERSION_INFO["supported_formats"]))
    print("支持提交类型:", ", ".join(VERSION_INFO["supported_commit_types"]))
    print()
    print("使用示例:")
    print("  from Q5 import quick_changelog")
    print("  changelog = quick_changelog()")
    print()
    print("或使用命令行:")
    print("  python ChangelogGenerator.py changelog")

# 初始化时的欢迎信息
def _welcome_message():
    """显示欢迎信息"""
    print("=" * 50)
    print("🚀 Q5变更日志生成器 v1.0.0")
    print("=" * 50)
    print("智能变更日志生成工具已就绪！")
    print()
    print("快速开始:")
    print("  python ChangelogGenerator.py changelog")
    print("  python ChangelogGenerator.py release v1.0.0")
    print()
    print("获取帮助:")
    print("  python ChangelogGenerator.py --help")
    print("=" * 50)

# 如果直接运行此文件，显示版本信息
if __name__ == "__main__":
    print_version_info()