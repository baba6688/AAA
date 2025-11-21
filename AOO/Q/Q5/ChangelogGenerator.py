#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5变更日志生成器
支持版本管理、变更分类、Git集成、多格式输出等功能
"""

import os
import re
import json
import datetime
import subprocess
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import argparse


@dataclass
class VersionInfo:
    """版本信息数据类"""
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    @classmethod
    def parse(cls, version_str: str) -> 'VersionInfo':
        """解析版本字符串"""
        # 语义化版本正则表达式
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        match = re.match(pattern, version_str)
        
        if not match:
            raise ValueError(f"无效的版本格式: {version_str}")
        
        major, minor, patch, prerelease, build = match.groups()
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease=prerelease or "",
            build=build or ""
        )
    
    def bump(self, part: str) -> 'VersionInfo':
        """递增版本号"""
        new_version = self.__class__(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            prerelease=self.prerelease,
            build=self.build
        )
        
        if part == 'major':
            new_version.major += 1
            new_version.minor = 0
            new_version.patch = 0
        elif part == 'minor':
            new_version.minor += 1
            new_version.patch = 0
        elif part == 'patch':
            new_version.patch += 1
        elif part == 'prerelease':
            if not new_version.prerelease:
                new_version.prerelease = "alpha.1"
            else:
                # 尝试递增预发布版本号
                match = re.match(r'^(.*?)(\d+)$', new_version.prerelease)
                if match:
                    prefix, num = match.groups()
                    new_version.prerelease = f"{prefix}{int(num) + 1}"
                else:
                    new_version.prerelease = f"{new_version.prerelease}.1"
        elif part == 'build':
            if not new_version.build:
                new_version.build = "build.1"
            else:
                match = re.match(r'^(.*?)(\d+)$', new_version.build)
                if match:
                    prefix, num = match.groups()
                    new_version.build = f"{prefix}{int(num) + 1}"
                else:
                    new_version.build = f"{new_version.build}.1"
        
        return new_version


@dataclass
class ChangeEntry:
    """变更条目数据类"""
    type: str  # feature, fix, docs, style, refactor, perf, test, chore
    scope: str = ""  # 变更范围
    description: str = ""  # 变更描述
    breaking: bool = False  # 是否为破坏性变更
    author: str = ""  # 提交者
    commit_hash: str = ""  # 提交哈希
    date: str = ""  # 提交日期
    body: str = ""  # 提交正文
    footer: str = ""  # 提交脚注
    
    def to_markdown(self) -> str:
        """转换为Markdown格式"""
        scope_part = f"({self.scope})" if self.scope else ""
        breaking_part = " [BREAKING]" if self.breaking else ""
        author_part = f" by @{self.author}" if self.author else ""
        
        return f"- {self.type}{scope_part}: {self.description}{breaking_part}{author_part}"
    
    def to_html(self) -> str:
        """转换为HTML格式"""
        scope_part = f"({self.scope})" if self.scope else ""
        breaking_part = " <span class='breaking'>[BREAKING]</span>" if self.breaking else ""
        author_part = f" by <span class='author'>@{self.author}</span>" if self.author else ""
        
        return f"<li><span class='type'>{self.type}</span>{scope_part}: {self.description}{breaking_part}{author_part}</li>"


@dataclass
class ReleaseInfo:
    """发布信息数据类"""
    version: VersionInfo
    date: str
    changes: List[ChangeEntry]
    contributors: List[str]
    breaking_changes: List[ChangeEntry]
    features: List[ChangeEntry]
    fixes: List[ChangeEntry]
    other_changes: List[ChangeEntry]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version': str(self.version),
            'date': self.date,
            'changes': [asdict(change) for change in self.changes],
            'contributors': self.contributors,
            'breaking_changes': [asdict(change) for change in self.breaking_changes],
            'features': [asdict(change) for change in self.features],
            'fixes': [asdict(change) for change in self.fixes],
            'other_changes': [asdict(change) for change in self.other_changes]
        }


class ChangelogGenerator:
    """变更日志生成器"""
    
    # 变更类型映射
    CHANGE_TYPES = {
        'feat': {'name': 'Features', 'icon': '✨'},
        'feature': {'name': 'Features', 'icon': '✨'},
        'fix': {'name': 'Bug Fixes', 'icon': '🐛'},
        'bug': {'name': 'Bug Fixes', 'icon': '🐛'},
        'docs': {'name': 'Documentation', 'icon': '📚'},
        'doc': {'name': 'Documentation', 'icon': '📚'},
        'style': {'name': 'Style', 'icon': '💄'},
        'refactor': {'name': 'Refactoring', 'icon': '♻️'},
        'perf': {'name': 'Performance', 'icon': '⚡'},
        'test': {'name': 'Tests', 'icon': '✅'},
        'chore': {'name': 'Chores', 'icon': '🔧'},
        'ci': {'name': 'CI/CD', 'icon': '🔄'},
        'build': {'name': 'Build', 'icon': '🏗️'}
    }
    
    def __init__(self, repo_path: str = ".", config: Optional[Dict[str, Any]] = None):
        """
        初始化变更日志生成器
        
        Args:
            repo_path: Git仓库路径
            config: 配置字典
        """
        self.repo_path = Path(repo_path)
        self.config = config or {}
        self.changelog_file = self.config.get('changelog_file', 'CHANGELOG.md')
        self.unreleased_file = self.config.get('unreleased_file', 'UNRELEASED.md')
        self.version_file = self.config.get('version_file', 'VERSION')
        
    def parse_conventional_commit(self, commit_message: str, commit_hash: str, 
                                author: str, date: str) -> Optional[ChangeEntry]:
        """
        解析 Conventional Commit 格式的提交信息
        
        Args:
            commit_message: 提交信息
            commit_hash: 提交哈希
            author: 提交者
            date: 提交日期
            
        Returns:
            变更条目或None
        """
        # Conventional Commit 正则表达式
        pattern = r'^(\w+)(\(([^\)]+)\))?(!)?:\s+(.+?)(?:\n\n(.+?)(?:\n\n(.+))?)?$'
        match = re.match(pattern, commit_message, re.DOTALL)
        
        if not match:
            return None
        
        type_part, scope_part, scope, breaking, description, body, footer = match.groups()
        
        change_type = type_part.lower()
        if change_type not in self.CHANGE_TYPES:
            return None
        
        return ChangeEntry(
            type=change_type,
            scope=scope or "",
            description=description.strip(),
            breaking=bool(breaking),
            author=author,
            commit_hash=commit_hash,
            date=date,
            body=body.strip() if body else "",
            footer=footer.strip() if footer else ""
        )
    
    def get_git_commits(self, since: Optional[str] = None, until: Optional[str] = None) -> List[Dict[str, str]]:
        """
        获取Git提交记录
        
        Args:
            since: 开始时间
            until: 结束时间
            
        Returns:
            提交记录列表
        """
        try:
            cmd = ['git', 'log', '--pretty=format:%H|%an|%ad|%s', '--date=short']
            
            if since:
                cmd.append(f'--since={since}')
            if until:
                cmd.append(f'--until={until}')
                
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        commits.append({
                            'hash': parts[0],
                            'author': parts[1],
                            'date': parts[2],
                            'message': parts[3]
                        })
            
            return commits
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"获取Git提交记录失败: {e}")
        except Exception as e:
            raise RuntimeError(f"Git操作失败: {e}")
    
    def get_tags(self) -> List[str]:
        """获取所有Git标签"""
        try:
            result = subprocess.run(['git', 'tag'], cwd=self.repo_path, capture_output=True, text=True)
            if result.returncode == 0:
                return sorted(result.stdout.strip().split('\n'), reverse=True)
            return []
        except Exception:
            return []
    
    def get_latest_tag(self) -> Optional[str]:
        """获取最新标签"""
        tags = self.get_tags()
        return tags[0] if tags else None
    
    def generate_changes_from_commits(self, commits: List[Dict[str, str]]) -> ReleaseInfo:
        """
        从提交记录生成变更信息
        
        Args:
            commits: 提交记录列表
            
        Returns:
            发布信息
        """
        changes = []
        contributors = set()
        
        for commit in commits:
            change = self.parse_conventional_commit(
                commit['message'], 
                commit['hash'], 
                commit['author'], 
                commit['date']
            )
            if change:
                changes.append(change)
                contributors.add(commit['author'])
        
        # 按类型分组
        features = [c for c in changes if c.type in ['feat', 'feature']]
        fixes = [c for c in changes if c.type == 'fix']
        breaking_changes = [c for c in changes if c.breaking]
        other_changes = [c for c in changes if c not in features + fixes + breaking_changes]
        
        # 创建虚拟版本（用于未发布变更）
        version = VersionInfo(0, 0, 0, "unreleased")
        
        return ReleaseInfo(
            version=version,
            date=datetime.date.today().isoformat(),
            changes=changes,
            contributors=list(contributors),
            breaking_changes=breaking_changes,
            features=features,
            fixes=fixes,
            other_changes=other_changes
        )
    
    def generate_markdown(self, release_info: ReleaseInfo) -> str:
        """
        生成Markdown格式的变更日志
        
        Args:
            release_info: 发布信息
            
        Returns:
            Markdown格式的变更日志
        """
        lines = []
        
        # 版本标题
        if release_info.version.prerelease == "unreleased":
            lines.append(f"## [{release_info.version.prerelease}] - {release_info.date}")
        else:
            lines.append(f"## [{release_info.version}] - {release_info.date}")
        
        lines.append("")
        
        # 破坏性变更
        if release_info.breaking_changes:
            lines.append("### ⚠️ Breaking Changes")
            lines.append("")
            for change in release_info.breaking_changes:
                lines.append(change.to_markdown())
            lines.append("")
        
        # 功能新增
        if release_info.features:
            lines.append("### ✨ Features")
            lines.append("")
            for change in release_info.features:
                lines.append(change.to_markdown())
            lines.append("")
        
        # 修复
        if release_info.fixes:
            lines.append("### 🐛 Bug Fixes")
            lines.append("")
            for change in release_info.fixes:
                lines.append(change.to_markdown())
            lines.append("")
        
        # 其他变更
        if release_info.other_changes:
            lines.append("### 🔧 Other Changes")
            lines.append("")
            for change in release_info.other_changes:
                lines.append(change.to_markdown())
            lines.append("")
        
        # 贡献者
        if release_info.contributors:
            lines.append("### 👥 Contributors")
            lines.append("")
            for contributor in sorted(release_info.contributors):
                lines.append(f"- @{contributor}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_html(self, release_info: ReleaseInfo) -> str:
        """
        生成HTML格式的变更日志
        
        Args:
            release_info: 发布信息
            
        Returns:
            HTML格式的变更日志
        """
        html = []
        
        # HTML头部
        html.append("<!DOCTYPE html>")
        html.append("<html lang='zh-CN'>")
        html.append("<head>")
        html.append("    <meta charset='UTF-8'>")
        html.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append("    <title>变更日志</title>")
        html.append("    <style>")
        html.append("        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }")
        html.append("        .version { border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }")
        html.append("        .version h2 { color: #0366d6; margin-bottom: 10px; }")
        html.append("        .change-section { margin: 20px 0; }")
        html.append("        .change-section h3 { color: #24292e; border-bottom: 1px solid #e1e4e8; padding-bottom: 5px; }")
        html.append("        .change-section ul { list-style: none; padding: 0; }")
        html.append("        .change-section li { margin: 5px 0; padding: 5px 0; }")
        html.append("        .type { background: #f1f8ff; color: #0366d6; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }")
        html.append("        .breaking { background: #ffeaa7; color: #d63031; font-weight: bold; }")
        html.append("        .author { color: #586069; font-style: italic; }")
        html.append("    </style>")
        html.append("</head>")
        html.append("<body>")
        
        # 版本标题
        if release_info.version.prerelease == "unreleased":
            html.append(f"    <div class='version'><h2>版本 {release_info.version.prerelease} - {release_info.date}</h2></div>")
        else:
            html.append(f"    <div class='version'><h2>版本 {release_info.version} - {release_info.date}</h2></div>")
        
        # 破坏性变更
        if release_info.breaking_changes:
            html.append("    <div class='change-section'>")
            html.append("        <h3>⚠️ 破坏性变更</h3>")
            html.append("        <ul>")
            for change in release_info.breaking_changes:
                html.append(f"            {change.to_html()}")
            html.append("        </ul>")
            html.append("    </div>")
        
        # 功能新增
        if release_info.features:
            html.append("    <div class='change-section'>")
            html.append("        <h3>✨ 新功能</h3>")
            html.append("        <ul>")
            for change in release_info.features:
                html.append(f"            {change.to_html()}")
            html.append("        </ul>")
            html.append("    </div>")
        
        # 修复
        if release_info.fixes:
            html.append("    <div class='change-section'>")
            html.append("        <h3>🐛 错误修复</h3>")
            html.append("        <ul>")
            for change in release_info.fixes:
                html.append(f"            {change.to_html()}")
            html.append("        </ul>")
            html.append("    </div>")
        
        # 其他变更
        if release_info.other_changes:
            html.append("    <div class='change-section'>")
            html.append("        <h3>🔧 其他变更</h3>")
            html.append("        <ul>")
            for change in release_info.other_changes:
                html.append(f"            {change.to_html()}")
            html.append("        </ul>")
            html.append("    </div>")
        
        # 贡献者
        if release_info.contributors:
            html.append("    <div class='change-section'>")
            html.append("        <h3>👥 贡献者</h3>")
            html.append("        <ul>")
            for contributor in sorted(release_info.contributors):
                html.append(f"            <li>@{contributor}</li>")
            html.append("        </ul>")
            html.append("    </div>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def generate_json(self, release_info: ReleaseInfo) -> str:
        """
        生成JSON格式的变更日志
        
        Args:
            release_info: 发布信息
            
        Returns:
            JSON格式的变更日志
        """
        return json.dumps(release_info.to_dict(), ensure_ascii=False, indent=2)
    
    def generate_release_notes(self, release_info: ReleaseInfo) -> str:
        """
        生成发布说明
        
        Args:
            release_info: 发布信息
            
        Returns:
            发布说明文本
        """
        lines = []
        
        if release_info.version.prerelease == "unreleased":
            lines.append(f"## 未发布版本变更")
        else:
            lines.append(f"## 版本 {release_info.version} 发布")
        
        lines.append("")
        
        # 概要
        if release_info.features:
            lines.append(f"✨ 本次发布包含 {len(release_info.features)} 个新功能")
        if release_info.fixes:
            lines.append(f"🐛 修复了 {len(release_info.fixes)} 个问题")
        if release_info.breaking_changes:
            lines.append(f"⚠️ 包含 {len(release_info.breaking_changes)} 个破坏性变更")
        
        lines.append("")
        
        # 主要变更
        if release_info.features:
            lines.append("### 主要新功能")
            lines.append("")
            for change in release_info.features[:5]:  # 只显示前5个
                lines.append(f"- {change.description}")
            if len(release_info.features) > 5:
                lines.append(f"- ... 还有 {len(release_info.features) - 5} 个其他功能")
            lines.append("")
        
        if release_info.breaking_changes:
            lines.append("### ⚠️ 重要变更")
            lines.append("")
            for change in release_info.breaking_changes:
                lines.append(f"- {change.description}")
            lines.append("")
        
        # 贡献者
        if release_info.contributors:
            lines.append("### 感谢贡献者")
            lines.append("")
            lines.append(f"感谢以下 {len(release_info.contributors)} 位贡献者的参与：")
            lines.append("")
            for contributor in sorted(release_info.contributors):
                lines.append(f"- @{contributor}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_changelog(self, since: Optional[str] = None, until: Optional[str] = None,
                          format: str = 'markdown', output_file: Optional[str] = None) -> str:
        """
        生成变更日志
        
        Args:
            since: 开始时间
            until: 结束时间
            format: 输出格式 (markdown, html, json)
            output_file: 输出文件路径
            
        Returns:
            生成的变更日志内容
        """
        # 获取提交记录
        commits = self.get_git_commits(since, until)
        
        # 生成变更信息
        release_info = self.generate_changes_from_commits(commits)
        
        # 生成对应格式的内容
        if format.lower() == 'html':
            content = self.generate_html(release_info)
        elif format.lower() == 'json':
            content = self.generate_json(release_info)
        else:
            content = self.generate_markdown(release_info)
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return content
    
    def generate_release(self, version: str, format: str = 'markdown', 
                        output_file: Optional[str] = None) -> str:
        """
        生成特定版本的发布信息
        
        Args:
            version: 版本号
            format: 输出格式
            output_file: 输出文件路径
            
        Returns:
            生成的发布信息内容
        """
        # 获取标签对应的提交
        try:
            result = subprocess.run(['git', 'show', '--pretty=format:%H|%an|%ad|%s', '--date=short', version], 
                                  cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"标签 {version} 不存在")
            
            # 获取该标签之后的所有提交
            since_tag = version
            commits = self.get_git_commits(since=since_tag)
            
            # 生成变更信息
            release_info = self.generate_changes_from_commits(commits)
            release_info.version = VersionInfo.parse(version)
            
            # 获取标签日期
            try:
                tag_result = subprocess.run(['git', 'show', '--format=%ci', '--date=short', version, '--no-patch'], 
                                          cwd=self.repo_path, capture_output=True, text=True)
                if tag_result.returncode == 0:
                    release_info.date = tag_result.stdout.strip().split()[0]
            except:
                pass
            
            # 生成对应格式的内容
            if format.lower() == 'html':
                content = self.generate_html(release_info)
            elif format.lower() == 'json':
                content = self.generate_json(release_info)
            else:
                content = self.generate_markdown(release_info)
            
            # 保存到文件
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            return content
            
        except Exception as e:
            raise RuntimeError(f"生成发布信息失败: {e}")
    
    def update_changelog(self, version: Optional[str] = None, format: str = 'markdown') -> str:
        """
        更新变更日志文件
        
        Args:
            version: 版本号，如果为None则更新未发布部分
            format: 输出格式
            
        Returns:
            生成的变更日志内容
        """
        if version:
            # 生成特定版本的发布信息
            return self.generate_release(version, format)
        else:
            # 生成未发布的变更
            return self.generate_changelog(format=format)
    
    def bump_version(self, part: str, current_version: Optional[str] = None) -> str:
        """
        递增版本号
        
        Args:
            part: 递增部分 (major, minor, patch, prerelease, build)
            current_version: 当前版本，如果为None则从版本文件读取
            
        Returns:
            新的版本号
        """
        if current_version:
            version = VersionInfo.parse(current_version)
        elif self.version_file and Path(self.version_file).exists():
            with open(self.version_file, 'r', encoding='utf-8') as f:
                version = VersionInfo.parse(f.read().strip())
        else:
            # 默认版本
            version = VersionInfo(1, 0, 0)
        
        new_version = version.bump(part)
        
        # 保存到版本文件
        if self.version_file:
            with open(self.version_file, 'w', encoding='utf-8') as f:
                f.write(str(new_version))
        
        return str(new_version)
    
    def tag_version(self, version: str, message: Optional[str] = None) -> None:
        """
        创建版本标签
        
        Args:
            version: 版本号
            message: 标签信息
        """
        try:
            tag_message = message or f"Release version {version}"
            subprocess.run(['git', 'tag', '-a', version, '-m', tag_message], 
                         cwd=self.repo_path, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"创建标签失败: {e}")
    
    def get_changelog_history(self) -> List[Dict[str, Any]]:
        """
        获取变更日志历史
        
        Returns:
            变更日志历史列表
        """
        history = []
        tags = self.get_tags()
        
        for tag in tags:
            try:
                content = self.generate_release(tag, 'json')
                release_info = json.loads(content)
                history.append(release_info)
            except Exception:
                continue
        
        return sorted(history, key=lambda x: x['version'], reverse=True)


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='Q5变更日志生成器')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 生成变更日志命令
    changelog_parser = subparsers.add_parser('changelog', help='生成变更日志')
    changelog_parser.add_argument('--since', help='开始时间')
    changelog_parser.add_argument('--until', help='结束时间')
    changelog_parser.add_argument('--format', choices=['markdown', 'html', 'json'], 
                                 default='markdown', help='输出格式')
    changelog_parser.add_argument('--output', '-o', help='输出文件路径')
    
    # 生成发布信息命令
    release_parser = subparsers.add_parser('release', help='生成发布信息')
    release_parser.add_argument('version', help='版本号')
    release_parser.add_argument('--format', choices=['markdown', 'html', 'json'], 
                               default='markdown', help='输出格式')
    release_parser.add_argument('--output', '-o', help='输出文件路径')
    
    # 递增版本命令
    bump_parser = subparsers.add_parser('bump', help='递增版本号')
    bump_parser.add_argument('part', choices=['major', 'minor', 'patch', 'prerelease', 'build'],
                           help='递增部分')
    bump_parser.add_argument('--current', help='当前版本号')
    
    # 创建标签命令
    tag_parser = subparsers.add_parser('tag', help='创建版本标签')
    tag_parser.add_argument('version', help='版本号')
    tag_parser.add_argument('--message', '-m', help='标签信息')
    
    # 历史命令
    history_parser = subparsers.add_parser('history', help='获取变更日志历史')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    generator = ChangelogGenerator()
    
    try:
        if args.command == 'changelog':
            content = generator.generate_changelog(
                since=args.since,
                until=args.until,
                format=args.format,
                output_file=args.output
            )
            if not args.output:
                print(content)
        
        elif args.command == 'release':
            content = generator.generate_release(
                version=args.version,
                format=args.format,
                output_file=args.output
            )
            if not args.output:
                print(content)
        
        elif args.command == 'bump':
            new_version = generator.bump_version(args.part, args.current)
            print(f"新版本: {new_version}")
        
        elif args.command == 'tag':
            generator.tag_version(args.version, args.message)
            print(f"标签 {args.version} 创建成功")
        
        elif args.command == 'history':
            history = generator.get_changelog_history()
            print(json.dumps(history, ensure_ascii=False, indent=2))
    
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())