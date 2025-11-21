#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q8代码文档生成器
一个功能完整的代码文档生成工具，支持多种编程语言
"""

import os
import re
import ast
import json
import subprocess
import inspect
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import tempfile
import shutil
from datetime import datetime
import argparse


@dataclass
class CodeElement:
    """代码元素基类"""
    name: str
    type: str
    line_number: int
    content: str
    docstring: str = ""
    parameters: List[Dict] = None
    return_type: str = ""
    decorators: List[str] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.decorators is None:
            self.decorators = []


@dataclass
class FunctionInfo(CodeElement):
    """函数信息"""
    is_async: bool = False
    is_method: bool = False
    class_name: str = ""
    complexity: int = 0


@dataclass
class ClassInfo(CodeElement):
    """类信息"""
    base_classes: List[str] = None
    methods: List[FunctionInfo] = None
    attributes: List[Dict] = None
    is_abstract: bool = False
    
    def __post_init__(self):
        if self.base_classes is None:
            self.base_classes = []
        if self.methods is None:
            self.methods = []
        if self.attributes is None:
            self.attributes = []


@dataclass
class ImportInfo:
    """导入信息"""
    module: str
    alias: str = ""
    is_from: bool = False
    items: List[str] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []


class PythonCodeParser:
    """Python代码解析器"""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.variables = []
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """解析Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            self._parse_ast(tree, file_path)
            
            return {
                'functions': self.functions,
                'classes': self.classes,
                'imports': self.imports,
                'variables': self.variables,
                'file_path': file_path,
                'total_lines': len(content.splitlines())
            }
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
            return {}
    
    def _parse_ast(self, tree: ast.AST, file_path: str):
        """解析AST"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._parse_function(node, file_path)
            elif isinstance(node, ast.ClassDef):
                self._parse_class(node, file_path)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._parse_import(node)
    
    def _parse_function(self, node: ast.FunctionDef, file_path: str):
        """解析函数"""
        # 提取参数
        parameters = []
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': self._infer_type(arg.annotation) if arg.annotation else 'Any',
                'default': None
            }
            parameters.append(param_info)
        
        # 提取返回类型
        return_type = self._infer_type(node.returns) if node.returns else 'Any'
        
        # 计算复杂度（简单版本）
        complexity = self._calculate_complexity(node)
        
        func_info = FunctionInfo(
            name=node.name,
            type='function',
            line_number=node.lineno,
            content=ast.get_source_segment(open(file_path, 'r').read(), node) or '',
            docstring=ast.get_docstring(node) or '',
            parameters=parameters,
            return_type=return_type,
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            complexity=complexity
        )
        
        self.functions.append(func_info)
    
    def _parse_class(self, node: ast.ClassDef, file_path: str):
        """解析类"""
        # 提取基类
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(ast.unparse(base))
        
        # 提取方法
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._parse_function(item, file_path)
                methods.append(self.functions[-1])
        
        class_info = ClassInfo(
            name=node.name,
            type='class',
            line_number=node.lineno,
            content=ast.get_source_segment(open(file_path, 'r').read(), node) or '',
            docstring=ast.get_docstring(node) or '',
            base_classes=base_classes,
            methods=methods
        )
        
        self.classes.append(class_info)
    
    def _parse_import(self, node):
        """解析导入"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_info = ImportInfo(
                    module=alias.name,
                    alias=alias.asname or alias.name
                )
                self.imports.append(import_info)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                import_info = ImportInfo(
                    module=node.module or '',
                    alias=alias.asname or alias.name,
                    is_from=True,
                    items=[alias.name]
                )
                self.imports.append(import_info)
    
    def _infer_type(self, node) -> str:
        """推断类型"""
        if not node:
            return 'Any'
        
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return ast.unparse(node)
        elif isinstance(node, ast.Subscript):
            return ast.unparse(node)
        elif isinstance(node, ast.Constant):
            return type(node.value).__name__
        else:
            return 'Any'
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数复杂度（简单版本）"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
        return complexity


class CodeCommentGenerator:
    """代码注释生成器"""
    
    def __init__(self):
        self.templates = {
            'function': self._function_template,
            'class': self._class_template,
            'method': self._method_template
        }
    
    def generate_function_comment(self, func_info: FunctionInfo) -> str:
        """生成函数注释"""
        template = self.templates['function']
        return template(func_info)
    
    def generate_class_comment(self, class_info: ClassInfo) -> str:
        """生成类注释"""
        template = self.templates['class']
        return template(class_info)
    
    def _function_template(self, func_info: FunctionInfo) -> str:
        """函数注释模板"""
        docstring = f'"""\n'
        docstring += f'{func_info.name} 函数\n\n'
        
        if func_info.parameters:
            docstring += '参数:\n'
            for param in func_info.parameters:
                docstring += f'    {param["name"]} ({param["type"]})'
                if param.get("default"):
                    docstring += f', 默认值: {param["default"]}'
                docstring += f': 参数说明\n'
            docstring += '\n'
        
        if func_info.return_type != 'None':
            docstring += f'返回:\n    {func_info.return_type}: 返回值说明\n\n'
        
        if func_info.decorators:
            docstring += '装饰器:\n'
            for decorator in func_info.decorators:
                docstring += f'    @{decorator}\n'
            docstring += '\n'
        
        docstring += f'复杂度: {func_info.complexity}\n'
        docstring += '"""'
        
        return docstring
    
    def _class_template(self, class_info: ClassInfo) -> str:
        """类注释模板"""
        docstring = f'"""\n'
        docstring += f'{class_info.name} 类\n\n'
        
        if class_info.base_classes:
            docstring += f'继承自: {", ".join(class_info.base_classes)}\n\n'
        
        docstring += f'方法数量: {len(class_info.methods)}\n'
        docstring += f'属性数量: {len(class_info.attributes)}\n'
        docstring += '"""'
        
        return docstring
    
    def _method_template(self, func_info: FunctionInfo) -> str:
        """方法注释模板"""
        return self._function_template(func_info)


class DependencyAnalyzer:
    """依赖关系分析器"""
    
    def __init__(self):
        self.dependencies = {}
        self.reverse_dependencies = {}
    
    def analyze_file(self, file_path: str, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析文件依赖关系"""
        dependencies = []
        
        # 分析导入依赖
        for import_info in parsed_data.get('imports', []):
            dep_info = {
                'type': 'import',
                'module': import_info.module,
                'alias': import_info.alias,
                'items': import_info.items,
                'is_external': self._is_external_module(import_info.module)
            }
            dependencies.append(dep_info)
        
        # 分析函数调用依赖
        self._analyze_function_calls(file_path, parsed_data, dependencies)
        
        return {
            'file_path': file_path,
            'dependencies': dependencies,
            'external_dependencies': [dep for dep in dependencies if dep.get('is_external', False)],
            'internal_dependencies': [dep for dep in dependencies if not dep.get('is_external', False)]
        }
    
    def _is_external_module(self, module_name: str) -> bool:
        """判断是否为外部模块"""
        external_modules = {
            'os', 'sys', 'json', 'yaml', 'requests', 'numpy', 'pandas',
            'matplotlib', 'tensorflow', 'torch', 'django', 'flask',
            'pytest', 'unittest', 'logging', 'datetime', 'pathlib'
        }
        return module_name.split('.')[0] in external_modules
    
    def _analyze_function_calls(self, file_path: str, parsed_data: Dict[str, Any], dependencies: List[Dict]):
        """分析函数调用依赖"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的函数调用模式匹配
            patterns = [
                r'(\w+)\.(\w+)\(',  # obj.method()
                r'(\w+)\(',         # function()
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) == 2:
                        dep_info = {
                            'type': 'function_call',
                            'object': match[0],
                            'method': match[1],
                            'is_method_call': True
                        }
                    else:
                        dep_info = {
                            'type': 'function_call',
                            'function': match[0],
                            'is_method_call': False
                        }
                    dependencies.append(dep_info)
        
        except Exception as e:
            print(f"分析函数调用时出错: {e}")


class FlowchartGenerator:
    """流程图生成器"""
    
    def __init__(self):
        self.diagrams = {}
    
    def generate_function_flowchart(self, func_info: FunctionInfo) -> str:
        """生成函数流程图（Mermaid格式）"""
        diagram = f"graph TD\n"
        diagram += f'    Start(["开始"]) --> Input{{"输入参数检查"}}\n'
        diagram += f'    Input -->|有效| Process["执行主要逻辑"]\n'
        diagram += f'    Input -->|无效| Error["抛出异常"]\n'
        diagram += f'    Process --> Output{{"返回结果"}}\n'
        diagram += f'    Output -->|有返回值| Return["返回结果"]\n'
        diagram += f'    Output -->|无返回值| End(["结束"])\n'
        diagram += f'    Return --> End\n'
        diagram += f'    Error --> End\n'
        
        return diagram
    
    def generate_class_diagram(self, class_info: ClassInfo) -> str:
        """生成类图（Mermaid格式）"""
        diagram = f"classDiagram\n"
        diagram += f'    class {class_info.name} {{\n'
        
        # 添加属性
        for attr in class_info.attributes:
            diagram += f'        +{attr.get("name", "attribute")} : {attr.get("type", "Any")}\n'
        
        # 添加方法
        for method in class_info.methods:
            params = ', '.join([f'{p["name"]}: {p["type"]}' for p in method.parameters])
            diagram += f'        +{method.name}({params}) {method.return_type}\n'
        
        diagram += '    }\n'
        
        # 添加继承关系
        for base in class_info.base_classes:
            diagram += f'    {base} <|-- {class_info.name}\n'
        
        return diagram
    
    def generate_architecture_diagram(self, project_data: Dict[str, Any]) -> str:
        """生成项目架构图"""
        diagram = "graph TB\n"
        diagram += "    subgraph '项目架构'\n"
        
        # 添加主要组件
        for file_path, data in project_data.items():
            file_name = Path(file_path).stem
            diagram += f'        {file_name}["{file_name}"]\n'
            
            # 添加类和函数
            for class_info in data.get('classes', []):
                diagram += f'        {file_name} --> {class_info.name}\n'
            
            for func_info in data.get('functions', []):
                diagram += f'        {file_name} --> {func_info.name}\n'
        
        diagram += "    end\n"
        
        return diagram


class QualityAnalyzer:
    """代码质量分析器"""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_file(self, file_path: str, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析文件质量"""
        metrics = {}
        
        # 基本指标
        metrics['total_lines'] = parsed_data.get('total_lines', 0)
        metrics['code_lines'] = self._count_code_lines(parsed_data)
        metrics['comment_lines'] = self._count_comment_lines(parsed_data)
        metrics['blank_lines'] = self._count_blank_lines(parsed_data)
        
        # 函数和类指标
        metrics['function_count'] = len(parsed_data.get('functions', []))
        metrics['class_count'] = len(parsed_data.get('classes', []))
        metrics['average_function_length'] = self._calculate_average_function_length(parsed_data)
        metrics['average_function_complexity'] = self._calculate_average_complexity(parsed_data)
        
        # 质量评分
        metrics['complexity_score'] = self._calculate_complexity_score(metrics)
        metrics['documentation_score'] = self._calculate_documentation_score(parsed_data)
        metrics['overall_quality'] = self._calculate_overall_quality(metrics)
        
        return metrics
    
    def _count_code_lines(self, parsed_data: Dict[str, Any]) -> int:
        """统计代码行数"""
        code_lines = 0
        for func in parsed_data.get('functions', []):
            code_lines += len(func.content.splitlines())
        for class_info in parsed_data.get('classes', []):
            code_lines += len(class_info.content.splitlines())
        return code_lines
    
    def _count_comment_lines(self, parsed_data: Dict[str, Any]) -> int:
        """统计注释行数"""
        comment_lines = 0
        for func in parsed_data.get('functions', []):
            if func.docstring:
                comment_lines += len(func.docstring.splitlines())
        for class_info in parsed_data.get('classes', []):
            if class_info.docstring:
                comment_lines += len(class_info.docstring.splitlines())
        return comment_lines
    
    def _count_blank_lines(self, parsed_data: Dict[str, Any]) -> int:
        """统计空行数"""
        # 简单估算
        total_lines = parsed_data.get('total_lines', 0)
        code_lines = self._count_code_lines(parsed_data)
        comment_lines = self._count_comment_lines(parsed_data)
        return max(0, total_lines - code_lines - comment_lines)
    
    def _calculate_average_function_length(self, parsed_data: Dict[str, Any]) -> float:
        """计算平均函数长度"""
        functions = parsed_data.get('functions', [])
        if not functions:
            return 0.0
        
        total_length = sum(len(func.content.splitlines()) for func in functions)
        return total_length / len(functions)
    
    def _calculate_average_complexity(self, parsed_data: Dict[str, Any]) -> float:
        """计算平均复杂度"""
        functions = parsed_data.get('functions', [])
        if not functions:
            return 0.0
        
        total_complexity = sum(func.complexity for func in functions)
        return total_complexity / len(functions)
    
    def _calculate_complexity_score(self, metrics: Dict[str, Any]) -> int:
        """计算复杂度评分"""
        avg_complexity = metrics.get('average_function_complexity', 0)
        if avg_complexity <= 5:
            return 100
        elif avg_complexity <= 10:
            return 80
        elif avg_complexity <= 20:
            return 60
        else:
            return 40
    
    def _calculate_documentation_score(self, parsed_data: Dict[str, Any]) -> int:
        """计算文档评分"""
        functions = parsed_data.get('functions', [])
        classes = parsed_data.get('classes', [])
        
        total_elements = len(functions) + len(classes)
        if total_elements == 0:
            return 100
        
        documented_elements = 0
        for func in functions:
            if func.docstring:
                documented_elements += 1
        for class_info in classes:
            if class_info.docstring:
                documented_elements += 1
        
        return int((documented_elements / total_elements) * 100)
    
    def _calculate_overall_quality(self, metrics: Dict[str, Any]) -> int:
        """计算总体质量评分"""
        complexity_score = metrics.get('complexity_score', 0)
        documentation_score = metrics.get('documentation_score', 0)
        
        return int((complexity_score + documentation_score) / 2)


class DocumentTemplate:
    """文档模板"""
    
    def __init__(self):
        self.templates = {
            'markdown': self._markdown_template,
            'html': self._html_template,
            'rst': self._rst_template,
            'api': self._api_template
        }
    
    def generate_documentation(self, project_data: Dict[str, Any], template_type: str = 'markdown') -> str:
        """生成文档"""
        if template_type not in self.templates:
            template_type = 'markdown'
        
        template = self.templates[template_type]
        return template(project_data)
    
    def _markdown_template(self, project_data: Dict[str, Any]) -> str:
        """Markdown模板"""
        doc = "# 项目文档\n\n"
        doc += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 项目概览
        doc += "## 项目概览\n\n"
        total_files = len(project_data)
        total_functions = sum(len(data.get('functions', [])) for data in project_data.values())
        total_classes = sum(len(data.get('classes', [])) for data in project_data.values())
        
        doc += f"- 文件数量: {total_files}\n"
        doc += f"- 函数数量: {total_functions}\n"
        doc += f"- 类数量: {total_classes}\n\n"
        
        # 文件详情
        for file_path, data in project_data.items():
            file_name = Path(file_path).name
            doc += f"## {file_name}\n\n"
            
            # 类文档
            if data.get('classes'):
                doc += "### 类\n\n"
                for class_info in data['classes']:
                    doc += f"#### {class_info.name}\n\n"
                    if class_info.docstring:
                        doc += f"{class_info.docstring}\n\n"
                    else:
                        doc += f"未文档化的类 {class_info.name}\n\n"
                    
                    if class_info.methods:
                        doc += "**方法:**\n\n"
                        for method in class_info.methods:
                            doc += f"- `{method.name}`: {method.docstring or '未文档化'}\n"
                        doc += "\n"
            
            # 函数文档
            if data.get('functions'):
                doc += "### 函数\n\n"
                for func_info in data['functions']:
                    doc += f"#### {func_info.name}\n\n"
                    if func_info.docstring:
                        doc += f"{func_info.docstring}\n\n"
                    else:
                        doc += f"未文档化的函数 {func_info.name}\n\n"
                    
                    if func_info.parameters:
                        doc += "**参数:**\n\n"
                        for param in func_info.parameters:
                            doc += f"- `{param['name']}` ({param['type']}): 参数说明\n"
                        doc += "\n"
                    
                    if func_info.return_type != 'None':
                        doc += f"**返回:** {func_info.return_type}\n\n"
        
        return doc
    
    def _html_template(self, project_data: Dict[str, Any]) -> str:
        """HTML模板"""
        doc = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>项目文档</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .function, .class { border-left: 4px solid #007acc; padding-left: 20px; margin: 10px 0; }
        .code { background-color: #f5f5f5; padding: 10px; border-radius: 3px; font-family: monospace; }
    </style>
</head>
<body>
    <div class="header">
        <h1>项目文档</h1>
        <p>生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
"""
        
        for file_path, data in project_data.items():
            file_name = Path(file_path).name
            doc += f'    <div class="section">\n'
            doc += f'        <h2>{file_name}</h2>\n'
            
            # 类文档
            if data.get('classes'):
                doc += '        <h3>类</h3>\n'
                for class_info in data['classes']:
                    doc += f'        <div class="class">\n'
                    doc += f'            <h4>{class_info.name}</h4>\n'
                    if class_info.docstring:
                        doc += f'            <p>{class_info.docstring}</p>\n'
                    else:
                        doc += f'            <p>未文档化的类 {class_info.name}</p>\n'
                    doc += '        </div>\n'
            
            # 函数文档
            if data.get('functions'):
                doc += '        <h3>函数</h3>\n'
                for func_info in data['functions']:
                    doc += f'        <div class="function">\n'
                    doc += f'            <h4>{func_info.name}</h4>\n'
                    if func_info.docstring:
                        doc += f'            <p>{func_info.docstring}</p>\n'
                    else:
                        doc += f'            <p>未文档化的函数 {func_info.name}</p>\n'
                    doc += '        </div>\n'
            
            doc += '    </div>\n'
        
        doc += """</body>
</html>"""
        
        return doc
    
    def _rst_template(self, project_data: Dict[str, Any]) -> str:
        """RST模板"""
        doc = "项目文档\n"
        doc += "=" * 20 + "\n\n"
        doc += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for file_path, data in project_data.items():
            file_name = Path(file_path).name
            doc += f"{file_name}\n"
            doc += "-" * len(file_name) + "\n\n"
            
            if data.get('classes'):
                doc += "类\n"
                doc += "~" * 10 + "\n\n"
                for class_info in data['classes']:
                    doc += f"{class_info.name}\n"
                    doc += "^" * len(class_info.name) + "\n\n"
                    if class_info.docstring:
                        doc += f"{class_info.docstring}\n\n"
                    else:
                        doc += f"未文档化的类 {class_info.name}\n\n"
            
            if data.get('functions'):
                doc += "函数\n"
                doc += "~" * 10 + "\n\n"
                for func_info in data['functions']:
                    doc += f"{func_info.name}\n"
                    doc += "^" * len(func_info.name) + "\n\n"
                    if func_info.docstring:
                        doc += f"{func_info.docstring}\n\n"
                    else:
                        doc += f"未文档化的函数 {func_info.name}\n\n"
        
        return doc
    
    def _api_template(self, project_data: Dict[str, Any]) -> str:
        """API文档模板"""
        doc = "# API 文档\n\n"
        doc += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for file_path, data in project_data.items():
            file_name = Path(file_path).name
            doc += f"## {file_name}\n\n"
            
            # API端点
            if data.get('functions'):
                for func_info in data['functions']:
                    doc += f"### `{func_info.name}`\n\n"
                    doc += f"**类型:** 函数\n"
                    doc += f"**行号:** {func_info.line_number}\n\n"
                    
                    if func_info.docstring:
                        doc += f"**描述:** {func_info.docstring}\n\n"
                    
                    if func_info.parameters:
                        doc += "**参数:**\n\n"
                        for param in func_info.parameters:
                            doc += f"- `{param['name']}` ({param['type']})\n"
                        doc += "\n"
                    
                    if func_info.return_type != 'None':
                        doc += f"**返回类型:** {func_info.return_type}\n\n"
                    
                    doc += "**示例:**\n\n"
                    doc += "```python\n"
                    doc += f"# 调用示例\nresult = {func_info.name}("
                    if func_info.parameters:
                        params_str = ", ".join([f'{p["name"]}=value' for p in func_info.parameters])
                        doc += params_str
                    doc += ")\n"
                    doc += "```\n\n"
            
            if data.get('classes'):
                for class_info in data['classes']:
                    doc += f"### `{class_info.name}`\n\n"
                    doc += f"**类型:** 类\n"
                    doc += f"**行号:** {class_info.line_number}\n\n"
                    
                    if class_info.docstring:
                        doc += f"**描述:** {class_info.docstring}\n\n"
                    
                    if class_info.base_classes:
                        doc += f"**继承:** {', '.join(class_info.base_classes)}\n\n"
                    
                    if class_info.methods:
                        doc += "**方法:**\n\n"
                        for method in class_info.methods:
                            doc += f"- `{method.name}`: {method.docstring or '未文档化'}\n"
                        doc += "\n"
        
        return doc


class CodeDocGenerator:
    """代码文档生成器主类"""
    
    def __init__(self):
        self.parser = PythonCodeParser()
        self.comment_generator = CodeCommentGenerator()
        self.dependency_analyzer = DependencyAnalyzer()
        self.flowchart_generator = FlowchartGenerator()
        self.quality_analyzer = QualityAnalyzer()
        self.document_template = DocumentTemplate()
        self.supported_languages = ['python']
    
    def generate_documentation(self, 
                             source_path: str, 
                             output_path: str = None,
                             template_type: str = 'markdown',
                             include_flowcharts: bool = True,
                             include_quality_analysis: bool = True) -> Dict[str, Any]:
        """生成完整文档"""
        print(f"开始分析代码: {source_path}")
        
        # 解析项目
        project_data = self._parse_project(source_path)
        
        if not project_data:
            print("未能解析任何代码文件")
            return {}
        
        # 生成文档
        documentation = {}
        
        # 生成基础文档
        documentation['main_doc'] = self.document_template.generate_documentation(
            project_data, template_type
        )
        
        # 生成API文档
        documentation['api_doc'] = self.document_template.generate_documentation(
            project_data, 'api'
        )
        
        # 生成流程图
        if include_flowcharts:
            documentation['flowcharts'] = self._generate_flowcharts(project_data)
        
        # 生成质量分析报告
        if include_quality_analysis:
            documentation['quality_report'] = self._generate_quality_report(project_data)
        
        # 生成依赖分析报告
        documentation['dependency_report'] = self._generate_dependency_report(project_data)
        
        # 保存文档
        if output_path:
            self._save_documentation(documentation, output_path, template_type)
        
        print("文档生成完成")
        return documentation
    
    def _parse_project(self, source_path: str) -> Dict[str, Any]:
        """解析整个项目"""
        project_data = {}
        
        if os.path.isfile(source_path):
            # 单个文件
            if source_path.endswith('.py'):
                parsed_data = self.parser.parse_file(source_path)
                if parsed_data:
                    project_data[source_path] = parsed_data
        elif os.path.isdir(source_path):
            # 目录
            for root, dirs, files in os.walk(source_path):
                # 跳过隐藏目录和特定目录
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        parsed_data = self.parser.parse_file(file_path)
                        if parsed_data:
                            project_data[file_path] = parsed_data
        
        return project_data
    
    def _generate_flowcharts(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成流程图"""
        flowcharts = {}
        
        for file_path, data in project_data.items():
            file_flowcharts = {}
            
            # 函数流程图
            for func_info in data.get('functions', []):
                flowchart = self.flowchart_generator.generate_function_flowchart(func_info)
                file_flowcharts[f"function_{func_info.name}"] = flowchart
            
            # 类图
            for class_info in data.get('classes', []):
                diagram = self.flowchart_generator.generate_class_diagram(class_info)
                file_flowcharts[f"class_{class_info.name}"] = diagram
            
            flowcharts[file_path] = file_flowcharts
        
        # 项目架构图
        architecture = self.flowchart_generator.generate_architecture_diagram(project_data)
        flowcharts['__project_architecture__'] = architecture
        
        return flowcharts
    
    def _generate_quality_report(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成质量分析报告"""
        quality_report = {}
        
        for file_path, data in project_data.items():
            metrics = self.quality_analyzer.analyze_file(file_path, data)
            quality_report[file_path] = metrics
        
        # 项目总体质量
        all_metrics = list(quality_report.values())
        if all_metrics:
            project_quality = {
                'average_complexity': sum(m.get('average_function_complexity', 0) for m in all_metrics) / len(all_metrics),
                'average_documentation_score': sum(m.get('documentation_score', 0) for m in all_metrics) / len(all_metrics),
                'total_files': len(all_metrics),
                'overall_quality': sum(m.get('overall_quality', 0) for m in all_metrics) / len(all_metrics)
            }
            quality_report['__project_summary__'] = project_quality
        
        return quality_report
    
    def _generate_dependency_report(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成依赖分析报告"""
        dependency_report = {}
        
        for file_path, data in project_data.items():
            deps = self.dependency_analyzer.analyze_file(file_path, data)
            dependency_report[file_path] = deps
        
        return dependency_report
    
    def _save_documentation(self, documentation: Dict[str, Any], output_path: str, template_type: str):
        """保存文档到文件"""
        os.makedirs(output_path, exist_ok=True)
        
        # 保存主要文档
        main_doc_path = os.path.join(output_path, f'documentation.{template_type}')
        with open(main_doc_path, 'w', encoding='utf-8') as f:
            f.write(documentation['main_doc'])
        
        # 保存API文档
        api_doc_path = os.path.join(output_path, 'api_documentation.md')
        with open(api_doc_path, 'w', encoding='utf-8') as f:
            f.write(documentation['api_doc'])
        
        # 保存流程图
        if 'flowcharts' in documentation:
            flowcharts_path = os.path.join(output_path, 'flowcharts.md')
            with open(flowcharts_path, 'w', encoding='utf-8') as f:
                f.write("# 代码流程图\n\n")
                for file_path, flowcharts in documentation['flowcharts'].items():
                    f.write(f"## {file_path}\n\n")
                    # 确保flowcharts是字典类型
                    if isinstance(flowcharts, dict):
                        for name, flowchart in flowcharts.items():
                            f.write(f"### {name}\n\n")
                            f.write(f"```{flowchart.split('\\n')[0]}\n")
                            for line in flowchart.split('\\n')[1:]:
                                f.write(f"{line}\n")
                            f.write("```\n\n")
                    else:
                        # 如果flowcharts是字符串，直接写入
                        f.write(flowcharts)
                        f.write("\n\n")
        
        # 保存质量报告
        if 'quality_report' in documentation:
            quality_path = os.path.join(output_path, 'quality_report.md')
            with open(quality_path, 'w', encoding='utf-8') as f:
                f.write("# 代码质量报告\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for file_path, metrics in documentation['quality_report'].items():
                    if file_path == '__project_summary__':
                        continue
                    
                    f.write(f"## {file_path}\n\n")
                    f.write(f"- 总行数: {metrics.get('total_lines', 0)}\n")
                    f.write(f"- 代码行数: {metrics.get('code_lines', 0)}\n")
                    f.write(f"- 函数数量: {metrics.get('function_count', 0)}\n")
                    f.write(f"- 类数量: {metrics.get('class_count', 0)}\n")
                    f.write(f"- 平均函数长度: {metrics.get('average_function_length', 0):.1f}\n")
                    f.write(f"- 平均复杂度: {metrics.get('average_function_complexity', 0):.1f}\n")
                    f.write(f"- 复杂度评分: {metrics.get('complexity_score', 0)}/100\n")
                    f.write(f"- 文档评分: {metrics.get('documentation_score', 0)}/100\n")
                    f.write(f"- 总体质量: {metrics.get('overall_quality', 0)}/100\n\n")
                
                # 项目总体质量
                if '__project_summary__' in documentation['quality_report']:
                    summary = documentation['quality_report']['__project_summary__']
                    f.write("## 项目总体质量\n\n")
                    f.write(f"- 平均复杂度: {summary.get('average_complexity', 0):.1f}\n")
                    f.write(f"- 平均文档评分: {summary.get('average_documentation_score', 0):.1f}/100\n")
                    f.write(f"- 文件数量: {summary.get('total_files', 0)}\n")
                    f.write(f"- 项目总体质量: {summary.get('overall_quality', 0):.1f}/100\n")
        
        # 保存依赖报告
        if 'dependency_report' in documentation:
            deps_path = os.path.join(output_path, 'dependency_report.md')
            with open(deps_path, 'w', encoding='utf-8') as f:
                f.write("# 依赖关系分析报告\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for file_path, deps in documentation['dependency_report'].items():
                    f.write(f"## {file_path}\n\n")
                    
                    external_deps = deps.get('external_dependencies', [])
                    internal_deps = deps.get('internal_dependencies', [])
                    
                    if external_deps:
                        f.write("### 外部依赖\n\n")
                        for dep in external_deps:
                            f.write(f"- {dep.get('module', 'Unknown')}\n")
                        f.write("\n")
                    
                    if internal_deps:
                        f.write("### 内部依赖\n\n")
                        for dep in internal_deps:
                            f.write(f"- {dep.get('module', 'Unknown')}\n")
                        f.write("\n")
        
        print(f"文档已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Q8代码文档生成器')
    parser.add_argument('source', help='源代码路径（文件或目录）')
    parser.add_argument('-o', '--output', help='输出目录', default='docs')
    parser.add_argument('-t', '--template', choices=['markdown', 'html', 'rst', 'api'], 
                       default='markdown', help='文档模板类型')
    parser.add_argument('--no-flowcharts', action='store_true', help='不生成流程图')
    parser.add_argument('--no-quality', action='store_true', help='不生成质量分析')
    
    args = parser.parse_args()
    
    # 创建文档生成器
    generator = CodeDocGenerator()
    
    # 生成文档
    documentation = generator.generate_documentation(
        source_path=args.source,
        output_path=args.output,
        template_type=args.template,
        include_flowcharts=not args.no_flowcharts,
        include_quality_analysis=not args.no_quality
    )
    
    if documentation:
        print("文档生成成功！")
        print(f"输出目录: {args.output}")
    else:
        print("文档生成失败！")


if __name__ == '__main__':
    main()