"""
Q1 API文档生成器核心实现

提供完整的API文档生成功能，包括代码解析、文档字符串解析、
API端点识别、参数文档化、示例代码生成、多格式输出等。
"""

import ast
import os
import re
import json
import inspect
import importlib
import importlib.util
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# 可选依赖
try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False


@dataclass
class FunctionInfo:
    """函数信息数据类"""
    name: str
    docstring: str
    parameters: List[Dict[str, Any]]
    returns: Optional[Dict[str, Any]]
    decorators: List[str]
    source_file: str
    line_number: int
    is_api_endpoint: bool = False
    http_method: Optional[str] = None
    endpoint_path: Optional[str] = None
    examples: List[str] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


@dataclass
class ClassInfo:
    """类信息数据类"""
    name: str
    docstring: str
    methods: List[FunctionInfo]
    attributes: List[Dict[str, Any]]
    inheritance: List[str]
    source_file: str
    line_number: int


@dataclass
class ModuleInfo:
    """模块信息数据类"""
    name: str
    docstring: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[str]
    source_file: str


class CodeParser:
    """代码解析器"""
    
    def __init__(self):
        self.parsed_modules = {}
    
    def parse_file(self, file_path: str) -> ModuleInfo:
        """解析Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            return self._parse_ast_tree(tree, module_name, file_path)
        
        except Exception as e:
            raise ValueError(f"解析文件 {file_path} 失败: {str(e)}")
    
    def parse_directory(self, directory_path: str) -> Dict[str, ModuleInfo]:
        """解析整个目录"""
        modules = {}
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    try:
                        module_info = self.parse_file(file_path)
                        modules[module_info.name] = module_info
                    except Exception as e:
                        print(f"警告: 无法解析文件 {file_path}: {str(e)}")
        
        return modules
    
    def _parse_ast_tree(self, tree: ast.AST, module_name: str, file_path: str) -> ModuleInfo:
        """解析AST树"""
        module_docstring = ast.get_docstring(tree) or ""
        
        functions = []
        classes = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._parse_function(node, file_path)
                functions.append(func_info)
            elif isinstance(node, ast.ClassDef):
                class_info = self._parse_class(node, file_path)
                classes.append(class_info)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_info = self._parse_import(node)
                imports.extend(import_info)
        
        return ModuleInfo(
            name=module_name,
            docstring=module_docstring,
            functions=functions,
            classes=classes,
            imports=imports,
            source_file=file_path
        )
    
    def _parse_function(self, node: ast.FunctionDef, file_path: str) -> FunctionInfo:
        """解析函数"""
        docstring = ast.get_docstring(node) or ""
        
        parameters = []
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': 'Any',
                'default': None,
                'description': ''
            }
            parameters.append(param_info)
        
        # 处理返回值
        returns = None
        if node.returns:
            returns = {
                'type': ast.unparse(node.returns) if hasattr(ast, 'unparse') else 'Any',
                'description': ''
            }
        
        decorators = [ast.unparse(dec) if hasattr(ast, 'unparse') else str(dec) for dec in node.decorator_list]
        
        return FunctionInfo(
            name=node.name,
            docstring=docstring,
            parameters=parameters,
            returns=returns,
            decorators=decorators,
            source_file=file_path,
            line_number=node.lineno
        )
    
    def _parse_class(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """解析类"""
        docstring = ast.get_docstring(node) or ""
        
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._parse_function(item, file_path)
                methods.append(method_info)
        
        # 处理继承
        inheritance = [ast.unparse(base) if hasattr(ast, 'unparse') else str(base) for base in node.bases]
        
        return ClassInfo(
            name=node.name,
            docstring=docstring,
            methods=methods,
            attributes=[],
            inheritance=inheritance,
            source_file=file_path,
            line_number=node.lineno
        )
    
    def _parse_import(self, node: Union[ast.Import, ast.ImportFrom]) -> List[str]:
        """解析导入语句"""
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            return [f"{module}.{name}" for name in names]
        return []


class DocstringParser:
    """文档字符串解析器"""
    
    def __init__(self):
        self.param_pattern = re.compile(r'@param\s+(\w+):\s*(.+)')
        self.return_pattern = re.compile(r'@return:\s*(.+)')
        self.example_pattern = re.compile(r'@example\s*\n(.*?)(?=\n@|\n\n|\Z)', re.DOTALL)
    
    def parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """解析文档字符串"""
        if not docstring:
            return {}
        
        # 分离描述和参数文档
        lines = docstring.split('\n')
        description_lines = []
        param_docs = {}
        return_docs = ""
        examples = []
        
        current_section = "description"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查参数文档
            param_match = self.param_pattern.match(line)
            if param_match:
                param_name, param_desc = param_match.groups()
                param_docs[param_name] = param_desc
                current_section = "params"
                continue
            
            # 检查返回值文档
            return_match = self.return_pattern.match(line)
            if return_match:
                return_docs = return_match.group(1)
                current_section = "return"
                continue
            
            # 检查示例
            if line.startswith('@example'):
                current_section = "example"
                continue
            
            # 收集内容
            if current_section == "description":
                description_lines.append(line)
            elif current_section == "example":
                examples.append(line)
        
        return {
            'description': '\n'.join(description_lines),
            'parameters': param_docs,
            'returns': return_docs,
            'examples': examples
        }
    
    def extract_parameters(self, docstring: str) -> Dict[str, str]:
        """提取参数文档"""
        parsed = self.parse_docstring(docstring)
        return parsed.get('parameters', {})
    
    def extract_examples(self, docstring: str) -> List[str]:
        """提取示例代码"""
        parsed = self.parse_docstring(docstring)
        return parsed.get('examples', [])


class EndpointDetector:
    """API端点检测器"""
    
    def __init__(self):
        self.decorator_patterns = {
            'flask': re.compile(r'@app\.route\(["\']([^"\']+)["\']'),
            'fastapi': re.compile(r'@(get|post|put|delete|patch)\(["\']([^"\']+)["\']'),
            'django': re.compile(r'@urlpatterns\.path\(["\']([^"\']+)["\']'),
            'generic': re.compile(r'@(route|endpoint)\(["\']([^"\']+)["\']')
        }
    
    def detect_endpoints(self, function_info: FunctionInfo) -> Dict[str, Any]:
        """检测API端点"""
        endpoint_info = {
            'is_endpoint': False,
            'http_methods': [],
            'paths': [],
            'decorators': []
        }
        
        # 检查装饰器
        for decorator in function_info.decorators:
            decorator_str = str(decorator)
            endpoint_info['decorators'].append(decorator_str)
            
            # Flask风格
            flask_match = self.decorator_patterns['flask'].search(decorator_str)
            if flask_match:
                endpoint_info['is_endpoint'] = True
                endpoint_info['paths'].append(flask_match.group(1))
                endpoint_info['http_methods'].append('GET')  # Flask默认GET
            
            # FastAPI风格
            fastapi_match = self.decorator_patterns['fastapi'].search(decorator_str)
            if fastapi_match:
                endpoint_info['is_endpoint'] = True
                endpoint_info['http_methods'].append(fastapi_match.group(1).upper())
                endpoint_info['paths'].append(fastapi_match.group(2))
            
            # Django风格
            django_match = self.decorator_patterns['django'].search(decorator_str)
            if django_match:
                endpoint_info['is_endpoint'] = True
                endpoint_info['paths'].append(django_match.group(1))
                endpoint_info['http_methods'].append('GET')  # Django默认GET
        
        # 检查函数名模式
        if not endpoint_info['is_endpoint']:
            if any(keyword in function_info.name.lower() for keyword in ['api', 'endpoint', 'rest']):
                endpoint_info['is_endpoint'] = True
                endpoint_info['paths'].append(f"/{function_info.name}")
                endpoint_info['http_methods'].append('GET')
        
        return endpoint_info


class ParameterDocumenter:
    """参数文档化器"""
    
    def __init__(self, docstring_parser: DocstringParser):
        self.docstring_parser = docstring_parser
    
    def document_parameters(self, function_info: FunctionInfo) -> List[Dict[str, Any]]:
        """文档化函数参数"""
        documented_params = []
        param_docs = self.docstring_parser.extract_parameters(function_info.docstring)
        
        for param in function_info.parameters:
            param_name = param['name']
            documented_param = param.copy()
            
            # 添加文档字符串中的描述
            if param_name in param_docs:
                documented_param['description'] = param_docs[param_name]
            
            documented_params.append(documented_param)
        
        return documented_params
    
    def document_return_value(self, function_info: FunctionInfo) -> Optional[Dict[str, Any]]:
        """文档化返回值"""
        if not function_info.returns:
            return None
        
        documented_return = function_info.returns.copy()
        
        # 添加文档字符串中的返回值描述
        return_docs = self.docstring_parser.parse_docstring(function_info.docstring).get('returns', '')
        if return_docs:
            documented_return['description'] = return_docs
        
        return documented_return


class ExampleGenerator:
    """示例代码生成器"""
    
    def __init__(self, docstring_parser: DocstringParser):
        self.docstring_parser = docstring_parser
    
    def generate_examples(self, function_info: FunctionInfo) -> List[str]:
        """生成使用示例"""
        examples = []
        
        # 从文档字符串提取示例
        docstring_examples = self.docstring_parser.extract_examples(function_info.docstring)
        examples.extend(docstring_examples)
        
        # 自动生成示例
        auto_example = self._generate_auto_example(function_info)
        if auto_example:
            examples.append(auto_example)
        
        return examples
    
    def _generate_auto_example(self, function_info: FunctionInfo) -> str:
        """自动生成示例代码"""
        if not function_info.parameters:
            return f"# {function_info.name}() 调用示例\n{function_info.name}()"
        
        # 生成参数示例
        param_examples = []
        for param in function_info.parameters:
            param_name = param['name']
            param_type = param.get('type', 'Any')
            
            # 根据类型生成示例值
            if param_type.lower() in ['int', 'integer']:
                example_value = "1"
            elif param_type.lower() in ['float', 'double']:
                example_value = "1.0"
            elif param_type.lower() in ['str', 'string']:
                example_value = f'"{param_name}_value"'
            elif param_type.lower() in ['bool', 'boolean']:
                example_value = "True"
            elif param_type.lower() in ['list', 'array']:
                example_value = "[]"
            elif param_type.lower() in ['dict', 'object']:
                example_value = "{}"
            else:
                example_value = "None"
            
            param_examples.append(f"{param_name}={example_value}")
        
        params_str = ", ".join(param_examples)
        return f"# {function_info.name}() 调用示例\n{function_info.name}({params_str})"


class OutputFormatter:
    """输出格式化器"""
    
    def __init__(self):
        self.formatters = {
            'markdown': self._format_markdown,
            'html': self._format_html,
            'json': self._format_json,
            'rst': self._format_rst
        }
    
    def format(self, module_info: ModuleInfo, format_type: str = 'markdown') -> str:
        """格式化输出"""
        if format_type not in self.formatters:
            raise ValueError(f"不支持的输出格式: {format_type}")
        
        return self.formatters[format_type](module_info)
    
    def _format_markdown(self, module_info: ModuleInfo) -> str:
        """Markdown格式"""
        md_content = []
        
        # 模块标题
        md_content.append(f"# {module_info.name}")
        md_content.append("")
        
        # 模块描述
        if module_info.docstring:
            md_content.append(module_info.docstring)
            md_content.append("")
        
        # 函数文档
        if module_info.functions:
            md_content.append("## 函数")
            md_content.append("")
            
            for func in module_info.functions:
                md_content.append(f"### {func.name}")
                md_content.append("")
                
                if func.docstring:
                    md_content.append(func.docstring)
                    md_content.append("")
                
                # 参数
                if func.parameters:
                    md_content.append("**参数:**")
                    md_content.append("")
                    for param in func.parameters:
                        param_desc = param.get('description', '')
                        param_type = param.get('type', 'Any')
                        param_default = param.get('default')
                        default_str = f" (默认: {param_default})" if param_default else ""
                        md_content.append(f"- `{param['name']}` ({param_type}){default_str}: {param_desc}")
                    md_content.append("")
                
                # 返回值
                if func.returns:
                    return_desc = func.returns.get('description', '')
                    return_type = func.returns.get('type', 'Any')
                    md_content.append(f"**返回值:** {return_type} - {return_desc}")
                    md_content.append("")
                
                # 示例
                if func.examples:
                    md_content.append("**示例:**")
                    md_content.append("")
                    for example in func.examples:
                        md_content.append("```python")
                        md_content.append(example)
                        md_content.append("```")
                        md_content.append("")
        
        # 类文档
        if module_info.classes:
            md_content.append("## 类")
            md_content.append("")
            
            for cls in module_info.classes:
                md_content.append(f"### {cls.name}")
                md_content.append("")
                
                if cls.docstring:
                    md_content.append(cls.docstring)
                    md_content.append("")
                
                # 方法
                if cls.methods:
                    md_content.append("**方法:**")
                    md_content.append("")
                    for method in cls.methods:
                        md_content.append(f"- `{method.name}()`: {method.docstring[:100]}...")
                    md_content.append("")
        
        return '\n'.join(md_content)
    
    def _format_html(self, module_info: ModuleInfo) -> str:
        """HTML格式"""
        html_content = []
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html>")
        html_content.append("<head>")
        html_content.append(f"<title>{module_info.name} - API文档</title>")
        html_content.append("<style>")
        html_content.append("""
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { border-bottom: 2px solid #333; padding-bottom: 10px; }
        .section { margin: 20px 0; }
        .function { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .parameter { margin: 5px 0; }
        .example { background: #e8f4f8; padding: 10px; border-left: 4px solid #2196F3; }
        code { background: #f0f0f0; padding: 2px 4px; border-radius: 3px; }
        pre { background: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto; }
        """)
        html_content.append("</style>")
        html_content.append("</head>")
        html_content.append("<body>")
        
        # 标题
        html_content.append(f"<div class='header'><h1>{module_info.name}</h1></div>")
        
        # 模块描述
        if module_info.docstring:
            html_content.append(f"<div class='section'><p>{module_info.docstring}</p></div>")
        
        # 函数文档
        if module_info.functions:
            html_content.append("<div class='section'><h2>函数</h2>")
            
            for func in module_info.functions:
                html_content.append(f"<div class='function'>")
                html_content.append(f"<h3>{func.name}</h3>")
                
                if func.docstring:
                    html_content.append(f"<p>{func.docstring}</p>")
                
                # 参数
                if func.parameters:
                    html_content.append("<h4>参数:</h4>")
                    for param in func.parameters:
                        param_desc = param.get('description', '')
                        param_type = param.get('type', 'Any')
                        param_default = param.get('default')
                        default_str = f" (默认: {param_default})" if param_default else ""
                        html_content.append(f"<div class='parameter'><code>{param['name']}</code> ({param_type}){default_str}: {param_desc}</div>")
                
                # 返回值
                if func.returns:
                    return_desc = func.returns.get('description', '')
                    return_type = func.returns.get('type', 'Any')
                    html_content.append(f"<h4>返回值:</h4> <p>{return_type} - {return_desc}</p>")
                
                # 示例
                if func.examples:
                    html_content.append("<h4>示例:</h4>")
                    for example in func.examples:
                        html_content.append(f"<div class='example'><pre><code>{example}</code></pre></div>")
                
                html_content.append("</div>")
            
            html_content.append("</div>")
        
        html_content.append("</body>")
        html_content.append("</html>")
        
        return '\n'.join(html_content)
    
    def _format_json(self, module_info: ModuleInfo) -> str:
        """JSON格式"""
        return json.dumps(asdict(module_info), ensure_ascii=False, indent=2)
    
    def _format_rst(self, module_info: ModuleInfo) -> str:
        """reStructuredText格式"""
        rst_content = []
        
        # 模块标题
        rst_content.append(module_info.name)
        rst_content.append("=" * len(module_info.name))
        rst_content.append("")
        
        # 模块描述
        if module_info.docstring:
            rst_content.append(module_info.docstring)
            rst_content.append("")
        
        # 函数文档
        if module_info.functions:
            rst_content.append("函数")
            rst_content.append("-" * 4)
            rst_content.append("")
            
            for func in module_info.functions:
                rst_content.append(func.name)
                rst_content.append("^" * len(func.name))
                rst_content.append("")
                
                if func.docstring:
                    rst_content.append(func.docstring)
                    rst_content.append("")
                
                # 参数
                if func.parameters:
                    rst_content.append("参数")
                    rst_content.append("~" * 4)
                    rst_content.append("")
                    for param in func.parameters:
                        param_desc = param.get('description', '')
                        param_type = param.get('type', 'Any')
                        param_default = param.get('default')
                        default_str = f" (默认: {param_default})" if param_default else ""
                        rst_content.append(f"{param['name']} ({param_type}){default_str}")
                        rst_content.append(f"    {param_desc}")
                        rst_content.append("")
        
        return '\n'.join(rst_content)


class InteractiveDocumentation:
    """交互式文档生成器"""
    
    def __init__(self):
        self.search_index = {}
    
    def generate_interactive_docs(self, modules: Dict[str, ModuleInfo], output_dir: str):
        """生成交互式文档"""
        # 生成搜索索引
        self._build_search_index(modules)
        
        # 生成HTML文件
        self._generate_html_docs(modules, output_dir)
        
        # 生成搜索脚本
        self._generate_search_script(output_dir)
    
    def _build_search_index(self, modules: Dict[str, ModuleInfo]):
        """构建搜索索引"""
        for module_name, module_info in modules.items():
            # 添加模块到索引
            self.search_index[f"module:{module_name}"] = {
                'type': 'module',
                'name': module_name,
                'description': module_info.docstring[:200] + '...' if len(module_info.docstring) > 200 else module_info.docstring,
                'url': f"{module_name}.html"
            }
            
            # 添加函数到索引
            for func in module_info.functions:
                key = f"function:{module_name}.{func.name}"
                self.search_index[key] = {
                    'type': 'function',
                    'name': func.name,
                    'module': module_name,
                    'description': func.docstring[:200] + '...' if len(func.docstring) > 200 else func.docstring,
                    'url': f"{module_name}.html#{func.name}"
                }
            
            # 添加类到索引
            for cls in module_info.classes:
                key = f"class:{module_name}.{cls.name}"
                self.search_index[key] = {
                    'type': 'class',
                    'name': cls.name,
                    'module': module_name,
                    'description': cls.docstring[:200] + '...' if len(cls.docstring) > 200 else cls.docstring,
                    'url': f"{module_name}.html#{cls.name}"
                }
    
    def _generate_html_docs(self, modules: Dict[str, ModuleInfo], output_dir: str):
        """生成HTML文档"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成主页面
        self._generate_index_page(modules, output_dir)
        
        # 生成各模块页面
        formatter = OutputFormatter()
        for module_name, module_info in modules.items():
            html_content = formatter.format(module_info, 'html')
            
            # 添加搜索功能
            html_content = self._add_search_functionality(html_content, module_name)
            
            with open(os.path.join(output_dir, f"{module_name}.html"), 'w', encoding='utf-8') as f:
                f.write(html_content)
    
    def _generate_index_page(self, modules: Dict[str, ModuleInfo], output_dir: str):
        """生成索引页面"""
        index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>API文档索引</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .search-container { margin: 20px 0; }
        .search-box { width: 100%; padding: 10px; font-size: 16px; }
        .module-list { margin: 20px 0; }
        .module-item { padding: 10px; margin: 5px 0; background: #f5f5f5; border-radius: 5px; }
        .search-results { margin: 20px 0; }
        .result-item { padding: 10px; margin: 5px 0; background: #e8f4f8; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>API文档索引</h1>
    
    <div class="search-container">
        <input type="text" class="search-box" id="searchBox" placeholder="搜索API...">
        <div class="search-results" id="searchResults"></div>
    </div>
    
    <div class="module-list">
        <h2>模块列表</h2>
"""
        
        for module_name in modules.keys():
            index_html += f'        <div class="module-item"><a href="{module_name}.html">{module_name}</a></div>\n'
        
        index_html += """
    </div>
    
    <script>
        const searchIndex = """ + json.dumps(self.search_index, ensure_ascii=False) + """;
        
        document.getElementById('searchBox').addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            const resultsDiv = document.getElementById('searchResults');
            
            if (query.length < 2) {
                resultsDiv.innerHTML = '';
                return;
            }
            
            const results = [];
            for (const [key, item] of Object.entries(searchIndex)) {
                if (item.name.toLowerCase().includes(query) || 
                    item.description.toLowerCase().includes(query)) {
                    results.push(item);
                }
            }
            
            resultsDiv.innerHTML = results.map(result => 
                `<div class="result-item" onclick="window.location.href='${result.url}'">
                    <strong>${result.name}</strong> (${result.type})<br>
                    ${result.description}
                </div>`
            ).join('');
        });
    </script>
</body>
</html>
"""
        
        with open(os.path.join(output_dir, "index.html"), 'w', encoding='utf-8') as f:
            f.write(index_html)
    
    def _add_search_functionality(self, html_content: str, module_name: str) -> str:
        """添加搜索功能到HTML"""
        search_script = f"""
    <script>
        function searchInPage() {{
            const query = document.getElementById('pageSearch').value.toLowerCase();
            const content = document.querySelector('.content');
            const items = content.querySelectorAll('.function, .class');
            
            items.forEach(item => {{
                const text = item.textContent.toLowerCase();
                if (text.includes(query)) {{
                    item.style.display = 'block';
                    if (query.length > 0) {{
                        item.style.background = '#fff3cd';
                    }}
                }} else {{
                    item.style.display = 'none';
                }}
            }});
        }}
    </script>
    
    <div class="search-container">
        <input type="text" id="pageSearch" placeholder="在当前页面搜索..." onkeyup="searchInPage()">
    </div>
"""
        
        # 在body标签后添加搜索框
        html_content = html_content.replace('<body>', '<body>\n' + search_script)
        
        # 为内容区域添加class
        html_content = html_content.replace('<div class=\'section\'>', '<div class="content">\n<div class=\'section\'>')
        
        return html_content
    
    def _generate_search_script(self, output_dir: str):
        """生成搜索脚本文件"""
        search_script = """
// 交互式文档搜索功能
class DocSearch {
    constructor() {
        this.index = {};
        this.initializeIndex();
    }
    
    initializeIndex() {
        // 初始化搜索索引
        this.buildIndex();
    }
    
    buildIndex() {
        // 构建搜索索引
        const functions = document.querySelectorAll('.function');
        const classes = document.querySelectorAll('.class');
        
        [...functions, ...classes].forEach(item => {
            const name = item.querySelector('h3, h4').textContent;
            const text = item.textContent.toLowerCase();
            this.index[name] = {
                element: item,
                text: text
            };
        });
    }
    
    search(query) {
        const results = [];
        const lowerQuery = query.toLowerCase();
        
        for (const [name, data] of Object.entries(this.index)) {
            if (name.toLowerCase().includes(lowerQuery) || 
                data.text.includes(lowerQuery)) {
                results.push(data.element);
            }
        }
        
        return results;
    }
    
    highlightResults(results) {
        // 高亮搜索结果
        results.forEach(result => {
            result.style.backgroundColor = '#fff3cd';
        });
    }
    
    clearHighlights() {
        // 清除高亮
        const highlighted = document.querySelectorAll('[style*="background-color: #fff3cd"]');
        highlighted.forEach(item => {
            item.style.backgroundColor = '';
        });
    }
}

// 初始化搜索
document.addEventListener('DOMContentLoaded', () => {
    window.docSearch = new DocSearch();
});
"""
        
        with open(os.path.join(output_dir, "search.js"), 'w', encoding='utf-8') as f:
            f.write(search_script)


class VersionManager:
    """版本管理器"""
    
    def __init__(self):
        self.versions = {}
        self.current_version = None
    
    def add_version(self, version: str, modules: Dict[str, ModuleInfo]):
        """添加版本"""
        self.versions[version] = {
            'modules': modules,
            'timestamp': datetime.now().isoformat(),
            'description': f"版本 {version}"
        }
        
        if self.current_version is None:
            self.current_version = version
    
    def set_current_version(self, version: str):
        """设置当前版本"""
        if version not in self.versions:
            raise ValueError(f"版本 {version} 不存在")
        self.current_version = version
    
    def get_version_info(self, version: str = None) -> Dict[str, Any]:
        """获取版本信息"""
        if version is None:
            version = self.current_version
        
        if version not in self.versions:
            raise ValueError(f"版本 {version} 不存在")
        
        return self.versions[version]
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """比较两个版本"""
        if version1 not in self.versions or version2 not in self.versions:
            raise ValueError("版本不存在")
        
        v1_info = self.versions[version1]
        v2_info = self.versions[version2]
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'changes': {
                'added': [],
                'removed': [],
                'modified': []
            }
        }
        
        # 比较模块
        v1_modules = set(v1_info['modules'].keys())
        v2_modules = set(v2_info['modules'].keys())
        
        comparison['changes']['added'] = list(v2_modules - v1_modules)
        comparison['changes']['removed'] = list(v1_modules - v2_modules)
        
        # 查找修改的模块
        common_modules = v1_modules & v2_modules
        for module_name in common_modules:
            # 这里可以添加更详细的比较逻辑
            comparison['changes']['modified'].append(module_name)
        
        return comparison
    
    def export_version_info(self, version: str = None) -> str:
        """导出版本信息"""
        if version is None:
            version = self.current_version
        
        info = self.get_version_info(version)
        return json.dumps(info, ensure_ascii=False, indent=2)


class APIDocGenerator:
    """主要的API文档生成器"""
    
    def __init__(self):
        self.code_parser = CodeParser()
        self.docstring_parser = DocstringParser()
        self.endpoint_detector = EndpointDetector()
        self.parameter_documenter = ParameterDocumenter(self.docstring_parser)
        self.example_generator = ExampleGenerator(self.docstring_parser)
        self.output_formatter = OutputFormatter()
        self.interactive_docs = InteractiveDocumentation()
        self.version_manager = VersionManager()
    
    def generate_documentation(self, 
                             source_path: str, 
                             output_dir: str, 
                             format_type: str = 'markdown',
                             include_interactive: bool = True,
                             version: str = None) -> str:
        """生成API文档
        
        Args:
            source_path: 源代码路径（文件或目录）
            output_dir: 输出目录
            format_type: 输出格式 (markdown, html, json, rst)
            include_interactive: 是否包含交互式文档
            version: 版本号
        
        Returns:
            生成的文档内容或路径
        """
        # 解析源代码
        if os.path.isfile(source_path):
            modules = {os.path.splitext(os.path.basename(source_path))[0]: 
                      self.code_parser.parse_file(source_path)}
        else:
            modules = self.code_parser.parse_directory(source_path)
        
        # 增强解析结果
        enhanced_modules = self._enhance_modules(modules)
        
        # 添加到版本管理
        if version:
            self.version_manager.add_version(version, enhanced_modules)
        
        # 生成文档
        os.makedirs(output_dir, exist_ok=True)
        
        if format_type == 'interactive' and include_interactive:
            # 生成交互式文档
            self.interactive_docs.generate_interactive_docs(enhanced_modules, output_dir)
            return os.path.join(output_dir, "index.html")
        else:
            # 生成单个或多个文档文件
            generated_files = []
            for module_name, module_info in enhanced_modules.items():
                content = self.output_formatter.format(module_info, format_type)
                
                file_extension = {
                    'markdown': '.md',
                    'html': '.html',
                    'json': '.json',
                    'rst': '.rst'
                }.get(format_type, '.txt')
                
                output_file = os.path.join(output_dir, f"{module_name}{file_extension}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files.append(output_file)
            
            return generated_files if len(generated_files) > 1 else generated_files[0]
    
    def _enhance_modules(self, modules: Dict[str, ModuleInfo]) -> Dict[str, ModuleInfo]:
        """增强模块信息"""
        enhanced_modules = {}
        
        for module_name, module_info in modules.items():
            enhanced_module = module_info
            
            # 增强函数信息
            enhanced_functions = []
            for func in module_info.functions:
                enhanced_func = self._enhance_function(func)
                enhanced_functions.append(enhanced_func)
            
            enhanced_module.functions = enhanced_functions
            
            # 增强类信息
            enhanced_classes = []
            for cls in module_info.classes:
                enhanced_cls = self._enhance_class(cls)
                enhanced_classes.append(enhanced_cls)
            
            enhanced_module.classes = enhanced_classes
            
            enhanced_modules[module_name] = enhanced_module
        
        return enhanced_modules
    
    def _enhance_function(self, function_info: FunctionInfo) -> FunctionInfo:
        """增强函数信息"""
        # 检测API端点
        endpoint_info = self.endpoint_detector.detect_endpoints(function_info)
        
        # 文档化参数
        documented_parameters = self.parameter_documenter.document_parameters(function_info)
        
        # 文档化返回值
        documented_return = self.parameter_documenter.document_return_value(function_info)
        
        # 生成示例
        examples = self.example_generator.generate_examples(function_info)
        
        # 更新函数信息
        enhanced_func = FunctionInfo(
            name=function_info.name,
            docstring=function_info.docstring,
            parameters=documented_parameters,
            returns=documented_return,
            decorators=function_info.decorators,
            source_file=function_info.source_file,
            line_number=function_info.line_number,
            is_api_endpoint=endpoint_info['is_endpoint'],
            http_method=endpoint_info['http_methods'][0] if endpoint_info['http_methods'] else None,
            endpoint_path=endpoint_info['paths'][0] if endpoint_info['paths'] else None,
            examples=examples
        )
        
        return enhanced_func
    
    def _enhance_class(self, class_info: ClassInfo) -> ClassInfo:
        """增强类信息"""
        enhanced_methods = []
        for method in class_info.methods:
            enhanced_method = self._enhance_function(method)
            enhanced_methods.append(enhanced_method)
        
        enhanced_class = ClassInfo(
            name=class_info.name,
            docstring=class_info.docstring,
            methods=enhanced_methods,
            attributes=class_info.attributes,
            inheritance=class_info.inheritance,
            source_file=class_info.source_file,
            line_number=class_info.line_number
        )
        
        return enhanced_class
    
    def generate_pdf_documentation(self, source_path: str, output_file: str, version: str = None):
        """生成PDF文档"""
        # 先生成HTML
        temp_dir = os.path.join(os.path.dirname(output_file), "temp_html")
        html_file = self.generate_documentation(
            source_path, 
            temp_dir, 
            format_type='html',
            include_interactive=False,
            version=version
        )
        
        # 这里可以集成PDF生成库，如weasyprint或reportlab
        # 由于依赖问题，这里提供接口，实际使用时需要安装相应库
        print(f"HTML文件已生成: {html_file}")
        print("注意: PDF生成需要安装额外的依赖库，如weasyprint")
        
        # 清理临时文件
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def get_api_summary(self, source_path: str) -> Dict[str, Any]:
        """获取API摘要信息"""
        if os.path.isfile(source_path):
            modules = {os.path.splitext(os.path.basename(source_path))[0]: 
                      self.code_parser.parse_file(source_path)}
        else:
            modules = self.code_parser.parse_directory(source_path)
        
        enhanced_modules = self._enhance_modules(modules)
        
        summary = {
            'total_modules': len(enhanced_modules),
            'total_functions': sum(len(m.functions) for m in enhanced_modules.values()),
            'total_classes': sum(len(m.classes) for m in enhanced_modules.values()),
            'api_endpoints': [],
            'modules': []
        }
        
        for module_name, module_info in enhanced_modules.items():
            module_summary = {
                'name': module_name,
                'functions': len(module_info.functions),
                'classes': len(module_info.classes),
                'endpoints': []
            }
            
            for func in module_info.functions:
                if func.is_api_endpoint:
                    endpoint = {
                        'name': func.name,
                        'path': func.endpoint_path,
                        'method': func.http_method,
                        'module': module_name
                    }
                    summary['api_endpoints'].append(endpoint)
                    module_summary['endpoints'].append(endpoint)
            
            summary['modules'].append(module_summary)
        
        return summary


# 使用示例
if __name__ == "__main__":
    # 创建文档生成器实例
    generator = APIDocGenerator()
    
    # 示例用法
    print("Q1 API文档生成器")
    print("================")
    print("支持的功能:")
    print("- 代码解析和文档字符串解析")
    print("- API端点识别")
    print("- 参数和返回值文档化")
    print("- 示例代码生成")
    print("- 多格式输出 (Markdown, HTML, JSON, RST)")
    print("- 交互式文档")
    print("- 版本管理")
    print()
    print("使用示例:")
    print("generator = APIDocGenerator()")
    print("generator.generate_documentation('path/to/source', 'output/docs', 'html')")