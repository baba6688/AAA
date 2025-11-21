"""
Q3技术文档生成器

一个功能完整的技术文档生成器，支持多种文档类型的自动生成。
"""

import os
import json
import datetime
import ast
import inspect
import importlib.util
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import re


@dataclass
class ArchitectureComponent:
    """架构组件数据类"""
    name: str
    type: str
    description: str
    dependencies: List[str]
    technologies: List[str]


@dataclass
class APIEndpoint:
    """API端点数据类"""
    method: str
    path: str
    description: str
    parameters: Dict[str, Any]
    responses: Dict[str, Any]
    examples: Dict[str, Any]


@dataclass
class DatabaseTable:
    """数据库表数据类"""
    name: str
    description: str
    columns: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    foreign_keys: List[Dict[str, Any]]


class TechnicalDocGenerator:
    """技术文档生成器主类"""
    
    def __init__(self, project_root: str, output_dir: str = "docs"):
        """
        初始化技术文档生成器
        
        Args:
            project_root: 项目根目录路径
            output_dir: 输出文档目录
        """
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.subdirs = {
            'architecture': self.output_dir / 'architecture',
            'api': self.output_dir / 'api',
            'code': self.output_dir / 'code',
            'database': self.output_dir / 'database',
            'deployment': self.output_dir / 'deployment',
            'development': self.output_dir / 'development',
            'performance': self.output_dir / 'performance',
            'changelog': self.output_dir / 'changelog'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
    
    def generate_architecture_docs(self, components: List[ArchitectureComponent]) -> str:
        """
        生成架构文档
        
        Args:
            components: 架构组件列表
            
        Returns:
            生成的文档路径
        """
        doc_content = self._generate_architecture_markdown(components)
        output_path = self.subdirs['architecture'] / '系统架构文档.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        # 生成架构图（Mermaid格式）
        self._generate_architecture_diagram(components)
        
        return str(output_path)
    
    def _generate_architecture_markdown(self, components: List[ArchitectureComponent]) -> str:
        """生成架构文档的Markdown内容"""
        content = """# 系统架构文档

## 概述

本文档描述了系统的整体架构设计，包括各个组件之间的关系、技术栈选择和设计原则。

## 架构图

![系统架构图](架构图.svg)

## 组件详情

"""
        
        for comp in components:
            content += f"""### {comp.name}

**类型**: {comp.type}
**描述**: {comp.description}

**依赖组件**: {', '.join(comp.dependencies) if comp.dependencies else '无'}
**技术栈**: {', '.join(comp.technologies) if comp.technologies else '无'}

---

"""
        
        content += """## 设计原则

1. **模块化设计**: 系统采用模块化设计，各组件职责清晰
2. **可扩展性**: 支持水平扩展和垂直扩展
3. **高可用性**: 通过负载均衡和故障转移保证高可用
4. **安全性**: 多层安全防护机制
5. **性能优化**: 缓存、异步处理等优化策略

## 部署架构

系统采用微服务架构，支持容器化部署和云原生部署。

## 监控与运维

- 应用性能监控(APM)
- 日志聚合和分析
- 健康检查和告警
- 自动化部署和回滚
"""
        
        return content
    
    def _generate_architecture_diagram(self, components: List[ArchitectureComponent]):
        """生成架构图（Mermaid格式）"""
        diagram = "graph TB\n"
        
        # 添加组件节点
        for comp in components:
            comp_id = comp.name.replace(" ", "_").lower()
            diagram += f'    {comp_id}["{comp.name}<br/>{comp.type}"]\n'
        
        # 添加依赖关系
        for comp in components:
            comp_id = comp.name.replace(" ", "_").lower()
            for dep in comp.dependencies:
                dep_id = dep.replace(" ", "_").lower()
                diagram += f'    {comp_id} --> {dep_id}\n'
        
        # 保存Mermaid图表
        mermaid_path = self.subdirs['architecture'] / '架构图.mmd'
        with open(mermaid_path, 'w', encoding='utf-8') as f:
            f.write(diagram)
    
    def generate_code_docs(self, source_dirs: List[str]) -> str:
        """
        生成代码文档
        
        Args:
            source_dirs: 源代码目录列表
            
        Returns:
            生成的文档路径
        """
        code_docs = []
        
        for source_dir in source_dirs:
            docs = self._analyze_source_code(source_dir)
            code_docs.extend(docs)
        
        doc_content = self._generate_code_markdown(code_docs)
        output_path = self.subdirs['code'] / '代码文档.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        return str(output_path)
    
    def _analyze_source_code(self, source_dir: str) -> List[Dict[str, Any]]:
        """分析源代码文件"""
        docs = []
        source_path = self.project_root / source_dir
        
        if not source_path.exists():
            return docs
        
        for file_path in source_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                file_doc = self._extract_python_docs(tree, str(file_path))
                if file_doc:
                    docs.append(file_doc)
            except Exception as e:
                print(f"分析文件 {file_path} 时出错: {e}")
        
        return docs
    
    def _extract_python_docs(self, tree: ast.AST, file_path: str) -> Optional[Dict[str, Any]]:
        """提取Python文档信息"""
        module_doc = ast.get_docstring(tree)
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'methods': []
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            'name': item.name,
                            'docstring': ast.get_docstring(item),
                            'args': [arg.arg for arg in item.args.args],
                            'is_private': item.name.startswith('_'),
                            'is_static': any(isinstance(decorator, ast.Name) and decorator.id == 'staticmethod' 
                                           for decorator in item.decorator_list)
                        }
                        class_info['methods'].append(method_info)
                
                classes.append(class_info)
            
            elif isinstance(node, ast.FunctionDef):
                # 检查是否在类定义内部
                is_in_class = False
                for parent_node in ast.walk(tree):
                    if isinstance(parent_node, ast.ClassDef):
                        for child in ast.walk(parent_node):
                            if child is node:
                                is_in_class = True
                                break
                        if is_in_class:
                            break
                
                if not is_in_class:
                    func_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'args': [arg.arg for arg in node.args.args],
                        'is_private': node.name.startswith('_'),
                        'is_main': node.name == 'main'
                    }
                    functions.append(func_info)
        
        if not (module_doc or classes or functions):
            return None
        
        return {
            'file_path': file_path,
            'module_doc': module_doc,
            'classes': classes,
            'functions': functions
        }
    
    def _generate_code_markdown(self, code_docs: List[Dict[str, Any]]) -> str:
        """生成代码文档的Markdown内容"""
        content = """# 代码文档

## 概述

本文档包含项目的代码结构、类说明和函数说明。

"""
        
        for doc in code_docs:
            content += f"""## {doc['file_path']}

"""
            
            if doc['module_doc']:
                content += f"**模块说明**: {doc['module_doc']}\n\n"
            
            if doc['classes']:
                content += "### 类\n\n"
                for cls in doc['classes']:
                    content += f"#### {cls['name']}\n\n"
                    if cls['docstring']:
                        content += f"{cls['docstring']}\n\n"
                    
                    if cls['methods']:
                        content += "**方法**:\n\n"
                        for method in cls['methods']:
                            if not method['is_private']:
                                content += f"- `{method['name']}({', '.join(method['args'])})`"
                                if method['docstring']:
                                    content += f": {method['docstring']}"
                                content += "\n"
                        content += "\n"
            
            if doc['functions']:
                content += "### 函数\n\n"
                for func in doc['functions']:
                    if not func['is_private']:
                        content += f"#### {func['name']}\n\n"
                        content += f"**签名**: `{func['name']}({', '.join(func['args'])})`\n\n"
                        if func['docstring']:
                            content += f"{func['docstring']}\n\n"
        
        return content
    
    def generate_database_docs(self, tables: List[DatabaseTable]) -> str:
        """
        生成数据库文档
        
        Args:
            tables: 数据库表列表
            
        Returns:
            生成的文档路径
        """
        doc_content = self._generate_database_markdown(tables)
        output_path = self.subdirs['database'] / '数据库设计文档.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        # 生成ER图
        self._generate_er_diagram(tables)
        
        return str(output_path)
    
    def _generate_database_markdown(self, tables: List[DatabaseTable]) -> str:
        """生成数据库文档的Markdown内容"""
        content = """# 数据库设计文档

## 概述

本文档描述了数据库的设计结构，包括表结构、关系和索引设计。

## ER图

![数据库ER图](ER图.svg)

## 数据表详情

"""
        
        for table in tables:
            content += f"""### {table.name}

**描述**: {table.description}

**表结构**:

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
"""
            
            for column in table.columns:
                constraints = []
                if column.get('primary_key'):
                    constraints.append('PRIMARY KEY')
                if column.get('not_null'):
                    constraints.append('NOT NULL')
                if column.get('unique'):
                    constraints.append('UNIQUE')
                if column.get('default'):
                    constraints.append(f'DEFAULT {column["default"]}')
                
                constraint_str = ', '.join(constraints) if constraints else ''
                
                content += f"| {column['name']} | {column['type']} | {constraint_str} | {column.get('description', '')} |\n"
            
            content += "\n"
            
            if table.indexes:
                content += "**索引**:\n\n"
                for index in table.indexes:
                    content += f"- {index['name']}: {index['columns']}\n"
                content += "\n"
            
            if table.foreign_keys:
                content += "**外键关系**:\n\n"
                for fk in table.foreign_keys:
                    content += f"- {fk['column']} -> {fk['reference_table']}.{fk['reference_column']}\n"
                content += "\n"
        
        content += """## 设计原则

1. **规范化设计**: 遵循数据库设计范式
2. **性能优化**: 合理的索引设计
3. **数据完整性**: 外键约束和检查约束
4. **可扩展性**: 考虑未来扩展需求
"""
        
        return content
    
    def _generate_er_diagram(self, tables: List[DatabaseTable]):
        """生成ER图（Mermaid格式）"""
        diagram = "erDiagram\n"
        
        # 添加表结构
        for table in tables:
            diagram += f'    {table.name} {{\n'
            for column in table.columns:
                constraints = []
                if column.get('primary_key'):
                    constraints.append('PK')
                if column.get('not_null'):
                    constraints.append('NN')
                
                constraint_str = ' '.join(constraints)
                diagram += f'        {column["type"]} {column["name"]} "{constraint_str}"\n'
            diagram += '    }\n\n'
        
        # 添加关系
        for table in tables:
            for fk in table.foreign_keys:
                diagram += f'    {table.name} ||--o{{ {fk["reference_table"]} : "{fk["column"]}"\n'
        
        # 保存ER图
        mermaid_path = self.subdirs['database'] / 'ER图.mmd'
        with open(mermaid_path, 'w', encoding='utf-8') as f:
            f.write(diagram)
    
    def generate_deployment_docs(self, config: Dict[str, Any]) -> str:
        """
        生成部署文档
        
        Args:
            config: 部署配置信息
            
        Returns:
            生成的文档路径
        """
        doc_content = self._generate_deployment_markdown(config)
        output_path = self.subdirs['deployment'] / '部署文档.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        return str(output_path)
    
    def _generate_deployment_markdown(self, config: Dict[str, Any]) -> str:
        """生成部署文档的Markdown内容"""
        content = f"""# 部署文档

## 概述

本文档描述了系统的部署流程、环境配置和运维指南。

## 环境要求

- **操作系统**: {config.get('os', 'Linux/Windows/macOS')}
- **Python版本**: {config.get('python_version', '3.8+')}
- **数据库**: {config.get('database', 'PostgreSQL/MySQL')}
- **内存**: {config.get('memory', '4GB+')}
- **磁盘**: {config.get('disk', '50GB+')}

## 环境配置

### 开发环境

```bash
# 克隆项目
git clone <repository-url>
cd <project-name>

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件配置数据库连接等
```

### 生产环境

```bash
# 使用Docker部署
docker-compose up -d

# 或使用Kubernetes
kubectl apply -f k8s/
```

## 部署流程

1. **环境准备**
   - 安装必要的软件和依赖
   - 配置环境变量
   - 初始化数据库

2. **应用部署**
   - 构建应用镜像
   - 部署到目标环境
   - 配置负载均衡

3. **验证部署**
   - 健康检查
   - 功能测试
   - 性能测试

## 配置说明

### 环境变量

"""
        
        env_vars = config.get('environment_variables', {})
        for var, description in env_vars.items():
            content += f"- `{var}`: {description}\n"
        
        content += f"""

### 数据库配置

```sql
-- 创建数据库
CREATE DATABASE {config.get('database_name', 'app_db')};

-- 创建用户
CREATE USER {config.get('db_user', 'app_user')} WITH PASSWORD '{config.get('db_password', 'password')}';

-- 授权
GRANT ALL PRIVILEGES ON DATABASE {config.get('database_name', 'app_db')} TO {config.get('db_user', 'app_user')};
```

## 监控与日志

- **应用监控**: 使用 {config.get('monitoring', 'Prometheus + Grafana')}
- **日志收集**: 使用 {config.get('logging', 'ELK Stack')}
- **告警通知**: 配置邮件/短信告警

## 故障排除

### 常见问题

1. **应用启动失败**
   - 检查环境变量配置
   - 查看应用日志
   - 验证数据库连接

2. **性能问题**
   - 检查系统资源使用情况
   - 分析数据库查询性能
   - 查看应用性能指标

3. **部署问题**
   - 验证Docker镜像
   - 检查网络连接
   - 查看Kubernetes事件

## 备份与恢复

### 数据备份

```bash
# 数据库备份
pg_dump -h localhost -U {config.get('db_user', 'app_user')} {config.get('database_name', 'app_db')} > backup.sql

# 文件备份
tar -czf files_backup.tar.gz /app/data/
```

### 数据恢复

```bash
# 数据库恢复
psql -h localhost -U {config.get('db_user', 'app_user')} {config.get('database_name', 'app_db')} < backup.sql

# 文件恢复
tar -xzf files_backup.tar.gz -C /
```
"""
        
        return content
    
    def generate_development_docs(self, info: Dict[str, Any]) -> str:
        """
        生成开发者指南
        
        Args:
            info: 开发者信息
            
        Returns:
            生成的文档路径
        """
        doc_content = self._generate_development_markdown(info)
        output_path = self.subdirs['development'] / '开发者指南.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        return str(output_path)
    
    def _generate_development_markdown(self, info: Dict[str, Any]) -> str:
        """生成开发者指南的Markdown内容"""
        content = f"""# 开发者指南

## 概述

欢迎加入{info.get('project_name', '项目')}开发团队！本文档将帮助您快速上手项目开发。

## 项目信息

- **项目名称**: {info.get('project_name', '未命名项目')}
- **版本**: {info.get('version', '1.0.0')}
- **技术栈**: {', '.join(info.get('tech_stack', []))}
- **开发团队**: {', '.join(info.get('team_members', []))}

## 开发环境搭建

### 前置要求

- Python 3.8+
- Git
- IDE (推荐使用 VS Code 或 PyCharm)
- 数据库客户端工具

### 快速开始

```bash
# 1. 克隆仓库
git clone <repository-url>
cd <project-directory>

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\\Scripts\\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 5. 初始化数据库
python manage.py migrate  # Django项目
# 或
flask db upgrade  # Flask项目

# 6. 运行测试
python -m pytest

# 7. 启动开发服务器
python manage.py runserver  # Django项目
# 或
flask run  # Flask项目
```

## 项目结构

```
{info.get('project_name', 'project')}/
├── src/                    # 源代码目录
├── tests/                  # 测试代码目录
├── docs/                   # 文档目录
├── scripts/                # 脚本目录
├── config/                 # 配置文件
├── requirements.txt        # 依赖列表
├── README.md              # 项目说明
└── .env.example           # 环境变量示例
```

## 开发规范

### 代码风格

- 遵循 PEP 8 Python代码规范
- 使用 Black 进行代码格式化
- 使用 isort 整理导入语句
- 使用 flake8 进行代码检查

```bash
# 代码格式化
black .
isort .

# 代码检查
flake8 .
```

### Git提交规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型(type)**:
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

**示例**:
```
feat(auth): 添加用户登录功能

实现JWT令牌认证机制，包括：
- 用户登录接口
- 令牌验证中间件
- 令牌刷新机制

Closes #123
```

### 测试规范

- 单元测试覆盖率不低于80%
- 使用 pytest 作为测试框架
- 集成测试覆盖核心业务流程
- 性能测试用于关键接口

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_auth.py

# 生成覆盖率报告
pytest --cov=src tests/
```

## API开发

### RESTful API设计原则

1. 使用HTTP方法表示操作
2. 使用名词表示资源
3. 使用复数形式表示集合
4. 合理使用HTTP状态码

### API文档

使用Swagger/OpenAPI生成API文档：

```bash
# 生成API文档
python manage.py spectacular --file schema.yml  # Django
# 或
flask spec --output api-docs.json  # Flask
```

## 数据库开发

### 迁移管理

```bash
# 创建迁移
python manage.py makemigrations  # Django
# 或
flask db migrate -m "描述信息"  # Flask

# 执行迁移
python manage.py migrate  # Django
# 或
flask db upgrade  # Flask
```

### 数据模型设计原则

1. 遵循数据库设计范式
2. 合理使用索引
3. 考虑数据完整性约束
4. 预留扩展字段

## 调试与问题排查

### 日志配置

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### 常见问题排查

1. **导入错误**
   - 检查PYTHONPATH
   - 验证包结构
   - 确认虚拟环境

2. **数据库连接问题**
   - 检查数据库服务状态
   - 验证连接字符串
   - 确认用户权限

3. **性能问题**
   - 使用性能分析工具
   - 检查数据库查询
   - 分析内存使用

## 部署流程

### 开发环境部署

```bash
# 构建Docker镜像
docker build -t app:dev .

# 运行容器
docker run -p 8000:8000 app:dev
```

### 生产环境部署

```bash
# 构建生产镜像
docker build -f Dockerfile.prod -t app:prod .

# 部署到生产环境
docker-compose -f docker-compose.prod.yml up -d
```

## 贡献指南

1. Fork项目仓库
2. 创建功能分支
3. 提交代码变更
4. 创建Pull Request
5. 代码审查
6. 合并代码

## 联系方式

如有问题，请联系：
- **项目负责人**: {info.get('project_lead', '未指定')}
- **技术负责人**: {info.get('tech_lead', '未指定')}
- **团队邮箱**: {info.get('team_email', '未指定')}
"""
        
        return content
    
    def generate_api_docs(self, endpoints: List[APIEndpoint]) -> str:
        """
        生成API参考文档
        
        Args:
            endpoints: API端点列表
            
        Returns:
            生成的文档路径
        """
        doc_content = self._generate_api_markdown(endpoints)
        output_path = self.subdirs['api'] / 'API参考文档.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        return str(output_path)
    
    def _generate_api_markdown(self, endpoints: List[APIEndpoint]) -> str:
        """生成API文档的Markdown内容"""
        content = """# API参考文档

## 概述

本文档提供了完整的API参考信息，包括所有端点、参数、响应和示例。

## 认证

API使用JWT令牌进行认证，请在请求头中包含：

```
Authorization: Bearer <your-token>
```

## 响应格式

所有API响应都遵循统一的JSON格式：

```json
{
    "success": true,
    "data": {},
    "message": "操作成功",
    "timestamp": "2023-01-01T00:00:00Z"
}
```

## 错误代码

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

## API端点

"""
        
        # 按模块分组端点
        modules = {}
        for endpoint in endpoints:
            module = endpoint.path.split('/')[1] if len(endpoint.path.split('/')) > 1 else 'general'
            if module not in modules:
                modules[module] = []
            modules[module].append(endpoint)
        
        for module, module_endpoints in modules.items():
            content += f"### {module.title()}\n\n"
            
            for endpoint in module_endpoints:
                content += f"""#### {endpoint.method} {endpoint.path}

**描述**: {endpoint.description}

"""
                
                if endpoint.parameters:
                    content += "**参数**:\n\n"
                    content += "| 参数名 | 类型 | 位置 | 必填 | 说明 |\n"
                    content += "|--------|------|------|------|------|\n"
                    
                    for param_name, param_info in endpoint.parameters.items():
                        param_type = param_info.get('type', 'string')
                        location = param_info.get('location', 'query')
                        required = param_info.get('required', False)
                        description = param_info.get('description', '')
                        
                        content += f"| {param_name} | {param_type} | {location} | {'是' if required else '否'} | {description} |\n"
                    content += "\n"
                
                if endpoint.responses:
                    content += "**响应**:\n\n"
                    for status_code, response_info in endpoint.responses.items():
                        content += f"- **{status_code}**: {response_info.get('description', '')}\n"
                        if 'schema' in response_info:
                            content += f"  ```json\n{json.dumps(response_info['schema'], indent=2, ensure_ascii=False)}\n  ```\n"
                    content += "\n"
                
                if endpoint.examples:
                    content += "**示例**:\n\n"
                    for example_type, example in endpoint.examples.items():
                        content += f"*{example_type.title()}*:\n"
                        content += f"```json\n{json.dumps(example, indent=2, ensure_ascii=False)}\n```\n\n"
                
                content += "---\n\n"
        
        content += """## 速率限制

API实施速率限制以确保服务质量：

- **普通用户**: 1000请求/小时
- **认证用户**: 5000请求/小时
- **管理员**: 10000请求/小时

## SDK和代码示例

### Python SDK

```python
from api_client import APIClient

client = APIClient(base_url='https://api.example.com', token='your-token')

# 获取用户信息
user = client.users.get(user_id=123)
print(user)
```

### JavaScript SDK

```javascript
import { APIClient } from 'api-client';

const client = new APIClient({
    baseURL: 'https://api.example.com',
    token: 'your-token'
});

// 获取用户信息
const user = await client.users.get(123);
console.log(user);
```

### cURL示例

```bash
# 获取用户信息
curl -X GET "https://api.example.com/users/123" \\
  -H "Authorization: Bearer your-token" \\
  -H "Content-Type: application/json"
```
"""
        
        return content
    
    def generate_changelog(self, changes: List[Dict[str, Any]]) -> str:
        """
        生成变更日志
        
        Args:
            changes: 变更记录列表
            
        Returns:
            生成的文档路径
        """
        doc_content = self._generate_changelog_markdown(changes)
        output_path = self.subdirs['changelog'] / '变更日志.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        return str(output_path)
    
    def _generate_changelog_markdown(self, changes: List[Dict[str, Any]]) -> str:
        """生成变更日志的Markdown内容"""
        # 按版本分组变更
        versions = {}
        for change in changes:
            version = change.get('version', 'unreleased')
            if version not in versions:
                versions[version] = []
            versions[version].append(change)
        
        content = """# 变更日志

本文档记录了项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 计划中
- 待定

"""
        
        # 按版本号排序（最新的在前）
        sorted_versions = sorted(versions.keys(), reverse=True)
        
        for version in sorted_versions:
            if version == 'unreleased':
                continue
                
            content += f"## [{version}] - {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
            
            # 按变更类型分组
            change_types = {
                'added': '新增',
                'changed': '变更',
                'deprecated': '废弃',
                'removed': '移除',
                'fixed': '修复',
                'security': '安全'
            }
            
            version_changes = versions[version]
            for change_type, change_type_cn in change_types.items():
                type_changes = [c for c in version_changes if c.get('type') == change_type]
                if type_changes:
                    content += f"### {change_type_cn}\n\n"
                    for change in type_changes:
                        content += f"- {change.get('description', '')}\n"
                        if change.get('details'):
                            content += f"  - {change['details']}\n"
                    content += "\n"
        
        content += """## 版本说明

### 版本号规则

版本号采用 `主版本号.次版本号.修订号` 的格式：

- **主版本号**: 不兼容的API修改
- **次版本号**: 向下兼容的功能性新增
- **修订号**: 向下兼容的问题修正

### 变更类型

- **新增**: 新功能
- **变更**: 对现有功能的修改
- **废弃**: 即将移除的功能
- **移除**: 已移除的功能
- **修复**: 问题修复
- **安全**: 安全相关修复
"""
        
        return content
    
    def generate_performance_docs(self, metrics: Dict[str, Any]) -> str:
        """
        生成性能文档
        
        Args:
            metrics: 性能指标数据
            
        Returns:
            生成的文档路径
        """
        doc_content = self._generate_performance_markdown(metrics)
        output_path = self.subdirs['performance'] / '性能文档.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        return str(output_path)
    
    def _generate_performance_markdown(self, metrics: Dict[str, Any]) -> str:
        """生成性能文档的Markdown内容"""
        content = f"""# 性能文档

## 概述

本文档描述了系统的性能指标、基准测试结果和优化建议。

## 性能指标

### 系统性能

- **响应时间**: {metrics.get('response_time', 'N/A')}
- **吞吐量**: {metrics.get('throughput', 'N/A')} 请求/秒
- **并发用户数**: {metrics.get('concurrent_users', 'N/A')}
- **资源使用率**:
  - CPU: {metrics.get('cpu_usage', 'N/A')}
  - 内存: {metrics.get('memory_usage', 'N/A')}
  - 磁盘I/O: {metrics.get('disk_io', 'N/A')}
  - 网络: {metrics.get('network_usage', 'N/A')}

### 数据库性能

- **查询响应时间**: {metrics.get('db_query_time', 'N/A')}
- **连接池使用率**: {metrics.get('db_connection_pool', 'N/A')}
- **慢查询数量**: {metrics.get('slow_queries', 'N/A')}

### 缓存性能

- **缓存命中率**: {metrics.get('cache_hit_rate', 'N/A')}
- **缓存响应时间**: {metrics.get('cache_response_time', 'N/A')}

## 基准测试

### 负载测试结果

| 测试场景 | 并发用户数 | 平均响应时间 | 95%响应时间 | 错误率 |
|----------|------------|--------------|-------------|--------|
"""
        
        load_tests = metrics.get('load_tests', [])
        for test in load_tests:
            content += f"| {test.get('scenario', '')} | {test.get('users', '')} | {test.get('avg_response', '')} | {test.get('p95_response', '')} | {test.get('error_rate', '')} |\n"
        
        content += """

### 压力测试结果

"""
        
        stress_tests = metrics.get('stress_tests', [])
        for test in stress_tests:
            content += f"- **{test.get('scenario', '')}**: {test.get('result', '')}\n"
        
        content += """

## 性能优化

### 已实施的优化

"""
        
        optimizations = metrics.get('optimizations', [])
        for opt in optimizations:
            content += f"- **{opt.get('area', '')}**: {opt.get('description', '')}\n"
            if opt.get('impact'):
                content += f"  - 性能提升: {opt['impact']}\n"
        
        content += """

### 建议的优化

1. **数据库优化**
   - 添加合适的索引
   - 优化慢查询
   - 考虑读写分离

2. **缓存优化**
   - 增加缓存层
   - 优化缓存策略
   - 实施缓存预热

3. **代码优化**
   - 减少不必要的计算
   - 优化算法复杂度
   - 实施异步处理

4. **架构优化**
   - 考虑微服务拆分
   - 实施负载均衡
   - 优化网络通信

## 监控与告警

### 关键指标监控

- **应用响应时间**: 阈值 {metrics.get('alert_threshold_response', 'N/A')}
- **错误率**: 阈值 {metrics.get('alert_threshold_error', 'N/A')}
- **CPU使用率**: 阈值 {metrics.get('alert_threshold_cpu', 'N/A')}
- **内存使用率**: 阈值 {metrics.get('alert_threshold_memory', 'N/A')}

### 告警规则

```yaml
alerts:
  - name: high_response_time
    condition: response_time > 1000ms
    duration: 5m
    severity: warning
    
  - name: high_error_rate
    condition: error_rate > 5%
    duration: 2m
    severity: critical
```

## 性能测试指南

### 测试环境

- **硬件配置**: 与生产环境一致
- **网络环境**: 模拟真实网络条件
- **数据规模**: 使用接近生产的数据量

### 测试工具

- **负载测试**: JMeter, Locust
- **性能监控**: Prometheus, Grafana
- **APM工具**: New Relic, AppDynamics

### 测试流程

1. **基线测试**: 建立性能基线
2. **负载测试**: 验证正常负载下的性能
3. **压力测试**: 确定系统极限
4. **容量测试**: 验证扩展性
5. **稳定性测试**: 验证长期稳定性

## 性能调优建议

### 应用层

1. **代码优化**
   - 减少数据库查询次数
   - 使用连接池
   - 实施缓存策略
   - 异步处理耗时操作

2. **配置优化**
   - 调整JVM参数（Java应用）
   - 配置线程池大小
   - 优化连接池配置

### 数据库层

1. **索引优化**
   - 为常用查询添加索引
   - 避免过度索引
   - 定期重建索引

2. **查询优化**
   - 优化SQL语句
   - 避免SELECT *
   - 使用适当的JOIN方式

### 系统层

1. **资源调优**
   - 调整系统内核参数
   - 优化文件系统
   - 配置网络参数

2. **架构优化**
   - 实施负载均衡
   - 使用CDN
   - 考虑分布式架构
"""
        
        return content
    
    def generate_all_docs(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        生成所有文档
        
        Args:
            config: 完整配置信息
            
        Returns:
            生成的文档路径字典
        """
        generated_docs = {}
        
        try:
            # 生成架构文档
            if 'architecture' in config:
                arch_path = self.generate_architecture_docs(config['architecture'])
                generated_docs['architecture'] = arch_path
            
            # 生成代码文档
            if 'source_dirs' in config:
                code_path = self.generate_code_docs(config['source_dirs'])
                generated_docs['code'] = code_path
            
            # 生成数据库文档
            if 'database' in config:
                db_path = self.generate_database_docs(config['database'])
                generated_docs['database'] = db_path
            
            # 生成部署文档
            if 'deployment' in config:
                deploy_path = self.generate_deployment_docs(config['deployment'])
                generated_docs['deployment'] = deploy_path
            
            # 生成开发者指南
            if 'development' in config:
                dev_path = self.generate_development_docs(config['development'])
                generated_docs['development'] = dev_path
            
            # 生成API文档
            if 'api_endpoints' in config:
                api_path = self.generate_api_docs(config['api_endpoints'])
                generated_docs['api'] = api_path
            
            # 生成变更日志
            if 'changelog' in config:
                changelog_path = self.generate_changelog(config['changelog'])
                generated_docs['changelog'] = changelog_path
            
            # 生成性能文档
            if 'performance' in config:
                perf_path = self.generate_performance_docs(config['performance'])
                generated_docs['performance'] = perf_path
            
            # 生成主索引文档
            index_path = self._generate_index_doc(generated_docs)
            generated_docs['index'] = index_path
            
        except Exception as e:
            print(f"生成文档时出错: {e}")
        
        return generated_docs
    
    def _generate_index_doc(self, docs: Dict[str, str]) -> str:
        """生成主索引文档"""
        content = """# 技术文档索引

欢迎使用Q3技术文档生成器！

## 文档导航

"""
        
        doc_names = {
            'architecture': '架构文档',
            'code': '代码文档',
            'database': '数据库文档',
            'deployment': '部署文档',
            'development': '开发者指南',
            'api': 'API参考文档',
            'changelog': '变更日志',
            'performance': '性能文档'
        }
        
        for doc_type, doc_name in doc_names.items():
            if doc_type in docs:
                relative_path = Path(docs[doc_type]).name
                content += f"- [{doc_name}](./{doc_type}/{relative_path})\n"
        
        content += """

## 快速开始

1. 查看 [架构文档](./architecture/系统架构文档.md) 了解系统整体设计
2. 阅读 [开发者指南](./development/开发者指南.md) 开始开发
3. 参考 [API参考文档](./api/API参考文档.md) 了解接口规范
4. 按照 [部署文档](./deployment/部署文档.md) 进行部署

## 文档维护

- 文档与代码同步更新
- 定期审查文档内容
- 欢迎提交文档改进建议

## 联系信息

如有疑问，请联系开发团队。

---

*本文档由Q3技术文档生成器自动生成*
"""
        
        index_path = self.output_dir / 'README.md'
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(index_path)


# 示例配置和用法
if __name__ == "__main__":
    # 示例配置
    config = {
        'architecture': [
            ArchitectureComponent(
                name="用户服务",
                type="微服务",
                description="处理用户相关的业务逻辑",
                dependencies=["数据库服务", "缓存服务"],
                technologies=["Python", "FastAPI", "PostgreSQL"]
            ),
            ArchitectureComponent(
                name="认证服务",
                type="微服务",
                description="处理用户认证和授权",
                dependencies=["用户服务"],
                technologies=["Python", "JWT", "Redis"]
            )
        ],
        'source_dirs': ['src'],
        'database': [
            DatabaseTable(
                name="users",
                description="用户信息表",
                columns=[
                    {"name": "id", "type": "SERIAL", "primary_key": True, "description": "用户ID"},
                    {"name": "username", "type": "VARCHAR(50)", "not_null": True, "unique": True, "description": "用户名"},
                    {"name": "email", "type": "VARCHAR(100)", "not_null": True, "unique": True, "description": "邮箱"},
                    {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP", "description": "创建时间"}
                ],
                indexes=[
                    {"name": "idx_users_username", "columns": "username"},
                    {"name": "idx_users_email", "columns": "email"}
                ],
                foreign_keys=[]
            )
        ],
        'deployment': {
            'os': 'Linux',
            'python_version': '3.9+',
            'database': 'PostgreSQL 13+',
            'memory': '4GB',
            'disk': '100GB',
            'environment_variables': {
                'DATABASE_URL': '数据库连接字符串',
                'SECRET_KEY': '应用密钥',
                'DEBUG': '调试模式'
            }
        },
        'development': {
            'project_name': '示例项目',
            'version': '1.0.0',
            'tech_stack': ['Python', 'FastAPI', 'PostgreSQL', 'Redis'],
            'team_members': ['张三', '李四', '王五'],
            'project_lead': '张三',
            'tech_lead': '李四',
            'team_email': 'dev@example.com'
        },
        'api_endpoints': [
            APIEndpoint(
                method="GET",
                path="/api/users",
                description="获取用户列表",
                parameters={
                    "page": {"type": "int", "location": "query", "required": False, "description": "页码"},
                    "limit": {"type": "int", "location": "query", "required": False, "description": "每页数量"}
                },
                responses={
                    "200": {"description": "成功", "schema": {"type": "object", "properties": {"data": {"type": "array"}}}}
                },
                examples={
                    "request": {"page": 1, "limit": 10},
                    "response": {"data": [], "total": 0}
                }
            )
        ],
        'changelog': [
            {"version": "1.0.0", "type": "added", "description": "初始版本发布"},
            {"version": "1.0.0", "type": "added", "description": "用户管理功能"}
        ],
        'performance': {
            'response_time': '< 200ms',
            'throughput': '1000',
            'concurrent_users': '100',
            'cpu_usage': '< 70%',
            'memory_usage': '< 80%',
            'load_tests': [
                {"scenario": "正常负载", "users": "100", "avg_response": "150ms", "p95_response": "300ms", "error_rate": "0.1%"}
            ]
        }
    }
    
    # 创建生成器实例
    generator = TechnicalDocGenerator(".", "docs")
    
    # 生成所有文档
    docs = generator.generate_all_docs(config)
    
    print("文档生成完成！")
    for doc_type, path in docs.items():
        print(f"- {doc_type}: {path}")