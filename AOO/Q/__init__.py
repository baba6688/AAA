"""
Q区 - 配置管理模块导出接口

Q区是一个功能完整的配置管理模块，提供API文档生成、用户手册生成、
技术文档生成、策略文档生成、变更日志生成、报告生成、教程生成、
代码文档生成和文档状态聚合等核心功能。

模块组成:
- Q1: API文档生成器 - 代码解析、API端点识别、多格式输出
- Q2: 用户手册生成器 - 内容结构化、模板系统、多媒体支持
- Q3: 技术文档生成器 - 架构文档、代码文档、数据库文档
- Q4: 策略文档生成器 - 策略描述、参数说明、风险分析
- Q5: 变更日志生成器 - 版本管理、Git集成、多格式输出
- Q6: 报告生成器 - 数据处理、图表生成、报告输出
- Q7: 教程生成器 - 内容结构化、步骤指导、互动元素
- Q8: 代码文档生成器 - 多语言支持、代码解析、文档生成
- Q9: 文档状态聚合器 - 状态监控、分析、报告

版本: 1.0.0
作者: Q区开发团队
创建时间: 2025-11-13
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "Q区开发团队"
__description__ = "配置管理模块 - 全面的文档和报告生成解决方案"

# ================================
# Q1: API文档生成器
# ================================
from .Q1 import (
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

# ================================
# Q2: 用户手册生成器
# ================================
from .Q2 import (
    UserManualGenerator,
    ContentStructure,
    TemplateManager,
    MultimediaHandler,
    StepByStepGuide,
    FAQManager,
    FeedbackHandler,
    MultiLanguageSupport,
    VersionControl
)

# ================================
# Q3: 技术文档生成器
# ================================
from .Q3 import TechnicalDocGenerator
from .Q3.TechnicalDocGenerator import (
    ArchitectureComponent,
    APIEndpoint,
    DatabaseTable
)

# ================================
# Q4: 策略文档生成器
# ================================
from .Q4 import StrategyDocGenerator
from .Q4.StrategyDocGenerator import (
    StrategyParameter,
    RiskFactor,
    PerformanceMetric,
    UsageExample,
    VersionInfo as StrategyVersionInfo
)

# ================================
# Q5: 变更日志生成器
# ================================
from .Q5 import (
    ChangelogGenerator,
    VersionInfo,
    ChangeEntry,
    ReleaseInfo
)

# ================================
# Q6: 报告生成器
# ================================
try:
    from .Q6 import (
        ReportGenerator,
        DataProcessor,
        TemplateManager as ReportTemplateManager,
        ChartGenerator,
        ReportScheduler
    )
except ImportError as e:
    print(f"警告: 无法导入Q6模块的某些组件: {e}")
    ReportGenerator = None
    DataProcessor = None
    ReportTemplateManager = None
    ChartGenerator = None
    ReportScheduler = None

# ================================
# Q7: 教程生成器
# ================================
from .Q7 import (
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

# ================================
# Q8: 代码文档生成器
# ================================
from .Q8 import (
    CodeDocGenerator,
    PythonCodeParser,
    CodeCommentGenerator,
    DependencyAnalyzer,
    FlowchartGenerator,
    QualityAnalyzer,
    DocumentTemplate,
    CodeElement,
    FunctionInfo,
    ClassInfo,
    ImportInfo
)

# ================================
# Q9: 文档状态聚合器
# ================================
try:
    from .Q9 import (
        DocumentStatusAggregator,
        StatusCollector,
        AlertManager,
        ReportGenerator as StatusReportGenerator,
        DocumentStatus,
        AlertLevel,
        DocumentInfo,
        StatusReport,
        Alert
    )
except ImportError as e:
    # 处理可能的导入错误
    print(f"警告: 无法导入Q9模块的某些组件: {e}")
    DocumentStatusAggregator = None
    StatusCollector = None
    AlertManager = None
    StatusReportGenerator = None
    DocumentStatus = None
    AlertLevel = None
    DocumentInfo = None
    StatusReport = None
    Alert = None

# ================================
# 完整的导出列表
# ================================
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__description__",
    
    # Q1: API文档生成器
    "APIDocGenerator",
    "CodeParser",
    "DocstringParser", 
    "EndpointDetector",
    "ParameterDocumenter",
    "ExampleGenerator",
    "OutputFormatter",
    "InteractiveDocumentation",
    "VersionManager",
    
    # Q2: 用户手册生成器
    "UserManualGenerator",
    "ContentStructure",
    "TemplateManager",
    "MultimediaHandler",
    "StepByStepGuide",
    "FAQManager",
    "FeedbackHandler",
    "MultiLanguageSupport",
    "VersionControl",
    
    # Q3: 技术文档生成器
    "TechnicalDocGenerator",
    "ArchitectureComponent",
    "APIEndpoint", 
    "DatabaseTable",
    
    # Q4: 策略文档生成器
    "StrategyDocGenerator",
    "StrategyParameter",
    "RiskFactor",
    "PerformanceMetric",
    "UsageExample",
    "StrategyVersionInfo",
    
    # Q5: 变更日志生成器
    "ChangelogGenerator",
    "VersionInfo",
    "ChangeEntry",
    "ReleaseInfo",
    
    # Q6: 报告生成器
    "ReportGenerator",
    "DataProcessor",
    "ReportTemplateManager",
    "ChartGenerator",
    "ReportScheduler",
    
    # Q7: 教程生成器
    "TutorialGenerator",
    "DifficultyLevel",
    "ContentType",
    "CodeExample",
    "QuizQuestion",
    "Exercise",
    "ContentBlock",
    "Chapter",
    "ProgressRecord",
    "create_sample_tutorial",
    
    # Q8: 代码文档生成器
    "CodeDocGenerator",
    "PythonCodeParser",
    "CodeCommentGenerator",
    "DependencyAnalyzer",
    "FlowchartGenerator",
    "QualityAnalyzer",
    "DocumentTemplate",
    "CodeElement",
    "FunctionInfo",
    "ClassInfo",
    "ImportInfo",
    
    # Q9: 文档状态聚合器 (可选导入)
    "DocumentStatusAggregator",
    "StatusCollector",
    "AlertManager", 
    "StatusReportGenerator",
    "DocumentStatus",
    "AlertLevel",
    "DocumentInfo",
    "StatusReport",
    "Alert"
]

# ================================
# 模块级工厂函数
# ================================

def create_api_doc_generator():
    """创建API文档生成器实例"""
    return APIDocGenerator()

def create_user_manual_generator(output_dir="output"):
    """创建用户手册生成器实例"""
    return UserManualGenerator(output_dir=output_dir)

def create_technical_doc_generator(project_root=".", output_dir="docs"):
    """创建技术文档生成器实例"""
    return TechnicalDocGenerator(project_root=project_root, output_dir=output_dir)

def create_strategy_doc_generator(strategy_name, strategy_type="通用"):
    """创建策略文档生成器实例"""
    return StrategyDocGenerator(strategy_name=strategy_name, strategy_type=strategy_type)

def create_changelog_generator(repo_path=".", config=None):
    """创建变更日志生成器实例"""
    return ChangelogGenerator(repo_path=repo_path, config=config)

def create_report_generator(output_dir="reports"):
    """创建报告生成器实例"""
    return ReportGenerator(output_dir=output_dir)

def create_tutorial_generator(tutorial_id, title, description):
    """创建教程生成器实例"""
    return TutorialGenerator(tutorial_id=tutorial_id, title=title, description=description)

def create_code_doc_generator():
    """创建代码文档生成器实例"""
    return CodeDocGenerator()

def create_document_status_aggregator(config=None):
    """创建文档状态聚合器实例"""
    if DocumentStatusAggregator is None:
        raise ImportError("DocumentStatusAggregator not available")
    return DocumentStatusAggregator(config=config)

# ================================
# 快速使用函数
# ================================

def quick_api_doc(source_path, output_dir="api_docs", format_type="markdown"):
    """快速生成API文档"""
    generator = create_api_doc_generator()
    return generator.generate_documentation(
        source_path=source_path,
        output_dir=output_dir,
        format_type=format_type
    )

def quick_changelog(repo_path=".", format="markdown", output_file=None):
    """快速生成变更日志"""
    generator = create_changelog_generator(repo_path=repo_path)
    return generator.generate_changelog(format=format, output_file=output_file)

def quick_code_doc(source_path, output_path="docs", template_type="markdown"):
    """快速生成代码文档"""
    generator = create_code_doc_generator()
    return generator.generate_documentation(
        source_path=source_path,
        output_path=output_path,
        template_type=template_type
    )

def quick_report(data_source, output_dir="reports", template_type="standard"):
    """快速生成报告"""
    generator = create_report_generator(output_dir=output_dir)
    return generator.generate_report(
        data_source=data_source,
        template_type=template_type
    )

# ================================
# 模块信息
# ================================

def get_module_info():
    """获取Q区模块信息"""
    return {
        "name": "配置管理模块",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "submodules": {
            "Q1": "API文档生成器 - 代码解析和API文档生成",
            "Q2": "用户手册生成器 - 用户手册和指南生成",
            "Q3": "技术文档生成器 - 技术文档和架构文档生成", 
            "Q4": "策略文档生成器 - 策略文档和参数说明生成",
            "Q5": "变更日志生成器 - 版本管理和变更记录生成",
            "Q6": "报告生成器 - 数据分析和报告生成",
            "Q7": "教程生成器 - 互动教程和步骤指导生成",
            "Q8": "代码文档生成器 - 多语言代码文档生成",
            "Q9": "文档状态聚合器 - 文档状态监控和分析"
        },
        "total_classes": len(__all__),
        "features": [
            "多格式文档输出 (Markdown, HTML, JSON, PDF)",
            "代码解析和自动文档生成",
            "版本管理和变更追踪",
            "多语言支持",
            "模板系统和自定义格式",
            "交互式文档和在线生成",
            "自动化报告生成",
            "状态监控和预警系统"
        ]
    }

def print_module_info():
    """打印模块信息"""
    info = get_module_info()
    print("=" * 60)
    print(f"🗂️  {info['name']} v{info['version']}")
    print("=" * 60)
    print(f"作者: {info['author']}")
    print(f"描述: {info['description']}")
    print()
    print("📋 子模块列表:")
    for key, desc in info['submodules'].items():
        print(f"  {key}: {desc}")
    print()
    print("⭐ 主要特性:")
    for feature in info['features']:
        print(f"  ✓ {feature}")
    print()
    print(f"📊 总计导出类: {info['total_classes']}")
    print("=" * 60)

# ================================
# 使用示例
# ================================

def example_usage():
    """使用示例"""
    print("\n🚀 Q区配置管理模块使用示例")
    print("=" * 50)
    
    print("\n1. API文档生成:")
    print("   from Q import create_api_doc_generator")
    print("   generator = create_api_doc_generator()")
    print("   generator.generate_documentation('path/to/code', 'api_docs')")
    
    print("\n2. 变更日志生成:")
    print("   from Q import quick_changelog")
    print("   changelog = quick_changelog(format='markdown')")
    
    print("\n3. 代码文档生成:")
    print("   from Q import quick_code_doc")
    print("   quick_code_doc('my_project/', 'docs/', 'html')")
    
    print("\n4. 报告生成:")
    print("   from Q import quick_report")
    print("   quick_report('data.csv', 'reports/')")
    
    print("\n5. 状态聚合:")
    print("   from Q import create_document_status_aggregator")
    print("   aggregator = create_document_status_aggregator()")
    
    print("\n" + "=" * 50)
    print("💡 提示: 使用 help(Q) 查看完整API文档")

# ================================
# 初始化和欢迎信息
# ================================

def _initialize():
    """初始化Q区模块"""
    try:
        # 设置日志级别
        import logging
        logging.getLogger(__name__).setLevel(logging.INFO)
        
        # 预检查关键组件
        required_components = [
            APIDocGenerator,
            UserManualGenerator,
            TechnicalDocGenerator,
            ChangelogGenerator
        ]
        
        for component in required_components:
            if component is None:
                raise ImportError(f"关键组件 {component} 导入失败")
                
        print("✅ Q区配置管理模块初始化成功")
        return True
        
    except Exception as e:
        print(f"❌ Q区模块初始化失败: {e}")
        return False

# 在导入时自动初始化
_initialization_success = _initialize()

# ================================
# 模块入口点
# ================================

if __name__ == "__main__":
    print_module_info()
    example_usage()