"""
S6分析服务包
提供全面的数据分析、统计计算、报告生成等功能

快速开始:
    from S6 import AnalysisService, quick_analysis, create_sample_data
    
    # 创建分析实例
    analyzer = AnalysisService()
    
    # 或者使用快速分析
    results = quick_analysis(your_data)

版本信息:
    当前版本: 1.0.0
    作者: S6分析服务团队
    更新时间: 2025-11-13
"""

# 导入核心类
from .AnalysisService import AnalysisService

# 版本信息
__version__ = "1.0.0"
__author__ = "S6分析服务团队"
__email__ = "s6-analysis@company.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 S6分析服务团队"
__update_date__ = "2025-11-13"

# 包级导出
__all__ = [
    "AnalysisService",
    "quick_analysis", 
    "create_sample_data",
    "load_config",
    "S6Config",
    "ANALYSIS_TYPES",
    "EXPORT_FORMATS",
    "DEFAULT_THRESHOLDS",
    "VERSION_INFO"
]

# =============================================================================
# 默认配置
# =============================================================================

class S6Config:
    """S6分析服务默认配置"""
    
    # 数据处理配置
    DATA_PROCESSING = {
        'default_missing_threshold': 0.5,
        'default_drop_duplicates': True,
        'default_handle_missing': 'drop',
        'max_data_rows': 1000000,
        'chunk_size': 10000
    }
    
    # 统计分析配置
    STATISTICS = {
        'correlation_methods': ['pearson', 'spearman', 'kendall'],
        'default_correlation_method': 'pearson',
        'significance_level': 0.05,
        'confidence_interval': 0.95
    }
    
    # 异常值检测配置
    OUTLIER_DETECTION = {
        'default_method': 'iqr',
        'iqr_multiplier': 1.5,
        'zscore_threshold': 3.0,
        'isolation_forest_contamination': 0.1
    }
    
    # 聚类分析配置
    CLUSTERING = {
        'default_n_clusters': 3,
        'max_clusters': 20,
        'default_features': None,
        'scaling_required': True
    }
    
    # 可视化配置
    VISUALIZATION = {
        'default_figure_size': (12, 8),
        'dpi': 300,
        'figure_format': 'png',
        'colormap': 'viridis',
        'style': 'seaborn'
    }
    
    # 报告生成配置
    REPORTING = {
        'default_output_format': 'markdown',
        'include_visualizations': True,
        'include_raw_data': False,
        'template_style': 'detailed'
    }
    
    # 文件导出配置
    EXPORT = {
        'default_output_dir': 'analysis_output',
        'encoding': 'utf-8',
        'compression': False,
        'backup_enabled': True
    }

# =============================================================================
# 常量定义
# =============================================================================

# 分析类型常量
ANALYSIS_TYPES = {
    'DESCRIPTIVE': 'descriptive',
    'CORRELATION': 'correlation', 
    'OUTLIER': 'outlier',
    'TREND': 'trend',
    'CLUSTERING': 'clustering',
    'HYPOTHESIS': 'hypothesis',
    'PERFORMANCE': 'performance',
    'COMPREHENSIVE': 'comprehensive'
}

# 导出格式常量
EXPORT_FORMATS = {
    'JSON': 'json',
    'CSV': 'csv', 
    'EXCEL': 'excel',
    'MARKDOWN': 'md',
    'HTML': 'html',
    'PDF': 'pdf',
    'PNG': 'png',
    'SVG': 'svg'
}

# 数据类型常量
DATA_TYPES = {
    'NUMERIC': 'numeric',
    'CATEGORICAL': 'categorical',
    'DATETIME': 'datetime',
    'TEXT': 'text',
    'BOOLEAN': 'boolean'
}

# 默认阈值配置
DEFAULT_THRESHOLDS = {
    'missing_value_ratio': 0.5,
    'correlation_significance': 0.05,
    'outlier_contamination': 0.1,
    'cluster_silhouette': 0.3,
    'significance_level': 0.05
}

# 性能配置
PERFORMANCE_CONFIG = {
    'max_workers': 4,
    'memory_limit_mb': 1024,
    'processing_timeout': 300,
    'cache_enabled': True,
    'parallel_processing': False
}

# 消息常量
MESSAGES = {
    'INIT_SUCCESS': 'S6分析服务初始化成功',
    'DATA_LOADED': '数据加载完成，形状: {}',
    'ANALYSIS_COMPLETE': '{}分析完成',
    'EXPORT_SUCCESS': '结果导出完成: {}',
    'ERROR_NO_DATA': '错误: 请先加载数据',
    'ERROR_INVALID_CONFIG': '错误: 无效的配置参数'
}

# =============================================================================
# 便利函数
# =============================================================================

def quick_analysis(data, analysis_types=None, output_dir='quick_analysis_output'):
    """
    快速分析函数 - 一键式数据分析
    
    Args:
        data: 输入数据 (DataFrame, 文件路径, 或字典)
        analysis_types: 要执行的分析类型列表，默认为综合分析
        output_dir: 输出目录
        
    Returns:
        AnalysisService实例，包含分析结果
        
    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
        >>> analyzer = quick_analysis(data, ['descriptive', 'correlation'])
        >>> print(analyzer.get_analysis_summary())
    """
    # 创建分析实例
    analyzer = AnalysisService()
    
    # 如果未指定分析类型，使用综合分析
    if analysis_types is None:
        analysis_types = [
            ANALYSIS_TYPES['DESCRIPTIVE'],
            ANALYSIS_TYPES['CORRELATION'],
            ANALYSIS_TYPES['OUTLIER'],
            ANALYSIS_TYPES['CLUSTERING']
        ]
    
    # 加载数据
    analyzer.load_data(data)
    
    # 数据清洗
    analyzer.data_cleaning(
        drop_duplicates=S6Config.DATA_PROCESSING['default_drop_duplicates'],
        handle_missing=S6Config.DATA_PROCESSING['default_handle_missing'],
        missing_threshold=S6Config.DATA_PROCESSING['default_missing_threshold']
    )
    
    # 执行分析
    for analysis_type in analysis_types:
        if analysis_type == ANALYSIS_TYPES['DESCRIPTIVE']:
            analyzer.descriptive_statistics()
        elif analysis_type == ANALYSIS_TYPES['CORRELATION']:
            analyzer.correlation_analysis()
        elif analysis_type == ANALYSIS_TYPES['OUTLIER']:
            analyzer.outlier_detection()
        elif analysis_type == ANALYSIS_TYPES['CLUSTERING']:
            analyzer.clustering_analysis()
        elif analysis_type == ANALYSIS_TYPES['TREND']:
            # 需要时间列才能进行趋势分析
            pass
    
    # 生成可视化
    try:
        analyzer.generate_visualizations(output_dir)
    except:
        pass  # 跳过可视化生成失败
    
    # 生成报告
    report_file = f"{output_dir}/quick_analysis_report.md"
    analyzer.generate_report(report_file)
    
    print(f"{MESSAGES['ANALYSIS_COMPLETE'].format('快速分析')}")
    return analyzer


def load_config(config_file=None):
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径，默认为内置配置
        
    Returns:
        配置字典
        
    Example:
        >>> config = load_config('my_config.json')
        >>> print(config['DATA_PROCESSING'])
    """
    import json
    import os
    
    # 默认配置
    config = {
        'DATA_PROCESSING': S6Config.DATA_PROCESSING,
        'STATISTICS': S6Config.STATISTICS,
        'OUTLIER_DETECTION': S6Config.OUTLIER_DETECTION,
        'CLUSTERING': S6Config.CLUSTERING,
        'VISUALIZATION': S6Config.VISUALIZATION,
        'REPORTING': S6Config.REPORTING,
        'EXPORT': S6Config.EXPORT
    }
    
    # 如果提供了配置文件，则加载并合并
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            # 深度合并配置
            for key, value in file_config.items():
                if key in config:
                    config[key].update(value)
                else:
                    config[key] = value
        except Exception as e:
            print(f"配置文件加载失败，使用默认配置: {e}")
    
    return config


def create_sample_data(n_samples=1000, random_state=42):
    """
    创建示例数据用于测试和演示
    
    Args:
        n_samples: 样本数量
        random_state: 随机种子
        
    Returns:
        示例DataFrame
        
    Example:
        >>> data = create_sample_data(500)
        >>> print(data.head())
    """
    from .AnalysisService import create_sample_data as _create_sample_data
    return _create_sample_data()


def validate_data(data):
    """
    验证输入数据
    
    Args:
        data: 待验证的数据
        
    Returns:
        是否为有效数据
        
    Raises:
        ValueError: 数据无效时抛出异常
    """
    import pandas as pd
    
    if data is None:
        raise ValueError("数据不能为空")
    
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("DataFrame不能为空")
        return True
    elif isinstance(data, str):
        import os
        if not os.path.exists(data):
            raise ValueError(f"文件不存在: {data}")
        return True
    elif isinstance(data, dict):
        if len(data) == 0:
            raise ValueError("字典数据不能为空")
        return True
    else:
        raise ValueError(f"不支持的数据类型: {type(data)}")


def get_version_info():
    """
    获取版本信息
    
    Returns:
        版本信息字典
        
    Example:
        >>> info = get_version_info()
        >>> print(f"版本: {info['version']}")
    """
    return {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'copyright': __copyright__,
        'update_date': __update_date__
    }


def list_available_methods():
    """
    列出可用的分析方法
    
    Returns:
        方法列表字典
        
    Example:
        >>> methods = list_available_methods()
        >>> print("可用的分析方法:", list(methods.keys()))
    """
    return {
        'data_processing': [
            'load_data',
            'data_cleaning'
        ],
        'statistical_analysis': [
            'descriptive_statistics',
            'correlation_analysis',
            'hypothesis_testing'
        ],
        'advanced_analysis': [
            'outlier_detection',
            'trend_analysis', 
            'clustering_analysis'
        ],
        'visualization': [
            'generate_visualizations'
        ],
        'reporting': [
            'generate_report',
            'export_results'
        ],
        'performance': [
            'performance_analysis',
            'get_analysis_summary'
        ]
    }

# =============================================================================
# 快速入门指南
# =============================================================================

QUICK_START_GUIDE = """
=== S6分析服务快速入门指南 ===

1. 基本使用流程:
   
   from S6 import AnalysisService, quick_analysis
   
   # 方法一：使用AnalysisService类
   analyzer = AnalysisService()
   analyzer.load_data('your_data.csv')
   analyzer.data_cleaning()
   stats = analyzer.descriptive_statistics()
   
   # 方法二：使用快速分析函数
   analyzer = quick_analysis('your_data.csv')
   
2. 数据输入格式:
   
   - pandas DataFrame
   - CSV文件路径
   - Excel文件路径 (.xlsx, .xls)
   - JSON文件路径
   - Python字典

3. 主要分析方法:
   
   - descriptive_statistics()     # 描述性统计
   - correlation_analysis()       # 相关性分析
   - outlier_detection()         # 异常值检测
   - trend_analysis()            # 趋势分析
   - clustering_analysis()       # 聚类分析
   - hypothesis_testing()        # 假设检验

4. 输出和导出:
   
   - generate_visualizations()    # 生成图表
   - generate_report()           # 生成报告
   - export_results()            # 导出所有结果

5. 配置定制:
   
   from S6 import S6Config, load_config
   
   # 使用默认配置
   config = S6Config()
   
   # 加载自定义配置
   config = load_config('my_config.json')

6. 完整示例:
   
   # 加载数据
   analyzer = AnalysisService()
   analyzer.load_data('sales_data.csv')
   
   # 数据清洗
   analyzer.data_cleaning(
       drop_duplicates=True,
       handle_missing='fill_mean'
   )
   
   # 执行分析
   analyzer.descriptive_statistics()
   analyzer.correlation_analysis()
   analyzer.outlier_detection()
   analyzer.clustering_analysis(n_clusters=3)
   
   # 生成结果
   analyzer.generate_visualizations()
   analyzer.generate_report('analysis_report.md')
   analyzer.export_results('output_folder')

更多详细信息请参考API文档和示例代码。
"""

def print_quick_start():
    """打印快速入门指南"""
    print(QUICK_START_GUIDE)

# =============================================================================
# 包初始化
# =============================================================================

def _initialize_package():
    """包初始化函数"""
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                     S6分析服务 v{__version__}                          ║
    ║                                                              ║
    ║  提供专业的数据分析、统计计算和报告生成功能                  ║
    ║                                                              ║
    ║  快速开始:                                                  ║
    ║    from S6 import AnalysisService, quick_analysis           ║
    ║                                                              ║
    ║  获取帮助:                                                  ║
    ║    print_quick_start()  # 显示快速入门指南                  ║
    ║    help(AnalysisService) # 显示API文档                      ║
    ║                                                              ║
    ║  {MESSAGES['INIT_SUCCESS']}                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

# 包导入时自动初始化
_initialize_package()

# =============================================================================
# 导出快捷方式
# =============================================================================

# 为了方便使用，提供一些快捷导入方式
# 这些导入在包的顶级命名空间中可用
try:
    # 便利函数的别名
    qa = quick_analysis
    csd = create_sample_data
    lc = load_config
    gvi = get_version_info
    vd = validate_data
    qs = print_quick_start
    
    print("✓ S6分析服务包加载完成")
    
except ImportError as e:
    print(f"⚠ 部分功能导入失败: {e}")