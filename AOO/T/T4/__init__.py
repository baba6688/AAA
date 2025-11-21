"""
T4数据转换器模块
================

T4数据转换器是一个功能强大的Python数据处理工具，支持多种数据格式转换、
结构转换、聚合分析、数据标准化等功能。

主要功能：
1. 数据格式转换（JSON、CSV、XML、Parquet、Excel等）
2. 数据结构转换（宽表转长表、长表转宽表）
3. 数据聚合和分组分析
4. 数据标准化和归一化
5. 数据编码和解码
6. 数据映射和匹配
7. 转换流水线管理
8. 性能优化和验证

版本: 1.0.0
作者: T4开发团队
日期: 2025-11-13
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "T4开发团队"
__email__ = "dev@minimax.chat"
__description__ = "T4数据转换器 - 功能强大的数据处理和转换工具"

# 导入所有主要类
from .DataTransformer import (
    # 枚举类
    DataFormat,
    NormalizationMethod,
    AggregationMethod,
    
    # 配置类
    TransformationConfig,
    TransformationResult,
    
    # 主转换器类
    DataTransformer,
    
    # 辅助函数
    create_sample_data,
    run_comprehensive_tests
)

# 默认配置
DEFAULT_CONFIG = TransformationConfig(
    source_format=DataFormat.JSON,
    target_format=DataFormat.CSV,
    encoding="utf-8",
    chunk_size=10000,
    parallel=False,
    max_workers=4,
    preserve_index=False,
    compress=False,
    validation=True
)

# 常用配置模板
CONFIG_TEMPLATES = {
    "csv_to_json": TransformationConfig(
        source_format=DataFormat.CSV,
        target_format=DataFormat.JSON,
        encoding="utf-8",
        preserve_index=False
    ),
    
    "json_to_excel": TransformationConfig(
        source_format=DataFormat.JSON,
        target_format=DataFormat.EXCEL,
        encoding="utf-8",
        preserve_index=True
    ),
    
    "parquet_optimized": TransformationConfig(
        source_format=DataFormat.PARQUET,
        target_format=DataFormat.PARQUET,
        compress=True,
        parallel=True,
        max_workers=8,
        chunk_size=50000
    ),
    
    "xml_to_dataframe": TransformationConfig(
        source_format=DataFormat.XML,
        target_format=DataFormat.CSV,
        encoding="utf-8",
        preserve_index=False
    ),
    
    "high_performance": TransformationConfig(
        source_format=DataFormat.JSON,
        target_format=DataFormat.CSV,
        parallel=True,
        max_workers=8,
        chunk_size=100000,
        validation=False
    )
}

# 常量定义
SUPPORTED_FORMATS = list(DataFormat.__members__.keys())
SUPPORTED_NORMALIZATION_METHODS = list(NormalizationMethod.__members__.keys())
SUPPORTED_AGGREGATION_METHODS = list(AggregationMethod.__members__.keys())

# 文件扩展名映射
FORMAT_EXTENSIONS = {
    DataFormat.JSON: ".json",
    DataFormat.CSV: ".csv",
    DataFormat.XML: ".xml",
    DataFormat.PARQUET: ".parquet",
    DataFormat.EXCEL: ".xlsx",
    DataFormat.HDF5: ".h5",
    DataFormat.PICKLE: ".pkl",
    DataFormat.SQL: ".sql"
}

# MIME类型映射
FORMAT_MIMETYPES = {
    DataFormat.JSON: "application/json",
    DataFormat.CSV: "text/csv",
    DataFormat.XML: "application/xml",
    DataFormat.PARQUET: "application/octet-stream",
    DataFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    DataFormat.HDF5: "application/x-hdf5",
    DataFormat.PICKLE: "application/octet-stream",
    DataFormat.SQL: "application/sql"
}

# 性能配置
PERFORMANCE_CONFIGS = {
    "fast": {
        "parallel": False,
        "chunk_size": 1000,
        "max_workers": 2,
        "validation": False
    },
    
    "balanced": {
        "parallel": True,
        "chunk_size": 10000,
        "max_workers": 4,
        "validation": True
    },
    
    "thorough": {
        "parallel": True,
        "chunk_size": 5000,
        "max_workers": 2,
        "validation": True
    }
}

# 便利函数
def create_transformer(config=None, performance_profile="balanced"):
    """
    创建数据转换器的便利函数
    
    Args:
        config: 自定义配置，如果为None则使用性能配置模板
        performance_profile: 性能配置模板 ('fast', 'balanced', 'thorough')
        
    Returns:
        DataTransformer: 配置好的数据转换器实例
    """
    if config is None:
        if performance_profile in PERFORMANCE_CONFIGS:
            perf_config = PERFORMANCE_CONFIGS[performance_profile]
            config = TransformationConfig(
                source_format=DataFormat.JSON,
                target_format=DataFormat.CSV,
                **perf_config
            )
        else:
            config = DEFAULT_CONFIG
    
    return DataTransformer(config)

def quick_convert(data, target_format="csv", **kwargs):
    """
    快速数据转换
    
    Args:
        data: 待转换的数据
        target_format: 目标格式
        **kwargs: 其他转换参数
        
    Returns:
        TransformationResult: 转换结果
    """
    transformer = create_transformer()
    config = TransformationConfig(
        source_format=DataFormat.JSON,  # 通用源格式
        target_format=DataFormat(target_format),
        **kwargs
    )
    return transformer.convert_format(data, config)

def batch_convert(data_list, target_format="csv", **kwargs):
    """
    批量数据转换
    
    Args:
        data_list: 数据列表
        target_format: 目标格式
        **kwargs: 其他转换参数
        
    Returns:
        List[TransformationResult]: 转换结果列表
    """
    transformer = create_transformer()
    config = TransformationConfig(
        source_format=DataFormat.JSON,
        target_format=DataFormat(target_format),
        parallel=True,
        max_workers=4,
        **kwargs
    )
    
    def process_func(data):
        return transformer.convert_format(data, config)
    
    return transformer.batch_process(data_list, process_func, {"parallel": True, "max_workers": 4})

def normalize_dataframe(df, columns=None, method="min_max", **kwargs):
    """
    快速数据标准化
    
    Args:
        df: 待标准化的DataFrame
        columns: 要标准化的列，默认为数值列
        method: 标准化方法
        **kwargs: 其他参数
        
    Returns:
        Tuple[DataFrame, Dict]: 标准化后的数据和参数
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    transformer = create_transformer()
    return transformer.normalize_data(
        df, 
        columns, 
        method=NormalizationMethod(method),
        **kwargs
    )

def aggregate_dataframe(df, group_by, agg_dict, **kwargs):
    """
    快速数据聚合
    
    Args:
        df: 待聚合的DataFrame
        group_by: 分组列
        agg_dict: 聚合配置
        **kwargs: 其他参数
        
    Returns:
        DataFrame: 聚合结果
    """
    transformer = create_transformer()
    return transformer.aggregate_data(df, group_by, agg_dict, **kwargs)

def encode_dataframe(df, columns, method="onehot", **kwargs):
    """
    快速数据编码
    
    Args:
        df: 待编码的DataFrame
        columns: 要编码的列
        method: 编码方法
        **kwargs: 其他参数
        
    Returns:
        Tuple[DataFrame, Dict]: 编码后的数据和编码映射
    """
    transformer = create_transformer()
    return transformer.encode_data(df, columns, method, **kwargs)

# 验证和诊断函数
def validate_data(df, rules=None):
    """
    快速数据验证
    
    Args:
        df: 待验证的DataFrame
        rules: 验证规则，默认使用基础规则
        
    Returns:
        Dict: 验证结果
    """
    if rules is None:
        rules = {
            "required_columns": list(df.columns[:3]) if len(df.columns) >= 3 else list(df.columns),
            "null_check": {col: 0.5 for col in df.select_dtypes(include=['object']).columns},
            "ranges": {}
        }
    
    transformer = create_transformer()
    return transformer.validate_data(df, rules)

def get_data_summary(df):
    """
    获取数据概览
    
    Args:
        df: 要分析的DataFrame
        
    Returns:
        Dict: 数据概览信息
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        "missing_values": df.isna().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist()
    }
    
    # 添加统计信息
    if len(df.select_dtypes(include=['number']).columns) > 0:
        summary["numeric_summary"] = df.describe().to_dict()
    
    return summary

# 转换流水线模板
PIPELINE_TEMPLATES = {
    "data_cleaning": [
        {"type": "validate", "config": {}},
        {"type": "aggregate", "config": {"group_by": [], "agg_dict": {}}},
        {"type": "normalize", "config": {"columns": [], "method": NormalizationMethod.MIN_MAX}}
    ],
    
    "feature_preparation": [
        {"type": "encode", "config": {"columns": [], "method": "onehot"}},
        {"type": "normalize", "config": {"columns": [], "method": NormalizationMethod.Z_SCORE}}
    ],
    
    "data_export": [
        {"type": "aggregate", "config": {"group_by": [], "agg_dict": {}}},
        {"type": "format_convert", "config": {"target_format": DataFormat.CSV}}
    ]
}

def create_pipeline(steps=None, template=None):
    """
    创建转换流水线的便利函数
    
    Args:
        steps: 自定义步骤列表
        template: 预定义模板名称
        
    Returns:
        Callable: 流水线函数
    """
    transformer = create_transformer()
    
    if template and template in PIPELINE_TEMPLATES:
        steps = PIPELINE_TEMPLATES[template]
    elif steps is None:
        steps = PIPELINE_TEMPLATES["data_cleaning"]
    
    return transformer.create_pipeline(steps)

# 快速入门指南
QUICK_START_GUIDE = """
=== T4数据转换器快速入门 ===

1. 基本导入：
   from T4 import DataTransformer, create_transformer, quick_convert

2. 创建转换器：
   # 使用默认配置
   transformer = create_transformer()
   
   # 使用性能配置
   transformer = create_transformer(performance_profile="fast")
   
   # 使用自定义配置
   from T4 import TransformationConfig, DataFormat
   config = TransformationConfig(
       source_format=DataFormat.JSON,
       target_format=DataFormat.CSV,
       parallel=True
   )
   transformer = create_transformer(config)

3. 快速转换：
   # 简单格式转换
   result = quick_convert(my_data, target_format="csv")
   
   # 批量转换
   results = batch_convert([data1, data2, data3], target_format="excel")

4. 数据处理：
   # 标准化
   normalized_data, params = normalize_dataframe(df, columns=['price', 'quantity'])
   
   # 聚合
   aggregated = aggregate_dataframe(df, group_by=['category'], agg_dict={'price': 'mean'})
   
   # 编码
   encoded_data, mapping = encode_dataframe(df, columns=['category'], method='onehot')

5. 验证数据：
   validation_result = validate_data(df)
   summary = get_data_summary(df)

6. 使用流水线：
   pipeline = create_pipeline(template="feature_preparation")
   result = pipeline(raw_data)

更多详细信息请参考文档和示例代码。
"""

def print_quick_start():
    """打印快速入门指南"""
    print(QUICK_START_GUIDE)

def get_version_info():
    """获取版本信息"""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": __description__,
        "supported_formats": SUPPORTED_FORMATS,
        "supported_normalization_methods": SUPPORTED_NORMALIZATION_METHODS,
        "supported_aggregation_methods": SUPPORTED_AGGREGATION_METHODS
    }

# 测试和演示函数
def demo_basic_conversion():
    """演示基本转换功能"""
    print("=== 基本转换演示 ===")
    
    # 创建示例数据
    sample_data = create_sample_data()
    print(f"创建示例数据: {len(sample_data)} 行")
    
    # 创建转换器
    transformer = create_transformer(performance_profile="fast")
    
    # 转换为JSON
    json_result = quick_convert(sample_data, target_format="json")
    print(f"JSON转换: {'成功' if json_result.success else '失败'}")
    
    # 转换为CSV
    csv_result = quick_convert(sample_data, target_format="csv")
    print(f"CSV转换: {'成功' if csv_result.success else '失败'}")
    
    return sample_data

def demo_data_processing():
    """演示数据处理功能"""
    print("=== 数据处理演示 ===")
    
    # 获取示例数据
    sample_data = create_sample_data()
    
    # 数据标准化
    normalized, params = normalize_dataframe(sample_data, columns=['price', 'quantity'])
    print(f"数据标准化完成: {len(sample_data.columns)} 列")
    
    # 数据聚合
    aggregated = aggregate_dataframe(
        sample_data, 
        group_by=['category', 'region'], 
        agg_dict={'price': 'mean', 'quantity': 'sum'}
    )
    print(f"数据聚合完成: {len(sample_data)} -> {len(aggregated)} 行")
    
    # 数据编码
    encoded, mapping = encode_dataframe(sample_data, columns=['category', 'region'])
    print(f"数据编码完成: {len(sample_data.columns)} -> {len(encoded.columns)} 列")
    
    return aggregated

def demo_pipeline():
    """演示流水线功能"""
    print("=== 流水线演示 ===")
    
    # 获取示例数据
    sample_data = create_sample_data()
    
    # 创建流水线
    pipeline = create_pipeline(template="feature_preparation")
    
    # 执行流水线
    try:
        result = pipeline(sample_data)
        print(f"流水线执行完成: {len(result)} 行")
    except Exception as e:
        print(f"流水线执行失败: {e}")

def run_demo():
    """运行完整演示"""
    print("T4数据转换器演示程序")
    print("=" * 50)
    
    demo_basic_conversion()
    print()
    demo_data_processing()
    print()
    demo_pipeline()
    
    print("\n演示完成！")
    print("输入 'help' 查看快速入门指南")
    print("输入 'version' 查看版本信息")

# 导出所有公共接口
__all__ = [
    # 主要类
    "DataFormat",
    "NormalizationMethod", 
    "AggregationMethod",
    "TransformationConfig",
    "TransformationResult",
    "DataTransformer",
    
    # 便利函数
    "create_transformer",
    "quick_convert",
    "batch_convert",
    "normalize_dataframe",
    "aggregate_dataframe",
    "encode_dataframe",
    "validate_data",
    "get_data_summary",
    "create_pipeline",
    
    # 常量和配置
    "DEFAULT_CONFIG",
    "CONFIG_TEMPLATES",
    "SUPPORTED_FORMATS",
    "SUPPORTED_NORMALIZATION_METHODS", 
    "SUPPORTED_AGGREGATION_METHODS",
    "FORMAT_EXTENSIONS",
    "FORMAT_MIMETYPES",
    "PERFORMANCE_CONFIGS",
    "PIPELINE_TEMPLATES",
    
    # 工具函数
    "create_sample_data",
    "run_comprehensive_tests",
    "run_demo",
    "print_quick_start",
    "get_version_info",
    
    # 演示函数
    "demo_basic_conversion",
    "demo_data_processing",
    "demo_pipeline"
]

# 模块初始化信息
__all__.sort()  # 保持有序

# 在导入时显示欢迎信息
def _show_welcome():
    """显示欢迎信息"""
    print(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║              T4数据转换器 v{__version__}                               ║
    ║                                                                      ║
    ║    功能强大的数据处理和转换工具，支持多种格式转换和数据分析          ║
    ║                                                                      ║
    ║    快速开始: from T4 import create_transformer, quick_convert        ║
    ║    帮助文档: print_quick_start()                                    ║
    ║    运行演示: run_demo()                                             ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

# 取消下面的注释以在导入时显示欢迎信息
# _show_welcome()
