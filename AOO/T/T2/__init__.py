"""
T2数据清洗器 - 导出接口
=====================

一个功能完整的企业级数据清洗解决方案，提供全面的数据质量管理和清洗功能。

功能特性:
- 缺失值处理（删除、填充、插值等11种策略）
- 重复值检测和处理
- 异常值检测和处理（6种检测方法）
- 数据格式标准化和清理
- 数据类型智能转换
- 数据去噪和过滤
- 数据一致性检查和修复
- 数据质量评估（6个维度）
- 清洗结果验证
- 详细的清洗报告生成

作者: T2数据处理团队
日期: 2025-11-13
版本: 1.0.0
许可证: MIT
"""

# ===========================
# 版本信息
# ===========================

__version__ = "1.0.0"
__author__ = "T2数据处理团队"
__email__ = "t2-team@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 T2数据处理团队"
__description__ = "企业级数据清洗和数据质量管理解决方案"

# ===========================
# 核心类和枚举
# ===========================

from .DataCleaner import (
    # 枚举类
    MissingValueStrategy,
    OutlierDetectionMethod,
    DataQualityMetric,
    
    # 数据类
    DataQualityReport,
    CleaningReport,
    
    # 主要类
    DataCleaner,
    
    # 工具函数
    create_sample_data,
    run_comprehensive_test
)

# ===========================
# 公共API导出
# ===========================

__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__email__', 
    '__license__',
    '__copyright__',
    '__description__',
    
    # 核心类
    'MissingValueStrategy',
    'OutlierDetectionMethod', 
    'DataQualityMetric',
    'DataQualityReport',
    'CleaningReport',
    'DataCleaner',
    
    # 工具函数
    'create_sample_data',
    'run_comprehensive_test',
    
    # 便利函数
    'create_cleaner',
    'quick_clean',
    'assess_quality',
    'load_and_clean',
    'get_default_config',
    'validate_data',
    
    # 常量
    'DEFAULT_QUALITY_THRESHOLD',
    'SUPPORTED_FORMATS',
    'QUALITY_SCORE_LEVELS'
]

# ===========================
# 默认配置
# ===========================

DEFAULT_CONFIG = {
    # 缺失值处理
    'missing_value_strategy': MissingValueStrategy.FILL_MEDIAN,
    'missing_value_constant': 0,
    
    # 异常值检测
    'outlier_detection_method': OutlierDetectionMethod.IQR,
    'outlier_threshold': 1.5,
    
    # 重复值处理
    'duplicate_keep': 'first',
    
    # 列类型配置
    'numeric_columns': None,
    'categorical_columns': None,
    'datetime_columns': None,
    
    # 格式标准化
    'standardize_formats': True,
    
    # 数据去噪
    'remove_noise': True,
    'noise_threshold': 0.1,
    
    # 一致性检查
    'consistency_check': True,
    
    # 质量阈值
    'quality_threshold': 0.8,
    
    # 随机种子
    'random_state': 42
}

# ===========================
# 常量定义
# ===========================

# 默认质量阈值
DEFAULT_QUALITY_THRESHOLD = 0.8

# 支持的文件格式
SUPPORTED_FORMATS = {
    'csv': 'CSV文件',
    'xlsx': 'Excel文件', 
    'xls': 'Excel文件',
    'json': 'JSON文件',
    'parquet': 'Parquet文件',
    'feather': 'Feather文件'
}

# 质量评分等级
QUALITY_SCORE_LEVELS = {
    'excellent': (0.9, 1.0),
    'good': (0.8, 0.9),
    'fair': (0.7, 0.8),
    'poor': (0.6, 0.7),
    'unacceptable': (0.0, 0.6)
}

# 缺失值处理策略说明
MISSING_VALUE_STRATEGIES = {
    MissingValueStrategy.DELETE: "删除包含缺失值的行",
    MissingValueStrategy.FILL_MEAN: "用均值填充数值列",
    MissingValueStrategy.FILL_MEDIAN: "用中位数填充数值列", 
    MissingValueStrategy.FILL_MODE: "用众数填充",
    MissingValueStrategy.FILL_FORWARD: "前向填充",
    MissingValueStrategy.FILL_BACKWARD: "后向填充",
    MissingValueStrategy.FILL_CONSTANT: "常量填充",
    MissingValueStrategy.INTERPOLATE_LINEAR: "线性插值",
    MissingValueStrategy.INTERPOLATE_POLYNOMIAL: "多项式插值",
    MissingValueStrategy.INTERPOLATE_SPLINE: "样条插值"
}

# 异常值检测方法说明
OUTLIER_DETECTION_METHODS = {
    OutlierDetectionMethod.ZSCORE: "Z-score方法（基于标准差）",
    OutlierDetectionMethod.IQR: "四分位距方法（推荐）",
    OutlierDetectionMethod.ISOLATION_FOREST: "孤立森林方法",
    OutlierDetectionMethod.LOCAL_OUTLIER_FACTOR: "局部异常因子",
    OutlierDetectionMethod.DBSCAN: "DBSCAN聚类方法",
    OutlierDetectionMethod.STATISTICAL: "统计方法"
}

# 数据质量指标说明
DATA_QUALITY_METRICS = {
    DataQualityMetric.COMPLETENESS: "完整性 - 数据的缺失程度",
    DataQualityMetric.ACCURACY: "准确性 - 数据的正确程度",
    DataQualityMetric.CONSISTENCY: "一致性 - 数据内部逻辑一致性",
    DataQualityMetric.VALIDITY: "有效性 - 数据的格式规范性",
    DataQualityMetric.UNIQUENESS: "唯一性 - 数据的去重程度",
    DataQualityMetric.TIMELINESS: "时效性 - 数据的更新时间"
}

# ===========================
# 便利函数
# ===========================

def create_cleaner(config: dict = None, **kwargs) -> DataCleaner:
    """
    创建数据清洗器的便利函数
    
    Args:
        config: 基础配置字典
        **kwargs: 额外的配置参数
        
    Returns:
        DataCleaner实例
        
    Examples:
        >>> # 使用默认配置
        >>> cleaner = create_cleaner()
        
        >>> # 自定义配置
        >>> cleaner = create_cleaner(
        ...     missing_value_strategy=MissingValueStrategy.FILL_MEAN,
        ...     outlier_threshold=2.0
        ... )
    """
    # 合并默认配置和自定义配置
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)
    final_config.update(kwargs)
    
    return DataCleaner(final_config)


def quick_clean(data, **kwargs) -> tuple:
    """
    快速数据清洗的便利函数
    
    Args:
        data: 输入数据（DataFrame、文件路径或字典）
        **kwargs: 清洗参数
        
    Returns:
        (清洗后数据, 清洗报告) 元组
        
    Examples:
        >>> # 快速清洗CSV文件
        >>> cleaned_data, report = quick_clean('data.csv')
        
        >>> # 自定义清洗参数
        >>> cleaned_data, report = quick_clean(
        ...     data,
        ...     missing_strategy=MissingValueStrategy.FILL_MEAN,
        ...     outlier_action='cap'
        ... )
    """
    cleaner = create_cleaner(**kwargs)
    return cleaner.clean_data(data)


def assess_quality(data) -> DataQualityReport:
    """
    快速数据质量评估
    
    Args:
        data: 待评估的数据
        
    Returns:
        数据质量报告
        
    Examples:
        >>> # 评估数据质量
        >>> report = assess_quality(data)
        >>> print(f"综合评分: {report.overall_score:.2%}")
        >>> print(f"主要问题: {report.issues}")
    """
    cleaner = create_cleaner()
    return cleaner.assess_data_quality(data)


def load_and_clean(filepath: str, **kwargs) -> tuple:
    """
    加载并清洗数据文件的便利函数
    
    Args:
        filepath: 文件路径
        **kwargs: 清洗参数
        
    Returns:
        (清洗后数据, 清洗报告) 元组
        
    Examples:
        >>> # 加载并清洗CSV文件
        >>> data, report = load_and_clean('data.csv')
        
        >>> # 指定类型转换
        >>> data, report = load_and_clean(
        ...     'data.xlsx',
        ...     type_mapping={'age': 'numeric', 'date': 'datetime'}
        ... )
    """
    if not isinstance(filepath, str):
        raise ValueError("filepath必须是字符串路径")
    
    # 检查文件格式
    file_ext = filepath.split('.')[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的文件格式: {file_ext}。支持格式: {list(SUPPORTED_FORMATS.keys())}")
    
    # 创建清洗器并处理
    cleaner = create_cleaner(**kwargs)
    return cleaner.clean_data(filepath)


def get_default_config() -> dict:
    """
    获取默认配置
    
    Returns:
        默认配置字典的副本
        
    Examples:
        >>> # 获取默认配置
        >>> config = get_default_config()
        >>> print(config.keys())
    """
    return DEFAULT_CONFIG.copy()


def validate_data(data) -> dict:
    """
    快速数据验证
    
    Args:
        data: 待验证的数据
        
    Returns:
        验证结果字典
        
    Examples:
        >>> # 验证数据
        >>> result = validate_data(data)
        >>> print(f"数据形状: {result['shape']}")
        >>> print(f"缺失值: {result['missing_values']}")
        >>> print(f"重复值: {result['duplicates']}")
    """
    import pandas as pd
    import numpy as np
    
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    result = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'duplicates': int(data.duplicated().sum()),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'has_infinity': bool(np.isinf(data.select_dtypes(include=[np.number])).any().any()),
        'column_types': {
            'numeric': data.select_dtypes(include=[np.number]).columns.tolist(),
            'datetime': data.select_dtypes(include=['datetime64']).columns.tolist(),
            'categorical': data.select_dtypes(include=['category', 'object']).columns.tolist()
        }
    }
    
    return result

# ===========================
# 工具函数
# ===========================

def get_quality_level(score: float) -> str:
    """
    根据质量评分获取质量等级
    
    Args:
        score: 质量评分 (0-1)
        
    Returns:
        质量等级字符串
        
    Examples:
        >>> get_quality_level(0.95)
        'excellent'
        >>> get_quality_level(0.75) 
        'fair'
    """
    for level, (min_score, max_score) in QUALITY_SCORE_LEVELS.items():
        if min_score <= score < max_score:
            return level
    return 'unacceptable'


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小显示
    
    Args:
        size_bytes: 字节数
        
    Returns:
        格式化的大小字符串
        
    Examples:
        >>> format_file_size(1024)
        '1.0 KB'
        >>> format_file_size(1048576)
        '1.0 MB'
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_cleaning_summary(cleaner: DataCleaner) -> dict:
    """
    获取清洗摘要信息
    
    Args:
        cleaner: 数据清洗器实例
        
    Returns:
        摘要信息字典
        
    Examples:
        >>> cleaner = create_cleaner()
        >>> data, report = cleaner.clean_data('data.csv')
        >>> summary = get_cleaning_summary(cleaner)
        >>> print(summary['quality_score'])
    """
    return cleaner.get_cleaning_summary()

# ===========================
# 快速入门指南
# ===========================

"""
快速入门指南
============

1. 基础使用
-----------

>>> from T.T2 import create_cleaner, quick_clean
>>> import pandas as pd

>>> # 方法1: 快速清洗
>>> data, report = quick_clean('data.csv')

>>> # 方法2: 使用清洗器
>>> cleaner = create_cleaner()
>>> data, report = cleaner.clean_data('data.csv')

2. 自定义配置
------------

>>> # 创建自定义配置
>>> cleaner = create_cleaner(
...     missing_value_strategy=MissingValueStrategy.FILL_MEAN,
...     outlier_detection_method=OutlierDetectionMethod.ZSCORE,
...     outlier_threshold=2.5,
...     quality_threshold=0.9
... )
>>> data, report = cleaner.clean_data(data)

3. 分步清洗
----------

>>> cleaner = create_cleaner()
>>> 
>>> # 加载数据
>>> data = cleaner.load_data('data.csv')
>>> 
>>> # 评估质量
>>> quality = cleaner.assess_data_quality(data)
>>> print(f"质量评分: {quality.overall_score:.2%}")
>>> 
>>> # 执行清洗
>>> data, report = cleaner.clean_data(data)

4. 质量评估
----------

>>> from T.T2 import assess_quality

>>> # 快速质量评估
>>> report = assess_quality(data)
>>> print(f"完整性: {report.completeness:.2%}")
>>> print(f"准确性: {report.accuracy:.2%}")
>>> print(f"主要问题: {report.issues}")

5. 高级功能
----------

>>> # 异常值检测
>>> outliers = cleaner.detect_outliers(data, method=OutlierDetectionMethod.IQR)
>>> 
>>> # 处理特定列的缺失值
>>> data = cleaner.handle_missing_values(
...     data, 
...     strategy=MissingValueStrategy.FILL_MEDIAN,
...     columns=['age', 'income']
... )
>>> 
>>> # 导出清洗报告
>>> cleaner.export_cleaning_report('cleaning_report.txt')

6. 常见问题处理
--------------

>>> # 缺失值处理
>>> data = cleaner.handle_missing_values(
...     data,
...     strategy=MissingValueStrategy.INTERPOLATE_LINEAR
... )
>>> 
>>> # 重复值处理  
>>> data = cleaner.detect_and_remove_duplicates(data, keep='first')
>>> 
>>> # 异常值处理
>>> outliers = cleaner.detect_outliers(data)
>>> data = cleaner.handle_outliers(data, outliers, action='cap')

7. 最佳实践
----------

1. 总是先评估数据质量，了解问题所在
2. 根据数据特点选择合适的处理策略  
3. 清洗前后都要验证结果质量
4. 保存清洗过程和结果的详细报告
5. 定期监控数据质量变化趋势

8. 性能优化建议
--------------

1. 对于大数据集，考虑分批处理
2. 合理设置质量阈值，避免过度清洗
3. 使用合适的数据类型减少内存占用
4. 避免不必要的计算，缓存中间结果

更多详细信息请参考完整文档和示例代码。
"""

# ===========================
# 模块初始化
# ===========================

# 设置日志级别（可选）
import logging
try:
    logging.getLogger('DataCleaner').setLevel(logging.INFO)
except:
    pass

# 版本兼容性检查
try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # 检查版本兼容性
    if pd.__version__ < "1.0.0":
        import warnings
        warnings.warn(
            f"Pandas版本 {pd.__version__} 可能不完全兼容，建议使用 1.0.0+",
            UserWarning
        )
        
except ImportError as e:
    import warnings
    warnings.warn(
        f"缺少必要的依赖包: {e}。请安装: pip install pandas numpy scipy scikit-learn",
        ImportWarning
    )

# 模块加载完成标志
_T2_DATACLEANER_LOADED = True

print(f"✓ T2数据清洗器 v{__version__} 加载完成")
print(f"  支持的文件格式: {', '.join(SUPPORTED_FORMATS.keys())}")
print(f"  缺失值策略: {len(MISSING_VALUE_STRATEGIES)} 种")
print(f"  异常值检测: {len(OUTLIER_DETECTION_METHODS)} 种")
print(f"  数据质量指标: {len(DATA_QUALITY_METRICS)} 个维度")
print("  使用 help(T.T2) 查看完整文档")