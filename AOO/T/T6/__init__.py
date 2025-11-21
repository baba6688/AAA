#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T6数据导出器 - 完整导出接口

一个高性能、多格式的数据导出工具，支持多种数据格式导出、自动化调度、
流式导出、压缩加密、性能优化等功能。

版本信息:
    Version: 1.0.0
    Author: T6模块开发团队
    Date: 2025-11-05

功能特性:
- 多格式数据导出 (CSV, Excel, JSON, XML, Parquet, HDF5等)
- 数据导出调度和自动化
- 数据分批导出和流式导出
- 数据导出压缩和加密
- 数据导出性能优化
- 数据导出错误处理
- 数据导出进度监控
- 数据导出安全控制
- 数据导出日志和审计

快速入门:
    >>> from T.T6 import DataExporter, ExportConfig
    >>> 
    >>> # 创建基本配置
    >>> config = ExportConfig(format='csv', enable_compression=True)
    >>> 
    >>> # 初始化导出器
    >>> exporter = DataExporter(config)
    >>> 
    >>> # 准备数据
    >>> data = [{"id": 1, "name": "张三", "age": 25}]
    >>> 
    >>> # 导出数据
    >>> result = exporter.export_data(data, 'output.csv')
    >>> 
    >>> # 检查结果
    >>> if result.success:
    ...     print(f"导出成功: {result.file_path}")
    
日期: 2025-11-13
版本: 1.0.0
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "T6模块开发团队"
__email__ = "t6-team@company.com"
__license__ = "MIT"

# 项目信息
PROJECT_NAME = "T6数据导出器"
PROJECT_DESCRIPTION = "高性能、多格式的数据导出工具"
PROJECT_URL = "https://github.com/company/T6-DataExporter"
RELEASE_DATE = "2025-11-05"

# 导出所有主要类和函数
from .DataExporter import (
    # 主要类
    DataExporter,
    ExportConfig,
    ExportResult,
    
    # 监控和管理类
    ProgressMonitor,
    SecurityManager,
    PerformanceMonitor,
    CompressionManager,
    EncryptionManager,
    
    # 格式处理器
    FormatHandler,
    CSVHandler,
    ExcelHandler,
    JSONHandler,
    XMLHandler,
    ParquetHandler,
    HDF5Handler,
    
    # 便利函数
    create_sample_data,
    data_generator,
    progress_callback,
    retry_on_failure,
    
    # 测试函数
    test_basic_export,
    test_batch_export,
    test_streaming_export,
    test_scheduled_export,
    test_security_features,
    main
)

# 默认配置
DEFAULT_CONFIG = ExportConfig(
    format="csv",
    compression=None,
    encryption_key=None,
    encoding="utf-8",
    batch_size=10000,
    enable_streaming=False,
    chunk_size=1000,
    max_workers=4,
    enable_compression=False,
    memory_limit_mb=512,
    schedule_type=None,
    schedule_time=None,
    schedule_cron=None,
    enable_audit=True,
    enable_access_control=True,
    allowed_users=[],
    enable_progress_monitor=True,
    progress_callback=None,
    enable_performance_monitor=True,
    log_level="INFO",
    log_file=None,
    enable_audit_log=True
)

# 支持的导出格式常量
SUPPORTED_FORMATS = {
    "csv": "逗号分隔值文件",
    "tsv": "制表符分隔值文件", 
    "excel": "Excel工作簿",
    "json": "JSON数据交换格式",
    "xml": "XML可扩展标记语言",
    "parquet": "Apache Parquet列式存储格式",
    "hdf5": "HDF5层次化数据格式"
}

# 支持的压缩格式
SUPPORTED_COMPRESSION = {
    "gzip": "GZIP压缩",
    "bz2": "BZ2压缩",
    "zip": "ZIP压缩"
}

# 日志级别
LOG_LEVELS = {
    "DEBUG": "调试信息",
    "INFO": "一般信息", 
    "WARNING": "警告信息",
    "ERROR": "错误信息",
    "CRITICAL": "严重错误"
}

# 调度类型
SCHEDULE_TYPES = {
    "daily": "每日调度",
    "weekly": "每周调度",
    "monthly": "每月调度",
    "cron": "Cron表达式调度"
}

# 文件编码
SUPPORTED_ENCODINGS = [
    "utf-8", "utf-8-sig", "gbk", "gb2312", "big5", 
    "ascii", "latin-1", "cp1252", "shift_jis"
]

# 便利函数

def create_exporter(config=None, **kwargs):
    """
    创建数据导出器的便利函数
    
    Args:
        config: 导出配置对象，如果为None则使用默认配置
        **kwargs: 配置参数，将覆盖默认配置
    
    Returns:
        DataExporter: 数据导出器实例
    """
    if config is None:
        config = ExportConfig(**kwargs)
    
    return DataExporter(config)


def quick_export(data, file_path, format=None, **kwargs):
    """
    快速导出数据的便利函数
    
    Args:
        data: 要导出的数据
        file_path: 输出文件路径
        format: 导出格式，如果为None则从文件扩展名推断
        **kwargs: 其他配置参数
    
    Returns:
        ExportResult: 导出结果
    """
    config = ExportConfig(**kwargs)
    exporter = DataExporter(config)
    
    with exporter:
        return exporter.export_data(data, file_path, format)


def batch_quick_export(data_list, output_dir, format="csv", **kwargs):
    """
    批量快速导出数据的便利函数
    
    Args:
        data_list: 数据列表
        output_dir: 输出目录
        format: 导出格式
        **kwargs: 其他配置参数
    
    Returns:
        List[ExportResult]: 导出结果列表
    """
    config = ExportConfig(**kwargs)
    exporter = DataExporter(config)
    
    with exporter:
        return exporter.export_data_batch(data_list, output_dir, format)


def export_csv(data, file_path, **kwargs):
    """
    快速导出CSV格式数据的便利函数
    
    Args:
        data: 要导出的数据
        file_path: 输出文件路径
        **kwargs: 其他配置参数
    
    Returns:
        ExportResult: 导出结果
    """
    return quick_export(data, file_path, format="csv", **kwargs)


def export_json(data, file_path, **kwargs):
    """
    快速导出JSON格式数据的便利函数
    
    Args:
        data: 要导出的数据
        file_path: 输出文件路径
        **kwargs: 其他配置参数
    
    Returns:
        ExportResult: 导出结果
    """
    return quick_export(data, file_path, format="json", **kwargs)


def export_excel(data, file_path, **kwargs):
    """
    快速导出Excel格式数据的便利函数
    
    Args:
        data: 要导出的数据
        file_path: 输出文件路径
        **kwargs: 其他配置参数
    
    Returns:
        ExportResult: 导出结果
    """
    return quick_export(data, file_path, format="excel", **kwargs)


def export_xml(data, file_path, **kwargs):
    """
    快速导出XML格式数据的便利函数
    
    Args:
        data: 要导出的数据
        file_path: 输出文件路径
        **kwargs: 其他配置参数
    
    Returns:
        ExportResult: 导出结果
    """
    return quick_export(data, file_path, format="xml", **kwargs)


def get_version_info():
    """
    获取版本信息
    
    Returns:
        dict: 版本信息字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "project": PROJECT_NAME,
        "description": PROJECT_DESCRIPTION,
        "url": PROJECT_URL,
        "release_date": RELEASE_DATE
    }


def check_dependencies():
    """
    检查依赖库是否可用
    
    Returns:
        dict: 依赖库检查结果
    """
    dependencies = {}
    
    try:
        import pandas as pd
        dependencies["pandas"] = {"available": True, "version": pd.__version__}
    except ImportError:
        dependencies["pandas"] = {"available": False, "version": None}
    
    try:
        import openpyxl
        dependencies["openpyxl"] = {"available": True, "version": openpyxl.__version__}
    except ImportError:
        dependencies["openpyxl"] = {"available": False, "version": None}
    
    try:
        import pyarrow
        dependencies["pyarrow"] = {"available": True, "version": pyarrow.__version__}
    except ImportError:
        dependencies["pyarrow"] = {"available": False, "version": None}
    
    try:
        import h5py
        dependencies["h5py"] = {"available": True, "version": h5py.__version__}
    except ImportError:
        dependencies["h5py"] = {"available": False, "version": None}
    
    try:
        from cryptography.fernet import Fernet
        dependencies["cryptography"] = {"available": True, "version": "unknown"}
    except ImportError:
        dependencies["cryptography"] = {"available": False, "version": None}
    
    try:
        import psutil
        dependencies["psutil"] = {"available": True, "version": psutil.__version__}
    except ImportError:
        dependencies["psutil"] = {"available": False, "version": None}
    
    return dependencies


def print_documentation():
    """
    打印使用文档
    """
    doc = f"""
{PROJECT_NAME} v{__version__}
{PROJECT_DESCRIPTION}

使用文档:
========

1. 基本使用
-----------
from T.T6 import DataExporter, ExportConfig

# 创建配置
config = ExportConfig(
    format='csv',
    enable_compression=True,
    compression='gzip'
)

# 初始化导出器
exporter = DataExporter(config)

# 导出数据
data = [{{"id": 1, "name": "张三", "age": 25}}]
result = exporter.export_data(data, 'output.csv')

2. 便利函数
-----------
from T.T6 import quick_export, export_csv, export_json

# 快速导出
result = quick_export(data, 'output.csv', format='csv')

# 直接导出特定格式
result = export_csv(data, 'output.csv')
result = export_json(data, 'output.json')

3. 批量导出
-----------
from T.T6 import batch_quick_export

data_list = [data1, data2, data3]
results = batch_quick_export(data_list, '/tmp/output/', format='csv')

4. 流式导出
-----------
from T.T6 import data_generator

def my_data_generator():
    for i in range(10000):
        yield {{"id": i, "value": f"数据{{i}}"}}

# 配置流式导出
config = ExportConfig(
    enable_streaming=True,
    chunk_size=1000,
    enable_progress_monitor=True
)

exporter = DataExporter(config)
result = exporter.export_data_streaming(
    my_data_generator(),
    'large_output.csv'
)

5. 调度导出
-----------
config = ExportConfig(
    schedule_type='daily',
    schedule_time='09:00'
)

exporter = DataExporter(config)
exporter.schedule_export(
    data_source=get_daily_data,
    file_path='/backup/daily_export.csv'
)

支持的格式:
----------
{chr(10).join([f"  {fmt}: {desc}" for fmt, desc in SUPPORTED_FORMATS.items()])}

支持的压缩格式:
-------------
{chr(10).join([f"  {fmt}: {desc}" for fmt, desc in SUPPORTED_COMPRESSION.items()])}

依赖库检查:
----------
{chr(10).join([f"  {name}: {'✓' if info['available'] else '✗'} {info['version'] or ''}" for name, info in check_dependencies().items()])}

更多详细信息请参考官方文档: {PROJECT_URL}
    """
    print(doc)


def demo():
    """
    运行演示程序
    """
    print(f"{PROJECT_NAME} v{__version__} 演示程序")
    print("=" * 50)
    
    # 检查依赖
    deps = check_dependencies()
    print("依赖库检查:")
    for name, info in deps.items():
        status = "✓" if info["available"] else "✗"
        version = info["version"] or "N/A"
        print(f"  {name}: {status} {version}")
    print()
    
    # 创建示例数据
    print("1. 创建示例数据...")
    data = create_sample_data()
    print(f"   数据量: {len(data)} 条记录")
    print(f"   示例: {data[0] if data else '无数据'}")
    print()
    
    # 测试各种格式导出
    print("2. 测试各种格式导出...")
    
    formats_to_test = ["csv", "json"]
    if deps.get("openpyxl", {}).get("available"):
        formats_to_test.append("excel")
    
    for fmt in formats_to_test:
        print(f"   测试 {fmt.upper()} 格式...")
        result = quick_export(
            data, 
            f'/tmp/demo.{fmt.replace("excel", "xlsx")}', 
            format=fmt,
            log_level="WARNING"  # 减少输出
        )
        
        if result.success:
            print(f"   ✓ {fmt.upper()} 导出成功")
            print(f"     文件: {result.file_path}")
            print(f"     大小: {result.file_size} bytes")
            print(f"     耗时: {result.export_time:.2f}s")
        else:
            print(f"   ✗ {fmt.upper()} 导出失败: {result.error_message}")
        print()
    
    # 测试批量导出
    print("3. 测试批量导出...")
    data_list = [data[:3], data[3:5], data[5:]]  # 分成3个小数据集
    results = batch_quick_export(
        data_list, 
        '/tmp/demo_batch/', 
        format='csv',
        log_level="WARNING"
    )
    
    success_count = sum(1 for r in results if r.success)
    print(f"   批量导出结果: {success_count}/{len(results)} 成功")
    print()
    
    print("演示程序完成!")
    print("要查看完整文档，请调用: print_documentation()")


# 模块级别的便利配置
# 快速导入常用配置
QUICK_CONFIGS = {
    "basic": ExportConfig(format="csv"),
    "compressed": ExportConfig(format="csv", enable_compression=True, compression="gzip"),
    "secure": ExportConfig(enable_audit=True, enable_access_control=True),
    "fast": ExportConfig(max_workers=8, batch_size=50000),
    "streaming": ExportConfig(enable_streaming=True, chunk_size=5000),
    "full": ExportConfig(
        format="csv",
        enable_compression=True,
        compression="gzip",
        enable_audit=True,
        enable_progress_monitor=True,
        log_level="INFO"
    )
}


def get_quick_config(config_name="basic"):
    """
    获取快速配置
    
    Args:
        config_name: 配置名称，可选值: basic, compressed, secure, fast, streaming, full
    
    Returns:
        ExportConfig: 配置对象
    """
    if config_name not in QUICK_CONFIGS:
        raise ValueError(f"未知的快速配置: {config_name}，可用选项: {list(QUICK_CONFIGS.keys())}")
    
    return QUICK_CONFIGS[config_name]


# 错误类定义
class T6ExportError(Exception):
    """T6导出器基础异常类"""
    pass


class FormatNotSupportedError(T6ExportError):
    """不支持的格式异常"""
    pass


class CompressionError(T6ExportError):
    """压缩相关异常"""
    pass


class EncryptionError(T6ExportError):
    """加密相关异常"""
    pass


class SecurityError(T6ExportError):
    """安全相关异常"""
    pass


class ValidationError(T6ExportError):
    """数据验证异常"""
    pass


# 导出所有可用的类和函数到模块级别
__all__ = [
    # 主要类
    "DataExporter",
    "ExportConfig", 
    "ExportResult",
    
    # 监控管理类
    "ProgressMonitor",
    "SecurityManager", 
    "PerformanceMonitor",
    "CompressionManager",
    "EncryptionManager",
    
    # 格式处理器
    "FormatHandler",
    "CSVHandler",
    "ExcelHandler", 
    "JSONHandler",
    "XMLHandler",
    "ParquetHandler",
    "HDF5Handler",
    
    # 便利函数
    "create_exporter",
    "quick_export",
    "batch_quick_export", 
    "export_csv",
    "export_json",
    "export_excel",
    "export_xml",
    "get_version_info",
    "check_dependencies",
    "print_documentation",
    "demo",
    "get_quick_config",
    
    # 测试和示例函数
    "create_sample_data",
    "data_generator", 
    "progress_callback",
    "retry_on_failure",
    "test_basic_export",
    "test_batch_export",
    "test_streaming_export",
    "test_scheduled_export",
    "test_security_features",
    "main",
    
    # 常量和配置
    "DEFAULT_CONFIG",
    "SUPPORTED_FORMATS",
    "SUPPORTED_COMPRESSION", 
    "SCHEDULE_TYPES",
    "LOG_LEVELS",
    "SUPPORTED_ENCODINGS",
    "QUICK_CONFIGS",
    
    # 异常类
    "T6ExportError",
    "FormatNotSupportedError",
    "CompressionError", 
    "EncryptionError",
    "SecurityError",
    "ValidationError"
]


# 模块初始化时的日志配置
import logging

# 创建模块级日志器
module_logger = logging.getLogger("T.T6")
module_logger.setLevel(logging.INFO)

# 添加默认控制台处理器（如果没有处理器的话）
if not module_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - T.T6 - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    module_logger.addHandler(console_handler)


def _initialize_module():
    """
    模块初始化时的设置
    """
    # 检查Python版本
    import sys
    if sys.version_info < (3, 7):
        module_logger.warning(f"建议使用Python 3.7或更高版本，当前版本: {sys.version}")
    
    # 检查关键依赖
    missing_critical = []
    try:
        import csv
    except ImportError:
        missing_critical.append("csv")
    
    try:
        import json
    except ImportError:
        missing_critical.append("json")
    
    if missing_critical:
        module_logger.error(f"缺少关键依赖: {', '.join(missing_critical)}")
    
    # 记录模块初始化完成
    module_logger.info(f"{PROJECT_NAME} v{__version__} 初始化完成")


# 执行模块初始化
_initialize_module()

# 在模块导入时打印欢迎信息（可选）
# module_logger.info(f"欢迎使用 {PROJECT_NAME} v{__version__} - {PROJECT_DESCRIPTION}")
# module_logger.info("输入 help(T.T6) 或 T.T6.print_documentation() 查看使用文档")