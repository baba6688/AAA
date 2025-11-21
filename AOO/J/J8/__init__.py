"""
J8调试工具模块

这个模块提供了全面的调试工具集，包括代码调试、性能调试、日志调试、
错误诊断、实时监控、调试报告生成等功能。支持异步和分布式调试。

主要导出类：
- J8DebuggingTools: 主要调试工具管理器
- ExtendedJ8DebuggingTools: 扩展的调试工具管理器

便捷函数：
- get_debugger: 获取全局调试工具实例
- debug_function: 函数调试装饰器
- log_debug: 便捷日志记录函数
- set_breakpoint: 便捷断点设置函数
- start_profiling: 便捷性能分析启动函数
- stop_profiling: 便捷性能分析停止函数
- generate_debug_report: 便捷调试报告生成函数

作者: J8 Team
版本: 1.0.0
创建时间: 2025-11-06
"""

from .DebuggingTools import (
    # 主要调试工具类
    J8DebuggingTools,
    ExtendedJ8DebuggingTools,
    
    # 调试组件类
    CodeDebugger,
    PerformanceDebugger,
    LogDebugger,
    ErrorDiagnostician,
    RealtimeDebugger,
    DebugReporter,
    DistributedDebugger,
    
    # 高级调试类
    AdvancedDebugger,
    NetworkDebugger,
    DatabaseDebugger,
    MemoryDebugger,
    SecurityDebugger,
    
    # 数据类和枚举
    DebugLevel,
    ErrorType,
    DebugInfo,
    PerformanceMetrics,
    ErrorInfo,
    Breakpoint,
    
    # 工具函数和装饰器
    get_debugger,
    debug_function,
    debug_async_function,
    debug_class_methods,
    performance_benchmark,
    memory_profiler,
    log_debug,
    set_breakpoint,
    start_profiling,
    stop_profiling,
    generate_debug_report
)

# 模块版本信息
__version__ = "1.0.0"
__author__ = "J8 Team"

# 默认调试工具实例
_default_debugger = None

def get_default_debugger():
    """获取默认调试工具实例"""
    global _default_debugger
    if _default_debugger is None:
        _default_debugger = ExtendedJ8DebuggingTools()
    return _default_debugger

# 便捷别名
debugger = get_default_debugger
