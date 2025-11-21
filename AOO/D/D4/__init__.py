"""
D4错误诊断器模块
提供智能错误诊断、分析、预防和恢复功能
"""

from .ErrorDiagnoser import (
    ErrorDiagnoser,
    ErrorInfo,
    ErrorAnalysis,
    ErrorPatternInfo,
    ErrorLevel,
    ErrorCategory,
    ErrorPattern
)

__version__ = "1.0.0"
__author__ = "D4 Error Diagnoser Team"

__all__ = [
    "ErrorDiagnoser",
    "ErrorInfo", 
    "ErrorAnalysis",
    "ErrorPatternInfo",
    "ErrorLevel",
    "ErrorCategory",
    "ErrorPattern"
]