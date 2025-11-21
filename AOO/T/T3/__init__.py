"""
T3数据验证器模块

这是一个全面的数据验证器，提供以下功能：
- 数据完整性验证
- 数据类型验证  
- 数据范围验证
- 数据格式验证
- 业务规则验证
- 数据依赖性验证
- 数据一致性验证
- 数据质量评分
- 验证报告生成

快速开始:
    from T.T3 import DataValidator, ValidationRule, ValidationLevel, DataType
    
    # 创建验证器
    validator = DataValidator()
    
    # 添加验证规则
    validator.add_validation_rule(ValidationRule(
        field_name="email",
        rule_type="type", 
        parameters={"type": DataType.EMAIL},
        level=ValidationLevel.HIGH
    ))
    
    # 验证数据
    data = {"email": "test@example.com"}
    report = validator.validate_data(data)

作者: T3系统
版本: 1.0.0
创建日期: 2025-11-05
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "T3系统"
__creation_date__ = "2025-11-05"
__description__ = "T3数据验证器 - 全面的数据验证和质量评估工具"

# 核心类导出
from .DataValidator import (
    ValidationLevel,
    DataType, 
    ValidationRule,
    ValidationResult,
    DataQualityScore,
    ValidationReport,
    DataValidator
)

# 便捷函数导出
from .DataValidator import (
    create_basic_validator,
    validate_user_data,
    validate_financial_data
)

# 默认配置
DEFAULT_CONFIG = {
    "strict_mode": False,
    "enable_logging": True,
    "quality_weights": {
        'completeness': 0.2,
        'accuracy': 0.25, 
        'consistency': 0.2,
        'validity': 0.25,
        'uniqueness': 0.05,
        'timeliness': 0.05
    },
    "score_thresholds": {
        "excellent": 90,
        "good": 80,
        "fair": 70,
        "poor": 60
    }
}

# 常用验证规则模板
COMMON_RULES = {
    "email_required": ValidationRule(
        field_name="email",
        rule_type="required",
        level=ValidationLevel.CRITICAL,
        error_message="邮箱为必填项"
    ),
    
    "email_format": ValidationRule(
        field_name="email", 
        rule_type="type",
        parameters={"type": DataType.EMAIL},
        level=ValidationLevel.HIGH,
        error_message="邮箱格式不正确"
    ),
    
    "phone_format": ValidationRule(
        field_name="phone",
        rule_type="type", 
        parameters={"type": DataType.PHONE},
        level=ValidationLevel.MEDIUM,
        error_message="手机号格式不正确"
    ),
    
    "age_range": ValidationRule(
        field_name="age",
        rule_type="range",
        parameters={"min": 0, "max": 150},
        level=ValidationLevel.MEDIUM,
        error_message="年龄必须在0-150之间"
    ),
    
    "name_length": ValidationRule(
        field_name="name",
        rule_type="range", 
        parameters={"min": 2, "max": 50},
        level=ValidationLevel.MEDIUM,
        error_message="姓名长度必须在2-50个字符之间"
    )
}

# 业务特定验证规则模板
BUSINESS_RULES = {
    # 用户数据验证
    "user_validation": [
        COMMON_RULES["email_required"],
        COMMON_RULES["email_format"],
        COMMON_RULES["phone_format"],
        COMMON_RULES["age_range"],
        ValidationRule(
            field_name="name",
            rule_type="required",
            level=ValidationLevel.HIGH,
            error_message="姓名为必填项"
        )
    ],
    
    # 金融数据验证
    "financial_validation": [
        ValidationRule(
            field_name="amount",
            rule_type="required",
            level=ValidationLevel.CRITICAL,
            error_message="金额为必填项"
        ),
        ValidationRule(
            field_name="amount",
            rule_type="range",
            parameters={"min": 0},
            level=ValidationLevel.CRITICAL,
            error_message="金额不能为负数"
        ),
        ValidationRule(
            field_name="currency",
            rule_type="business_rule",
            parameters={"expression": "value in ['USD', 'EUR', 'CNY', 'JPY', 'GBP']"},
            level=ValidationLevel.HIGH,
            error_message="不支持的货币类型"
        ),
        ValidationRule(
            field_name="date",
            rule_type="format",
            parameters={"format": "date"},
            level=ValidationLevel.HIGH,
            error_message="日期格式不正确"
        )
    ],
    
    # 产品数据验证
    "product_validation": [
        ValidationRule(
            field_name="name",
            rule_type="required",
            level=ValidationLevel.CRITICAL,
            error_message="产品名称为必填项"
        ),
        ValidationRule(
            field_name="price",
            rule_type="required",
            level=ValidationLevel.CRITICAL,
            error_message="产品价格为必填项"
        ),
        ValidationRule(
            field_name="price",
            rule_type="range",
            parameters={"min": 0},
            level=ValidationLevel.HIGH,
            error_message="产品价格不能为负数"
        ),
        ValidationRule(
            field_name="sku",
            rule_type="format",
            parameters={"pattern": r"^[A-Z0-9]{8,12}$"},
            level=ValidationLevel.MEDIUM,
            error_message="SKU格式不正确"
        )
    ]
}

# 数据质量标准
QUALITY_STANDARDS = {
    "excellent": {
        "min_score": 90,
        "description": "数据质量优秀，可直接使用"
    },
    "good": {
        "min_score": 80,
        "description": "数据质量良好，略有瑕疵"
    },
    "fair": {
        "min_score": 70,
        "description": "数据质量一般，需要改进"
    },
    "poor": {
        "min_score": 0,
        "description": "数据质量较差，需要大量修复"
    }
}

# 便利函数
def create_validator(config: dict = None) -> DataValidator:
    """
    创建数据验证器
    
    Args:
        config: 配置字典，如果为None则使用默认配置
        
    Returns:
        DataValidator实例
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    return DataValidator(
        strict_mode=config.get("strict_mode", False),
        enable_logging=config.get("enable_logging", True)
    )

def create_user_validator() -> DataValidator:
    """创建用户数据验证器"""
    validator = create_validator()
    
    # 添加用户验证规则
    for rule in BUSINESS_RULES["user_validation"]:
        validator.add_validation_rule(rule)
    
    return validator

def create_financial_validator() -> DataValidator:
    """创建金融数据验证器"""
    validator = create_validator()
    
    # 添加金融验证规则
    for rule in BUSINESS_RULES["financial_validation"]:
        validator.add_validation_rule(rule)
    
    return validator

def create_product_validator() -> DataValidator:
    """创建产品数据验证器"""
    validator = create_validator()
    
    # 添加产品验证规则
    for rule in BUSINESS_RULES["product_validation"]:
        validator.add_validation_rule(rule)
    
    return validator

def quick_validate(data: dict, validation_type: str = "basic") -> ValidationReport:
    """
    快速验证数据
    
    Args:
        data: 要验证的数据
        validation_type: 验证类型 ("basic", "user", "financial", "product")
        
    Returns:
        验证报告
    """
    if validation_type == "basic":
        validator = create_basic_validator()
    elif validation_type == "user":
        validator = create_user_validator()
    elif validation_type == "financial":
        validator = create_financial_validator()
    elif validation_type == "product":
        validator = create_product_validator()
    else:
        raise ValueError(f"不支持的验证类型: {validation_type}")
    
    return validator.validate_data(data)

def get_quality_level(score: float) -> str:
    """
    根据评分获取质量等级
    
    Args:
        score: 质量评分 (0-100)
        
    Returns:
        质量等级字符串
    """
    for level, standards in QUALITY_STANDARDS.items():
        if score >= standards["min_score"]:
            return level
    return "poor"

def get_quality_description(score: float) -> str:
    """
    根据评分获取质量描述
    
    Args:
        score: 质量评分 (0-100)
        
    Returns:
        质量描述字符串
    """
    level = get_quality_level(score)
    return QUALITY_STANDARDS[level]["description"]

def batch_validate(data_list: list, validation_type: str = "basic") -> list:
    """
    批量验证数据
    
    Args:
        data_list: 数据列表
        validation_type: 验证类型
        
    Returns:
        验证报告列表
    """
    reports = []
    for data in data_list:
        report = quick_validate(data, validation_type)
        reports.append(report)
    return reports

def create_custom_rule(field_name: str, rule_type: str, **kwargs) -> ValidationRule:
    """
    创建自定义验证规则
    
    Args:
        field_name: 字段名
        rule_type: 规则类型
        **kwargs: 其他参数
        
    Returns:
        ValidationRule实例
    """
    return ValidationRule(
        field_name=field_name,
        rule_type=rule_type,
        **kwargs
    )

# 工具函数
def format_report_summary(report: ValidationReport) -> str:
    """
    格式化报告摘要
    
    Args:
        report: 验证报告
        
    Returns:
        格式化的摘要字符串
    """
    quality_level = get_quality_level(report.quality_score.overall_score)
    quality_desc = get_quality_description(report.quality_score.overall_score)
    
    summary = f"""
数据验证报告摘要
================
报告ID: {report.report_id}
生成时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

验证统计:
- 总记录数: {report.total_records}
- 有效记录: {report.valid_records} ({report.valid_records/report.total_records*100:.1f}%)
- 无效记录: {report.invalid_records} ({report.invalid_records/report.total_records*100:.1f}%)

数据质量评分:
- 总体评分: {report.quality_score.overall_score} ({quality_level})
- 质量描述: {quality_desc}
- 完整性: {report.quality_score.completeness_score}
- 准确性: {report.quality_score.accuracy_score}
- 一致性: {report.quality_score.consistency_score}
- 有效性: {report.quality_score.validity_score}

主要问题:
"""
    
    if report.summary.get('main_issues'):
        for issue in report.summary['main_issues']:
            summary += f"- {issue}\n"
    else:
        summary += "- 无主要问题\n"
    
    summary += "\n改进建议:\n"
    if report.recommendations:
        for rec in report.recommendations:
            summary += f"- {rec}\n"
    else:
        summary += "- 无特殊建议\n"
    
    return summary

def get_validation_stats(report: ValidationReport) -> dict:
    """
    获取验证统计信息
    
    Args:
        report: 验证报告
        
    Returns:
        统计信息字典
    """
    return {
        "total_validations": report.total_records,
        "valid_count": report.valid_records,
        "invalid_count": report.invalid_records,
        "success_rate": round(report.valid_records / report.total_records * 100, 2) if report.total_records > 0 else 0,
        "overall_score": report.quality_score.overall_score,
        "quality_level": get_quality_level(report.quality_score.overall_score),
        "level_distribution": report.summary.get('level_distribution', {}),
        "rule_type_distribution": report.summary.get('rule_type_distribution', {})
    }

# 导出所有公共接口
__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__creation_date__",
    "__description__",
    
    # 核心类
    "ValidationLevel",
    "DataType",
    "ValidationRule", 
    "ValidationResult",
    "DataQualityScore",
    "ValidationReport",
    "DataValidator",
    
    # 便捷函数
    "create_basic_validator",
    "validate_user_data",
    "validate_financial_data",
    "create_validator",
    "create_user_validator", 
    "create_financial_validator",
    "create_product_validator",
    "quick_validate",
    "batch_validate",
    "create_custom_rule",
    
    # 工具函数
    "format_report_summary",
    "get_validation_stats",
    "get_quality_level",
    "get_quality_description",
    
    # 常量
    "DEFAULT_CONFIG",
    "COMMON_RULES", 
    "BUSINESS_RULES",
    "QUALITY_STANDARDS"
]

# 模块文档
"""
快速入门指南
============

1. 基础使用
-----------
from T.T3 import DataValidator, ValidationRule, ValidationLevel

# 创建验证器
validator = DataValidator()

# 添加验证规则
validator.add_validation_rule(ValidationRule(
    field_name="email",
    rule_type="type",
    parameters={"type": DataType.EMAIL},
    level=ValidationLevel.HIGH
))

# 验证数据
data = {"email": "test@example.com"}
report = validator.validate_data(data)

2. 使用便捷函数
--------------
from T.T3 import quick_validate, create_user_validator

# 快速验证
report = quick_validate({"email": "test@example.com"}, "user")

# 使用专用验证器
validator = create_user_validator()
report = validator.validate_data(user_data)

3. 自定义验证规则
----------------
from T.T3 import create_custom_rule, DataType

# 创建自定义规则
rule = create_custom_rule(
    field_name="username",
    rule_type="range",
    parameters={"min": 3, "max": 20},
    level=ValidationLevel.MEDIUM,
    error_message="用户名长度必须在3-20个字符之间"
)

4. 数据质量评估
--------------
from T.T3 import get_quality_level, format_report_summary

# 获取质量等级
quality_level = get_quality_level(report.quality_score.overall_score)

# 格式化报告摘要
summary = format_report_summary(report)
print(summary)

5. 批量验证
-----------
from T.T3 import batch_validate

# 批量验证
data_list = [data1, data2, data3]
reports = batch_validate(data_list, "user")

支持的验证规则类型
===============
- "required": 必填字段验证
- "type": 数据类型验证
- "range": 数据范围验证  
- "format": 数据格式验证
- "business_rule": 业务规则验证
- "dependency": 数据依赖性验证
- "consistency": 数据一致性验证

支持的验证级别
=============
- CRITICAL: 关键级别，验证失败将阻止数据处理
- HIGH: 高级别，严重问题
- MEDIUM: 中级别，一般问题  
- LOW: 低级别，轻微问题
- INFO: 信息级别，仅供参考

支持的数据类型
=============
- STRING: 字符串
- INTEGER: 整数
- FLOAT: 浮点数
- BOOLEAN: 布尔值
- DATE: 日期
- DATETIME: 日期时间
- EMAIL: 邮箱
- PHONE: 手机号
- URL: URL地址
- JSON: JSON格式
- LIST: 列表
- DICT: 字典

更多详细信息请参考 DataValidator 类的完整文档。
"""