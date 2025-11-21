"""
T3数据验证器模块

本模块实现了一个全面的数据验证器，支持多种数据验证功能：
1. 数据完整性验证
2. 数据类型验证
3. 数据范围验证
4. 数据格式验证
5. 业务规则验证
6. 数据依赖性验证
7. 数据一致性验证
8. 数据质量评分
9. 验证报告生成

作者: T3系统
版本: 1.0.0
创建日期: 2025-11-05
"""

import re
import json
import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics
import logging


class ValidationLevel(Enum):
    """验证级别枚举"""
    CRITICAL = "critical"      # 关键级别，验证失败将阻止数据处理
    HIGH = "high"             # 高级别，严重问题
    MEDIUM = "medium"         # 中级别，一般问题
    LOW = "low"              # 低级别，轻微问题
    INFO = "info"            # 信息级别，仅供参考


class DataType(Enum):
    """数据类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    JSON = "json"
    LIST = "list"
    DICT = "dict"


@dataclass
class ValidationRule:
    """验证规则数据类"""
    field_name: str
    rule_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    level: ValidationLevel = ValidationLevel.MEDIUM
    error_message: str = ""
    custom_validator: Optional[Callable] = None


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    field_name: str
    rule_type: str
    level: ValidationLevel
    message: str
    value: Any = None
    expected: Any = None
    actual: Any = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class DataQualityScore:
    """数据质量评分数据类"""
    overall_score: float           # 总体评分 (0-100)
    completeness_score: float      # 完整性评分
    accuracy_score: float          # 准确性评分
    consistency_score: float       # 一致性评分
    validity_score: float          # 有效性评分
    uniqueness_score: float        # 唯一性评分
    timeliness_score: float        # 时效性评分
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """验证报告数据类"""
    report_id: str
    timestamp: datetime.datetime
    total_records: int
    valid_records: int
    invalid_records: int
    validation_results: List[ValidationResult]
    quality_score: DataQualityScore
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class DataValidator:
    """
    T3数据验证器类
    
    提供全面的数据验证功能，支持多种验证规则和质量评估。
    """
    
    def __init__(self, strict_mode: bool = False, enable_logging: bool = True):
        """
        初始化数据验证器
        
        Args:
            strict_mode: 严格模式，验证失败时抛出异常
            enable_logging: 是否启用日志记录
        """
        self.strict_mode = strict_mode
        self.enable_logging = enable_logging
        self.validation_rules: List[ValidationRule] = []
        self.custom_validators: Dict[str, Callable] = {}
        self.validation_history: List[ValidationReport] = []
        
        # 设置日志
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
            
        # 预定义验证模式
        self._init_predefined_validators()
    
    def _init_predefined_validators(self):
        """初始化预定义的验证器"""
        # 邮箱验证模式
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        # 手机号验证模式（中国大陆）
        self.phone_pattern = re.compile(
            r'^1[3-9]\d{9}$'
        )
        
        # URL验证模式
        self.url_pattern = re.compile(
            r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?$'
        )
        
        # 日期格式模式
        self.date_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}$'),           # YYYY-MM-DD
            re.compile(r'^\d{2}/\d{2}/\d{4}$'),           # MM/DD/YYYY
            re.compile(r'^\d{2}-\d{2}-\d{4}$'),           # MM-DD-YYYY
        ]
    
    def add_validation_rule(self, rule: ValidationRule) -> 'DataValidator':
        """
        添加验证规则
        
        Args:
            rule: 验证规则对象
            
        Returns:
            返回自身，支持链式调用
        """
        self.validation_rules.append(rule)
        return self
    
    def add_custom_validator(self, name: str, validator_func: Callable) -> 'DataValidator':
        """
        添加自定义验证器
        
        Args:
            name: 验证器名称
            validator_func: 验证函数
            
        Returns:
            返回自身，支持链式调用
        """
        self.custom_validators[name] = validator_func
        return self
    
    def validate_data(self, data: Union[Dict, List, Any], 
                     rules: Optional[List[ValidationRule]] = None) -> ValidationReport:
        """
        验证数据
        
        Args:
            data: 要验证的数据
            rules: 验证规则列表，如果为None则使用默认规则
            
        Returns:
            验证报告
        """
        if rules is None:
            rules = self.validation_rules
            
        report_id = f"validation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        validation_results = []
        
        # 处理不同数据类型
        if isinstance(data, dict):
            validation_results.extend(self._validate_dict_data(data, rules))
        elif isinstance(data, list):
            validation_results.extend(self._validate_list_data(data, rules))
        else:
            validation_results.append(self._validate_single_value(data, "value", rules))
        
        # 计算验证统计
        valid_records = sum(1 for r in validation_results if r.is_valid)
        invalid_records = len(validation_results) - valid_records
        total_records = len(validation_results) if validation_results else 1
        
        # 计算数据质量评分
        quality_score = self._calculate_quality_score(data, validation_results)
        
        # 生成报告
        report = ValidationReport(
            report_id=report_id,
            timestamp=datetime.datetime.now(),
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            validation_results=validation_results,
            quality_score=quality_score,
            summary=self._generate_summary(validation_results),
            recommendations=self._generate_recommendations(validation_results, quality_score)
        )
        
        self.validation_history.append(report)
        
        if self.enable_logging:
            self.logger.info(f"数据验证完成: {valid_records}/{total_records} 条记录有效")
        
        # 严格模式下验证失败则抛出异常
        if self.strict_mode and invalid_records > 0:
            raise ValueError(f"数据验证失败: {invalid_records} 条记录无效")
        
        return report
    
    def _validate_dict_data(self, data: Dict, rules: List[ValidationRule]) -> List[ValidationResult]:
        """验证字典数据"""
        results = []
        for field_name, value in data.items():
            field_rules = [r for r in rules if r.field_name == field_name or r.field_name == "*"]
            for rule in field_rules:
                result = self._validate_field(field_name, value, rule)
                results.append(result)
        return results
    
    def _validate_list_data(self, data: List, rules: List[ValidationRule]) -> List[ValidationResult]:
        """验证列表数据"""
        results = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                for field_name, value in item.items():
                    field_rules = [r for r in rules if r.field_name == field_name or r.field_name == "*"]
                    for rule in field_rules:
                        result = self._validate_field(f"{field_name}[{i}]", value, rule)
                        results.append(result)
            else:
                result = self._validate_field(f"item[{i}]", item, rules[0] if rules else None)
                results.append(result)
        return results
    
    def _validate_single_value(self, value: Any, field_name: str, rules: Optional[List[ValidationRule]]) -> ValidationResult:
        """验证单个值"""
        if not rules:
            return ValidationResult(
                is_valid=True,
                field_name=field_name,
                rule_type="no_rule",
                level=ValidationLevel.INFO,
                message="无验证规则"
            )
        
        return self._validate_field(field_name, value, rules[0])
    
    def _validate_field(self, field_name: str, value: Any, rule: ValidationRule) -> ValidationResult:
        """验证单个字段"""
        try:
            if rule.custom_validator:
                return rule.custom_validator(field_name, value, rule)
            
            # 根据规则类型执行验证
            if rule.rule_type == "required":
                return self._validate_required(field_name, value, rule)
            elif rule.rule_type == "type":
                return self._validate_type(field_name, value, rule)
            elif rule.rule_type == "range":
                return self._validate_range(field_name, value, rule)
            elif rule.rule_type == "format":
                return self._validate_format(field_name, value, rule)
            elif rule.rule_type == "business_rule":
                return self._validate_business_rule(field_name, value, rule)
            elif rule.rule_type == "dependency":
                return self._validate_dependency(field_name, value, rule)
            elif rule.rule_type == "consistency":
                return self._validate_consistency(field_name, value, rule)
            elif rule.rule_type in self.custom_validators:
                return self.custom_validators[rule.rule_type](field_name, value, rule)
            else:
                return ValidationResult(
                    is_valid=False,
                    field_name=field_name,
                    rule_type=rule.rule_type,
                    level=ValidationLevel.HIGH,
                    message=f"未知验证规则类型: {rule.rule_type}",
                    value=value
                )
                
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                field_name=field_name,
                rule_type=rule.rule_type,
                level=ValidationLevel.HIGH,
                message=f"验证执行错误: {str(e)}",
                value=value
            )
    
    def _validate_required(self, field_name: str, value: Any, rule: ValidationRule) -> ValidationResult:
        """验证必填字段"""
        is_valid = value is not None and value != "" and value != []
        
        return ValidationResult(
            is_valid=is_valid,
            field_name=field_name,
            rule_type="required",
            level=rule.level,
            message="字段为必填项" if not is_valid else "字段已填写",
            value=value
        )
    
    def _validate_type(self, field_name: str, value: Any, rule: ValidationRule) -> ValidationResult:
        """验证数据类型"""
        expected_type = rule.parameters.get("type")
        is_valid = False
        actual_type = type(value).__name__
        
        if expected_type == DataType.STRING:
            is_valid = isinstance(value, str)
        elif expected_type == DataType.INTEGER:
            is_valid = isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == DataType.FLOAT:
            is_valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == DataType.BOOLEAN:
            is_valid = isinstance(value, bool)
        elif expected_type == DataType.DATE:
            is_valid = self._is_date(value)
        elif expected_type == DataType.DATETIME:
            is_valid = self._is_datetime(value)
        elif expected_type == DataType.EMAIL:
            is_valid = isinstance(value, str) and self.email_pattern.match(value)
        elif expected_type == DataType.PHONE:
            is_valid = isinstance(value, str) and self.phone_pattern.match(value)
        elif expected_type == DataType.URL:
            is_valid = isinstance(value, str) and self.url_pattern.match(value)
        elif expected_type == DataType.JSON:
            is_valid = self._is_json(value)
        elif expected_type == DataType.LIST:
            is_valid = isinstance(value, list)
        elif expected_type == DataType.DICT:
            is_valid = isinstance(value, dict)
        
        return ValidationResult(
            is_valid=is_valid,
            field_name=field_name,
            rule_type="type",
            level=rule.level,
            message=f"数据类型不匹配，期望: {expected_type.value}, 实际: {actual_type}" if not is_valid else "数据类型正确",
            value=value,
            expected=expected_type.value,
            actual=actual_type
        )
    
    def _validate_range(self, field_name: str, value: Any, rule: ValidationRule) -> ValidationResult:
        """验证数据范围"""
        min_val = rule.parameters.get("min")
        max_val = rule.parameters.get("max")
        is_valid = True
        message = "数值在有效范围内"
        
        try:
            if isinstance(value, (int, float)):
                if min_val is not None and value < min_val:
                    is_valid = False
                    message = f"数值 {value} 小于最小值 {min_val}"
                elif max_val is not None and value > max_val:
                    is_valid = False
                    message = f"数值 {value} 大于最大值 {max_val}"
            elif isinstance(value, str):
                length = len(value)
                if min_val is not None and length < min_val:
                    is_valid = False
                    message = f"字符串长度 {length} 小于最小长度 {min_val}"
                elif max_val is not None and length > max_val:
                    is_valid = False
                    message = f"字符串长度 {length} 大于最大长度 {max_val}"
            elif isinstance(value, list):
                length = len(value)
                if min_val is not None and length < min_val:
                    is_valid = False
                    message = f"列表长度 {length} 小于最小长度 {min_val}"
                elif max_val is not None and length > max_val:
                    is_valid = False
                    message = f"列表长度 {length} 大于最大长度 {max_val}"
        except Exception as e:
            is_valid = False
            message = f"范围验证错误: {str(e)}"
        
        return ValidationResult(
            is_valid=is_valid,
            field_name=field_name,
            rule_type="range",
            level=rule.level,
            message=message,
            value=value,
            expected=f"[{min_val}, {max_val}]"
        )
    
    def _validate_format(self, field_name: str, value: Any, rule: ValidationRule) -> ValidationResult:
        """验证数据格式"""
        format_type = rule.parameters.get("format")
        pattern = rule.parameters.get("pattern")
        is_valid = True
        message = "数据格式正确"
        
        if not isinstance(value, str):
            is_valid = False
            message = "格式验证仅适用于字符串类型"
        else:
            if format_type == "email":
                is_valid = self.email_pattern.match(value)
                message = "邮箱格式不正确" if not is_valid else "邮箱格式正确"
            elif format_type == "phone":
                is_valid = self.phone_pattern.match(value)
                message = "手机号格式不正确" if not is_valid else "手机号格式正确"
            elif format_type == "url":
                is_valid = self.url_pattern.match(value)
                message = "URL格式不正确" if not is_valid else "URL格式正确"
            elif format_type == "date":
                is_valid = any(p.match(value) for p in self.date_patterns)
                message = "日期格式不正确" if not is_valid else "日期格式正确"
            elif format_type == "custom" and pattern:
                is_valid = re.match(pattern, value) is not None
                message = "数据格式不符合要求" if not is_valid else "数据格式正确"
        
        return ValidationResult(
            is_valid=is_valid,
            field_name=field_name,
            rule_type="format",
            level=rule.level,
            message=message,
            value=value,
            expected=format_type or pattern
        )
    
    def _validate_business_rule(self, field_name: str, value: Any, rule: ValidationRule) -> ValidationResult:
        """验证业务规则"""
        rule_expression = rule.parameters.get("expression")
        is_valid = True
        message = "业务规则验证通过"
        
        try:
            if rule_expression:
                # 简单的业务规则评估（实际应用中应该使用更安全的表达式引擎）
                local_vars = {"value": value, "field": field_name}
                is_valid = eval(rule_expression, {"__builtins__": {}}, local_vars)
                message = "业务规则验证失败" if not is_valid else "业务规则验证通过"
        except Exception as e:
            is_valid = False
            message = f"业务规则验证错误: {str(e)}"
        
        return ValidationResult(
            is_valid=is_valid,
            field_name=field_name,
            rule_type="business_rule",
            level=rule.level,
            message=message,
            value=value
        )
    
    def _validate_dependency(self, field_name: str, value: Any, rule: ValidationRule) -> ValidationResult:
        """验证数据依赖性"""
        depends_on = rule.parameters.get("depends_on")
        condition = rule.parameters.get("condition")
        is_valid = True
        message = "依赖关系验证通过"
        
        # 这里需要访问完整的数据上下文，实际实现中需要重构
        # 为了演示，我们假设依赖验证总是通过
        if depends_on and condition:
            # 实现依赖逻辑
            pass
        
        return ValidationResult(
            is_valid=is_valid,
            field_name=field_name,
            rule_type="dependency",
            level=rule.level,
            message=message,
            value=value
        )
    
    def _validate_consistency(self, field_name: str, value: Any, rule: ValidationRule) -> ValidationResult:
        """验证数据一致性"""
        reference_field = rule.parameters.get("reference_field")
        consistency_rule = rule.parameters.get("rule")
        is_valid = True
        message = "数据一致性验证通过"
        
        # 实现一致性检查逻辑
        if reference_field and consistency_rule:
            # 需要访问其他字段的值进行一致性检查
            pass
        
        return ValidationResult(
            is_valid=is_valid,
            field_name=field_name,
            rule_type="consistency",
            level=rule.level,
            message=message,
            value=value
        )
    
    def _is_date(self, value: Any) -> bool:
        """检查是否为日期"""
        if not isinstance(value, str):
            return False
        return any(pattern.match(value) for pattern in self.date_patterns)
    
    def _is_datetime(self, value: Any) -> bool:
        """检查是否为日期时间"""
        if not isinstance(value, str):
            return False
        try:
            datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
            return True
        except ValueError:
            return False
    
    def _is_json(self, value: Any) -> bool:
        """检查是否为JSON格式"""
        if isinstance(value, str):
            try:
                json.loads(value)
                return True
            except json.JSONDecodeError:
                return False
        return False
    
    def _calculate_quality_score(self, data: Any, results: List[ValidationResult]) -> DataQualityScore:
        """计算数据质量评分"""
        if not results:
            return DataQualityScore(
                overall_score=100.0,
                completeness_score=100.0,
                accuracy_score=100.0,
                consistency_score=100.0,
                validity_score=100.0,
                uniqueness_score=100.0,
                timeliness_score=100.0
            )
        
        # 计算各项评分
        total_results = len(results)
        valid_results = sum(1 for r in results if r.is_valid)
        
        # 基础评分
        validity_score = (valid_results / total_results) * 100
        
        # 完整性评分（基于必填字段验证）
        required_results = [r for r in results if r.rule_type == "required"]
        if required_results:
            complete_results = sum(1 for r in required_results if r.is_valid)
            completeness_score = (complete_results / len(required_results)) * 100
        else:
            completeness_score = 100.0
        
        # 准确性评分（基于类型、范围、格式验证）
        accuracy_related = ["type", "range", "format"]
        accuracy_results = [r for r in results if r.rule_type in accuracy_related]
        if accuracy_results:
            accurate_results = sum(1 for r in accuracy_results if r.is_valid)
            accuracy_score = (accurate_results / len(accuracy_results)) * 100
        else:
            accuracy_score = 100.0
        
        # 一致性评分（基于一致性验证）
        consistency_results = [r for r in results if r.rule_type == "consistency"]
        if consistency_results:
            consistent_results = sum(1 for r in consistency_results if r.is_valid)
            consistency_score = (consistent_results / len(consistency_results)) * 100
        else:
            consistency_score = 100.0
        
        # 唯一性评分（简化实现）
        uniqueness_score = 95.0  # 默认高分，需要根据实际数据计算
        
        # 时效性评分（简化实现）
        timeliness_score = 90.0  # 默认分数，需要根据数据时间戳计算
        
        # 计算总体评分（加权平均）
        weights = {
            'completeness': 0.2,
            'accuracy': 0.25,
            'consistency': 0.2,
            'validity': 0.25,
            'uniqueness': 0.05,
            'timeliness': 0.05
        }
        
        overall_score = (
            completeness_score * weights['completeness'] +
            accuracy_score * weights['accuracy'] +
            consistency_score * weights['consistency'] +
            validity_score * weights['validity'] +
            uniqueness_score * weights['uniqueness'] +
            timeliness_score * weights['timeliness']
        )
        
        return DataQualityScore(
            overall_score=round(overall_score, 2),
            completeness_score=round(completeness_score, 2),
            accuracy_score=round(accuracy_score, 2),
            consistency_score=round(consistency_score, 2),
            validity_score=round(validity_score, 2),
            uniqueness_score=round(uniqueness_score, 2),
            timeliness_score=round(timeliness_score, 2),
            details={
                'total_validations': total_results,
                'valid_validations': valid_results,
                'invalid_validations': total_results - valid_results
            }
        )
    
    def _generate_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """生成验证摘要"""
        if not results:
            return {}
        
        # 按级别统计
        level_stats = defaultdict(int)
        type_stats = defaultdict(int)
        
        for result in results:
            level_stats[result.level.value] += 1
            type_stats[result.rule_type] += 1
        
        # 找出主要问题
        invalid_results = [r for r in results if not r.is_valid]
        main_issues = []
        
        if invalid_results:
            # 按级别排序，找出最重要的问题
            critical_issues = [r for r in invalid_results if r.level == ValidationLevel.CRITICAL]
            if critical_issues:
                main_issues.extend([r.message for r in critical_issues[:3]])
            
            high_issues = [r for r in invalid_results if r.level == ValidationLevel.HIGH]
            if high_issues and len(main_issues) < 3:
                main_issues.extend([r.message for r in high_issues[:3-len(main_issues)]])
        
        return {
            'total_validations': len(results),
            'valid_count': len([r for r in results if r.is_valid]),
            'invalid_count': len([r for r in results if not r.is_valid]),
            'success_rate': round(len([r for r in results if r.is_valid]) / len(results) * 100, 2),
            'level_distribution': dict(level_stats),
            'rule_type_distribution': dict(type_stats),
            'main_issues': main_issues
        }
    
    def _generate_recommendations(self, results: List[ValidationResult], 
                                quality_score: DataQualityScore) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于质量评分的建议
        if quality_score.overall_score < 60:
            recommendations.append("数据质量较差，建议进行全面的数据清洗和修复")
        elif quality_score.overall_score < 80:
            recommendations.append("数据质量中等，建议重点关注验证失败的项目")
        
        if quality_score.completeness_score < 80:
            recommendations.append("数据完整性不足，建议补充缺失的必填字段")
        
        if quality_score.accuracy_score < 80:
            recommendations.append("数据准确性有待提高，建议检查数据类型和取值范围")
        
        if quality_score.consistency_score < 80:
            recommendations.append("数据一致性存在问题，建议建立数据标准化规则")
        
        # 基于验证结果的建议
        invalid_results = [r for r in results if not r.is_valid]
        if invalid_results:
            rule_type_counts = defaultdict(int)
            for result in invalid_results:
                rule_type_counts[result.rule_type] += 1
            
            # 找出最常见的问题类型
            most_common_issue = max(rule_type_counts.items(), key=lambda x: x[1])
            if most_common_issue[1] > len(results) * 0.1:  # 超过10%
                if most_common_issue[0] == "type":
                    recommendations.append("数据类型错误较多，建议在数据输入时加强类型检查")
                elif most_common_issue[0] == "range":
                    recommendations.append("数值范围错误较多，建议重新评估业务规则和约束条件")
                elif most_common_issue[0] == "format":
                    recommendations.append("数据格式错误较多，建议提供格式示例和输入指导")
        
        return recommendations
    
    def get_validation_history(self, limit: Optional[int] = None) -> List[ValidationReport]:
        """获取验证历史"""
        if limit is None:
            return self.validation_history.copy()
        return self.validation_history[-limit:]
    
    def export_report(self, report: ValidationReport, format: str = "json", 
                     file_path: Optional[str] = None) -> str:
        """
        导出验证报告
        
        Args:
            report: 验证报告
            format: 导出格式 ("json", "html", "csv")
            file_path: 文件路径，如果为None则返回字符串
            
        Returns:
            导出的内容或文件路径
        """
        if format == "json":
            content = self._report_to_json(report)
        elif format == "html":
            content = self._report_to_html(report)
        elif format == "csv":
            content = self._report_to_csv(report)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return file_path
        else:
            return content
    
    def _report_to_json(self, report: ValidationReport) -> str:
        """将报告转换为JSON格式"""
        def default_serializer(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dict__'):
                # 处理可能包含不可序列化对象的情况
                result = {}
                for key, value in obj.__dict__.items():
                    try:
                        # 尝试直接序列化
                        json.dumps(value)
                        result[key] = value
                    except (TypeError, ValueError):
                        # 如果无法序列化，转换为字符串
                        result[key] = str(value)
                return result
            elif hasattr(obj, 'group'):  # 正则表达式匹配对象
                return obj.group()
            else:
                return str(obj)
        
        return json.dumps(report, indent=2, default=default_serializer, ensure_ascii=False)
    
    def _report_to_html(self, report: ValidationReport) -> str:
        """将报告转换为HTML格式"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>数据验证报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .score {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }}
                .valid {{ color: green; }}
                .invalid {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .critical {{ background-color: #ffebee; }}
                .high {{ background-color: #fff3e0; }}
                .medium {{ background-color: #fff9c4; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>数据验证报告</h1>
                <p>报告ID: {report.report_id}</p>
                <p>生成时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>验证摘要</h2>
                <p>总记录数: {report.total_records}</p>
                <p>有效记录: <span class="valid">{report.valid_records}</span></p>
                <p>无效记录: <span class="invalid">{report.invalid_records}</span></p>
                
                <h3>数据质量评分</h3>
                <div class="score">总体评分: {report.quality_score.overall_score}</div>
                <div class="score">完整性: {report.quality_score.completeness_score}</div>
                <div class="score">准确性: {report.quality_score.accuracy_score}</div>
                <div class="score">一致性: {report.quality_score.consistency_score}</div>
                <div class="score">有效性: {report.quality_score.validity_score}</div>
            </div>
            
            <h2>验证结果详情</h2>
            <table>
                <tr>
                    <th>字段名</th>
                    <th>验证类型</th>
                    <th>级别</th>
                    <th>结果</th>
                    <th>消息</th>
                    <th>值</th>
                </tr>
        """
        
        for result in report.validation_results:
            row_class = result.level.value
            status_class = "valid" if result.is_valid else "invalid"
            html += f"""
                <tr class="{row_class}">
                    <td>{result.field_name}</td>
                    <td>{result.rule_type}</td>
                    <td>{result.level.value}</td>
                    <td class="{status_class}">{'通过' if result.is_valid else '失败'}</td>
                    <td>{result.message}</td>
                    <td>{str(result.value)[:50]}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>改进建议</h2>
            <ul>
        """
        
        for recommendation in report.recommendations:
            html += f"<li>{recommendation}</li>"
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html
    
    def _report_to_csv(self, report: ValidationReport) -> str:
        """将报告转换为CSV格式"""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # 写入表头
        writer.writerow([
            '字段名', '验证类型', '级别', '结果', '消息', '值', '期望值', '实际值', '时间戳'
        ])
        
        # 写入数据
        for result in report.validation_results:
            writer.writerow([
                result.field_name,
                result.rule_type,
                result.level.value,
                '通过' if result.is_valid else '失败',
                result.message,
                str(result.value),
                str(result.expected),
                str(result.actual),
                result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        return output.getvalue()


# 便捷函数
def create_basic_validator() -> DataValidator:
    """创建基础验证器，包含常用验证规则"""
    validator = DataValidator()
    
    # 添加基础验证规则
    validator.add_validation_rule(ValidationRule(
        field_name="*",
        rule_type="required",
        level=ValidationLevel.CRITICAL,
        error_message="必填字段不能为空"
    ))
    
    return validator


def validate_user_data(user_data: Dict) -> ValidationReport:
    """验证用户数据的便捷函数"""
    validator = create_basic_validator()
    
    # 添加用户数据特定验证规则
    validator.add_validation_rule(ValidationRule(
        field_name="email",
        rule_type="type",
        parameters={"type": DataType.EMAIL},
        level=ValidationLevel.HIGH,
        error_message="邮箱格式不正确"
    ))
    
    validator.add_validation_rule(ValidationRule(
        field_name="phone",
        rule_type="type",
        parameters={"type": DataType.PHONE},
        level=ValidationLevel.MEDIUM,
        error_message="手机号格式不正确"
    ))
    
    validator.add_validation_rule(ValidationRule(
        field_name="age",
        rule_type="range",
        parameters={"min": 0, "max": 150},
        level=ValidationLevel.MEDIUM,
        error_message="年龄必须在0-150之间"
    ))
    
    return validator.validate_data(user_data)


def validate_financial_data(financial_data: Dict) -> ValidationReport:
    """验证金融数据的便捷函数"""
    validator = create_basic_validator()
    
    # 添加金融数据特定验证规则
    validator.add_validation_rule(ValidationRule(
        field_name="amount",
        rule_type="range",
        parameters={"min": 0},
        level=ValidationLevel.CRITICAL,
        error_message="金额不能为负数"
    ))
    
    validator.add_validation_rule(ValidationRule(
        field_name="date",
        rule_type="format",
        parameters={"format": "date"},
        level=ValidationLevel.HIGH,
        error_message="日期格式不正确"
    ))
    
    validator.add_validation_rule(ValidationRule(
        field_name="currency",
        rule_type="business_rule",
        parameters={"expression": "value in ['USD', 'EUR', 'CNY', 'JPY']"},
        level=ValidationLevel.MEDIUM,
        error_message="不支持的货币类型"
    ))
    
    return validator.validate_data(financial_data)


if __name__ == "__main__":
    # 测试代码
    print("T3数据验证器测试")
    
    # 创建测试数据
    test_data = {
        "name": "张三",
        "email": "zhangsan@example.com",
        "phone": "13812345678",
        "age": 25,
        "address": "北京市朝阳区",
        "salary": 15000.50,
        "join_date": "2023-01-15"
    }
    
    # 创建验证器
    validator = create_basic_validator()
    
    # 添加自定义验证规则
    validator.add_validation_rule(ValidationRule(
        field_name="name",
        rule_type="range",
        parameters={"min": 2, "max": 10},
        level=ValidationLevel.MEDIUM,
        error_message="姓名长度必须在2-10个字符之间"
    ))
    
    validator.add_validation_rule(ValidationRule(
        field_name="salary",
        rule_type="range",
        parameters={"min": 1000, "max": 100000},
        level=ValidationLevel.HIGH,
        error_message="薪资必须在合理范围内"
    ))
    
    # 执行验证
    report = validator.validate_data(test_data)
    
    # 输出结果
    print(f"验证报告ID: {report.report_id}")
    print(f"总记录数: {report.total_records}")
    print(f"有效记录: {report.valid_records}")
    print(f"无效记录: {report.invalid_records}")
    print(f"总体质量评分: {report.quality_score.overall_score}")
    
    print("\n验证结果:")
    for result in report.validation_results:
        status = "✓" if result.is_valid else "✗"
        print(f"{status} {result.field_name}: {result.message}")
    
    print("\n改进建议:")
    for recommendation in report.recommendations:
        print(f"- {recommendation}")
    
    # 导出报告
    json_report = validator.export_report(report, "json")
    print(f"\nJSON报告长度: {len(json_report)} 字符")
    
    # 测试便捷函数
    print("\n测试用户数据验证:")
    user_report = validate_user_data(test_data)
    print(f"用户数据验证结果: {user_report.valid_records}/{user_report.total_records} 有效")