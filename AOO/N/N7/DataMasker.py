#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据脱敏器 (DataMasker)

一个全面的数据脱敏系统，支持多种脱敏算法和策略，确保敏感数据的安全处理。
包含动态脱敏、静态脱敏、合规性检查、性能优化等功能。

主要功能：
1. 敏感数据识别 - 自动识别各种类型的敏感数据
2. 数据脱敏算法 - 提供多种脱敏算法
3. 动态数据脱敏 - 实时数据处理
4. 静态数据脱敏 - 批量数据处理
5. 数据脱敏策略 - 可配置的脱敏规则
6. 数据脱敏验证 - 脱敏效果验证
7. 数据脱敏日志 - 完整的操作记录
8. 数据脱敏性能优化 - 高效的数据处理
9. 数据脱敏合规性 - 符合法规要求


日期: 2025-11-06
版本: 1.0.0
"""

import re
import hashlib
import logging
import json
import time
import uuid
import random
import string
import base64
import logging.handlers
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from abc import ABC, abstractmethod


# =============================================================================
# 枚举类型定义
# =============================================================================

class SensitiveDataType(Enum):
    """敏感数据类型枚举"""
    PERSONAL_ID = "personal_id"  # 身份证号
    PHONE = "phone"  # 手机号
    EMAIL = "email"  # 邮箱
    CREDIT_CARD = "credit_card"  # 信用卡号
    BANK_ACCOUNT = "bank_account"  # 银行账号
    NAME = "name"  # 姓名
    ADDRESS = "address"  # 地址
    IP_ADDRESS = "ip_address"  # IP地址
    MAC_ADDRESS = "mac_address"  # MAC地址
    LICENSE_PLATE = "license_plate"  # 车牌号
    PASSPORT = "passport"  # 护照号
    DRIVER_LICENSE = "driver_license"  # 驾驶证号
    SOCIAL_SECURITY = "social_security"  # 社保号
    MEDICAL_RECORD = "medical_record"  # 医疗记录
    FINANCIAL_ACCOUNT = "financial_account"  # 金融账户


class MaskingStrategy(Enum):
    """脱敏策略枚举"""
    HASH = "hash"  # 哈希脱敏
    REPLACE = "replace"  # 替换脱敏
    SHUFFLE = "shuffle"  # 随机替换
    PARTIAL_MASK = "partial_mask"  # 部分遮蔽
    ENCRYPTION = "encryption"  # 加密脱敏
    TOKENIZATION = "tokenization"  # 令牌化
    NULLIFICATION = "nullification"  # 置空
    RANGE_GENERALIZATION = "range_generalization"  # 范围泛化
    DATE_SHIFT = "date_shift"  # 日期偏移
    NUMBER_ROUNDING = "number_rounding"  # 数字舍入


class ComplianceStandard(Enum):
    """合规标准枚举"""
    GDPR = "gdpr"  # 欧盟通用数据保护条例
    CCPA = "ccpa"  # 加州消费者隐私法案
    PIPEDA = "pipeda"  # 加拿大个人信息保护法
    HIPAA = "hipaa"  # 健康保险流通与责任法案
    SOX = "sox"  # 萨班斯-奥克斯利法案
    PCI_DSS = "pci_dss"  # 支付卡行业数据安全标准
    ISO27001 = "iso27001"  # 信息安全管理体系
    LOCAL_LAW = "local_law"  # 当地法律


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class SensitiveDataPattern:
    """敏感数据模式定义"""
    data_type: SensitiveDataType
    regex_pattern: str
    sample_format: str
    priority: int = 1  # 识别优先级，数字越大优先级越高
    validation_function: Optional[Callable[[str], bool]] = None


@dataclass
class MaskingRule:
    """脱敏规则定义"""
    data_type: SensitiveDataType
    strategy: MaskingStrategy
    parameters: Dict[str, Any]
    enabled: bool = True
    compliance_standards: List[ComplianceStandard] = None


@dataclass
class MaskingResult:
    """脱敏结果"""
    original_data: str
    masked_data: str
    data_type: SensitiveDataType
    strategy: MaskingStrategy
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


@dataclass
class MaskingStatistics:
    """脱敏统计信息"""
    total_processed: int = 0
    successful_masked: int = 0
    failed_masked: int = 0
    processing_time: float = 0.0
    data_types_processed: Dict[SensitiveDataType, int] = None
    strategies_used: Dict[MaskingStrategy, int] = None

    def __post_init__(self):
        if self.data_types_processed is None:
            self.data_types_processed = {}
        if self.strategies_used is None:
            self.strategies_used = {}


# =============================================================================
# 脱敏算法接口
# =============================================================================

class MaskingAlgorithm(ABC):
    """脱敏算法基类"""

    @abstractmethod
    def mask(self, data: str, parameters: Dict[str, Any]) -> str:
        """执行脱敏算法
        
        Args:
            data: 原始数据
            parameters: 算法参数
            
        Returns:
            脱敏后的数据
        """
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证算法参数
        
        Args:
            parameters: 算法参数
            
        Returns:
            参数是否有效
        """
        pass


# =============================================================================
# 具体脱敏算法实现
# =============================================================================

class HashMaskingAlgorithm(MaskingAlgorithm):
    """哈希脱敏算法"""

    def mask(self, data: str, parameters: Dict[str, Any]) -> str:
        """执行哈希脱敏"""
        algorithm = parameters.get("algorithm", "sha256")
        salt = parameters.get("salt", "")
        
        if algorithm == "sha256":
            return hashlib.sha256((data + salt).encode()).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5((data + salt).encode()).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1((data + salt).encode()).hexdigest()
        else:
            raise ValueError(f"不支持的哈希算法: {algorithm}")

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        algorithm = parameters.get("algorithm", "sha256")
        return algorithm in ["sha256", "md5", "sha1"]


class ReplaceMaskingAlgorithm(MaskingAlgorithm):
    """替换脱敏算法"""

    def mask(self, data: str, parameters: Dict[str, Any]) -> str:
        """执行替换脱敏"""
        replacement = parameters.get("replacement", "*")
        preserve_length = parameters.get("preserve_length", True)
        
        if preserve_length:
            return replacement * len(data)
        else:
            return replacement

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        return isinstance(parameters.get("replacement", "*"), str)


class PartialMaskingAlgorithm(MaskingAlgorithm):
    """部分遮蔽脱敏算法"""

    def mask(self, data: str, parameters: Dict[str, Any]) -> str:
        """执行部分遮蔽"""
        mask_char = parameters.get("mask_char", "*")
        keep_start = parameters.get("keep_start", 3)
        keep_end = parameters.get("keep_end", 4)
        
        if len(data) <= keep_start + keep_end:
            return mask_char * len(data)
        
        return data[:keep_start] + mask_char * (len(data) - keep_start - keep_end) + data[-keep_end:]

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        keep_start = parameters.get("keep_start", 3)
        keep_end = parameters.get("keep_end", 4)
        return isinstance(keep_start, int) and isinstance(keep_end, int) and keep_start >= 0 and keep_end >= 0


class ShuffleMaskingAlgorithm(MaskingAlgorithm):
    """随机替换脱敏算法"""

    def mask(self, data: str, parameters: Dict[str, Any]) -> str:
        """执行随机替换"""
        mask_char = parameters.get("mask_char", "*")
        keep_original = parameters.get("keep_original", False)
        
        if keep_original:
            # 保持部分字符不变
            preserve_ratio = parameters.get("preserve_ratio", 0.3)
            preserve_count = max(1, int(len(data) * preserve_ratio))
            preserve_indices = random.sample(range(len(data)), preserve_count)
            
            result = list(data)
            for i in range(len(data)):
                if i not in preserve_indices:
                    result[i] = mask_char
            return "".join(result)
        else:
            return mask_char * len(data)

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        preserve_ratio = parameters.get("preserve_ratio", 0.3)
        return 0 <= preserve_ratio <= 1


class TokenizationMaskingAlgorithm(MaskingAlgorithm):
    """令牌化脱敏算法"""

    def __init__(self):
        self.token_map: Dict[str, str] = {}
        self.lock = threading.Lock()

    def mask(self, data: str, parameters: Dict[str, Any]) -> str:
        """执行令牌化脱敏"""
        with self.lock:
            if data in self.token_map:
                return self.token_map[data]
            
            token_length = parameters.get("token_length", 16)
            token_prefix = parameters.get("token_prefix", "TKN_")
            
            # 生成唯一令牌
            token = token_prefix + ''.join(random.choices(string.ascii_uppercase + string.digits, k=token_length))
            self.token_map[data] = token
            return token

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        token_length = parameters.get("token_length", 16)
        token_prefix = parameters.get("token_prefix", "TKN_")
        return isinstance(token_length, int) and token_length > 0 and isinstance(token_prefix, str)


class DateShiftMaskingAlgorithm(MaskingAlgorithm):
    """日期偏移脱敏算法"""

    def mask(self, data: str, parameters: Dict[str, Any]) -> str:
        """执行日期偏移"""
        try:
            # 尝试解析日期
            date_formats = parameters.get("date_formats", ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"])
            date_obj = None
            
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(data, fmt)
                    break
                except ValueError:
                    continue
            
            if date_obj is None:
                return data  # 无法解析日期，返回原数据
            
            # 应用偏移
            shift_days = parameters.get("shift_days", 30)
            shifted_date = date_obj + timedelta(days=shift_days)
            
            # 保持原始格式
            return shifted_date.strftime(date_formats[0])
            
        except Exception:
            return data

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        shift_days = parameters.get("shift_days", 30)
        date_formats = parameters.get("date_formats", ["%Y-%m-%d"])
        return isinstance(shift_days, int) and isinstance(date_formats, list)


class EncryptionMaskingAlgorithm(MaskingAlgorithm):
    """加密脱敏算法"""

    def mask(self, data: str, parameters: Dict[str, Any]) -> str:
        """执行加密脱敏"""
        try:
            # 简单的Base64编码作为示例（实际应用中应使用更强的加密）
            key = parameters.get("key", "default_key")
            encoded_data = base64.b64encode((data + key).encode()).decode()
            return f"ENC_{encoded_data}"
        except Exception:
            return data

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        return True  # 简化验证


# =============================================================================
# 敏感数据识别器
# =============================================================================

class SensitiveDataIdentifier:
    """敏感数据识别器"""

    def __init__(self):
        self.patterns: List[SensitiveDataPattern] = self._initialize_patterns()

    def _initialize_patterns(self) -> List[SensitiveDataPattern]:
        """初始化敏感数据识别模式"""
        return [
            # 身份证号
            SensitiveDataPattern(
                data_type=SensitiveDataType.PERSONAL_ID,
                regex_pattern=r'[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]',
                sample_format="110101199001011234",
                priority=10
            ),
            # 手机号
            SensitiveDataPattern(
                data_type=SensitiveDataType.PHONE,
                regex_pattern=r'1[3-9]\d{9}',
                sample_format="13800138000",
                priority=9
            ),
            # 邮箱
            SensitiveDataPattern(
                data_type=SensitiveDataType.EMAIL,
                regex_pattern=r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                sample_format="user@example.com",
                priority=8
            ),
            # 信用卡号
            SensitiveDataPattern(
                data_type=SensitiveDataType.CREDIT_CARD,
                regex_pattern=r'4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12}',
                sample_format="4111111111111111",
                priority=10
            ),
            # 银行账号
            SensitiveDataPattern(
                data_type=SensitiveDataType.BANK_ACCOUNT,
                regex_pattern=r'\d{16,19}',
                sample_format="6222021234567890123",
                priority=7
            ),
            # IP地址
            SensitiveDataPattern(
                data_type=SensitiveDataType.IP_ADDRESS,
                regex_pattern=r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',
                sample_format="192.168.1.1",
                priority=6
            ),
            # MAC地址
            SensitiveDataPattern(
                data_type=SensitiveDataType.MAC_ADDRESS,
                regex_pattern=r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})',
                sample_format="00:1B:44:11:3A:B7",
                priority=6
            ),
            # 车牌号
            SensitiveDataPattern(
                data_type=SensitiveDataType.LICENSE_PLATE,
                regex_pattern=r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{4}[A-Z0-9挂学警港澳]',
                sample_format="京A12345",
                priority=5
            ),
            # 护照号
            SensitiveDataPattern(
                data_type=SensitiveDataType.PASSPORT,
                regex_pattern=r'[EG]\d{8}',
                sample_format="E12345678",
                priority=8
            ),
            # 驾驶证号
            SensitiveDataPattern(
                data_type=SensitiveDataType.DRIVER_LICENSE,
                regex_pattern=r'\d{18}',
                sample_format="110101199001011234",
                priority=7
            ),
            # 社保号
            SensitiveDataPattern(
                data_type=SensitiveDataType.SOCIAL_SECURITY,
                regex_pattern=r'\d{18}|\d{15}',
                sample_format="110101199001011234",
                priority=9
            ),
        ]

    def identify_sensitive_data(self, data: str) -> List[Tuple[SensitiveDataType, int, int]]:
        """识别敏感数据
        
        Args:
            data: 待识别数据
            
        Returns:
            敏感数据类型、开始位置、结束位置的列表
        """
        results = []
        
        for pattern in self.patterns:
            matches = re.finditer(pattern.regex_pattern, data)
            for match in matches:
                results.append((pattern.data_type, match.start(), match.end()))
        
        # 按优先级排序
        results.sort(key=lambda x: self._get_pattern_priority(x[0]), reverse=True)
        return results

    def _get_pattern_priority(self, data_type: SensitiveDataType) -> int:
        """获取数据类型优先级"""
        for pattern in self.patterns:
            if pattern.data_type == data_type:
                return pattern.priority
        return 1

    def validate_sensitive_data(self, data: str, data_type: SensitiveDataType) -> bool:
        """验证敏感数据格式"""
        for pattern in self.patterns:
            if pattern.data_type == data_type:
                if pattern.validation_function:
                    return pattern.validation_function(data)
                else:
                    return bool(re.match(pattern.regex_pattern, data))
        return False


# =============================================================================
# 数据脱敏器主类
# =============================================================================

class DataMasker:
    """数据脱敏器主类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化数据脱敏器
        
        Args:
            config_file: 配置文件路径
        """
        self.identifier = SensitiveDataIdentifier()
        self.algorithms: Dict[MaskingStrategy, MaskingAlgorithm] = self._initialize_algorithms()
        self.masking_rules: Dict[SensitiveDataType, MaskingRule] = {}
        self.statistics = MaskingStatistics()
        self.logger = self._setup_logger()
        self.compliance_standards = set()
        
        # 加载配置
        if config_file:
            self.load_config(config_file)
        else:
            self._setup_default_rules()
        
        # 性能优化
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 1000

    def _initialize_algorithms(self) -> Dict[MaskingStrategy, MaskingAlgorithm]:
        """初始化脱敏算法"""
        return {
            MaskingStrategy.HASH: HashMaskingAlgorithm(),
            MaskingStrategy.REPLACE: ReplaceMaskingAlgorithm(),
            MaskingStrategy.SHUFFLE: ShuffleMaskingAlgorithm(),
            MaskingStrategy.PARTIAL_MASK: PartialMaskingAlgorithm(),
            MaskingStrategy.TOKENIZATION: TokenizationMaskingAlgorithm(),
            MaskingStrategy.DATE_SHIFT: DateShiftMaskingAlgorithm(),
            MaskingStrategy.ENCRYPTION: EncryptionMaskingAlgorithm(),
        }

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("DataMasker")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                'data_masker.log', maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger

    def _setup_default_rules(self):
        """设置默认脱敏规则"""
        default_rules = [
            MaskingRule(
                data_type=SensitiveDataType.PERSONAL_ID,
                strategy=MaskingStrategy.PARTIAL_MASK,
                parameters={"keep_start": 6, "keep_end": 4, "mask_char": "*"},
                compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.CCPA]
            ),
            MaskingRule(
                data_type=SensitiveDataType.PHONE,
                strategy=MaskingStrategy.PARTIAL_MASK,
                parameters={"keep_start": 3, "keep_end": 4, "mask_char": "*"},
                compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.CCPA]
            ),
            MaskingRule(
                data_type=SensitiveDataType.EMAIL,
                strategy=MaskingStrategy.PARTIAL_MASK,
                parameters={"keep_start": 2, "keep_end": 0, "mask_char": "*"},
                compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.CCPA]
            ),
            MaskingRule(
                data_type=SensitiveDataType.CREDIT_CARD,
                strategy=MaskingStrategy.PARTIAL_MASK,
                parameters={"keep_start": 4, "keep_end": 4, "mask_char": "*"},
                compliance_standards=[ComplianceStandard.PCI_DSS, ComplianceStandard.GDPR]
            ),
            MaskingRule(
                data_type=SensitiveDataType.NAME,
                strategy=MaskingStrategy.HASH,
                parameters={"algorithm": "sha256"},
                compliance_standards=[ComplianceStandard.GDPR]
            ),
            MaskingRule(
                data_type=SensitiveDataType.ADDRESS,
                strategy=MaskingStrategy.TOKENIZATION,
                parameters={"token_length": 16},
                compliance_standards=[ComplianceStandard.GDPR]
            ),
        ]
        
        for rule in default_rules:
            self.masking_rules[rule.data_type] = rule

    def load_config(self, config_file: str):
        """加载配置文件
        
        Args:
            config_file: 配置文件路径
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 加载脱敏规则
            if 'masking_rules' in config:
                for rule_config in config['masking_rules']:
                    data_type = SensitiveDataType(rule_config['data_type'])
                    strategy = MaskingStrategy(rule_config['strategy'])
                    rule = MaskingRule(
                        data_type=data_type,
                        strategy=strategy,
                        parameters=rule_config['parameters'],
                        compliance_standards=[ComplianceStandard(std) for std in rule_config.get('compliance_standards', [])]
                    )
                    self.masking_rules[data_type] = rule
            
            # 加载合规标准
            if 'compliance_standards' in config:
                self.compliance_standards = {ComplianceStandard(std) for std in config['compliance_standards']}
            
            self.logger.info(f"成功加载配置文件: {config_file}")
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            self._setup_default_rules()

    def save_config(self, config_file: str):
        """保存配置文件
        
        Args:
            config_file: 配置文件路径
        """
        try:
            config = {
                'masking_rules': [
                    {
                        'data_type': rule.data_type.value,
                        'strategy': rule.strategy.value,
                        'parameters': rule.parameters,
                        'compliance_standards': [std.value for std in rule.compliance_standards or []]
                    }
                    for rule in self.masking_rules.values()
                ],
                'compliance_standards': [std.value for std in self.compliance_standards]
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功保存配置文件: {config_file}")
            
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")

    def add_masking_rule(self, rule: MaskingRule):
        """添加脱敏规则
        
        Args:
            rule: 脱敏规则
        """
        self.masking_rules[rule.data_type] = rule
        self.logger.info(f"添加脱敏规则: {rule.data_type.value} -> {rule.strategy.value}")

    def remove_masking_rule(self, data_type: SensitiveDataType):
        """移除脱敏规则
        
        Args:
            data_type: 敏感数据类型
        """
        if data_type in self.masking_rules:
            del self.masking_rules[data_type]
            self.logger.info(f"移除脱敏规则: {data_type.value}")

    def mask_data(self, data: str, data_type: Optional[SensitiveDataType] = None) -> MaskingResult:
        """脱敏单个数据项
        
        Args:
            data: 原始数据
            data_type: 数据类型（如果为None则自动识别）
            
        Returns:
            脱敏结果
        """
        start_time = time.time()
        
        try:
            # 自动识别数据类型
            if data_type is None:
                identified_types = self.identifier.identify_sensitive_data(data)
                if identified_types:
                    data_type = identified_types[0][0]
                else:
                    # 未识别到敏感数据，返回原数据
                    return MaskingResult(
                        original_data=data,
                        masked_data=data,
                        data_type=SensitiveDataType.PERSONAL_ID,  # 默认值
                        strategy=MaskingStrategy.REPLACE,
                        timestamp=datetime.now(),
                        success=False,
                        error_message="未识别到敏感数据"
                    )
            
            # 获取脱敏规则
            rule = self.masking_rules.get(data_type)
            if not rule or not rule.enabled:
                return MaskingResult(
                    original_data=data,
                    masked_data=data,
                    data_type=data_type,
                    strategy=MaskingStrategy.REPLACE,
                    timestamp=datetime.now(),
                    success=False,
                    error_message="未找到对应的脱敏规则"
                )
            
            # 检查缓存
            cache_key = f"{data_type.value}:{rule.strategy.value}:{hash(str(rule.parameters))}:{data}"
            with self._cache_lock:
                if cache_key in self._cache:
                    cached_result = self._cache[cache_key]
                    # 更新统计信息
                    self._update_statistics(start_time, data_type, rule.strategy, True)
                    return cached_result
            
            # 执行脱敏
            algorithm = self.algorithms.get(rule.strategy)
            if not algorithm:
                raise ValueError(f"不支持的脱敏策略: {rule.strategy}")
            
            if not algorithm.validate_parameters(rule.parameters):
                raise ValueError(f"脱敏参数无效: {rule.parameters}")
            
            masked_data = algorithm.mask(data, rule.parameters)
            
            # 创建结果
            result = MaskingResult(
                original_data=data,
                masked_data=masked_data,
                data_type=data_type,
                strategy=rule.strategy,
                timestamp=datetime.now(),
                success=True
            )
            
            # 缓存结果
            with self._cache_lock:
                if len(self._cache) >= self._max_cache_size:
                    # 清理缓存（简单策略：清空一半）
                    keys_to_remove = list(self._cache.keys())[:self._max_cache_size//2]
                    for key in keys_to_remove:
                        del self._cache[key]
                self._cache[cache_key] = result
            
            # 更新统计信息
            self._update_statistics(start_time, data_type, rule.strategy, True)
            
            self.logger.debug(f"脱敏成功: {data_type.value} - {data[:10]}... -> {masked_data[:10]}...")
            return result
            
        except Exception as e:
            # 更新统计信息
            self._update_statistics(start_time, data_type or SensitiveDataType.PERSONAL_ID, 
                                  MaskingStrategy.REPLACE, False)
            
            self.logger.error(f"脱敏失败: {e}")
            return MaskingResult(
                original_data=data,
                masked_data=data,
                data_type=data_type or SensitiveDataType.PERSONAL_ID,
                strategy=MaskingStrategy.REPLACE,
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )

    def mask_batch(self, data_list: List[str], max_workers: int = 4) -> List[MaskingResult]:
        """批量脱敏数据
        
        Args:
            data_list: 数据列表
            max_workers: 最大工作线程数
            
        Returns:
            脱敏结果列表
        """
        self.logger.info(f"开始批量脱敏 {len(data_list)} 条数据")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_data = {executor.submit(self.mask_data, data): data for data in data_list}
            
            # 收集结果
            for future in as_completed(future_to_data):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    data = future_to_data[future]
                    results.append(MaskingResult(
                        original_data=data,
                        masked_data=data,
                        data_type=SensitiveDataType.PERSONAL_ID,
                        strategy=MaskingStrategy.REPLACE,
                        timestamp=datetime.now(),
                        success=False,
                        error_message=str(e)
                    ))
        
        self.logger.info(f"批量脱敏完成，成功: {sum(1 for r in results if r.success)}/{len(results)}")
        return results

    def mask_file(self, input_file: str, output_file: str, 
                  data_type: Optional[SensitiveDataType] = None) -> bool:
        """脱敏文件数据
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            data_type: 数据类型
            
        Returns:
            是否成功
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            results = []
            for line in lines:
                line = line.strip()
                if line:
                    result = self.mask_data(line, data_type)
                    results.append(result.masked_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for masked_line in results:
                    f.write(masked_line + '\n')
            
            self.logger.info(f"文件脱敏完成: {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"文件脱敏失败: {e}")
            return False

    def validate_masking(self, original_data: str, masked_data: str, 
                        data_type: SensitiveDataType) -> Dict[str, Any]:
        """验证脱敏效果
        
        Args:
            original_data: 原始数据
            masked_data: 脱敏后数据
            data_type: 数据类型
            
        Returns:
            验证结果
        """
        validation_result = {
            "is_valid": True,
            "checks": {},
            "score": 0.0
        }
        
        # 检查1: 数据是否被修改
        is_modified = original_data != masked_data
        validation_result["checks"]["data_modified"] = is_modified
        validation_result["score"] += 20 if is_modified else 0
        
        # 检查2: 是否包含原始数据的片段
        has_original_segments = False
        if len(original_data) > 0:
            segment_length = max(1, len(original_data) // 4)
            for i in range(len(original_data) - segment_length + 1):
                segment = original_data[i:i + segment_length]
                if segment in masked_data:
                    has_original_segments = True
                    break
        
        validation_result["checks"]["no_original_segments"] = not has_original_segments
        validation_result["score"] += 30 if not has_original_segments else 0
        
        # 检查3: 脱敏后数据格式是否合理
        rule = self.masking_rules.get(data_type)
        if rule:
            algorithm = self.algorithms.get(rule.strategy)
            if algorithm:
                try:
                    # 重新脱敏原始数据，看结果是否一致
                    re_masked = algorithm.mask(original_data, rule.parameters)
                    is_consistent = re_masked == masked_data
                    validation_result["checks"]["consistent_masking"] = is_consistent
                    validation_result["score"] += 25 if is_consistent else 0
                except:
                    validation_result["checks"]["consistent_masking"] = False
        
        # 检查4: 敏感数据识别器验证
        is_still_sensitive = False
        identified_types = self.identifier.identify_sensitive_data(masked_data)
        for identified_type, _, _ in identified_types:
            if identified_type == data_type:
                is_still_sensitive = True
                break
        
        validation_result["checks"]["no_sensitive_data"] = not is_still_sensitive
        validation_result["score"] += 25 if not is_still_sensitive else 0
        
        # 计算总分
        validation_result["is_valid"] = validation_result["score"] >= 70
        validation_result["score"] = min(100, validation_result["score"])
        
        self.logger.info(f"脱敏验证完成: {data_type.value}, 得分: {validation_result['score']:.1f}")
        return validation_result

    def check_compliance(self, data_type: SensitiveDataType, 
                        standard: ComplianceStandard) -> Dict[str, Any]:
        """检查合规性
        
        Args:
            data_type: 数据类型
            standard: 合规标准
            
        Returns:
            合规性检查结果
        """
        compliance_result = {
            "compliant": True,
            "requirements": [],
            "recommendations": []
        }
        
        rule = self.masking_rules.get(data_type)
        if not rule:
            compliance_result["compliant"] = False
            compliance_result["requirements"].append(f"缺少 {data_type.value} 的脱敏规则")
            return compliance_result
        
        # GDPR 要求
        if standard == ComplianceStandard.GDPR:
            if rule.strategy in [MaskingStrategy.HASH, MaskingStrategy.ENCRYPTION, MaskingStrategy.TOKENIZATION]:
                compliance_result["requirements"].append(f"{data_type.value} 使用了强脱敏方法")
            else:
                compliance_result["recommendations"].append(f"建议对 {data_type.value} 使用更强的脱敏方法")
        
        # PCI DSS 要求
        elif standard == ComplianceStandard.PCI_DSS:
            if data_type == SensitiveDataType.CREDIT_CARD:
                if rule.strategy == MaskingStrategy.PARTIAL_MASK:
                    compliance_result["requirements"].append("信用卡号已部分遮蔽")
                else:
                    compliance_result["recommendations"].append("信用卡号建议使用部分遮蔽")
        
        # HIPAA 要求
        elif standard == ComplianceStandard.HIPAA:
            if data_type in [SensitiveDataType.MEDICAL_RECORD, SensitiveDataType.PERSONAL_ID]:
                if rule.strategy in [MaskingStrategy.HASH, MaskingStrategy.ENCRYPTION]:
                    compliance_result["requirements"].append(f"{data_type.value} 使用了安全脱敏方法")
                else:
                    compliance_result["recommendations"].append(f"建议对 {data_type.value} 使用安全脱敏方法")
        
        compliance_result["compliant"] = len(compliance_result["requirements"]) == 0
        
        self.logger.debug(f"合规性检查: {data_type.value} vs {standard.value} - {'通过' if compliance_result['compliant'] else '需改进'}")
        return compliance_result

    def get_statistics(self) -> Dict[str, Any]:
        """获取脱敏统计信息
        
        Returns:
            统计信息字典
        """
        stats_dict = asdict(self.statistics)
        
        # 转换枚举类型为字符串
        stats_dict["data_types_processed"] = {
            dt.value: count for dt, count in self.statistics.data_types_processed.items()
        }
        stats_dict["strategies_used"] = {
            strategy.value: count for strategy, count in self.statistics.strategies_used.items()
        }
        
        return stats_dict

    def reset_statistics(self):
        """重置统计信息"""
        self.statistics = MaskingStatistics()
        self.logger.info("统计信息已重置")

    def _update_statistics(self, start_time: float, data_type: SensitiveDataType, 
                          strategy: MaskingStrategy, success: bool):
        """更新统计信息"""
        processing_time = time.time() - start_time
        
        self.statistics.total_processed += 1
        self.statistics.processing_time += processing_time
        
        if success:
            self.statistics.successful_masked += 1
        else:
            self.statistics.failed_masked += 1
        
        # 更新数据类型统计
        if data_type in self.statistics.data_types_processed:
            self.statistics.data_types_processed[data_type] += 1
        else:
            self.statistics.data_types_processed[data_type] = 1
        
        # 更新策略统计
        if strategy in self.statistics.strategies_used:
            self.statistics.strategies_used[strategy] += 1
        else:
            self.statistics.strategies_used[strategy] = 1

    def clear_cache(self):
        """清理缓存"""
        with self._cache_lock:
            self._cache.clear()
        self.logger.info("缓存已清理")

    def export_masking_log(self, output_file: str, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None):
        """导出脱敏日志
        
        Args:
            output_file: 输出文件路径
            start_date: 开始日期
            end_date: 结束日期
        """
        # 这里应该从日志文件中提取脱敏相关的日志记录
        # 简化实现，创建一个示例日志
        log_data = {
            "export_time": datetime.now().isoformat(),
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "statistics": self.get_statistics(),
            "masking_rules": [
                {
                    "data_type": rule.data_type.value,
                    "strategy": rule.strategy.value,
                    "enabled": rule.enabled
                }
                for rule in self.masking_rules.values()
            ]
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"脱敏日志已导出到: {output_file}")
        except Exception as e:
            self.logger.error(f"导出脱敏日志失败: {e}")


# =============================================================================
# 装饰器函数
# =============================================================================

def mask_sensitive_data(data_type: Optional[SensitiveDataType] = None, 
                       strategy: Optional[MaskingStrategy] = None):
    """数据脱敏装饰器
    
    Args:
        data_type: 敏感数据类型
        strategy: 脱敏策略
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建临时的数据脱敏器
            masker = DataMasker()
            
            # 如果指定了策略，创建临时规则
            if data_type and strategy:
                temp_rule = MaskingRule(
                    data_type=data_type,
                    strategy=strategy,
                    parameters={}
                )
                masker.add_masking_rule(temp_rule)
            
            # 处理返回值
            result = func(*args, **kwargs)
            
            # 如果结果是字符串，进行脱敏
            if isinstance(result, str):
                return masker.mask_data(result, data_type).masked_data
            elif isinstance(result, list):
                # 如果是列表，对每个元素脱敏
                return [masker.mask_data(item, data_type).masked_data if isinstance(item, str) else item 
                       for item in result]
            else:
                return result
        
        return wrapper
    return decorator


# =============================================================================
# 测试用例
# =============================================================================

def test_data_masker():
    """数据脱敏器测试函数"""
    print("=== 数据脱敏器测试 ===\n")
    
    # 创建脱敏器实例
    masker = DataMasker()
    
    # 测试数据
    test_data = [
        ("张三", SensitiveDataType.NAME),
        ("13800138000", SensitiveDataType.PHONE),
        ("user@example.com", SensitiveDataType.EMAIL),
        ("110101199001011234", SensitiveDataType.PERSONAL_ID),
        ("4111111111111111", SensitiveDataType.CREDIT_CARD),
        ("192.168.1.1", SensitiveDataType.IP_ADDRESS),
    ]
    
    print("1. 单个数据脱敏测试:")
    for data, data_type in test_data:
        result = masker.mask_data(data, data_type)
        print(f"  原始: {data}")
        print(f"  脱敏: {result.masked_data}")
        print(f"  策略: {result.strategy.value}")
        print(f"  成功: {result.success}")
        print()
    
    print("2. 批量脱敏测试:")
    data_list = [item[0] for item in test_data]
    results = masker.mask_batch(data_list)
    for original, result in zip(data_list, results):
        print(f"  {original} -> {result.masked_data}")
    print()
    
    print("3. 自动识别测试:")
    auto_data = "张三的手机号是13800138000，邮箱是user@example.com"
    result = masker.mask_data(auto_data)
    print(f"  原始: {auto_data}")
    print(f"  脱敏: {result.masked_data}")
    print()
    
    print("4. 脱敏验证测试:")
    original = "13800138000"
    masked = masker.mask_data(original, SensitiveDataType.PHONE).masked_data
    validation = masker.validate_masking(original, masked, SensitiveDataType.PHONE)
    print(f"  验证结果: {validation}")
    print()
    
    print("5. 合规性检查测试:")
    compliance = masker.check_compliance(SensitiveDataType.PHONE, ComplianceStandard.GDPR)
    print(f"  GDPR合规性: {compliance}")
    print()
    
    print("6. 统计信息:")
    stats = masker.get_statistics()
    print(f"  总处理数: {stats['total_processed']}")
    print(f"  成功脱敏: {stats['successful_masked']}")
    print(f"  失败脱敏: {stats['failed_masked']}")
    print(f"  平均处理时间: {stats['processing_time']/max(1, stats['total_processed']):.4f}秒")
    print()
    
    print("7. 性能测试:")
    import time
    large_data_list = ["13800138000"] * 1000
    start_time = time.time()
    large_results = masker.mask_batch(large_data_list)
    end_time = time.time()
    print(f"  处理1000条数据耗时: {end_time - start_time:.2f}秒")
    print(f"  平均每条数据: {(end_time - start_time)/1000*1000:.2f}毫秒")
    print()


def test_masking_strategies():
    """测试不同脱敏策略"""
    print("=== 脱敏策略测试 ===\n")
    
    masker = DataMasker()
    test_data = "13800138000"
    
    strategies = [
        (MaskingStrategy.PARTIAL_MASK, {"keep_start": 3, "keep_end": 4, "mask_char": "*"}),
        (MaskingStrategy.REPLACE, {"replacement": "X", "preserve_length": True}),
        (MaskingStrategy.HASH, {"algorithm": "sha256"}),
        (MaskingStrategy.TOKENIZATION, {"token_length": 16}),
        (MaskingStrategy.ENCRYPTION, {"key": "test_key"}),
    ]
    
    for strategy, params in strategies:
        rule = MaskingRule(
            data_type=SensitiveDataType.PHONE,
            strategy=strategy,
            parameters=params
        )
        masker.add_masking_rule(rule)
        
        result = masker.mask_data(test_data, SensitiveDataType.PHONE)
        print(f"策略: {strategy.value}")
        print(f"参数: {params}")
        print(f"原始: {test_data}")
        print(f"脱敏: {result.masked_data}")
        print()


def test_compliance_standards():
    """测试合规性标准"""
    print("=== 合规性标准测试 ===\n")
    
    masker = DataMasker()
    standards = [ComplianceStandard.GDPR, ComplianceStandard.CCPA, 
                ComplianceStandard.PCI_DSS, ComplianceStandard.HIPAA]
    
    for standard in standards:
        print(f"合规标准: {standard.value}")
        for data_type in SensitiveDataType:
            compliance = masker.check_compliance(data_type, standard)
            status = "✓" if compliance["compliant"] else "✗"
            print(f"  {status} {data_type.value}: {compliance['compliant']}")
        print()


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===\n")
    
    masker = DataMasker()
    
    # 测试不同数据量的性能
    data_sizes = [100, 1000, 5000, 10000]
    
    for size in data_sizes:
        print(f"测试数据量: {size}")
        
        # 生成测试数据
        test_data = [f"1380013800{i%10:02d}" for i in range(size)]
        
        # 单线程测试
        start_time = time.time()
        results = masker.mask_batch(test_data, max_workers=1)
        single_thread_time = time.time() - start_time
        
        # 多线程测试
        start_time = time.time()
        results = masker.mask_batch(test_data, max_workers=4)
        multi_thread_time = time.time() - start_time
        
        print(f"  单线程: {single_thread_time:.2f}秒")
        print(f"  多线程: {multi_thread_time:.2f}秒")
        print(f"  加速比: {single_thread_time/multi_thread_time:.2f}x")
        print(f"  成功率: {sum(1 for r in results if r.success)/len(results)*100:.1f}%")
        print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("数据脱敏器 (DataMasker) v1.0.0")
    print("=" * 50)
    
    # 运行所有测试
    test_data_masker()
    test_masking_strategies()
    test_compliance_standards()
    performance_benchmark()
    
    print("所有测试完成!")