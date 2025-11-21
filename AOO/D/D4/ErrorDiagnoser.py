"""
D4错误诊断器
实现多层次错误诊断框架，提供智能错误识别、分析、评估、预防、修复和学习功能
"""

import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import statistics
import re


class ErrorLevel(Enum):
    """错误严重级别"""
    CRITICAL = "critical"      # 严重错误
    HIGH = "high"             # 高级错误
    MEDIUM = "medium"         # 中级错误
    LOW = "low"              # 低级错误
    INFO = "info"            # 信息性错误


class ErrorCategory(Enum):
    """错误分类"""
    SYSTEM = "system"                    # 系统错误
    NETWORK = "network"                  # 网络错误
    DATA = "data"                       # 数据错误
    LOGIC = "logic"                     # 逻辑错误
    RESOURCE = "resource"               # 资源错误
    SECURITY = "security"               # 安全错误
    PERFORMANCE = "performance"         # 性能错误
    CONFIGURATION = "configuration"     # 配置错误
    DEPENDENCY = "dependency"           # 依赖错误
    USER_INPUT = "user_input"           # 用户输入错误


class ErrorPattern(Enum):
    """错误模式"""
    RECURRENT = "recurrent"             # 重复性错误
    CASCADING = "cascading"             # 级联错误
    INTERMITTENT = "intermittent"       # 间歇性错误
    PROGRESSIVE = "progressive"         # 渐进性错误
    BURST = "burst"                     # 突发性错误
    SEASONAL = "seasonal"               # 季节性错误
    LOAD_RELATED = "load_related"       # 负载相关错误


@dataclass
class ErrorInfo:
    """错误信息结构"""
    error_id: str
    timestamp: datetime
    level: ErrorLevel
    category: ErrorCategory
    message: str
    stack_trace: str
    context: Dict[str, Any]
    source_module: str
    affected_components: List[str]
    error_code: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class ErrorAnalysis:
    """错误分析结果"""
    error_id: str
    root_cause: str
    contributing_factors: List[str]
    impact_assessment: Dict[str, Any]
    pattern_type: Optional[ErrorPattern]
    confidence_score: float
    recommendations: List[str]
    prevention_measures: List[str]
    recovery_actions: List[str]


@dataclass
class ErrorPatternInfo:
    """错误模式信息"""
    pattern_id: str
    pattern_type: ErrorPattern
    frequency: int
    last_occurrence: datetime
    severity_distribution: Dict[ErrorLevel, int]
    affected_modules: List[str]
    characteristics: Dict[str, Any]
    prevention_effectiveness: float


class ErrorDiagnoser:
    """错误诊断器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化错误诊断器"""
        self.config = config or self._default_config()
        self.logger = self._setup_logger()
        
        # 错误存储
        self.error_history: deque = deque(maxlen=self.config['max_error_history'])
        self.error_patterns: Dict[str, ErrorPatternInfo] = {}
        self.error_statistics: Dict[str, Any] = defaultdict(int)
        
        # 诊断引擎
        self.classification_engine = ErrorClassificationEngine()
        self.analysis_engine = ErrorAnalysisEngine()
        self.pattern_engine = ErrorPatternEngine()
        self.prevention_engine = ErrorPreventionEngine()
        self.recovery_engine = ErrorRecoveryEngine()
        self.learning_engine = ErrorLearningEngine()
        
        # 监控和阈值
        self.error_thresholds = self.config['error_thresholds']
        self.monitoring_active = True
        
        # 知识库
        self.error_knowledge_base = self._initialize_knowledge_base()
        
        self.logger.info("错误诊断器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'max_error_history': 10000,
            'pattern_detection_window': 3600,  # 1小时
            'error_thresholds': {
                'critical': 1,
                'high': 5,
                'medium': 20,
                'low': 100,
                'info': 500
            },
            'pattern_min_frequency': 3,
            'learning_rate': 0.1,
            'auto_recovery_enabled': True,
            'prevention_enabled': True
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"ErrorDiagnoser_{datetime.now().strftime('%Y%m%d')}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """初始化错误知识库"""
        return {
            'common_errors': {
                'connection_timeout': {
                    'category': ErrorCategory.NETWORK,
                    'typical_causes': ['网络延迟', '服务器负载', '防火墙阻断'],
                    'common_solutions': ['重试机制', '连接池优化', '超时时间调整']
                },
                'data_corruption': {
                    'category': ErrorCategory.DATA,
                    'typical_causes': ['存储故障', '并发写入', '程序异常'],
                    'common_solutions': ['数据校验', '备份恢复', '事务回滚']
                },
                'memory_leak': {
                    'category': ErrorCategory.RESOURCE,
                    'typical_causes': ['循环引用', '未释放资源', '缓存累积'],
                    'common_solutions': ['内存分析', '资源清理', '垃圾回收']
                }
            },
            'error_patterns': {
                'recurrent': {
                    'detection_criteria': ['相同错误重复出现', '时间间隔规律'],
                    'prevention_strategies': ['根本原因修复', '监控告警', '自动恢复']
                },
                'cascading': {
                    'detection_criteria': ['错误传播链', '依赖组件失败'],
                    'prevention_strategies': ['隔离机制', '熔断器模式', '降级策略']
                }
            }
        }
    
    def diagnose_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorAnalysis:
        """诊断错误"""
        try:
            # 构建错误信息
            error_info = self._build_error_info(error, context)
            
            # 存储错误历史
            self.error_history.append(error_info)
            
            # 执行多层次诊断
            analysis_result = self._comprehensive_diagnosis(error_info)
            
            # 更新统计信息
            self._update_error_statistics(error_info)
            
            # 模式识别和更新
            self._update_error_patterns(error_info)
            
            # 学习机制
            self.learning_engine.learn_from_error(error_info, analysis_result)
            
            self.logger.info(f"错误诊断完成: {error_info.error_id}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"错误诊断过程异常: {str(e)}")
            return self._create_fallback_analysis(str(e))
    
    def _build_error_info(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """构建错误信息"""
        error_id = self._generate_error_id(error, context)
        timestamp = datetime.now()
        
        # 错误分类和级别判断
        level, category = self.classification_engine.classify_error(error, context)
        
        # 提取堆栈信息
        stack_trace = traceback.format_exc() if error.__traceback__ else str(error)
        
        # 提取上下文信息
        context = context or {}
        context.update({
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': timestamp.isoformat()
        })
        
        # 确定源模块和影响组件
        source_module = self._extract_source_module(stack_trace)
        affected_components = self._identify_affected_components(error, context)
        
        return ErrorInfo(
            error_id=error_id,
            timestamp=timestamp,
            level=level,
            category=category,
            message=str(error),
            stack_trace=stack_trace,
            context=context,
            source_module=source_module,
            affected_components=affected_components
        )
    
    def _generate_error_id(self, error: Exception, context: Dict[str, Any] = None) -> str:
        """生成错误ID"""
        content = f"{type(error).__name__}_{str(error)}_{datetime.now().isoformat()}"
        if context:
            content += f"_{hashlib.md5(str(context).encode()).hexdigest()[:8]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _extract_source_module(self, stack_trace: str) -> str:
        """提取源模块"""
        lines = stack_trace.split('\n')
        for line in lines:
            if 'File "' in line and '.py"' in line:
                match = re.search(r'File "([^"]+)"', line)
                if match:
                    return match.group(1).split('/')[-1].replace('.py', '')
        return 'unknown'
    
    def _identify_affected_components(self, error: Exception, context: Dict[str, Any]) -> List[str]:
        """识别受影响的组件"""
        components = []
        
        # 基于错误类型识别
        error_type = type(error).__name__
        if 'Connection' in error_type or 'Network' in error_type:
            components.append('network')
        if 'Data' in error_type or 'Database' in error_type:
            components.append('database')
        if 'Memory' in error_type or 'Resource' in error_type:
            components.append('memory')
        
        # 基于上下文识别
        if context:
            if 'module' in context:
                components.append(context['module'])
            if 'service' in context:
                components.append(context['service'])
        
        return list(set(components))
    
    def _comprehensive_diagnosis(self, error_info: ErrorInfo) -> ErrorAnalysis:
        """综合诊断"""
        # 错误分类
        classification = self.classification_engine.analyze_classification(error_info)
        
        # 原因分析
        root_cause, contributing_factors = self.analysis_engine.analyze_root_cause(error_info)
        
        # 影响评估
        impact_assessment = self.analysis_engine.assess_impact(error_info)
        
        # 模式识别
        pattern_type = self.pattern_engine.identify_pattern(error_info)
        
        # 置信度评分
        confidence_score = self._calculate_confidence_score(error_info, classification)
        
        # 生成建议
        recommendations = self._generate_recommendations(error_info, root_cause, pattern_type)
        
        # 预防措施
        prevention_measures = self.prevention_engine.generate_prevention_measures(error_info, pattern_type)
        
        # 恢复行动
        recovery_actions = self.recovery_engine.generate_recovery_actions(error_info, root_cause)
        
        return ErrorAnalysis(
            error_id=error_info.error_id,
            root_cause=root_cause,
            contributing_factors=contributing_factors,
            impact_assessment=impact_assessment,
            pattern_type=pattern_type,
            confidence_score=confidence_score,
            recommendations=recommendations,
            prevention_measures=prevention_measures,
            recovery_actions=recovery_actions
        )
    
    def _calculate_confidence_score(self, error_info: ErrorInfo, classification: Dict[str, Any]) -> float:
        """计算置信度评分"""
        base_score = 0.5
        
        # 基于错误级别的调整
        level_weights = {
            ErrorLevel.CRITICAL: 0.2,
            ErrorLevel.HIGH: 0.15,
            ErrorLevel.MEDIUM: 0.1,
            ErrorLevel.LOW: 0.05,
            ErrorLevel.INFO: 0.0
        }
        base_score += level_weights.get(error_info.level, 0)
        
        # 基于历史相似错误的调整
        similar_errors = self._find_similar_errors(error_info)
        if similar_errors:
            base_score += min(0.3, len(similar_errors) * 0.05)
        
        # 基于知识库的匹配度
        kb_match_score = self._calculate_kb_match_score(error_info)
        base_score += kb_match_score * 0.2
        
        return min(1.0, base_score)
    
    def _find_similar_errors(self, error_info: ErrorInfo) -> List[ErrorInfo]:
        """查找相似错误"""
        similar_errors = []
        
        for historical_error in list(self.error_history)[-100:]:  # 最近100个错误
            if (historical_error.category == error_info.category and
                historical_error.source_module == error_info.source_module and
                abs((historical_error.timestamp - error_info.timestamp).total_seconds()) < 3600):
                similar_errors.append(historical_error)
        
        return similar_errors
    
    def _calculate_kb_match_score(self, error_info: ErrorInfo) -> float:
        """计算知识库匹配评分"""
        match_score = 0.0
        
        for error_name, error_data in self.error_knowledge_base['common_errors'].items():
            if error_data['category'] == error_info.category:
                # 检查错误消息匹配
                if any(keyword in error_info.message.lower() for keyword in error_name.split('_')):
                    match_score += 0.5
                
                # 检查上下文匹配
                context_keys = set(error_info.context.keys())
                if context_keys.intersection({'module', 'service', 'operation'}):
                    match_score += 0.3
        
        return min(1.0, match_score)
    
    def _generate_recommendations(self, error_info: ErrorInfo, root_cause: str, pattern_type: Optional[ErrorPattern]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于根本原因的建议
        if 'network' in root_cause.lower():
            recommendations.extend([
                "检查网络连接状态",
                "优化网络超时设置",
                "实施重试机制"
            ])
        elif 'memory' in root_cause.lower():
            recommendations.extend([
                "分析内存使用情况",
                "检查内存泄漏",
                "优化垃圾回收策略"
            ])
        elif 'data' in root_cause.lower():
            recommendations.extend([
                "验证数据完整性",
                "检查数据源质量",
                "实施数据校验机制"
            ])
        
        # 基于错误模式的建议
        if pattern_type == ErrorPattern.RECURRENT:
            recommendations.append("建立根本原因修复流程")
        elif pattern_type == ErrorPattern.CASCADING:
            recommendations.append("实施错误隔离机制")
        elif pattern_type == ErrorPattern.INTERMITTENT:
            recommendations.append("增加详细日志记录")
        
        # 基于错误级别的建议
        if error_info.level in [ErrorLevel.CRITICAL, ErrorLevel.HIGH]:
            recommendations.append("立即进行人工干预")
            recommendations.append("通知相关技术人员")
        
        return recommendations
    
    def _update_error_statistics(self, error_info: ErrorInfo):
        """更新错误统计信息"""
        self.error_statistics['total_errors'] += 1
        self.error_statistics[f'{error_info.level.value}_errors'] += 1
        self.error_statistics[f'{error_info.category.value}_errors'] += 1
        self.error_statistics[f'module_{error_info.source_module}_errors'] += 1
        
        # 按时间窗口统计
        hour_key = error_info.timestamp.strftime('%Y%m%d_%H')
        self.error_statistics[f'hourly_{hour_key}'] += 1
    
    def _update_error_patterns(self, error_info: ErrorInfo):
        """更新错误模式"""
        # 检查是否形成新的模式
        recent_errors = [e for e in self.error_history 
                        if (error_info.timestamp - e.timestamp).total_seconds() < self.config['pattern_detection_window']]
        
        if len(recent_errors) >= self.config['pattern_min_frequency']:
            # 分析模式特征
            pattern_characteristics = self.pattern_engine.analyze_pattern_characteristics(recent_errors)
            
            # 生成模式ID
            pattern_id = self._generate_pattern_id(recent_errors)
            
            # 更新或创建模式信息
            if pattern_id not in self.error_patterns:
                self.error_patterns[pattern_id] = ErrorPatternInfo(
                    pattern_id=pattern_id,
                    pattern_type=pattern_characteristics['type'],
                    frequency=len(recent_errors),
                    last_occurrence=error_info.timestamp,
                    severity_distribution=pattern_characteristics['severity_distribution'],
                    affected_modules=list(set(e.source_module for e in recent_errors)),
                    characteristics=pattern_characteristics['characteristics'],
                    prevention_effectiveness=0.0
                )
            else:
                # 更新现有模式
                existing_pattern = self.error_patterns[pattern_id]
                existing_pattern.frequency += 1
                existing_pattern.last_occurrence = error_info.timestamp
    
    def _generate_pattern_id(self, errors: List[ErrorInfo]) -> str:
        """生成模式ID"""
        # 基于错误类型、模块和时间的模式识别
        pattern_signature = f"{errors[0].category.value}_{errors[0].source_module}"
        for error in errors[1:]:
            pattern_signature += f"_{error.category.value}"
        
        return hashlib.md5(pattern_signature.encode()).hexdigest()[:16]
    
    def _create_fallback_analysis(self, fallback_message: str) -> ErrorAnalysis:
        """创建回退分析"""
        return ErrorAnalysis(
            error_id="fallback",
            root_cause="诊断过程异常",
            contributing_factors=[fallback_message],
            impact_assessment={'level': 'unknown', 'scope': 'unknown'},
            pattern_type=None,
            confidence_score=0.0,
            recommendations=["检查诊断系统状态", "联系技术支持"],
            prevention_measures=["增强系统稳定性"],
            recovery_actions=["重启诊断服务"]
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return dict(self.error_statistics)
    
    def get_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """获取错误模式信息"""
        patterns_dict = {}
        for pid, pattern in self.error_patterns.items():
            pattern_dict = asdict(pattern)
            # 转换枚举值为字符串
            if 'pattern_type' in pattern_dict and hasattr(pattern_dict['pattern_type'], 'value'):
                pattern_dict['pattern_type'] = pattern_dict['pattern_type'].value
            if 'severity_distribution' in pattern_dict:
                severity_dist = {}
                for level, count in pattern_dict['severity_distribution'].items():
                    if hasattr(level, 'value'):
                        severity_dist[level.value] = count
                    else:
                        severity_dist[str(level)] = count
                pattern_dict['severity_distribution'] = severity_dist
            patterns_dict[pid] = pattern_dict
        return patterns_dict
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的错误"""
        recent_errors = list(self.error_history)[-count:]
        errors_list = []
        for error in recent_errors:
            error_dict = asdict(error)
            # 转换枚举值为字符串
            if 'level' in error_dict and hasattr(error_dict['level'], 'value'):
                error_dict['level'] = error_dict['level'].value
            if 'category' in error_dict and hasattr(error_dict['category'], 'value'):
                error_dict['category'] = error_dict['category'].value
            errors_list.append(error_dict)
        return errors_list
    
    def export_diagnosis_report(self, file_path: str):
        """导出诊断报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_error_statistics(),
            'patterns': self.get_error_patterns(),
            'recent_errors': self.get_recent_errors(20),
            'knowledge_base_summary': {
                'common_errors_count': len(self.error_knowledge_base['common_errors']),
                'pattern_types_count': len(self.error_knowledge_base['error_patterns'])
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"诊断报告已导出到: {file_path}")


class ErrorClassificationEngine:
    """错误分类引擎"""
    
    def __init__(self):
        self.classification_rules = self._initialize_classification_rules()
    
    def _initialize_classification_rules(self) -> Dict[str, Dict]:
        """初始化分类规则"""
        return {
            'network_errors': {
                'patterns': ['connection', 'timeout', 'network', 'socket', 'dns'],
                'category': ErrorCategory.NETWORK,
                'level_mapping': {
                    'connection_refused': ErrorLevel.HIGH,
                    'timeout': ErrorLevel.MEDIUM,
                    'dns_error': ErrorLevel.MEDIUM
                }
            },
            'data_errors': {
                'patterns': ['data', 'database', 'sql', 'corruption', 'validation'],
                'category': ErrorCategory.DATA,
                'level_mapping': {
                    'corruption': ErrorLevel.CRITICAL,
                    'validation': ErrorLevel.MEDIUM,
                    'sql_error': ErrorLevel.HIGH
                }
            },
            'system_errors': {
                'patterns': ['system', 'os', 'permission', 'resource'],
                'category': ErrorCategory.SYSTEM,
                'level_mapping': {
                    'permission_denied': ErrorLevel.HIGH,
                    'resource_exhausted': ErrorLevel.CRITICAL
                }
            },
            'logic_errors': {
                'patterns': ['logic', 'algorithm', 'calculation', 'business'],
                'category': ErrorCategory.LOGIC,
                'level_mapping': {
                    'division_by_zero': ErrorLevel.HIGH,
                    'invalid_state': ErrorLevel.MEDIUM
                }
            }
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorLevel, ErrorCategory]:
        """分类错误"""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # 匹配分类规则
        for rule_name, rule in self.classification_rules.items():
            if any(pattern in error_message or pattern in error_type for pattern in rule['patterns']):
                category = rule['category']
                
                # 确定错误级别
                level = self._determine_error_level(error, rule['level_mapping'])
                return level, category
        
        # 默认分类
        return ErrorLevel.MEDIUM, ErrorCategory.SYSTEM
    
    def _determine_error_level(self, error: Exception, level_mapping: Dict[str, ErrorLevel]) -> ErrorLevel:
        """确定错误级别"""
        error_message = str(error).lower()
        
        for pattern, level in level_mapping.items():
            if pattern in error_message:
                return level
        
        # 基于错误类型的基本判断
        error_type = type(error).__name__
        if any(keyword in error_type for keyword in ['Critical', 'Fatal', 'System']):
            return ErrorLevel.CRITICAL
        elif any(keyword in error_type for keyword in ['Error', 'Exception']):
            return ErrorLevel.HIGH
        else:
            return ErrorLevel.MEDIUM
    
    def analyze_classification(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """分析分类结果"""
        return {
            'primary_category': error_info.category,
            'confidence': self._calculate_classification_confidence(error_info),
            'alternative_categories': self._get_alternative_categories(error_info)
        }
    
    def _calculate_classification_confidence(self, error_info: ErrorInfo) -> float:
        """计算分类置信度"""
        # 基于匹配规则数量计算置信度
        error_message = error_info.message.lower()
        matched_rules = 0
        
        for rule in self.classification_rules.values():
            if any(pattern in error_message for pattern in rule['patterns']):
                matched_rules += 1
        
        return min(1.0, matched_rules / len(self.classification_rules))
    
    def _get_alternative_categories(self, error_info: ErrorInfo) -> List[ErrorCategory]:
        """获取备选分类"""
        alternatives = []
        error_message = error_info.message.lower()
        
        for rule in self.classification_rules.values():
            if any(pattern in error_message for pattern in rule['patterns']):
                if rule['category'] != error_info.category:
                    alternatives.append(rule['category'])
        
        return alternatives


class ErrorAnalysisEngine:
    """错误分析引擎"""
    
    def analyze_root_cause(self, error_info: ErrorInfo) -> Tuple[str, List[str]]:
        """分析根本原因"""
        # 基于错误类型和模式分析根本原因
        root_cause = self._identify_root_cause(error_info)
        contributing_factors = self._identify_contributing_factors(error_info)
        
        return root_cause, contributing_factors
    
    def _identify_root_cause(self, error_info: ErrorInfo) -> str:
        """识别根本原因"""
        error_type = type(error_info.message).__name__
        message = error_info.message.lower()
        
        # 网络相关错误
        if error_info.category == ErrorCategory.NETWORK:
            if 'timeout' in message:
                return "网络连接超时 - 可能是网络延迟或服务器响应慢"
            elif 'connection' in message:
                return "网络连接失败 - 可能是网络中断或服务器不可达"
            elif 'dns' in message:
                return "DNS解析失败 - 域名解析配置问题"
        
        # 数据相关错误
        elif error_info.category == ErrorCategory.DATA:
            if 'corruption' in message:
                return "数据损坏 - 存储设备故障或并发写入冲突"
            elif 'validation' in message:
                return "数据验证失败 - 输入数据格式或内容不符合要求"
            elif 'sql' in message:
                return "数据库操作错误 - SQL语法错误或权限问题"
        
        # 系统相关错误
        elif error_info.category == ErrorCategory.SYSTEM:
            if 'memory' in message:
                return "内存不足 - 内存泄漏或配置不足"
            elif 'disk' in message:
                return "磁盘空间不足 - 日志文件过大或数据增长过快"
            elif 'permission' in message:
                return "权限不足 - 用户权限配置错误"
        
        # 资源相关错误
        elif error_info.category == ErrorCategory.RESOURCE:
            if 'resource' in message:
                return "资源耗尽 - 连接池、线程池等资源不足"
            elif 'leak' in message:
                return "资源泄漏 - 资源未正确释放"
        
        # 默认分析
        return f"未识别的{error_info.category.value}错误，需要进一步分析"
    
    def _identify_contributing_factors(self, error_info: ErrorInfo) -> List[str]:
        """识别促成因素"""
        factors = []
        
        # 基于错误上下文的分析
        context = error_info.context
        
        # 时间相关因素
        hour = error_info.timestamp.hour
        if 0 <= hour <= 6:
            factors.append("夜间运行 - 系统负载可能较低但监控不足")
        elif 9 <= hour <= 17:
            factors.append("工作时间 - 用户活动频繁，负载较高")
        
        # 负载相关因素
        if 'load' in context.get('system_state', {}):
            factors.append("高负载状态 - 系统资源紧张")
        
        # 配置相关因素
        if 'config_error' in error_info.message.lower():
            factors.append("配置错误 - 系统参数设置不当")
        
        # 依赖相关因素
        if len(error_info.affected_components) > 1:
            factors.append("多组件依赖 - 错误可能通过依赖链传播")
        
        return factors
    
    def assess_impact(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """评估错误影响"""
        impact = {
            'severity': error_info.level.value,
            'scope': self._assess_scope(error_info),
            'duration': self._estimate_duration(error_info),
            'user_impact': self._assess_user_impact(error_info),
            'business_impact': self._assess_business_impact(error_info)
        }
        
        return impact
    
    def _assess_scope(self, error_info: ErrorInfo) -> str:
        """评估影响范围"""
        affected_count = len(error_info.affected_components)
        
        if affected_count >= 5:
            return "广泛影响 - 多个核心组件受影响"
        elif affected_count >= 3:
            return "中等影响 - 部分关键组件受影响"
        elif affected_count >= 1:
            return "局部影响 - 单个组件受影响"
        else:
            return "最小影响 - 影响范围有限"
    
    def _estimate_duration(self, error_info: ErrorInfo) -> str:
        """估计持续时间"""
        if error_info.level == ErrorLevel.CRITICAL:
            return "立即处理 - 需要紧急修复"
        elif error_info.level == ErrorLevel.HIGH:
            return "短时间 - 几小时内需要处理"
        elif error_info.level == ErrorLevel.MEDIUM:
            return "中等时间 - 当天内处理"
        else:
            return "可延后 - 可以在下次维护时处理"
    
    def _assess_user_impact(self, error_info: ErrorInfo) -> str:
        """评估用户影响"""
        if error_info.level in [ErrorLevel.CRITICAL, ErrorLevel.HIGH]:
            return "严重影响 - 用户无法正常使用服务"
        elif error_info.level == ErrorLevel.MEDIUM:
            return "中等影响 - 部分功能可能受影响"
        else:
            return "轻微影响 - 用户体验可能略有下降"
    
    def _assess_business_impact(self, error_info: ErrorInfo) -> str:
        """评估业务影响"""
        if error_info.level == ErrorLevel.CRITICAL:
            return "严重业务影响 - 可能导致业务中断和损失"
        elif error_info.level == ErrorLevel.HIGH:
            return "重要业务影响 - 影响关键业务流程"
        elif error_info.level == ErrorLevel.MEDIUM:
            return "一般业务影响 - 可能影响部分业务指标"
        else:
            return "微小业务影响 - 对业务影响很小"


class ErrorPatternEngine:
    """错误模式引擎"""
    
    def identify_pattern(self, error_info: ErrorInfo) -> Optional[ErrorPattern]:
        """识别错误模式"""
        # 获取最近的错误进行分析
        recent_errors = self._get_recent_errors_for_pattern_analysis(error_info)
        
        if len(recent_errors) < 2:
            return None
        
        # 模式识别逻辑
        if self._is_recurrent_pattern(recent_errors):
            return ErrorPattern.RECURRENT
        elif self._is_cascading_pattern(recent_errors):
            return ErrorPattern.CASCADING
        elif self._is_intermittent_pattern(recent_errors):
            return ErrorPattern.INTERMITTENT
        elif self._is_burst_pattern(recent_errors):
            return ErrorPattern.BURST
        elif self._is_load_related_pattern(recent_errors):
            return ErrorPattern.LOAD_RELATED
        
        return None
    
    def _get_recent_errors_for_pattern_analysis(self, current_error: ErrorInfo) -> List[ErrorInfo]:
        """获取用于模式分析的最近错误"""
        # 获取最近1小时内的错误
        window_start = current_error.timestamp - timedelta(hours=1)
        recent_errors = [e for e in [current_error] if e.timestamp >= window_start]
        
        # 添加历史错误
        for historical_error in reversed(list(current_error.__class__.__module__.split('.'))):
            pass  # 这里需要从错误历史中获取
        
        return recent_errors
    
    def _is_recurrent_pattern(self, errors: List[ErrorInfo]) -> bool:
        """判断是否为重复模式"""
        if len(errors) < 3:
            return False
        
        # 检查相同或相似的错误是否重复出现
        error_signatures = [f"{e.category.value}_{e.source_module}" for e in errors]
        signature_counts = defaultdict(int)
        
        for signature in error_signatures:
            signature_counts[signature] += 1
        
        # 如果某个签名出现频率超过50%，认为是重复模式
        max_frequency = max(signature_counts.values())
        return max_frequency / len(errors) >= 0.5
    
    def _is_cascading_pattern(self, errors: List[ErrorInfo]) -> bool:
        """判断是否为级联模式"""
        # 检查错误是否按依赖顺序出现
        component_dependencies = self._get_component_dependencies()
        
        for i in range(len(errors) - 1):
            current_components = set(errors[i].affected_components)
            next_components = set(errors[i + 1].affected_components)
            
            # 如果下一个错误的组件依赖于当前错误的组件，则可能是级联
            if any(comp in component_dependencies.get(dep, []) for comp in current_components 
                   for dep in next_components):
                return True
        
        return False
    
    def _is_intermittent_pattern(self, errors: List[ErrorInfo]) -> bool:
        """判断是否为间歇模式"""
        if len(errors) < 3:
            return False
        
        # 检查错误发生的时间间隔
        intervals = []
        for i in range(1, len(errors)):
            interval = (errors[i].timestamp - errors[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        # 如果时间间隔变化很大，可能是间歇性错误
        if len(intervals) > 1:
            std_dev = statistics.stdev(intervals)
            mean_interval = statistics.mean(intervals)
            coefficient_of_variation = std_dev / mean_interval if mean_interval > 0 else 0
            
            return coefficient_of_variation > 0.5
        
        return False
    
    def _is_burst_pattern(self, errors: List[ErrorInfo]) -> bool:
        """判断是否为突发模式"""
        if len(errors) < 5:
            return False
        
        # 检查短时间内是否集中出现大量错误
        time_window = timedelta(minutes=10)
        burst_threshold = 5
        
        for i in range(len(errors)):
            window_errors = [e for e in errors 
                           if abs((e.timestamp - errors[i].timestamp).total_seconds()) <= time_window.total_seconds()]
            if len(window_errors) >= burst_threshold:
                return True
        
        return False
    
    def _is_load_related_pattern(self, errors: List[ErrorInfo]) -> bool:
        """判断是否为负载相关模式"""
        # 检查错误是否与系统负载相关
        load_indicators = ['timeout', 'resource', 'memory', 'cpu', 'load']
        
        for error in errors:
            if any(indicator in error.message.lower() for indicator in load_indicators):
                return True
        
        return False
    
    def _get_component_dependencies(self) -> Dict[str, List[str]]:
        """获取组件依赖关系"""
        # 简化的组件依赖关系映射
        return {
            'database': ['cache', 'api'],
            'api': ['web', 'mobile'],
            'cache': ['api'],
            'web': ['cdn'],
            'mobile': ['api']
        }
    
    def analyze_pattern_characteristics(self, errors: List[ErrorInfo]) -> Dict[str, Any]:
        """分析模式特征"""
        characteristics = {
            'type': self.identify_pattern(errors[0]) if errors else None,
            'severity_distribution': self._calculate_severity_distribution(errors),
            'characteristics': self._extract_pattern_characteristics(errors)
        }
        
        return characteristics
    
    def _calculate_severity_distribution(self, errors: List[ErrorInfo]) -> Dict[ErrorLevel, int]:
        """计算严重级别分布"""
        distribution = defaultdict(int)
        for error in errors:
            distribution[error.level] += 1
        return dict(distribution)
    
    def _extract_pattern_characteristics(self, errors: List[ErrorInfo]) -> Dict[str, Any]:
        """提取模式特征"""
        return {
            'error_count': len(errors),
            'time_span_minutes': (errors[-1].timestamp - errors[0].timestamp).total_seconds() / 60,
            'affected_modules': list(set(e.source_module for e in errors)),
            'primary_category': max(set(e.category for e in errors), key=[e.category for e in errors].count),
            'avg_severity': statistics.mean([self._severity_to_numeric(e.level) for e in errors])
        }
    
    def _severity_to_numeric(self, level: ErrorLevel) -> int:
        """将错误级别转换为数值"""
        mapping = {
            ErrorLevel.CRITICAL: 5,
            ErrorLevel.HIGH: 4,
            ErrorLevel.MEDIUM: 3,
            ErrorLevel.LOW: 2,
            ErrorLevel.INFO: 1
        }
        return mapping.get(level, 3)


class ErrorPreventionEngine:
    """错误预防引擎"""
    
    def generate_prevention_measures(self, error_info: ErrorInfo, pattern_type: Optional[ErrorPattern]) -> List[str]:
        """生成预防措施"""
        measures = []
        
        # 基于错误类别的预防措施
        measures.extend(self._get_category_specific_measures(error_info.category))
        
        # 基于错误模式的预防措施
        if pattern_type:
            measures.extend(self._get_pattern_specific_measures(pattern_type))
        
        # 基于错误级别的预防措施
        measures.extend(self._get_level_specific_measures(error_info.level))
        
        # 基于根本原因的预防措施
        measures.extend(self._get_root_cause_prevention(error_info))
        
        return list(set(measures))  # 去重
    
    def _get_category_specific_measures(self, category: ErrorCategory) -> List[str]:
        """获取类别特定的预防措施"""
        measures = {
            ErrorCategory.NETWORK: [
                "实施连接池管理",
                "配置合理的超时时间",
                "部署网络监控",
                "实施重试机制"
            ],
            ErrorCategory.DATA: [
                "实施数据校验",
                "配置数据备份",
                "使用事务机制",
                "监控数据质量"
            ],
            ErrorCategory.SYSTEM: [
                "监控系统资源",
                "配置资源限制",
                "实施健康检查",
                "配置自动重启"
            ],
            ErrorCategory.RESOURCE: [
                "优化资源使用",
                "实施资源监控",
                "配置资源告警",
                "清理无用资源"
            ],
            ErrorCategory.SECURITY: [
                "加强权限控制",
                "实施安全审计",
                "配置防火墙",
                "加密敏感数据"
            ]
        }
        
        return measures.get(category, ["加强系统监控", "完善错误处理"])
    
    def _get_pattern_specific_measures(self, pattern_type: ErrorPattern) -> List[str]:
        """获取模式特定的预防措施"""
        measures = {
            ErrorPattern.RECURRING: [
                "建立根本原因分析流程",
                "实施预防性维护",
                "加强监控告警",
                "建立知识库"
            ],
            ErrorPattern.CASCADING: [
                "实施错误隔离",
                "配置熔断器",
                "设计降级策略",
                "加强依赖管理"
            ],
            ErrorPattern.INTERMITTENT: [
                "增加详细日志",
                "延长监控时间",
                "分析触发条件",
                "实施预防性检查"
            ],
            ErrorPattern.BURST: [
                "实施流量控制",
                "配置负载均衡",
                "优化资源分配",
                "建立应急预案"
            ],
            ErrorPattern.LOAD_RELATED: [
                "优化性能",
                "扩容资源",
                "实施缓存",
                "优化算法"
            ]
        }
        
        return measures.get(pattern_type, ["加强系统稳定性"])
    
    def _get_level_specific_measures(self, level: ErrorLevel) -> List[str]:
        """获取级别特定的预防措施"""
        measures = {
            ErrorLevel.CRITICAL: [
                "立即建立监控",
                "准备应急预案",
                "通知相关人员",
                "准备回滚方案"
            ],
            ErrorLevel.HIGH: [
                "加强监控频率",
                "准备快速响应",
                "分析影响范围",
                "制定修复计划"
            ],
            ErrorLevel.MEDIUM: [
                "定期检查",
                "优化配置",
                "改进流程",
                "培训人员"
            ],
            ErrorLevel.LOW: [
                "记录日志",
                "定期审查",
                "持续改进",
                "文档更新"
            ]
        }
        
        return measures.get(level, ["持续监控"])
    
    def _get_root_cause_prevention(self, error_info: ErrorInfo) -> List[str]:
        """基于根本原因的预防措施"""
        message = error_info.message.lower()
        
        if 'timeout' in message:
            return ["优化网络配置", "增加超时时间", "实施异步处理"]
        elif 'memory' in message:
            return ["优化内存使用", "增加内存监控", "实施内存清理"]
        elif 'connection' in message:
            return ["检查网络状态", "优化连接管理", "实施健康检查"]
        elif 'data' in message:
            return ["加强数据验证", "实施数据备份", "优化数据处理"]
        
        return ["深入分析根本原因", "制定针对性预防措施"]


class ErrorRecoveryEngine:
    """错误恢复引擎"""
    
    def generate_recovery_actions(self, error_info: ErrorInfo, root_cause: str) -> List[str]:
        """生成恢复行动"""
        actions = []
        
        # 即时恢复行动
        actions.extend(self._get_immediate_actions(error_info))
        
        # 基于根本原因的恢复行动
        actions.extend(self._get_root_cause_recovery(error_info, root_cause))
        
        # 基于错误级别的恢复行动
        actions.extend(self._get_level_specific_recovery(error_info.level))
        
        # 验证和测试行动
        actions.extend(self._get_verification_actions(error_info))
        
        return actions
    
    def _get_immediate_actions(self, error_info: ErrorInfo) -> List[str]:
        """获取即时行动"""
        actions = ["记录错误详情", "通知相关人员"]
        
        if error_info.level in [ErrorLevel.CRITICAL, ErrorLevel.HIGH]:
            actions.extend([
                "立即停止受影响服务",
                "启动应急预案",
                "准备回滚操作"
            ])
        
        return actions
    
    def _get_root_cause_recovery(self, error_info: ErrorInfo, root_cause: str) -> List[str]:
        """基于根本原因的恢复行动"""
        if 'network' in root_cause.lower():
            return [
                "检查网络连接",
                "重启网络服务",
                "验证DNS配置",
                "测试网络连通性"
            ]
        elif 'memory' in root_cause.lower():
            return [
                "清理内存缓存",
                "重启应用服务",
                "检查内存泄漏",
                "优化内存配置"
            ]
        elif 'data' in root_cause.lower():
            return [
                "验证数据完整性",
                "从备份恢复数据",
                "检查数据库连接",
                "验证数据格式"
            ]
        elif 'resource' in root_cause.lower():
            return [
                "释放无用资源",
                "重启相关服务",
                "优化资源分配",
                "监控资源使用"
            ]
        
        return ["深入分析根本原因", "制定针对性修复方案"]
    
    def _get_level_specific_recovery(self, level: ErrorLevel) -> List[str]:
        """基于级别的恢复行动"""
        actions = {
            ErrorLevel.CRITICAL: [
                "立即执行修复",
                "验证修复效果",
                "监控恢复状态",
                "准备应急预案"
            ],
            ErrorLevel.HIGH: [
                "尽快执行修复",
                "测试修复方案",
                "监控修复过程",
                "通知相关团队"
            ],
            ErrorLevel.MEDIUM: [
                "计划修复时间",
                "准备修复方案",
                "测试修复效果",
                "更新文档"
            ],
            ErrorLevel.LOW: [
                "安排合适时间修复",
                "记录修复过程",
                "验证修复结果",
                "总结经验"
            ]
        }
        
        return actions.get(level, ["评估修复优先级"])
    
    def _get_verification_actions(self, error_info: ErrorInfo) -> List[str]:
        """获取验证行动"""
        return [
            "验证服务状态",
            "测试核心功能",
            "检查相关日志",
            "确认错误已解决"
        ]


class ErrorLearningEngine:
    """错误学习引擎"""
    
    def __init__(self):
        self.learning_history = []
        self.pattern_effectiveness = defaultdict(float)
    
    def learn_from_error(self, error_info: ErrorInfo, analysis: ErrorAnalysis):
        """从错误中学习"""
        learning_record = {
            'timestamp': datetime.now(),
            'error_info': error_info,
            'analysis': analysis,
            'outcome': None  # 待后续填充
        }
        
        self.learning_history.append(learning_record)
        
        # 更新模式有效性
        if analysis.pattern_type:
            self._update_pattern_effectiveness(analysis.pattern_type, analysis)
        
        # 更新预防措施有效性
        self._update_prevention_effectiveness(error_info, analysis)
    
    def _update_pattern_effectiveness(self, pattern_type: ErrorPattern, analysis: ErrorAnalysis):
        """更新模式有效性"""
        # 基于建议的采纳情况和效果来评估有效性
        # 这里简化处理，实际应该基于后续的监控数据
        effectiveness = analysis.confidence_score
        
        self.pattern_effectiveness[pattern_type.value] = (
            self.pattern_effectiveness[pattern_type.value] * 0.9 + effectiveness * 0.1
        )
    
    def _update_prevention_effectiveness(self, error_info: ErrorInfo, analysis: ErrorAnalysis):
        """更新预防措施有效性"""
        # 基于错误是否重复出现来评估预防措施的有效性
        # 这里简化处理
        pass
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        return {
            'total_learning_records': len(self.learning_history),
            'pattern_effectiveness': dict(self.pattern_effectiveness),
            'recent_insights': self._extract_recent_insights(),
            'recommendations': self._generate_learning_recommendations()
        }
    
    def _extract_recent_insights(self) -> List[str]:
        """提取最近的洞察"""
        insights = []
        
        if len(self.learning_history) >= 10:
            recent_records = self.learning_history[-10:]
            
            try:
                # 分析最常见的错误类别
                categories = []
                levels = []
                
                for record in recent_records:
                    if isinstance(record['error_info'], ErrorInfo):
                        categories.append(record['error_info'].category)
                        levels.append(record['error_info'].level)
                    elif isinstance(record['error_info'], str):
                        # 如果存储的是错误ID，跳过这部分分析
                        continue
                
                if categories:
                    most_common_category = max(set(categories), key=categories.count)
                    insights.append(f"最近最常见的错误类别: {most_common_category.value}")
                
                if levels:
                    most_common_level = max(set(levels), key=levels.count)
                    insights.append(f"最近最常见的错误级别: {most_common_level.value}")
                
            except Exception as e:
                # 如果分析失败，提供通用洞察
                insights.append("学习数据正在积累中，请稍后再查看详细洞察")
        
        return insights
    
    def _generate_learning_recommendations(self) -> List[str]:
        """生成学习建议"""
        recommendations = []
        
        if self.pattern_effectiveness:
            # 找出最有效的模式预防措施
            most_effective_pattern = max(self.pattern_effectiveness.items(), key=lambda x: x[1])
            recommendations.append(f"加强{most_effective_pattern[0]}模式的预防措施")
            
            # 找出效果不佳的模式
            least_effective_patterns = [pattern for pattern, effectiveness in self.pattern_effectiveness.items() 
                                      if effectiveness < 0.5]
            if least_effective_patterns:
                recommendations.append(f"改进以下模式的预防策略: {', '.join(least_effective_patterns)}")
        
        return recommendations


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建错误诊断器实例
    diagnoser = ErrorDiagnoser()
    
    # 模拟各种错误进行测试
    test_errors = [
        ConnectionError("连接到数据库失败"),
        MemoryError("内存不足"),
        ValueError("数据验证失败"),
        TimeoutError("网络请求超时")
    ]
    
    print("=== D4错误诊断器测试 ===\n")
    
    for i, error in enumerate(test_errors, 1):
        print(f"测试错误 {i}: {type(error).__name__}")
        print(f"错误信息: {str(error)}")
        
        # 诊断错误
        context = {
            'module': 'test_module',
            'operation': 'test_operation',
            'user_id': 'test_user'
        }
        
        analysis = diagnoser.diagnose_error(error, context)
        
        print(f"错误ID: {analysis.error_id}")
        print(f"根本原因: {analysis.root_cause}")
        print(f"影响评估: {analysis.impact_assessment}")
        print(f"置信度: {analysis.confidence_score:.2f}")
        print(f"建议措施: {analysis.recommendations[:3]}")  # 只显示前3个建议
        print("-" * 50)
    
    # 导出诊断报告
    diagnoser.export_diagnosis_report("error_diagnosis_report.json")
    
    print("\n=== 统计信息 ===")
    stats = diagnoser.get_error_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== 错误模式 ===")
    patterns = diagnoser.get_error_patterns()
    for pattern_id, pattern_info in patterns.items():
        print(f"模式ID: {pattern_id}")
        print(f"模式类型: {pattern_info['pattern_type']}")
        print(f"频率: {pattern_info['frequency']}")
    
    print("\nD4错误诊断器测试完成！")