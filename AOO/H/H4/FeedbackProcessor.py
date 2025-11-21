#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H4反馈处理器
实现反馈信息的收集、分类、分析、处理策略制定、效果评估和历史跟踪
"""

import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """反馈类型枚举"""
    BUG = "bug"                    # 错误反馈
    FEATURE = "feature"            # 功能建议
    IMPROVEMENT = "improvement"    # 改进建议
    PERFORMANCE = "performance"    # 性能问题
    UI_UX = "ui_ux"               # 界面体验
    DOCUMENTATION = "documentation" # 文档问题
    SECURITY = "security"         # 安全问题
    COMPLIANCE = "compliance"     # 合规问题


class Priority(Enum):
    """优先级枚举"""
    CRITICAL = 1    # 关键
    HIGH = 2        # 高
    MEDIUM = 3      # 中
    LOW = 4         # 低


class Status(Enum):
    """处理状态枚举"""
    PENDING = "pending"          # 待处理
    IN_PROGRESS = "in_progress"  # 处理中
    COMPLETED = "completed"      # 已完成
    CLOSED = "closed"           # 已关闭
    REJECTED = "rejected"       # 已拒绝


@dataclass
class FeedbackItem:
    """反馈项目数据模型"""
    id: str
    title: str
    description: str
    feedback_type: FeedbackType
    priority: Priority
    status: Status
    submit_time: datetime.datetime
    submitter: str
    assignee: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_history: List[Dict] = field(default_factory=list)
    evaluation_score: Optional[float] = None
    resolution_time: Optional[datetime.datetime] = None


@dataclass
class ProcessingStrategy:
    """处理策略数据模型"""
    id: str
    name: str
    description: str
    applicable_types: List[FeedbackType]
    applicable_priorities: List[Priority]
    steps: List[Dict[str, Any]]
    estimated_time: int  # 预计处理时间（小时）
    success_rate: float = 0.0
    usage_count: int = 0


@dataclass
class ProcessingResult:
    """处理结果数据模型"""
    feedback_id: str
    strategy_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    status: Status = Status.PENDING
    result: Optional[Dict[str, Any]] = None
    issues_encountered: List[str] = field(default_factory=list)
    satisfaction_score: Optional[float] = None


class FeedbackCollector:
    """反馈信息收集器"""
    
    def __init__(self):
        self.sources = {}
        self.collection_rules = {}
        
    def register_source(self, source_id: str, source_config: Dict[str, Any]):
        """注册反馈源"""
        self.sources[source_id] = source_config
        logger.info(f"注册反馈源: {source_id}")
    
    def collect_feedback(self, source_id: str, filters: Optional[Dict] = None) -> List[Dict]:
        """从指定源收集反馈"""
        if source_id not in self.sources:
            raise ValueError(f"未知的反馈源: {source_id}")
        
        source_config = self.sources[source_id]
        # 模拟从不同源收集反馈
        feedbacks = []
        
        if source_config.get('type') == 'api':
            # 从API收集
            feedbacks = self._collect_from_api(source_config, filters)
        elif source_config.get('type') == 'database':
            # 从数据库收集
            feedbacks = self._collect_from_database(source_config, filters)
        elif source_config.get('type') == 'file':
            # 从文件收集
            feedbacks = self._collect_from_file(source_config, filters)
        
        logger.info(f"从源 {source_id} 收集到 {len(feedbacks)} 条反馈")
        return feedbacks
    
    def _collect_from_api(self, config: Dict, filters: Optional[Dict]) -> List[Dict]:
        """从API收集反馈"""
        # 模拟API调用
        return [
            {
                'id': 'fb_001',
                'title': '页面加载缓慢',
                'description': '首页加载时间超过5秒',
                'type': 'performance',
                'priority': 'high',
                'submitter': 'user001',
                'tags': ['performance', 'loading']
            }
        ]
    
    def _collect_from_database(self, config: Dict, filters: Optional[Dict]) -> List[Dict]:
        """从数据库收集反馈"""
        # 模拟数据库查询
        return [
            {
                'id': 'fb_002',
                'title': '搜索功能异常',
                'description': '搜索结果不准确',
                'type': 'bug',
                'priority': 'medium',
                'submitter': 'user002',
                'tags': ['search', 'bug']
            }
        ]
    
    def _collect_from_file(self, config: Dict, filters: Optional[Dict]) -> List[Dict]:
        """从文件收集反馈"""
        # 模拟文件读取
        return [
            {
                'id': 'fb_003',
                'title': '建议增加导出功能',
                'description': '希望支持Excel导出',
                'type': 'feature',
                'priority': 'low',
                'submitter': 'user003',
                'tags': ['export', 'feature']
            }
        ]


class FeedbackClassifier:
    """反馈分类器"""
    
    def __init__(self):
        self.type_keywords = {
            FeedbackType.BUG: ['错误', 'bug', '异常', '失败', '崩溃', '无法', '问题'],
            FeedbackType.FEATURE: ['功能', 'feature', '增加', '添加', '新功能', '建议'],
            FeedbackType.IMPROVEMENT: ['改进', '优化', '改善', '提升', '更好'],
            FeedbackType.PERFORMANCE: ['性能', '慢', '卡顿', '响应', '加载', '速度'],
            FeedbackType.UI_UX: ['界面', 'UI', 'UX', '美观', '易用', '用户体验'],
            FeedbackType.DOCUMENTATION: ['文档', '说明', '帮助', '教程', '手册'],
            FeedbackType.SECURITY: ['安全', '漏洞', '加密', '权限', '认证'],
            FeedbackType.COMPLIANCE: ['合规', '标准', '规范', '法律', '政策']
        }
        
        self.priority_keywords = {
            Priority.CRITICAL: ['紧急', 'critical', '严重', '系统崩溃', '数据丢失'],
            Priority.HIGH: ['重要', 'high', '影响大', '很多用户', '频繁'],
            Priority.MEDIUM: ['一般', 'medium', '普通', '偶尔'],
            Priority.LOW: ['轻微', 'low', '小问题', '建议']
        }
    
    def classify_feedback(self, feedback: Dict[str, Any]) -> Tuple[FeedbackType, Priority]:
        """对反馈进行分类"""
        text = (feedback.get('title', '') + ' ' + feedback.get('description', '')).lower()
        
        # 类型分类
        type_score = {}
        for fb_type, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                type_score[fb_type] = score
        
        feedback_type = max(type_score.items(), key=lambda x: x[1])[0] if type_score else FeedbackType.IMPROVEMENT
        
        # 优先级分类
        priority_score = {}
        for priority, keywords in self.priority_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                priority_score[priority] = score
        
        priority = max(priority_score.items(), key=lambda x: x[1])[0] if priority_score else Priority.MEDIUM
        
        return feedback_type, priority
    
    def auto_tag_feedback(self, feedback: Dict[str, Any]) -> List[str]:
        """自动标记反馈标签"""
        text = (feedback.get('title', '') + ' ' + feedback.get('description', '')).lower()
        tags = []
        
        # 基于关键词自动标记
        keyword_tags = {
            '登录': ['auth'],
            '注册': ['auth'],
            '支付': ['payment'],
            '订单': ['order'],
            '搜索': ['search'],
            '导出': ['export'],
            '导入': ['import'],
            '报表': ['report'],
            '数据': ['data'],
            '用户': ['user'],
            '管理': ['admin']
        }
        
        for keyword, tag in keyword_tags.items():
            if keyword in text:
                tags.extend(tag)
        
        return list(set(tags))  # 去重


class FeedbackAnalyzer:
    """反馈分析器"""
    
    def __init__(self):
        self.patterns = {}
        self.trends = {}
        
    def analyze_patterns(self, feedbacks: List[FeedbackItem]) -> Dict[str, Any]:
        """分析反馈模式"""
        analysis = {
            'type_distribution': Counter(f.feedback_type for f in feedbacks),
            'priority_distribution': Counter(f.priority for f in feedbacks),
            'status_distribution': Counter(f.status for f in feedbacks),
            'submitter_analysis': Counter(f.submitter for f in feedbacks),
            'time_patterns': self._analyze_time_patterns(feedbacks),
            'tag_analysis': self._analyze_tags(feedbacks),
            'correlation_analysis': self._analyze_correlations(feedbacks)
        }
        
        return analysis
    
    def _analyze_time_patterns(self, feedbacks: List[FeedbackItem]) -> Dict[str, Any]:
        """分析时间模式"""
        hourly_dist = Counter(f.submit_time.hour for f in feedbacks)
        daily_dist = Counter(f.submit_time.weekday() for f in feedbacks)
        
        return {
            'hourly_distribution': dict(hourly_dist),
            'daily_distribution': dict(daily_dist),
            'peak_hours': sorted(hourly_dist.items(), key=lambda x: x[1], reverse=True)[:3],
            'peak_days': sorted(daily_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _analyze_tags(self, feedbacks: List[FeedbackItem]) -> Dict[str, Any]:
        """分析标签模式"""
        all_tags = []
        for f in feedbacks:
            all_tags.extend(f.tags)
        
        tag_counter = Counter(all_tags)
        type_tag_correlation = defaultdict(lambda: defaultdict(int))
        
        for f in feedbacks:
            for tag in f.tags:
                type_tag_correlation[f.feedback_type][tag] += 1
        
        return {
            'most_common_tags': tag_counter.most_common(10),
            'type_tag_correlation': dict(type_tag_correlation),
            'unique_tags_count': len(set(all_tags))
        }
    
    def _analyze_correlations(self, feedbacks: List[FeedbackItem]) -> Dict[str, Any]:
        """分析相关性"""
        # 分析类型和优先级的相关性
        type_priority_corr = defaultdict(lambda: defaultdict(int))
        for f in feedbacks:
            type_priority_corr[f.feedback_type][f.priority] += 1
        
        return {
            'type_priority_correlation': dict(type_priority_corr)
        }
    
    def detect_trends(self, feedbacks: List[FeedbackItem], time_window: int = 7) -> Dict[str, Any]:
        """检测反馈趋势"""
        now = datetime.datetime.now()
        window_start = now - datetime.timedelta(days=time_window)
        
        recent_feedbacks = [f for f in feedbacks if f.submit_time >= window_start]
        older_feedbacks = [f for f in feedbacks if f.submit_time < window_start]
        
        trends = {
            'recent_count': len(recent_feedbacks),
            'older_count': len(older_feedbacks),
            'growth_rate': (len(recent_feedbacks) - len(older_feedbacks)) / max(len(older_feedbacks), 1),
            'recent_type_trends': Counter(f.feedback_type for f in recent_feedbacks),
            'recent_priority_trends': Counter(f.priority for f in recent_feedbacks)
        }
        
        return trends


class StrategyManager:
    """策略管理器"""
    
    def __init__(self):
        self.strategies = {}
        self.performance_metrics = {}
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """初始化默认策略"""
        default_strategies = [
            ProcessingStrategy(
                id="bug_fix_immediate",
                name="错误立即修复",
                description="针对严重错误的立即处理策略",
                applicable_types=[FeedbackType.BUG],
                applicable_priorities=[Priority.CRITICAL, Priority.HIGH],
                steps=[
                    {"step": 1, "action": "立即评估影响范围", "time": 1},
                    {"step": 2, "action": "分配给开发团队", "time": 0.5},
                    {"step": 3, "action": "实施修复", "time": 8},
                    {"step": 4, "action": "测试验证", "time": 2},
                    {"step": 5, "action": "部署上线", "time": 1}
                ],
                estimated_time=12.5,
                success_rate=0.95
            ),
            ProcessingStrategy(
                id="feature_request_review",
                name="功能请求评估",
                description="新功能请求的评估和规划策略",
                applicable_types=[FeedbackType.FEATURE],
                applicable_priorities=[Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW],
                steps=[
                    {"step": 1, "action": "需求分析", "time": 4},
                    {"step": 2, "action": "技术可行性评估", "time": 2},
                    {"step": 3, "action": "资源评估", "time": 1},
                    {"step": 4, "action": "优先级排序", "time": 0.5},
                    {"step": 5, "action": "排期规划", "time": 1}
                ],
                estimated_time=8.5,
                success_rate=0.80
            ),
            ProcessingStrategy(
                id="performance_optimization",
                name="性能优化",
                description="性能问题的分析和优化策略",
                applicable_types=[FeedbackType.PERFORMANCE],
                applicable_priorities=[Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM],
                steps=[
                    {"step": 1, "action": "性能监控分析", "time": 2},
                    {"step": 2, "action": "瓶颈识别", "time": 3},
                    {"step": 3, "action": "优化方案设计", "time": 4},
                    {"step": 4, "action": "实施优化", "time": 16},
                    {"step": 5, "action": "性能测试", "time": 4}
                ],
                estimated_time=29,
                success_rate=0.85
            )
        ]
        
        for strategy in default_strategies:
            self.add_strategy(strategy)
    
    def add_strategy(self, strategy: ProcessingStrategy):
        """添加处理策略"""
        self.strategies[strategy.id] = strategy
        self.performance_metrics[strategy.id] = {
            'total_usage': 0,
            'success_count': 0,
            'average_time': 0,
            'satisfaction_scores': []
        }
        logger.info(f"添加处理策略: {strategy.name}")
    
    def recommend_strategy(self, feedback: FeedbackItem) -> Optional[ProcessingStrategy]:
        """推荐处理策略"""
        suitable_strategies = []
        
        for strategy in self.strategies.values():
            # 检查类型和优先级是否匹配
            if (feedback.feedback_type in strategy.applicable_types and 
                feedback.priority in strategy.applicable_priorities):
                suitable_strategies.append(strategy)
        
        if not suitable_strategies:
            return None
        
        # 按成功率和预计时间排序
        suitable_strategies.sort(key=lambda s: (s.success_rate, -s.estimated_time), reverse=True)
        
        recommended = suitable_strategies[0]
        self.performance_metrics[recommended.id]['total_usage'] += 1
        
        return recommended
    
    def update_strategy_performance(self, strategy_id: str, success: bool, 
                                  actual_time: float, satisfaction: float):
        """更新策略性能指标"""
        if strategy_id not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[strategy_id]
        metrics['total_usage'] += 1
        
        if success:
            metrics['success_count'] += 1
        
        # 更新平均时间
        total_time = metrics['average_time'] * (metrics['total_usage'] - 1) + actual_time
        metrics['average_time'] = total_time / metrics['total_usage']
        
        # 更新满意度
        metrics['satisfaction_scores'].append(satisfaction)
        if len(metrics['satisfaction_scores']) > 100:  # 保持最近100次记录
            metrics['satisfaction_scores'] = metrics['satisfaction_scores'][-100:]


class EffectEvaluator:
    """效果评估器"""
    
    def __init__(self):
        self.evaluation_metrics = {}
        self.benchmarks = {}
    
    def evaluate_processing_effect(self, feedback: FeedbackItem, 
                                 result: ProcessingResult) -> Dict[str, Any]:
        """评估处理效果"""
        evaluation = {
            'feedback_id': feedback.id,
            'processing_time': self._calculate_processing_time(feedback, result),
            'resolution_quality': self._evaluate_resolution_quality(feedback, result),
            'user_satisfaction': result.satisfaction_score or 0,
            'efficiency_score': self._calculate_efficiency_score(feedback, result),
            'recommendations': []
        }
        
        # 生成改进建议
        evaluation['recommendations'] = self._generate_recommendations(evaluation)
        
        return evaluation
    
    def _calculate_processing_time(self, feedback: FeedbackItem, result: ProcessingResult) -> float:
        """计算处理时间"""
        if result.end_time:
            return (result.end_time - result.start_time).total_seconds() / 3600  # 小时
        return 0
    
    def _evaluate_resolution_quality(self, feedback: FeedbackItem, result: ProcessingResult) -> float:
        """评估解决质量"""
        if result.status != Status.COMPLETED:
            return 0
        
        # 基于多个因素评估质量
        quality_score = 0
        
        # 检查是否完全解决了原始问题
        if result.result and result.result.get('issue_resolved', False):
            quality_score += 0.4
        
        # 检查是否按时完成
        if feedback.metadata.get('deadline'):
            deadline = datetime.datetime.fromisoformat(feedback.metadata['deadline'])
            if result.end_time and result.end_time <= deadline:
                quality_score += 0.3
        
        # 检查是否产生了负面影响
        if not result.issues_encountered:
            quality_score += 0.3
        
        return min(quality_score, 1.0)
    
    def _calculate_efficiency_score(self, feedback: FeedbackItem, result: ProcessingResult) -> float:
        """计算效率分数"""
        if not result.end_time:
            return 0
        
        processing_time = (result.end_time - result.start_time).total_seconds() / 3600
        
        # 基于优先级设定期望时间
        expected_times = {
            Priority.CRITICAL: 4,   # 4小时
            Priority.HIGH: 24,      # 1天
            Priority.MEDIUM: 72,    # 3天
            Priority.LOW: 168       # 1周
        }
        
        expected_time = expected_times.get(feedback.priority, 72)
        efficiency = expected_time / max(processing_time, 1)
        
        return min(efficiency, 1.0)
    
    def _generate_recommendations(self, evaluation: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if evaluation['processing_time'] > 48:
            recommendations.append("处理时间过长，建议优化流程或增加资源")
        
        if evaluation['resolution_quality'] < 0.7:
            recommendations.append("解决质量有待提升，建议加强问题分析")
        
        if evaluation['user_satisfaction'] < 3:
            recommendations.append("用户满意度较低，建议改进沟通和反馈机制")
        
        if evaluation['efficiency_score'] < 0.5:
            recommendations.append("处理效率偏低，建议重新评估处理策略")
        
        return recommendations


class HistoryTracker:
    """历史跟踪器"""
    
    def __init__(self):
        self.history_records = []
        self.statistics = {}
    
    def record_processing_step(self, feedback_id: str, step: str, 
                             details: Dict[str, Any]):
        """记录处理步骤"""
        record = {
            'feedback_id': feedback_id,
            'timestamp': datetime.datetime.now(),
            'step': step,
            'details': details
        }
        self.history_records.append(record)
        
        logger.info(f"记录处理步骤: {feedback_id} - {step}")
    
    def get_feedback_history(self, feedback_id: str) -> List[Dict]:
        """获取反馈处理历史"""
        return [r for r in self.history_records if r['feedback_id'] == feedback_id]
    
    def generate_statistics(self, time_range: Optional[Tuple[datetime.datetime, datetime.datetime]] = None) -> Dict[str, Any]:
        """生成统计信息"""
        records = self.history_records
        
        if time_range:
            start, end = time_range
            records = [r for r in records if start <= r['timestamp'] <= end]
        
        stats = {
            'total_records': len(records),
            'step_frequency': Counter(r['step'] for r in records),
            'daily_activity': self._calculate_daily_activity(records),
            'process_bottlenecks': self._identify_bottlenecks(records)
        }
        
        return stats
    
    def _calculate_daily_activity(self, records: List[Dict]) -> Dict[str, int]:
        """计算每日活动量"""
        daily_counts = Counter(r['timestamp'].date() for r in records)
        return dict(daily_counts)
    
    def _identify_bottlenecks(self, records: List[Dict]) -> List[str]:
        """识别处理瓶颈"""
        # 分析各步骤的执行频率和耗时
        step_times = defaultdict(list)
        
        for record in records:
            if 'duration' in record.get('details', {}):
                step_times[record['step']].append(record['details']['duration'])
        
        bottlenecks = []
        for step, times in step_times.items():
            if times:
                avg_time = sum(times) / len(times)
                if avg_time > 24:  # 平均超过24小时的步骤
                    bottlenecks.append(f"{step}: 平均耗时 {avg_time:.1f}小时")
        
        return bottlenecks


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """初始化报告模板"""
        self.templates = {
            'daily_summary': {
                'title': '每日反馈处理摘要',
                'sections': ['overview', 'type_distribution', 'priority_analysis', 'processing_status']
            },
            'weekly_analysis': {
                'title': '周度反馈分析报告',
                'sections': ['trends', 'patterns', 'performance', 'recommendations']
            },
            'monthly_report': {
                'title': '月度反馈处理报告',
                'sections': ['comprehensive_analysis', 'strategy_effectiveness', 'improvement_plan']
            }
        }
    
    def generate_report(self, report_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告"""
        if report_type not in self.templates:
            raise ValueError(f"未知的报告类型: {report_type}")
        
        template = self.templates[report_type]
        report = {
            'type': report_type,
            'title': template['title'],
            'generated_time': datetime.datetime.now(),
            'sections': {}
        }
        
        for section in template['sections']:
            report['sections'][section] = self._generate_section(section, data)
        
        return report
    
    def _generate_section(self, section: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告章节"""
        if section == 'overview':
            return {
                'total_feedbacks': len(data.get('feedbacks', [])),
                'processed_count': len([f for f in data.get('feedbacks', []) if f.status == Status.COMPLETED]),
                'pending_count': len([f for f in data.get('feedbacks', []) if f.status == Status.PENDING]),
                'average_processing_time': self._calculate_average_time(data.get('feedbacks', []))
            }
        
        elif section == 'type_distribution':
            feedbacks = data.get('feedbacks', [])
            type_dist = Counter(f.feedback_type for f in feedbacks)
            return {'distribution': dict(type_dist)}
        
        elif section == 'priority_analysis':
            feedbacks = data.get('feedbacks', [])
            priority_dist = Counter(f.priority for f in feedbacks)
            return {'distribution': dict(priority_dist)}
        
        elif section == 'trends':
            analyzer = FeedbackAnalyzer()
            trends = analyzer.detect_trends(data.get('feedbacks', []))
            return trends
        
        elif section == 'performance':
            return data.get('performance_metrics', {})
        
        else:
            return {}
    
    def _calculate_average_time(self, feedbacks: List[FeedbackItem]) -> float:
        """计算平均处理时间"""
        completed_feedbacks = [f for f in feedbacks if f.status == Status.COMPLETED and f.resolution_time]
        
        if not completed_feedbacks:
            return 0
        
        total_time = sum((f.resolution_time - f.submit_time).total_seconds() for f in completed_feedbacks)
        return total_time / len(completed_feedbacks) / 3600  # 转换为小时


class FeedbackProcessor:
    """反馈处理器主类"""
    
    def __init__(self):
        self.collector = FeedbackCollector()
        self.classifier = FeedbackClassifier()
        self.analyzer = FeedbackAnalyzer()
        self.strategy_manager = StrategyManager()
        self.evaluator = EffectEvaluator()
        self.tracker = HistoryTracker()
        self.reporter = ReportGenerator()
        
        self.feedbacks = {}  # 存储所有反馈
        self.processing_results = {}  # 存储处理结果
        
        logger.info("反馈处理器初始化完成")
    
    def collect_and_process_feedback(self, source_id: str, filters: Optional[Dict] = None) -> List[str]:
        """收集并处理反馈"""
        try:
            # 1. 收集反馈
            raw_feedbacks = self.collector.collect_feedback(source_id, filters)
            
            processed_ids = []
            
            for raw_feedback in raw_feedbacks:
                # 2. 分类反馈
                feedback_type, priority = self.classifier.classify_feedback(raw_feedback)
                tags = self.classifier.auto_tag_feedback(raw_feedback)
                
                # 3. 创建反馈项目
                feedback_item = FeedbackItem(
                    id=raw_feedback['id'],
                    title=raw_feedback['title'],
                    description=raw_feedback['description'],
                    feedback_type=feedback_type,
                    priority=priority,
                    status=Status.PENDING,
                    submit_time=datetime.datetime.now(),
                    submitter=raw_feedback['submitter'],
                    tags=tags
                )
                
                self.feedbacks[feedback_item.id] = feedback_item
                processed_ids.append(feedback_item.id)
                
                # 4. 推荐处理策略
                strategy = self.strategy_manager.recommend_strategy(feedback_item)
                
                # 5. 开始处理
                if strategy:
                    self._start_processing(feedback_item, strategy)
                
                logger.info(f"处理反馈: {feedback_item.id} - {feedback_item.title}")
            
            return processed_ids
            
        except Exception as e:
            logger.error(f"收集和处理反馈时出错: {e}")
            raise
    
    def _start_processing(self, feedback: FeedbackItem, strategy: ProcessingStrategy):
        """开始处理反馈"""
        result = ProcessingResult(
            feedback_id=feedback.id,
            strategy_id=strategy.id,
            start_time=datetime.datetime.now()
        )
        
        self.processing_results[feedback.id] = result
        
        # 记录处理开始
        self.tracker.record_processing_step(
            feedback.id,
            'processing_started',
            {'strategy': strategy.name, 'estimated_time': strategy.estimated_time}
        )
        
        # 更新反馈状态
        feedback.status = Status.IN_PROGRESS
        feedback.assignee = "auto_assigned"  # 实际应用中应该有具体的分配逻辑
    
    def process_feedback(self, feedback_id: str, action: str, details: Dict[str, Any]) -> bool:
        """处理特定反馈"""
        if feedback_id not in self.feedbacks:
            raise ValueError(f"反馈不存在: {feedback_id}")
        
        feedback = self.feedbacks[feedback_id]
        
        if action == "complete":
            return self._complete_feedback(feedback, details)
        elif action == "reject":
            return self._reject_feedback(feedback, details)
        else:
            raise ValueError(f"未知的处理动作: {action}")
    
    def _complete_feedback(self, feedback: FeedbackItem, details: Dict[str, Any]) -> bool:
        """完成反馈处理"""
        feedback.status = Status.COMPLETED
        feedback.resolution_time = datetime.datetime.now()
        
        if feedback_id := feedback.id in self.processing_results:
            result = self.processing_results[feedback.id]
            result.end_time = feedback.resolution_time
            result.status = Status.COMPLETED
            result.result = details
        
        # 记录完成
        self.tracker.record_processing_step(
            feedback.id,
            'processing_completed',
            details
        )
        
        # 评估效果
        if feedback.id in self.processing_results:
            evaluation = self.evaluator.evaluate_processing_effect(feedback, self.processing_results[feedback.id])
            feedback.evaluation_score = evaluation.get('user_satisfaction', 0)
        
        logger.info(f"完成反馈处理: {feedback.id}")
        return True
    
    def _reject_feedback(self, feedback: FeedbackItem, details: Dict[str, Any]) -> bool:
        """拒绝反馈处理"""
        feedback.status = Status.REJECTED
        
        if feedback.id in self.processing_results:
            result = self.processing_results[feedback.id]
            result.status = Status.REJECTED
            result.result = details
        
        # 记录拒绝
        self.tracker.record_processing_step(
            feedback.id,
            'processing_rejected',
            details
        )
        
        logger.info(f"拒绝反馈处理: {feedback.id}")
        return True
    
    def analyze_feedbacks(self, time_range: Optional[Tuple[datetime.datetime, datetime.datetime]] = None) -> Dict[str, Any]:
        """分析反馈"""
        feedbacks = list(self.feedbacks.values())
        
        if time_range:
            start, end = time_range
            feedbacks = [f for f in feedbacks if start <= f.submit_time <= end]
        
        analysis = {
            'patterns': self.analyzer.analyze_patterns(feedbacks),
            'trends': self.analyzer.detect_trends(feedbacks),
            'statistics': self.tracker.generate_statistics(time_range)
        }
        
        return analysis
    
    def generate_report(self, report_type: str, time_range: Optional[Tuple[datetime.datetime, datetime.datetime]] = None) -> Dict[str, Any]:
        """生成报告"""
        data = {
            'feedbacks': list(self.feedbacks.values()),
            'processing_results': list(self.processing_results.values()),
            'performance_metrics': self.strategy_manager.performance_metrics
        }
        
        return self.reporter.generate_report(report_type, data)
    
    def get_feedback_status(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """获取反馈状态"""
        if feedback_id not in self.feedbacks:
            return None
        
        feedback = self.feedbacks[feedback_id]
        result = self.processing_results.get(feedback_id)
        history = self.tracker.get_feedback_history(feedback_id)
        
        return {
            'feedback': feedback,
            'processing_result': result,
            'history': history
        }
    
    def get_statistics(self, period: str = 'week') -> Dict[str, Any]:
        """获取统计信息"""
        now = datetime.datetime.now()
        
        if period == 'day':
            start = now - datetime.timedelta(days=1)
        elif period == 'week':
            start = now - datetime.timedelta(weeks=1)
        elif period == 'month':
            start = now - datetime.timedelta(days=30)
        else:
            start = now - datetime.timedelta(weeks=1)
        
        feedbacks = [f for f in self.feedbacks.values() if f.submit_time >= start]
        
        return {
            'period': period,
            'start_time': start,
            'end_time': now,
            'total_feedbacks': len(feedbacks),
            'type_distribution': Counter(f.feedback_type for f in feedbacks),
            'priority_distribution': Counter(f.priority for f in feedbacks),
            'status_distribution': Counter(f.status for f in feedbacks),
            'completion_rate': len([f for f in feedbacks if f.status == Status.COMPLETED]) / max(len(feedbacks), 1),
            'average_processing_time': self._calculate_average_processing_time(feedbacks)
        }
    
    def _calculate_average_processing_time(self, feedbacks: List[FeedbackItem]) -> float:
        """计算平均处理时间"""
        completed = [f for f in feedbacks if f.status == Status.COMPLETED and f.resolution_time]
        
        if not completed:
            return 0
        
        total_time = sum((f.resolution_time - f.submit_time).total_seconds() for f in completed)
        return total_time / len(completed) / 3600  # 小时


# 使用示例
if __name__ == "__main__":
    # 创建反馈处理器
    processor = FeedbackProcessor()
    
    # 注册反馈源
    processor.collector.register_source("api_source", {"type": "api", "endpoint": "https://api.example.com/feedback"})
    processor.collector.register_source("database_source", {"type": "database", "connection": "db_connection"})
    
    # 收集和处理反馈
    feedback_ids = processor.collect_and_process_feedback("api_source")
    print(f"处理了 {len(feedback_ids)} 条反馈")
    
    # 分析反馈
    analysis = processor.analyze_feedbacks()
    print("反馈分析完成")
    
    # 生成报告
    daily_report = processor.generate_report("daily_summary")
    print("日报生成完成")
    
    # 获取统计信息
    stats = processor.get_statistics("week")
    print(f"本周收到 {stats['total_feedbacks']} 条反馈")
    
    # 获取反馈状态
    if feedback_ids:
        status = processor.get_feedback_status(feedback_ids[0])
        print(f"反馈状态: {status['feedback'].status.value}")
