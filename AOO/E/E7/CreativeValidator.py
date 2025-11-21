"""
E7 创意验证器实现
================

创意验证器提供全面的创意评估和验证功能，包括可行性、有效性、一致性、
质量、风险评估以及改进建议。

主要组件：
- CreativeValidator: 主要验证器类
- CreativeFeasibilityValidator: 可行性验证器
- CreativeValidityTester: 有效性测试器
- CreativeConsistencyChecker: 一致性检验器
- CreativeQualityAssessor: 质量评估器
- CreativeRiskAssessor: 风险评估器
- CreativeImprovementAdvisor: 改进建议器
- CreativeValidationReport: 验证报告生成器


创建时间：2025-11-05
"""

import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import math
import statistics
from abc import ABC, abstractmethod


class ValidationStatus(Enum):
    """验证状态枚举"""
    PENDING = "待验证"
    PASSED = "通过"
    FAILED = "失败"
    WARNING = "警告"
    PARTIAL = "部分通过"


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    CRITICAL = "极高风险"


class QualityGrade(Enum):
    """质量等级枚举"""
    EXCELLENT = "优秀"
    GOOD = "良好"
    AVERAGE = "一般"
    POOR = "较差"
    UNACCEPTABLE = "不可接受"


@dataclass
class ValidationResult:
    """验证结果数据类"""
    status: ValidationStatus
    score: float  # 0-100分
    details: Dict[str, Any]
    timestamp: str
    suggestions: List[str]


@dataclass
class CreativeIdea:
    """创意想法数据类"""
    id: str
    title: str
    description: str
    category: str
    tags: List[str]
    target_audience: str
    resources_required: List[str]
    expected_outcomes: List[str]
    constraints: List[str]
    created_at: str
    metadata: Dict[str, Any] = None


class CreativeValidator:
    """创意验证器主类"""
    
    def __init__(self):
        self.feasibility_validator = CreativeFeasibilityValidator()
        self.validity_tester = CreativeValidityTester()
        self.consistency_checker = CreativeConsistencyChecker()
        self.quality_assessor = CreativeQualityAssessor()
        self.risk_assessor = CreativeRiskAssessor()
        self.improvement_advisor = CreativeImprovementAdvisor()
        self.report_generator = CreativeValidationReport()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_creative(self, idea: CreativeIdea) -> Dict[str, Any]:
        """
        对创意进行全面验证
        
        Args:
            idea: 创意想法对象
            
        Returns:
            完整的验证结果
        """
        self.logger.info(f"开始验证创意: {idea.title}")
        
        # 执行各项验证
        feasibility_result = self.feasibility_validator.validate(idea)
        validity_result = self.validity_tester.test(idea)
        consistency_result = self.consistency_checker.check(idea)
        quality_result = self.quality_assessor.assess(idea)
        risk_result = self.risk_assessor.assess(idea)
        
        # 生成改进建议
        improvement_suggestions = self.improvement_advisor.generate_suggestions(
            idea, feasibility_result, validity_result, consistency_result, 
            quality_result, risk_result
        )
        
        # 生成综合报告
        comprehensive_report = self.report_generator.generate_report(
            idea, feasibility_result, validity_result, consistency_result,
            quality_result, risk_result, improvement_suggestions
        )
        
        return comprehensive_report


class CreativeFeasibilityValidator:
    """创意可行性验证器"""
    
    def __init__(self):
        self.feasibility_criteria = {
            'technical_feasibility': 0.25,  # 技术可行性
            'economic_feasibility': 0.25,   # 经济可行性
            'operational_feasibility': 0.25, # 运营可行性
            'legal_feasibility': 0.15,      # 法律可行性
            'time_feasibility': 0.10        # 时间可行性
        }
    
    def validate(self, idea: CreativeIdea) -> ValidationResult:
        """验证创意的可行性"""
        details = {}
        total_score = 0
        
        # 技术可行性验证
        tech_score = self._validate_technical_feasibility(idea)
        details['technical_feasibility'] = tech_score
        total_score += tech_score * self.feasibility_criteria['technical_feasibility']
        
        # 经济可行性验证
        economic_score = self._validate_economic_feasibility(idea)
        details['economic_feasibility'] = economic_score
        total_score += economic_score * self.feasibility_criteria['economic_feasibility']
        
        # 运营可行性验证
        operational_score = self._validate_operational_feasibility(idea)
        details['operational_feasibility'] = operational_score
        total_score += operational_score * self.feasibility_criteria['operational_feasibility']
        
        # 法律可行性验证
        legal_score = self._validate_legal_feasibility(idea)
        details['legal_feasibility'] = legal_score
        total_score += legal_score * self.feasibility_criteria['legal_feasibility']
        
        # 时间可行性验证
        time_score = self._validate_time_feasibility(idea)
        details['time_feasibility'] = time_score
        total_score += time_score * self.feasibility_criteria['time_feasibility']
        
        # 确定验证状态
        if total_score >= 80:
            status = ValidationStatus.PASSED
        elif total_score >= 60:
            status = ValidationStatus.PARTIAL
        elif total_score >= 40:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED
        
        suggestions = self._generate_feasibility_suggestions(details)
        
        return ValidationResult(
            status=status,
            score=total_score,
            details=details,
            timestamp=datetime.datetime.now().isoformat(),
            suggestions=suggestions
        )
    
    def _validate_technical_feasibility(self, idea: CreativeIdea) -> float:
        """验证技术可行性"""
        score = 100.0
        
        # 检查技术复杂度
        complexity_keywords = ['AI', '机器学习', '区块链', '量子计算', '基因工程']
        for keyword in complexity_keywords:
            if keyword in idea.description:
                score -= 15
        
        # 检查资源需求
        if len(idea.resources_required) > 10:
            score -= 20
        elif len(idea.resources_required) > 5:
            score -= 10
        
        # 检查技术成熟度
        mature_tech = ['云计算', '移动应用', 'Web开发', '数据分析']
        for tech in mature_tech:
            if tech in idea.description:
                score += 10
        
        return max(0, min(100, score))
    
    def _validate_economic_feasibility(self, idea: CreativeIdea) -> float:
        """验证经济可行性"""
        score = 100.0
        
        # 检查目标市场规模
        market_keywords = ['全球', '全国', '大众', '消费者']
        market_mentions = sum(1 for keyword in market_keywords if keyword in idea.description)
        score += market_mentions * 10
        
        # 检查商业模式清晰度
        business_keywords = ['付费', '订阅', '广告', '佣金', '服务费']
        business_mentions = sum(1 for keyword in business_keywords if keyword in idea.description)
        score += business_mentions * 15
        
        # 检查成本控制
        if '低成本' in idea.description or '节约' in idea.description:
            score += 10
        
        return max(0, min(100, score))
    
    def _validate_operational_feasibility(self, idea: CreativeIdea) -> float:
        """验证运营可行性"""
        score = 100.0
        
        # 检查团队要求
        team_keywords = ['团队', '合作', '协作', '分工']
        team_mentions = sum(1 for keyword in team_keywords if keyword in idea.description)
        score += team_mentions * 10
        
        # 检查流程复杂度
        process_keywords = ['自动化', '标准化', '流程化', '系统化']
        process_mentions = sum(1 for keyword in process_keywords if keyword in idea.description)
        score += process_mentions * 12
        
        # 检查可扩展性
        scalability_keywords = ['扩展', '复制', '规模化', '推广']
        scalability_mentions = sum(1 for keyword in scalability_keywords if keyword in idea.description)
        score += scalability_mentions * 8
        
        return max(0, min(100, score))
    
    def _validate_legal_feasibility(self, idea: CreativeIdea) -> float:
        """验证法律可行性"""
        score = 100.0
        
        # 检查合规风险
        risk_keywords = ['监管', '合规', '法律', '政策', '标准']
        risk_mentions = sum(1 for keyword in risk_keywords if keyword in idea.description)
        score += risk_mentions * 5  # 正面提及表示考虑了合规
        
        # 检查知识产权
        ip_keywords = ['专利', '版权', '商标', '知识产权']
        ip_mentions = sum(1 for keyword in ip_keywords if keyword in idea.description)
        score += ip_mentions * 8
        
        # 检查敏感领域
        sensitive_areas = ['金融', '医疗', '教育', '数据隐私']
        for area in sensitive_areas:
            if area in idea.description:
                score -= 15  # 敏感领域需要更多合规考虑
        
        return max(0, min(100, score))
    
    def _validate_time_feasibility(self, idea: CreativeIdea) -> float:
        """验证时间可行性"""
        score = 100.0
        
        # 检查时间敏感性
        urgency_keywords = ['紧急', '即时', '快速', '时效']
        urgency_mentions = sum(1 for keyword in urgency_keywords if keyword in idea.description)
        score -= urgency_mentions * 10
        
        # 检查开发周期
        development_keywords = ['原型', 'MVP', '测试', '迭代']
        development_mentions = sum(1 for keyword in development_keywords if keyword in idea.description)
        score += development_mentions * 8  # 考虑开发周期是好的
        
        return max(0, min(100, score))
    
    def _generate_feasibility_suggestions(self, details: Dict[str, float]) -> List[str]:
        """生成可行性改进建议"""
        suggestions = []
        
        if details.get('technical_feasibility', 0) < 70:
            suggestions.append("建议简化技术方案或寻找更成熟的技术解决方案")
        
        if details.get('economic_feasibility', 0) < 70:
            suggestions.append("建议明确商业模式和盈利路径，考虑成本控制策略")
        
        if details.get('operational_feasibility', 0) < 70:
            suggestions.append("建议制定详细的运营计划和团队配置方案")
        
        if details.get('legal_feasibility', 0) < 70:
            suggestions.append("建议咨询法律专家，确保合规性和知识产权保护")
        
        if details.get('time_feasibility', 0) < 70:
            suggestions.append("建议制定现实可行的时间表和里程碑计划")
        
        return suggestions


class CreativeValidityTester:
    """创意有效性测试器"""
    
    def test(self, idea: CreativeIdea) -> ValidationResult:
        """测试创意的有效性"""
        details = {}
        total_score = 0
        
        # 市场需求验证
        market_validity = self._test_market_demand(idea)
        details['market_demand'] = market_validity
        total_score += market_validity * 0.3
        
        # 用户价值验证
        user_value = self._test_user_value(idea)
        details['user_value'] = user_value
        total_score += user_value * 0.25
        
        # 竞争优势验证
        competitive_advantage = self._test_competitive_advantage(idea)
        details['competitive_advantage'] = competitive_advantage
        total_score += competitive_advantage * 0.25
        
        # 创新性验证
        innovation_level = self._test_innovation_level(idea)
        details['innovation_level'] = innovation_level
        total_score += innovation_level * 0.2
        
        # 确定测试状态
        if total_score >= 75:
            status = ValidationStatus.PASSED
        elif total_score >= 60:
            status = ValidationStatus.PARTIAL
        elif total_score >= 45:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED
        
        suggestions = self._generate_validity_suggestions(details)
        
        return ValidationResult(
            status=status,
            score=total_score,
            details=details,
            timestamp=datetime.datetime.now().isoformat(),
            suggestions=suggestions
        )
    
    def _test_market_demand(self, idea: CreativeIdea) -> float:
        """测试市场需求"""
        score = 50.0  # 基础分
        
        # 市场需求关键词
        demand_keywords = ['痛点', '需求', '问题', '困难', '挑战', '不足']
        demand_score = sum(15 for keyword in demand_keywords if keyword in idea.description)
        score += demand_score
        
        # 目标用户明确性
        if idea.target_audience:
            score += 20
        
        # 市场规模暗示
        market_size_keywords = ['大量', '广泛', '普遍', '常见', '普遍存在']
        market_score = sum(10 for keyword in market_size_keywords if keyword in idea.description)
        score += market_score
        
        return min(100, score)
    
    def _test_user_value(self, idea: CreativeIdea) -> float:
        """测试用户价值"""
        score = 50.0
        
        # 价值关键词
        value_keywords = ['便利', '效率', '节省', '提升', '改善', '优化', '解决']
        value_score = sum(12 for keyword in value_keywords if keyword in idea.description)
        score += value_score
        
        # 预期成果
        if idea.expected_outcomes:
            score += len(idea.expected_outcomes) * 5
        
        # 解决具体问题
        problem_solving_keywords = ['解决', '消除', '避免', '预防', '改善']
        problem_score = sum(10 for keyword in problem_solving_keywords if keyword in idea.description)
        score += problem_score
        
        return min(100, score)
    
    def _test_competitive_advantage(self, idea: CreativeIdea) -> float:
        """测试竞争优势"""
        score = 40.0
        
        # 独特性关键词
        unique_keywords = ['独特', '创新', '新颖', '首创', '独家', '特色', '差异化']
        unique_score = sum(15 for keyword in unique_keywords if keyword in idea.description)
        score += unique_score
        
        # 优势表述
        advantage_keywords = ['优势', '领先', '超越', '更好', '更优', '更强']
        advantage_score = sum(10 for keyword in advantage_keywords if keyword in idea.description)
        score += advantage_score
        
        return min(100, score)
    
    def _test_innovation_level(self, idea: CreativeIdea) -> float:
        """测试创新水平"""
        score = 30.0
        
        # 创新关键词
        innovation_keywords = ['创新', '突破', '革命性', '颠覆性', '变革', '改进']
        innovation_score = sum(20 for keyword in innovation_keywords if keyword in idea.description)
        score += innovation_score
        
        # 技术创新
        tech_innovation = ['AI', '机器学习', '区块链', '物联网', '大数据']
        tech_score = sum(15 for keyword in tech_innovation if keyword in idea.description)
        score += tech_score
        
        return min(100, score)
    
    def _generate_validity_suggestions(self, details: Dict[str, float]) -> List[str]:
        """生成有效性改进建议"""
        suggestions = []
        
        if details.get('market_demand', 0) < 60:
            suggestions.append("建议深入调研市场需求，明确目标用户群体和痛点")
        
        if details.get('user_value', 0) < 60:
            suggestions.append("建议明确用户价值主张，突出核心价值和优势")
        
        if details.get('competitive_advantage', 0) < 60:
            suggestions.append("建议明确差异化竞争策略，突出独特优势")
        
        if details.get('innovation_level', 0) < 60:
            suggestions.append("建议增加创新元素，提升创意的独特性和突破性")
        
        return suggestions


class CreativeConsistencyChecker:
    """创意一致性检验器"""
    
    def check(self, idea: CreativeIdea) -> ValidationResult:
        """检验创意的一致性"""
        details = {}
        total_score = 0
        
        # 目标一致性
        goal_consistency = self._check_goal_consistency(idea)
        details['goal_consistency'] = goal_consistency
        total_score += goal_consistency * 0.3
        
        # 逻辑一致性
        logic_consistency = self._check_logic_consistency(idea)
        details['logic_consistency'] = logic_consistency
        total_score += logic_consistency * 0.3
        
        # 资源一致性
        resource_consistency = self._check_resource_consistency(idea)
        details['resource_consistency'] = resource_consistency
        total_score += resource_consistency * 0.2
        
        # 时间一致性
        time_consistency = self._check_time_consistency(idea)
        details['time_consistency'] = time_consistency
        total_score += time_consistency * 0.2
        
        # 确定检验状态
        if total_score >= 80:
            status = ValidationStatus.PASSED
        elif total_score >= 65:
            status = ValidationStatus.PARTIAL
        elif total_score >= 50:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED
        
        suggestions = self._generate_consistency_suggestions(details)
        
        return ValidationResult(
            status=status,
            score=total_score,
            details=details,
            timestamp=datetime.datetime.now().isoformat(),
            suggestions=suggestions
        )
    
    def _check_goal_consistency(self, idea: CreativeIdea) -> float:
        """检查目标一致性"""
        score = 100.0
        
        # 检查描述与预期成果的一致性
        description_words = set(idea.description.lower().split())
        outcome_words = set(' '.join(idea.expected_outcomes).lower().split())
        
        # 计算词汇重叠度
        overlap = len(description_words.intersection(outcome_words))
        total_words = len(description_words.union(outcome_words))
        
        if total_words > 0:
            consistency_ratio = overlap / total_words
            score = consistency_ratio * 100
        
        # 检查目标受众与价值主张的一致性
        audience_value_consistency = self._check_audience_value_match(idea)
        score = (score + audience_value_consistency) / 2
        
        return score
    
    def _check_logic_consistency(self, idea: CreativeIdea) -> float:
        """检查逻辑一致性"""
        score = 100.0
        
        # 检查因果逻辑
        cause_effect_keywords = ['因为', '由于', '导致', '造成', '因此', '所以', '从而']
        logic_indicators = sum(1 for keyword in cause_effect_keywords if keyword in idea.description)
        
        if logic_indicators == 0:
            score -= 20  # 缺乏逻辑连接词
        
        # 检查步骤逻辑
        sequence_keywords = ['首先', '然后', '接着', '最后', '下一步', '随后']
        sequence_indicators = sum(1 for keyword in sequence_keywords if keyword in idea.description)
        score += sequence_indicators * 10
        
        # 检查资源与目标的匹配度
        resource_target_match = self._check_resource_target_match(idea)
        score = (score + resource_target_match) / 2
        
        return max(0, min(100, score))
    
    def _check_resource_consistency(self, idea: CreativeIdea) -> float:
        """检查资源一致性"""
        score = 100.0
        
        # 检查资源需求的合理性
        if len(idea.resources_required) == 0:
            score -= 30  # 缺乏资源考虑
        
        # 检查资源与创意的匹配度
        resource_idea_match = self._calculate_resource_idea_match(idea)
        score = resource_idea_match
        
        # 检查资源约束的合理性
        constraint_keywords = ['有限', '限制', '约束', '预算', '人力']
        constraint_mentions = sum(1 for keyword in constraint_keywords if keyword in idea.description)
        score += constraint_mentions * 5
        
        return max(0, min(100, score))
    
    def _check_time_consistency(self, idea: CreativeIdea) -> float:
        """检查时间一致性"""
        score = 100.0
        
        # 检查时间表述的一致性
        time_keywords = ['短期', '长期', '立即', '未来', '近期', '远期']
        time_mentions = sum(1 for keyword in time_keywords if keyword in idea.description)
        
        if time_mentions == 0:
            score -= 20  # 缺乏时间考虑
        
        # 检查时间期望的现实性
        urgency_keywords = ['立即', '马上', '即刻']
        urgency_count = sum(1 for keyword in urgency_keywords if keyword in idea.description)
        
        if urgency_count > 2:
            score -= 30  # 过于急迫
        
        return max(0, min(100, score))
    
    def _check_audience_value_match(self, idea: CreativeIdea) -> float:
        """检查受众与价值匹配度"""
        if not idea.target_audience:
            return 50.0
        
        # 简单的匹配度计算
        audience_words = set(idea.target_audience.lower().split())
        value_words = set(' '.join(idea.expected_outcomes).lower().split())
        
        overlap = len(audience_words.intersection(value_words))
        total_words = len(audience_words.union(value_words))
        
        if total_words == 0:
            return 50.0
        
        return (overlap / total_words) * 100
    
    def _check_resource_target_match(self, idea: CreativeIdea) -> float:
        """检查资源与目标匹配度"""
        if not idea.resources_required or not idea.expected_outcomes:
            return 70.0
        
        # 简单的匹配度计算
        resource_words = set(' '.join(idea.resources_required).lower().split())
        target_words = set(' '.join(idea.expected_outcomes).lower().split())
        
        overlap = len(resource_words.intersection(target_words))
        total_words = len(resource_words.union(target_words))
        
        if total_words == 0:
            return 70.0
        
        return (overlap / total_words) * 100
    
    def _calculate_resource_idea_match(self, idea: CreativeIdea) -> float:
        """计算资源与创意匹配度"""
        # 基于创意复杂度和资源数量的简单评估
        complexity_score = 100.0
        
        # 复杂度指标
        complexity_indicators = len(idea.description.split()) / 10  # 描述长度
        complexity_score -= complexity_indicators * 5
        
        # 资源充足度
        resource_adequacy = len(idea.resources_required) * 10
        score = min(100, resource_adequacy + complexity_score - 50)
        
        return max(0, min(100, score))
    
    def _generate_consistency_suggestions(self, details: Dict[str, float]) -> List[str]:
        """生成一致性改进建议"""
        suggestions = []
        
        if details.get('goal_consistency', 0) < 70:
            suggestions.append("建议确保创意描述与预期成果保持一致，明确目标导向")
        
        if details.get('logic_consistency', 0) < 70:
            suggestions.append("建议加强逻辑链条，确保因果关系清晰，步骤合理")
        
        if details.get('resource_consistency', 0) < 70:
            suggestions.append("建议重新评估资源需求，确保资源与创意目标匹配")
        
        if details.get('time_consistency', 0) < 70:
            suggestions.append("建议制定合理的时间规划，确保时间期望现实可行")
        
        return suggestions


class CreativeQualityAssessor:
    """创意质量评估器"""
    
    def assess(self, idea: CreativeIdea) -> ValidationResult:
        """评估创意质量"""
        details = {}
        total_score = 0
        
        # 完整性评估
        completeness = self._assess_completeness(idea)
        details['completeness'] = completeness
        total_score += completeness * 0.25
        
        # 清晰度评估
        clarity = self._assess_clarity(idea)
        details['clarity'] = clarity
        total_score += clarity * 0.2
        
        # 可行性评估
        practicality = self._assess_practicality(idea)
        details['practicality'] = practicality
        total_score += practicality * 0.2
        
        # 吸引力评估
        attractiveness = self._assess_attractiveness(idea)
        details['attractiveness'] = attractiveness
        total_score += attractiveness * 0.15
        
        # 可扩展性评估
        scalability = self._assess_scalability(idea)
        details['scalability'] = scalability
        total_score += scalability * 0.2
        
        # 确定质量等级
        if total_score >= 85:
            status = ValidationStatus.PASSED
            grade = QualityGrade.EXCELLENT
        elif total_score >= 75:
            status = ValidationStatus.PASSED
            grade = QualityGrade.GOOD
        elif total_score >= 60:
            status = ValidationStatus.PARTIAL
            grade = QualityGrade.AVERAGE
        elif total_score >= 45:
            status = ValidationStatus.WARNING
            grade = QualityGrade.POOR
        else:
            status = ValidationStatus.FAILED
            grade = QualityGrade.UNACCEPTABLE
        
        details['quality_grade'] = grade.value
        suggestions = self._generate_quality_suggestions(details)
        
        return ValidationResult(
            status=status,
            score=total_score,
            details=details,
            timestamp=datetime.datetime.now().isoformat(),
            suggestions=suggestions
        )
    
    def _assess_completeness(self, idea: CreativeIdea) -> float:
        """评估完整性"""
        score = 0.0
        max_score = 100.0
        
        # 检查基本信息完整性
        if idea.title:
            score += 15
        if idea.description:
            score += 20
        if idea.category:
            score += 10
        if idea.target_audience:
            score += 15
        
        # 检查详细程度
        if idea.tags:
            score += len(idea.tags) * 2  # 每个标签2分，最多10分
        if idea.resources_required:
            score += min(len(idea.resources_required) * 3, 15)  # 每个资源3分，最多15分
        if idea.expected_outcomes:
            score += min(len(idea.expected_outcomes) * 4, 20)  # 每个成果4分，最多20分
        if idea.constraints:
            score += min(len(idea.constraints) * 2, 10)  # 每个约束2分，最多10分
        
        return min(score, max_score)
    
    def _assess_clarity(self, idea: CreativeIdea) -> float:
        """评估清晰度"""
        score = 100.0
        
        # 检查描述长度
        desc_length = len(idea.description)
        if desc_length < 50:
            score -= 30  # 描述太短
        elif desc_length > 500:
            score -= 20  # 描述太长
        elif 100 <= desc_length <= 300:
            score += 10  # 描述长度适中
        
        # 检查语言清晰度
        unclear_keywords = ['可能', '也许', '大概', '似乎', '或许', '差不多']
        unclear_count = sum(1 for keyword in unclear_keywords if keyword in idea.description)
        score -= unclear_count * 8
        
        # 检查专业术语使用
        technical_terms = ['算法', '架构', '系统', '平台', '框架', '接口']
        tech_count = sum(1 for term in technical_terms if term in idea.description)
        if tech_count > 3:
            score -= 15  # 技术术语过多
        elif 1 <= tech_count <= 3:
            score += 5  # 适度的技术表述
        
        return max(0, min(100, score))
    
    def _assess_practicality(self, idea: CreativeIdea) -> float:
        """评估实用性"""
        score = 60.0  # 基础分
        
        # 实用性关键词
        practical_keywords = ['实用', '可行', '现实', '有效', '解决', '改善', '优化']
        practical_score = sum(8 for keyword in practical_keywords if keyword in idea.description)
        score += practical_score
        
        # 避免过于抽象
        abstract_keywords = ['概念', '理论', '理念', '思想', '愿景']
        abstract_count = sum(1 for keyword in abstract_keywords if keyword in idea.description)
        score -= abstract_count * 10
        
        # 考虑实施细节
        implementation_keywords = ['实施', '执行', '操作', '应用', '部署']
        impl_count = sum(1 for keyword in implementation_keywords if keyword in idea.description)
        score += impl_count * 6
        
        return max(0, min(100, score))
    
    def _assess_attractiveness(self, idea: CreativeIdea) -> float:
        """评估吸引力"""
        score = 50.0  # 基础分
        
        # 吸引力关键词
        attractive_keywords = ['吸引', '有趣', '新颖', '独特', '令人兴奋', '惊喜', '震撼']
        attractive_score = sum(10 for keyword in attractive_keywords if keyword in idea.description)
        score += attractive_score
        
        # 情感词汇
        emotional_keywords = ['快乐', '兴奋', '满足', '惊喜', '感动', '震撼']
        emotional_score = sum(8 for keyword in emotional_keywords if keyword in idea.description)
        score += emotional_score
        
        # 视觉化程度
        visual_keywords = ['图像', '视觉', '美观', '设计', '界面', '体验']
        visual_score = sum(6 for keyword in visual_keywords if keyword in idea.description)
        score += visual_score
        
        return max(0, min(100, score))
    
    def _assess_scalability(self, idea: CreativeIdea) -> float:
        """评估可扩展性"""
        score = 40.0  # 基础分
        
        # 扩展性关键词
        scalable_keywords = ['扩展', '复制', '推广', '规模化', '复制', '复制', '普及']
        scalable_score = sum(12 for keyword in scalable_keywords if keyword in idea.description)
        score += scalable_score
        
        # 网络效应
        network_keywords = ['网络', '平台', '社区', '连接', '互动', '分享']
        network_score = sum(8 for keyword in network_keywords if keyword in idea.description)
        score += network_score
        
        # 技术可扩展性
        tech_scalability = ['云端', '分布式', '模块化', 'API', '自动化']
        tech_score = sum(10 for keyword in tech_scalability if keyword in idea.description)
        score += tech_score
        
        return max(0, min(100, score))
    
    def _generate_quality_suggestions(self, details: Dict[str, float]) -> List[str]:
        """生成质量改进建议"""
        suggestions = []
        
        if details.get('completeness', 0) < 70:
            suggestions.append("建议完善创意的基本信息，包括详细描述、目标受众、资源需求等")
        
        if details.get('clarity', 0) < 70:
            suggestions.append("建议使用更清晰明确的语言表达，避免模糊不清的表述")
        
        if details.get('practicality', 0) < 70:
            suggestions.append("建议增强创意的实用性，提供更多具体的实施细节")
        
        if details.get('attractiveness', 0) < 70:
            suggestions.append("建议增加创意的吸引力，使用更有感染力的表达方式")
        
        if details.get('scalability', 0) < 70:
            suggestions.append("建议考虑创意的可扩展性，设计规模化推广的策略")
        
        return suggestions


class CreativeRiskAssessor:
    """创意风险评估器"""
    
    def assess(self, idea: CreativeIdea) -> ValidationResult:
        """评估创意风险"""
        details = {}
        total_risk_score = 0
        
        # 技术风险
        technical_risk = self._assess_technical_risk(idea)
        details['technical_risk'] = technical_risk
        total_risk_score += technical_risk * 0.25
        
        # 市场风险
        market_risk = self._assess_market_risk(idea)
        details['market_risk'] = market_risk
        total_risk_score += market_risk * 0.25
        
        # 财务风险
        financial_risk = self._assess_financial_risk(idea)
        details['financial_risk'] = financial_risk
        total_risk_score += financial_risk * 0.2
        
        # 运营风险
        operational_risk = self._assess_operational_risk(idea)
        details['operational_risk'] = operational_risk
        total_risk_score += operational_risk * 0.15
        
        # 法律风险
        legal_risk = self._assess_legal_risk(idea)
        details['legal_risk'] = legal_risk
        total_risk_score += legal_risk * 0.15
        
        # 确定风险等级
        if total_risk_score <= 30:
            risk_level = RiskLevel.LOW
            status = ValidationStatus.PASSED
        elif total_risk_score <= 50:
            risk_level = RiskLevel.MEDIUM
            status = ValidationStatus.PARTIAL
        elif total_risk_score <= 70:
            risk_level = RiskLevel.HIGH
            status = ValidationStatus.WARNING
        else:
            risk_level = RiskLevel.CRITICAL
            status = ValidationStatus.FAILED
        
        details['risk_level'] = risk_level.value
        suggestions = self._generate_risk_mitigation_suggestions(details)
        
        return ValidationResult(
            status=status,
            score=100 - total_risk_score,  # 风险越低，分数越高
            details=details,
            timestamp=datetime.datetime.now().isoformat(),
            suggestions=suggestions
        )
    
    def _assess_technical_risk(self, idea: CreativeIdea) -> float:
        """评估技术风险"""
        risk_score = 0.0
        
        # 高风险技术
        high_risk_tech = ['量子计算', '脑机接口', '基因编辑', '纳米技术']
        for tech in high_risk_tech:
            if tech in idea.description:
                risk_score += 30
        
        # 中等风险技术
        medium_risk_tech = ['AI', '机器学习', '区块链', '物联网']
        for tech in medium_risk_tech:
            if tech in idea.description:
                risk_score += 15
        
        # 技术复杂度
        complexity_indicators = len(idea.description.split()) / 5
        risk_score += complexity_indicators
        
        # 技术依赖性
        dependency_keywords = ['依赖', '基于', '使用', '采用']
        dependency_count = sum(1 for keyword in dependency_keywords if keyword in idea.description)
        risk_score += dependency_count * 5
        
        return min(100, risk_score)
    
    def _assess_market_risk(self, idea: CreativeIdea) -> float:
        """评估市场风险"""
        risk_score = 20.0  # 基础市场风险
        
        # 竞争激烈程度
        competitive_keywords = ['竞争', '竞争激烈', '红海', '同质化']
        competitive_count = sum(1 for keyword in competitive_keywords if keyword in idea.description)
        risk_score += competitive_count * 15
        
        # 市场不确定性
        uncertain_keywords = ['不确定', '变化', '波动', '新兴']
        uncertain_count = sum(1 for keyword in uncertain_keywords if keyword in idea.description)
        risk_score += uncertain_count * 10
        
        # 用户接受度风险
        acceptance_risk = ['用户接受', '习惯改变', '学习成本']
        for risk in acceptance_risk:
            if risk in idea.description:
                risk_score += 12
        
        # 技术接受度
        if '新技术' in idea.description or '创新技术' in idea.description:
            risk_score += 15
        
        return min(100, risk_score)
    
    def _assess_financial_risk(self, idea: CreativeIdea) -> float:
        """评估财务风险"""
        risk_score = 15.0  # 基础财务风险
        
        # 投资需求
        investment_keywords = ['投资', '融资', '资金', '预算', '成本']
        investment_count = sum(1 for keyword in investment_keywords if keyword in idea.description)
        risk_score += investment_count * 8
        
        # 盈利不确定性
        if '盈利模式' not in idea.description and '商业模式' not in idea.description:
            risk_score += 20  # 缺乏明确的商业模式
        
        # 财务复杂度
        financial_complexity = ['多层次', '复杂', '多元化']
        complexity_count = sum(1 for keyword in financial_complexity if keyword in idea.description)
        risk_score += complexity_count * 10
        
        # 现金流风险
        cashflow_keywords = ['现金流', '回款', '周期']
        cashflow_count = sum(1 for keyword in cashflow_keywords if keyword in idea.description)
        risk_score += cashflow_count * 12
        
        return min(100, risk_score)
    
    def _assess_operational_risk(self, idea: CreativeIdea) -> float:
        """评估运营风险"""
        risk_score = 10.0  # 基础运营风险
        
        # 人员依赖性
        personnel_keywords = ['关键人员', '专业人才', '团队依赖']
        personnel_count = sum(1 for keyword in personnel_keywords if keyword in idea.description)
        risk_score += personnel_count * 15
        
        # 运营复杂度
        operational_keywords = ['复杂', '多样化', '多环节']
        operational_count = sum(1 for keyword in operational_keywords if keyword in idea.description)
        risk_score += operational_count * 8
        
        # 供应链风险
        supply_keywords = ['供应链', '供应商', '依赖外部']
        supply_count = sum(1 for keyword in supply_keywords if keyword in idea.description)
        risk_score += supply_count * 12
        
        # 质量控制
        if '质量控制' not in idea.description:
            risk_score += 10  # 缺乏质量控制考虑
        
        return min(100, risk_score)
    
    def _assess_legal_risk(self, idea: CreativeIdea) -> float:
        """评估法律风险"""
        risk_score = 5.0  # 基础法律风险
        
        # 监管风险行业
        regulated_industries = ['金融', '医疗', '教育', '数据', '隐私']
        for industry in regulated_industries:
            if industry in idea.description:
                risk_score += 25
        
        # 知识产权风险
        ip_keywords = ['专利', '版权', '商标', '知识产权']
        ip_count = sum(1 for keyword in ip_keywords if keyword in idea.description)
        risk_score += ip_count * 8
        
        # 合规要求
        compliance_keywords = ['合规', '监管', '政策', '标准']
        compliance_count = sum(1 for keyword in compliance_keywords if keyword in idea.description)
        risk_score += compliance_count * 5
        
        # 跨国业务
        if '国际' in idea.description or '跨国' in idea.description:
            risk_score += 15
        
        return min(100, risk_score)
    
    def _generate_risk_mitigation_suggestions(self, details: Dict[str, float]) -> List[str]:
        """生成风险缓解建议"""
        suggestions = []
        
        if details.get('technical_risk', 0) > 50:
            suggestions.append("建议进行技术可行性研究，制定技术风险缓解计划")
        
        if details.get('market_risk', 0) > 50:
            suggestions.append("建议进行市场调研，了解竞争态势和用户需求")
        
        if details.get('financial_risk', 0) > 50:
            suggestions.append("建议制定详细的财务计划和风险控制措施")
        
        if details.get('operational_risk', 0) > 50:
            suggestions.append("建议建立完善的运营流程和质量控制体系")
        
        if details.get('legal_risk', 0) > 50:
            suggestions.append("建议咨询法律专家，确保合规性和知识产权保护")
        
        return suggestions


class CreativeImprovementAdvisor:
    """创意改进建议器"""
    
    def generate_suggestions(self, idea: CreativeIdea, feasibility_result: ValidationResult,
                          validity_result: ValidationResult, consistency_result: ValidationResult,
                          quality_result: ValidationResult, risk_result: ValidationResult) -> List[Dict[str, Any]]:
        """生成综合改进建议"""
        all_suggestions = []
        
        # 基于各项评估结果生成建议
        suggestion_categories = {
            'feasibility': feasibility_result.suggestions,
            'validity': validity_result.suggestions,
            'consistency': consistency_result.suggestions,
            'quality': quality_result.suggestions,
            'risk': risk_result.suggestions
        }
        
        for category, suggestions in suggestion_categories.items():
            for suggestion in suggestions:
                all_suggestions.append({
                    'category': category,
                    'suggestion': suggestion,
                    'priority': self._calculate_priority(category, suggestion),
                    'impact': self._estimate_impact(category, suggestion),
                    'effort': self._estimate_effort(category, suggestion)
                })
        
        # 按优先级排序
        all_suggestions.sort(key=lambda x: x['priority'], reverse=True)
        
        return all_suggestions
    
    def _calculate_priority(self, category: str, suggestion: str) -> int:
        """计算建议优先级"""
        priority_score = 50  # 基础优先级
        
        # 基于类别的权重
        category_weights = {
            'feasibility': 25,
            'validity': 20,
            'consistency': 15,
            'quality': 20,
            'risk': 30
        }
        priority_score += category_weights.get(category, 10)
        
        # 基于建议内容的紧急程度
        urgent_keywords = ['紧急', '立即', '马上', '关键', '重要']
        urgent_count = sum(1 for keyword in urgent_keywords if keyword in suggestion)
        priority_score += urgent_count * 10
        
        # 基于建议的影响范围
        impact_keywords = ['全面', '整体', '核心', '基础']
        impact_count = sum(1 for keyword in impact_keywords if keyword in suggestion)
        priority_score += impact_count * 8
        
        return min(100, priority_score)
    
    def _estimate_impact(self, category: str, suggestion: str) -> str:
        """估算建议影响"""
        high_impact_keywords = ['核心', '关键', '基础', '根本', '重要']
        medium_impact_keywords = ['改进', '优化', '提升', '增强']
        
        for keyword in high_impact_keywords:
            if keyword in suggestion:
                return '高'
        
        for keyword in medium_impact_keywords:
            if keyword in suggestion:
                return '中'
        
        return '低'
    
    def _estimate_effort(self, category: str, suggestion: str) -> str:
        """估算实施难度"""
        high_effort_keywords = ['重新设计', '重构', '全面', '系统']
        medium_effort_keywords = ['改进', '优化', '调整', '完善']
        
        for keyword in high_effort_keywords:
            if keyword in suggestion:
                return '高'
        
        for keyword in medium_effort_keywords:
            if keyword in suggestion:
                return '中'
        
        return '低'


class CreativeValidationReport:
    """创意验证报告生成器"""
    
    def generate_report(self, idea: CreativeIdea, feasibility_result: ValidationResult,
                       validity_result: ValidationResult, consistency_result: ValidationResult,
                       quality_result: ValidationResult, risk_result: ValidationResult,
                       improvement_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成完整的验证报告"""
        
        # 计算综合评分
        overall_score = self._calculate_overall_score(
            feasibility_result, validity_result, consistency_result, 
            quality_result, risk_result
        )
        
        # 确定整体状态
        overall_status = self._determine_overall_status(
            feasibility_result, validity_result, consistency_result,
            quality_result, risk_result
        )
        
        # 生成执行摘要
        executive_summary = self._generate_executive_summary(
            idea, overall_score, overall_status
        )
        
        # 生成详细分析
        detailed_analysis = self._generate_detailed_analysis(
            feasibility_result, validity_result, consistency_result,
            quality_result, risk_result
        )
        
        # 生成行动计划
        action_plan = self._generate_action_plan(improvement_suggestions)
        
        report = {
            'report_info': {
                'report_id': f"CV-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                'generated_at': datetime.datetime.now().isoformat(),
                'validator_version': '1.0.0'
            },
            'idea_info': asdict(idea),
            'executive_summary': executive_summary,
            'overall_assessment': {
                'score': overall_score,
                'status': overall_status.value,
                'grade': self._determine_quality_grade(overall_score)
            },
            'detailed_analysis': detailed_analysis,
            'improvement_suggestions': improvement_suggestions,
            'action_plan': action_plan,
            'risk_alerts': self._generate_risk_alerts(risk_result),
            'recommendations': self._generate_recommendations(overall_score, overall_status)
        }
        
        return report
    
    def _calculate_overall_score(self, feasibility_result: ValidationResult,
                               validity_result: ValidationResult, consistency_result: ValidationResult,
                               quality_result: ValidationResult, risk_result: ValidationResult) -> float:
        """计算综合评分"""
        weights = {
            'feasibility': 0.25,
            'validity': 0.25,
            'consistency': 0.20,
            'quality': 0.20,
            'risk': 0.10
        }
        
        total_score = (
            feasibility_result.score * weights['feasibility'] +
            validity_result.score * weights['validity'] +
            consistency_result.score * weights['consistency'] +
            quality_result.score * weights['quality'] +
            risk_result.score * weights['risk']
        )
        
        return round(total_score, 2)
    
    def _determine_overall_status(self, feasibility_result: ValidationResult,
                                validity_result: ValidationResult, consistency_result: ValidationResult,
                                quality_result: ValidationResult, risk_result: ValidationResult) -> ValidationStatus:
        """确定整体状态"""
        results = [feasibility_result, validity_result, consistency_result, quality_result, risk_result]
        
        # 如果有任何关键项失败，整体状态为失败
        failed_results = [r for r in results if r.status == ValidationStatus.FAILED]
        if failed_results:
            return ValidationStatus.FAILED
        
        # 计算通过率
        passed_results = [r for r in results if r.status == ValidationStatus.PASSED]
        pass_rate = len(passed_results) / len(results)
        
        if pass_rate >= 0.8:
            return ValidationStatus.PASSED
        elif pass_rate >= 0.6:
            return ValidationStatus.PARTIAL
        elif pass_rate >= 0.4:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.FAILED
    
    def _determine_quality_grade(self, score: float) -> str:
        """确定质量等级"""
        if score >= 85:
            return QualityGrade.EXCELLENT.value
        elif score >= 75:
            return QualityGrade.GOOD.value
        elif score >= 60:
            return QualityGrade.AVERAGE.value
        elif score >= 45:
            return QualityGrade.POOR.value
        else:
            return QualityGrade.UNACCEPTABLE.value
    
    def _generate_executive_summary(self, idea: CreativeIdea, overall_score: float, status: ValidationStatus) -> Dict[str, Any]:
        """生成执行摘要"""
        return {
            'idea_title': idea.title,
            'overall_score': overall_score,
            'overall_status': status.value,
            'key_findings': self._extract_key_findings(idea, overall_score),
            'primary_recommendations': self._extract_primary_recommendations(overall_score, status),
            'next_steps': self._suggest_next_steps(status)
        }
    
    def _generate_detailed_analysis(self, feasibility_result: ValidationResult,
                                  validity_result: ValidationResult, consistency_result: ValidationResult,
                                  quality_result: ValidationResult, risk_result: ValidationResult) -> Dict[str, Any]:
        """生成详细分析"""
        return {
            'feasibility_analysis': {
                'score': feasibility_result.score,
                'status': feasibility_result.status.value,
                'details': feasibility_result.details,
                'key_points': self._extract_key_points(feasibility_result.details)
            },
            'validity_analysis': {
                'score': validity_result.score,
                'status': validity_result.status.value,
                'details': validity_result.details,
                'key_points': self._extract_key_points(validity_result.details)
            },
            'consistency_analysis': {
                'score': consistency_result.score,
                'status': consistency_result.status.value,
                'details': consistency_result.details,
                'key_points': self._extract_key_points(consistency_result.details)
            },
            'quality_analysis': {
                'score': quality_result.score,
                'status': quality_result.status.value,
                'details': quality_result.details,
                'key_points': self._extract_key_points(quality_result.details)
            },
            'risk_analysis': {
                'score': risk_result.score,
                'status': risk_result.status.value,
                'details': risk_result.details,
                'key_points': self._extract_key_points(risk_result.details)
            }
        }
    
    def _generate_action_plan(self, improvement_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成行动计划"""
        high_priority = [s for s in improvement_suggestions if s['priority'] >= 70]
        medium_priority = [s for s in improvement_suggestions if 40 <= s['priority'] < 70]
        low_priority = [s for s in improvement_suggestions if s['priority'] < 40]
        
        return {
            'immediate_actions': high_priority[:5],  # 前5个高优先级建议
            'short_term_actions': medium_priority[:8],  # 前8个中优先级建议
            'long_term_actions': low_priority[:10],  # 前10个低优先级建议
            'timeline': self._create_action_timeline(high_priority, medium_priority, low_priority)
        }
    
    def _generate_risk_alerts(self, risk_result: ValidationResult) -> List[Dict[str, Any]]:
        """生成风险警报"""
        alerts = []
        
        for risk_type, risk_score in risk_result.details.items():
            if risk_type == 'risk_level':
                continue
                
            if risk_score > 70:
                alerts.append({
                    'type': risk_type,
                    'level': '高',
                    'message': f"{risk_type}风险过高，需要立即关注",
                    'score': risk_score
                })
            elif risk_score > 50:
                alerts.append({
                    'type': risk_type,
                    'level': '中',
                    'message': f"{risk_type}风险中等，建议制定应对措施",
                    'score': risk_score
                })
        
        return alerts
    
    def _generate_recommendations(self, overall_score: float, status: ValidationStatus) -> List[str]:
        """生成总体建议"""
        recommendations = []
        
        if status == ValidationStatus.PASSED:
            recommendations.append("创意整体表现优秀，建议进入实施阶段")
            recommendations.append("可以开始制定详细的实施计划和时间表")
        elif status == ValidationStatus.PARTIAL:
            recommendations.append("创意基本可行，但需要解决部分问题")
            recommendations.append("建议优先处理高优先级改进建议")
        elif status == ValidationStatus.WARNING:
            recommendations.append("创意存在较多问题，需要大幅改进")
            recommendations.append("建议重新审视创意核心概念")
        else:
            recommendations.append("创意当前不适合实施")
            recommendations.append("建议重新构思或寻找替代方案")
        
        return recommendations
    
    def _extract_key_findings(self, idea: CreativeIdea, overall_score: float) -> List[str]:
        """提取关键发现"""
        findings = []
        
        if overall_score >= 80:
            findings.append("创意具有很强的市场潜力")
        elif overall_score >= 60:
            findings.append("创意具有一定的发展前景")
        else:
            findings.append("创意需要重大改进")
        
        # 基于创意内容的关键发现
        if len(idea.tags) > 5:
            findings.append("创意标签丰富，覆盖面广")
        
        if len(idea.expected_outcomes) > 3:
            findings.append("预期成果明确且多样化")
        
        return findings
    
    def _extract_primary_recommendations(self, overall_score: float, status: ValidationStatus) -> List[str]:
        """提取主要建议"""
        if status == ValidationStatus.PASSED:
            return ["立即启动项目", "制定实施计划", "组建执行团队"]
        elif status == ValidationStatus.PARTIAL:
            return ["解决关键问题", "完善创意细节", "降低主要风险"]
        elif status == ValidationStatus.WARNING:
            return ["重新评估创意", "寻找改进方向", "考虑替代方案"]
        else:
            return ["暂停项目", "重新构思", "寻求专业建议"]
    
    def _suggest_next_steps(self, status: ValidationStatus) -> List[str]:
        """建议下一步行动"""
        if status == ValidationStatus.PASSED:
            return [
                "制定详细的项目计划",
                "确定资源需求和预算",
                "组建项目团队",
                "开始原型开发"
            ]
        elif status == ValidationStatus.PARTIAL:
            return [
                "优先解决关键问题",
                "进行可行性研究",
                "收集用户反馈",
                "完善商业模式"
            ]
        elif status == ValidationStatus.WARNING:
            return [
                "深入分析问题根源",
                "寻求专家意见",
                "探索改进方案",
                "重新评估可行性"
            ]
        else:
            return [
                "暂停当前方向",
                "重新市场调研",
                "寻找新机会",
                "考虑合作或收购"
            ]
    
    def _extract_key_points(self, details: Dict[str, Any]) -> List[str]:
        """从详情中提取关键点"""
        key_points = []
        
        for key, value in details.items():
            if isinstance(value, (int, float)) and value >= 80:
                key_points.append(f"{key}表现优秀")
            elif isinstance(value, (int, float)) and value <= 40:
                key_points.append(f"{key}需要改进")
        
        return key_points
    
    def _create_action_timeline(self, high_priority: List, medium_priority: List, low_priority: List) -> Dict[str, List]:
        """创建行动时间表"""
        return {
            'immediate': [s['suggestion'] for s in high_priority[:3]],
            'this_month': [s['suggestion'] for s in high_priority[3:6] + medium_priority[:2]],
            'this_quarter': [s['suggestion'] for s in medium_priority[2:6]],
            'long_term': [s['suggestion'] for s in low_priority[:5]]
        }


# 使用示例和测试函数
def demo_creative_validator():
    """演示创意验证器的使用"""
    
    # 创建示例创意
    sample_idea = CreativeIdea(
        id="idea_001",
        title="智能健康监测手环",
        description="基于AI技术的智能健康监测手环，可以实时监测心率、血压、血氧等生理指标，并通过机器学习算法预测健康风险，提供个性化健康建议。产品采用低功耗蓝牙技术，支持与手机APP无缝连接，数据云端同步存储。",
        category="智能硬件",
        tags=["AI", "健康监测", "可穿戴设备", "机器学习", "物联网"],
        target_audience="关注健康的中老年人群和健身爱好者",
        resources_required=["硬件研发团队", "AI算法工程师", "移动APP开发", "云服务器", "医疗认证"],
        expected_outcomes=["实时健康监测", "风险预警", "个性化建议", "数据同步", "用户健康改善"],
        constraints=["医疗设备认证要求", "电池续航时间", "成本控制", "用户隐私保护"],
        created_at=datetime.datetime.now().isoformat(),
        metadata={"industry": "健康科技", "target_market": "消费级"}
    )
    
    # 创建验证器实例
    validator = CreativeValidator()
    
    # 执行验证
    print("开始创意验证...")
    result = validator.validate_creative(sample_idea)
    
    # 输出结果
    print(f"\n=== 创意验证报告 ===")
    print(f"创意标题: {result['idea_info']['title']}")
    print(f"综合评分: {result['overall_assessment']['score']}")
    print(f"验证状态: {result['overall_assessment']['status']}")
    print(f"质量等级: {result['overall_assessment']['grade']}")
    
    print(f"\n=== 执行摘要 ===")
    summary = result['executive_summary']
    for finding in summary['key_findings']:
        print(f"• {finding}")
    
    print(f"\n=== 改进建议 ===")
    for i, suggestion in enumerate(result['improvement_suggestions'][:5], 1):
        print(f"{i}. [{suggestion['category']}] {suggestion['suggestion']} (优先级: {suggestion['priority']})")
    
    print(f"\n=== 风险警报 ===")
    for alert in result['risk_alerts']:
        print(f"• {alert['type']}风险 ({alert['level']}): {alert['message']}")
    
    return result


if __name__ == "__main__":
    # 运行演示
    demo_result = demo_creative_validator()
    
    # 保存报告到文件
    with open('/workspace/creative_validation_report.json', 'w', encoding='utf-8') as f:
        json.dump(demo_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n验证报告已保存到: creative_validation_report.json")