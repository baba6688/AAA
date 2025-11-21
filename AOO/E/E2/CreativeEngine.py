"""
E2 创意引擎实现
Enhanced Creative Engine - 提供多模态创意生成、评估和优化功能

主要功能：
1. 创意生成算法（随机、联想、类比等）
2. 多模态创意生成（文本、图像、代码等）
3. 创意组合和融合
4. 创意质量评估和筛选
5. 创意进化和优化
6. 创意库管理和检索
7. 创意应用和实现
"""

import json
import random
import numpy as np
import hashlib
import sqlite3
import pickle
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import re
import math
from collections import defaultdict, Counter
import itertools


class CreativeType(Enum):
    """创意类型枚举"""
    TEXT = "text"
    IMAGE = "image" 
    CODE = "code"
    MUSIC = "music"
    VIDEO = "video"
    CONCEPT = "concept"
    SOLUTION = "solution"
    HYPOTHESIS = "hypothesis"
    DESIGN = "design"


class QualityMetric(Enum):
    """质量评估指标枚举"""
    ORIGINALITY = "originality"        # 原创性
    FEASIBILITY = "feasibility"        # 可行性
    INNOVATION = "innovation"          # 创新性
    RELEVANCE = "relevance"           # 相关性
    COMPLEXITY = "complexity"         # 复杂性
    BEAUTY = "beauty"                 # 美观性
    IMPACT = "impact"                 # 影响力
    SIMPLICITY = "simplicity"         # 简洁性


@dataclass
class CreativeIdea:
    """创意Idea数据结构"""
    id: str
    content: Any
    type: CreativeType
    metadata: Dict[str, Any]
    quality_scores: Dict[QualityMetric, float]
    created_at: datetime
    updated_at: datetime
    parent_ids: List[str]
    tags: Set[str]
    source: str
    confidence: float
    description: str = ""
    
    def __post_init__(self):
        if isinstance(self.quality_scores, dict) and len(self.quality_scores) > 0:
            if not isinstance(list(self.quality_scores.keys())[0], QualityMetric):
                # 转换字符串键为QualityMetric枚举
                new_scores = {}
                for k, v in self.quality_scores.items():
                    if isinstance(k, str):
                        new_scores[QualityMetric(k)] = v
                    else:
                        new_scores[k] = v
                self.quality_scores = new_scores
        
        if isinstance(self.tags, list):
            self.tags = set(self.tags)


class CreativeGenerator(ABC):
    """创意生成器抽象基类"""
    
    @abstractmethod
    def generate(self, seed: Any, context: Dict[str, Any]) -> List[CreativeIdea]:
        """生成创意"""
        pass


class RandomCreativeGenerator(CreativeGenerator):
    """随机创意生成器"""
    
    def __init__(self):
        self.text_templates = [
            "一个关于{subject}的{adjective}想法",
            "如何用{approach}解决{problem}",
            "想象一个{adjective}的{subject}",
            "创造一个{adjective}的{concept}",
            "设计一个{adjective}的{solution}"
        ]
        
        self.code_templates = [
            "实现一个{function}的算法",
            "创建一个{tool}工具",
            "开发一个{application}应用",
            "构建一个{framework}框架",
            "设计一个{api}接口"
        ]
        
        self.concept_templates = [
            "新的{field}理论",
            "创新的{method}方法",
            "革命性的{technology}技术",
            "突破性的{approach}方法",
            "颠覆性的{concept}概念"
        ]
    
    def generate(self, seed: Any, context: Dict[str, Any]) -> List[CreativeIdea]:
        """基于种子生成随机创意"""
        ideas = []
        seed_str = str(seed)
        
        # 文本创意生成
        for _ in range(3):
            template = random.choice(self.text_templates)
            idea = self._fill_template(template, context)
            creative_idea = CreativeIdea(
                id=self._generate_id(idea),
                content=idea,
                type=CreativeType.TEXT,
                metadata={"template": template, "seed": seed_str},
                quality_scores=self._evaluate_random_quality(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                parent_ids=[],
                tags=set(),
                source="random_generator",
                confidence=random.uniform(0.3, 0.8)
            )
            ideas.append(creative_idea)
        
        # 代码创意生成
        for _ in range(2):
            template = random.choice(self.code_templates)
            idea = self._fill_template(template, context)
            creative_idea = CreativeIdea(
                id=self._generate_id(idea),
                content=idea,
                type=CreativeType.CODE,
                metadata={"template": template, "seed": seed_str},
                quality_scores=self._evaluate_random_quality(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                parent_ids=[],
                tags=set(),
                source="random_generator",
                confidence=random.uniform(0.3, 0.8)
            )
            ideas.append(creative_idea)
        
        # 概念创意生成
        for _ in range(2):
            template = random.choice(self.concept_templates)
            idea = self._fill_template(template, context)
            creative_idea = CreativeIdea(
                id=self._generate_id(idea),
                content=idea,
                type=CreativeType.CONCEPT,
                metadata={"template": template, "seed": seed_str},
                quality_scores=self._evaluate_random_quality(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                parent_ids=[],
                tags=set(),
                source="random_generator",
                confidence=random.uniform(0.3, 0.8)
            )
            ideas.append(creative_idea)
        
        return ideas
    
    def _fill_template(self, template: str, context: Dict[str, Any]) -> str:
        """填充模板"""
        words = {
            "subject": random.choice(["AI", "机器人", "算法", "系统", "平台"]),
            "adjective": random.choice(["创新", "智能", "高效", "优雅", "简洁"]),
            "approach": random.choice(["机器学习", "深度学习", "强化学习", "联邦学习"]),
            "problem": random.choice(["优化", "预测", "分类", "聚类", "生成"]),
            "concept": random.choice(["模型", "架构", "框架", "方法", "理论"]),
            "solution": random.choice(["算法", "系统", "工具", "平台", "服务"]),
            "function": random.choice(["排序", "搜索", "优化", "预测", "分类"]),
            "tool": random.choice(["分析", "可视化", "监控", "测试", "部署"]),
            "application": random.choice(["推荐", "聊天", "翻译", "识别", "生成"]),
            "framework": random.choice(["Web", "数据", "机器学习", "区块链", "物联网"]),
            "api": random.choice(["REST", "GraphQL", "WebSocket", "gRPC", "Webhook"]),
            "field": random.choice(["计算机科学", "数学", "物理", "生物", "心理学"]),
            "method": random.choice(["研究", "实验", "分析", "建模", "验证"]),
            "technology": random.choice(["量子计算", "脑机接口", "纳米技术", "生物技术", "新能源"]),
        }
        
        return template.format(**words)
    
    def _generate_id(self, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(f"{content}_{time.time()}".encode()).hexdigest()
    
    def _evaluate_random_quality(self) -> Dict[QualityMetric, float]:
        """评估随机创意的质量"""
        return {
            QualityMetric.ORIGINALITY: random.uniform(0.3, 0.9),
            QualityMetric.FEASIBILITY: random.uniform(0.4, 0.8),
            QualityMetric.INNOVATION: random.uniform(0.3, 0.8),
            QualityMetric.RELEVANCE: random.uniform(0.5, 0.9),
            QualityMetric.COMPLEXITY: random.uniform(0.3, 0.7),
            QualityMetric.BEAUTY: random.uniform(0.4, 0.8),
            QualityMetric.IMPACT: random.uniform(0.3, 0.7),
            QualityMetric.SIMPLICITY: random.uniform(0.4, 0.9)
        }


class AssociativeCreativeGenerator(CreativeGenerator):
    """联想创意生成器"""
    
    def __init__(self):
        self.association_networks = {
            "AI": ["机器学习", "神经网络", "深度学习", "算法", "数据", "模型", "训练", "预测"],
            "机器人": ["自动化", "机械", "传感器", "控制", "智能", "执行", "感知", "交互"],
            "算法": ["优化", "搜索", "排序", "图论", "动态规划", "贪心", "分治", "回溯"],
            "系统": ["架构", "设计", "性能", "可扩展", "可靠性", "安全性", "维护", "部署"],
            "平台": ["服务", "接口", "集成", "生态系统", "用户", "数据", "API", "云"]
        }
    
    def generate(self, seed: Any, context: Dict[str, Any]) -> List[CreativeIdea]:
        """基于联想生成创意"""
        ideas = []
        seed_str = str(seed)
        
        # 找到相关的联想词汇
        associations = self._find_associations(seed_str)
        
        # 生成联想创意
        for i, (word, strength) in enumerate(associations[:5]):
            idea = f"将{seed_str}与{word}结合，创造新的{self._get_domain_type(seed_str)}"
            creative_idea = CreativeIdea(
                id=self._generate_id(idea),
                content=idea,
                type=CreativeType.CONCEPT,
                metadata={"seed": seed_str, "association": word, "strength": strength},
                quality_scores=self._evaluate_associative_quality(strength),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                parent_ids=[],
                tags={seed_str, word},
                source="associative_generator",
                confidence=strength
            )
            ideas.append(creative_idea)
        
        return ideas
    
    def _find_associations(self, word: str) -> List[Tuple[str, float]]:
        """找到词汇的联想词汇"""
        associations = []
        
        # 直接匹配
        if word in self.association_networks:
            for associated_word in self.association_networks[word]:
                associations.append((associated_word, random.uniform(0.7, 1.0)))
        
        # 模糊匹配和语义联想
        for key, values in self.association_networks.items():
            if key != word and any(char in word for char in key):
                for associated_word in values:
                    strength = random.uniform(0.4, 0.7)
                    associations.append((associated_word, strength))
        
        # 随机联想
        for _ in range(3):
            random_word = random.choice(list(self.association_networks.keys()))
            associations.append((random_word, random.uniform(0.3, 0.6)))
        
        return sorted(associations, key=lambda x: x[1], reverse=True)
    
    def _get_domain_type(self, word: str) -> str:
        """获取词汇的领域类型"""
        domain_mapping = {
            "AI": "智能系统",
            "机器人": "自动化解决方案", 
            "算法": "计算方法",
            "系统": "技术架构",
            "平台": "服务平台"
        }
        return domain_mapping.get(word, "创新方案")
    
    def _generate_id(self, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(f"{content}_{time.time()}".encode()).hexdigest()
    
    def _evaluate_associative_quality(self, strength: float) -> Dict[QualityMetric, float]:
        """评估联想创意的质量"""
        return {
            QualityMetric.ORIGINALITY: min(0.9, strength + 0.2),
            QualityMetric.FEASIBILITY: min(0.9, strength + 0.1),
            QualityMetric.INNOVATION: min(0.9, strength + 0.15),
            QualityMetric.RELEVANCE: min(0.95, strength + 0.25),
            QualityMetric.COMPLEXITY: random.uniform(0.4, 0.8),
            QualityMetric.BEAUTY: random.uniform(0.5, 0.9),
            QualityMetric.IMPACT: min(0.9, strength + 0.1),
            QualityMetric.SIMPLICITY: random.uniform(0.5, 0.9)
        }


class AnalogicalCreativeGenerator(CreativeGenerator):
    """类比创意生成器"""
    
    def __init__(self):
        self.analogy_patterns = {
            "自然现象": {
                "pattern": "模仿{自然现象}的{特征}，应用到{应用领域}",
                "examples": ["鸟类的飞行", "蜂巢的结构", "蚂蚁的协作", "蜘蛛的网络"]
            },
            "生物系统": {
                "pattern": "借鉴{生物系统}的{机制}，设计{目标系统}",
                "examples": ["神经网络", "免疫系统", "DNA结构", "生态系统"]
            },
            "物理原理": {
                "pattern": "运用{物理原理}，创造{技术方案}",
                "examples": ["量子纠缠", "电磁感应", "重力原理", "光的传播"]
            },
            "社会系统": {
                "pattern": "参考{社会系统}的组织方式，构建{组织结构}",
                "examples": ["民主制度", "市场经济", "企业架构", "团队协作"]
            }
        }
    
    def generate(self, seed: Any, context: Dict[str, Any]) -> List[CreativeIdea]:
        """基于类比生成创意"""
        ideas = []
        seed_str = str(seed)
        
        # 选择类比模式
        for category, pattern_info in self.analogy_patterns.items():
            pattern = pattern_info["pattern"]
            examples = pattern_info["examples"]
            
            # 生成类比创意
            for example in examples[:2]:
                idea = pattern.format(
                    自然现象=example if "自然现象" in category else "",
                    生物系统=example if "生物系统" in category else "",
                    物理原理=example if "物理原理" in category else "",
                    社会系统=example if "社会系统" in category else "",
                    特征=random.choice(["结构", "行为", "机制", "原理"]),
                    应用领域=seed_str,
                    机制=random.choice["工作原理", "组织方式", "运行机制"],
                    目标系统=seed_str,
                    技术方案=seed_str,
                    组织结构=seed_str
                )
                
                creative_idea = CreativeIdea(
                    id=self._generate_id(idea),
                    content=idea,
                    type=CreativeType.CONCEPT,
                    metadata={"seed": seed_str, "analogy": example, "category": category},
                    quality_scores=self._evaluate_analogical_quality(),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    parent_ids=[],
                    tags={seed_str, category},
                    source="analogical_generator",
                    confidence=random.uniform(0.6, 0.9)
                )
                ideas.append(creative_idea)
        
        return ideas
    
    def _generate_id(self, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(f"{content}_{time.time()}".encode()).hexdigest()
    
    def _evaluate_analogical_quality(self) -> Dict[QualityMetric, float]:
        """评估类比创意的质量"""
        return {
            QualityMetric.ORIGINALITY: random.uniform(0.6, 0.9),
            QualityMetric.FEASIBILITY: random.uniform(0.5, 0.8),
            QualityMetric.INNOVATION: random.uniform(0.6, 0.9),
            QualityMetric.RELEVANCE: random.uniform(0.5, 0.8),
            QualityMetric.COMPLEXITY: random.uniform(0.5, 0.8),
            QualityMetric.BEAUTY: random.uniform(0.6, 0.9),
            QualityMetric.IMPACT: random.uniform(0.5, 0.8),
            QualityMetric.SIMPLICITY: random.uniform(0.4, 0.7)
        }


class QualityEvaluator:
    """创意质量评估器"""
    
    def __init__(self):
        self.evaluation_weights = {
            QualityMetric.ORIGINALITY: 0.2,
            QualityMetric.FEASIBILITY: 0.15,
            QualityMetric.INNOVATION: 0.2,
            QualityMetric.RELEVANCE: 0.15,
            QualityMetric.COMPLEXITY: 0.1,
            QualityMetric.BEAUTY: 0.1,
            QualityMetric.IMPACT: 0.05,
            QualityMetric.SIMPLICITY: 0.05
        }
    
    def evaluate_idea(self, idea: CreativeIdea) -> Dict[QualityMetric, float]:
        """评估单个创意的质量"""
        scores = {}
        
        # 原创性评估
        scores[QualityMetric.ORIGINALITY] = self._evaluate_originality(idea)
        
        # 可行性评估
        scores[QualityMetric.FEASIBILITY] = self._evaluate_feasibility(idea)
        
        # 创新性评估
        scores[QualityMetric.INNOVATION] = self._evaluate_innovation(idea)
        
        # 相关性评估
        scores[QualityMetric.RELEVANCE] = self._evaluate_relevance(idea)
        
        # 复杂性评估
        scores[QualityMetric.COMPLEXITY] = self._evaluate_complexity(idea)
        
        # 美观性评估
        scores[QualityMetric.BEAUTY] = self._evaluate_beauty(idea)
        
        # 影响力评估
        scores[QualityMetric.IMPACT] = self._evaluate_impact(idea)
        
        # 简洁性评估
        scores[QualityMetric.SIMPLICITY] = self._evaluate_simplicity(idea)
        
        return scores
    
    def _evaluate_originality(self, idea: CreativeIdea) -> float:
        """评估原创性"""
        # 基于内容长度、独特词汇等评估
        content_str = str(idea.content)
        unique_words = len(set(content_str.split()))
        total_words = len(content_str.split())
        
        if total_words == 0:
            return 0.0
        
        uniqueness_ratio = unique_words / total_words
        length_factor = min(1.0, len(content_str) / 100)
        
        return min(1.0, (uniqueness_ratio * 0.7 + length_factor * 0.3) * 1.2)
    
    def _evaluate_feasibility(self, idea: CreativeIdea) -> float:
        """评估可行性"""
        # 基于关键词和模式评估
        content_str = str(idea.content).lower()
        
        feasible_keywords = ["实现", "开发", "创建", "构建", "设计", "算法", "系统", "平台"]
        infeasible_keywords = ["完美", "无限", "万能", "绝对", "永远", "瞬间"]
        
        feasible_score = sum(1 for keyword in feasible_keywords if keyword in content_str)
        infeasible_score = sum(1 for keyword in infeasible_keywords if keyword in content_str)
        
        base_score = 0.5
        feasible_bonus = min(0.4, feasible_score * 0.1)
        infeasible_penalty = min(0.3, infeasible_score * 0.15)
        
        return max(0.1, min(1.0, base_score + feasible_bonus - infeasible_penalty))
    
    def _evaluate_innovation(self, idea: CreativeIdea) -> float:
        """评估创新性"""
        # 基于新颖词汇组合评估
        content_str = str(idea.content)
        
        innovative_patterns = [
            r"结合.*和.*", r"融合.*与.*", r"将.*应用于.*", 
            r"基于.*的.*", r"创新的.*方法", r"革命性的.*技术"
        ]
        
        innovation_score = 0.0
        for pattern in innovative_patterns:
            if re.search(pattern, content_str):
                innovation_score += 0.2
        
        # 基于技术词汇密度
        tech_words = ["AI", "机器学习", "深度学习", "区块链", "量子", "纳米", "生物技术"]
        tech_density = sum(1 for word in tech_words if word in content_str) / max(1, len(content_str.split()))
        
        return min(1.0, innovation_score + tech_density * 0.5 + 0.3)
    
    def _evaluate_relevance(self, idea: CreativeIdea) -> float:
        """评估相关性"""
        # 基于与上下文的匹配度
        if not idea.metadata.get("seed"):
            return 0.5
        
        seed = str(idea.metadata["seed"]).lower()
        content = str(idea.content).lower()
        
        # 计算词汇重叠度
        seed_words = set(seed.split())
        content_words = set(content.split())
        
        if len(seed_words) == 0:
            return 0.5
        
        overlap = len(seed_words.intersection(content_words))
        relevance = overlap / len(seed_words)
        
        return min(1.0, relevance + 0.3)
    
    def _evaluate_complexity(self, idea: CreativeIdea) -> float:
        """评估复杂性"""
        content = str(idea.content)
        
        # 基于句子长度、词汇复杂度等
        sentences = content.split('。')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # 技术术语密度
        tech_terms = ["算法", "架构", "框架", "系统", "模型", "网络", "协议", "接口"]
        tech_density = sum(1 for term in tech_terms if term in content) / max(1, len(content.split()))
        
        complexity = min(1.0, (avg_sentence_length / 20 + tech_density) / 2)
        
        return complexity
    
    def _evaluate_beauty(self, idea: CreativeIdea) -> float:
        """评估美观性"""
        content = str(idea.content)
        
        # 基于语言优美程度
        positive_words = ["优雅", "简洁", "美观", "和谐", "平衡", "对称", "流畅", "自然"]
        negative_words = ["复杂", "混乱", "冗余", "繁琐", "笨重", "粗糙"]
        
        positive_score = sum(1 for word in positive_words if word in content)
        negative_score = sum(1 for word in negative_words if word in content)
        
        base_beauty = 0.5
        positive_bonus = min(0.3, positive_score * 0.1)
        negative_penalty = min(0.2, negative_score * 0.1)
        
        return max(0.1, min(1.0, base_beauty + positive_bonus - negative_penalty))
    
    def _evaluate_impact(self, idea: CreativeIdea) -> float:
        """评估影响力"""
        content = str(idea.content)
        
        impact_keywords = ["革命性", "颠覆性", "突破性", "重大", "重要", "关键", "核心"]
        impact_score = sum(1 for keyword in impact_keywords if keyword in content)
        
        # 基于应用范围
        application_words = ["全球", "世界", "社会", "行业", "领域", "大众", "普及"]
        application_score = sum(1 for word in application_words if word in content)
        
        return min(1.0, (impact_score * 0.2 + application_score * 0.15 + 0.3))
    
    def _evaluate_simplicity(self, idea: CreativeIdea) -> float:
        """评估简洁性"""
        content = str(idea.content)
        
        # 基于内容长度和词汇简洁度
        word_count = len(content.split())
        char_count = len(content)
        
        # 简洁的指标
        simplicity_score = 1.0 - min(0.8, (word_count / 20 + char_count / 200) / 2)
        
        # 简洁词汇
        simple_words = ["简单", "直接", "明确", "清晰", "简洁", "明了"]
        simple_bonus = sum(1 for word in simple_words if word in content) * 0.1
        
        return min(1.0, simplicity_score + simple_bonus)
    
    def calculate_overall_score(self, scores: Dict[QualityMetric, float]) -> float:
        """计算综合质量分数"""
        weighted_sum = sum(
            scores[metric] * weight 
            for metric, weight in self.evaluation_weights.items()
            if metric in scores
        )
        
        total_weight = sum(
            weight for metric, weight in self.evaluation_weights.items()
            if metric in scores
        )
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class IdeaFusion:
    """创意融合器"""
    
    def __init__(self):
        self.fusion_strategies = [
            self._combine_concepts,
            self._merge_solutions,
            self._synthesize_patterns,
            self._blend_approaches,
            self._integrate_systems
        ]
    
    def fuse_ideas(self, ideas: List[CreativeIdea]) -> List[CreativeIdea]:
        """融合多个创意"""
        fused_ideas = []
        
        # 两两融合
        for i, idea1 in enumerate(ideas):
            for j, idea2 in enumerate(ideas[i+1:], i+1):
                fusion_result = self._fuse_pair(idea1, idea2)
                if fusion_result:
                    fused_ideas.append(fusion_result)
        
        # 多创意融合
        if len(ideas) > 2:
            multi_fusion = self._fuse_multiple(ideas)
            if multi_fusion:
                fused_ideas.extend(multi_fusion)
        
        return fused_ideas
    
    def _fuse_pair(self, idea1: CreativeIdea, idea2: CreativeIdea) -> Optional[CreativeIdea]:
        """融合两个创意"""
        # 选择融合策略
        strategy = random.choice(self.fusion_strategies)
        
        try:
            fused_content = strategy(idea1, idea2)
            if not fused_content:
                return None
            
            # 计算融合后的质量分数
            fused_scores = self._calculate_fused_scores(idea1, idea2)
            
            # 生成新创意
            fused_idea = CreativeIdea(
                id=self._generate_id(fused_content),
                content=fused_content,
                type=self._determine_fused_type(idea1.type, idea2.type),
                metadata={
                    "parent_ids": [idea1.id, idea2.id],
                    "fusion_strategy": strategy.__name__,
                    "original_types": [idea1.type.value, idea2.type.value]
                },
                quality_scores=fused_scores,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                parent_ids=[idea1.id, idea2.id],
                tags=idea1.tags.union(idea2.tags),
                source="fusion_engine",
                confidence=min(idea1.confidence, idea2.confidence) * 0.8
            )
            
            return fused_idea
            
        except Exception as e:
            print(f"融合失败: {e}")
            return None
    
    def _combine_concepts(self, idea1: CreativeIdea, idea2: CreativeIdea) -> str:
        """组合概念"""
        content1 = str(idea1.content)
        content2 = str(idea2.content)
        
        # 提取关键概念
        concepts1 = self._extract_concepts(content1)
        concepts2 = self._extract_concepts(content2)
        
        # 组合概念
        combined_concepts = concepts1[:2] + concepts2[:2]
        return f"结合{' '.join(combined_concepts)}，创造新的解决方案"
    
    def _merge_solutions(self, idea1: CreativeIdea, idea2: CreativeIdea) -> str:
        """合并解决方案"""
        return f"将{idea1.content}的{self._extract_solution_type(idea1.content)}与{idea2.content}的{self._extract_solution_type(idea2.content)}相结合"
    
    def _synthesize_patterns(self, idea1: CreativeIdea, idea2: CreativeIdea) -> str:
        """综合模式"""
        return f"综合{idea1.content}和{idea2.content}的模式，创造新的方法论"
    
    def _blend_approaches(self, idea1: CreativeIdea, idea2: CreativeIdea) -> str:
        """混合方法"""
        return f"融合{idea1.content}的方法和{idea2.content}的技巧，形成混合方案"
    
    def _integrate_systems(self, idea1: CreativeIdea, idea2: CreativeIdea) -> str:
        """整合系统"""
        return f"将{idea1.content}系统与{idea2.content}系统进行深度整合"
    
    def _extract_concepts(self, content: str) -> List[str]:
        """提取概念"""
        # 简单的关键词提取
        concepts = []
        keywords = ["算法", "系统", "方法", "技术", "模型", "框架", "平台", "工具"]
        
        for keyword in keywords:
            if keyword in content:
                concepts.append(keyword)
        
        return concepts if concepts else ["创新", "方案"]
    
    def _extract_solution_type(self, content: str) -> str:
        """提取解决方案类型"""
        solution_types = ["算法", "系统", "平台", "工具", "方法", "技术"]
        
        for solution_type in solution_types:
            if solution_type in content:
                return solution_type
        
        return "方案"
    
    def _determine_fused_type(self, type1: CreativeType, type2: CreativeType) -> CreativeType:
        """确定融合后的类型"""
        if type1 == type2:
            return type1
        elif type1 == CreativeType.CONCEPT or type2 == CreativeType.CONCEPT:
            return CreativeType.CONCEPT
        elif type1 == CreativeType.SOLUTION or type2 == CreativeType.SOLUTION:
            return CreativeType.SOLUTION
        else:
            return CreativeType.TEXT
    
    def _calculate_fused_scores(self, idea1: CreativeIdea, idea2: CreativeIdea) -> Dict[QualityMetric, float]:
        """计算融合后的质量分数"""
        fused_scores = {}
        
        # 对每个质量指标进行融合
        all_metrics = set(idea1.quality_scores.keys()).union(set(idea2.quality_scores.keys()))
        
        for metric in all_metrics:
            score1 = idea1.quality_scores.get(metric, 0.5)
            score2 = idea2.quality_scores.get(metric, 0.5)
            
            # 使用加权平均，考虑置信度
            weight1 = idea1.confidence
            weight2 = idea2.confidence
            
            fused_scores[metric] = (score1 * weight1 + score2 * weight2) / (weight1 + weight2)
        
        return fused_scores
    
    def _fuse_multiple(self, ideas: List[CreativeIdea]) -> List[CreativeIdea]:
        """融合多个创意"""
        if len(ideas) < 3:
            return []
        
        # 选择三个创意进行融合
        selected_ideas = random.sample(ideas, min(3, len(ideas)))
        
        # 综合所有内容
        all_content = " ".join(str(idea.content) for idea in selected_ideas)
        fused_content = f"综合{len(selected_ideas)}个创意：{all_content[:100]}..."
        
        # 计算综合质量分数
        avg_scores = {}
        all_metrics = set()
        for idea in selected_ideas:
            all_metrics.update(idea.quality_scores.keys())
        
        for metric in all_metrics:
            scores = [idea.quality_scores.get(metric, 0.5) for idea in selected_ideas]
            avg_scores[metric] = sum(scores) / len(scores)
        
        # 生成融合创意
        fused_idea = CreativeIdea(
            id=self._generate_id(fused_content),
            content=fused_content,
            type=CreativeType.CONCEPT,
            metadata={
                "parent_ids": [idea.id for idea in selected_ideas],
                "fusion_type": "multiple_fusion",
                "count": len(selected_ideas)
            },
            quality_scores=avg_scores,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_ids=[idea.id for idea in selected_ideas],
            tags=set().union(*[idea.tags for idea in selected_ideas]),
            source="multi_fusion_engine",
            confidence=min(idea.confidence for idea in selected_ideas) * 0.7
        )
        
        return [fused_idea]
    
    def _generate_id(self, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(f"{content}_{time.time()}_{random.random()}".encode()).hexdigest()


class EvolutionEngine:
    """创意进化引擎"""
    
    def __init__(self):
        self.evolution_strategies = [
            self._mutate_content,
            self._enhance_quality,
            self._simplify_complexity,
            self._increase_innovation,
            self._improve_feasibility
        ]
    
    def evolve_ideas(self, ideas: List[CreativeIdea], generations: int = 3) -> List[CreativeIdea]:
        """进化创意"""
        current_population = ideas.copy()
        
        for generation in range(generations):
            print(f"进化第 {generation + 1} 代...")
            
            # 选择和繁殖
            offspring = self._generate_offspring(current_population)
            
            # 评估和选择
            current_population = self._select_best(current_population + offspring)
        
        return current_population
    
    def _generate_offspring(self, population: List[CreativeIdea]) -> List[CreativeIdea]:
        """生成后代"""
        offspring = []
        
        # 对每个个体应用进化策略
        for idea in population:
            for strategy in self.evolution_strategies:
                try:
                    evolved_idea = strategy(idea)
                    if evolved_idea:
                        offspring.append(evolved_idea)
                except Exception as e:
                    print(f"进化策略失败: {e}")
        
        return offspring
    
    def _mutate_content(self, idea: CreativeIdea) -> Optional[CreativeIdea]:
        """变异内容"""
        content = str(idea.content)
        
        # 随机替换一些词汇
        words = content.split()
        if len(words) < 3:
            return None
        
        # 选择要变异的词汇
        mutation_count = max(1, len(words) // 5)
        mutation_indices = random.sample(range(len(words)), mutation_count)
        
        # 替换词汇
        replacement_words = ["创新", "智能", "高效", "优雅", "简洁", "强大", "灵活", "可靠"]
        
        mutated_words = words.copy()
        for idx in mutation_indices:
            mutated_words[idx] = random.choice(replacement_words)
        
        mutated_content = " ".join(mutated_words)
        
        # 创建新的创意
        mutated_idea = CreativeIdea(
            id=self._generate_id(mutated_content),
            content=mutated_content,
            type=idea.type,
            metadata={
                **idea.metadata,
                "evolution_type": "mutation",
                "parent_id": idea.id,
                "mutated_words": mutation_count
            },
            quality_scores=idea.quality_scores.copy(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_ids=[idea.id],
            tags=idea.tags.copy(),
            source="evolution_engine",
            confidence=idea.confidence * 0.9
        )
        
        return mutated_idea
    
    def _enhance_quality(self, idea: CreativeIdea) -> Optional[CreativeIdea]:
        """增强质量"""
        enhanced_scores = {}
        
        for metric, score in idea.quality_scores.items():
            # 随机增强某些质量指标
            if random.random() < 0.3:  # 30%概率增强
                enhancement = random.uniform(0.05, 0.15)
                enhanced_scores[metric] = min(1.0, score + enhancement)
            else:
                enhanced_scores[metric] = score
        
        # 轻微调整内容以反映质量提升
        content = str(idea.content)
        if "高质量" not in content:
            enhanced_content = f"高质量的{content}"
        else:
            enhanced_content = content
        
        enhanced_idea = CreativeIdea(
            id=self._generate_id(enhanced_content),
            content=enhanced_content,
            type=idea.type,
            metadata={
                **idea.metadata,
                "evolution_type": "quality_enhancement",
                "parent_id": idea.id
            },
            quality_scores=enhanced_scores,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_ids=[idea.id],
            tags=idea.tags.copy(),
            source="evolution_engine",
            confidence=min(1.0, idea.confidence * 1.1)
        )
        
        return enhanced_idea
    
    def _simplify_complexity(self, idea: CreativeIdea) -> Optional[CreativeIdea]:
        """简化复杂性"""
        content = str(idea.content)
        
        # 移除复杂的修饰词
        complex_words = ["复杂的", "高级的", "精密的", "详细的", "全面的", "深度的"]
        simple_words = ["简单的", "基本的", "直接的", "清晰的", "核心的", "本质的"]
        
        simplified_content = content
        for complex_word in complex_words:
            if complex_word in simplified_content:
                simple_word = random.choice(simple_words)
                simplified_content = simplified_content.replace(complex_word, simple_word, 1)
        
        # 简化句子结构
        sentences = simplified_content.split('。')
        simplified_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 0:
                # 简化过长的句子
                if len(sentence.split()) > 15:
                    simplified_sentences.append(sentence[:50] + "...")
                else:
                    simplified_sentences.append(sentence)
        
        final_content = "。".join(simplified_sentences)
        
        simplified_idea = CreativeIdea(
            id=self._generate_id(final_content),
            content=final_content,
            type=idea.type,
            metadata={
                **idea.metadata,
                "evolution_type": "complexity_simplification",
                "parent_id": idea.id
            },
            quality_scores=idea.quality_scores.copy(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_ids=[idea.id],
            tags=idea.tags.copy(),
            source="evolution_engine",
            confidence=idea.confidence
        )
        
        return simplified_idea
    
    def _increase_innovation(self, idea: CreativeIdea) -> Optional[CreativeIdea]:
        """增加创新性"""
        content = str(idea.content)
        
        # 添加创新前缀
        innovation_prefixes = ["革命性的", "颠覆性的", "突破性的", "开创性的", "前瞻性的"]
        
        # 随机选择一个前缀
        prefix = random.choice(innovation_prefixes)
        innovative_content = f"{prefix}{content}"
        
        # 添加创新关键词
        innovation_keywords = ["创新", "突破", "变革", "革新", "颠覆"]
        if not any(keyword in innovative_content for keyword in innovation_keywords):
            innovative_content += f"，实现{random.choice(innovation_keywords)}"
        
        innovative_idea = CreativeIdea(
            id=self._generate_id(innovative_content),
            content=innovative_content,
            type=idea.type,
            metadata={
                **idea.metadata,
                "evolution_type": "innovation_enhancement",
                "parent_id": idea.id
            },
            quality_scores=idea.quality_scores.copy(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_ids=[idea.id],
            tags=idea.tags.copy(),
            source="evolution_engine",
            confidence=min(1.0, idea.confidence * 1.05)
        )
        
        # 提升创新性分数
        innovative_idea.quality_scores[QualityMetric.INNOVATION] = min(
            1.0, 
            idea.quality_scores.get(QualityMetric.INNOVATION, 0.5) + 0.1
        )
        
        return innovative_idea
    
    def _improve_feasibility(self, idea: CreativeIdea) -> Optional[CreativeIdea]:
        """提高可行性"""
        content = str(idea.content)
        
        # 添加可行性描述
        feasibility_phrases = [
            "可实际部署的", "易于实现的", "成本可控的", "技术成熟的", "风险较低的"
        ]
        
        phrase = random.choice(feasibility_phrases)
        feasible_content = f"{phrase}{content}"
        
        feasible_idea = CreativeIdea(
            id=self._generate_id(feasible_content),
            content=feasible_content,
            type=idea.type,
            metadata={
                **idea.metadata,
                "evolution_type": "feasibility_improvement",
                "parent_id": idea.id
            },
            quality_scores=idea.quality_scores.copy(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_ids=[idea.id],
            tags=idea.tags.copy(),
            source="evolution_engine",
            confidence=min(1.0, idea.confidence * 1.08)
        )
        
        # 提升可行性分数
        feasible_idea.quality_scores[QualityMetric.FEASIBILITY] = min(
            1.0,
            idea.quality_scores.get(QualityMetric.FEASIBILITY, 0.5) + 0.12
        )
        
        return feasible_idea
    
    def _select_best(self, population: List[CreativeIdea]) -> List[CreativeIdea]:
        """选择最优个体"""
        # 按质量分数排序
        evaluator = QualityEvaluator()
        
        scored_population = []
        for idea in population:
            overall_score = evaluator.calculate_overall_score(idea.quality_scores)
            scored_population.append((idea, overall_score))
        
        # 按分数排序
        scored_population.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前50%
        selection_count = max(1, len(scored_population) // 2)
        selected = [idea for idea, score in scored_population[:selection_count]]
        
        return selected
    
    def _generate_id(self, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(f"{content}_{time.time()}_{random.random()}".encode()).hexdigest()


class CreativeLibrary:
    """创意库管理器"""
    
    def __init__(self, db_path: str = "creative_library.db"):
        self.db_path = db_path
        self._init_database()
        self._lock = threading.Lock()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建创意表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ideas (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                metadata TEXT,
                quality_scores TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                parent_ids TEXT,
                tags TEXT,
                source TEXT,
                confidence REAL,
                description TEXT
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON ideas(type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON ideas(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON ideas(source)')
        
        conn.commit()
        conn.close()
    
    def store_idea(self, idea: CreativeIdea) -> bool:
        """存储创意"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO ideas 
                    (id, content, type, metadata, quality_scores, created_at, updated_at, 
                     parent_ids, tags, source, confidence, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    idea.id,
                    str(idea.content),
                    idea.type.value,
                    json.dumps(idea.metadata, ensure_ascii=False, default=str),
                    json.dumps(asdict(idea.quality_scores), ensure_ascii=False, default=str),
                    idea.created_at,
                    idea.updated_at,
                    json.dumps(idea.parent_ids),
                    json.dumps(list(idea.tags)),
                    idea.source,
                    idea.confidence,
                    idea.description
                ))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                print(f"存储创意失败: {e}")
                return False
    
    def retrieve_idea(self, idea_id: str) -> Optional[CreativeIdea]:
        """检索创意"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM ideas WHERE id = ?', (idea_id,))
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return self._row_to_idea(row)
                return None
                
            except Exception as e:
                print(f"检索创意失败: {e}")
                return None
    
    def search_ideas(self, 
                    query: str = None,
                    type_filter: CreativeType = None,
                    min_quality: float = 0.0,
                    tags: List[str] = None,
                    limit: int = 100) -> List[CreativeIdea]:
        """搜索创意"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                sql = "SELECT * FROM ideas WHERE 1=1"
                params = []
                
                # 文本搜索
                if query:
                    sql += " AND content LIKE ?"
                    params.append(f"%{query}%")
                
                # 类型过滤
                if type_filter:
                    sql += " AND type = ?"
                    params.append(type_filter.value)
                
                # 质量过滤
                if min_quality > 0:
                    sql += " AND confidence >= ?"
                    params.append(min_quality)
                
                # 标签过滤
                if tags:
                    for tag in tags:
                        sql += " AND tags LIKE ?"
                        params.append(f"%{tag}%")
                
                sql += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                conn.close()
                
                return [self._row_to_idea(row) for row in rows]
                
            except Exception as e:
                print(f"搜索创意失败: {e}")
                return []
    
    def get_similar_ideas(self, idea: CreativeIdea, limit: int = 10) -> List[CreativeIdea]:
        """获取相似创意"""
        # 基于内容相似度搜索
        content = str(idea.content)
        
        # 提取关键词
        keywords = self._extract_keywords(content)
        query = " ".join(keywords[:3])  # 使用前3个关键词
        
        return self.search_ideas(query=query, limit=limit + 1)
    
    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = content.split()
        
        # 过滤停用词
        stop_words = {"的", "和", "与", "或", "在", "是", "有", "了", "一个", "一种"}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 返回频次最高的词汇
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]
    
    def _row_to_idea(self, row) -> CreativeIdea:
        """数据库行转换为创意对象"""
        return CreativeIdea(
            id=row[0],
            content=row[1],
            type=CreativeType(row[2]),
            metadata=json.loads(row[3]) if row[3] else {},
            quality_scores=self._parse_quality_scores(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            parent_ids=json.loads(row[7]) if row[7] else [],
            tags=set(json.loads(row[8])) if row[8] else set(),
            source=row[9],
            confidence=row[10],
            description=row[11] or ""
        )
    
    def _parse_quality_scores(self, scores_str: str) -> Dict[QualityMetric, float]:
        """解析质量分数"""
        if not scores_str:
            return {}
        
        scores_dict = json.loads(scores_str)
        parsed_scores = {}
        
        for key, value in scores_dict.items():
            if isinstance(key, str):
                try:
                    metric = QualityMetric(key)
                    parsed_scores[metric] = float(value)
                except ValueError:
                    continue
        
        return parsed_scores
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取创意库统计信息"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 总数量
                cursor.execute('SELECT COUNT(*) FROM ideas')
                total_count = cursor.fetchone()[0]
                
                # 按类型统计
                cursor.execute('SELECT type, COUNT(*) FROM ideas GROUP BY type')
                type_stats = dict(cursor.fetchall())
                
                # 按来源统计
                cursor.execute('SELECT source, COUNT(*) FROM ideas GROUP BY source')
                source_stats = dict(cursor.fetchall())
                
                # 平均质量分数
                cursor.execute('SELECT AVG(confidence) FROM ideas')
                avg_quality = cursor.fetchone()[0] or 0.0
                
                # 最近创建的数量
                cursor.execute('SELECT COUNT(*) FROM ideas WHERE created_at > ?', 
                             (datetime.now() - timedelta(days=30),))
                recent_count = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    "total_ideas": total_count,
                    "type_distribution": type_stats,
                    "source_distribution": source_stats,
                    "average_quality": avg_quality,
                    "recent_ideas_30_days": recent_count
                }
                
            except Exception as e:
                print(f"获取统计信息失败: {e}")
                return {}


class ApplicationFramework:
    """创意应用框架"""
    
    def __init__(self):
        self.application_templates = {
            "产品设计": self._product_design_template,
            "技术方案": self._tech_solution_template,
            "商业模式": self._business_model_template,
            "研究项目": self._research_project_template,
            "创意写作": self._creative_writing_template,
            "艺术创作": self._art_creation_template
        }
    
    def apply_ideas(self, 
                   ideas: List[CreativeIdea], 
                   application_type: str,
                   context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """应用创意"""
        if application_type not in self.application_templates:
            raise ValueError(f"不支持的应用类型: {application_type}")
        
        template_func = self.application_templates[application_type]
        applications = []
        
        for idea in ideas:
            try:
                application = template_func(idea, context or {})
                if application:
                    applications.append(application)
            except Exception as e:
                print(f"应用创意失败: {e}")
        
        return applications
    
    def _product_design_template(self, idea: CreativeIdea, context: Dict[str, Any]) -> Dict[str, Any]:
        """产品设计模板"""
        return {
            "type": "产品设计",
            "idea_id": idea.id,
            "concept": str(idea.content),
            "target_user": context.get("target_user", "目标用户"),
            "key_features": self._extract_features(idea.content),
            "value_proposition": f"为用户提供{self._extract_value(idea.content)}",
            "implementation_roadmap": self._create_roadmap(idea),
            "success_metrics": ["用户满意度", "市场占有率", "收入增长"],
            "risk_assessment": self._assess_risks(idea),
            "estimated_timeline": "6-12个月",
            "resource_requirements": ["技术团队", "设计团队", "产品经理"]
        }
    
    def _tech_solution_template(self, idea: CreativeIdea, context: Dict[str, Any]) -> Dict[str, Any]:
        """技术方案模板"""
        return {
            "type": "技术方案",
            "idea_id": idea.id,
            "technical_concept": str(idea.content),
            "architecture_overview": self._design_architecture(idea),
            "technology_stack": self._suggest_tech_stack(idea),
            "implementation_phases": self._create_implementation_phases(idea),
            "performance_requirements": self._define_performance_requirements(idea),
            "scalability_plan": "支持水平和垂直扩展",
            "security_considerations": ["数据加密", "访问控制", "审计日志"],
            "testing_strategy": ["单元测试", "集成测试", "性能测试"],
            "deployment_plan": "容器化部署，支持自动扩缩容"
        }
    
    def _business_model_template(self, idea: CreativeIdea, context: Dict[str, Any]) -> Dict[str, Any]:
        """商业模式模板"""
        return {
            "type": "商业模式",
            "idea_id": idea.id,
            "business_concept": str(idea.content),
            "value_proposition": self._define_value_proposition(idea),
            "target_market": context.get("target_market", "目标市场"),
            "revenue_model": self._design_revenue_model(idea),
            "cost_structure": ["研发成本", "运营成本", "营销成本"],
            "competitive_advantage": self._identify_competitive_advantages(idea),
            "go_to_market_strategy": self._create_gtm_strategy(idea),
            "financial_projections": {
                "year_1": "盈亏平衡",
                "year_2": "20%增长",
                "year_3": "50%增长"
            },
            "funding_requirements": "种子轮100万，A轮500万"
        }
    
    def _research_project_template(self, idea: CreativeIdea, context: Dict[str, Any]) -> Dict[str, Any]:
        """研究项目模板"""
        return {
            "type": "研究项目",
            "idea_id": idea.id,
            "research_question": str(idea.content),
            "objectives": self._define_research_objectives(idea),
            "methodology": self._design_methodology(idea),
            "expected_outcomes": ["理论贡献", "实践应用", "社会影响"],
            "timeline": "18-24个月",
            "resources_needed": ["研究团队", "实验设备", "数据分析工具"],
            "milestones": self._create_research_milestones(idea),
            "risk_mitigation": ["技术风险", "时间风险", "资源风险"],
            "dissemination_plan": ["学术论文", "会议报告", "开源代码"]
        }
    
    def _creative_writing_template(self, idea: CreativeIdea, context: Dict[str, Any]) -> Dict[str, Any]:
        """创意写作模板"""
        return {
            "type": "创意写作",
            "idea_id": idea.id,
            "writing_concept": str(idea.content),
            "genre": context.get("genre", "科幻"),
            "target_audience": context.get("target_audience", "成人读者"),
            "plot_outline": self._create_plot_outline(idea),
            "character_development": self._develop_characters(idea),
            "themes": self._identify_themes(idea),
            "writing_schedule": "每日1000字，预计3个月完成",
            "editing_process": ["初稿", "修改", "校对", "定稿"],
            "publication_strategy": ["电子书", "纸质书", "有声书"]
        }
    
    def _art_creation_template(self, idea: CreativeIdea, context: Dict[str, Any]) -> Dict[str, Any]:
        """艺术创作模板"""
        return {
            "type": "艺术创作",
            "idea_id": idea.id,
            "artistic_concept": str(idea.content),
            "medium": context.get("medium", "数字艺术"),
            "style": context.get("style", "现代主义"),
            "color_palette": self._suggest_color_palette(idea),
            "composition": self._design_composition(idea),
            "technical_approach": self._define_technical_approach(idea),
            "creation_timeline": "2-4周",
            "exhibition_plan": ["在线展览", "画廊展示", "艺术节"],
            "audience_engagement": ["互动体验", "艺术家对话", "作品解读"]
        }
    
    def _extract_features(self, content: str) -> List[str]:
        """提取功能特性"""
        features = []
        feature_keywords = ["功能", "特性", "能力", "特点", "优势"]
        
        for keyword in feature_keywords:
            if keyword in content:
                features.append(f"支持{keyword}")
        
        return features if features else ["核心功能", "辅助功能"]
    
    def _extract_value(self, content: str) -> str:
        """提取价值主张"""
        value_keywords = ["效率", "便利", "创新", "优化", "改进"]
        
        for keyword in value_keywords:
            if keyword in content:
                return f"{keyword}价值"
        
        return "创新价值"
    
    def _create_roadmap(self, idea: CreativeIdea) -> List[Dict[str, str]]:
        """创建产品路线图"""
        return [
            {"phase": "MVP", "duration": "3个月", "description": "核心功能实现"},
            {"phase": "版本1.0", "duration": "6个月", "description": "完整功能发布"},
            {"phase": "版本2.0", "duration": "12个月", "description": "高级功能扩展"}
        ]
    
    def _assess_risks(self, idea: CreativeIdea) -> List[Dict[str, str]]:
        """评估风险"""
        return [
            {"risk": "技术风险", "level": "中等", "mitigation": "技术预研和验证"},
            {"risk": "市场风险", "level": "低", "mitigation": "用户调研和验证"},
            {"risk": "竞争风险", "level": "中等", "mitigation": "差异化定位"}
        ]
    
    def _design_architecture(self, idea: CreativeIdea) -> str:
        """设计架构"""
        return "微服务架构，支持高并发和可扩展性"
    
    def _suggest_tech_stack(self, idea: CreativeIdea) -> List[str]:
        """建议技术栈"""
        return ["Python", "React", "PostgreSQL", "Redis", "Docker", "Kubernetes"]
    
    def _create_implementation_phases(self, idea: CreativeIdea) -> List[Dict[str, str]]:
        """创建实施阶段"""
        return [
            {"phase": "架构设计", "duration": "2周", "deliverable": "技术架构文档"},
            {"phase": "核心开发", "duration": "8周", "deliverable": "核心功能模块"},
            {"phase": "集成测试", "duration": "2周", "deliverable": "测试报告"},
            {"phase": "部署上线", "duration": "1周", "deliverable": "生产环境"}
        ]
    
    def _define_performance_requirements(self, idea: CreativeIdea) -> Dict[str, str]:
        """定义性能要求"""
        return {
            "响应时间": "< 200ms",
            "并发用户": "10000+",
            "可用性": "99.9%",
            "数据一致性": "强一致性"
        }
    
    def _define_value_proposition(self, idea: CreativeIdea) -> str:
        """定义价值主张"""
        return f"通过{str(idea.content)}为客户创造独特价值"
    
    def _design_revenue_model(self, idea: CreativeIdea) -> str:
        """设计收入模型"""
        return "订阅制 + 按使用量计费"
    
    def _identify_competitive_advantages(self, idea: CreativeIdea) -> List[str]:
        """识别竞争优势"""
        return ["技术创新", "用户体验", "成本优势", "生态系统"]
    
    def _create_gtm_strategy(self, idea: CreativeIdea) -> Dict[str, str]:
        """创建市场进入策略"""
        return {
            "目标客户": "中小企业",
            "销售渠道": "直销 + 合作伙伴",
            "营销策略": "内容营销 + 行业展会",
            "定价策略": "价值定价"
        }
    
    def _define_research_objectives(self, idea: CreativeIdea) -> List[str]:
        """定义研究目标"""
        return ["理论创新", "方法改进", "应用验证"]
    
    def _design_methodology(self, idea: CreativeIdea) -> str:
        """设计研究方法"""
        return "定量与定性相结合的研究方法"
    
    def _create_research_milestones(self, idea: CreativeIdea) -> List[Dict[str, str]]:
        """创建研究里程碑"""
        return [
            {"milestone": "文献综述", "timeline": "3个月"},
            {"milestone": "实验设计", "timeline": "6个月"},
            {"milestone": "数据收集", "timeline": "12个月"},
            {"milestone": "论文撰写", "timeline": "18个月"}
        ]
    
    def _create_plot_outline(self, idea: CreativeIdea) -> str:
        """创建情节大纲"""
        return f"基于{str(idea.content)}构建完整故事情节"
    
    def _develop_characters(self, idea: CreativeIdea) -> Dict[str, str]:
        """开发角色"""
        return {
            "主角": "具有创新精神的角色",
            "配角": "支持主角发展的角色",
            "反派": "制造冲突的角色"
        }
    
    def _identify_themes(self, idea: CreativeIdea) -> List[str]:
        """识别主题"""
        return ["创新与变革", "人性与科技", "梦想与现实"]
    
    def _suggest_color_palette(self, idea: CreativeIdea) -> List[str]:
        """建议色彩方案"""
        return ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    
    def _design_composition(self, idea: CreativeIdea) -> str:
        """设计构图"""
        return "遵循三分法则，突出主题元素"
    
    def _define_technical_approach(self, idea: CreativeIdea) -> str:
        """定义技术方法"""
        return "结合传统技法与现代数字工具"


class CreativeEngine:
    """E2 创意引擎主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化创意引擎"""
        self.config = config or {}
        
        # 初始化各个组件
        self.random_generator = RandomCreativeGenerator()
        self.associative_generator = AssociativeCreativeGenerator()
        self.analogical_generator = AnalogicalCreativeGenerator()
        self.quality_evaluator = QualityEvaluator()
        self.idea_fusion = IdeaFusion()
        self.evolution_engine = EvolutionEngine()
        self.creative_library = CreativeLibrary(
            self.config.get("db_path", "creative_engine.db")
        )
        self.application_framework = ApplicationFramework()
        
        # 创意生成器列表
        self.generators = {
            "random": self.random_generator,
            "associative": self.associative_generator,
            "analogical": self.analogical_generator
        }
        
        print("E2 创意引擎初始化完成")
    
    def generate_ideas(self, 
                      seed: Any, 
                      context: Dict[str, Any] = None,
                      generator_types: List[str] = None,
                      count: int = 10) -> List[CreativeIdea]:
        """生成创意"""
        if generator_types is None:
            generator_types = ["random", "associative", "analogical"]
        
        if context is None:
            context = {}
        
        all_ideas = []
        
        print(f"开始生成创意，种子: {seed}")
        
        for gen_type in generator_types:
            if gen_type not in self.generators:
                print(f"未知的生成器类型: {gen_type}")
                continue
            
            generator = self.generators[gen_type]
            try:
                ideas = generator.generate(seed, context)
                all_ideas.extend(ideas)
                print(f"生成器 {gen_type} 生成了 {len(ideas)} 个创意")
            except Exception as e:
                print(f"生成器 {gen_type} 失败: {e}")
        
        # 限制数量
        if len(all_ideas) > count:
            # 按质量分数排序
            scored_ideas = []
            for idea in all_ideas:
                overall_score = self.quality_evaluator.calculate_overall_score(idea.quality_scores)
                scored_ideas.append((idea, overall_score))
            
            scored_ideas.sort(key=lambda x: x[1], reverse=True)
            all_ideas = [idea for idea, score in scored_ideas[:count]]
        
        print(f"总共生成 {len(all_ideas)} 个创意")
        return all_ideas
    
    def evaluate_ideas(self, ideas: List[CreativeIdea]) -> List[CreativeIdea]:
        """评估创意质量"""
        evaluated_ideas = []
        
        print("开始评估创意质量...")
        
        for i, idea in enumerate(ideas):
            try:
                # 重新评估质量
                new_scores = self.quality_evaluator.evaluate_idea(idea)
                idea.quality_scores = new_scores
                
                # 计算综合分数
                overall_score = self.quality_evaluator.calculate_overall_score(new_scores)
                
                evaluated_ideas.append(idea)
                
                if (i + 1) % 5 == 0:
                    print(f"已评估 {i + 1}/{len(ideas)} 个创意")
                    
            except Exception as e:
                print(f"评估创意失败: {e}")
        
        print(f"质量评估完成，平均分数: {sum(self.quality_evaluator.calculate_overall_score(idea.quality_scores) for idea in evaluated_ideas) / len(evaluated_ideas):.3f}")
        
        return evaluated_ideas
    
    def fuse_ideas(self, ideas: List[CreativeIdea]) -> List[CreativeIdea]:
        """融合创意"""
        if len(ideas) < 2:
            print("需要至少2个创意进行融合")
            return []
        
        print(f"开始融合 {len(ideas)} 个创意...")
        
        try:
            fused_ideas = self.idea_fusion.fuse_ideas(ideas)
            print(f"融合生成 {len(fused_ideas)} 个新创意")
            return fused_ideas
        except Exception as e:
            print(f"创意融合失败: {e}")
            return []
    
    def evolve_ideas(self, ideas: List[CreativeIdea], generations: int = 3) -> List[CreativeIdea]:
        """进化创意"""
        if not ideas:
            print("没有创意可以进行进化")
            return []
        
        print(f"开始进化 {len(ideas)} 个创意，进化 {generations} 代...")
        
        try:
            evolved_ideas = self.evolution_engine.evolve_ideas(ideas, generations)
            print(f"进化完成，生成 {len(evolved_ideas)} 个进化后的创意")
            return evolved_ideas
        except Exception as e:
            print(f"创意进化失败: {e}")
            return []
    
    def store_ideas(self, ideas: List[CreativeIdea]) -> int:
        """存储创意到库中"""
        if not ideas:
            return 0
        
        print(f"开始存储 {len(ideas)} 个创意到库中...")
        
        stored_count = 0
        for idea in ideas:
            try:
                if self.creative_library.store_idea(idea):
                    stored_count += 1
                else:
                    print(f"存储创意失败: {idea.id}")
            except Exception as e:
                print(f"存储创意异常: {e}")
        
        print(f"成功存储 {stored_count}/{len(ideas)} 个创意")
        return stored_count
    
    def search_ideas(self, 
                    query: str = None,
                    type_filter: CreativeType = None,
                    min_quality: float = 0.0,
                    tags: List[str] = None,
                    limit: int = 100) -> List[CreativeIdea]:
        """搜索创意库"""
        print(f"搜索创意: query={query}, type={type_filter}, min_quality={min_quality}")
        
        try:
            ideas = self.creative_library.search_ideas(
                query=query,
                type_filter=type_filter,
                min_quality=min_quality,
                tags=tags,
                limit=limit
            )
            print(f"搜索到 {len(ideas)} 个创意")
            return ideas
        except Exception as e:
            print(f"搜索创意失败: {e}")
            return []
    
    def apply_ideas(self, 
                   ideas: List[CreativeIdea], 
                   application_type: str,
                   context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """应用创意"""
        if not ideas:
            print("没有创意可以应用")
            return []
        
        print(f"开始应用 {len(ideas)} 个创意到 {application_type}")
        
        try:
            applications = self.application_framework.apply_ideas(
                ideas, application_type, context
            )
            print(f"生成 {len(applications)} 个应用方案")
            return applications
        except Exception as e:
            print(f"应用创意失败: {e}")
            return []
    
    def get_library_statistics(self) -> Dict[str, Any]:
        """获取创意库统计信息"""
        try:
            stats = self.creative_library.get_statistics()
            print("获取创意库统计信息成功")
            return stats
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {}
    
    def full_creative_pipeline(self, 
                              seed: Any,
                              context: Dict[str, Any] = None,
                              target_count: int = 20,
                              application_type: str = None) -> Dict[str, Any]:
        """完整的创意流程"""
        print("=" * 50)
        print("开始完整的创意生成流程")
        print("=" * 50)
        
        results = {
            "seed": seed,
            "context": context or {},
            "generated_ideas": [],
            "evaluated_ideas": [],
            "fused_ideas": [],
            "evolved_ideas": [],
            "applications": [],
            "statistics": {}
        }
        
        try:
            # 1. 生成创意
            print("\n步骤1: 生成创意")
            generated_ideas = self.generate_ideas(seed, context, count=target_count)
            results["generated_ideas"] = generated_ideas
            
            # 2. 评估创意
            print("\n步骤2: 评估创意质量")
            evaluated_ideas = self.evaluate_ideas(generated_ideas)
            results["evaluated_ideas"] = evaluated_ideas
            
            # 3. 融合创意
            print("\n步骤3: 融合创意")
            if len(evaluated_ideas) >= 2:
                fused_ideas = self.fuse_ideas(evaluated_ideas[:10])  # 限制数量避免爆炸
                results["fused_ideas"] = fused_ideas
                all_ideas_for_evolution = evaluated_ideas + fused_ideas
            else:
                all_ideas_for_evolution = evaluated_ideas
            
            # 4. 进化创意
            print("\n步骤4: 进化创意")
            if all_ideas_for_evolution:
                evolved_ideas = self.evolve_ideas(all_ideas_for_evolution, generations=2)
                results["evolved_ideas"] = evolved_ideas
            
            # 5. 存储创意
            print("\n步骤5: 存储创意")
            all_final_ideas = evaluated_ideas + results.get("fused_ideas", []) + results.get("evolved_ideas", [])
            stored_count = self.store_ideas(all_final_ideas)
            
            # 6. 应用创意
            if application_type and all_final_ideas:
                print(f"\n步骤6: 应用创意到 {application_type}")
                applications = self.apply_ideas(all_final_ideas[:5], application_type, context)  # 限制数量
                results["applications"] = applications
            
            # 7. 获取统计信息
            print("\n步骤7: 获取统计信息")
            statistics = self.get_library_statistics()
            results["statistics"] = statistics
            
            print("\n" + "=" * 50)
            print("创意流程完成")
            print(f"生成创意: {len(generated_ideas)}")
            print(f"评估创意: {len(evaluated_ideas)}")
            print(f"融合创意: {len(results.get('fused_ideas', []))}")
            print(f"进化创意: {len(results.get('evolved_ideas', []))}")
            print(f"存储创意: {stored_count}")
            print(f"应用方案: {len(results.get('applications', []))}")
            print("=" * 50)
            
            return results
            
        except Exception as e:
            print(f"创意流程执行失败: {e}")
            return results


# 使用示例和测试函数
def demo_creative_engine():
    """演示创意引擎的使用"""
    print("E2 创意引擎演示")
    print("=" * 60)
    
    # 初始化创意引擎
    engine = CreativeEngine()
    
    # 设置种子和上下文
    seed = "人工智能"
    context = {
        "domain": "技术",
        "target_audience": "开发者",
        "constraints": ["可实现性", "成本控制"]
    }
    
    # 执行完整的创意流程
    results = engine.full_creative_pipeline(
        seed=seed,
        context=context,
        target_count=15,
        application_type="技术方案"
    )
    
    # 展示结果
    print("\n生成的部分创意:")
    for i, idea in enumerate(results["generated_ideas"][:5]):
        score = engine.quality_evaluator.calculate_overall_score(idea.quality_scores)
        print(f"{i+1}. {idea.content} (质量分数: {score:.3f})")
    
    if results["applications"]:
        print("\n生成的应用方案示例:")
        for i, app in enumerate(results["applications"][:2]):
            print(f"{i+1}. {app['type']}: {app.get('technical_concept', app.get('concept', ''))}")
    
    print("\n创意库统计信息:")
    stats = results["statistics"]
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return results


if __name__ == "__main__":
    # 运行演示
    demo_results = demo_creative_engine()