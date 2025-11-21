#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5 概念创新器
===============

概念创新器是智能系统中的核心组件，负责：
1. 新概念生成和发明
2. 概念组合和融合
3. 概念进化和变异
4. 概念有效性验证
5. 概念应用和实现
6. 概念库管理和更新
7. 概念影响评估

作者：智能系统
版本：1.0
创建时间：2025-11-05
"""

import json
import random
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import math
import statistics
from collections import defaultdict, Counter
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptType(Enum):
    """概念类型枚举"""
    BASIC = "基础概念"
    COMPOSITE = "复合概念"
    ABSTRACT = "抽象概念"
    CONCRETE = "具体概念"
    TEMPORAL = "时间概念"
    SPATIAL = "空间概念"
    CAUSAL = "因果概念"
    RELATIONAL = "关系概念"


class ValidationStatus(Enum):
    """验证状态枚举"""
    UNVALIDATED = "未验证"
    VALIDATING = "验证中"
    VALID = "有效"
    INVALID = "无效"
    PARTIAL = "部分有效"


class EvolutionType(Enum):
    """进化类型枚举"""
    MUTATION = "变异"
    CROSSOVER = "交叉"
    SELECTION = "选择"
    ADAPTATION = "适应"
    INNOVATION = "创新"


@dataclass
class ConceptProperty:
    """概念属性"""
    name: str
    value: Any
    type: str
    weight: float = 1.0
    confidence: float = 1.0


@dataclass
class ConceptRelation:
    """概念关系"""
    target_concept_id: str
    relation_type: str
    strength: float
    bidirectional: bool = False


@dataclass
class ConceptValidation:
    """概念验证结果"""
    concept_id: str
    status: ValidationStatus
    score: float
    criteria: Dict[str, float]
    timestamp: datetime.datetime
    validator: str
    notes: str = ""


@dataclass
class ConceptEvolution:
    """概念进化记录"""
    concept_id: str
    parent_id: str
    evolution_type: EvolutionType
    changes: Dict[str, Any]
    fitness_score: float
    timestamp: datetime.datetime


@dataclass
class ConceptApplication:
    """概念应用记录"""
    concept_id: str
    application_domain: str
    implementation: Dict[str, Any]
    success_rate: float
    usage_count: int
    last_used: datetime.datetime


@dataclass
class ConceptImpact:
    """概念影响评估"""
    concept_id: str
    impact_score: float
    influence_network: Dict[str, float]
    innovation_potential: float
    practical_value: float
    theoretical_value: float
    timestamp: datetime.datetime


class Concept:
    """概念类"""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 concept_type: ConceptType,
                 properties: List[ConceptProperty] = None,
                 relations: List[ConceptRelation] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.concept_type = concept_type
        self.properties = properties or []
        self.relations = relations or []
        self.created_time = datetime.datetime.now()
        self.modified_time = datetime.datetime.now()
        self.version = 1
        self.tags = set()
        self.confidence_score = 0.5
        self.usage_count = 0
        self.parent_concepts = []
        self.child_concepts = []
        
    def add_property(self, property_obj: ConceptProperty):
        """添加属性"""
        # 检查属性是否已存在
        for prop in self.properties:
            if prop.name == property_obj.name:
                prop.value = property_obj.value
                prop.confidence = property_obj.confidence
                return
        self.properties.append(property_obj)
        self._update_modified_time()
        
    def add_relation(self, relation: ConceptRelation):
        """添加关系"""
        self.relations.append(relation)
        self._update_modified_time()
        
    def get_property(self, name: str) -> Optional[ConceptProperty]:
        """获取属性"""
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None
        
    def get_relations_by_type(self, relation_type: str) -> List[ConceptRelation]:
        """根据类型获取关系"""
        return [rel for rel in self.relations if rel.relation_type == relation_type]
        
    def _update_modified_time(self):
        """更新修改时间"""
        self.modified_time = datetime.datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'concept_type': self.concept_type.value,
            'properties': [asdict(prop) for prop in self.properties],
            'relations': [asdict(rel) for rel in self.relations],
            'created_time': self.created_time.isoformat(),
            'modified_time': self.modified_time.isoformat(),
            'version': self.version,
            'tags': list(self.tags),
            'confidence_score': self.confidence_score,
            'usage_count': self.usage_count,
            'parent_concepts': self.parent_concepts,
            'child_concepts': self.child_concepts
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Concept':
        """从字典创建概念"""
        concept = cls(
            name=data['name'],
            description=data['description'],
            concept_type=ConceptType(data['concept_type'])
        )
        
        concept.id = data['id']
        concept.created_time = datetime.datetime.fromisoformat(data['created_time'])
        concept.modified_time = datetime.datetime.fromisoformat(data['modified_time'])
        concept.version = data['version']
        concept.tags = set(data['tags'])
        concept.confidence_score = data['confidence_score']
        concept.usage_count = data['usage_count']
        concept.parent_concepts = data['parent_concepts']
        concept.child_concepts = data['child_concepts']
        
        # 重建属性
        for prop_data in data['properties']:
            prop = ConceptProperty(**prop_data)
            concept.properties.append(prop)
            
        # 重建关系
        for rel_data in data['relations']:
            rel = ConceptRelation(**rel_data)
            concept.relations.append(rel)
            
        return concept


class ConceptInnovator:
    """概念创新器主类"""
    
    def __init__(self, 
                 concept_library_path: str = "concept_library.json",
                 max_concepts: int = 10000):
        self.concept_library_path = concept_library_path
        self.max_concepts = max_concepts
        self.concepts: Dict[str, Concept] = {}
        self.validation_records: Dict[str, ConceptValidation] = {}
        self.evolution_records: List[ConceptEvolution] = []
        self.application_records: Dict[str, ConceptApplication] = {}
        self.impact_assessments: Dict[str, ConceptImpact] = {}
        
        # 算法参数
        self.mutation_rate = 0.1
        self.crossover_rate = 0.3
        self.selection_pressure = 0.5
        self.validation_threshold = 0.7
        
        # 初始化概念库
        self._initialize_concept_library()
        
    def _initialize_concept_library(self):
        """初始化概念库"""
        # 加载现有概念库
        try:
            with open(self.concept_library_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for concept_data in data.get('concepts', []):
                    concept = Concept.from_dict(concept_data)
                    self.concepts[concept.id] = concept
            logger.info(f"已加载 {len(self.concepts)} 个概念")
        except FileNotFoundError:
            logger.info("概念库文件不存在，创建新的概念库")
            self._create_basic_concepts()
            
    def _create_basic_concepts(self):
        """创建基础概念"""
        basic_concepts = [
            ("物质", "构成宇宙的基本实体", ConceptType.BASIC),
            ("能量", "做功的能力", ConceptType.BASIC),
            ("信息", "有意义的信号或数据", ConceptType.BASIC),
            ("空间", "物质存在的三维扩展", ConceptType.SPATIAL),
            ("时间", "事件发生的顺序", ConceptType.TEMPORAL),
            ("因果", "原因与结果的联系", ConceptType.CAUSAL),
            ("关系", "事物之间的联系", ConceptType.RELATIONAL),
            ("系统", "相互作用的组件集合", ConceptType.ABSTRACT),
            ("过程", "状态变化的序列", ConceptType.TEMPORAL),
            ("结构", "组成部分的排列方式", ConceptType.ABSTRACT)
        ]
        
        for name, desc, concept_type in basic_concepts:
            concept = Concept(name, desc, concept_type)
            self.concepts[concept.id] = concept
            
        logger.info(f"已创建 {len(basic_concepts)} 个基础概念")
        
    def save_concept_library(self):
        """保存概念库"""
        def serialize_enum(obj):
            """序列化枚举对象"""
            if hasattr(obj, 'value'):
                return obj.value
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return str(obj)
        
        data = {
            'concepts': [concept.to_dict() for concept in self.concepts.values()],
            'validation_records': [self._serialize_validation_record(record) for record in self.validation_records.values()],
            'evolution_records': [self._serialize_evolution_record(record) for record in self.evolution_records],
            'application_records': [self._serialize_application_record(record) for record in self.application_records.values()],
            'impact_assessments': [self._serialize_impact_assessment(record) for record in self.impact_assessments.values()],
            'metadata': {
                'total_concepts': len(self.concepts),
                'last_updated': datetime.datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        
        with open(self.concept_library_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=serialize_enum)
            
        logger.info("概念库已保存")
        
    def _serialize_validation_record(self, record: ConceptValidation) -> Dict[str, Any]:
        """序列化验证记录"""
        return {
            'concept_id': record.concept_id,
            'status': record.status.value,
            'score': record.score,
            'criteria': record.criteria,
            'timestamp': record.timestamp.isoformat(),
            'validator': record.validator,
            'notes': record.notes
        }
        
    def _serialize_evolution_record(self, record: ConceptEvolution) -> Dict[str, Any]:
        """序列化进化记录"""
        return {
            'concept_id': record.concept_id,
            'parent_id': record.parent_id,
            'evolution_type': record.evolution_type.value,
            'changes': record.changes,
            'fitness_score': record.fitness_score,
            'timestamp': record.timestamp.isoformat()
        }
        
    def _serialize_application_record(self, record: ConceptApplication) -> Dict[str, Any]:
        """序列化应用记录"""
        return {
            'concept_id': record.concept_id,
            'application_domain': record.application_domain,
            'implementation': record.implementation,
            'success_rate': record.success_rate,
            'usage_count': record.usage_count,
            'last_used': record.last_used.isoformat()
        }
        
    def _serialize_impact_assessment(self, record: ConceptImpact) -> Dict[str, Any]:
        """序列化影响评估"""
        return {
            'concept_id': record.concept_id,
            'impact_score': record.impact_score,
            'influence_network': record.influence_network,
            'innovation_potential': record.innovation_potential,
            'practical_value': record.practical_value,
            'theoretical_value': record.theoretical_value,
            'timestamp': record.timestamp.isoformat()
        }
        
    # ==================== 1. 新概念生成和发明 ====================
    
    def generate_new_concept(self, 
                           context: Dict[str, Any] = None,
                           target_domain: str = "general") -> Concept:
        """生成新概念"""
        context = context or {}
        
        # 分析上下文
        domain_knowledge = self._extract_domain_knowledge(target_domain)
        existing_concepts = self._find_relevant_concepts(context)
        
        # 生成新概念
        concept_name = self._generate_concept_name(context, existing_concepts)
        concept_description = self._generate_concept_description(context, domain_knowledge)
        concept_type = self._determine_concept_type(context, existing_concepts)
        
        # 创建概念
        new_concept = Concept(concept_name, concept_description, concept_type)
        
        # 添加属性
        self._generate_concept_properties(new_concept, context, existing_concepts)
        
        # 建立关系
        self._establish_concept_relations(new_concept, existing_concepts)
        
        # 存储概念
        self.concepts[new_concept.id] = new_concept
        
        logger.info(f"生成了新概念: {new_concept.name}")
        return new_concept
        
    def _extract_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """提取领域知识"""
        # 简化的领域知识提取
        domain_knowledge = {
            "physics": {
                "fundamental_concepts": ["force", "mass", "acceleration", "energy", "momentum"],
                "laws": ["Newton's laws", "conservation laws", "thermodynamics"],
                "relationships": ["causal", "mathematical", "temporal"]
            },
            "biology": {
                "fundamental_concepts": ["cell", "gene", "evolution", "ecosystem", "metabolism"],
                "laws": ["natural selection", "genetic inheritance", "homeostasis"],
                "relationships": ["hierarchical", "functional", "evolutionary"]
            },
            "computer_science": {
                "fundamental_concepts": ["algorithm", "data structure", "computation", "information"],
                "laws": ["complexity theory", "computability", "information theory"],
                "relationships": ["logical", "computational", "algorithmic"]
            }
        }
        
        return domain_knowledge.get(domain, {})
        
    def _find_relevant_concepts(self, context: Dict[str, Any]) -> List[Concept]:
        """查找相关概念"""
        relevant_concepts = []
        context_keywords = set()
        
        # 提取上下文关键词
        if 'keywords' in context:
            context_keywords.update(context['keywords'])
        if 'description' in context:
            # 简单的关键词提取
            words = context['description'].lower().split()
            context_keywords.update(words)
            
        # 查找匹配的概念
        for concept in self.concepts.values():
            if self._is_concept_relevant(concept, context_keywords):
                relevant_concepts.append(concept)
                
        return relevant_concepts
        
    def _is_concept_relevant(self, concept: Concept, keywords: Set[str]) -> bool:
        """判断概念是否相关"""
        # 检查概念名称和描述
        concept_text = f"{concept.name} {concept.description}".lower()
        
        # 检查属性
        for prop in concept.properties:
            concept_text += f" {prop.name} {prop.value}"
            
        # 检查关键词匹配
        for keyword in keywords:
            if keyword in concept_text:
                return True
                
        return False
        
    def _generate_concept_name(self, context: Dict[str, Any], existing_concepts: List[Concept]) -> str:
        """生成概念名称"""
        # 基于上下文和现有概念生成名称
        base_words = []
        
        if 'keywords' in context:
            base_words.extend(context['keywords'][:3])  # 取前3个关键词
            
        # 如果没有足够的关键词，从现有概念中提取
        if len(base_words) < 2:
            for concept in existing_concepts[:2]:
                base_words.append(concept.name)
                
        # 生成组合名称
        if len(base_words) >= 2:
            return f"{base_words[0]}-{base_words[1]}"
        elif len(base_words) == 1:
            return f"新型{base_words[0]}"
        else:
            return f"创新概念_{random.randint(1000, 9999)}"
            
    def _generate_concept_description(self, context: Dict[str, Any], domain_knowledge: Dict[str, Any]) -> str:
        """生成概念描述"""
        description_parts = []
        
        if 'description' in context:
            description_parts.append(context['description'])
            
        # 基于领域知识生成描述
        if 'fundamental_concepts' in domain_knowledge:
            concepts = domain_knowledge['fundamental_concepts']
            description_parts.append(f"涉及{', '.join(concepts[:3])}等核心概念")
            
        # 基于现有概念生成描述
        if context.get('based_on_existing'):
            description_parts.append("基于现有概念的创新组合")
            
        return " ".join(description_parts) if description_parts else "新生成的概念"
        
    def _determine_concept_type(self, context: Dict[str, Any], existing_concepts: List[Concept]) -> ConceptType:
        """确定概念类型"""
        if 'type_hint' in context:
            type_hint = context['type_hint'].lower()
            type_mapping = {
                'basic': ConceptType.BASIC,
                'composite': ConceptType.COMPOSITE,
                'abstract': ConceptType.ABSTRACT,
                'concrete': ConceptType.CONCRETE,
                'temporal': ConceptType.TEMPORAL,
                'spatial': ConceptType.SPATIAL,
                'causal': ConceptType.CAUSAL,
                'relational': ConceptType.RELATIONAL
            }
            return type_mapping.get(type_hint, ConceptType.ABSTRACT)
            
        # 基于上下文判断
        if 'temporal' in str(context).lower():
            return ConceptType.TEMPORAL
        elif 'spatial' in str(context).lower():
            return ConceptType.SPATIAL
        elif 'causal' in str(context).lower():
            return ConceptType.CAUSAL
        elif len(existing_concepts) > 1:
            return ConceptType.COMPOSITE
        else:
            return ConceptType.ABSTRACT
            
    def _generate_concept_properties(self, concept: Concept, context: Dict[str, Any], existing_concepts: List[Concept]):
        """生成概念属性"""
        # 基于上下文生成属性
        for key, value in context.items():
            if key not in ['keywords', 'description', 'type_hint', 'based_on_existing']:
                prop = ConceptProperty(
                    name=key,
                    value=value,
                    type=type(value).__name__,
                    weight=random.uniform(0.5, 1.0),
                    confidence=random.uniform(0.7, 1.0)
                )
                concept.add_property(prop)
                
        # 基于现有概念继承属性
        for existing_concept in existing_concepts[:2]:
            for prop in existing_concept.properties[:2]:
                if concept.get_property(prop.name) is None:
                    new_prop = ConceptProperty(
                        name=f"继承_{prop.name}",
                        value=prop.value,
                        type=prop.type,
                        weight=prop.weight * 0.8,  # 继承的属性权重降低
                        confidence=prop.confidence * 0.9
                    )
                    concept.add_property(new_prop)
                    
    def _establish_concept_relations(self, concept: Concept, existing_concepts: List[Concept]):
        """建立概念关系"""
        relation_types = ["包含", "被包含", "相关", "对立", "因果", "相似"]
        
        for existing_concept in existing_concepts[:5]:  # 最多关联5个概念
            relation_type = random.choice(relation_types)
            relation = ConceptRelation(
                target_concept_id=existing_concept.id,
                relation_type=relation_type,
                strength=random.uniform(0.3, 1.0),
                bidirectional=random.choice([True, False])
            )
            concept.add_relation(relation)
            
    # ==================== 2. 概念组合和融合 ====================
    
    def combine_concepts(self, concept_ids: List[str], combination_strategy: str = "merge") -> Concept:
        """组合概念"""
        if len(concept_ids) < 2:
            raise ValueError("至少需要2个概念进行组合")
            
        # 获取要组合的概念
        source_concepts = [self.concepts[cid] for cid in concept_ids if cid in self.concepts]
        if not source_concepts:
            raise ValueError("未找到有效的概念")
            
        # 根据策略组合概念
        if combination_strategy == "merge":
            return self._merge_concepts(source_concepts)
        elif combination_strategy == "blend":
            return self._blend_concepts(source_concepts)
        elif combination_strategy == "hybrid":
            return self._hybridize_concepts(source_concepts)
        else:
            raise ValueError(f"未知的组合策略: {combination_strategy}")
            
    def _merge_concepts(self, concepts: List[Concept]) -> Concept:
        """合并概念"""
        # 选择主要概念
        primary_concept = max(concepts, key=lambda c: c.confidence_score)
        
        # 创建新概念
        merged_name = f"{primary_concept.name}融合体"
        merged_desc = f"融合了{len(concepts)}个概念的创新概念"
        merged_concept = Concept(merged_name, merged_desc, ConceptType.COMPOSITE)
        
        # 合并属性
        all_properties = []
        for concept in concepts:
            all_properties.extend(concept.properties)
            
        # 去重并合并属性
        property_names = set()
        for prop in all_properties:
            if prop.name not in property_names:
                merged_concept.add_property(prop)
                property_names.add(prop.name)
                
        # 建立关系
        for concept in concepts:
            relation = ConceptRelation(
                target_concept_id=concept.id,
                relation_type="融合来源",
                strength=1.0,
                bidirectional=True
            )
            merged_concept.add_relation(relation)
            
        # 设置父概念
        merged_concept.parent_concepts = [c.id for c in concepts]
        
        # 存储新概念
        self.concepts[merged_concept.id] = merged_concept
        
        logger.info(f"成功合并 {len(concepts)} 个概念为: {merged_concept.name}")
        return merged_concept
        
    def _blend_concepts(self, concepts: List[Concept]) -> Concept:
        """混合概念"""
        # 创建混合概念
        blended_name = f"{concepts[0].name}×{concepts[1].name}"
        blended_desc = f"概念混合体，融合了多个概念的特性"
        blended_concept = Concept(blended_name, blended_desc, ConceptType.COMPOSITE)
        
        # 混合属性值
        for prop_name in set(prop.name for concept in concepts for prop in concept.properties):
            # 获取所有相关属性的值
            values = []
            for concept in concepts:
                prop = concept.get_property(prop_name)
                if prop:
                    values.append(prop.value)
                    
            if values:
                # 混合属性值（简单平均）
                if all(isinstance(v, (int, float)) for v in values):
                    mixed_value = sum(values) / len(values)
                else:
                    mixed_value = f"混合_{prop_name}"
                    
                blended_prop = ConceptProperty(
                    name=prop_name,
                    value=mixed_value,
                    type=type(mixed_value).__name__,
                    weight=0.8,
                    confidence=0.7
                )
                blended_concept.add_property(blended_prop)
                
        # 存储概念
        self.concepts[blended_concept.id] = blended_concept
        
        logger.info(f"成功混合概念: {blended_concept.name}")
        return blended_concept
        
    def _hybridize_concepts(self, concepts: List[Concept]) -> Concept:
        """杂交概念"""
        # 创建杂交概念
        hybrid_name = f"{concepts[0].name}杂交体"
        hybrid_desc = f"概念杂交产物，结合了多个概念的优点"
        hybrid_concept = Concept(hybrid_name, hybrid_desc, ConceptType.COMPOSITE)
        
        # 选择性继承属性
        for concept in concepts:
            # 每个概念贡献最好的属性
            best_properties = sorted(concept.properties, 
                                   key=lambda p: p.weight * p.confidence, 
                                   reverse=True)[:3]
            
            for prop in best_properties:
                if hybrid_concept.get_property(prop.name) is None:
                    hybrid_prop = ConceptProperty(
                        name=prop.name,
                        value=prop.value,
                        type=prop.type,
                        weight=prop.weight * 0.9,
                        confidence=prop.confidence * 0.9
                    )
                    hybrid_concept.add_property(hybrid_prop)
                    
        # 存储概念
        self.concepts[hybrid_concept.id] = hybrid_concept
        
        logger.info(f"成功杂交概念: {hybrid_concept.name}")
        return hybrid_concept
        
    # ==================== 3. 概念进化和变异 ====================
    
    def evolve_concept(self, concept_id: str, evolution_type: EvolutionType = None) -> Concept:
        """进化概念"""
        if concept_id not in self.concepts:
            raise ValueError(f"概念 {concept_id} 不存在")
            
        original_concept = self.concepts[concept_id]
        evolution_type = evolution_type or random.choice(list(EvolutionType))
        
        # 创建进化版本
        evolved_concept = self._create_evolved_concept(original_concept, evolution_type)
        
        # 记录进化过程
        evolution_record = ConceptEvolution(
            concept_id=evolved_concept.id,
            parent_id=original_concept.id,
            evolution_type=evolution_type,
            changes=self._calculate_changes(original_concept, evolved_concept),
            fitness_score=self._calculate_fitness_score(evolved_concept),
            timestamp=datetime.datetime.now()
        )
        self.evolution_records.append(evolution_record)
        
        # 存储新概念
        self.concepts[evolved_concept.id] = evolved_concept
        
        # 更新父子关系
        evolved_concept.parent_concepts = [original_concept.id]
        original_concept.child_concepts.append(evolved_concept.id)
        
        logger.info(f"概念 {original_concept.name} 通过 {evolution_type.value} 进化为 {evolved_concept.name}")
        return evolved_concept
        
    def _create_evolved_concept(self, original_concept: Concept, evolution_type: EvolutionType) -> Concept:
        """创建进化后的概念"""
        # 创建新概念
        evolved_name = f"{original_concept.name}_进化版"
        evolved_desc = f"基于{evolution_type.value}进化的概念"
        evolved_concept = Concept(evolved_name, evolved_desc, original_concept.concept_type)
        
        # 复制原有属性
        for prop in original_concept.properties:
            evolved_concept.add_property(prop)
            
        # 根据进化类型进行变异
        if evolution_type == EvolutionType.MUTATION:
            self._apply_mutation(evolved_concept)
        elif evolution_type == EvolutionType.CROSSOVER:
            self._apply_crossover(evolved_concept, original_concept)
        elif evolution_type == EvolutionType.ADAPTATION:
            self._apply_adaptation(evolved_concept, original_concept)
        elif evolution_type == EvolutionType.INNOVATION:
            self._apply_innovation(evolved_concept, original_concept)
            
        return evolved_concept
        
    def _apply_mutation(self, concept: Concept):
        """应用变异"""
        # 随机修改属性
        if concept.properties and random.random() < self.mutation_rate:
            prop_to_mutate = random.choice(concept.properties)
            
            # 修改属性值
            if isinstance(prop_to_mutate.value, (int, float)):
                prop_to_mutate.value *= random.uniform(0.8, 1.2)
            elif isinstance(prop_to_mutate.value, str):
                # 字符串变异
                original = prop_to_mutate.value
                if len(original) > 3:
                    # 替换部分字符
                    chars = list(original)
                    idx = random.randint(0, len(chars) - 1)
                    chars[idx] = chr(ord('a') + random.randint(0, 25))
                    prop_to_mutate.value = ''.join(chars)
                    
            # 调整置信度
            prop_to_mutate.confidence *= random.uniform(0.9, 1.1)
            
        # 添加新属性
        if random.random() < 0.3:
            new_prop = ConceptProperty(
                name=f"变异属性_{random.randint(100, 999)}",
                value=random.randint(1, 100),
                type="int",
                weight=random.uniform(0.3, 0.8),
                confidence=random.uniform(0.5, 0.9)
            )
            concept.add_property(new_prop)
            
    def _apply_crossover(self, concept: Concept, parent: Concept):
        """应用交叉"""
        # 找到另一个可交叉的概念
        candidates = [c for c in self.concepts.values() 
                     if c.id != parent.id and c.concept_type == parent.concept_type]
        
        if candidates:
            other_parent = random.choice(candidates)
            
            # 交换部分属性
            for prop in other_parent.properties[:3]:
                if concept.get_property(prop.name) is None:
                    crossover_prop = ConceptProperty(
                        name=prop.name,
                        value=prop.value,
                        type=prop.type,
                        weight=prop.weight * 0.8,
                        confidence=prop.confidence * 0.8
                    )
                    concept.add_property(crossover_prop)
                    
    def _apply_adaptation(self, concept: Concept, parent: Concept):
        """应用适应"""
        # 增强适应性属性
        for prop in concept.properties:
            if "适应性" in prop.name or "适应" in prop.name:
                if isinstance(prop.value, (int, float)):
                    prop.value *= random.uniform(1.1, 1.5)
                    
        # 添加适应相关属性
        adaptation_prop = ConceptProperty(
            name="适应度",
            value=random.uniform(0.7, 1.0),
            type="float",
            weight=0.9,
            confidence=0.8
        )
        concept.add_property(adaptation_prop)
        
    def _apply_innovation(self, concept: Concept, parent: Concept):
        """应用创新"""
        # 添加创新属性
        innovation_prop = ConceptProperty(
            name="创新度",
            value=random.uniform(0.8, 1.0),
            type="float",
            weight=1.0,
            confidence=0.9
        )
        concept.add_property(innovation_prop)
        
        # 添加新颖的关系
        if random.random() < 0.5:
            other_concepts = [c for c in self.concepts.values() if c.id != parent.id]
            if other_concepts:
                target = random.choice(other_concepts)
                new_relation = ConceptRelation(
                    target_concept_id=target.id,
                    relation_type="创新关联",
                    strength=random.uniform(0.6, 1.0),
                    bidirectional=True
                )
                concept.add_relation(new_relation)
                
    def _calculate_changes(self, original: Concept, evolved: Concept) -> Dict[str, Any]:
        """计算变化"""
        changes = {
            "property_changes": 0,
            "relation_changes": 0,
            "confidence_change": evolved.confidence_score - original.confidence_score,
            "new_properties": [],
            "modified_properties": [],
            "new_relations": []
        }
        
        # 计算属性变化
        original_props = {prop.name: prop for prop in original.properties}
        evolved_props = {prop.name: prop for prop in evolved.properties}
        
        for name, evolved_prop in evolved_props.items():
            if name not in original_props:
                changes["new_properties"].append(name)
                changes["property_changes"] += 1
            else:
                original_prop = original_props[name]
                if original_prop.value != evolved_prop.value:
                    changes["modified_properties"].append(name)
                    changes["property_changes"] += 1
                    
        # 计算关系变化
        original_relations = {(rel.target_concept_id, rel.relation_type) for rel in original.relations}
        evolved_relations = {(rel.target_concept_id, rel.relation_type) for rel in evolved.relations}
        
        new_relations = evolved_relations - original_relations
        changes["new_relations"] = [str(rel) for rel in new_relations]
        changes["relation_changes"] = len(new_relations)
        
        return changes
        
    def _calculate_fitness_score(self, concept: Concept) -> float:
        """计算适应度分数"""
        # 基于多个因素计算适应度
        factors = []
        
        # 置信度因子
        factors.append(concept.confidence_score * 0.3)
        
        # 属性质量因子
        if concept.properties:
            avg_prop_quality = sum(prop.weight * prop.confidence for prop in concept.properties) / len(concept.properties)
            factors.append(avg_prop_quality * 0.3)
            
        # 关系丰富度因子
        relation_factor = min(len(concept.relations) / 10, 1.0) * 0.2
        factors.append(relation_factor)
        
        # 创新度因子（如果有相关属性）
        innovation_prop = concept.get_property("创新度")
        innovation_factor = innovation_prop.value if innovation_prop else 0.5
        factors.append(innovation_factor * 0.2)
        
        return sum(factors)
        
    # ==================== 4. 概念有效性验证 ====================
    
    def validate_concept(self, concept_id: str, validation_criteria: Dict[str, float] = None) -> ConceptValidation:
        """验证概念有效性"""
        if concept_id not in self.concepts:
            raise ValueError(f"概念 {concept_id} 不存在")
            
        concept = self.concepts[concept_id]
        validation_criteria = validation_criteria or self._get_default_validation_criteria()
        
        # 执行验证
        validation_result = self._perform_validation(concept, validation_criteria)
        
        # 记录验证结果
        validation_record = ConceptValidation(
            concept_id=concept_id,
            status=validation_result["status"],
            score=validation_result["score"],
            criteria=validation_result["criteria_scores"],
            timestamp=datetime.datetime.now(),
            validator="ConceptInnovator",
            notes=validation_result["notes"]
        )
        
        self.validation_records[concept_id] = validation_record
        
        # 更新概念置信度
        concept.confidence_score = validation_result["score"]
        
        logger.info(f"概念 {concept.name} 验证完成，得分: {validation_result['score']:.2f}")
        return validation_record
        
    def _get_default_validation_criteria(self) -> Dict[str, float]:
        """获取默认验证标准"""
        return {
            "逻辑一致性": 0.3,
            "概念完整性": 0.25,
            "实用性": 0.2,
            "创新性": 0.15,
            "可理解性": 0.1
        }
        
    def _perform_validation(self, concept: Concept, criteria: Dict[str, float]) -> Dict[str, Any]:
        """执行验证"""
        criteria_scores = {}
        total_score = 0
        
        # 逻辑一致性验证
        consistency_score = self._validate_logical_consistency(concept)
        criteria_scores["逻辑一致性"] = consistency_score
        total_score += consistency_score * criteria["逻辑一致性"]
        
        # 概念完整性验证
        completeness_score = self._validate_completeness(concept)
        criteria_scores["概念完整性"] = completeness_score
        total_score += completeness_score * criteria["概念完整性"]
        
        # 实用性验证
        practicality_score = self._validate_practicality(concept)
        criteria_scores["实用性"] = practicality_score
        total_score += practicality_score * criteria["实用性"]
        
        # 创新性验证
        innovation_score = self._validate_innovation(concept)
        criteria_scores["创新性"] = innovation_score
        total_score += innovation_score * criteria["创新性"]
        
        # 可理解性验证
        clarity_score = self._validate_clarity(concept)
        criteria_scores["可理解性"] = clarity_score
        total_score += clarity_score * criteria["可理解性"]
        
        # 确定验证状态
        if total_score >= self.validation_threshold:
            status = ValidationStatus.VALID
        elif total_score >= self.validation_threshold * 0.6:
            status = ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.INVALID
            
        return {
            "status": status,
            "score": total_score,
            "criteria_scores": criteria_scores,
            "notes": f"验证完成，总分: {total_score:.2f}"
        }
        
    def _validate_logical_consistency(self, concept: Concept) -> float:
        """验证逻辑一致性"""
        score = 1.0
        
        # 检查属性冲突
        for i, prop1 in enumerate(concept.properties):
            for prop2 in concept.properties[i+1:]:
                if prop1.name != prop2.name and prop1.value == prop2.value:
                    # 相同值但不同名称的属性可能存在逻辑冲突
                    score -= 0.1
                    
        # 检查关系一致性
        for relation in concept.relations:
            # 简化检查：如果关系强度过高但概念本身置信度低，可能不一致
            if relation.strength > 0.9 and concept.confidence_score < 0.5:
                score -= 0.2
                
        return max(0.0, score)
        
    def _validate_completeness(self, concept: Concept) -> float:
        """验证概念完整性"""
        score = 0.5  # 基础分数
        
        # 基础属性检查
        if concept.properties:
            score += min(len(concept.properties) / 5, 0.3)  # 最多0.3分
            
        # 关系完整性检查
        if concept.relations:
            score += min(len(concept.relations) / 3, 0.2)  # 最多0.2分
            
        return min(1.0, score)
        
    def _validate_practicality(self, concept: Concept) -> float:
        """验证实用性"""
        score = 0.5  # 基础分数
        
        # 基于属性实用性评分
        for prop in concept.properties:
            if prop.confidence > 0.7 and prop.weight > 0.5:
                score += 0.1
                
        # 基于关系实用性评分
        for relation in concept.relations:
            if relation.strength > 0.6:
                score += 0.05
                
        return min(1.0, score)
        
    def _validate_innovation(self, concept: Concept) -> float:
        """验证创新性"""
        score = 0.3  # 基础分数
        
        # 检查是否是全新概念
        is_new = len(concept.parent_concepts) == 0
        if is_new:
            score += 0.3
            
        # 检查创新属性
        innovation_prop = concept.get_property("创新度")
        if innovation_prop:
            score += innovation_prop.value * 0.4
            
        # 检查新颖关系
        novel_relations = [rel for rel in concept.relations if "创新" in rel.relation_type]
        if novel_relations:
            score += 0.2
            
        return min(1.0, score)
        
    def _validate_clarity(self, concept: Concept) -> float:
        """验证可理解性"""
        score = 1.0
        
        # 基于描述长度和清晰度
        if len(concept.description) < 10:
            score -= 0.3
        elif len(concept.description) > 200:
            score -= 0.2
            
        # 基于属性命名清晰度
        unclear_props = 0
        for prop in concept.properties:
            if len(prop.name) < 3 or "变异属性" in prop.name:
                unclear_props += 1
                
        if unclear_props > 0:
            score -= unclear_props * 0.1
            
        return max(0.0, score)
        
    # ==================== 5. 概念应用和实现 ====================
    
    def apply_concept(self, concept_id: str, application_domain: str, implementation_params: Dict[str, Any] = None) -> ConceptApplication:
        """应用概念"""
        if concept_id not in self.concepts:
            raise ValueError(f"概念 {concept_id} 不存在")
            
        concept = self.concepts[concept_id]
        implementation_params = implementation_params or {}
        
        # 创建应用记录
        application = ConceptApplication(
            concept_id=concept_id,
            application_domain=application_domain,
            implementation=implementation_params,
            success_rate=0.0,  # 初始成功率为0
            usage_count=1,
            last_used=datetime.datetime.now()
        )
        
        self.application_records[concept_id] = application
        
        # 更新概念使用统计
        concept.usage_count += 1
        
        # 模拟应用过程
        self._simulate_concept_application(concept, application)
        
        logger.info(f"概念 {concept.name} 在 {application_domain} 领域得到应用")
        return application
        
    def _simulate_concept_application(self, concept: Concept, application: ConceptApplication):
        """模拟概念应用过程"""
        # 基于概念质量计算成功率
        quality_factors = [
            concept.confidence_score,
            sum(prop.confidence for prop in concept.properties) / max(len(concept.properties), 1),
            min(len(concept.relations) / 5, 1.0)
        ]
        
        base_success_rate = sum(quality_factors) / len(quality_factors)
        
        # 添加随机因素
        random_factor = random.uniform(0.8, 1.2)
        final_success_rate = min(1.0, base_success_rate * random_factor)
        
        application.success_rate = final_success_rate
        
        # 更新应用参数
        application.implementation.update({
            "quality_score": base_success_rate,
            "random_factor": random_factor,
            "applied_at": datetime.datetime.now().isoformat()
        })
        
    def get_applicable_domains(self, concept_id: str) -> List[str]:
        """获取概念适用领域"""
        if concept_id not in self.concepts:
            return []
            
        concept = self.concepts[concept_id]
        applicable_domains = []
        
        # 基于概念类型确定适用领域
        domain_mapping = {
            ConceptType.BASIC: ["科学", "工程", "教育"],
            ConceptType.COMPOSITE: ["研究", "开发", "创新"],
            ConceptType.ABSTRACT: ["理论", "哲学", "研究"],
            ConceptType.CONCRETE: ["应用", "实践", "工程"],
            ConceptType.TEMPORAL: ["时间分析", "过程管理", "历史研究"],
            ConceptType.SPATIAL: ["地理", "空间分析", "建筑设计"],
            ConceptType.CAUSAL: ["因果分析", "预测", "决策支持"],
            ConceptType.RELATIONAL: ["网络分析", "关系建模", "社会分析"]
        }
        
        applicable_domains = domain_mapping.get(concept.concept_type, ["通用"])
        
        # 基于属性调整
        for prop in concept.properties:
            if "领域" in prop.name:
                applicable_domains.append(str(prop.value))
                
        return list(set(applicable_domains))
        
    # ==================== 6. 概念库管理和更新 ====================
    
    def update_concept_library(self, auto_cleanup: bool = True):
        """更新概念库"""
        logger.info("开始更新概念库...")
        
        # 清理无效概念
        if auto_cleanup:
            self._cleanup_invalid_concepts()
            
        # 优化概念结构
        self._optimize_concept_structure()
        
        # 更新概念关联
        self._update_concept_relationships()
        
        # 重新计算影响分数
        self._recalculate_impact_scores()
        
        # 保存更新后的概念库
        self.save_concept_library()
        
        logger.info("概念库更新完成")
        
    def _cleanup_invalid_concepts(self):
        """清理无效概念"""
        invalid_concepts = []
        
        for concept_id, concept in self.concepts.items():
            # 检查概念有效性
            if self._is_concept_invalid(concept):
                invalid_concepts.append(concept_id)
                
        # 删除无效概念
        for concept_id in invalid_concepts:
            del self.concepts[concept_id]
            logger.info(f"删除了无效概念: {concept_id}")
            
    def _is_concept_invalid(self, concept: Concept) -> bool:
        """判断概念是否无效"""
        # 基本有效性检查
        if not concept.name or not concept.description:
            return True
            
        if concept.confidence_score < 0.1:  # 置信度过低
            return True
            
        if len(concept.properties) == 0 and len(concept.relations) == 0:
            return True  # 没有任何属性和关系
            
        return False
        
    def _optimize_concept_structure(self):
        """优化概念结构"""
        for concept in self.concepts.values():
            # 合并相似属性
            self._merge_similar_properties(concept)
            
            # 优化关系
            self._optimize_relations(concept)
            
            # 更新置信度
            self._update_confidence_score(concept)
            
    def _merge_similar_properties(self, concept: Concept):
        """合并相似属性"""
        property_groups = defaultdict(list)
        
        # 按属性名分组
        for prop in concept.properties:
            base_name = prop.name.replace("继承_", "").replace("变异属性_", "")
            property_groups[base_name].append(prop)
            
        # 合并每组属性
        for base_name, props in property_groups.items():
            if len(props) > 1:
                # 保留权重最高的属性
                best_prop = max(props, key=lambda p: p.weight * p.confidence)
                
                # 删除其他属性
                for prop in props:
                    if prop != best_prop:
                        concept.properties.remove(prop)
                        
    def _optimize_relations(self, concept: Concept):
        """优化关系"""
        # 合并重复关系
        relation_groups = defaultdict(list)
        
        for relation in concept.relations:
            key = (relation.target_concept_id, relation.relation_type)
            relation_groups[key].append(relation)
            
        # 保留强度最高的关系
        for key, relations in relation_groups.items():
            if len(relations) > 1:
                best_relation = max(relations, key=lambda r: r.strength)
                
                for relation in relations:
                    if relation != best_relation:
                        concept.relations.remove(relation)
                        
    def _update_confidence_score(self, concept: Concept):
        """更新置信度分数"""
        if not concept.properties:
            concept.confidence_score = 0.3
            return
            
        # 基于属性质量计算置信度
        property_quality = sum(prop.confidence * prop.weight for prop in concept.properties) / len(concept.properties)
        
        # 基于关系质量计算置信度
        relation_quality = sum(rel.strength for rel in concept.relations) / max(len(concept.relations), 1)
        
        # 综合计算
        concept.confidence_score = (property_quality * 0.7 + relation_quality * 0.3)
        
    def _update_concept_relationships(self):
        """更新概念关系"""
        # 重建所有概念的关系
        for concept in self.concepts.values():
            # 移除过时关系
            concept.relations = [rel for rel in concept.relations 
                               if rel.target_concept_id in self.concepts]
            
            # 添加新的潜在关系
            self._discover_new_relations(concept)
            
    def _discover_new_relations(self, concept: Concept):
        """发现新关系"""
        # 基于属性相似性发现关系
        for other_concept in self.concepts.values():
            if other_concept.id == concept.id:
                continue
                
            similarity = self._calculate_concept_similarity(concept, other_concept)
            
            if similarity > 0.6:  # 相似度阈值
                new_relation = ConceptRelation(
                    target_concept_id=other_concept.id,
                    relation_type="相似",
                    strength=similarity,
                    bidirectional=True
                )
                
                # 检查是否已存在类似关系
                exists = any(rel.target_concept_id == other_concept.id and 
                           rel.relation_type == "相似" for rel in concept.relations)
                
                if not exists:
                    concept.relations.append(new_relation)
                    
    def _calculate_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算概念相似度"""
        similarity_score = 0.0
        
        # 基于属性相似性
        common_props = 0
        for prop1 in concept1.properties:
            for prop2 in concept2.properties:
                if prop1.name == prop2.name and prop1.value == prop2.value:
                    common_props += 1
                    
        if concept1.properties and concept2.properties:
            attribute_similarity = common_props / max(len(concept1.properties), len(concept2.properties))
            similarity_score += attribute_similarity * 0.6
            
        # 基于类型相似性
        if concept1.concept_type == concept2.concept_type:
            similarity_score += 0.2
            
        # 基于关系相似性
        common_relations = 0
        for rel1 in concept1.relations:
            for rel2 in concept2.relations:
                if rel1.target_concept_id == rel2.target_concept_id:
                    common_relations += 1
                    
        if concept1.relations and concept2.relations:
            relation_similarity = common_relations / max(len(concept1.relations), len(concept2.relations))
            similarity_score += relation_similarity * 0.2
            
        return similarity_score
        
    def _recalculate_impact_scores(self):
        """重新计算影响分数"""
        for concept in self.concepts.values():
            impact = self._calculate_concept_impact(concept)
            self.impact_assessments[concept.id] = impact
            
    def _calculate_concept_impact(self, concept: Concept) -> ConceptImpact:
        """计算概念影响"""
        # 计算影响网络
        influence_network = {}
        for relation in concept.relations:
            if relation.target_concept_id in self.concepts:
                influence_network[relation.target_concept_id] = relation.strength
                
        # 计算创新潜力
        innovation_potential = self._calculate_innovation_potential(concept)
        
        # 计算实用价值
        practical_value = self._calculate_practical_value(concept)
        
        # 计算理论价值
        theoretical_value = self._calculate_theoretical_value(concept)
        
        # 综合影响分数
        impact_score = (innovation_potential * 0.3 + 
                       practical_value * 0.4 + 
                       theoretical_value * 0.3)
        
        return ConceptImpact(
            concept_id=concept.id,
            impact_score=impact_score,
            influence_network=influence_network,
            innovation_potential=innovation_potential,
            practical_value=practical_value,
            theoretical_value=theoretical_value,
            timestamp=datetime.datetime.now()
        )
        
    def _calculate_innovation_potential(self, concept: Concept) -> float:
        """计算创新潜力"""
        potential = 0.3  # 基础潜力
        
        # 基于属性创新性
        for prop in concept.properties:
            if "创新" in prop.name or "新" in prop.name:
                potential += prop.confidence * 0.2
                
        # 基于关系创新性
        for relation in concept.relations:
            if "创新" in relation.relation_type:
                potential += relation.strength * 0.1
                
        # 基于父概念数量（跨领域融合潜力）
        if len(concept.parent_concepts) > 1:
            potential += 0.2
            
        return min(1.0, potential)
        
    def _calculate_practical_value(self, concept: Concept) -> float:
        """计算实用价值"""
        value = 0.4  # 基础价值
        
        # 基于应用记录
        if concept.id in self.application_records:
            app = self.application_records[concept.id]
            value += app.success_rate * 0.3
            
        # 基于使用次数
        if concept.usage_count > 0:
            usage_factor = min(concept.usage_count / 10, 1.0)
            value += usage_factor * 0.2
            
        # 基于关系强度
        avg_relation_strength = sum(rel.strength for rel in concept.relations) / max(len(concept.relations), 1)
        value += avg_relation_strength * 0.1
        
        return min(1.0, value)
        
    def _calculate_theoretical_value(self, concept: Concept) -> float:
        """计算理论价值"""
        value = 0.3  # 基础价值
        
        # 基于概念完整性
        if concept.properties:
            value += min(len(concept.properties) / 10, 0.3)
            
        # 基于关系丰富度
        if concept.relations:
            value += min(len(concept.relations) / 8, 0.2)
            
        # 基于抽象程度
        if concept.concept_type in [ConceptType.ABSTRACT, ConceptType.BASIC]:
            value += 0.2
            
        return min(1.0, value)
        
    # ==================== 7. 概念影响评估 ====================
    
    def assess_concept_impact(self, concept_id: str) -> ConceptImpact:
        """评估概念影响"""
        if concept_id not in self.concepts:
            raise ValueError(f"概念 {concept_id} 不存在")
            
        concept = self.concepts[concept_id]
        impact = self._calculate_concept_impact(concept)
        
        self.impact_assessments[concept_id] = impact
        
        logger.info(f"概念 {concept.name} 影响评估完成，分数: {impact.impact_score:.2f}")
        return impact
        
    def get_top_influential_concepts(self, limit: int = 10) -> List[Tuple[Concept, ConceptImpact]]:
        """获取最有影响力的概念"""
        impacts = []
        
        for concept_id, impact in self.impact_assessments.items():
            if concept_id in self.concepts:
                concept = self.concepts[concept_id]
                impacts.append((concept, impact))
                
        # 按影响分数排序
        impacts.sort(key=lambda x: x[1].impact_score, reverse=True)
        
        return impacts[:limit]
        
    def get_concept_influence_network(self, concept_id: str, depth: int = 2) -> Dict[str, Any]:
        """获取概念影响网络"""
        if concept_id not in self.concepts:
            return {}
            
        network = {
            "central_concept": concept_id,
            "direct_influences": {},
            "indirect_influences": {},
            "influence_paths": []
        }
        
        # 直接影响
        concept = self.concepts[concept_id]
        for relation in concept.relations:
            if relation.target_concept_id in self.concepts:
                network["direct_influences"][relation.target_concept_id] = {
                    "strength": relation.strength,
                    "type": relation.relation_type
                }
                
        # 间接影响（递归搜索）
        if depth > 1:
            for target_id in network["direct_influences"].keys():
                indirect = self._get_concept_influences(target_id, depth - 1)
                network["indirect_influences"].update(indirect)
                
        return network
        
    def _get_concept_influences(self, concept_id: str, depth: int) -> Dict[str, Any]:
        """获取概念影响（递归）"""
        if depth <= 0 or concept_id not in self.concepts:
            return {}
            
        influences = {}
        concept = self.concepts[concept_id]
        
        for relation in concept.relations:
            if relation.target_concept_id in self.concepts:
                influences[relation.target_concept_id] = {
                    "strength": relation.strength,
                    "type": relation.relation_type,
                    "path_length": depth
                }
                
                # 递归获取更深层影响
                if depth > 1:
                    deeper = self._get_concept_influences(relation.target_concept_id, depth - 1)
                    for k, v in deeper.items():
                        influences[k] = v
                        
        return influences
        
    def analyze_concept_evolution_trends(self) -> Dict[str, Any]:
        """分析概念进化趋势"""
        if not self.evolution_records:
            return {"message": "没有进化记录可供分析"}
            
        trends = {
            "total_evolutions": len(self.evolution_records),
            "evolution_types": {},
            "fitness_trends": [],
            "success_rate": 0.0,
            "most_successful_evolution": None
        }
        
        # 统计进化类型
        for record in self.evolution_records:
            evo_type = record.evolution_type.value
            trends["evolution_types"][evo_type] = trends["evolution_types"].get(evo_type, 0) + 1
            
        # 分析适应度趋势
        fitness_scores = [record.fitness_score for record in self.evolution_records]
        trends["fitness_trends"] = {
            "average": statistics.mean(fitness_scores),
            "median": statistics.median(fitness_scores),
            "max": max(fitness_scores),
            "min": min(fitness_scores),
            "std_dev": statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0
        }
        
        # 计算成功率（适应度超过父概念的比例）
        successful_evolutions = 0
        for record in self.evolution_records:
            if record.parent_id in self.concepts:
                parent_concept = self.concepts[record.parent_id]
                parent_fitness = self._calculate_fitness_score(parent_concept)
                if record.fitness_score > parent_fitness:
                    successful_evolutions += 1
                    
        trends["success_rate"] = successful_evolutions / len(self.evolution_records)
        
        # 最成功的进化
        best_record = max(self.evolution_records, key=lambda r: r.fitness_score)
        trends["most_successful_evolution"] = {
            "concept_id": best_record.concept_id,
            "parent_id": best_record.parent_id,
            "evolution_type": best_record.evolution_type.value,
            "fitness_score": best_record.fitness_score
        }
        
        return trends
        
    def get_concept_statistics(self) -> Dict[str, Any]:
        """获取概念统计信息"""
        total_concepts = len(self.concepts)
        if total_concepts == 0:
            return {"message": "概念库为空"}
            
        # 概念类型分布
        type_distribution = Counter(concept.concept_type.value for concept in self.concepts.values())
        
        # 属性统计
        total_properties = sum(len(concept.properties) for concept in self.concepts.values())
        avg_properties = total_properties / total_concepts
        
        # 关系统计
        total_relations = sum(len(concept.relations) for concept in self.concepts.values())
        avg_relations = total_relations / total_concepts
        
        # 置信度统计
        confidence_scores = [concept.confidence_score for concept in self.concepts.values()]
        
        # 使用统计
        total_usage = sum(concept.usage_count for concept in self.concepts.values())
        avg_usage = total_usage / total_concepts
        
        return {
            "total_concepts": total_concepts,
            "type_distribution": dict(type_distribution),
            "average_properties_per_concept": avg_properties,
            "average_relations_per_concept": avg_relations,
            "confidence_statistics": {
                "average": statistics.mean(confidence_scores),
                "median": statistics.median(confidence_scores),
                "max": max(confidence_scores),
                "min": min(confidence_scores)
            },
            "usage_statistics": {
                "total_usage": total_usage,
                "average_usage_per_concept": avg_usage,
                "most_used_concept": max(self.concepts.values(), key=lambda c: c.usage_count).name if self.concepts else None
            },
            "validation_statistics": {
                "validated_concepts": len(self.validation_records),
                "validation_rate": len(self.validation_records) / total_concepts
            },
            "application_statistics": {
                "applied_concepts": len(self.application_records),
                "application_rate": len(self.application_records) / total_concepts
            }
        }
        
    # ==================== 公共接口方法 ====================
    
    def search_concepts(self, query: str, max_results: int = 10) -> List[Concept]:
        """搜索概念"""
        query_lower = query.lower()
        matching_concepts = []
        
        for concept in self.concepts.values():
            # 检查名称匹配
            if query_lower in concept.name.lower():
                matching_concepts.append((concept, 1.0))
                continue
                
            # 检查描述匹配
            if query_lower in concept.description.lower():
                matching_concepts.append((concept, 0.8))
                continue
                
            # 检查属性匹配
            for prop in concept.properties:
                if query_lower in prop.name.lower() or query_lower in str(prop.value).lower():
                    matching_concepts.append((concept, 0.6))
                    break
                    
        # 按匹配度排序
        matching_concepts.sort(key=lambda x: x[1], reverse=True)
        
        return [concept for concept, score in matching_concepts[:max_results]]
        
    def get_concept_by_id(self, concept_id: str) -> Optional[Concept]:
        """根据ID获取概念"""
        return self.concepts.get(concept_id)
        
    def get_all_concepts(self) -> List[Concept]:
        """获取所有概念"""
        return list(self.concepts.values())
        
    def export_concept_network(self, format: str = "json") -> str:
        """导出概念网络"""
        network_data = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "total_concepts": len(self.concepts),
                "export_time": datetime.datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        # 添加节点
        for concept in self.concepts.values():
            network_data["nodes"].append({
                "id": concept.id,
                "label": concept.name,
                "type": concept.concept_type.value,
                "confidence": concept.confidence_score,
                "properties": len(concept.properties),
                "relations": len(concept.relations)
            })
            
        # 添加边
        for concept in self.concepts.values():
            for relation in concept.relations:
                network_data["edges"].append({
                    "source": concept.id,
                    "target": relation.target_concept_id,
                    "type": relation.relation_type,
                    "strength": relation.strength
                })
                
        if format.lower() == "json":
            return json.dumps(network_data, ensure_ascii=False, indent=2)
        else:
            # 可以扩展其他格式
            raise ValueError(f"不支持的导出格式: {format}")
            
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行综合分析"""
        analysis_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "concept_statistics": self.get_concept_statistics(),
            "evolution_trends": self.analyze_concept_evolution_trends(),
            "top_influential_concepts": [
                {
                    "concept_name": concept.name,
                    "impact_score": impact.impact_score,
                    "innovation_potential": impact.innovation_potential,
                    "practical_value": impact.practical_value
                }
                for concept, impact in self.get_top_influential_concepts(5)
            ],
            "validation_summary": {
                "total_validated": len(self.validation_records),
                "validation_rate": len(self.validation_records) / max(len(self.concepts), 1),
                "average_validation_score": statistics.mean([record.score for record in self.validation_records.values()]) if self.validation_records else 0
            },
            "application_summary": {
                "total_applied": len(self.application_records),
                "application_rate": len(self.application_records) / max(len(self.concepts), 1),
                "average_success_rate": statistics.mean([app.success_rate for app in self.application_records.values()]) if self.application_records else 0
            }
        }
        
        return analysis_results


def demo_concept_innovator():
    """演示概念创新器的功能"""
    print("=== E5 概念创新器演示 ===\n")
    
    # 创建概念创新器
    innovator = ConceptInnovator()
    
    # 1. 生成新概念
    print("1. 生成新概念:")
    new_concept = innovator.generate_new_concept(
        context={
            "keywords": ["智能", "系统", "创新"],
            "description": "基于人工智能的创新系统概念",
            "type_hint": "abstract"
        },
        target_domain="computer_science"
    )
    print(f"   生成概念: {new_concept.name}")
    print(f"   描述: {new_concept.description}")
    print(f"   属性数量: {len(new_concept.properties)}")
    print()
    
    # 2. 概念组合
    print("2. 概念组合:")
    concept_ids = list(innovator.concepts.keys())[:3]
    if len(concept_ids) >= 2:
        combined_concept = innovator.combine_concepts(concept_ids[:2], "merge")
        print(f"   组合概念: {combined_concept.name}")
        print(f"   组合策略: merge")
    print()
    
    # 3. 概念进化
    print("3. 概念进化:")
    if innovator.concepts:
        first_concept_id = list(innovator.concepts.keys())[0]
        evolved_concept = innovator.evolve_concept(first_concept_id, EvolutionType.MUTATION)
        print(f"   原始概念: {innovator.concepts[first_concept_id].name}")
        print(f"   进化概念: {evolved_concept.name}")
        print(f"   进化类型: {EvolutionType.MUTATION.value}")
    print()
    
    # 4. 概念验证
    print("4. 概念验证:")
    if innovator.concepts:
        concept_to_validate = list(innovator.concepts.keys())[0]
        validation = innovator.validate_concept(concept_to_validate)
        print(f"   验证概念: {innovator.concepts[concept_to_validate].name}")
        print(f"   验证状态: {validation.status.value}")
        print(f"   验证分数: {validation.score:.2f}")
    print()
    
    # 5. 概念应用
    print("5. 概念应用:")
    if innovator.concepts:
        concept_to_apply = list(innovator.concepts.keys())[0]
        application = innovator.apply_concept(concept_to_apply, "研究开发")
        print(f"   应用概念: {innovator.concepts[concept_to_apply].name}")
        print(f"   应用领域: {application.application_domain}")
        print(f"   成功率: {application.success_rate:.2f}")
    print()
    
    # 6. 概念搜索
    print("6. 概念搜索:")
    search_results = innovator.search_concepts("智能", max_results=3)
    print(f"   搜索关键词: '智能'")
    print(f"   找到 {len(search_results)} 个相关概念:")
    for concept in search_results:
        print(f"   - {concept.name}")
    print()
    
    # 7. 统计分析
    print("7. 统计分析:")
    stats = innovator.get_concept_statistics()
    print(f"   总概念数: {stats['total_concepts']}")
    print(f"   平均属性数: {stats['average_properties_per_concept']:.1f}")
    print(f"   平均关系数: {stats['average_relations_per_concept']:.1f}")
    print(f"   平均置信度: {stats['confidence_statistics']['average']:.2f}")
    print()
    
    # 8. 综合分析
    print("8. 综合分析:")
    analysis = innovator.run_comprehensive_analysis()
    print(f"   分析时间: {analysis['timestamp']}")
    print(f"   验证率: {analysis['validation_summary']['validation_rate']:.2%}")
    print(f"   应用率: {analysis['application_summary']['application_rate']:.2%}")
    
    # 保存概念库
    innovator.save_concept_library()
    print("\n概念库已保存!")


if __name__ == "__main__":
    demo_concept_innovator()