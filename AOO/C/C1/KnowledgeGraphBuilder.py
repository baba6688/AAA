"""
C1知识图谱构建器
================

实现完整的知识图谱构建、存储、查询和可视化功能。

功能特性：
1. 实体识别和抽取（公司、行业、人物、事件等）
2. 关系抽取和建模（因果关系、相关关系、时间关系等）
3. 知识图谱构建和更新
4. 图数据库存储和管理
5. 知识图谱查询和检索
6. 知识图谱可视化和分析
7. 知识图谱质量评估和验证


创建时间：2025-11-05
"""

import networkx as nx
import pandas as pd
import numpy as np
import json
import pickle
import sqlite3
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体类"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'properties': self.properties,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class Relation:
    """关系类"""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'properties': self.properties,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class KnowledgeTriple:
    """知识三元组"""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subject': self.subject.to_dict(),
            'predicate': self.predicate,
            'object': self.object.to_dict(),
            'confidence': self.confidence,
            'source': self.source
        }


class EntityExtractor:
    """实体识别和抽取器"""
    
    def __init__(self):
        self.entity_patterns = {
            'company': [
                r'([A-Z][a-zA-Z\s]*(?:有限公司|股份有限公司|集团|科技|网络|数据|智能))',
                r'([A-Z][a-zA-Z\s]*(?:Corp|Inc|Ltd|Co\.))',
                r'(腾讯|阿里巴巴|百度|字节跳动|京东|美团|小米|华为|苹果|微软|谷歌|亚马逊)'
            ],
            'industry': [
                r'(人工智能|机器学习|深度学习|自然语言处理|计算机视觉)',
                r'(金融科技|区块链|云计算|大数据|物联网|5G|新能源|生物医药)',
                r'(电商|社交媒体|游戏|教育|医疗|汽车|房地产|制造业|零售)'
            ],
            'person': [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(马云|马化腾|李彦宏|刘强东|张一鸣|雷军|任正非|柳传志)'
            ],
            'event': [
                r'(\d{4}年.*?(?:发布|收购|合并|上市|投资|融资|合作))',
                r'(COVID-19|疫情|金融危机|贸易战|脱欧)',
                r'(IPO|并购|重组|破产|诉讼|监管)'
            ],
            'location': [
                r'(北京|上海|深圳|杭州|广州|南京|成都|武汉|西安|苏州)',
                r'(美国|英国|德国|法国|日本|韩国|新加坡|中国香港|中国台湾)',
                r'(硅谷|中关村|陆家嘴|金融街)'
            ],
            'technology': [
                r'(Python|Java|JavaScript|Go|Rust|Scala|Kotlin)',
                r'(TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy)',
                r'(React|Vue|Angular|Node\.js|Django|Flask)'
            ]
        }
        
        self.entity_keywords = {
            'company': ['公司', '企业', '集团', '股份有限公司', '有限公司', 'Corp', 'Inc', 'Ltd'],
            'industry': ['行业', '领域', '市场', '产业', '板块'],
            'person': ['先生', '女士', 'CEO', 'CTO', '创始人', '董事长', '总裁'],
            'event': ['事件', '会议', '发布会', '峰会', '论坛', '展览'],
            'location': ['城市', '国家', '地区', '省', '市'],
            'technology': ['技术', '框架', '语言', '平台', '工具']
        }
    
    def extract_entities(self, text: str, min_confidence: float = 0.5) -> List[Entity]:
        """从文本中提取实体"""
        entities = []
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group(1).strip()
                    if len(entity_name) >= 2:
                        # 计算置信度
                        confidence = self._calculate_confidence(entity_name, entity_type, text)
                        
                        if confidence >= min_confidence:
                            entity_id = f"{entity_type}_{hashlib.md5(entity_name.encode()).hexdigest()[:8]}"
                            entity = Entity(
                                id=entity_id,
                                name=entity_name,
                                type=entity_type,
                                properties={'source_text': text, 'match_start': match.start(), 'match_end': match.end()},
                                confidence=confidence,
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            entities.append(entity)
        
        # 去重
        unique_entities = {}
        for entity in entities:
            if entity.name not in unique_entities or entity.confidence > unique_entities[entity.name].confidence:
                unique_entities[entity.name] = entity
        
        return list(unique_entities.values())
    
    def _calculate_confidence(self, entity_name: str, entity_type: str, text: str) -> float:
        """计算实体识别置信度"""
        confidence = 0.5  # 基础置信度
        
        # 基于关键词的置信度提升
        if entity_type in self.entity_keywords:
            for keyword in self.entity_keywords[entity_type]:
                if keyword in text:
                    confidence += 0.1
        
        # 基于实体名称长度的置信度调整
        if len(entity_name) > 3:
            confidence += 0.1
        
        # 基于模式匹配的置信度
        if re.match(r'^[A-Z]', entity_name):
            confidence += 0.1
        
        return min(confidence, 1.0)


class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self):
        self.relation_patterns = {
            'causal': [
                r'(.+)导致(.+)',
                r'(.+)引起(.+)',
                r'(.+)造成(.+)',
                r'(.+)引发(.+)',
                r'(.+)促成(.+)',
                r'(.+)推动(.+)'
            ],
            'temporal': [
                r'(.+)在(.+)之前',
                r'(.+)在(.+)之后',
                r'(.+)期间(.+)',
                r'(.+)同时(.+)',
                r'(.+)终于(.+)'
            ],
            'comparative': [
                r'(.+)比(.+)',
                r'(.+)优于(.+)',
                r'(.+)低于(.+)',
                r'(.+)类似于(.+)',
                r'(.+)不同于(.+)'
            ],
            'associative': [
                r'(.+)与(.+)相关',
                r'(.+)和(.+)有关',
                r'(.+)涉及(.+)',
                r'(.+)包含(.+)',
                r'(.+)属于(.+)'
            ],
            'ownership': [
                r'(.+)拥有(.+)',
                r'(.+)控制(.+)',
                r'(.+)持有(.+)',
                r'(.+)收购(.+)',
                r'(.+)投资(.+)'
            ],
            'hierarchical': [
                r'(.+)是(.+)的一部分',
                r'(.+)属于(.+)类别',
                r'(.+)是(.+)的子集',
                r'(.+)隶属于(.+)'
            ]
        }
    
    def extract_relations(self, text: str, entities: List[Entity], min_confidence: float = 0.5) -> List[Relation]:
        """从文本中提取关系"""
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    source_text = match.group(1).strip()
                    target_text = match.group(2).strip()
                    
                    # 找到对应的实体
                    source_entity = self._find_matching_entity(source_text, entities)
                    target_entity = self._find_matching_entity(target_text, entities)
                    
                    if source_entity and target_entity:
                        confidence = self._calculate_relation_confidence(
                            source_entity, target_entity, relation_type, text
                        )
                        
                        if confidence >= min_confidence:
                            relation_id = f"{relation_type}_{hashlib.md5(f'{source_entity.id}_{target_entity.id}'.encode()).hexdigest()[:8]}"
                            relation = Relation(
                                id=relation_id,
                                source_id=source_entity.id,
                                target_id=target_entity.id,
                                relation_type=relation_type,
                                properties={
                                    'source_text': text,
                                    'pattern': pattern,
                                    'match_start': match.start(),
                                    'match_end': match.end()
                                },
                                confidence=confidence,
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            relations.append(relation)
        
        return relations
    
    def _find_matching_entity(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """找到匹配的实体"""
        for entity in entities:
            if entity.name in text or text in entity.name:
                return entity
        return None
    
    def _calculate_relation_confidence(self, source_entity: Entity, target_entity: Entity, 
                                     relation_type: str, text: str) -> float:
        """计算关系抽取置信度"""
        confidence = 0.5  # 基础置信度
        
        # 基于实体置信度的置信度提升
        confidence += (source_entity.confidence + target_entity.confidence) / 4
        
        # 基于关系类型的置信度调整
        if relation_type in ['causal', 'temporal']:
            confidence += 0.1
        
        # 基于实体类型的匹配度
        if self._are_entities_compatible(source_entity.type, target_entity.type, relation_type):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _are_entities_compatible(self, source_type: str, target_type: str, relation_type: str) -> bool:
        """检查实体类型是否兼容"""
        compatibility_rules = {
            'causal': [('company', 'event'), ('person', 'event'), ('technology', 'event')],
            'temporal': [('company', 'company'), ('person', 'person'), ('event', 'event')],
            'comparative': [('company', 'company'), ('person', 'person'), ('industry', 'industry')],
            'associative': [('company', 'technology'), ('person', 'company'), ('industry', 'company')],
            'ownership': [('company', 'company'), ('person', 'company'), ('company', 'technology')],
            'hierarchical': [('technology', 'technology'), ('industry', 'industry'), ('company', 'industry')]
        }
        
        if relation_type in compatibility_rules:
            return (source_type, target_type) in compatibility_rules[relation_type]
        return False


class KnowledgeGraphBuilder:
    """知识图谱构建器主类"""
    
    def __init__(self, storage_path: str = "knowledge_graph"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # 初始化组件
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        
        # 初始化图数据库
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        
        # 版本管理
        self.versions: List[Dict[str, Any]] = []
        self.current_version = 0
        
        # 统计信息
        self.stats = {
            'total_entities': 0,
            'total_relations': 0,
            'entity_types': Counter(),
            'relation_types': Counter(),
            'last_updated': None
        }
        
        # 初始化数据库
        self._init_database()
        
        logger.info(f"知识图谱构建器初始化完成，存储路径: {self.storage_path}")
    
    def _init_database(self):
        """初始化SQLite数据库"""
        self.db_path = self.storage_path / "knowledge_graph.db"
        conn = sqlite3.connect(str(self.db_path))
        
        # 创建表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT,
                confidence REAL,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                properties TEXT,
                confidence REAL,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (source_id) REFERENCES entities (id),
                FOREIGN KEY (target_id) REFERENCES entities (id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS versions (
                version_id INTEGER PRIMARY KEY,
                timestamp TEXT,
                entity_count INTEGER,
                relation_count INTEGER,
                description TEXT,
                checksum TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_text(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """从文本构建知识图谱"""
        logger.info(f"开始处理文本，来源: {source}")
        
        # 提取实体
        entities = self.entity_extractor.extract_entities(text)
        logger.info(f"提取到 {len(entities)} 个实体")
        
        # 提取关系
        relations = self.relation_extractor.extract_relations(text, entities)
        logger.info(f"提取到 {len(relations)} 个关系")
        
        # 更新图数据库
        added_entities = 0
        added_relations = 0
        
        for entity in entities:
            if entity.id not in self.entities:
                self.entities[entity.id] = entity
                self._add_entity_to_graph(entity)
                self._save_entity_to_db(entity)
                added_entities += 1
        
        for relation in relations:
            if relation.id not in self.relations:
                self.relations[relation.id] = relation
                self._add_relation_to_graph(relation)
                self._save_relation_to_db(relation)
                added_relations += 1
        
        # 更新统计信息
        self._update_stats()
        
        # 创建版本快照
        self._create_version_snapshot(f"添加文本来源: {source}")
        
        result = {
            'entities_added': added_entities,
            'relations_added': added_relations,
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'stats': self.stats
        }
        
        logger.info(f"文本处理完成: {result}")
        return result
    
    def _add_entity_to_graph(self, entity: Entity):
        """添加实体到图数据库"""
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            confidence=entity.confidence,
            **entity.properties
        )
    
    def _add_relation_to_graph(self, relation: Relation):
        """添加关系到图数据库"""
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation.id,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            **relation.properties
        )
    
    def _save_entity_to_db(self, entity: Entity):
        """保存实体到数据库"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            '''INSERT OR REPLACE INTO entities 
               (id, name, type, properties, confidence, created_at, updated_at) 
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (
                entity.id, entity.name, entity.type, json.dumps(entity.properties),
                entity.confidence, entity.created_at.isoformat(), entity.updated_at.isoformat()
            )
        )
        conn.commit()
        conn.close()
    
    def _save_relation_to_db(self, relation: Relation):
        """保存关系到数据库"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            '''INSERT OR REPLACE INTO relations 
               (id, source_id, target_id, relation_type, properties, confidence, created_at, updated_at) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                relation.id, relation.source_id, relation.target_id, relation.relation_type,
                json.dumps(relation.properties), relation.confidence,
                relation.created_at.isoformat(), relation.updated_at.isoformat()
            )
        )
        conn.commit()
        conn.close()
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats['total_entities'] = len(self.entities)
        self.stats['total_relations'] = len(self.relations)
        self.stats['entity_types'] = Counter(entity.type for entity in self.entities.values())
        self.stats['relation_types'] = Counter(relation.relation_type for relation in self.relations.values())
        self.stats['last_updated'] = datetime.now().isoformat()
    
    def _create_version_snapshot(self, description: str):
        """创建版本快照"""
        self.current_version += 1
        
        # 计算校验和
        entities_data = json.dumps({k: v.to_dict() for k, v in self.entities.items()}, sort_keys=True)
        relations_data = json.dumps({k: v.to_dict() for k, v in self.relations.items()}, sort_keys=True)
        checksum = hashlib.md5((entities_data + relations_data).encode()).hexdigest()
        
        version_info = {
            'version_id': self.current_version,
            'timestamp': datetime.now().isoformat(),
            'entity_count': len(self.entities),
            'relation_count': len(self.relations),
            'description': description,
            'checksum': checksum
        }
        
        self.versions.append(version_info)
        
        # 保存到数据库
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            '''INSERT INTO versions 
               (version_id, timestamp, entity_count, relation_count, description, checksum) 
               VALUES (?, ?, ?, ?, ?, ?)''',
            (
                version_info['version_id'], version_info['timestamp'],
                version_info['entity_count'], version_info['relation_count'],
                version_info['description'], version_info['checksum']
            )
        )
        conn.commit()
        conn.close()
        
        # 保存图数据
        self._save_graph_data()
    
    def _save_graph_data(self):
        """保存图数据"""
        graph_data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': [(u, v, k, d) for u, v, k, d in self.graph.edges(keys=True, data=True)],
            'entities': {k: v.to_dict() for k, v in self.entities.items()},
            'relations': {k: v.to_dict() for k, v in self.relations.items()},
            'versions': self.versions
        }
        
        with open(self.storage_path / f"graph_v{self.current_version}.pkl", 'wb') as f:
            pickle.dump(graph_data, f)
    
    def load_graph_data(self, version: int = None):
        """加载图数据"""
        if version is None:
            version = self.current_version
        
        graph_file = self.storage_path / f"graph_v{version}.pkl"
        if graph_file.exists():
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            # 重建图
            self.graph.clear()
            self.graph.add_nodes_from(graph_data['nodes'])
            self.graph.add_edges_from(graph_data['edges'])
            
            # 重建实体和关系
            self.entities = {}
            for entity_id, entity_data in graph_data['entities'].items():
                entity = Entity(**entity_data)
                entity.created_at = datetime.fromisoformat(entity_data['created_at'])
                entity.updated_at = datetime.fromisoformat(entity_data['updated_at'])
                self.entities[entity_id] = entity
            
            self.relations = {}
            for relation_id, relation_data in graph_data['relations'].items():
                relation = Relation(**relation_data)
                relation.created_at = datetime.fromisoformat(relation_data['created_at'])
                relation.updated_at = datetime.fromisoformat(relation_data['updated_at'])
                self.relations[relation_id] = relation
            
            self.versions = graph_data['versions']
            self.current_version = version
            
            self._update_stats()
            logger.info(f"成功加载版本 {version} 的图数据")
        else:
            logger.warning(f"版本 {version} 的图数据文件不存在")
    
    def query_entities(self, entity_type: str = None, name_pattern: str = None, 
                      min_confidence: float = 0.0) -> List[Entity]:
        """查询实体"""
        results = []
        
        for entity in self.entities.values():
            if entity_type and entity.type != entity_type:
                continue
            if name_pattern and name_pattern.lower() not in entity.name.lower():
                continue
            if entity.confidence < min_confidence:
                continue
            results.append(entity)
        
        return results
    
    def query_relations(self, relation_type: str = None, source_type: str = None, 
                       target_type: str = None, min_confidence: float = 0.0) -> List[Relation]:
        """查询关系"""
        results = []
        
        for relation in self.relations.values():
            if relation_type and relation.relation_type != relation_type:
                continue
            if relation.confidence < min_confidence:
                continue
            
            source_entity = self.entities.get(relation.source_id)
            target_entity = self.entities.get(relation.target_id)
            
            if source_type and source_entity and source_entity.type != source_type:
                continue
            if target_type and target_entity and target_entity.type != target_type:
                continue
            
            results.append(relation)
        
        return results
    
    def find_path(self, source_entity: str, target_entity: str, max_length: int = 5) -> List[List[str]]:
        """查找两个实体之间的路径"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source_entity, target_entity, cutoff=max_length
            ))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_subgraph(self, entities: List[str], include_relations: bool = True) -> nx.MultiDiGraph:
        """获取子图"""
        subgraph = self.graph.subgraph(entities).copy()
        
        if not include_relations:
            # 只保留节点
            for node in list(subgraph.nodes()):
                subgraph.remove_node(node)
            subgraph.add_nodes_from(entities)
        
        return subgraph
    
    def analyze_centrality(self) -> Dict[str, Dict[str, float]]:
        """分析实体中心性"""
        # 转换为简单图进行中心性计算
        simple_graph = nx.Graph(self.graph)
        
        centrality_metrics = {}
        try:
            centrality_metrics['degree'] = nx.degree_centrality(simple_graph)
        except Exception as e:
            logger.warning(f"度中心性计算失败: {e}")
            centrality_metrics['degree'] = {}
        
        try:
            centrality_metrics['betweenness'] = nx.betweenness_centrality(simple_graph)
        except Exception as e:
            logger.warning(f"介数中心性计算失败: {e}")
            centrality_metrics['betweenness'] = {}
        
        try:
            centrality_metrics['closeness'] = nx.closeness_centrality(simple_graph)
        except Exception as e:
            logger.warning(f"接近中心性计算失败: {e}")
            centrality_metrics['closeness'] = {}
        
        try:
            centrality_metrics['eigenvector'] = nx.eigenvector_centrality(simple_graph, max_iter=1000)
        except Exception as e:
            logger.warning(f"特征向量中心性计算失败: {e}")
            centrality_metrics['eigenvector'] = {}
        
        return centrality_metrics
    
    def detect_communities(self) -> Dict[int, List[str]]:
        """检测实体社区"""
        try:
            # 转换为无向图进行社区检测
            undirected_graph = self.graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected_graph)
            
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[i] = list(community)
            
            return community_dict
        except Exception as e:
            logger.error(f"社区检测失败: {e}")
            return {}
    
    def quality_assessment(self) -> Dict[str, Any]:
        """知识图谱质量评估"""
        assessment = {
            'completeness': self._assess_completeness(),
            'consistency': self._assess_consistency(),
            'accuracy': self._assess_accuracy(),
            'connectivity': self._assess_connectivity(),
            'coverage': self._assess_coverage()
        }
        
        # 计算综合质量分数
        weights = {'completeness': 0.3, 'consistency': 0.25, 'accuracy': 0.25, 'connectivity': 0.1, 'coverage': 0.1}
        overall_score = sum(assessment[key] * weights[key] for key in weights)
        assessment['overall_score'] = overall_score
        
        return assessment
    
    def _assess_completeness(self) -> float:
        """评估完整性"""
        if not self.entities:
            return 0.0
        
        # 检查实体属性完整性
        total_entities = len(self.entities)
        entities_with_properties = sum(1 for e in self.entities.values() if e.properties)
        property_completeness = entities_with_properties / total_entities
        
        # 检查关系完整性
        total_relations = len(self.relations)
        if total_relations == 0:
            relation_completeness = 0.0
        else:
            relations_with_properties = sum(1 for r in self.relations.values() if r.properties)
            relation_completeness = relations_with_properties / total_relations
        
        return (property_completeness + relation_completeness) / 2
    
    def _assess_consistency(self) -> float:
        """评估一致性"""
        # 检查重复实体
        name_groups = defaultdict(list)
        for entity in self.entities.values():
            name_groups[entity.name.lower()].append(entity)
        
        duplicates = sum(1 for group in name_groups.values() if len(group) > 1)
        name_consistency = 1.0 - (duplicates / len(self.entities)) if self.entities else 1.0
        
        # 检查关系一致性
        relation_consistency = 1.0  # 简化实现
        
        return (name_consistency + relation_consistency) / 2
    
    def _assess_accuracy(self) -> float:
        """评估准确性"""
        if not self.entities:
            return 0.0
        
        # 基于置信度的准确性评估
        avg_confidence = np.mean([e.confidence for e in self.entities.values()])
        return avg_confidence
    
    def _assess_connectivity(self) -> float:
        """评估连通性"""
        if len(self.graph.nodes()) == 0:
            return 0.0
        
        # 计算连通组件
        num_components = nx.number_weakly_connected_components(self.graph)
        total_nodes = len(self.graph.nodes())
        
        # 最大连通组件的比例
        largest_component_size = max(len(c) for c in nx.weakly_connected_components(self.graph))
        connectivity = largest_component_size / total_nodes
        
        return connectivity
    
    def _assess_coverage(self) -> float:
        """评估覆盖度"""
        # 检查实体类型多样性
        entity_types = set(e.type for e in self.entities.values())
        max_possible_types = len(self.entity_extractor.entity_patterns)
        type_coverage = len(entity_types) / max_possible_types
        
        # 检查关系类型多样性
        relation_types = set(r.relation_type for r in self.relations.values())
        max_possible_relations = len(self.relation_extractor.relation_patterns)
        relation_coverage = len(relation_types) / max_possible_relations
        
        return (type_coverage + relation_coverage) / 2
    
    def clean_knowledge_graph(self, min_confidence: float = 0.3, remove_duplicates: bool = True):
        """清洗知识图谱"""
        logger.info("开始清洗知识图谱")
        
        original_entities = len(self.entities)
        original_relations = len(self.relations)
        
        # 移除低置信度实体
        low_confidence_entities = [
            entity_id for entity_id, entity in self.entities.items()
            if entity.confidence < min_confidence
        ]
        
        for entity_id in low_confidence_entities:
            del self.entities[entity_id]
            if entity_id in self.graph:
                self.graph.remove_node(entity_id)
        
        # 移除低置信度关系
        low_confidence_relations = [
            relation_id for relation_id, relation in self.relations.items()
            if relation.confidence < min_confidence or 
            relation.source_id not in self.entities or 
            relation.target_id not in self.entities
        ]
        
        for relation_id in low_confidence_relations:
            del self.relations[relation_id]
            # 从图中移除关系
            relation = next((r for r in self.graph.edges(keys=True, data=True) 
                           if r[2] == relation_id), None)
            if relation:
                self.graph.remove_edge(relation[0], relation[1], key=relation_id)
        
        # 移除重复实体
        if remove_duplicates:
            name_groups = defaultdict(list)
            for entity_id, entity in self.entities.items():
                name_groups[entity.name.lower()].append((entity_id, entity))
            
            for name, group in name_groups.items():
                if len(group) > 1:
                    # 保留置信度最高的实体
                    group.sort(key=lambda x: x[1].confidence, reverse=True)
                    keep_entity_id, keep_entity = group[0]
                    
                    # 移除其他实体
                    for entity_id, entity in group[1:]:
                        del self.entities[entity_id]
                        if entity_id in self.graph:
                            self.graph.remove_node(entity_id)
                        
                        # 更新相关关系
                        for relation in list(self.relations.values()):
                            if relation.source_id == entity_id:
                                relation.source_id = keep_entity_id
                            elif relation.target_id == entity_id:
                                relation.target_id = keep_entity_id
        
        # 更新统计信息
        self._update_stats()
        
        # 创建版本快照
        self._create_version_snapshot(f"清洗知识图谱 (置信度>{min_confidence})")
        
        cleaned_entities = original_entities - len(self.entities)
        cleaned_relations = original_relations - len(self.relations)
        
        logger.info(f"清洗完成: 移除 {cleaned_entities} 个实体, {cleaned_relations} 个关系")
        
        return {
            'entities_removed': cleaned_entities,
            'relations_removed': cleaned_relations,
            'remaining_entities': len(self.entities),
            'remaining_relations': len(self.relations)
        }
    
    def visualize_graph(self, output_path: str = None, layout: str = 'spring', 
                       node_size_factor: float = 1000, figsize: Tuple[int, int] = (15, 10)):
        """可视化知识图谱"""
        if not self.graph.nodes():
            logger.warning("图谱为空，无法可视化")
            return
        
        plt.figure(figsize=figsize)
        
        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'hierarchical':
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        else:
            pos = nx.random_layout(self.graph)
        
        # 准备节点数据
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        color_map = {
            'company': '#FF6B6B',
            'industry': '#4ECDC4', 
            'person': '#45B7D1',
            'event': '#96CEB4',
            'location': '#FECA57',
            'technology': '#FF9FF3'
        }
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            entity_type = node_data.get('type', 'unknown')
            confidence = node_data.get('confidence', 0.5)
            
            node_colors.append(color_map.get(entity_type, '#CCCCCC'))
            node_sizes.append(confidence * node_size_factor)
            node_labels[node] = node_data.get('name', node)
        
        # 准备边数据
        edge_colors = []
        edge_widths = []
        
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            relation_type = data.get('relation_type', 'unknown')
            confidence = data.get('relation_confidence', 0.5)
            
            # 为不同关系类型设置不同颜色
            relation_colors = {
                'causal': '#FF4757',
                'temporal': '#3742FA', 
                'comparative': '#2ED573',
                'associative': '#FFA502',
                'ownership': '#FF3838',
                'hierarchical': '#5F27CD'
            }
            
            edge_colors.append(relation_colors.get(relation_type, '#747D8C'))
            edge_widths.append(confidence * 3)
        
        # 绘制图谱
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes,
                              alpha=0.8)
        
        nx.draw_networkx_edges(self.graph, pos,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.6,
                              arrows=True,
                              arrowsize=20)
        
        nx.draw_networkx_labels(self.graph, pos, node_labels, 
                               font_size=8, font_weight='bold')
        
        # 添加图例
        legend_elements = []
        for entity_type, color in color_map.items():
            if entity_type in [self.graph.nodes[node].get('type') for node in self.graph.nodes()]:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10, label=entity_type))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.title(f"知识图谱可视化 (节点: {len(self.graph.nodes())}, 边: {len(self.graph.edges())})", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"图谱可视化保存到: {output_path}")
        
        plt.show()
        plt.close()
    
    def export_to_neo4j(self, neo4j_uri: str = None, username: str = None, password: str = None):
        """导出到Neo4j数据库"""
        try:
            from neo4j import GraphDatabase
            
            if not all([neo4j_uri, username, password]):
                logger.error("Neo4j连接参数不完整")
                return False
            
            driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))
            
            with driver.session() as session:
                # 清空数据库
                session.run("MATCH (n) DETACH DELETE n")
                
                # 导入实体
                for entity in self.entities.values():
                    session.run(
                        "CREATE (e:Entity {id: $id, name: $name, type: $type, confidence: $confidence})",
                        id=entity.id, name=entity.name, type=entity.type, confidence=entity.confidence
                    )
                
                # 导入关系
                for relation in self.relations.values():
                    session.run(
                        """
                        MATCH (s:Entity {id: $source_id}), (t:Entity {id: $target_id})
                        CREATE (s)-[:RELATION {type: $relation_type, confidence: $confidence}]->(t)
                        """,
                        source_id=relation.source_id,
                        target_id=relation.target_id,
                        relation_type=relation.relation_type,
                        confidence=relation.confidence
                    )
            
            driver.close()
            logger.info("成功导出到Neo4j数据库")
            return True
            
        except ImportError:
            logger.error("请安装 neo4j Python驱动: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"导出到Neo4j失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        stats = {
            'basic_info': {
                'total_entities': len(self.entities),
                'total_relations': len(self.relations),
                'total_nodes': len(self.graph.nodes()),
                'total_edges': len(self.graph.edges()),
                'versions': len(self.versions)
            },
            'entity_distribution': dict(self.stats['entity_types']),
            'relation_distribution': dict(self.stats['relation_types']),
            'graph_metrics': {
                'density': self._safe_density_calculation(),
                'average_clustering': self._safe_clustering_calculation(),
                'number_of_components': nx.number_weakly_connected_components(self.graph)
            },
            'quality_metrics': self.quality_assessment(),
            'last_updated': self.stats['last_updated']
        }
        
        return stats
    
    def _safe_density_calculation(self) -> float:
        """安全地计算图密度"""
        try:
            return nx.density(self.graph)
        except Exception as e:
            logger.warning(f"图密度计算失败: {e}")
            return 0.0
    
    def _safe_clustering_calculation(self) -> float:
        """安全地计算平均聚类系数"""
        try:
            undirected_graph = self.graph.to_undirected()
            return nx.average_clustering(undirected_graph)
        except Exception as e:
            logger.warning(f"聚类系数计算失败: {e}")
            return 0.0
    
    def save_graph(self, file_path: str = None):
        """保存图谱到文件"""
        if file_path is None:
            file_path = self.storage_path / f"knowledge_graph_v{self.current_version}.json"
        else:
            file_path = Path(file_path)
        
        graph_data = {
            'metadata': {
                'version': self.current_version,
                'created_at': datetime.now().isoformat(),
                'statistics': self.get_statistics()
            },
            'entities': {k: v.to_dict() for k, v in self.entities.items()},
            'relations': {k: v.to_dict() for k, v in self.relations.items()},
            'versions': self.versions
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"图谱保存到: {file_path}")
    
    def load_graph(self, file_path: str):
        """从文件加载图谱"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # 清空现有数据
            self.entities.clear()
            self.relations.clear()
            self.graph.clear()
            self.versions.clear()
            
            # 重建实体
            for entity_id, entity_data in graph_data['entities'].items():
                entity = Entity(**entity_data)
                entity.created_at = datetime.fromisoformat(entity_data['created_at'])
                entity.updated_at = datetime.fromisoformat(entity_data['updated_at'])
                self.entities[entity_id] = entity
            
            # 重建关系
            for relation_id, relation_data in graph_data['relations'].items():
                relation = Relation(**relation_data)
                relation.created_at = datetime.fromisoformat(relation_data['created_at'])
                relation.updated_at = datetime.fromisoformat(relation_data['updated_at'])
                self.relations[relation_id] = relation
            
            # 重建图
            for entity in self.entities.values():
                self._add_entity_to_graph(entity)
            
            for relation in self.relations.values():
                self._add_relation_to_graph(relation)
            
            self.versions = graph_data['versions']
            self.current_version = graph_data['metadata']['version']
            
            self._update_stats()
            
            logger.info(f"成功从文件加载图谱: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载图谱失败: {e}")
            return False


def demo_knowledge_graph_builder():
    """知识图谱构建器演示"""
    print("=== 知识图谱构建器演示 ===\n")
    
    # 初始化构建器
    builder = KnowledgeGraphBuilder("demo_knowledge_graph")
    
    # 示例文本数据
    sample_texts = [
        """
        腾讯公司是中国领先的互联网科技公司，主要业务包括社交、游戏、金融科技等。
        马化腾是腾讯的创始人兼CEO。在2020年，腾讯收购了Supercell游戏公司。
        人工智能技术在腾讯的业务中发挥重要作用，特别是在微信和QQ等产品中。
        """,
        """
        阿里巴巴集团是全球知名的电子商务公司，旗下拥有淘宝、天猫、支付宝等品牌。
        马云是阿里巴巴的创始人。2021年，阿里巴巴在云计算领域投入大量资源。
        云计算技术对阿里巴巴的业务发展至关重要。
        """,
        """
        百度是中国最大的搜索引擎公司，在人工智能领域也有重要布局。
        李彦宏是百度的CEO。百度在自动驾驶和智能音箱方面投入研发。
        自然语言处理技术是百度的核心竞争力之一。
        """
    ]
    
    # 处理文本
    for i, text in enumerate(sample_texts, 1):
        print(f"处理文本 {i}...")
        result = builder.add_text(text, f"sample_text_{i}")
        print(f"  - 添加实体: {result['entities_added']}")
        print(f"  - 添加关系: {result['relations_added']}")
        print(f"  - 总实体数: {result['total_entities']}")
        print(f"  - 总关系数: {result['total_relations']}\n")
    
    # 查询实体
    print("=== 实体查询 ===")
    companies = builder.query_entities(entity_type='company')
    print(f"公司实体 ({len(companies)} 个):")
    for company in companies:
        print(f"  - {company.name} (置信度: {company.confidence:.2f})")
    print()
    
    # 查询关系
    print("=== 关系查询 ===")
    causal_relations = builder.query_relations(relation_type='causal')
    print(f"因果关系 ({len(causal_relations)} 个):")
    for relation in causal_relations:
        source_entity = builder.entities[relation.source_id]
        target_entity = builder.entities[relation.target_id]
        print(f"  - {source_entity.name} -> {target_entity.name} (置信度: {relation.confidence:.2f})")
    print()
    
    # 路径查找
    print("=== 路径查找 ===")
    if len(builder.entities) >= 2:
        entity_ids = list(builder.entities.keys())
        source_id = entity_ids[0]
        target_id = entity_ids[1] if len(entity_ids) > 1 else entity_ids[0]
        
        paths = builder.find_path(source_id, target_id)
        print(f"从 {builder.entities[source_id].name} 到 {builder.entities[target_id].name} 的路径:")
        for i, path in enumerate(paths, 1):
            path_names = [builder.entities[node_id].name for node_id in path]
            print(f"  路径 {i}: {' -> '.join(path_names)}")
        print()
    
    # 社区检测
    print("=== 社区检测 ===")
    communities = builder.detect_communities()
    print(f"检测到 {len(communities)} 个社区:")
    for community_id, members in communities.items():
        member_names = [builder.entities[node_id].name for node_id in members]
        print(f"  社区 {community_id}: {', '.join(member_names)}")
    print()
    
    # 质量评估
    print("=== 质量评估 ===")
    quality = builder.quality_assessment()
    print(f"综合质量分数: {quality['overall_score']:.2f}")
    for metric, score in quality.items():
        if metric != 'overall_score':
            print(f"  - {metric}: {score:.2f}")
    print()
    
    # 统计信息
    print("=== 统计信息 ===")
    stats = builder.get_statistics()
    print(f"基础信息:")
    for key, value in stats['basic_info'].items():
        print(f"  - {key}: {value}")
    print()
    
    # 保存图谱
    print("=== 保存图谱 ===")
    builder.save_graph("demo_knowledge_graph.json")
    
    # 可视化
    print("=== 可视化图谱 ===")
    try:
        builder.visualize_graph("demo_knowledge_graph.png")
        print("图谱可视化已保存到 demo_knowledge_graph.png")
    except Exception as e:
        print(f"可视化失败: {e}")
    
    print("\n=== 演示完成 ===")
    return builder


if __name__ == "__main__":
    # 运行演示
    demo_knowledge_graph_builder()