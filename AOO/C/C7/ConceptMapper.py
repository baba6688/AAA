"""
C7概念映射器
实现概念对齐和映射功能，支持跨领域概念转换、相似度计算、层次映射等
"""

import numpy as np
import json
import pickle
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jaccard
from scipy.stats import pearsonr
import re
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Concept:
    """概念类"""
    id: str
    name: str
    description: str = ""
    domain: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if not self.embedding:
            self.embedding = self._generate_embedding()
    
    def _generate_embedding(self) -> np.ndarray:
        """生成概念嵌入向量"""
        # 简单的词袋模型嵌入
        text = f"{self.name} {self.description} {self.domain}".lower()
        words = re.findall(r'\w+', text)
        
        # 创建词汇表
        vocab = defaultdict(int)
        for word in words:
            vocab[word] += 1
        
        # 生成向量
        vector = np.zeros(100)  # 固定维度
        for i, (word, count) in enumerate(vocab.items()):
            if i < 100:
                vector[i] = count
        
        # L2标准化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

@dataclass
class MappingResult:
    """映射结果类"""
    source_concept: Concept
    target_concept: Concept
    similarity_score: float
    mapping_type: str
    confidence: float
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConceptMapper:
    """概念映射器主类"""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.mappings: List[MappingResult] = []
        self.domains: Set[str] = set()
        self.similarity_threshold = 0.5
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # 领域特定词汇
        self.domain_vocab = {
            'finance': ['investment', 'market', 'risk', 'return', 'portfolio', 'asset'],
            'medicine': ['patient', 'diagnosis', 'treatment', 'disease', 'symptom', 'therapy'],
            'technology': ['algorithm', 'data', 'system', 'performance', 'optimization', 'architecture'],
            'education': ['learning', 'student', 'curriculum', 'assessment', 'knowledge', 'skill'],
            'business': ['strategy', 'customer', 'revenue', 'profit', 'market', 'competition']
        }
        
    def add_concept(self, concept: Concept) -> None:
        """添加概念"""
        self.concepts[concept.id] = concept
        self.domains.add(concept.domain)
        logger.info(f"添加概念: {concept.name} (领域: {concept.domain})")
    
    def add_concepts(self, concepts: List[Concept]) -> None:
        """批量添加概念"""
        for concept in concepts:
            self.add_concept(concept)
    
    def calculate_semantic_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算语义相似度"""
        # 基于嵌入向量的余弦相似度
        if concept1.embedding is not None and concept2.embedding is not None:
            embedding_sim = cosine_similarity([concept1.embedding], [concept2.embedding])[0][0]
        else:
            embedding_sim = 0.0
        
        # 基于文本的TF-IDF相似度
        texts = [f"{concept1.name} {concept1.description}", f"{concept2.name} {concept2.description}"]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            text_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            text_sim = 0.0
        
        # 基于属性的相似度
        attr_sim = self._calculate_attribute_similarity(concept1, concept2)
        
        # 基于关系的相似度
        relation_sim = self._calculate_relation_similarity(concept1, concept2)
        
        # 综合相似度
        total_similarity = (embedding_sim * 0.3 + text_sim * 0.3 + 
                          attr_sim * 0.2 + relation_sim * 0.2)
        
        return max(0.0, min(1.0, total_similarity))
    
    def _calculate_attribute_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算属性相似度"""
        if not concept1.attributes or not concept2.attributes:
            return 0.0
        
        attrs1 = set(concept1.attributes.keys())
        attrs2 = set(concept2.attributes.keys())
        
        if not attrs1 or not attrs2:
            return 0.0
        
        # Jaccard相似度
        intersection = attrs1 & attrs2
        union = attrs1 | attrs2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_relation_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算关系相似度"""
        if not concept1.relations or not concept2.relations:
            return 0.0
        
        relations1 = set()
        relations2 = set()
        
        for rel_type, targets in concept1.relations.items():
            relations1.add((rel_type, tuple(sorted(targets))))
        
        for rel_type, targets in concept2.relations.items():
            relations2.add((rel_type, tuple(sorted(targets))))
        
        if not relations1 or not relations2:
            return 0.0
        
        intersection = relations1 & relations2
        union = relations1 | relations2
        
        return len(intersection) / len(union) if union else 0.0
    
    def find_similar_concepts(self, query_concept: Concept, top_k: int = 10) -> List[Tuple[Concept, float]]:
        """查找相似概念"""
        similarities = []
        
        for concept_id, concept in self.concepts.items():
            if concept_id != query_concept.id:
                similarity = self.calculate_semantic_similarity(query_concept, concept)
                similarities.append((concept, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def create_cross_domain_mapping(self, source_domain: str, target_domain: str, 
                                  threshold: float = None) -> List[MappingResult]:
        """创建跨领域概念映射"""
        if threshold is None:
            threshold = self.similarity_threshold
        
        source_concepts = [c for c in self.concepts.values() if c.domain == source_domain]
        target_concepts = [c for c in self.concepts.values() if c.domain == target_domain]
        
        mappings = []
        
        for source_concept in source_concepts:
            best_match = None
            best_similarity = 0.0
            
            for target_concept in target_concepts:
                similarity = self.calculate_semantic_similarity(source_concept, target_concept)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = target_concept
            
            if best_similarity >= threshold:
                mapping = MappingResult(
                    source_concept=source_concept,
                    target_concept=best_match,
                    similarity_score=best_similarity,
                    mapping_type="cross_domain",
                    confidence=best_similarity,
                    reasoning=f"跨领域映射: {source_domain} -> {target_domain}"
                )
                mappings.append(mapping)
        
        self.mappings.extend(mappings)
        logger.info(f"创建跨领域映射: {len(mappings)} 个映射关系")
        
        return mappings
    
    def detect_concept_conflicts(self) -> List[Dict[str, Any]]:
        """检测概念冲突"""
        conflicts = []
        
        # 按领域分组概念
        domain_concepts = defaultdict(list)
        for concept in self.concepts.values():
            domain_concepts[concept.domain].append(concept)
        
        # 检测同领域内的冲突
        for domain, concepts in domain_concepts.items():
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    similarity = self.calculate_semantic_similarity(concept1, concept2)
                    
                    if similarity > 0.8:  # 高相似度可能是重复概念
                        conflicts.append({
                            'type': 'potential_duplicate',
                            'domain': domain,
                            'concept1': concept1,
                            'concept2': concept2,
                            'similarity': similarity,
                            'description': f"同领域内高相似度概念: {concept1.name} vs {concept2.name}"
                        })
        
        # 检测映射冲突
        mapping_conflicts = self._detect_mapping_conflicts()
        conflicts.extend(mapping_conflicts)
        
        logger.info(f"检测到 {len(conflicts)} 个概念冲突")
        
        return conflicts
    
    def _detect_mapping_conflicts(self) -> List[Dict[str, Any]]:
        """检测映射冲突"""
        conflicts = []
        
        # 按源概念分组映射
        source_mappings = defaultdict(list)
        for mapping in self.mappings:
            source_mappings[mapping.source_concept.id].append(mapping)
        
        # 检测一对多映射冲突
        for source_id, mappings in source_mappings.items():
            if len(mappings) > 1:
                # 检查是否应该是一对一映射
                high_similarity_mappings = [m for m in mappings if m.similarity_score > 0.7]
                
                if len(high_similarity_mappings) > 1:
                    conflicts.append({
                        'type': 'one_to_many_conflict',
                        'source_concept': mappings[0].source_concept,
                        'mappings': high_similarity_mappings,
                        'description': f"源概念 {mappings[0].source_concept.name} 有多个高相似度映射"
                    })
        
        return conflicts
    
    def validate_mappings(self) -> Dict[str, Any]:
        """验证映射质量"""
        if not self.mappings:
            return {
                'total_mappings': 0,
                'high_confidence_mappings': 0,
                'medium_confidence_mappings': 0,
                'low_confidence_mappings': 0,
                'cross_domain_mappings': 0,
                'average_similarity': 0.0,
                'quality_score': 0.0,
                'domain_pairs': 0
            }
        
        validation_results = {
            'total_mappings': len(self.mappings),
            'high_confidence_mappings': 0,
            'medium_confidence_mappings': 0,
            'low_confidence_mappings': 0,
            'cross_domain_mappings': 0,
            'average_similarity': 0.0,
            'quality_score': 0.0,
            'domain_pairs': 0
        }
        
        # 统计映射质量
        similarities = []
        domain_pairs = set()
        
        for mapping in self.mappings:
            similarities.append(mapping.similarity_score)
            
            if mapping.similarity_score >= 0.8:
                validation_results['high_confidence_mappings'] += 1
            elif mapping.similarity_score >= 0.6:
                validation_results['medium_confidence_mappings'] += 1
            else:
                validation_results['low_confidence_mappings'] += 1
            
            if mapping.mapping_type == 'cross_domain':
                validation_results['cross_domain_mappings'] += 1
            
            domain_pair = (mapping.source_concept.domain, mapping.target_concept.domain)
            domain_pairs.add(domain_pair)
        
        # 计算平均相似度
        validation_results['average_similarity'] = np.mean(similarities) if similarities else 0.0
        
        # 计算质量分数
        quality_score = (
            validation_results['high_confidence_mappings'] * 1.0 +
            validation_results['medium_confidence_mappings'] * 0.7 +
            validation_results['low_confidence_mappings'] * 0.3
        ) / validation_results['total_mappings']
        
        validation_results['quality_score'] = quality_score
        validation_results['domain_pairs'] = len(domain_pairs)
        
        logger.info(f"映射验证完成，质量分数: {quality_score:.2f}")
        
        return validation_results
    
    def optimize_mappings(self, strategy: str = 'quality') -> List[MappingResult]:
        """优化映射结果"""
        if not self.mappings:
            return []
        
        optimized_mappings = []
        
        if strategy == 'quality':
            # 基于质量优化：保留高相似度映射
            optimized_mappings = [m for m in self.mappings if m.similarity_score >= 0.6]
        elif strategy == 'coverage':
            # 基于覆盖率优化：确保每个概念都有映射
            mapped_concepts = set()
            for mapping in self.mappings:
                mapped_concepts.add(mapping.source_concept.id)
                mapped_concepts.add(mapping.target_concept.id)
            
            all_concepts = set(self.concepts.keys())
            unmapped_concepts = all_concepts - mapped_concepts
            
            # 为未映射的概念寻找最佳映射
            for unmapped_id in unmapped_concepts:
                unmapped_concept = self.concepts[unmapped_id]
                similar_concepts = self.find_similar_concepts(unmapped_concept, top_k=1)
                
                if similar_concepts:
                    target_concept, similarity = similar_concepts[0]
                    if similarity >= self.similarity_threshold:
                        mapping = MappingResult(
                            source_concept=unmapped_concept,
                            target_concept=target_concept,
                            similarity_score=similarity,
                            mapping_type='coverage_optimized',
                            confidence=similarity
                        )
                        optimized_mappings.append(mapping)
            
            # 添加原有高质量映射
            optimized_mappings.extend([m for m in self.mappings if m.similarity_score >= 0.5])
        
        # 去重
        unique_mappings = []
        seen_pairs = set()
        for mapping in optimized_mappings:
            pair = (mapping.source_concept.id, mapping.target_concept.id)
            if pair not in seen_pairs:
                unique_mappings.append(mapping)
                seen_pairs.add(pair)
        
        self.mappings = unique_mappings
        logger.info(f"映射优化完成，保留 {len(unique_mappings)} 个映射")
        
        return unique_mappings
    
    def create_hierarchy_mapping(self) -> Dict[str, Any]:
        """创建概念层次映射"""
        hierarchy = {
            'domains': {},
            'concepts_by_level': defaultdict(list),
            'hierarchical_mappings': []
        }
        
        # 按领域组织概念
        for domain in self.domains:
            domain_concepts = [c for c in self.concepts.values() if c.domain == domain]
            
            # 基于相似度创建层次结构
            if len(domain_concepts) > 1:
                # 计算概念间相似度矩阵
                concept_list = list(domain_concepts)
                similarity_matrix = np.zeros((len(concept_list), len(concept_list)))
                
                for i, c1 in enumerate(concept_list):
                    for j, c2 in enumerate(concept_list):
                        if i != j:
                            similarity_matrix[i][j] = self.calculate_semantic_similarity(c1, c2)
                
                # 层次聚类
                try:
                    n_clusters = min(5, len(concept_list) // 2 + 1)  # 动态确定聚类数
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(similarity_matrix)
                    
                    # 组织层次结构
                    clusters = defaultdict(list)
                    for i, label in enumerate(cluster_labels):
                        clusters[label].append(concept_list[i])
                    
                    hierarchy['domains'][domain] = {
                        'clusters': dict(clusters),
                        'cluster_centers': kmeans.cluster_centers_,
                        'total_concepts': len(domain_concepts)
                    }
                    
                    # 创建层次映射
                    for cluster_id, cluster_concepts in clusters.items():
                        if len(cluster_concepts) > 1:
                            # 选择聚类中心概念
                            center_idx = np.argmin([
                                np.mean([similarity_matrix[i][j] for j in range(len(concept_list)) 
                                       if j != i])
                                for i in range(len(concept_list))
                                if concept_list[i] in cluster_concepts
                            ])
                            
                            center_concept = [c for c in concept_list if c in cluster_concepts][center_idx]
                            
                            for concept in cluster_concepts:
                                if concept != center_concept:
                                    similarity = self.calculate_semantic_similarity(concept, center_concept)
                                    mapping = MappingResult(
                                        source_concept=concept,
                                        target_concept=center_concept,
                                        similarity_score=similarity,
                                        mapping_type='hierarchical',
                                        confidence=similarity,
                                        reasoning=f"层次映射: {concept.name} -> {center_concept.name}"
                                    )
                                    hierarchy['hierarchical_mappings'].append(mapping)
                
                except Exception as e:
                    logger.warning(f"层次聚类失败: {e}")
                    hierarchy['domains'][domain] = {'error': str(e)}
        
        self.mappings.extend(hierarchy['hierarchical_mappings'])
        logger.info(f"创建层次映射: {len(hierarchy['hierarchical_mappings'])} 个")
        
        return hierarchy
    
    def visualize_mappings(self, output_path: str = "concept_mappings.png") -> None:
        """可视化概念映射"""
        if not self.mappings:
            logger.warning("没有映射可可视化")
            return
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for concept in self.concepts.values():
            G.add_node(concept.id, 
                      name=concept.name, 
                      domain=concept.domain,
                      label=f"{concept.name}\n({concept.domain})")
        
        # 添加边（映射关系）
        for mapping in self.mappings:
            if mapping.similarity_score >= 0.3:  # 只显示相似度较高的映射
                G.add_edge(
                    mapping.source_concept.id,
                    mapping.target_concept.id,
                    weight=mapping.similarity_score,
                    label=f"{mapping.similarity_score:.2f}"
                )
        
        # 创建可视化
        plt.figure(figsize=(20, 15))
        
        # 设置布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 按领域着色
        domains = list(self.domains)
        colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
        domain_colors = dict(zip(domains, colors))
        
        # 绘制节点
        for domain in domains:
            domain_nodes = [node for node, data in G.nodes(data=True) 
                          if data.get('domain') == domain]
            nx.draw_networkx_nodes(G, pos, nodelist=domain_nodes,
                                 node_color=[domain_colors[domain]] * len(domain_nodes),
                                 node_size=1000, alpha=0.8)
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w * 3 for w in weights], 
                             alpha=0.6, edge_color='gray')
        
        # 绘制标签
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # 绘制边标签
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        plt.title("概念映射网络图", fontsize=16, fontweight='bold')
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=domain_colors[domain], 
                                    markersize=10, label=domain)
                         for domain in domains]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"映射可视化已保存到: {output_path}")
    
    def visualize_similarity_matrix(self, output_path: str = "similarity_matrix.png") -> None:
        """可视化相似度矩阵"""
        if len(self.concepts) < 2:
            logger.warning("概念数量不足，无法生成相似度矩阵")
            return
        
        concept_list = list(self.concepts.values())
        n = len(concept_list)
        similarity_matrix = np.zeros((n, n))
        
        # 计算相似度矩阵
        for i, concept1 in enumerate(concept_list):
            for j, concept2 in enumerate(concept_list):
                if i != j:
                    similarity_matrix[i][j] = self.calculate_semantic_similarity(concept1, concept2)
        
        # 创建标签
        labels = [f"{c.name}\n({c.domain})" for c in concept_list]
        
        # 可视化
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, 
                   xticklabels=labels, 
                   yticklabels=labels,
                   annot=True, 
                   fmt='.2f',
                   cmap='viridis',
                   square=True)
        
        plt.title("概念相似度矩阵", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"相似度矩阵已保存到: {output_path}")
    
    def generate_mapping_report(self) -> Dict[str, Any]:
        """生成映射报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_concepts': len(self.concepts),
                'total_mappings': len(self.mappings),
                'domains': list(self.domains),
                'mapping_types': Counter([m.mapping_type for m in self.mappings])
            },
            'quality_metrics': self.validate_mappings(),
            'conflict_analysis': {
                'conflicts_detected': len(self.detect_concept_conflicts()),
                'conflict_types': Counter([c['type'] for c in self.detect_concept_conflicts()])
            },
            'domain_statistics': {},
            'top_similarities': []
        }
        
        # 领域统计
        for domain in self.domains:
            domain_concepts = [c for c in self.concepts.values() if c.domain == domain]
            domain_mappings = [m for m in self.mappings 
                             if m.source_concept.domain == domain or m.target_concept.domain == domain]
            
            report['domain_statistics'][domain] = {
                'concept_count': len(domain_concepts),
                'mapping_count': len(domain_mappings),
                'average_similarity': np.mean([m.similarity_score for m in domain_mappings]) if domain_mappings else 0.0
            }
        
        # 最高相似度映射
        sorted_mappings = sorted(self.mappings, key=lambda x: x.similarity_score, reverse=True)
        report['top_similarities'] = [
            {
                'source': m.source_concept.name,
                'target': m.target_concept.name,
                'similarity': m.similarity_score,
                'type': m.mapping_type
            }
            for m in sorted_mappings[:10]
        ]
        
        logger.info("映射报告生成完成")
        
        return report
    
    def save_state(self, filepath: str) -> None:
        """保存状态"""
        state = {
            'concepts': {k: {
                'id': v.id,
                'name': v.name,
                'description': v.description,
                'domain': v.domain,
                'attributes': v.attributes,
                'relations': v.relations,
                'confidence': v.confidence
            } for k, v in self.concepts.items()},
            'mappings': [
                {
                    'source_id': m.source_concept.id,
                    'target_id': m.target_concept.id,
                    'similarity_score': m.similarity_score,
                    'mapping_type': m.mapping_type,
                    'confidence': m.confidence,
                    'reasoning': m.reasoning,
                    'metadata': m.metadata
                }
                for m in self.mappings
            ],
            'domains': list(self.domains),
            'similarity_threshold': self.similarity_threshold
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"状态已保存到: {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """加载状态"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # 重建概念
        self.concepts = {}
        for concept_data in state['concepts'].values():
            concept = Concept(**concept_data)
            self.concepts[concept.id] = concept
        
        # 重建映射
        self.mappings = []
        for mapping_data in state['mappings']:
            source_concept = self.concepts[mapping_data['source_id']]
            target_concept = self.concepts[mapping_data['target_id']]
            
            mapping = MappingResult(
                source_concept=source_concept,
                target_concept=target_concept,
                similarity_score=mapping_data['similarity_score'],
                mapping_type=mapping_data['mapping_type'],
                confidence=mapping_data['confidence'],
                reasoning=mapping_data['reasoning'],
                metadata=mapping_data['metadata']
            )
            self.mappings.append(mapping)
        
        self.domains = set(state['domains'])
        self.similarity_threshold = state['similarity_threshold']
        
        logger.info(f"状态已从 {filepath} 加载")

# 使用示例和测试函数
def create_sample_concepts() -> List[Concept]:
    """创建示例概念"""
    concepts = [
        # 金融领域概念
        Concept("fin_001", "投资组合", "包含多种投资资产的组合", "finance"),
        Concept("fin_002", "风险管理", "识别和评估投资风险的过程", "finance"),
        Concept("fin_003", "资产配置", "将资金分配到不同资产类别的策略", "finance"),
        
        # 医学领域概念
        Concept("med_001", "患者管理", "对患者进行全面照顾的过程", "medicine"),
        Concept("med_002", "诊断流程", "确定疾病类型和严重程度的过程", "medicine"),
        Concept("med_003", "治疗方案", "针对特定疾病的治疗计划", "medicine"),
        
        # 技术领域概念
        Concept("tech_001", "系统架构", "软件系统的整体设计和组织", "technology"),
        Concept("tech_002", "性能优化", "提高系统运行效率的方法", "technology"),
        Concept("tech_003", "数据结构", "组织和存储数据的方式", "technology"),
        
        # 教育领域概念
        Concept("edu_001", "课程设计", "制定教学内容和方法的计划", "education"),
        Concept("edu_002", "学习评估", "测量学习成果的过程", "education"),
        Concept("edu_003", "知识管理", "获取、组织和应用知识的方法", "education"),
    ]
    
    return concepts

def demo_concept_mapping():
    """演示概念映射功能"""
    print("=== C7概念映射器演示 ===\n")
    
    # 创建概念映射器
    mapper = ConceptMapper()
    
    # 添加示例概念
    concepts = create_sample_concepts()
    mapper.add_concepts(concepts)
    
    print(f"已添加 {len(concepts)} 个概念")
    print(f"涉及领域: {mapper.domains}\n")
    
    # 1. 概念相似度计算
    print("1. 概念相似度计算示例:")
    concept1 = mapper.concepts["fin_001"]  # 投资组合
    concept2 = mapper.concepts["tech_001"]  # 系统架构
    similarity = mapper.calculate_semantic_similarity(concept1, concept2)
    print(f"   投资组合 vs 系统架构: {similarity:.3f}")
    
    concept3 = mapper.concepts["fin_002"]  # 风险管理
    concept4 = mapper.concepts["med_002"]  # 诊断流程
    similarity2 = mapper.calculate_semantic_similarity(concept3, concept4)
    print(f"   风险管理 vs 诊断流程: {similarity2:.3f}\n")
    
    # 2. 跨领域概念映射
    print("2. 跨领域概念映射:")
    cross_domain_mappings = mapper.create_cross_domain_mapping("finance", "technology")
    for mapping in cross_domain_mappings[:3]:  # 显示前3个
        print(f"   {mapping.source_concept.name} -> {mapping.target_concept.name} "
              f"(相似度: {mapping.similarity_score:.3f})")
    print()
    
    # 3. 概念冲突检测
    print("3. 概念冲突检测:")
    conflicts = mapper.detect_concept_conflicts()
    if conflicts:
        for conflict in conflicts[:2]:  # 显示前2个
            print(f"   类型: {conflict['type']}")
            print(f"   描述: {conflict['description']}")
    else:
        print("   未检测到概念冲突")
    print()
    
    # 4. 映射验证
    print("4. 映射质量验证:")
    validation = mapper.validate_mappings()
    print(f"   总映射数: {validation['total_mappings']}")
    print(f"   高质量映射: {validation['high_confidence_mappings']}")
    print(f"   平均相似度: {validation['average_similarity']:.3f}")
    print(f"   质量分数: {validation['quality_score']:.3f}\n")
    
    # 5. 映射优化
    print("5. 映射优化:")
    optimized_mappings = mapper.optimize_mappings(strategy='quality')
    print(f"   优化后映射数: {len(optimized_mappings)}\n")
    
    # 6. 层次映射
    print("6. 创建层次映射:")
    hierarchy = mapper.create_hierarchy_mapping()
    print(f"   领域数: {len(hierarchy['domains'])}")
    print(f"   层次映射数: {len(hierarchy['hierarchical_mappings'])}\n")
    
    # 7. 生成报告
    print("7. 生成映射报告:")
    report = mapper.generate_mapping_report()
    print(f"   报告生成时间: {report['timestamp']}")
    print(f"   质量指标: {list(report['quality_metrics'].keys())}\n")
    
    # 8. 可视化
    print("8. 生成可视化图表:")
    mapper.visualize_mappings("/workspace/concept_mappings.png")
    mapper.visualize_similarity_matrix("/workspace/similarity_matrix.png")
    print("   图表已保存\n")
    
    # 9. 保存状态
    print("9. 保存状态:")
    mapper.save_state("/workspace/concept_mapper_state.json")
    print("   状态已保存\n")
    
    print("=== 演示完成 ===")
    return mapper

if __name__ == "__main__":
    # 运行演示
    mapper = demo_concept_mapping()
    
    # 保存演示结果
    print("保存演示结果...")
    
    # 生成详细报告
    detailed_report = mapper.generate_mapping_report()
    with open("/workspace/concept_mapping_report.json", 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, ensure_ascii=False, indent=2)
    
    print("演示结果已保存到 /workspace/concept_mapping_report.json")