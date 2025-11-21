"""
C3抽象学习器 - 抽象概念提取和建模系统
实现抽象概念提取、层次结构构建、关系映射和推理功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict, deque
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Concept:
    """概念数据结构"""
    id: str
    name: str
    attributes: Dict[str, Any]
    instances: List[Any]
    level: int
    parent: Optional[str] = None
    children: List[str] = None
    similarity_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.similarity_scores is None:
            self.similarity_scores = {}


@dataclass
class ConceptRelation:
    """概念关系数据结构"""
    source_concept: str
    target_concept: str
    relation_type: str  # 'is-a', 'part-of', 'similar-to', 'causes', etc.
    strength: float
    confidence: float


class AbstractLearner:
    """抽象学习器主类"""
    
    def __init__(self, 
                 clustering_method: str = 'kmeans',
                 similarity_threshold: float = 0.7,
                 max_levels: int = 5,
                 min_concept_size: int = 3):
        """
        初始化抽象学习器
        
        Args:
            clustering_method: 聚类方法 ('kmeans', 'dbscan', 'hierarchical')
            similarity_threshold: 相似度阈值
            max_levels: 最大概念层次
            min_concept_size: 最小概念实例数
        """
        self.clustering_method = clustering_method
        self.similarity_threshold = similarity_threshold
        self.max_levels = max_levels
        self.min_concept_size = min_concept_size
        
        # 核心数据结构
        self.concepts: Dict[str, Concept] = {}
        self.relations: List[ConceptRelation] = []
        self.hierarchy: Dict[int, Set[str]] = defaultdict(set)
        self.raw_data: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        
        # 学习状态
        self.is_trained = False
        self.learning_history: List[Dict] = []
        
        # 可视化工具
        self.visualizer = ConceptVisualizer()
    
    def extract_concepts(self, data: np.ndarray, feature_names: List[str] = None) -> Dict[str, Concept]:
        """
        使用无监督学习算法提取抽象概念
        
        Args:
            data: 输入数据 [n_samples, n_features]
            feature_names: 特征名称列表
            
        Returns:
            提取的概念字典
        """
        self.raw_data = data
        self.feature_names = feature_names or [f'feature_{i}' for i in range(data.shape[1])]
        
        print("开始概念提取...")
        
        # 数据预处理
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # 确定最优聚类数
        optimal_clusters = self._find_optimal_clusters(scaled_data)
        print(f"确定最优聚类数: {optimal_clusters}")
        
        # 执行聚类
        if self.clustering_method == 'kmeans':
            clusters = self._kmeans_clustering(scaled_data, optimal_clusters)
        elif self.clustering_method == 'dbscan':
            clusters = self._dbscan_clustering(scaled_data)
        elif self.clustering_method == 'hierarchical':
            clusters = self._hierarchical_clustering(scaled_data, optimal_clusters)
        else:
            raise ValueError(f"不支持的聚类方法: {self.clustering_method}")
        
        # 构建概念
        concepts = self._build_concepts_from_clusters(clusters, scaled_data)
        
        print(f"提取了 {len(concepts)} 个概念")
        return concepts
    
    def build_hierarchy(self, concepts: Dict[str, Concept]) -> Dict[int, Set[str]]:
        """
        构建概念层次结构
        
        Args:
            concepts: 概念字典
            
        Returns:
            层次结构字典 {level: {concept_ids}}
        """
        print("构建概念层次结构...")
        
        self.concepts = concepts
        self.hierarchy.clear()
        
        # 初始化层次结构
        for concept_id, concept in concepts.items():
            self.hierarchy[concept.level].add(concept_id)
        
        # 构建父子关系
        self._build_parent_child_relations()
        
        # 计算概念相似度
        self._calculate_concept_similarities()
        
        print(f"构建了 {len(self.hierarchy)} 层概念层次")
        return self.hierarchy
    
    def map_relations(self) -> List[ConceptRelation]:
        """
        映射概念间的关系
        
        Returns:
            概念关系列表
        """
        print("映射概念关系...")
        
        self.relations.clear()
        
        # 1. 层次关系 (is-a)
        self._map_hierarchical_relations()
        
        # 2. 相似关系 (similar-to)
        self._map_similarity_relations()
        
        # 3. 属性关系 (has-property)
        self._map_attribute_relations()
        
        # 4. 实例关系 (instance-of)
        self._map_instance_relations()
        
        print(f"映射了 {len(self.relations)} 个关系")
        return self.relations
    
    def reason(self, query: str, context: Dict = None) -> Any:
        """
        基于概念关系进行推理
        
        Args:
            query: 查询字符串
            context: 上下文信息
            
        Returns:
            推理结果
        """
        print(f"执行推理查询: {query}")
        
        # 解析查询类型
        if "相似" in query or "similar" in query.lower():
            return self._reason_similarity(query, context)
        elif "层次" in query or "hierarchy" in query.lower():
            return self._reason_hierarchy(query, context)
        elif "属性" in query or "attribute" in query.lower():
            return self._reason_attributes(query, context)
        else:
            return self._general_reasoning(query, context)
    
    def generalize(self, concept_id: str, target_level: int = None) -> Concept:
        """
        概念泛化
        
        Args:
            concept_id: 源概念ID
            target_level: 目标层次（如果为None则泛化到上一层）
            
        Returns:
            泛化后的概念
        """
        if concept_id not in self.concepts:
            raise ValueError(f"概念 {concept_id} 不存在")
        
        concept = self.concepts[concept_id]
        target_level = target_level or (concept.level + 1)
        
        if target_level >= self.max_levels:
            raise ValueError(f"目标层次 {target_level} 超过最大层次 {self.max_levels}")
        
        print(f"将概念 {concept_id} 泛化到层次 {target_level}")
        
        # 收集所有子概念
        all_children = self._collect_all_children(concept_id)
        all_instances = []
        all_attributes = defaultdict(list)
        
        for child_id in all_children:
            child = self.concepts[child_id]
            all_instances.extend(child.instances)
            for attr, value in child.attributes.items():
                all_attributes[attr].append(value)
        
        # 合并属性
        generalized_attributes = {}
        for attr, values in all_attributes.items():
            if isinstance(values[0], (int, float)):
                generalized_attributes[attr] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                # 分类属性，取最常见的值
                value_counts = pd.Series(values).value_counts()
                generalized_attributes[attr] = {
                    'mode': value_counts.index[0],
                    'distribution': value_counts.to_dict()
                }
        
        # 创建泛化概念
        generalized_concept = Concept(
            id=f"{concept_id}_generalized_{target_level}",
            name=f"泛化_{concept.name}",
            attributes=generalized_attributes,
            instances=all_instances,
            level=target_level,
            parent=concept.parent,
            children=[concept_id] + all_children
        )
        
        self.concepts[generalized_concept.id] = generalized_concept
        self.hierarchy[target_level].add(generalized_concept.id)
        
        return generalized_concept
    
    def specialize(self, concept_id: str, specialization_criteria: Dict) -> List[Concept]:
        """
        概念特化
        
        Args:
            concept_id: 源概念ID
            specialization_criteria: 特化标准
            
        Returns:
            特化后的概念列表
        """
        if concept_id not in self.concepts:
            raise ValueError(f"概念 {concept_id} 不存在")
        
        concept = self.concepts[concept_id]
        print(f"将概念 {concept_id} 基于标准 {specialization_criteria} 进行特化")
        
        specialized_concepts = []
        
        # 基于属性值进行特化
        for attr, criteria in specialization_criteria.items():
            if attr in concept.attributes:
                # 找到符合特定条件的实例
                matching_instances = []
                for instance in concept.instances:
                    if self._matches_criteria(instance, attr, criteria):
                        matching_instances.append(instance)
                
                if len(matching_instances) >= self.min_concept_size:
                    # 创建特化概念
                    specialized_concept = Concept(
                        id=f"{concept.id}_specialized_{attr}_{criteria}",
                        name=f"特化_{concept.name}_{attr}_{criteria}",
                        attributes={attr: criteria},
                        instances=matching_instances,
                        level=concept.level - 1,
                        parent=concept_id
                    )
                    
                    specialized_concepts.append(specialized_concept)
                    self.concepts[specialized_concept.id] = specialized_concept
                    self.hierarchy[specialized_concept.level].add(specialized_concept.id)
        
        return specialized_concepts
    
    def calculate_similarity(self, concept1_id: str, concept2_id: str) -> float:
        """
        计算两个概念间的相似度
        
        Args:
            concept1_id: 概念1 ID
            concept2_id: 概念2 ID
            
        Returns:
            相似度分数 [0, 1]
        """
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            raise ValueError("概念不存在")
        
        concept1 = self.concepts[concept1_id]
        concept2 = self.concepts[concept2_id]
        
        # 1. 属性相似度
        attr_similarity = self._calculate_attribute_similarity(concept1, concept2)
        
        # 2. 层次距离相似度
        hierarchy_similarity = self._calculate_hierarchy_similarity(concept1, concept2)
        
        # 3. 实例重叠相似度
        instance_similarity = self._calculate_instance_similarity(concept1, concept2)
        
        # 综合相似度
        total_similarity = (attr_similarity * 0.4 + 
                          hierarchy_similarity * 0.3 + 
                          instance_similarity * 0.3)
        
        return min(1.0, max(0.0, total_similarity))
    
    def apply_knowledge(self, new_data: np.ndarray) -> Dict[str, Any]:
        """
        应用抽象知识到新数据
        
        Args:
            new_data: 新数据 [n_samples, n_features] 或 [n_features]
            
        Returns:
            分类和推理结果
        """
        print("应用抽象知识到新数据...")
        
        results = {
            'classifications': {},
            'predictions': {},
            'reasoning': []
        }
        
        # 确保数据格式正确
        if new_data.ndim == 1:
            new_data = new_data.reshape(1, -1)
        
        # 对每个数据点进行概念匹配
        for i, data_point in enumerate(new_data):
            point_classifications = {}
            
            for concept_id, concept in self.concepts.items():
                similarity_score = self._calculate_data_concept_similarity(data_point, concept)
                if similarity_score > self.similarity_threshold:
                    point_classifications[concept_id] = similarity_score
            
            if point_classifications:
                best_match = max(point_classifications.items(), key=lambda x: x[1])
                concept_id, score = best_match
                results['classifications'][f'point_{i}'] = {
                    'concept': concept_id,
                    'confidence': score,
                    'all_matches': point_classifications
                }
        
        # 基于分类结果进行推理
        if results['classifications']:
            results['reasoning'].append({
                'type': 'classification_summary',
                'total_classified': len(results['classifications']),
                'concepts_used': list(set([v['concept'] for v in results['classifications'].values()])),
                'explanation': f"对 {len(new_data)} 个数据点进行了概念匹配"
            })
        
        return results
    
    def validate_knowledge(self, test_data: np.ndarray, ground_truth: List[str] = None) -> Dict[str, float]:
        """
        验证抽象知识的有效性
        
        Args:
            test_data: 测试数据
            ground_truth: 真实标签（可选）
            
        Returns:
            验证指标
        """
        print("验证抽象知识...")
        
        predictions = []
        confidences = []
        
        for data_point in test_data:
            result = self.apply_knowledge(data_point.reshape(1, -1))
            if result['classifications']:
                best_concept = max(result['classifications'].items(), key=lambda x: x[1])
                predictions.append(best_concept[0])
                confidences.append(best_concept[1])
            else:
                predictions.append('unknown')
                confidences.append(0.0)
        
        # 计算验证指标
        metrics = {
            'coverage': len([p for p in predictions if p != 'unknown']) / len(predictions),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'confidence_std': np.std(confidences) if confidences else 0.0
        }
        
        if ground_truth:
            # 计算准确率
            correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
            metrics['accuracy'] = correct / len(ground_truth)
        
        return metrics
    
    def visualize_concepts(self, save_path: str = None, figsize: Tuple[int, int] = (15, 10)):
        """
        可视化概念层次结构
        
        Args:
            save_path: 保存路径
            figsize: 图像大小
        """
        print("生成概念可视化...")
        
        try:
            # 设置matplotlib后端
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('C3抽象学习器 - 概念分析可视化', fontsize=16)
            
            # 1. 概念层次结构
            ax1 = axes[0, 0]
            self.visualizer.plot_hierarchy(self.hierarchy, ax1)
            ax1.set_title('概念层次结构')
            
            # 2. 概念相似度矩阵
            ax2 = axes[0, 1]
            self.visualizer.plot_similarity_matrix(self.concepts, ax2)
            ax2.set_title('概念相似度矩阵')
            
            # 3. 概念关系网络
            ax3 = axes[1, 0]
            self.visualizer.plot_concept_network(self.concepts, self.relations, ax3)
            ax3.set_title('概念关系网络')
            
            # 4. 概念统计信息
            ax4 = axes[1, 1]
            self.visualizer.plot_concept_statistics(self.concepts, ax4)
            ax4.set_title('概念统计信息')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"可视化结果已保存到: {save_path}")
            
            # 在无GUI环境中不显示图像
            # plt.show()
            
        except Exception as e:
            print(f"可视化生成失败: {e}")
            # 创建简单的文本报告作为备选方案
            self._create_text_visualization(save_path)
    
    def _create_text_visualization(self, save_path: str = None):
        """创建文本形式的可视化报告"""
        report_lines = []
        report_lines.append("C3抽象学习器 - 概念分析报告")
        report_lines.append("=" * 50)
        
        # 概念层次信息
        report_lines.append("\n概念层次结构:")
        for level, concept_ids in self.hierarchy.items():
            report_lines.append(f"  层次 {level}: {len(concept_ids)} 个概念")
            for concept_id in concept_ids:
                concept = self.concepts[concept_id]
                report_lines.append(f"    - {concept.name} (实例数: {len(concept.instances)})")
        
        # 概念相似度信息
        report_lines.append("\n概念相似度矩阵:")
        concept_ids = list(self.concepts.keys())
        if len(concept_ids) > 1:
            # 打印相似度表头
            header = "概念".ljust(15)
            for cid in concept_ids[:10]:  # 只显示前10个概念
                header += cid.split('_')[-1].ljust(8)
            report_lines.append(header)
            
            # 打印相似度数据
            for i, concept1_id in enumerate(concept_ids[:10]):
                row = concept1_id.split('_')[-1].ljust(15)
                for concept2_id in concept_ids[:10]:
                    similarity = self.concepts[concept1_id].similarity_scores.get(concept2_id, 0.0)
                    row += f"{similarity:.2f}".ljust(8)
                report_lines.append(row)
        
        # 关系统计
        report_lines.append(f"\n关系统计: 共 {len(self.relations)} 个关系")
        relation_types = {}
        for relation in self.relations:
            rel_type = relation.relation_type
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        
        for rel_type, count in relation_types.items():
            report_lines.append(f"  {rel_type}: {count} 个")
        
        # 保存报告
        if save_path:
            text_path = save_path.replace('.png', '_report.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"文本报告已保存到: {text_path}")
        
        return '\n'.join(report_lines)
    
    def save_model(self, filepath: str):
        """保存模型到文件"""
        model_data = {
            'concepts': {k: {
                'id': v.id,
                'name': v.name,
                'attributes': v.attributes,
                'instances': v.instances,
                'level': v.level,
                'parent': v.parent,
                'children': v.children
            } for k, v in self.concepts.items()},
            'relations': [{'source': r.source_concept, 'target': r.target_concept, 
                          'type': r.relation_type, 'strength': r.strength, 
                          'confidence': r.confidence} for r in self.relations],
            'hierarchy': {k: list(v) for k, v in self.hierarchy.items()},
            'config': {
                'clustering_method': self.clustering_method,
                'similarity_threshold': self.similarity_threshold,
                'max_levels': self.max_levels,
                'min_concept_size': self.min_concept_size
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """从文件加载模型"""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # 重建概念
        self.concepts = {}
        for k, v in model_data['concepts'].items():
            self.concepts[k] = Concept(
                id=v['id'], name=v['name'], attributes=v['attributes'],
                instances=v['instances'], level=v['level'], parent=v['parent'],
                children=v['children']
            )
        
        # 重建关系
        self.relations = [ConceptRelation(r['source'], r['target'], r['type'], 
                                        r['strength'], r['confidence']) 
                         for r in model_data['relations']]
        
        # 重建层次结构
        self.hierarchy = defaultdict(set)
        for k, v in model_data['hierarchy'].items():
            self.hierarchy[k] = set(v)
        
        # 重建配置
        config = model_data['config']
        self.clustering_method = config['clustering_method']
        self.similarity_threshold = config['similarity_threshold']
        self.max_levels = config['max_levels']
        self.min_concept_size = config['min_concept_size']
        
        self.is_trained = True
        print(f"模型已从 {filepath} 加载")
    
    # ==================== 私有方法 ====================
    
    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """确定最优聚类数"""
        max_clusters = min(20, len(data) // 2)
        
        if self.clustering_method == 'kmeans':
            # 使用肘部法和轮廓系数
            inertias = []
            silhouette_scores = []
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                inertias.append(kmeans.inertia_)
                
                if k < len(data):
                    sil_score = silhouette_score(data, labels)
                    silhouette_scores.append(sil_score)
            
            # 选择轮廓系数最高的k
            if silhouette_scores:
                optimal_k = np.argmax(silhouette_scores) + 2
            else:
                optimal_k = 3
                
        else:
            optimal_k = min(10, len(data) // 3)
        
        return max(2, optimal_k)
    
    def _kmeans_clustering(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """K-means聚类"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(data)
    
    def _dbscan_clustering(self, data: np.ndarray) -> np.ndarray:
        """DBSCAN聚类"""
        # 自动选择eps参数
        distances = pdist(data)
        eps = np.percentile(distances, 10)
        
        dbscan = DBSCAN(eps=eps, min_samples=self.min_concept_size)
        return dbscan.fit_predict(data)
    
    def _hierarchical_clustering(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """层次聚类"""
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return clustering.fit_predict(data)
    
    def _build_concepts_from_clusters(self, clusters: np.ndarray, data: np.ndarray) -> Dict[str, Concept]:
        """从聚类结果构建概念"""
        concepts = {}
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # DBSCAN噪声点
                continue
                
            cluster_mask = clusters == cluster_id
            cluster_data = data[cluster_mask]
            cluster_instances = list(np.where(cluster_mask)[0])
            
            # 计算概念属性
            attributes = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_values = cluster_data[:, i]
                attributes[feature_name] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'min': np.min(feature_values),
                    'max': np.max(feature_values),
                    'count': len(feature_values)
                }
            
            # 创建概念
            concept = Concept(
                id=f"concept_{cluster_id}",
                name=f"概念_{cluster_id}",
                attributes=attributes,
                instances=cluster_instances,
                level=0
            )
            
            concepts[concept.id] = concept
        
        return concepts
    
    def _build_parent_child_relations(self):
        """构建父子关系"""
        for level in range(1, self.max_levels):
            current_level_concepts = list(self.hierarchy[level])
            parent_level_concepts = list(self.hierarchy.get(level - 1, set()))
            
            for child_id in current_level_concepts:
                child = self.concepts[child_id]
                
                # 找到最相似的父概念
                best_parent = None
                best_similarity = 0
                
                for parent_id in parent_level_concepts:
                    parent = self.concepts[parent_id]
                    similarity = self._calculate_concept_similarity(child, parent)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_parent = parent_id
                
                if best_parent and best_similarity > self.similarity_threshold:
                    child.parent = best_parent
                    self.concepts[best_parent].children.append(child_id)
    
    def _calculate_concept_similarities(self):
        """计算概念间相似度"""
        concept_ids = list(self.concepts.keys())
        
        for i, concept1_id in enumerate(concept_ids):
            for concept2_id in concept_ids[i+1:]:
                similarity = self.calculate_similarity(concept1_id, concept2_id)
                self.concepts[concept1_id].similarity_scores[concept2_id] = similarity
                self.concepts[concept2_id].similarity_scores[concept1_id] = similarity
    
    def _map_hierarchical_relations(self):
        """映射层次关系"""
        for concept_id, concept in self.concepts.items():
            if concept.parent:
                relation = ConceptRelation(
                    source_concept=concept.parent,
                    target_concept=concept_id,
                    relation_type='is-a',
                    strength=1.0,
                    confidence=1.0
                )
                self.relations.append(relation)
    
    def _map_similarity_relations(self):
        """映射相似关系"""
        for concept1_id, concept1 in self.concepts.items():
            for concept2_id, similarity in concept1.similarity_scores.items():
                if (similarity > self.similarity_threshold and 
                    concept1_id < concept2_id):  # 避免重复
                    
                    relation = ConceptRelation(
                        source_concept=concept1_id,
                        target_concept=concept2_id,
                        relation_type='similar-to',
                        strength=similarity,
                        confidence=similarity
                    )
                    self.relations.append(relation)
    
    def _map_attribute_relations(self):
        """映射属性关系"""
        for concept_id, concept in self.concepts.items():
            for attr_name in concept.attributes.keys():
                relation = ConceptRelation(
                    source_concept=concept_id,
                    target_concept=attr_name,
                    relation_type='has-property',
                    strength=1.0,
                    confidence=0.8
                )
                self.relations.append(relation)
    
    def _map_instance_relations(self):
        """映射实例关系"""
        for concept_id, concept in self.concepts.items():
            for instance_id in concept.instances:
                relation = ConceptRelation(
                    source_concept=str(instance_id),
                    target_concept=concept_id,
                    relation_type='instance-of',
                    strength=1.0,
                    confidence=1.0
                )
                self.relations.append(relation)
    
    def _calculate_attribute_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算属性相似度"""
        common_attrs = set(concept1.attributes.keys()) & set(concept2.attributes.keys())
        if not common_attrs:
            return 0.0
        
        similarities = []
        for attr in common_attrs:
            attr1 = concept1.attributes[attr]
            attr2 = concept2.attributes[attr]
            
            # 数值属性相似度
            if isinstance(attr1, dict) and 'mean' in attr1:
                mean1, mean2 = attr1['mean'], attr2['mean']
                std1, std2 = attr1.get('std', 0), attr2.get('std', 0)
                
                # 基于均值和方差的相似度
                mean_diff = abs(mean1 - mean2)
                std_sum = std1 + std2 + 1e-8
                attr_sim = max(0, 1 - mean_diff / std_sum)
            else:
                # 分类属性相似度
                attr_sim = 1.0 if attr1 == attr2 else 0.0
            
            similarities.append(attr_sim)
        
        return np.mean(similarities)
    
    def _calculate_hierarchy_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算层次相似度"""
        # 如果在同一层次，相似度高
        if concept1.level == concept2.level:
            return 1.0
        
        # 计算层次距离
        level_diff = abs(concept1.level - concept2.level)
        return max(0, 1 - level_diff / self.max_levels)
    
    def _calculate_instance_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算实例重叠相似度"""
        if not concept1.instances or not concept2.instances:
            return 0.0
        
        overlap = len(set(concept1.instances) & set(concept2.instances))
        union = len(set(concept1.instances) | set(concept2.instances))
        
        return overlap / union if union > 0 else 0.0
    
    def _calculate_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """计算两个概念的综合相似度"""
        return self.calculate_similarity(concept1.id, concept2.id)
    
    def _calculate_data_concept_similarity(self, data: np.ndarray, concept: Concept) -> float:
        """计算数据点与概念的相似度"""
        if not concept.instances:
            return 0.0
        
        # 计算数据点与概念实例的平均距离
        concept_data = self.raw_data[concept.instances]
        distances = np.sqrt(np.sum((data - concept_data) ** 2, axis=1))
        
        # 转换为相似度
        max_distance = np.max(distances)
        similarity = 1.0 - (np.mean(distances) / (max_distance + 1e-8))
        
        return max(0.0, min(1.0, similarity))
    
    def _collect_all_children(self, concept_id: str) -> List[str]:
        """递归收集所有子概念"""
        children = []
        concept = self.concepts[concept_id]
        
        for child_id in concept.children:
            children.append(child_id)
            children.extend(self._collect_all_children(child_id))
        
        return children
    
    def _matches_criteria(self, instance: Any, attr: str, criteria: Any) -> bool:
        """检查实例是否匹配标准"""
        # 这里需要根据具体的数据结构实现
        # 简化实现
        return True
    
    def _reason_similarity(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """相似性推理"""
        # 提取查询中的概念
        result = {
            'type': 'similarity_reasoning',
            'similar_concepts': [],
            'reasoning_process': []
        }
        
        for concept_id, concept in self.concepts.items():
            similarities = concept.similarity_scores
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            result['similar_concepts'].append({
                'concept': concept_id,
                'most_similar': sorted_similarities[:3]
            })
        
        return result
    
    def _reason_hierarchy(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """层次推理"""
        result = {
            'type': 'hierarchy_reasoning',
            'hierarchy_structure': dict(self.hierarchy),
            'reasoning_process': []
        }
        
        return result
    
    def _reason_attributes(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """属性推理"""
        result = {
            'type': 'attribute_reasoning',
            'attribute_analysis': {},
            'reasoning_process': []
        }
        
        for concept_id, concept in self.concepts.items():
            result['attribute_analysis'][concept_id] = concept.attributes
        
        return result
    
    def _general_reasoning(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """一般推理"""
        result = {
            'type': 'general_reasoning',
            'query': query,
            'available_concepts': list(self.concepts.keys()),
            'available_relations': len(self.relations),
            'reasoning_process': ['基于概念映射进行推理']
        }
        
        return result


class ConceptVisualizer:
    """概念可视化工具类"""
    
    def plot_hierarchy(self, hierarchy: Dict[int, Set[str]], ax):
        """绘制层次结构"""
        levels = list(hierarchy.keys())
        concept_counts = [len(hierarchy[level]) for level in levels]
        
        ax.bar(levels, concept_counts, alpha=0.7, color='skyblue')
        ax.set_xlabel('概念层次')
        ax.set_ylabel('概念数量')
        ax.set_title('概念层次分布')
        
        # 添加数值标签
        for i, v in enumerate(concept_counts):
            ax.text(levels[i], v + 0.1, str(v), ha='center', va='bottom')
    
    def plot_similarity_matrix(self, concepts: Dict[str, Concept], ax):
        """绘制相似度矩阵"""
        concept_ids = list(concepts.keys())
        n_concepts = len(concept_ids)
        
        if n_concepts == 0:
            ax.text(0.5, 0.5, '无概念数据', ha='center', va='center', transform=ax.transAxes)
            return
        
        similarity_matrix = np.zeros((n_concepts, n_concepts))
        
        for i, concept1_id in enumerate(concept_ids):
            for j, concept2_id in enumerate(concept_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = concepts[concept1_id].similarity_scores.get(concept2_id, 0.0)
        
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(n_concepts))
        ax.set_yticks(range(n_concepts))
        ax.set_xticklabels([f'C{i}' for i in range(n_concepts)], rotation=45)
        ax.set_yticklabels([f'C{i}' for i in range(n_concepts)])
        ax.set_title('概念相似度矩阵')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def plot_concept_network(self, concepts: Dict[str, Concept], relations: List[ConceptRelation], ax):
        """绘制概念网络图"""
        G = nx.DiGraph()
        
        # 添加节点
        for concept_id in concepts.keys():
            G.add_node(concept_id)
        
        # 添加边
        for relation in relations:
            if (relation.source_concept in concepts and 
                relation.target_concept in concepts):
                G.add_edge(relation.source_concept, relation.target_concept, 
                          weight=relation.strength, label=relation.relation_type)
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, '无网络数据', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 布局
        pos = nx.spring_layout(G, seed=42)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                              node_size=500, alpha=0.7)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                              alpha=0.6, arrows=True, arrowsize=10)
        
        # 绘制标签
        labels = {node: node.split('_')[-1] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        ax.set_title('概念关系网络')
        ax.axis('off')
    
    def plot_concept_statistics(self, concepts: Dict[str, Concept], ax):
        """绘制概念统计信息"""
        if not concepts:
            ax.text(0.5, 0.5, '无统计数据', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 统计信息
        levels = [concept.level for concept in concepts.values()]
        instance_counts = [len(concept.instances) for concept in concepts.values()]
        
        # 创建子图
        fig = ax.get_figure()
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 层次分布
        ax1 = fig.add_subplot(gs[0, 0])
        level_counts = pd.Series(levels).value_counts().sort_index()
        ax1.bar(level_counts.index, level_counts.values, alpha=0.7, color='lightcoral')
        ax1.set_title('概念层次分布')
        ax1.set_xlabel('层次')
        ax1.set_ylabel('数量')
        
        # 实例数量分布
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(instance_counts, bins=10, alpha=0.7, color='lightgreen')
        ax2.set_title('概念实例数量分布')
        ax2.set_xlabel('实例数量')
        ax2.set_ylabel('概念数量')
        
        # 概念大小统计
        ax3 = fig.add_subplot(gs[1, :])
        concept_names = [f'C{i}' for i in range(len(concepts))]
        ax3.bar(range(len(concepts)), instance_counts, alpha=0.7, color='gold')
        ax3.set_title('各概念实例数量')
        ax3.set_xlabel('概念')
        ax3.set_ylabel('实例数量')
        ax3.set_xticks(range(len(concepts)))
        ax3.set_xticklabels(concept_names, rotation=45)
        
        # 移除原始轴
        ax.remove()


# 使用示例和测试代码
def demo_abstract_learner():
    """演示抽象学习器的使用"""
    print("=== C3抽象学习器演示 ===")
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 300
    n_features = 4
    
    # 创建三个不同的数据簇
    data1 = np.random.normal([2, 2, 2, 2], 0.5, (100, n_features))
    data2 = np.random.normal([5, 5, 5, 5], 0.7, (100, n_features))
    data3 = np.random.normal([8, 2, 8, 2], 0.6, (100, n_features))
    
    data = np.vstack([data1, data2, data3])
    feature_names = ['特征1', '特征2', '特征3', '特征4']
    
    print(f"生成数据: {data.shape}")
    
    # 初始化学习器
    learner = AbstractLearner(
        clustering_method='kmeans',
        similarity_threshold=0.6,
        max_levels=3,
        min_concept_size=5
    )
    
    # 1. 提取概念
    print("\n1. 提取概念...")
    concepts = learner.extract_concepts(data, feature_names)
    print(f"提取了 {len(concepts)} 个概念")
    
    # 2. 构建层次结构
    print("\n2. 构建层次结构...")
    hierarchy = learner.build_hierarchy(concepts)
    for level, concept_ids in hierarchy.items():
        print(f"  层次 {level}: {len(concept_ids)} 个概念")
    
    # 3. 映射关系
    print("\n3. 映射关系...")
    relations = learner.map_relations()
    print(f"映射了 {len(relations)} 个关系")
    
    # 4. 推理示例
    print("\n4. 推理示例...")
    similarity_result = learner.reason("找出相似的概念")
    print(f"相似性推理结果: {len(similarity_result.get('similar_concepts', []))} 组概念")
    
    # 5. 概念泛化
    print("\n5. 概念泛化...")
    first_concept = list(concepts.keys())[0]
    generalized = learner.generalize(first_concept)
    print(f"泛化概念: {generalized.id}")
    
    # 6. 相似度计算
    print("\n6. 相似度计算...")
    concept_ids = list(concepts.keys())
    if len(concept_ids) >= 2:
        similarity = learner.calculate_similarity(concept_ids[0], concept_ids[1])
        print(f"概念 {concept_ids[0]} 和 {concept_ids[1]} 的相似度: {similarity:.3f}")
    
    # 7. 知识应用
    print("\n7. 知识应用...")
    test_data = np.random.normal([3, 3, 3, 3], 0.3, (10, n_features))
    application_result = learner.apply_knowledge(test_data)
    print(f"应用结果: {len(application_result['classifications'])} 个分类")
    
    # 8. 知识验证
    print("\n8. 知识验证...")
    validation_metrics = learner.validate_knowledge(test_data[:5])
    print(f"验证指标: {validation_metrics}")
    
    # 9. 可视化
    print("\n9. 生成可视化...")
    learner.visualize_concepts(save_path='concept_analysis.png')
    
    # 10. 保存模型
    print("\n10. 保存模型...")
    learner.save_model('abstract_learner_model.json')
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_abstract_learner()