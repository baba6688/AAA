"""
E3价值观形成器
实现价值观识别、建模、冲突检测、一致性检验、演化发展、影响评估和决策指导功能
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValueType(Enum):
    """价值观类型枚举"""
    CORE = "核心价值观"          # 核心价值观
    DERIVED = "衍生价值观"       # 衍生价值观
    SITUATIONAL = "情境价值观"    # 情境价值观
    TEMPORARY = "临时价值观"      # 临时价值观
    CONFLICTING = "冲突价值观"    # 冲突价值观


class ValueLevel(Enum):
    """价值观层级枚举"""
    ABSTRACT = "抽象层"          # 抽象层
    CONCEPTUAL = "概念层"        # 概念层
    CONCRETE = "具体层"          # 具体层
    OPERATIONAL = "操作层"       # 操作层


class ConflictType(Enum):
    """冲突类型枚举"""
    DIRECT = "直接冲突"          # 直接冲突
    INDIRECT = "间接冲突"        # 间接冲突
    HIERARCHICAL = "层级冲突"     # 层级冲突
    TEMPORAL = "时间冲突"        # 时间冲突
    CONTEXTUAL = "情境冲突"      # 情境冲突


@dataclass
class Value:
    """价值观数据类"""
    id: str
    name: str
    description: str
    value_type: ValueType
    level: ValueLevel
    strength: float  # 价值观强度 (0-1)
    priority: int    # 优先级 (1-10)
    context: str     # 适用情境
    source: str      # 价值观来源
    timestamp: datetime
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'value_type': self.value_type.value,
            'level': self.level.value,
            'strength': self.strength,
            'priority': self.priority,
            'context': self.context,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'attributes': self.attributes
        }


@dataclass
class ValueRelationship:
    """价值观关系数据类"""
    source_value_id: str
    target_value_id: str
    relationship_type: str  # 支持、冲突、继承、相关等
    strength: float         # 关系强度 (0-1)
    description: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'source_value_id': self.source_value_id,
            'target_value_id': self.target_value_id,
            'relationship_type': self.relationship_type,
            'strength': self.strength,
            'description': self.description,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ValueConflict:
    """价值观冲突数据类"""
    id: str
    conflict_type: ConflictType
    value_ids: List[str]
    description: str
    severity: float         # 冲突严重程度 (0-1)
    resolution_suggestions: List[str]
    timestamp: datetime
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'conflict_type': self.conflict_type.value,
            'value_ids': self.value_ids,
            'description': self.description,
            'severity': self.severity,
            'resolution_suggestions': self.resolution_suggestions,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved
        }


class ValueFormation:
    """价值观形成器主类"""
    
    def __init__(self, db_path: str = "value_formation.db"):
        """
        初始化价值观形成器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.values: Dict[str, Value] = {}
        self.relationships: List[ValueRelationship] = []
        self.conflicts: List[ValueConflict] = []
        self.value_history: deque = deque(maxlen=1000)  # 价值观演化历史
        self.value_evolution_tracks: Dict[str, List[Dict]] = {}  # 价值观演化轨迹
        
        # 初始化数据库
        self._init_database()
        
        # 价值观识别组件
        self.value_extractor = ValueExtractor()
        
        # 价值观建模组件
        self.value_modeler = ValueModeler()
        
        # 冲突检测组件
        self.conflict_detector = ConflictDetector()
        
        # 一致性检验组件
        self.consistency_checker = ConsistencyChecker()
        
        # 演化追踪组件
        self.evolution_tracker = EvolutionTracker()
        
        # 影响评估组件
        self.impact_evaluator = ImpactEvaluator()
        
        # 决策指导组件
        self.decision_guider = DecisionGuider()
        
        logger.info("价值观形成器初始化完成")
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建价值观表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "values" (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                value_type TEXT,
                level TEXT,
                strength REAL,
                priority INTEGER,
                context TEXT,
                source TEXT,
                timestamp TEXT,
                attributes TEXT
            )
        ''')
        
        # 创建价值观关系表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS value_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_value_id TEXT,
                target_value_id TEXT,
                relationship_type TEXT,
                strength REAL,
                description TEXT,
                timestamp TEXT
            )
        ''')
        
        # 创建价值观冲突表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS value_conflicts (
                id TEXT PRIMARY KEY,
                conflict_type TEXT,
                value_ids TEXT,
                description TEXT,
                severity REAL,
                resolution_suggestions TEXT,
                timestamp TEXT,
                resolved BOOLEAN
            )
        ''')
        
        # 创建价值观演化记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS value_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value_id TEXT,
                change_type TEXT,
                old_value TEXT,
                new_value TEXT,
                timestamp TEXT,
                context TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def identify_values(self, text_data: str, source: str = "文本分析") -> List[Value]:
        """
        价值观识别和提取
        
        Args:
            text_data: 待分析的文本数据
            source: 数据来源
            
        Returns:
            识别出的价值观列表
        """
        logger.info("开始价值观识别和提取")
        
        # 使用价值观提取器识别价值观
        extracted_values = self.value_extractor.extract_values(text_data, source)
        
        # 将识别的价值观添加到系统中
        for value_data in extracted_values:
            value = Value(**value_data)
            self.values[value.id] = value
            
            # 保存到数据库
            self._save_value_to_db(value)
            
            # 记录演化轨迹
            self._track_value_evolution(value.id, "identified", None, value.to_dict())
        
        logger.info(f"识别出 {len(extracted_values)} 个价值观")
        return list(self.values.values())
    
    def model_values(self) -> Dict[str, Any]:
        """
        价值观建模和表示
        
        Returns:
            价值观模型
        """
        logger.info("开始价值观建模和表示")
        
        # 使用价值观建模器构建模型
        model = self.value_modeler.build_value_model(self.values, self.relationships)
        
        logger.info("价值观建模完成")
        return model
    
    def detect_conflicts(self) -> List[ValueConflict]:
        """
        价值观冲突检测
        
        Returns:
            检测到的冲突列表
        """
        logger.info("开始价值观冲突检测")
        
        # 使用冲突检测器检测冲突
        conflicts = self.conflict_detector.detect_conflicts(self.values, self.relationships)
        
        # 更新冲突列表
        self.conflicts.extend(conflicts)
        
        # 保存冲突到数据库
        for conflict in conflicts:
            self._save_conflict_to_db(conflict)
        
        logger.info(f"检测到 {len(conflicts)} 个价值观冲突")
        return conflicts
    
    def resolve_conflicts(self, conflict_id: str, resolution: str) -> bool:
        """
        价值观冲突解决
        
        Args:
            conflict_id: 冲突ID
            resolution: 解决方案
            
        Returns:
            是否成功解决
        """
        logger.info(f"尝试解决冲突: {conflict_id}")
        
        # 查找冲突
        conflict = next((c for c in self.conflicts if c.id == conflict_id), None)
        if not conflict:
            logger.warning(f"未找到冲突: {conflict_id}")
            return False
        
        # 应用解决方案
        success = self.conflict_detector.resolve_conflict(conflict, resolution)
        
        if success:
            conflict.resolved = True
            # 更新数据库
            self._update_conflict_in_db(conflict)
            logger.info(f"成功解决冲突: {conflict_id}")
        else:
            logger.warning(f"解决冲突失败: {conflict_id}")
        
        return success
    
    def check_consistency(self) -> Dict[str, Any]:
        """
        价值观一致性检验
        
        Returns:
            一致性检验结果
        """
        logger.info("开始价值观一致性检验")
        
        # 使用一致性检验器检查一致性
        consistency_result = self.consistency_checker.check_consistency(
            self.values, self.relationships, self.conflicts
        )
        
        logger.info("价值观一致性检验完成")
        return consistency_result
    
    def track_evolution(self, value_id: str, new_attributes: Dict[str, Any]) -> bool:
        """
        价值观演化追踪
        
        Args:
            value_id: 价值观ID
            new_attributes: 新的属性
            
        Returns:
            是否成功追踪
        """
        if value_id not in self.values:
            logger.warning(f"未找到价值观: {value_id}")
            return False
        
        old_value = self.values[value_id]
        old_dict = old_value.to_dict()
        
        # 更新价值观
        for key, value in new_attributes.items():
            if hasattr(old_value, key):
                setattr(old_value, key, value)
        
        new_dict = self.values[value_id].to_dict()
        
        # 追踪演化
        self.evolution_tracker.track_change(value_id, old_dict, new_dict)
        
        # 记录演化轨迹
        self._track_value_evolution(value_id, "updated", old_dict, new_dict)
        
        # 保存到数据库
        self._save_value_to_db(self.values[value_id])
        
        logger.info(f"成功追踪价值观演化: {value_id}")
        return True
    
    def evaluate_impact(self, decision_context: str) -> Dict[str, float]:
        """
        价值观影响评估
        
        Args:
            decision_context: 决策情境
            
        Returns:
            影响评估结果
        """
        logger.info("开始价值观影响评估")
        
        # 使用影响评估器评估影响
        impact_scores = self.impact_evaluator.evaluate_impact(
            self.values, decision_context
        )
        
        logger.info("价值观影响评估完成")
        return impact_scores
    
    def guide_decision(self, decision_options: List[str], context: str) -> Dict[str, Any]:
        """
        价值观指导决策
        
        Args:
            decision_options: 决策选项列表
            context: 决策情境
            
        Returns:
            决策指导结果
        """
        logger.info("开始价值观指导决策")
        
        # 使用决策指导器提供决策建议
        decision_result = self.decision_guider.guide_decision(
            self.values, decision_options, context
        )
        
        logger.info("价值观指导决策完成")
        return decision_result
    
    def get_value_hierarchy(self) -> Dict[str, Any]:
        """
        获取价值观层级结构
        
        Returns:
            价值观层级结构
        """
        hierarchy = {}
        
        # 按类型分组
        for value_type in ValueType:
            type_values = [
                v for v in self.values.values() 
                if v.value_type == value_type
            ]
            hierarchy[value_type.value] = [v.to_dict() for v in type_values]
        
        return hierarchy
    
    def export_values(self) -> Dict[str, Any]:
        """
        导出价值观数据
        
        Returns:
            价值观数据
        """
        return {
            'values': [v.to_dict() for v in self.values.values()],
            'relationships': [r.to_dict() for r in self.relationships],
            'conflicts': [c.to_dict() for c in self.conflicts],
            'evolution_tracks': self.value_evolution_tracks,
            'export_time': datetime.now().isoformat()
        }
    
    def import_values(self, data: Dict[str, Any]) -> bool:
        """
        导入价值观数据
        
        Args:
            data: 价值观数据
            
        Returns:
            是否成功导入
        """
        try:
            # 导入价值观
            for value_dict in data.get('values', []):
                value = Value(**value_dict)
                # 转换时间戳
                value.timestamp = datetime.fromisoformat(value_dict['timestamp'])
                self.values[value.id] = value
            
            # 导入关系
            for rel_dict in data.get('relationships', []):
                relationship = ValueRelationship(**rel_dict)
                relationship.timestamp = datetime.fromisoformat(rel_dict['timestamp'])
                self.relationships.append(relationship)
            
            # 导入冲突
            for conflict_dict in data.get('conflicts', []):
                conflict = ValueConflict(**conflict_dict)
                conflict.timestamp = datetime.fromisoformat(conflict_dict['timestamp'])
                conflict.conflict_type = ConflictType(conflict_dict['conflict_type'])
                self.conflicts.append(conflict)
            
            # 导入演化轨迹
            self.value_evolution_tracks = data.get('evolution_tracks', {})
            
            logger.info("成功导入价值观数据")
            return True
            
        except Exception as e:
            logger.error(f"导入价值观数据失败: {e}")
            return False
    
    def _save_value_to_db(self, value: Value):
        """保存价值观到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO "values" 
            (id, name, description, value_type, level, strength, priority, 
             context, source, timestamp, attributes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            value.id, value.name, value.description, value.value_type.value,
            value.level.value, value.strength, value.priority, value.context,
            value.source, value.timestamp.isoformat(), json.dumps(value.attributes)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_conflict_to_db(self, conflict: ValueConflict):
        """保存冲突到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO value_conflicts 
            (id, conflict_type, value_ids, description, severity, 
             resolution_suggestions, timestamp, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conflict.id, conflict.conflict_type.value, json.dumps(conflict.value_ids),
            conflict.description, conflict.severity, json.dumps(conflict.resolution_suggestions),
            conflict.timestamp.isoformat(), conflict.resolved
        ))
        
        conn.commit()
        conn.close()
    
    def _update_conflict_in_db(self, conflict: ValueConflict):
        """更新数据库中的冲突"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE value_conflicts 
            SET resolved = ? 
            WHERE id = ?
        ''', (conflict.resolved, conflict.id))
        
        conn.commit()
        conn.close()
    
    def _track_value_evolution(self, value_id: str, change_type: str, 
                              old_value: Dict, new_value: Dict):
        """追踪价值观演化"""
        if value_id not in self.value_evolution_tracks:
            self.value_evolution_tracks[value_id] = []
        
        evolution_record = {
            'change_type': change_type,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': datetime.now().isoformat()
        }
        
        self.value_evolution_tracks[value_id].append(evolution_record)


class ValueExtractor:
    """价值观提取器"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.value_patterns = {
            'core_values': ['诚信', '责任', '创新', '合作', '卓越', '尊重'],
            'derived_values': ['效率', '质量', '服务', '学习', '成长'],
            'situational_values': ['紧急', '重要', '临时', '特定']
        }
    
    def extract_values(self, text_data: str, source: str) -> List[Dict[str, Any]]:
        """
        从文本中提取价值观
        
        Args:
            text_data: 待分析的文本
            source: 数据来源
            
        Returns:
            提取的价值观数据列表
        """
        # 使用TF-IDF提取关键词
        if text_data.strip():
            tfidf_matrix = self.vectorizer.fit_transform([text_data])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # 获取高分关键词
            top_indices = np.argsort(tfidf_scores)[-10:]
            keywords = [feature_names[i] for i in top_indices]
        else:
            keywords = []
        
        # 基于模式匹配提取价值观
        extracted_values = []
        
        # 核心价值观识别
        for value_name in self.value_patterns['core_values']:
            if value_name in text_data:
                value_data = {
                    'id': f"core_{value_name}_{datetime.now().timestamp()}",
                    'name': value_name,
                    'description': f"从{source}中识别的核心价值观: {value_name}",
                    'value_type': ValueType.CORE,
                    'level': ValueLevel.ABSTRACT,
                    'strength': self._calculate_strength(text_data, value_name),
                    'priority': self._calculate_priority(value_name),
                    'context': '通用',
                    'source': source,
                    'timestamp': datetime.now()
                }
                extracted_values.append(value_data)
        
        # 关键词价值观识别
        for keyword in keywords:
            if keyword not in [v['name'] for v in extracted_values]:
                value_data = {
                    'id': f"keyword_{keyword}_{datetime.now().timestamp()}",
                    'name': keyword,
                    'description': f"从{source}中提取的关键词价值观: {keyword}",
                    'value_type': ValueType.DERIVED,
                    'level': ValueLevel.CONCEPTUAL,
                    'strength': 0.5,
                    'priority': 5,
                    'context': '分析提取',
                    'source': source,
                    'timestamp': datetime.now()
                }
                extracted_values.append(value_data)
        
        return extracted_values
    
    def _calculate_strength(self, text: str, value_name: str) -> float:
        """计算价值观强度"""
        count = text.count(value_name)
        return min(count / 10.0, 1.0)  # 归一化到0-1
    
    def _calculate_priority(self, value_name: str) -> int:
        """计算价值观优先级"""
        priority_map = {
            '诚信': 10, '责任': 9, '创新': 8, '合作': 7, '卓越': 8, '尊重': 7,
            '效率': 6, '质量': 7, '服务': 6, '学习': 5, '成长': 5
        }
        return priority_map.get(value_name, 5)


class ValueModeler:
    """价值观建模器"""
    
    def build_value_model(self, values: Dict[str, Value], 
                         relationships: List[ValueRelationship]) -> Dict[str, Any]:
        """
        构建价值观模型
        
        Args:
            values: 价值观字典
            relationships: 价值观关系列表
            
        Returns:
            价值观模型
        """
        model = {
            'hierarchy': self._build_hierarchy(values),
            'network': self._build_network(values, relationships),
            'clusters': self._cluster_values(values),
            'metrics': self._calculate_metrics(values, relationships)
        }
        
        return model
    
    def _build_hierarchy(self, values: Dict[str, Value]) -> Dict[str, Any]:
        """构建价值观层级结构"""
        hierarchy = {
            'levels': {},
            'dependencies': {}
        }
        
        # 按层级分组
        for value in values.values():
            level = value.level.value
            if level not in hierarchy['levels']:
                hierarchy['levels'][level] = []
            hierarchy['levels'][level].append(value.to_dict())
        
        return hierarchy
    
    def _build_network(self, values: Dict[str, Value], 
                      relationships: List[ValueRelationship]) -> Dict[str, Any]:
        """构建价值观网络"""
        network = {
            'nodes': list(values.keys()),
            'edges': [],
            'communities': []
        }
        
        # 构建边
        for rel in relationships:
            network['edges'].append({
                'source': rel.source_value_id,
                'target': rel.target_value_id,
                'type': rel.relationship_type,
                'strength': rel.strength
            })
        
        return network
    
    def _cluster_values(self, values: Dict[str, Value]) -> List[List[str]]:
        """价值观聚类"""
        if len(values) < 2:
            return [list(values.keys())]
        
        # 构建特征矩阵
        features = []
        value_ids = []
        
        for value_id, value in values.items():
            features.append([
                value.strength,
                value.priority,
                hash(value.value_type.value) % 100,
                hash(value.level.value) % 100
            ])
            value_ids.append(value_id)
        
        # K-means聚类
        n_clusters = min(3, len(values))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features)
            
            # 按聚类分组
            clustered_values = {}
            for i, cluster in enumerate(clusters):
                if cluster not in clustered_values:
                    clustered_values[cluster] = []
                clustered_values[cluster].append(value_ids[i])
            
            return list(clustered_values.values())
        else:
            return [value_ids]
    
    def _calculate_metrics(self, values: Dict[str, Value], 
                          relationships: List[ValueRelationship]) -> Dict[str, float]:
        """计算价值观指标"""
        if not values:
            return {}
        
        # 计算平均强度
        avg_strength = np.mean([v.strength for v in values.values()])
        
        # 计算平均优先级
        avg_priority = np.mean([v.priority for v in values.values()])
        
        # 计算关系密度
        max_possible_relations = len(values) * (len(values) - 1)
        actual_relations = len(relationships)
        density = actual_relations / max_possible_relations if max_possible_relations > 0 else 0
        
        # 计算一致性分数
        consistency_scores = []
        for value in values.values():
            # 基于强度和优先级的简单一致性计算
            consistency = (value.strength + value.priority / 10.0) / 2.0
            consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores)
        
        return {
            'average_strength': avg_strength,
            'average_priority': avg_priority,
            'relationship_density': density,
            'consistency_score': avg_consistency,
            'total_values': len(values),
            'total_relationships': len(relationships)
        }


class ConflictDetector:
    """冲突检测器"""
    
    def detect_conflicts(self, values: Dict[str, Value], 
                        relationships: List[ValueRelationship]) -> List[ValueConflict]:
        """
        检测价值观冲突
        
        Args:
            values: 价值观字典
            relationships: 价值观关系列表
            
        Returns:
            检测到的冲突列表
        """
        conflicts = []
        
        # 直接冲突检测
        direct_conflicts = self._detect_direct_conflicts(values)
        conflicts.extend(direct_conflicts)
        
        # 间接冲突检测
        indirect_conflicts = self._detect_indirect_conflicts(values, relationships)
        conflicts.extend(indirect_conflicts)
        
        # 层级冲突检测
        hierarchical_conflicts = self._detect_hierarchical_conflicts(values)
        conflicts.extend(hierarchical_conflicts)
        
        return conflicts
    
    def _detect_direct_conflicts(self, values: Dict[str, Value]) -> List[ValueConflict]:
        """检测直接冲突"""
        conflicts = []
        value_list = list(values.values())
        
        # 查找优先级冲突
        for i, value1 in enumerate(value_list):
            for j, value2 in enumerate(value_list[i+1:], i+1):
                # 如果两个价值观在相同情境下有冲突的优先级
                if (value1.context == value2.context and 
                    abs(value1.priority - value2.priority) <= 1 and
                    abs(value1.strength - value2.strength) <= 0.1):
                    
                    conflict = ValueConflict(
                        id=f"direct_{value1.id}_{value2.id}_{datetime.now().timestamp()}",
                        conflict_type=ConflictType.DIRECT,
                        value_ids=[value1.id, value2.id],
                        description=f"价值观'{value1.name}'和'{value2.name}'存在直接冲突",
                        severity=0.7,
                        resolution_suggestions=[
                            "调整优先级顺序",
                            "明确适用情境",
                            "寻找平衡点"
                        ],
                        timestamp=datetime.now()
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_indirect_conflicts(self, values: Dict[str, Value], 
                                  relationships: List[ValueRelationship]) -> List[ValueConflict]:
        """检测间接冲突"""
        conflicts = []
        
        # 构建关系图
        relation_graph = defaultdict(list)
        for rel in relationships:
            if rel.relationship_type == 'conflict':
                relation_graph[rel.source_value_id].append(rel.target_value_id)
                relation_graph[rel.target_value_id].append(rel.source_value_id)
        
        # 查找间接冲突链
        for value_id, conflicts_list in relation_graph.items():
            if len(conflicts_list) > 1:
                # 多个价值观与同一个价值观冲突，可能存在间接冲突
                for i, conflict_id1 in enumerate(conflicts_list):
                    for conflict_id2 in conflicts_list[i+1:]:
                        if conflict_id1 in values and conflict_id2 in values:
                            conflict = ValueConflict(
                                id=f"indirect_{value_id}_{conflict_id1}_{conflict_id2}_{datetime.now().timestamp()}",
                                conflict_type=ConflictType.INDIRECT,
                                value_ids=[value_id, conflict_id1, conflict_id2],
                                description=f"通过'{values[value_id].name}'形成的间接冲突",
                                severity=0.5,
                                resolution_suggestions=[
                                    "重新评估关系",
                                    "调整冲突链",
                                    "寻找共同点"
                                ],
                                timestamp=datetime.now()
                            )
                            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_hierarchical_conflicts(self, values: Dict[str, Value]) -> List[ValueConflict]:
        """检测层级冲突"""
        conflicts = []
        
        # 按层级分组价值观
        level_groups = defaultdict(list)
        for value in values.values():
            level_groups[value.level].append(value)
        
        # 检测跨层级的优先级冲突
        levels = list(level_groups.keys())
        for i, level1 in enumerate(levels):
            for level2 in levels[i+1:]:
                for value1 in level_groups[level1]:
                    for value2 in level_groups[level2]:
                        # 如果低层级的价值观优先级高于高层级，可能存在层级冲突
                        if (value1.priority > value2.priority and 
                            level1.value in ['具体层', '操作层'] and
                            level2.value in ['抽象层', '概念层']):
                            
                            conflict = ValueConflict(
                                id=f"hierarchical_{value1.id}_{value2.id}_{datetime.now().timestamp()}",
                                conflict_type=ConflictType.HIERARCHICAL,
                                value_ids=[value1.id, value2.id],
                                description=f"层级冲突: '{value1.name}'({level1.value})优先级高于'{value2.name}'({level2.value})",
                                severity=0.6,
                                resolution_suggestions=[
                                    "调整层级结构",
                                    "重新分配优先级",
                                    "明确适用范围"
                                ],
                                timestamp=datetime.now()
                            )
                            conflicts.append(conflict)
        
        return conflicts
    
    def resolve_conflict(self, conflict: ValueConflict, resolution: str) -> bool:
        """
        解决价值观冲突
        
        Args:
            conflict: 冲突对象
            resolution: 解决方案
            
        Returns:
            是否成功解决
        """
        # 这里实现具体的冲突解决逻辑
        # 简化实现：基于解决方案类型进行不同的处理
        
        if "调整优先级" in resolution:
            # 重新分配优先级
            return self._resolve_by_priority_adjustment(conflict)
        elif "明确情境" in resolution:
            # 明确适用情境
            return self._resolve_by_context_clarification(conflict)
        elif "寻找平衡" in resolution:
            # 寻找平衡点
            return self._resolve_by_finding_balance(conflict)
        else:
            # 默认处理
            return True
    
    def _resolve_by_priority_adjustment(self, conflict: ValueConflict) -> bool:
        """通过优先级调整解决冲突"""
        # 实现优先级调整逻辑
        return True
    
    def _resolve_by_context_clarification(self, conflict: ValueConflict) -> bool:
        """通过情境明确解决冲突"""
        # 实现情境明确逻辑
        return True
    
    def _resolve_by_finding_balance(self, conflict: ValueConflict) -> bool:
        """通过寻找平衡点解决冲突"""
        # 实现平衡点寻找逻辑
        return True


class ConsistencyChecker:
    """一致性检验器"""
    
    def check_consistency(self, values: Dict[str, Value], 
                         relationships: List[ValueRelationship],
                         conflicts: List[ValueConflict]) -> Dict[str, Any]:
        """
        检查价值观一致性
        
        Args:
            values: 价值观字典
            relationships: 价值观关系列表
            conflicts: 价值观冲突列表
            
        Returns:
            一致性检验结果
        """
        result = {
            'overall_score': 0.0,
            'consistency_metrics': {},
            'inconsistencies': [],
            'recommendations': []
        }
        
        # 检查内部一致性
        internal_consistency = self._check_internal_consistency(values)
        result['consistency_metrics']['internal'] = internal_consistency
        
        # 检查关系一致性
        relational_consistency = self._check_relational_consistency(values, relationships)
        result['consistency_metrics']['relational'] = relational_consistency
        
        # 检查层级一致性
        hierarchical_consistency = self._check_hierarchical_consistency(values)
        result['consistency_metrics']['hierarchical'] = hierarchical_consistency
        
        # 计算总体一致性分数
        scores = [
            internal_consistency['score'],
            relational_consistency['score'],
            hierarchical_consistency['score']
        ]
        result['overall_score'] = np.mean(scores)
        
        # 生成不一致项
        result['inconsistencies'] = self._identify_inconsistencies(
            values, relationships, conflicts
        )
        
        # 生成建议
        result['recommendations'] = self._generate_recommendations(result)
        
        return result
    
    def _check_internal_consistency(self, values: Dict[str, Value]) -> Dict[str, Any]:
        """检查内部一致性"""
        if not values:
            return {'score': 1.0, 'details': '无价值观数据'}
        
        consistency_scores = []
        
        for value in values.values():
            # 检查强度和优先级的一致性
            strength_priority_consistency = 1.0 - abs(value.strength - value.priority / 10.0)
            
            # 检查描述和强度的匹配度
            description_strength_match = min(value.strength * 2, 1.0)
            
            value_consistency = (strength_priority_consistency + description_strength_match) / 2.0
            consistency_scores.append(value_consistency)
        
        return {
            'score': np.mean(consistency_scores),
            'details': f'检查了{len(values)}个价值观的内部一致性'
        }
    
    def _check_relational_consistency(self, values: Dict[str, Value], 
                                     relationships: List[ValueRelationship]) -> Dict[str, Any]:
        """检查关系一致性"""
        if not relationships:
            return {'score': 1.0, 'details': '无关系数据'}
        
        consistency_scores = []
        
        for rel in relationships:
            # 检查关系强度的一致性
            if rel.source_value_id in values and rel.target_value_id in values:
                source_strength = values[rel.source_value_id].strength
                target_strength = values[rel.target_value_id].strength
                
                # 关系强度应该与价值观强度相关
                expected_strength = (source_strength + target_strength) / 2.0
                strength_consistency = 1.0 - abs(rel.strength - expected_strength)
                
                consistency_scores.append(strength_consistency)
        
        return {
            'score': np.mean(consistency_scores) if consistency_scores else 1.0,
            'details': f'检查了{len(relationships)}个关系的一致性'
        }
    
    def _check_hierarchical_consistency(self, values: Dict[str, Value]) -> Dict[str, Any]:
        """检查层级一致性"""
        if not values:
            return {'score': 1.0, 'details': '无价值观数据'}
        
        # 按层级分组
        level_groups = defaultdict(list)
        for value in values.values():
            level_groups[value.level].append(value)
        
        consistency_scores = []
        
        # 检查层级间的优先级一致性
        levels = list(level_groups.keys())
        for i, level1 in enumerate(levels):
            for level2 in levels[i+1:]:
                avg_priority1 = np.mean([v.priority for v in level_groups[level1]])
                avg_priority2 = np.mean([v.priority for v in level_groups[level2]])
                
                # 抽象层级应该有更高的优先级
                if level1.value in ['抽象层', '概念层'] and level2.value in ['具体层', '操作层']:
                    if avg_priority1 >= avg_priority2:
                        consistency_scores.append(1.0)
                    else:
                        consistency_scores.append(0.5)
                else:
                    consistency_scores.append(0.8)  # 默认分数
        
        return {
            'score': np.mean(consistency_scores) if consistency_scores else 1.0,
            'details': f'检查了{len(levels)}个层级的层级一致性'
        }
    
    def _identify_inconsistencies(self, values: Dict[str, Value], 
                                 relationships: List[ValueRelationship],
                                 conflicts: List[ValueConflict]) -> List[Dict[str, Any]]:
        """识别不一致项"""
        inconsistencies = []
        
        # 价值观强度异常
        for value in values.values():
            if value.strength > 0.9 and value.priority < 3:
                inconsistencies.append({
                    'type': 'strength_priority_mismatch',
                    'description': f"价值观'{value.name}'强度很高但优先级很低",
                    'value_id': value.id,
                    'severity': 'medium'
                })
        
        # 未解决的冲突
        for conflict in conflicts:
            if not conflict.resolved:
                inconsistencies.append({
                    'type': 'unresolved_conflict',
                    'description': f"存在未解决的冲突: {conflict.description}",
                    'conflict_id': conflict.id,
                    'severity': 'high'
                })
        
        return inconsistencies
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if result['overall_score'] < 0.7:
            recommendations.append("总体一致性较低，建议重新评估价值观体系")
        
        if result['consistency_metrics']['internal']['score'] < 0.6:
            recommendations.append("内部一致性需要改进，调整价值观的强度和优先级")
        
        if result['consistency_metrics']['hierarchical']['score'] < 0.6:
            recommendations.append("层级一致性需要改进，重新组织价值观的层级结构")
        
        if len(result['inconsistencies']) > 0:
            recommendations.append("存在不一致项，需要逐一解决")
        
        return recommendations


class EvolutionTracker:
    """演化追踪器"""
    
    def __init__(self):
        self.evolution_history = defaultdict(list)
        self.change_patterns = {}
    
    def track_change(self, value_id: str, old_value: Dict[str, Any], 
                    new_value: Dict[str, Any]):
        """
        追踪价值观变化
        
        Args:
            value_id: 价值观ID
            old_value: 旧价值观数据
            new_value: 新价值观数据
        """
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'old_value': old_value,
            'new_value': new_value,
            'changes': self._identify_changes(old_value, new_value)
        }
        
        self.evolution_history[value_id].append(change_record)
        
        # 分析变化模式
        self._analyze_change_patterns(value_id, change_record)
    
    def _identify_changes(self, old_value: Dict[str, Any], 
                         new_value: Dict[str, Any]) -> Dict[str, Any]:
        """识别具体变化"""
        changes = {}
        
        for key in old_value.keys():
            if key in new_value:
                if old_value[key] != new_value[key]:
                    changes[key] = {
                        'old': old_value[key],
                        'new': new_value[key]
                    }
        
        return changes
    
    def _analyze_change_patterns(self, value_id: str, change_record: Dict[str, Any]):
        """分析变化模式"""
        if value_id not in self.change_patterns:
            self.change_patterns[value_id] = {
                'change_frequency': 0,
                'common_changes': defaultdict(int),
                'stability_score': 1.0
            }
        
        pattern = self.change_patterns[value_id]
        pattern['change_frequency'] += 1
        
        # 统计常见变化
        for change_type in change_record['changes'].keys():
            pattern['common_changes'][change_type] += 1
        
        # 计算稳定性分数
        pattern['stability_score'] = max(0.1, 1.0 - pattern['change_frequency'] / 100.0)
    
    def get_evolution_summary(self, value_id: str) -> Dict[str, Any]:
        """获取演化摘要"""
        if value_id not in self.evolution_history:
            return {'status': 'no_evolution_data'}
        
        history = self.evolution_history[value_id]
        pattern = self.change_patterns.get(value_id, {})
        
        return {
            'total_changes': len(history),
            'change_frequency': pattern.get('change_frequency', 0),
            'stability_score': pattern.get('stability_score', 1.0),
            'common_change_types': dict(pattern.get('common_changes', {})),
            'latest_change': history[-1] if history else None,
            'evolution_timeline': [record['timestamp'] for record in history]
        }


class ImpactEvaluator:
    """影响评估器"""
    
    def evaluate_impact(self, values: Dict[str, Value], 
                       decision_context: str) -> Dict[str, float]:
        """
        评估价值观对决策的影响
        
        Args:
            values: 价值观字典
            decision_context: 决策情境
            
        Returns:
            影响评估结果
        """
        impact_scores = {}
        
        for value_id, value in values.items():
            # 计算情境相关性
            context_relevance = self._calculate_context_relevance(value, decision_context)
            
            # 计算影响强度
            impact_intensity = value.strength * value.priority / 10.0
            
            # 综合影响分数
            impact_scores[value_id] = context_relevance * impact_intensity
        
        # 归一化影响分数
        if impact_scores:
            max_impact = max(impact_scores.values())
            if max_impact > 0:
                impact_scores = {k: v / max_impact for k, v in impact_scores.items()}
        
        return impact_scores
    
    def _calculate_context_relevance(self, value: Value, context: str) -> float:
        """计算情境相关性"""
        # 简单的关键词匹配
        context_words = context.lower().split()
        value_context = value.context.lower()
        
        relevance = 0.0
        for word in context_words:
            if word in value_context:
                relevance += 1.0
        
        return min(relevance / len(context_words), 1.0) if context_words else 0.0


class DecisionGuider:
    """决策指导器"""
    
    def guide_decision(self, values: Dict[str, Value], 
                      decision_options: List[str], context: str) -> Dict[str, Any]:
        """
        基于价值观指导决策
        
        Args:
            values: 价值观字典
            decision_options: 决策选项列表
            context: 决策情境
            
        Returns:
            决策指导结果
        """
        # 计算每个选项的价值观匹配度
        option_scores = {}
        
        for option in decision_options:
            score = self._calculate_option_value_match(values, option, context)
            option_scores[option] = score
        
        # 排序选项
        sorted_options = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 生成决策建议
        recommendations = self._generate_decision_recommendations(
            sorted_options, values, context
        )
        
        return {
            'recommended_option': sorted_options[0][0] if sorted_options else None,
            'option_scores': option_scores,
            'ranked_options': [option for option, score in sorted_options],
            'confidence': self._calculate_confidence(sorted_options),
            'recommendations': recommendations,
            'value_alignment': self._analyze_value_alignment(sorted_options, values)
        }
    
    def _calculate_option_value_match(self, values: Dict[str, Value], 
                                     option: str, context: str) -> float:
        """计算选项与价值观的匹配度"""
        total_score = 0.0
        total_weight = 0.0
        
        for value in values.values():
            # 计算价值观与选项的匹配度
            match_score = self._calculate_match_score(value, option, context)
            
            # 权重基于价值观的强度和优先级
            weight = value.strength * value.priority / 10.0
            
            total_score += match_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_match_score(self, value: Value, option: str, context: str) -> float:
        """计算单个价值观与选项的匹配度"""
        # 简化实现：基于关键词匹配
        option_lower = option.lower()
        context_lower = context.lower()
        
        # 检查价值观名称是否在选项或情境中出现
        value_name = value.name.lower()
        
        if value_name in option_lower or value_name in context_lower:
            return value.strength
        else:
            return 0.1  # 默认低分
    
    def _generate_decision_recommendations(self, sorted_options: List[Tuple[str, float]], 
                                          values: Dict[str, Value], context: str) -> List[str]:
        """生成决策建议"""
        recommendations = []
        
        if not sorted_options:
            recommendations.append("没有可用的决策选项")
            return recommendations
        
        best_option, best_score = sorted_options[0]
        
        recommendations.append(f"推荐选择: {best_option}")
        recommendations.append(f"匹配度分数: {best_score:.2f}")
        
        # 分析价值观一致性
        top_values = sorted(values.values(), key=lambda v: v.strength * v.priority, reverse=True)[:3]
        recommendations.append(f"主要价值观: {', '.join([v.name for v in top_values])}")
        
        # 风险提示
        if len(sorted_options) > 1:
            score_diff = sorted_options[0][1] - sorted_options[1][1]
            if score_diff < 0.1:
                recommendations.append("注意：前两个选项得分接近，建议进一步分析")
        
        return recommendations
    
    def _calculate_confidence(self, sorted_options: List[Tuple[str, float]]) -> float:
        """计算决策置信度"""
        if len(sorted_options) < 2:
            return 1.0
        
        best_score = sorted_options[0][1]
        second_score = sorted_options[1][1]
        
        # 置信度基于最佳选项与次佳选项的差距
        score_gap = best_score - second_score
        confidence = min(score_gap * 10, 1.0)
        
        return confidence
    
    def _analyze_value_alignment(self, sorted_options: List[Tuple[str, float]], 
                                values: Dict[str, Value]) -> Dict[str, Any]:
        """分析价值观一致性"""
        if not sorted_options:
            return {'status': 'no_options'}
        
        best_option = sorted_options[0][0]
        
        # 分析最佳选项与价值观的契合度
        alignment_analysis = {
            'best_option': best_option,
            'core_values_aligned': [],
            'conflicting_values': [],
            'overall_alignment_score': 0.0
        }
        
        # 按价值观类型分析
        core_values = [v for v in values.values() if v.value_type == ValueType.CORE]
        if core_values:
            alignment_analysis['core_values_aligned'] = [v.name for v in core_values]
        
        return alignment_analysis


# 示例使用函数
def demo_value_formation():
    """演示价值观形成器的使用"""
    print("=== E3价值观形成器演示 ===")
    
    # 创建价值观形成器实例
    vf = ValueFormation()
    
    # 1. 价值观识别和提取
    print("\n1. 价值观识别和提取")
    sample_text = """
    我们公司秉承诚信经营的理念，注重创新发展。
    团队合作是我们的核心价值观，追求卓越是我们的目标。
    我们尊重每一位员工，致力于提供优质的服务。
    持续学习和成长是我们发展的重要驱动力。
    """
    
    identified_values = vf.identify_values(sample_text, "公司文化文档")
    print(f"识别出 {len(identified_values)} 个价值观")
    
    # 2. 价值观建模和表示
    print("\n2. 价值观建模和表示")
    value_model = vf.model_values()
    print("价值观模型构建完成")
    print(f"层级结构: {list(value_model['hierarchy']['levels'].keys())}")
    
    # 3. 价值观冲突检测
    print("\n3. 价值观冲突检测")
    conflicts = vf.detect_conflicts()
    print(f"检测到 {len(conflicts)} 个冲突")
    
    # 4. 价值观一致性检验
    print("\n4. 价值观一致性检验")
    consistency_result = vf.check_consistency()
    print(f"一致性分数: {consistency_result['overall_score']:.2f}")
    
    # 5. 价值观演化追踪
    print("\n5. 价值观演化追踪")
    if identified_values:
        first_value = identified_values[0]
        vf.track_evolution(first_value.id, {'strength': 0.8, 'priority': 7})
        print(f"追踪价值观 {first_value.name} 的演化")
    
    # 6. 价值观影响评估
    print("\n6. 价值观影响评估")
    impact_scores = vf.evaluate_impact("重大投资决策情境")
    print(f"影响评估完成，评估了 {len(impact_scores)} 个价值观")
    
    # 7. 价值观指导决策
    print("\n7. 价值观指导决策")
    decision_options = ["保守投资", "激进投资", "平衡投资"]
    decision_result = vf.guide_decision(decision_options, "投资策略选择")
    print(f"推荐决策: {decision_result['recommended_option']}")
    print(f"置信度: {decision_result['confidence']:.2f}")
    
    # 导出结果
    print("\n8. 导出价值观数据")
    exported_data = vf.export_values()
    print(f"导出 {len(exported_data['values'])} 个价值观数据")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_value_formation()