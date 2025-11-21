"""
F7经验管理器（进步经验存储）
实现多层次经验存储、智能检索、质量评估和应用推荐系统

功能模块：
1. 多层次经验存储和管理（短期、中期、长期经验）
2. 经验检索和智能关联分析
3. 经验学习效果评估和量化
4. 经验知识自动提取和转化
5. 经验质量评估和智能清洗
6. 经验智能应用和推荐系统
7. 经验历史数据跟踪和管理
"""

import json
import sqlite3
import threading
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor
import uuid


class ExperienceLevel(Enum):
    """经验层次枚举"""
    SHORT_TERM = "短期经验"      # 1-7天
    MEDIUM_TERM = "中期经验"     # 1-4周
    LONG_TERM = "长期经验"       # 1个月以上


class ExperienceType(Enum):
    """经验类型枚举"""
    SUCCESS = "成功经验"
    FAILURE = "失败经验"
    LEARNING = "学习经验"
    PROBLEM_SOLVING = "问题解决经验"
    DECISION_MAKING = "决策经验"
    CREATIVE = "创新经验"


class QualityLevel(Enum):
    """经验质量等级"""
    HIGH = "高质量"
    MEDIUM = "中等质量"
    LOW = "低质量"
    NEEDS_IMPROVEMENT = "需要改进"


@dataclass
class Experience:
    """经验数据结构"""
    id: str
    title: str
    content: str
    experience_type: ExperienceType
    level: ExperienceLevel
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    tags: List[str]
    quality_score: float
    created_at: datetime
    updated_at: datetime
    access_count: int
    success_rate: float
    related_experiences: List[str]
    knowledge_extracted: List[str]
    application_scenarios: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['experience_type'] = self.experience_type.value
        data['level'] = self.level.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """从字典创建实例"""
        data['experience_type'] = ExperienceType(data['experience_type'])
        data['level'] = ExperienceLevel(data['level'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ExperienceMetrics:
    """经验指标数据结构"""
    total_experiences: int
    success_rate: float
    average_quality_score: float
    most_used_experiences: List[str]
    learning_effectiveness: float
    knowledge_retention_rate: float
    application_success_rate: float


class ExperienceStorage:
    """经验存储管理器"""
    
    def __init__(self, db_path: str = "experience_storage.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建经验表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    experience_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    context TEXT,
                    outcome TEXT,
                    tags TEXT,
                    quality_score REAL,
                    created_at TEXT,
                    updated_at TEXT,
                    access_count INTEGER,
                    success_rate REAL,
                    related_experiences TEXT,
                    knowledge_extracted TEXT,
                    application_scenarios TEXT
                )
            ''')
            
            # 创建经验关联表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experience_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experience1_id TEXT,
                    experience2_id TEXT,
                    relation_type TEXT,
                    strength REAL,
                    created_at TEXT
                )
            ''')
            
            # 创建应用记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS application_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experience_id TEXT,
                    application_context TEXT,
                    success BOOLEAN,
                    effectiveness_score REAL,
                    applied_at TEXT
                )
            ''')
            
            # 创建知识提取表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_extractions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experience_id TEXT,
                    knowledge_type TEXT,
                    content TEXT,
                    confidence_score REAL,
                    extracted_at TEXT
                )
            ''')
            
            conn.commit()
    
    def store_experience(self, experience: Experience) -> bool:
        """存储经验"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO experiences 
                        (id, title, content, experience_type, level, context, outcome, 
                         tags, quality_score, created_at, updated_at, access_count, 
                         success_rate, related_experiences, knowledge_extracted, 
                         application_scenarios)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        experience.id, experience.title, experience.content,
                        experience.experience_type.value, experience.level.value,
                        json.dumps(experience.context), json.dumps(experience.outcome),
                        json.dumps(experience.tags), experience.quality_score,
                        experience.created_at.isoformat(), experience.updated_at.isoformat(),
                        experience.access_count, experience.success_rate,
                        json.dumps(experience.related_experiences),
                        json.dumps(experience.knowledge_extracted),
                        json.dumps(experience.application_scenarios)
                    ))
                    conn.commit()
                    return True
        except Exception as e:
            logging.error(f"存储经验失败: {e}")
            return False
    
    def retrieve_experience(self, experience_id: str) -> Optional[Experience]:
        """检索经验"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM experiences WHERE id = ?', (experience_id,))
                row = cursor.fetchone()
                
                if row:
                    return Experience.from_dict({
                        'id': row[0], 'title': row[1], 'content': row[2],
                        'experience_type': row[3], 'level': row[4],
                        'context': json.loads(row[5] or '{}'),
                        'outcome': json.loads(row[6] or '{}'),
                        'tags': json.loads(row[7] or '[]'),
                        'quality_score': row[8], 'created_at': row[9],
                        'updated_at': row[10], 'access_count': row[11],
                        'success_rate': row[12],
                        'related_experiences': json.loads(row[13] or '[]'),
                        'knowledge_extracted': json.loads(row[14] or '[]'),
                        'application_scenarios': json.loads(row[15] or '[]')
                    })
        except Exception as e:
            logging.error(f"检索经验失败: {e}")
        return None
    
    def search_experiences(self, query: str, filters: Dict[str, Any] = None) -> List[Experience]:
        """搜索经验"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                sql = "SELECT * FROM experiences WHERE (title LIKE ? OR content LIKE ?)"
                params = [f"%{query}%", f"%{query}%"]
                
                if filters:
                    if 'type' in filters:
                        sql += " AND experience_type = ?"
                        params.append(filters['type'])
                    if 'level' in filters:
                        sql += " AND level = ?"
                        params.append(filters['level'])
                    if 'min_quality' in filters:
                        sql += " AND quality_score >= ?"
                        params.append(filters['min_quality'])
                
                sql += " ORDER BY quality_score DESC, access_count DESC"
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                experiences = []
                for row in rows:
                    experiences.append(Experience.from_dict({
                        'id': row[0], 'title': row[1], 'content': row[2],
                        'experience_type': row[3], 'level': row[4],
                        'context': json.loads(row[5] or '{}'),
                        'outcome': json.loads(row[6] or '{}'),
                        'tags': json.loads(row[7] or '[]'),
                        'quality_score': row[8], 'created_at': row[9],
                        'updated_at': row[10], 'access_count': row[11],
                        'success_rate': row[12],
                        'related_experiences': json.loads(row[13] or '[]'),
                        'knowledge_extracted': json.loads(row[14] or '[]'),
                        'application_scenarios': json.loads(row[15] or '[]')
                    }))
                return experiences
        except Exception as e:
            logging.error(f"搜索经验失败: {e}")
            return []
    
    def get_related_experiences(self, experience_id: str) -> List[Experience]:
        """获取相关经验"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT e2.* FROM experiences e1
                    JOIN experience_relations er ON e1.id = er.experience1_id
                    JOIN experiences e2 ON e2.id = er.experience2_id
                    WHERE e1.id = ? AND er.strength > 0.5
                    ORDER BY er.strength DESC
                ''', (experience_id,))
                
                rows = cursor.fetchall()
                related_experiences = []
                for row in rows:
                    related_experiences.append(Experience.from_dict({
                        'id': row[0], 'title': row[1], 'content': row[2],
                        'experience_type': row[3], 'level': row[4],
                        'context': json.loads(row[5] or '{}'),
                        'outcome': json.loads(row[6] or '{}'),
                        'tags': json.loads(row[7] or '[]'),
                        'quality_score': row[8], 'created_at': row[9],
                        'updated_at': row[10], 'access_count': row[11],
                        'success_rate': row[12],
                        'related_experiences': json.loads(row[13] or '[]'),
                        'knowledge_extracted': json.loads(row[14] or '[]'),
                        'application_scenarios': json.loads(row[15] or '[]')
                    }))
                return related_experiences
        except Exception as e:
            logging.error(f"获取相关经验失败: {e}")
            return []


class ExperienceAnalyzer:
    """经验分析器"""
    
    def __init__(self):
        self.keywords = defaultdict(list)
        self.patterns = defaultdict(list)
    
    def analyze_experience_quality(self, experience: Experience) -> QualityLevel:
        """分析经验质量"""
        score = 0.0
        
        # 内容完整性评分
        if len(experience.content) > 100:
            score += 0.2
        if experience.context:
            score += 0.2
        if experience.outcome:
            score += 0.2
        
        # 标签和分类评分
        if len(experience.tags) >= 3:
            score += 0.1
        if experience.knowledge_extracted:
            score += 0.1
        
        # 成功率和访问次数评分
        if experience.success_rate > 0.8:
            score += 0.1
        if experience.access_count > 5:
            score += 0.1
        
        if score >= 0.8:
            return QualityLevel.HIGH
        elif score >= 0.6:
            return QualityLevel.MEDIUM
        elif score >= 0.4:
            return QualityLevel.LOW
        else:
            return QualityLevel.NEEDS_IMPROVEMENT
    
    def extract_knowledge(self, experience: Experience) -> List[str]:
        """提取知识"""
        knowledge = []
        
        # 从内容中提取关键概念
        content_words = experience.content.lower().split()
        for word in content_words:
            if len(word) > 4 and word not in ['this', 'that', 'with', 'from', 'they', 'have']:
                knowledge.append(word)
        
        # 从上下文中提取方法论
        if 'method' in experience.context:
            knowledge.append(f"方法: {experience.context['method']}")
        
        if 'principle' in experience.context:
            knowledge.append(f"原理: {experience.context['principle']}")
        
        # 从结果中提取教训
        if experience.outcome.get('lesson'):
            knowledge.append(f"教训: {experience.outcome['lesson']}")
        
        return list(set(knowledge))  # 去重
    
    def find_patterns(self, experiences: List[Experience]) -> Dict[str, List[str]]:
        """发现经验模式"""
        patterns = defaultdict(list)
        
        # 按类型分组
        type_groups = defaultdict(list)
        for exp in experiences:
            type_groups[exp.experience_type].append(exp)
        
        # 分析每种类型的模式
        for exp_type, exps in type_groups.items():
            if len(exps) >= 3:
                # 分析共同特征
                common_tags = set(exps[0].tags)
                for exp in exps[1:]:
                    common_tags &= set(exp.tags)
                
                if common_tags:
                    patterns[f"{exp_type.value}_模式"] = list(common_tags)
        
        return dict(patterns)
    
    def calculate_similarity(self, exp1: Experience, exp2: Experience) -> float:
        """计算经验相似度"""
        similarity = 0.0
        
        # 标签相似度
        if exp1.tags and exp2.tags:
            common_tags = set(exp1.tags) & set(exp2.tags)
            total_tags = set(exp1.tags) | set(exp2.tags)
            similarity += len(common_tags) / len(total_tags) * 0.4
        
        # 类型相似度
        if exp1.experience_type == exp2.experience_type:
            similarity += 0.3
        
        # 层次相似度
        if exp1.level == exp2.level:
            similarity += 0.2
        
        # 质量相似度
        quality_diff = abs(exp1.quality_score - exp2.quality_score)
        similarity += (1 - quality_diff) * 0.1
        
        return similarity


class ExperienceRecommender:
    """经验推荐系统"""
    
    def __init__(self, storage: ExperienceStorage):
        self.storage = storage
        self.analyzer = ExperienceAnalyzer()
    
    def recommend_experiences(self, context: Dict[str, Any], 
                            user_preferences: Dict[str, Any] = None,
                            limit: int = 5) -> List[Experience]:
        """推荐相关经验"""
        try:
            # 构建搜索查询
            query_parts = []
            
            if 'domain' in context:
                query_parts.append(context['domain'])
            if 'problem_type' in context:
                query_parts.append(context['problem_type'])
            if 'goal' in context:
                query_parts.append(context['goal'])
            
            query = ' '.join(query_parts)
            
            # 搜索相关经验
            experiences = self.storage.search_experiences(query)
            
            # 应用用户偏好过滤
            if user_preferences:
                experiences = self._apply_preferences(experiences, user_preferences)
            
            # 按相关性和质量排序
            experiences = self._rank_experiences(experiences, context)
            
            return experiences[:limit]
            
        except Exception as e:
            logging.error(f"推荐经验失败: {e}")
            return []
    
    def _apply_preferences(self, experiences: List[Experience], 
                          preferences: Dict[str, Any]) -> List[Experience]:
        """应用用户偏好"""
        filtered = []
        
        for exp in experiences:
            score = 1.0
            
            # 经验类型偏好
            if 'preferred_types' in preferences:
                if exp.experience_type.value in preferences['preferred_types']:
                    score *= 1.2
                else:
                    score *= 0.8
            
            # 质量偏好
            if 'min_quality' in preferences:
                if exp.quality_score >= preferences['min_quality']:
                    score *= 1.1
                else:
                    score *= 0.9
            
            # 经验层次偏好
            if 'preferred_levels' in preferences:
                if exp.level.value in preferences['preferred_levels']:
                    score *= 1.1
                else:
                    score *= 0.9
            
            if score > 0.5:  # 过滤掉相关性太低的经验
                filtered.append(exp)
        
        return filtered
    
    def _rank_experiences(self, experiences: List[Experience], 
                         context: Dict[str, Any]) -> List[Experience]:
        """排序经验"""
        def rank_score(exp: Experience) -> float:
            score = exp.quality_score * 0.4
            score += exp.success_rate * 0.3
            score += min(exp.access_count / 10.0, 1.0) * 0.2
            
            # 上下文相关性
            context_relevance = 0.0
            if 'domain' in context:
                if context['domain'] in exp.tags:
                    context_relevance += 0.3
            if 'problem_type' in context:
                if context['problem_type'] in exp.content.lower():
                    context_relevance += 0.2
            
            score += context_relevance * 0.1
            
            return score
        
        return sorted(experiences, key=rank_score, reverse=True)


class ExperienceManager:
    """经验管理器主类"""
    
    def __init__(self, db_path: str = "experience_manager.db"):
        self.storage = ExperienceStorage(db_path)
        self.analyzer = ExperienceAnalyzer()
        self.recommender = ExperienceRecommender(self.storage)
        self.metrics = ExperienceMetrics(0, 0.0, 0.0, [], 0.0, 0.0, 0.0)
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 启动后台任务
        self._start_background_tasks()
    
    def create_experience(self, title: str, content: str, 
                         experience_type: ExperienceType,
                         context: Dict[str, Any] = None,
                         outcome: Dict[str, Any] = None,
                         tags: List[str] = None) -> str:
        """创建新经验"""
        try:
            experience_id = str(uuid.uuid4())
            
            # 确定经验层次
            level = self._determine_level(content, context)
            
            # 创建经验对象
            experience = Experience(
                id=experience_id,
                title=title,
                content=content,
                experience_type=experience_type,
                level=level,
                context=context or {},
                outcome=outcome or {},
                tags=tags or [],
                quality_score=0.0,  # 初始质量分数
                created_at=datetime.now(),
                updated_at=datetime.now(),
                access_count=0,
                success_rate=0.0,
                related_experiences=[],
                knowledge_extracted=[],
                application_scenarios=[]
            )
            
            # 分析和提取知识
            self._process_experience(experience)
            
            # 存储经验
            if self.storage.store_experience(experience):
                self._update_metrics()
                return experience_id
            else:
                raise Exception("存储经验失败")
                
        except Exception as e:
            logging.error(f"创建经验失败: {e}")
            raise
    
    def _determine_level(self, content: str, context: Dict[str, Any]) -> ExperienceLevel:
        """确定经验层次"""
        content_length = len(content)
        
        # 基于内容长度和复杂度判断
        if content_length < 200:
            return ExperienceLevel.SHORT_TERM
        elif content_length < 1000:
            return ExperienceLevel.MEDIUM_TERM
        else:
            return ExperienceLevel.LONG_TERM
    
    def _process_experience(self, experience: Experience):
        """处理经验（分析、提取知识、评估质量）"""
        try:
            # 提取知识
            experience.knowledge_extracted = self.analyzer.extract_knowledge(experience)
            
            # 评估质量
            quality_level = self.analyzer.analyze_experience_quality(experience)
            experience.quality_score = self._quality_level_to_score(quality_level)
            
            # 查找相关经验并建立关联
            self._find_and_link_related_experiences(experience)
            
            # 更新经验
            experience.updated_at = datetime.now()
            
        except Exception as e:
            logging.error(f"处理经验失败: {e}")
    
    def _quality_level_to_score(self, quality_level: QualityLevel) -> float:
        """将质量等级转换为分数"""
        score_map = {
            QualityLevel.HIGH: 0.9,
            QualityLevel.MEDIUM: 0.7,
            QualityLevel.LOW: 0.5,
            QualityLevel.NEEDS_IMPROVEMENT: 0.3
        }
        return score_map.get(quality_level, 0.5)
    
    def _find_and_link_related_experiences(self, experience: Experience):
        """查找并链接相关经验"""
        try:
            # 搜索相似经验
            similar_experiences = self.storage.search_experiences(
                experience.title, {'min_quality': 0.5}
            )
            
            related_ids = []
            for similar_exp in similar_experiences:
                if similar_exp.id != experience.id:
                    similarity = self.analyzer.calculate_similarity(experience, similar_exp)
                    if similarity > 0.6:
                        related_ids.append(similar_exp.id)
                        
                        # 建立双向关联（在实际实现中需要在数据库中创建关联记录）
                        if experience.id not in similar_exp.related_experiences:
                            similar_exp.related_experiences.append(experience.id)
                            self.storage.store_experience(similar_exp)
            
            experience.related_experiences = related_ids
            
        except Exception as e:
            logging.error(f"查找相关经验失败: {e}")
    
    def retrieve_experience(self, experience_id: str) -> Optional[Experience]:
        """检索经验"""
        experience = self.storage.retrieve_experience(experience_id)
        if experience:
            # 更新访问统计
            experience.access_count += 1
            experience.updated_at = datetime.now()
            self.storage.store_experience(experience)
        return experience
    
    def search_experiences(self, query: str, filters: Dict[str, Any] = None) -> List[Experience]:
        """搜索经验"""
        return self.storage.search_experiences(query, filters)
    
    def get_recommendations(self, context: Dict[str, Any], 
                          user_preferences: Dict[str, Any] = None,
                          limit: int = 5) -> List[Experience]:
        """获取经验推荐"""
        return self.recommender.recommend_experiences(context, user_preferences, limit)
    
    def apply_experience(self, experience_id: str, application_context: Dict[str, Any],
                        success: bool, effectiveness_score: float = 0.0) -> bool:
        """记录经验应用"""
        try:
            experience = self.retrieve_experience(experience_id)
            if not experience:
                return False
            
            # 更新成功率
            total_applications = experience.access_count
            current_success_rate = experience.success_rate
            
            if total_applications > 0:
                new_success_rate = (current_success_rate * (total_applications - 1) + 
                                  (1.0 if success else 0.0)) / total_applications
            else:
                new_success_rate = 1.0 if success else 0.0
            
            experience.success_rate = new_success_rate
            
            # 更新应用场景
            scenario = f"{application_context.get('situation', '')}"
            if scenario and scenario not in experience.application_scenarios:
                experience.application_scenarios.append(scenario)
            
            # 重新评估质量
            quality_level = self.analyzer.analyze_experience_quality(experience)
            experience.quality_score = self._quality_level_to_score(quality_level)
            
            experience.updated_at = datetime.now()
            
            # 存储更新后的经验
            success = self.storage.store_experience(experience)
            
            if success:
                self._log_application(experience_id, application_context, success, effectiveness_score)
                self._update_metrics()
            
            return success
            
        except Exception as e:
            logging.error(f"记录经验应用失败: {e}")
            return False
    
    def _log_application(self, experience_id: str, context: Dict[str, Any],
                        success: bool, effectiveness_score: float):
        """记录应用日志"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO application_logs 
                    (experience_id, application_context, success, effectiveness_score, applied_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    experience_id,
                    json.dumps(context),
                    success,
                    effectiveness_score,
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"记录应用日志失败: {e}")
    
    def get_experience_analytics(self) -> ExperienceMetrics:
        """获取经验分析指标"""
        return self.metrics
    
    def _update_metrics(self):
        """更新指标"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取总体统计
                cursor.execute('SELECT COUNT(*) FROM experiences')
                total_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT AVG(success_rate) FROM experiences WHERE success_rate > 0')
                avg_success_rate = cursor.fetchone()[0] or 0.0
                
                cursor.execute('SELECT AVG(quality_score) FROM experiences')
                avg_quality = cursor.fetchone()[0] or 0.0
                
                cursor.execute('SELECT id FROM experiences ORDER BY access_count DESC LIMIT 5')
                most_used = [row[0] for row in cursor.fetchall()]
                
                # 计算学习效果
                learning_effectiveness = self._calculate_learning_effectiveness()
                
                # 计算知识保留率
                retention_rate = self._calculate_retention_rate()
                
                # 计算应用成功率
                application_success = self._calculate_application_success()
                
                self.metrics = ExperienceMetrics(
                    total_experiences=total_count,
                    success_rate=avg_success_rate,
                    average_quality_score=avg_quality,
                    most_used_experiences=most_used,
                    learning_effectiveness=learning_effectiveness,
                    knowledge_retention_rate=retention_rate,
                    application_success_rate=application_success
                )
                
        except Exception as e:
            logging.error(f"更新指标失败: {e}")
    
    def _calculate_learning_effectiveness(self) -> float:
        """计算学习效果"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT AVG(quality_score) FROM experiences 
                    WHERE created_at > datetime('now', '-30 days')
                ''')
                recent_quality = cursor.fetchone()[0] or 0.0
                return recent_quality
        except:
            return 0.0
    
    def _calculate_retention_rate(self) -> float:
        """计算知识保留率"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM knowledge_extractions')
                total_extractions = cursor.fetchone()[0] or 1
                
                cursor.execute('SELECT COUNT(DISTINCT experience_id) FROM experiences')
                experiences_with_knowledge = cursor.fetchone()[0] or 1
                
                return min(experiences_with_knowledge / total_extractions, 1.0)
        except:
            return 0.0
    
    def _calculate_application_success(self) -> float:
        """计算应用成功率"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM application_logs WHERE success = 1')
                successful_apps = cursor.fetchone()[0] or 0
                
                cursor.execute('SELECT COUNT(*) FROM application_logs')
                total_apps = cursor.fetchone()[0] or 1
                
                return successful_apps / total_apps
        except:
            return 0.0
    
    def clean_low_quality_experiences(self, min_quality: float = 0.3) -> int:
        """清理低质量经验"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM experiences WHERE quality_score < ?', (min_quality,))
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self._update_metrics()
                
                return deleted_count
        except Exception as e:
            logging.error(f"清理低质量经验失败: {e}")
            return 0
    
    def export_experiences(self, format_type: str = "json") -> str:
        """导出经验数据"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM experiences')
                rows = cursor.fetchall()
                
                experiences = []
                for row in rows:
                    exp_dict = {
                        'id': row[0], 'title': row[1], 'content': row[2],
                        'experience_type': row[3], 'level': row[4],
                        'context': json.loads(row[5] or '{}'),
                        'outcome': json.loads(row[6] or '{}'),
                        'tags': json.loads(row[7] or '[]'),
                        'quality_score': row[8], 'created_at': row[9],
                        'updated_at': row[10], 'access_count': row[11],
                        'success_rate': row[12],
                        'related_experiences': json.loads(row[13] or '[]'),
                        'knowledge_extracted': json.loads(row[14] or '[]'),
                        'application_scenarios': json.loads(row[15] or '[]')
                    }
                    experiences.append(exp_dict)
            
            if format_type.lower() == "json":
                return json.dumps(experiences, ensure_ascii=False, indent=2)
            else:
                # 可以扩展其他格式
                return json.dumps(experiences, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"导出经验数据失败: {e}")
            return ""
    
    def _start_background_tasks(self):
        """启动后台任务"""
        def periodic_cleanup():
            while True:
                try:
                    time.sleep(3600)  # 每小时执行一次
                    self.clean_low_quality_experiences()
                    self._update_metrics()
                except Exception as e:
                    logging.error(f"后台清理任务失败: {e}")
        
        # 启动后台线程
        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
        cleanup_thread.start()
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "total_experiences": self.metrics.total_experiences,
            "average_quality": self.metrics.average_quality_score,
            "success_rate": self.metrics.success_rate,
            "learning_effectiveness": self.metrics.learning_effectiveness,
            "knowledge_retention_rate": self.metrics.knowledge_retention_rate,
            "application_success_rate": self.metrics.application_success_rate,
            "database_path": self.storage.db_path,
            "status": "运行中"
        }


# 使用示例和测试代码
if __name__ == "__main__":
    # 初始化经验管理器
    manager = ExperienceManager()
    
    # 创建示例经验
    exp_id = manager.create_experience(
        title="项目风险识别经验",
        content="在项目管理过程中，通过定期风险评估会议和头脑风暴，可以提前识别潜在风险。建议每周举行一次风险评审会议，使用SWOT分析框架。",
        experience_type=ExperienceType.PROBLEM_SOLVING,
        context={"domain": "项目管理", "method": "SWOT分析"},
        outcome={"success": True, "lessons": "定期评审很重要"},
        tags=["风险管理", "项目管理", "SWOT", "团队协作"]
    )
    
    print(f"创建经验ID: {exp_id}")
    
    # 搜索经验
    results = manager.search_experiences("风险")
    print(f"找到 {len(results)} 个相关经验")
    
    # 获取推荐
    recommendations = manager.get_recommendations(
        context={"domain": "项目管理", "problem_type": "风险控制"},
        limit=3
    )
    print(f"推荐了 {len(recommendations)} 个经验")
    
    # 记录经验应用
    manager.apply_experience(
        exp_id,
        {"situation": "新项目启动", "team_size": 5},
        success=True,
        effectiveness_score=0.8
    )
    
    # 获取系统状态
    status = manager.get_system_status()
    print(f"系统状态: {status}")
    
    # 获取分析指标
    metrics = manager.get_experience_analytics()
    print(f"经验指标: {metrics}")