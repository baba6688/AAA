#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A6地缘政治监控器
Geopolitical Risk Monitor

功能：
1. 全球政治事件监控
2. 贸易政策变化跟踪
3. 制裁和关税影响分析
4. 地缘政治风险评分
5. 市场影响评估
6. 风险预警系统


日期: 2025-11-05
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EventSeverity(Enum):
    """事件严重程度枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EventCategory(Enum):
    """事件类别枚举"""
    POLITICAL_INSTABILITY = "政治不稳定"
    TRADE_DISPUTE = "贸易争端"
    SANCTIONS = "制裁"
    MILITARY_CONFLICT = "军事冲突"
    ELECTION = "选举"
    POLICY_CHANGE = "政策变化"
    DIPLOMATIC_TENSION = "外交紧张"
    ECONOMIC_SANCTION = "经济制裁"
    REGULATORY_CHANGE = "监管变化"


class AssetClass(Enum):
    """资产类别枚举"""
    EQUITIES = "股票"
    BONDS = "债券"
    COMMODITIES = "商品"
    CURRENCIES = "货币"
    CRYPTO = "加密货币"
    REAL_ESTATE = "房地产"


@dataclass
class GeopoliticalEvent:
    """地缘政治事件数据类"""
    event_id: str
    title: str
    description: str
    category: EventCategory
    severity: EventSeverity
    countries: List[str]
    date_time: datetime
    source: str
    impact_score: float = 0.0
    affected_assets: List[AssetClass] = None
    market_impact: Dict[str, float] = None
    
    def __post_init__(self):
        if self.affected_assets is None:
            self.affected_assets = []
        if self.market_impact is None:
            self.market_impact = {}


@dataclass
class RiskAssessment:
    """风险评估数据类"""
    region: str
    risk_score: float
    trend: str  # "increasing", "decreasing", "stable"
    key_factors: List[str]
    last_updated: datetime
    confidence_level: float


@dataclass
class MarketImpact:
    """市场影响数据类"""
    asset_class: AssetClass
    region: str
    expected_impact: float
    volatility_change: float
    liquidity_impact: float
    time_horizon: str  # "immediate", "short_term", "medium_term", "long_term"


class DataSource:
    """数据源基类"""
    def __init__(self, name: str, api_key: str = None):
        self.name = name
        self.api_key = api_key
        self.base_url = ""
        
    async def fetch_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """获取事件数据"""
        raise NotImplementedError
        
    def validate_data(self, data: Dict) -> bool:
        """验证数据格式"""
        return True


class NewsAPI(DataSource):
    """新闻API数据源"""
    def __init__(self, api_key: str):
        super().__init__("NewsAPI", api_key)
        self.base_url = "https://newsapi.org/v2"
        
    async def fetch_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """获取新闻事件"""
        headers = {'X-API-Key': self.api_key}
        params = {
            'q': 'geopolitical OR politics OR sanctions OR trade war OR election',
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt'
        }
        
        try:
            response = requests.get(f"{self.base_url}/everything", headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            events = []
            for article in data.get('articles', []):
                events.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'url': article.get('url', '')
                })
            return events
        except Exception as e:
            logger.error(f"NewsAPI获取数据失败: {e}")
            return []


class OfficialAnnouncements(DataSource):
    """官方公告数据源"""
    def __init__(self):
        super().__init__("OfficialAnnouncements")
        
    async def fetch_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """获取官方公告"""
        # 模拟官方公告数据
        current_time = datetime.now()
        events = [
            {
                'title': '美国宣布新的贸易制裁措施',
                'description': '美国政府宣布对特定国家实施新的贸易制裁',
                'source': '美国白宫',
                'published_at': current_time.isoformat(),
                'type': 'official_statement'
            },
            {
                'title': '欧盟发布贸易政策更新',
                'description': '欧盟委员会发布最新的贸易政策指导方针',
                'source': '欧盟委员会',
                'published_at': current_time.isoformat(),
                'type': 'policy_update'
            },
            {
                'title': '中国发布新的贸易政策',
                'description': '中国政府宣布调整对外贸易政策',
                'source': '中国商务部',
                'published_at': current_time.isoformat(),
                'type': 'policy_update'
            }
        ]
        return events


class GeopoliticalMonitor:
    """地缘政治监控器主类"""
    
    def __init__(self, db_path: str = "geopolitical_monitor.db"):
        self.db_path = db_path
        self.events_db = {}
        self.risk_scores = {}
        self.market_impacts = {}
        self.alert_thresholds = {
            EventSeverity.CRITICAL: 8.0,
            EventSeverity.HIGH: 6.0,
            EventSeverity.MEDIUM: 4.0,
            EventSeverity.LOW: 2.0
        }
        
        # 数据源
        self.data_sources = {}
        
        # 缓存
        self.event_cache = deque(maxlen=1000)
        self.risk_cache = {}
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 初始化数据库
        self._init_database()
        
        # 启动监控线程
        self.monitoring_active = False
        self.monitor_thread = None
        
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    category TEXT,
                    severity TEXT,
                    countries TEXT,
                    date_time TEXT,
                    source TEXT,
                    impact_score REAL,
                    affected_assets TEXT,
                    market_impact TEXT,
                    created_at TEXT
                )
            ''')
            
            # 创建风险评分表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_scores (
                    region TEXT PRIMARY KEY,
                    risk_score REAL,
                    trend TEXT,
                    key_factors TEXT,
                    last_updated TEXT,
                    confidence_level REAL
                )
            ''')
            
            # 创建市场影响表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_impacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_class TEXT,
                    region TEXT,
                    expected_impact REAL,
                    volatility_change REAL,
                    liquidity_impact REAL,
                    time_horizon TEXT,
                    timestamp TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("数据库初始化成功")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
    
    def add_data_source(self, source: DataSource):
        """添加数据源"""
        self.data_sources[source.name] = source
        logger.info(f"添加数据源: {source.name}")
    
    def classify_event(self, event_data: Dict) -> Tuple[EventCategory, EventSeverity]:
        """事件分类和严重程度评估"""
        title = event_data.get('title', '').lower()
        description = event_data.get('description', '').lower()
        content = f"{title} {description}"
        
        # 关键词匹配
        category_keywords = {
            EventCategory.POLITICAL_INSTABILITY: ['政治不稳定', '政变', '抗议', '政治危机', 'political instability', 'coup', 'protest'],
            EventCategory.TRADE_DISPUTE: ['贸易争端', '贸易战', '关税', 'trade dispute', 'trade war', 'tariff'],
            EventCategory.SANCTIONS: ['制裁', '限制', 'sanction', 'embargo', 'restriction'],
            EventCategory.MILITARY_CONFLICT: ['军事冲突', '战争', '攻击', 'military conflict', 'war', 'attack'],
            EventCategory.ELECTION: ['选举', '投票', 'election', 'vote', 'campaign'],
            EventCategory.POLICY_CHANGE: ['政策变化', '新政策', 'policy change', 'new policy'],
            EventCategory.DIPLOMATIC_TENSION: ['外交紧张', '外交危机', 'diplomatic tension', 'diplomatic crisis'],
            EventCategory.ECONOMIC_SANCTION: ['经济制裁', '经济限制', 'economic sanction', 'economic restriction'],
            EventCategory.REGULATORY_CHANGE: ['监管变化', '新法规', 'regulatory change', 'new regulation']
        }
        
        # 严重程度关键词
        severity_keywords = {
            EventSeverity.CRITICAL: ['紧急', '严重', 'critical', 'emergency', 'severe', 'major'],
            EventSeverity.HIGH: ['重大', '重要', 'high', 'major', 'significant'],
            EventSeverity.MEDIUM: ['中等', 'moderate', 'medium'],
            EventSeverity.LOW: ['轻微', 'low', 'minor', 'small']
        }
        
        # 分类
        category = EventCategory.POLICY_CHANGE  # 默认分类
        max_score = 0
        
        for cat, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > max_score:
                max_score = score
                category = cat
        
        # 严重程度评估
        severity = EventSeverity.MEDIUM  # 默认严重程度
        max_severity_score = 0
        
        for sev, keywords in severity_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > max_severity_score:
                max_severity_score = score
                severity = sev
        
        return category, severity
    
    def assess_market_impact(self, event: GeopoliticalEvent) -> Dict[AssetClass, float]:
        """评估市场影响"""
        impact_scores = {}
        
        # 基于事件类别的基本影响映射
        category_impact = {
            EventCategory.POLITICAL_INSTABILITY: {
                AssetClass.EQUITIES: -0.7,
                AssetClass.BONDS: 0.3,
                AssetClass.COMMODITIES: -0.4,
                AssetClass.CURRENCIES: -0.6,
                AssetClass.CRYPTO: -0.5,
                AssetClass.REAL_ESTATE: -0.3
            },
            EventCategory.TRADE_DISPUTE: {
                AssetClass.EQUITIES: -0.5,
                AssetClass.BONDS: 0.2,
                AssetClass.COMMODITIES: -0.3,
                AssetClass.CURRENCIES: -0.4,
                AssetClass.CRYPTO: -0.2,
                AssetClass.REAL_ESTATE: -0.2
            },
            EventCategory.SANCTIONS: {
                AssetClass.EQUITIES: -0.6,
                AssetClass.BONDS: 0.1,
                AssetClass.COMMODITIES: -0.5,
                AssetClass.CURRENCIES: -0.7,
                AssetClass.CRYPTO: -0.3,
                AssetClass.REAL_ESTATE: -0.4
            },
            EventCategory.MILITARY_CONFLICT: {
                AssetClass.EQUITIES: -0.8,
                AssetClass.BONDS: 0.5,
                AssetClass.COMMODITIES: -0.6,
                AssetClass.CURRENCIES: -0.8,
                AssetClass.CRYPTO: -0.7,
                AssetClass.REAL_ESTATE: -0.5
            },
            EventCategory.ELECTION: {
                AssetClass.EQUITIES: -0.3,
                AssetClass.BONDS: 0.1,
                AssetClass.COMMODITIES: -0.2,
                AssetClass.CURRENCIES: -0.4,
                AssetClass.CRYPTO: -0.1,
                AssetClass.REAL_ESTATE: -0.1
            }
        }
        
        # 获取基础影响
        base_impact = category_impact.get(event.category, {})
        
        # 根据严重程度调整影响
        severity_multiplier = {
            EventSeverity.LOW: 0.3,
            EventSeverity.MEDIUM: 0.6,
            EventSeverity.HIGH: 0.8,
            EventSeverity.CRITICAL: 1.0
        }
        
        multiplier = severity_multiplier.get(event.severity, 0.6)
        
        # 计算最终影响
        for asset_class, base_score in base_impact.items():
            impact_scores[asset_class] = base_score * multiplier
        
        return impact_scores
    
    def calculate_risk_score(self, events: List[GeopoliticalEvent], region: str) -> float:
        """计算地缘政治风险评分"""
        if not events:
            return 0.0
        
        # 事件权重
        severity_weights = {
            EventSeverity.LOW: 1.0,
            EventSeverity.MEDIUM: 2.5,
            EventSeverity.HIGH: 5.0,
            EventSeverity.CRITICAL: 10.0
        }
        
        # 时间衰减因子
        now = datetime.now()
        total_score = 0.0
        total_weight = 0.0
        
        for event in events:
            # 时间衰减 (7天半衰期)
            days_diff = (now - event.date_time).days
            time_decay = np.exp(-days_diff / 7.0)
            
            # 事件权重
            event_weight = severity_weights.get(event.severity, 1.0)
            
            # 地区相关性
            region_relevance = 1.0 if region in event.countries else 0.3
            
            # 计算加权得分
            weighted_score = event.impact_score * event_weight * time_decay * region_relevance
            total_score += weighted_score
            total_weight += event_weight * time_decay
        
        # 标准化到0-10分
        if total_weight > 0:
            normalized_score = min(10.0, total_score / total_weight * 2.0)
        else:
            normalized_score = 0.0
        
        return round(normalized_score, 2)
    
    def detect_trend(self, risk_scores: List[float]) -> str:
        """检测风险趋势"""
        if len(risk_scores) < 2:
            return "stable"
        
        recent_scores = risk_scores[-5:]  # 最近5个评分
        if len(recent_scores) < 2:
            return "stable"
        
        # 计算趋势
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        if slope > 0.2:
            return "increasing"
        elif slope < -0.2:
            return "decreasing"
        else:
            return "stable"
    
    async def collect_events(self, start_date: datetime, end_date: datetime) -> List[GeopoliticalEvent]:
        """收集地缘政治事件"""
        all_events = []
        
        # 从各个数据源收集事件
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_source = {
                executor.submit(
                    asyncio.run, source.fetch_events(start_date, end_date)
                ): source.name 
                for source in self.data_sources.values()
            }
            
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    events_data = future.result()
                    for event_data in events_data:
                        # 分类事件
                        category, severity = self.classify_event(event_data)
                        
                        # 创建事件对象
                        event = GeopoliticalEvent(
                            event_id=f"{source_name}_{hash(str(event_data))}",
                            title=event_data.get('title', ''),
                            description=event_data.get('description', ''),
                            category=category,
                            severity=severity,
                            countries=[],  # 需要从内容中提取
                            date_time=datetime.fromisoformat(
                                event_data.get('published_at', datetime.now().isoformat())
                            ),
                            source=source_name
                        )
                        
                        # 评估市场影响
                        market_impact = self.assess_market_impact(event)
                        event.market_impact = {k.value: v for k, v in market_impact.items()}
                        if market_impact:
                            event.impact_score = abs(sum(market_impact.values())) / len(market_impact)
                        else:
                            event.impact_score = 0.0
                        
                        all_events.append(event)
                        
                except Exception as e:
                    logger.error(f"从{source_name}收集事件失败: {e}")
        
        return all_events
    
    def save_event(self, event: GeopoliticalEvent):
        """保存事件到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO events 
                (event_id, title, description, category, severity, countries, 
                 date_time, source, impact_score, affected_assets, market_impact, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.title,
                event.description,
                event.category.value,
                event.severity.value,
                json.dumps(event.countries),
                event.date_time.isoformat(),
                event.source,
                event.impact_score,
                json.dumps([asset.value for asset in event.affected_assets]),
                json.dumps(event.market_impact),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存事件失败: {e}")
    
    def update_risk_scores(self, events: List[GeopoliticalEvent]):
        """更新风险评分"""
        # 按地区分组事件
        region_events = defaultdict(list)
        for event in events:
            if event.countries:
                for country in event.countries:
                    region_events[country].append(event)
            else:
                region_events["全球"].append(event)
        
        # 计算每个地区的风险评分
        for region, region_event_list in region_events.items():
            risk_score = self.calculate_risk_score(region_event_list, region)
            
            # 获取历史评分用于趋势分析
            historical_scores = []
            if region in self.risk_scores:
                historical_scores.append(self.risk_scores[region]['score'])
            historical_scores.append(risk_score)
            
            trend = self.detect_trend(historical_scores)
            
            # 识别关键风险因素
            key_factors = []
            for event in region_event_list[-5:]:  # 最近5个事件
                if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
                    key_factors.append(f"{event.category.value}: {event.title[:50]}...")
            
            risk_assessment = RiskAssessment(
                region=region,
                risk_score=risk_score,
                trend=trend,
                key_factors=key_factors[:5],  # 最多5个关键因素
                last_updated=datetime.now(),
                confidence_level=min(0.9, len(region_event_list) * 0.1 + 0.3)
            )
            
            self.risk_scores[region] = asdict(risk_assessment)
            
            # 保存到数据库
            self.save_risk_assessment(risk_assessment)
    
    def save_risk_assessment(self, assessment: RiskAssessment):
        """保存风险评估到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO risk_scores
                (region, risk_score, trend, key_factors, last_updated, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                assessment.region,
                assessment.risk_score,
                assessment.trend,
                json.dumps(assessment.key_factors),
                assessment.last_updated.isoformat(),
                assessment.confidence_level
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存风险评估失败: {e}")
    
    def generate_alerts(self, risk_scores: Dict[str, Any]) -> List[Dict]:
        """生成风险预警"""
        alerts = []
        
        for region, data in risk_scores.items():
            risk_score = data['risk_score']
            trend = data['trend']
            
            # 检查是否超过预警阈值
            for severity, threshold in self.alert_thresholds.items():
                if risk_score >= threshold:
                    alert = {
                        'alert_id': f"alert_{region}_{int(time.time())}",
                        'region': region,
                        'severity': severity.value,
                        'risk_score': risk_score,
                        'trend': trend,
                        'message': f"{region}地缘政治风险评分达到{risk_score}分（{severity.value}级别）",
                        'timestamp': datetime.now().isoformat(),
                        'key_factors': data.get('key_factors', [])
                    }
                    alerts.append(alert)
                    break
            
            # 检查风险趋势
            if trend == "increasing" and risk_score > 5.0:
                alert = {
                    'alert_id': f"trend_{region}_{int(time.time())}",
                    'region': region,
                    'severity': "趋势预警",
                    'risk_score': risk_score,
                    'trend': trend,
                    'message': f"{region}地缘政治风险呈上升趋势，当前评分{risk_score}分",
                    'timestamp': datetime.now().isoformat(),
                    'key_factors': data.get('key_factors', [])
                }
                alerts.append(alert)
        
        return alerts
    
    def backtest_historical_impact(self, historical_events: List[GeopoliticalEvent], 
                                 market_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """历史事件影响回测"""
        results = {}
        
        for asset_class in AssetClass:
            if asset_class.value not in market_data:
                continue
                
            asset_returns = market_data[asset_class.value]
            
            # 计算事件窗口期影响
            event_impacts = []
            for event in historical_events:
                # 查找事件发生前后的市场数据
                event_date = event.date_time
                
                # 简化处理：假设有足够的历史数据
                if hasattr(event, 'market_impact') and asset_class.value in event.market_impact:
                    predicted_impact = event.market_impact[asset_class.value]
                    event_impacts.append(predicted_impact)
            
            if event_impacts:
                # 计算预测准确性
                mean_predicted = np.mean(event_impacts)
                volatility = np.std(event_impacts)
                
                results[asset_class.value] = {
                    'mean_predicted_impact': round(mean_predicted, 4),
                    'volatility': round(volatility, 4),
                    'event_count': len(event_impacts),
                    'accuracy_score': round(1.0 / (1.0 + volatility), 4)  # 简化的准确性评分
                }
        
        return results
    
    async def start_monitoring(self, interval_minutes: int = 60):
        """启动实时监控"""
        if self.monitoring_active:
            logger.warning("监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval_minutes,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"地缘政治监控已启动，监控间隔: {interval_minutes}分钟")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("地缘政治监控已停止")
    
    def _monitoring_loop(self, interval_minutes: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集最近24小时的事件
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)
                
                events = asyncio.run(self.collect_events(start_date, end_date))
                
                # 更新风险评分
                self.update_risk_scores(events)
                
                # 生成预警
                alerts = self.generate_alerts(self.risk_scores)
                
                # 处理预警
                for alert in alerts:
                    self._handle_alert(alert)
                
                # 保存事件
                for event in events:
                    self.save_event(event)
                
                logger.info(f"监控周期完成: 收集到{len(events)}个事件，生成{len(alerts)}个预警")
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
            
            # 等待下次监控
            time.sleep(interval_minutes * 60)
    
    def _handle_alert(self, alert: Dict):
        """处理预警"""
        logger.warning(f"地缘政治风险预警: {alert['message']}")
        
        # 这里可以添加具体的预警处理逻辑
        # 例如：发送邮件、短信、Slack消息等
        
    def get_current_risk_scores(self) -> Dict[str, Any]:
        """获取当前风险评分"""
        return self.risk_scores.copy()
    
    def get_recent_events(self, days: int = 7) -> List[GeopoliticalEvent]:
        """获取最近事件"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            cursor.execute('''
                SELECT * FROM events 
                WHERE date_time >= ? 
                ORDER BY date_time DESC
            ''', (start_date.isoformat(),))
            
            events = []
            for row in cursor.fetchall():
                event = GeopoliticalEvent(
                    event_id=row[0],
                    title=row[1],
                    description=row[2],
                    category=EventCategory(row[3]),
                    severity=EventSeverity(row[4]),
                    countries=json.loads(row[5]),
                    date_time=datetime.fromisoformat(row[6]),
                    source=row[7],
                    impact_score=row[8],
                    affected_assets=[AssetClass(a) for a in json.loads(row[9])],
                    market_impact=json.loads(row[10])
                )
                events.append(event)
            
            conn.close()
            return events
            
        except Exception as e:
            logger.error(f"获取最近事件失败: {e}")
            return []
    
    def export_risk_report(self, output_path: str = "geopolitical_risk_report.json"):
        """导出风险报告"""
        # 序列化风险评分数据
        serializable_risk_scores = {}
        for region, data in self.risk_scores.items():
            serializable_data = data.copy()
            if 'last_updated' in serializable_data and isinstance(serializable_data['last_updated'], datetime):
                serializable_data['last_updated'] = serializable_data['last_updated'].isoformat()
            serializable_risk_scores[region] = serializable_data
        
        # 直接查询数据库获取最近事件数量
        recent_events_count = 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            start_date = datetime.now() - timedelta(days=7)
            cursor.execute('SELECT COUNT(*) FROM events WHERE date_time >= ?', (start_date.isoformat(),))
            result = cursor.fetchone()
            recent_events_count = result[0] if result else 0
            conn.close()
        except Exception as e:
            logger.warning(f"无法获取最近事件数量: {e}")
        
        # 生成预警（确保可序列化）
        alerts = []
        try:
            raw_alerts = self.generate_alerts(self.risk_scores)
            for alert in raw_alerts:
                serializable_alert = alert.copy()
                # 确保所有值都是可序列化的
                for key, value in serializable_alert.items():
                    if isinstance(value, datetime):
                        serializable_alert[key] = value.isoformat()
                alerts.append(serializable_alert)
        except Exception as e:
            logger.warning(f"生成预警失败: {e}")
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'risk_scores': serializable_risk_scores,
            'recent_events_count': recent_events_count,
            'active_alerts': alerts,
            'summary': {
                'highest_risk_region': max(self.risk_scores.items(), key=lambda x: x[1]['risk_score']) if self.risk_scores else None,
                'average_risk_score': np.mean([data['risk_score'] for data in self.risk_scores.values()]) if self.risk_scores else 0,
                'regions_at_risk': len([data for data in self.risk_scores.values() if data['risk_score'] > 5.0])
            }
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"风险报告已导出到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"导出报告失败: {e}")
            return False


# 使用示例和测试函数
def create_sample_monitor() -> GeopoliticalMonitor:
    """创建示例监控器"""
    monitor = GeopoliticalMonitor()
    
    # 添加数据源（示例）
    # monitor.add_data_source(NewsAPI("your_newsapi_key"))
    monitor.add_data_source(OfficialAnnouncements())
    
    return monitor


def demo_monitor_functionality():
    """演示监控器功能"""
    print("=== A6地缘政治监控器演示 ===\n")
    
    # 创建监控器
    monitor = create_sample_monitor()
    
    # 模拟收集事件
    print("1. 收集地缘政治事件...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    events = asyncio.run(monitor.collect_events(start_date, end_date))
    print(f"收集到 {len(events)} 个事件")
    
    # 显示事件详情
    for i, event in enumerate(events[:3]):  # 只显示前3个
        print(f"\n事件 {i+1}:")
        print(f"  标题: {event.title}")
        print(f"  类别: {event.category.value}")
        print(f"  严重程度: {event.severity.value}")
        print(f"  影响评分: {event.impact_score:.2f}")
        print(f"  市场影响: {event.market_impact}")
    
    # 更新风险评分
    print("\n2. 更新风险评分...")
    monitor.update_risk_scores(events)
    
    # 显示风险评分
    print("\n当前风险评分:")
    for region, data in monitor.get_current_risk_scores().items():
        print(f"  {region}: {data['risk_score']}分 (趋势: {data['trend']})")
    
    # 生成预警
    print("\n3. 生成风险预警...")
    alerts = monitor.generate_alerts(monitor.get_current_risk_scores())
    
    if alerts:
        print("发现以下预警:")
        for alert in alerts:
            print(f"  - {alert['message']}")
    else:
        print("暂无预警")
    
    # 导出报告
    print("\n4. 导出风险报告...")
    success = monitor.export_risk_report()
    if success:
        print("风险报告导出成功")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    # 运行演示
    demo_monitor_functionality()