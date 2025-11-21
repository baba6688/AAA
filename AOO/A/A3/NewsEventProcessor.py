"""
A3新闻事件处理器
实现多源新闻数据抓取、情感分析、事件分类和实时处理功能
"""

import asyncio
import json
import hashlib
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty

import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """新闻数据结构"""
    id: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    category: str
    sentiment_score: float = 0.0
    importance_score: float = 0.0
    tags: List[str] = None
    impact_score: float = 0.0
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.keywords is None:
            self.keywords = []

@dataclass
class Event:
    """事件数据结构"""
    id: str
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    category: str
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    related_news: List[str]  # 新闻ID列表
    timeline: List[Dict] = None
    
    def __post_init__(self):
        if self.timeline is None:
            self.timeline = []

class NewsAPIClient:
    """新闻API客户端"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.headers = {
            "X-API-Key": api_key,
            "User-Agent": "NewsEventProcessor/1.0"
        }
    
    def get_top_headlines(self, country: str = "us", category: str = None, 
                         page_size: int = 100) -> List[Dict]:
        """获取头条新闻"""
        url = f"{self.base_url}/top-headlines"
        params = {
            "country": country,
            "pageSize": page_size
        }
        if category:
            params["category"] = category
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("articles", [])
        except Exception as e:
            logger.error(f"获取头条新闻失败: {e}")
            return []
    
    def search_news(self, query: str, from_date: str = None, 
                   to_date: str = None, language: str = "en") -> List[Dict]:
        """搜索新闻"""
        url = f"{self.base_url}/everything"
        params = {
            "q": query,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": 100
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("articles", [])
        except Exception as e:
            logger.error(f"搜索新闻失败: {e}")
            return []

class AlphaVantageNewsClient:
    """Alpha Vantage新闻客户端"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_news(self, tickers: List[str] = None, topics: List[str] = None) -> List[Dict]:
        """获取财经新闻"""
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key
        }
        if tickers:
            params["tickers"] = ",".join(tickers)
        if topics:
            params["topics"] = ",".join(topics)
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("feed", [])
        except Exception as e:
            logger.error(f"获取Alpha Vantage新闻失败: {e}")
            return []

class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self):
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        if not text:
            return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}
        
        # 使用TextBlob进行情感分析
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        # 使用VADER进行情感分析
        vader_scores = self.sia.polarity_scores(text)
        
        # 综合评分
        compound_score = (textblob_score + vader_scores["compound"]) / 2
        
        return {
            "compound": compound_score,
            "pos": vader_scores["pos"],
            "neu": vader_scores["neu"],
            "neg": vader_scores["neg"],
            "textblob": textblob_score
        }
    
    def get_sentiment_label(self, score: float) -> str:
        """获取情感标签"""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"

class KeywordExtractor:
    """关键词提取器"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
    
    def extract_keywords(self, texts: List[str], top_k: int = 10) -> List[str]:
        """提取关键词"""
        if not texts:
            return []
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # 计算每个文档的TF-IDF分数
            scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # 获取top-k关键词
            top_indices = scores.argsort()[-top_k:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            return keywords
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return []
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 分词
        tokens = word_tokenize(text)
        
        # 移除停用词和短词
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

class EventClassifier:
    """事件分类器"""
    
    def __init__(self):
        # 预定义的事件类别和关键词
        self.categories = {
            "market": ["stock", "market", "trading", "earnings", "revenue", "profit", "loss"],
            "economy": ["inflation", "gdp", "recession", "unemployment", "interest rate", "fed"],
            "technology": ["ai", "technology", "software", "hardware", "innovation", "startup"],
            "politics": ["election", "policy", "government", "congress", "president", "law"],
            "international": ["china", "europe", "asia", "trade war", "sanctions", "diplomacy"],
            "energy": ["oil", "gas", "renewable", "solar", "wind", "nuclear"],
            "healthcare": ["health", "medical", "pharmaceutical", "fda", "vaccine", "treatment"],
            "real_estate": ["housing", "real estate", "mortgage", "property", "construction"]
        }
    
    def classify_event(self, text: str) -> Tuple[str, float]:
        """分类事件"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = score
        
        if not scores:
            return "general", 0.0
        
        # 返回得分最高的类别
        best_category = max(scores.items(), key=lambda x: x[1])
        return best_category[0], best_category[1] / len(text.split())

class NewsDeduplicator:
    """新闻去重器"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.news_cache = deque(maxlen=1000)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            return 0.0
    
    def is_duplicate(self, news_item: NewsItem) -> bool:
        """检查是否为重复新闻"""
        for cached_news in self.news_cache:
            # 计算标题相似度
            title_sim = self.calculate_similarity(news_item.title, cached_news.title)
            
            # 计算内容相似度
            content_sim = self.calculate_similarity(news_item.content, cached_news.content)
            
            # 如果相似度超过阈值，认为是重复
            if title_sim > self.similarity_threshold or content_sim > self.similarity_threshold:
                return True
        
        return False
    
    def add_news(self, news_item: NewsItem):
        """添加新闻到缓存"""
        self.news_cache.append(news_item)

class ImpactAnalyzer:
    """影响度分析器"""
    
    def __init__(self):
        self.source_weights = {
            "reuters": 0.9,
            "bloomberg": 0.9,
            "cnn": 0.8,
            "bbc": 0.8,
            "financial times": 0.9,
            "wall street journal": 0.9,
            "marketwatch": 0.8,
            "yahoo finance": 0.7,
            "google news": 0.6
        }
    
    def calculate_impact_score(self, news_item: NewsItem, related_news: List[NewsItem]) -> float:
        """计算新闻影响度分数"""
        # 基础分数
        base_score = 0.5
        
        # 来源权重
        source_weight = self.source_weights.get(news_item.source.lower(), 0.5)
        
        # 情感强度
        sentiment_intensity = abs(news_item.sentiment_score)
        
        # 相关新闻数量
        related_count = len(related_news)
        related_score = min(related_count / 10, 1.0)  # 最多10篇相关新闻
        
        # 时间因子（越新的新闻影响越大）
        hours_old = (datetime.now() - news_item.published_at).total_seconds() / 3600
        time_factor = max(0, 1 - hours_old / 24)  # 24小时后影响度减为0
        
        # 综合计算
        impact_score = (
            base_score * 0.2 +
            source_weight * 0.3 +
            sentiment_intensity * 0.2 +
            related_score * 0.2 +
            time_factor * 0.1
        )
        
        return min(impact_score, 1.0)

class TimelineBuilder:
    """时间线构建器"""
    
    def __init__(self):
        self.events = {}
    
    def add_event(self, event: Event):
        """添加事件"""
        self.events[event.id] = event
    
    def build_timeline(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """构建时间线"""
        timeline = []
        
        for event in self.events.values():
            if start_date <= event.start_time <= end_date:
                timeline.append({
                    "id": event.id,
                    "title": event.title,
                    "start_time": event.start_time.isoformat(),
                    "end_time": event.end_time.isoformat() if event.end_time else None,
                    "category": event.category,
                    "impact_level": event.impact_level,
                    "description": event.description,
                    "timeline_points": event.timeline
                })
        
        # 按时间排序
        timeline.sort(key=lambda x: x["start_time"])
        return timeline
    
    def update_event_timeline(self, event_id: str, news_item: NewsItem):
        """更新事件时间线"""
        if event_id in self.events:
            event = self.events[event_id]
            event.timeline.append({
                "timestamp": datetime.now().isoformat(),
                "news_id": news_item.id,
                "title": news_item.title,
                "sentiment": news_item.sentiment_score,
                "impact": news_item.impact_score
            })

class NewsEventProcessor:
    """主新闻事件处理器"""
    
    def __init__(self, news_api_key: str = None, alpha_vantage_key: str = None):
        self.news_client = NewsAPIClient(news_api_key) if news_api_key else None
        self.alpha_client = AlphaVantageNewsClient(alpha_vantage_key) if alpha_vantage_key else None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.event_classifier = EventClassifier()
        self.deduplicator = NewsDeduplicator()
        self.impact_analyzer = ImpactAnalyzer()
        self.timeline_builder = TimelineBuilder()
        
        # 存储
        self.news_storage = {}
        self.event_storage = {}
        self.news_queue = Queue()
        self.processing_thread = None
        self.is_running = False
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info("新闻事件处理器初始化完成")
    
    def generate_news_id(self, news_item: NewsItem) -> str:
        """生成新闻唯一ID"""
        content = f"{news_item.title}{news_item.content}{news_item.source}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_news_item(self, raw_news: Dict) -> Optional[NewsItem]:
        """处理单条新闻"""
        try:
            # 解析新闻数据
            title = raw_news.get("title", "")
            content = raw_news.get("description", "") or raw_news.get("content", "")
            source = raw_news.get("source", {}).get("name", "") if isinstance(raw_news.get("source"), dict) else raw_news.get("source", "")
            url = raw_news.get("url", "")
            
            # 解析发布时间
            published_at_str = raw_news.get("publishedAt", "")
            if published_at_str:
                published_at = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
            else:
                published_at = datetime.now()
            
            # 创建新闻对象
            news_item = NewsItem(
                id="",  # 临时ID，稍后生成
                title=title,
                content=content,
                source=source,
                url=url,
                published_at=published_at,
                category="general"
            )
            
            # 生成唯一ID
            news_item.id = self.generate_news_id(news_item)
            
            # 检查重复
            if self.deduplicator.is_duplicate(news_item):
                return None
            
            # 情感分析
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(f"{title} {content}")
            news_item.sentiment_score = sentiment_result["compound"]
            
            # 事件分类
            category, confidence = self.event_classifier.classify_event(f"{title} {content}")
            news_item.category = category
            
            # 关键词提取
            news_item.keywords = self.keyword_extractor.extract_keywords([f"{title} {content}"], top_k=5)
            
            # 添加标签
            news_item.tags = [category, self.sentiment_analyzer.get_sentiment_label(news_item.sentiment_score)]
            
            # 添加到去重缓存
            self.deduplicator.add_news(news_item)
            
            return news_item
            
        except Exception as e:
            logger.error(f"处理新闻失败: {e}")
            return None
    
    def fetch_news_from_sources(self) -> List[NewsItem]:
        """从多个源获取新闻"""
        all_news = []
        
        # 从NewsAPI获取新闻
        if self.news_client:
            try:
                # 获取财经新闻
                business_news = self.news_client.get_top_headlines(category="business")
                technology_news = self.news_client.get_top_headlines(category="technology")
                
                for news in business_news + technology_news:
                    processed_news = self.process_news_item(news)
                    if processed_news:
                        all_news.append(processed_news)
                        
            except Exception as e:
                logger.error(f"从NewsAPI获取新闻失败: {e}")
        
        # 从Alpha Vantage获取财经新闻
        if self.alpha_client:
            try:
                financial_news = self.alpha_client.get_news(
                    tickers=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
                    topics=["technology", "finance"]
                )
                
                for news in financial_news:
                    processed_news = self.process_news_item(news)
                    if processed_news:
                        all_news.append(processed_news)
                        
            except Exception as e:
                logger.error(f"从Alpha Vantage获取新闻失败: {e}")
        
        return all_news
    
    def find_related_news(self, news_item: NewsItem, threshold: float = 0.6) -> List[NewsItem]:
        """查找相关新闻"""
        related = []
        
        for stored_news in self.news_storage.values():
            if stored_news.id != news_item.id:
                # 计算相似度
                similarity = self.deduplicator.calculate_similarity(
                    f"{news_item.title} {news_item.content}",
                    f"{stored_news.title} {stored_news.content}"
                )
                
                if similarity > threshold:
                    related.append(stored_news)
        
        return related
    
    def calculate_importance_score(self, news_item: NewsItem) -> float:
        """计算重要性分数"""
        # 基础分数
        base_score = 0.5
        
        # 情感强度
        sentiment_strength = abs(news_item.sentiment_score)
        
        # 关键词数量
        keyword_score = min(len(news_item.keywords) / 10, 1.0)
        
        # 来源权威性
        source_score = self.impact_analyzer.source_weights.get(news_item.source.lower(), 0.5)
        
        # 时间新鲜度
        hours_old = (datetime.now() - news_item.published_at).total_seconds() / 3600
        freshness_score = max(0, 1 - hours_old / 12)  # 12小时后新鲜度下降
        
        # 综合计算
        importance_score = (
            base_score * 0.2 +
            sentiment_strength * 0.3 +
            keyword_score * 0.2 +
            source_score * 0.2 +
            freshness_score * 0.1
        )
        
        return min(importance_score, 1.0)
    
    def create_event(self, news_item: NewsItem, related_news: List[NewsItem]) -> Event:
        """创建事件"""
        event_id = hashlib.md5(f"{news_item.title}{news_item.category}".encode()).hexdigest()
        
        # 确定影响级别
        impact_score = news_item.impact_score
        if impact_score >= 0.8:
            impact_level = "critical"
        elif impact_score >= 0.6:
            impact_level = "high"
        elif impact_score >= 0.4:
            impact_level = "medium"
        else:
            impact_level = "low"
        
        # 创建事件
        event = Event(
            id=event_id,
            title=news_item.title,
            description=news_item.content[:500] + "..." if len(news_item.content) > 500 else news_item.content,
            start_time=news_item.published_at,
            end_time=None,
            category=news_item.category,
            impact_level=impact_level,
            related_news=[news_item.id] + [n.id for n in related_news]
        )
        
        # 添加到时间线
        self.timeline_builder.add_event(event)
        
        return event
    
    def process_news_batch(self, news_items: List[NewsItem]):
        """批量处理新闻"""
        for news_item in news_items:
            try:
                # 查找相关新闻
                related_news = self.find_related_news(news_item)
                
                # 计算重要性分数
                news_item.importance_score = self.calculate_importance_score(news_item)
                
                # 计算影响度分数
                news_item.impact_score = self.impact_analyzer.calculate_impact_score(news_item, related_news)
                
                # 存储新闻
                self.news_storage[news_item.id] = news_item
                
                # 如果是重要新闻，创建事件
                if news_item.importance_score > 0.7 or len(related_news) > 3:
                    event = self.create_event(news_item, related_news)
                    self.event_storage[event.id] = event
                    
                    # 更新事件时间线
                    self.timeline_builder.update_event_timeline(event.id, news_item)
                
                logger.info(f"处理新闻完成: {news_item.title[:50]}...")
                
            except Exception as e:
                logger.error(f"处理新闻批次失败: {e}")
    
    def start_real_time_processing(self, interval: int = 300):
        """启动实时新闻处理"""
        if self.is_running:
            logger.warning("实时处理已在运行中")
            return
        
        self.is_running = True
        
        def processing_loop():
            logger.info("启动实时新闻处理...")
            
            while self.is_running:
                try:
                    # 获取新闻
                    news_items = self.fetch_news_from_sources()
                    
                    if news_items:
                        # 批量处理
                        self.process_news_batch(news_items)
                        logger.info(f"处理了 {len(news_items)} 条新闻")
                    
                    # 等待下一次处理
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"实时处理错误: {e}")
                    time.sleep(60)  # 错误时等待1分钟
            
            logger.info("实时新闻处理已停止")
        
        self.processing_thread = threading.Thread(target=processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_real_time_processing(self):
        """停止实时新闻处理"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=10)
        logger.info("实时新闻处理已停止")
    
    def get_news_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取新闻摘要"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_news = [
            news for news in self.news_storage.values()
            if news.published_at >= cutoff_time
        ]
        
        # 统计信息
        total_count = len(recent_news)
        category_stats = defaultdict(int)
        sentiment_stats = defaultdict(int)
        
        for news in recent_news:
            category_stats[news.category] += 1
            sentiment_stats[self.sentiment_analyzer.get_sentiment_label(news.sentiment_score)] += 1
        
        # 重要新闻
        important_news = sorted(
            recent_news, 
            key=lambda x: x.importance_score, 
            reverse=True
        )[:10]
        
        # 活跃事件
        active_events = [
            event for event in self.event_storage.values()
            if event.start_time >= cutoff_time
        ]
        
        return {
            "total_news": total_count,
            "category_distribution": dict(category_stats),
            "sentiment_distribution": dict(sentiment_stats),
            "important_news": [
                {
                    "id": news.id,
                    "title": news.title,
                    "source": news.source,
                    "importance_score": news.importance_score,
                    "sentiment_score": news.sentiment_score,
                    "published_at": news.published_at.isoformat()
                }
                for news in important_news
            ],
            "active_events": [
                {
                    "id": event.id,
                    "title": event.title,
                    "category": event.category,
                    "impact_level": event.impact_level,
                    "start_time": event.start_time.isoformat(),
                    "related_news_count": len(event.related_news)
                }
                for event in active_events
            ]
        }
    
    def get_timeline(self, start_date: str, end_date: str) -> List[Dict]:
        """获取事件时间线"""
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            return self.timeline_builder.build_timeline(start_dt, end_dt)
        except Exception as e:
            logger.error(f"获取时间线失败: {e}")
            return []
    
    def export_data(self, filename: str):
        """导出数据"""
        try:
            export_data = {
                "news": [asdict(news) for news in self.news_storage.values()],
                "events": [asdict(event) for event in self.event_storage.values()],
                "timeline": self.timeline_builder.events,
                "export_time": datetime.now().isoformat()
            }
            
            # 转换datetime对象为字符串
            for news in export_data["news"]:
                if "published_at" in news and isinstance(news["published_at"], datetime):
                    news["published_at"] = news["published_at"].isoformat()
            
            for event in export_data["events"]:
                if "start_time" in event and isinstance(event["start_time"], datetime):
                    event["start_time"] = event["start_time"].isoformat()
                if "end_time" in event and isinstance(event["end_time"], datetime):
                    event["end_time"] = event["end_time"].isoformat()
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据已导出到 {filename}")
            
        except Exception as e:
            logger.error(f"导出数据失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 配置API密钥（需要从环境变量或配置文件获取）
    NEWS_API_KEY = "your_news_api_key"  # 从 https://newsapi.org/ 获取
    ALPHA_VANTAGE_KEY = "your_alpha_vantage_key"  # 从 https://www.alphavantage.co/ 获取
    
    # 创建处理器
    processor = NewsEventProcessor(
        news_api_key=NEWS_API_KEY,
        alpha_vantage_key=ALPHA_VANTAGE_KEY
    )
    
    try:
        # 获取新闻摘要
        print("获取新闻摘要...")
        summary = processor.get_news_summary(hours=24)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        # 获取事件时间线
        print("\n获取事件时间线...")
        timeline = processor.get_timeline(
            start_date=(datetime.now() - timedelta(days=1)).isoformat(),
            end_date=datetime.now().isoformat()
        )
        print(json.dumps(timeline, indent=2, ensure_ascii=False))
        
        # 启动实时处理（可选）
        print("\n启动实时新闻处理...")
        processor.start_real_time_processing(interval=300)  # 5分钟间隔
        
        # 运行一段时间
        time.sleep(30)
        
        # 停止实时处理
        processor.stop_real_time_processing()
        
        # 导出数据
        processor.export_data("news_data.json")
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        processor.stop_real_time_processing()
    except Exception as e:
        print(f"程序运行错误: {e}")
        processor.stop_real_time_processing()