"""
A8情绪指标分析器
市场情绪分析系统，集成多维度情绪指标计算

功能包括：
1. 社交媒体情绪监控（Twitter、Reddit、微博等）
2. 新闻情感分析
3. 恐惧贪婪指数计算
4. 资金流向情绪分析
5. 市场参与者情绪调查
6. 情绪指标综合评分
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import re
import math
import warnings
warnings.filterwarnings('ignore')

# 尝试导入自然语言处理库
try:
    import nltk
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK/TextBlob未安装，将使用基础情感分析")

try:
    import tweepy
    import praw
    TWITTER_REDDIT_AVAILABLE = True
except ImportError:
    TWITTER_REDDIT_AVAILABLE = False
    logging.warning("Twitter/Reddit API未安装，将使用模拟数据")

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logging.warning("Web scraping库未安装，将使用模拟数据")


@dataclass
class SentimentData:
    """情绪数据类"""
    timestamp: datetime
    source: str
    content: str
    sentiment_score: float  # -1到1，-1最消极，1最积极
    confidence: float  # 0到1，置信度
    metadata: Dict[str, Any]


@dataclass
class FearGreedIndex:
    """恐惧贪婪指数类"""
    timestamp: datetime
    value: float  # 0-100，0极度恐惧，100极度贪婪
    components: Dict[str, float]
    interpretation: str


class SentimentAnalyzer:
    """市场情绪分析器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化情绪分析器
        
        Args:
            config: 配置字典，包含API密钥等
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化情感分析器
        self._init_sentiment_analyzers()
        
        # 缓存
        self._sentiment_cache = {}
        self._fear_greed_cache = {}
        
        # 情绪历史数据
        self.sentiment_history = []
        self.fear_greed_history = []
        
        # 权重配置
        self.weights = {
            'social_media': 0.25,
            'news': 0.20,
            'fear_greed': 0.25,
            'fund_flow': 0.15,
            'survey': 0.15
        }
    
    def _init_sentiment_analyzers(self):
        """初始化情感分析器"""
        if NLTK_AVAILABLE:
            try:
                # 下载必要的NLTK数据
                nltk.download('punkt', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                self.vader_analyzer = SentimentIntensityAnalyzer()
            except:
                self.vader_analyzer = None
        else:
            self.vader_analyzer = None
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        分析文本情绪
        
        Args:
            text: 待分析的文本
            
        Returns:
            包含情绪分析结果的字典
        """
        if not text.strip():
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        
        # 使用VADER情感分析器
        if self.vader_analyzer:
            scores = self.vader_analyzer.polarity_scores(text)
            return scores
        
        # 使用TextBlob作为备选
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return {
                'compound': polarity,
                'pos': max(0, polarity),
                'neg': max(0, -polarity),
                'neu': 1 - abs(polarity)
            }
        except:
            # 基础词典方法
            return self._basic_sentiment_analysis(text)
    
    def _basic_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """基础情感分析方法"""
        # 简单的积极/消极词汇匹配
        positive_words = ['涨', '好', '牛', '买', '涨涨', '盈利', '收益', '上涨', '上升', '强', '乐观']
        negative_words = ['跌', '坏', '熊', '卖', '亏损', '下跌', '下降', '弱', '悲观', '恐慌']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        
        compound = (pos_count - neg_count) / total
        return {
            'compound': compound,
            'pos': pos_count / total,
            'neg': neg_count / total,
            'neu': 0.0
        }
    
    async def collect_social_media_sentiment(self, keywords: List[str], 
                                           platforms: List[str] = None,
                                           limit: int = 100) -> List[SentimentData]:
        """
        收集社交媒体情绪数据
        
        Args:
            keywords: 关键词列表
            platforms: 平台列表 ['twitter', 'reddit', 'weibo']
            limit: 收集限制
            
        Returns:
            情绪数据列表
        """
        if platforms is None:
            platforms = ['twitter', 'reddit', 'weibo']
        
        sentiment_data = []
        
        for platform in platforms:
            try:
                if platform == 'twitter' and TWITTER_REDDIT_AVAILABLE:
                    data = await self._collect_twitter_sentiment(keywords, limit // len(platforms))
                elif platform == 'reddit' and TWITTER_REDDIT_AVAILABLE:
                    data = await self._collect_reddit_sentiment(keywords, limit // len(platforms))
                elif platform == 'weibo' and WEB_SCRAPING_AVAILABLE:
                    data = await self._collect_weibo_sentiment(keywords, limit // len(platforms))
                else:
                    # 使用模拟数据
                    data = self._generate_mock_social_data(keywords, platform, limit // len(platforms))
                
                sentiment_data.extend(data)
            except Exception as e:
                self.logger.error(f"收集{platform}数据失败: {e}")
        
        return sentiment_data
    
    async def _collect_twitter_sentiment(self, keywords: List[str], limit: int) -> List[SentimentData]:
        """收集Twitter情绪数据"""
        # 这里需要实际的Twitter API配置
        # 由于API限制，使用模拟数据
        return self._generate_mock_social_data(keywords, 'twitter', limit)
    
    async def _collect_reddit_sentiment(self, keywords: List[str], limit: int) -> List[SentimentData]:
        """收集Reddit情绪数据"""
        # 这里需要实际的Reddit API配置
        # 由于API限制，使用模拟数据
        return self._generate_mock_social_data(keywords, 'reddit', limit)
    
    async def _collect_weibo_sentiment(self, keywords: List[str], limit: int) -> List[SentimentData]:
        """收集微博情绪数据"""
        # 这里需要实际的微博API配置
        # 由于API限制，使用模拟数据
        return self._generate_mock_social_data(keywords, 'weibo', limit)
    
    def _generate_mock_social_data(self, keywords: List[str], platform: str, limit: int) -> List[SentimentData]:
        """生成模拟社交媒体数据"""
        mock_data = []
        
        for i in range(limit):
            keyword = np.random.choice(keywords)
            
            # 模拟不同情绪的文本模板
            positive_templates = [
                f"{keyword}看起来要涨了！",
                f"{keyword}基本面很好，值得投资",
                f"{keyword}技术面突破，后市看好",
                f"{keyword}消息面利好，持有待涨"
            ]
            
            negative_templates = [
                f"{keyword}可能要跌了",
                f"{keyword}风险太大，不敢入手",
                f"{keyword}技术面破位，谨慎观望",
                f"{keyword}市场情绪悲观，先退出"
            ]
            
            neutral_templates = [
                f"关于{keyword}的分析",
                f"{keyword}的最新消息",
                f"{keyword}的技术指标",
                f"{keyword}的基本面情况"
            ]
            
            template_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                          p=[0.3, 0.3, 0.4])
            
            if template_type == 'positive':
                content = np.random.choice(positive_templates).format(keyword=keyword)
                sentiment_score = np.random.uniform(0.1, 0.8)
            elif template_type == 'negative':
                content = np.random.choice(negative_templates).format(keyword=keyword)
                sentiment_score = np.random.uniform(-0.8, -0.1)
            else:
                content = np.random.choice(neutral_templates).format(keyword=keyword)
                sentiment_score = np.random.uniform(-0.1, 0.1)
            
            mock_data.append(SentimentData(
                timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
                source=f"{platform}_{i}",
                content=content,
                sentiment_score=sentiment_score,
                confidence=np.random.uniform(0.6, 1.0),
                metadata={'platform': platform, 'keyword': keyword}
            ))
        
        return mock_data
    
    async def analyze_news_sentiment(self, keywords: List[str], 
                                   sources: List[str] = None) -> List[SentimentData]:
        """
        分析新闻情绪
        
        Args:
            keywords: 关键词列表
            sources: 新闻源列表
            
        Returns:
            新闻情绪数据列表
        """
        if sources is None:
            sources = ['reuters', 'bloomberg', 'cnn', 'bbc']
        
        sentiment_data = []
        
        # 模拟新闻数据
        for source in sources:
            try:
                news_items = self._generate_mock_news_data(keywords, source, 20)
                for item in news_items:
                    sentiment_result = self.analyze_text_sentiment(item['content'])
                    
                    sentiment_data.append(SentimentData(
                        timestamp=item['timestamp'],
                        source=item['source'],
                        content=item['content'],
                        sentiment_score=sentiment_result['compound'],
                        confidence=abs(sentiment_result['compound']),
                        metadata={'source': source, 'url': item.get('url', '')}
                    ))
            except Exception as e:
                self.logger.error(f"分析{source}新闻失败: {e}")
        
        return sentiment_data
    
    def _generate_mock_news_data(self, keywords: List[str], source: str, count: int) -> List[Dict]:
        """生成模拟新闻数据"""
        news_templates = [
            "市场分析师认为{keyword}在未来几个月内有望上涨",
            "由于经济不确定性，投资者对{keyword}保持谨慎态度",
            "{keyword}公司发布财报，业绩超出预期",
            "专家警告{keyword}可能面临调整风险",
            "{keyword}行业迎来政策利好，相关股票大涨",
            "分析师下调{keyword}目标价，建议减持",
            "{keyword}技术面突破，支撑位明确",
            "市场恐慌情绪蔓延，{keyword}承压下跌"
        ]
        
        news_data = []
        for i in range(count):
            keyword = np.random.choice(keywords)
            template = np.random.choice(news_templates)
            content = template.format(keyword=keyword)
            
            news_data.append({
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 24)),
                'source': source,
                'content': content,
                'url': f"https://{source}.com/news/{i}"
            })
        
        return news_data
    
    def calculate_fear_greed_index(self, market_data: Dict[str, Any] = None) -> FearGreedIndex:
        """
        计算恐惧贪婪指数
        
        Args:
            market_data: 市场数据，包含价格、成交量等
            
        Returns:
            恐惧贪婪指数对象
        """
        # 如果没有提供市场数据，使用模拟数据
        if market_data is None:
            market_data = self._generate_mock_market_data()
        
        components = {}
        
        # 1. 价格动量 (25%)
        price_momentum = self._calculate_price_momentum(market_data)
        components['price_momentum'] = price_momentum
        
        # 2. 市场波动率 (25%)
        volatility = self._calculate_volatility(market_data)
        components['volatility'] = volatility
        
        # 3. 安全性需求 (15%)
        safe_haven_demand = self._calculate_safe_haven_demand(market_data)
        components['safe_haven_demand'] = safe_haven_demand
        
        # 4. 垃圾债券需求 (10%)
        junk_bond_demand = self._calculate_junk_bond_demand(market_data)
        components['junk_bond_demand'] = junk_bond_demand
        
        # 5. 市场广度 (15%)
        market_breadth = self._calculate_market_breadth(market_data)
        components['market_breadth'] = market_breadth
        
        # 6. 股票vs债券收益率 (10%)
        stock_bond_yield = self._calculate_stock_bond_yield(market_data)
        components['stock_bond_yield'] = stock_bond_yield
        
        # 计算综合指数
        weights = {
            'price_momentum': 0.25,
            'volatility': 0.25,
            'safe_haven_demand': 0.15,
            'junk_bond_demand': 0.10,
            'market_breadth': 0.15,
            'stock_bond_yield': 0.10
        }
        
        fear_greed_value = sum(components[key] * weights[key] for key in components)
        
        # 解释
        if fear_greed_value <= 25:
            interpretation = "极度恐惧"
        elif fear_greed_value <= 45:
            interpretation = "恐惧"
        elif fear_greed_value <= 55:
            interpretation = "中性"
        elif fear_greed_value <= 75:
            interpretation = "贪婪"
        else:
            interpretation = "极度贪婪"
        
        return FearGreedIndex(
            timestamp=datetime.now(),
            value=fear_greed_value,
            components=components,
            interpretation=interpretation
        )
    
    def _generate_mock_market_data(self) -> Dict[str, Any]:
        """生成模拟市场数据"""
        return {
            'prices': np.random.randn(252).cumsum() + 100,  # 模拟一年价格数据
            'volumes': np.random.randint(1000000, 10000000, 252),
            'vix': np.random.uniform(10, 40),
            'treasury_yield': np.random.uniform(1, 5),
            'junk_bond_yield': np.random.uniform(5, 15),
            'advance_decline': np.random.randint(-100, 100, 252)
        }
    
    def _calculate_price_momentum(self, market_data: Dict[str, Any]) -> float:
        """计算价格动量"""
        prices = market_data.get('prices', [])
        if len(prices) < 20:
            return 50.0
        
        # 计算短期和长期移动平均
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
        
        momentum = (short_ma - long_ma) / long_ma * 100
        # 标准化到0-100
        return max(0, min(100, 50 + momentum * 10))
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """计算市场波动率"""
        vix = market_data.get('vix', 20)
        # VIX越高，恐惧情绪越强
        return max(0, min(100, 100 - vix * 2))
    
    def _calculate_safe_haven_demand(self, market_data: Dict[str, Any]) -> float:
        """计算安全性需求"""
        treasury_yield = market_data.get('treasury_yield', 3)
        # 国债收益率越低，安全需求越高
        return max(0, min(100, 100 - treasury_yield * 10))
    
    def _calculate_junk_bond_demand(self, market_data: Dict[str, Any]) -> float:
        """计算垃圾债券需求"""
        junk_yield = market_data.get('junk_bond_yield', 8)
        treasury_yield = market_data.get('treasury_yield', 3)
        spread = junk_yield - treasury_yield
        # 利差越大，贪婪情绪越强
        return max(0, min(100, spread * 10))
    
    def _calculate_market_breadth(self, market_data: Dict[str, Any]) -> float:
        """计算市场广度"""
        advance_decline = market_data.get('advance_decline', [])
        if len(advance_decline) < 10:
            return 50.0
        
        # 计算上涨下跌比率
        recent_a_d = advance_decline[-10:]
        positive_days = sum(1 for x in recent_a_d if x > 0)
        breadth_ratio = positive_days / len(recent_a_d)
        
        return breadth_ratio * 100
    
    def _calculate_stock_bond_yield(self, market_data: Dict[str, Any]) -> float:
        """计算股票债券收益率比较"""
        treasury_yield = market_data.get('treasury_yield', 3)
        # 假设股票收益率为5%，比较股票vs债券吸引力
        stock_yield = 5.0
        yield_diff = stock_yield - treasury_yield
        # 股票相对吸引力越高，贪婪情绪越强
        return max(0, min(100, 50 + yield_diff * 10))
    
    def analyze_fund_flow_sentiment(self, fund_data: Dict[str, Any] = None) -> Dict[str, float]:
        """
        分析资金流向情绪
        
        Args:
            fund_data: 资金流向数据
            
        Returns:
            资金流向情绪指标
        """
        if fund_data is None:
            fund_data = self._generate_mock_fund_data()
        
        # 计算各项指标
        retail_flow = self._calculate_retail_flow(fund_data)
        institutional_flow = self._calculate_institutional_flow(fund_data)
        foreign_flow = self._calculate_foreign_flow(fund_data)
        smart_money_flow = self._calculate_smart_money_flow(fund_data)
        
        return {
            'retail_sentiment': retail_flow,
            'institutional_sentiment': institutional_flow,
            'foreign_sentiment': foreign_flow,
            'smart_money_sentiment': smart_money_flow,
            'overall_fund_sentiment': (retail_flow + institutional_flow + 
                                     foreign_flow + smart_money_flow) / 4
        }
    
    def _generate_mock_fund_data(self) -> Dict[str, Any]:
        """生成模拟资金流向数据"""
        return {
            'retail_flow': np.random.uniform(-100, 100),
            'institutional_flow': np.random.uniform(-50, 50),
            'foreign_flow': np.random.uniform(-80, 80),
            'smart_money_flow': np.random.uniform(-30, 30)
        }
    
    def _calculate_retail_flow(self, fund_data: Dict[str, Any]) -> float:
        """计算散户资金流向情绪"""
        flow = fund_data.get('retail_flow', 0)
        # 散户流入为正，流出为负
        return max(-100, min(100, flow))
    
    def _calculate_institutional_flow(self, fund_data: Dict[str, Any]) -> float:
        """计算机构资金流向情绪"""
        flow = fund_data.get('institutional_flow', 0)
        return max(-100, min(100, flow))
    
    def _calculate_foreign_flow(self, fund_data: Dict[str, Any]) -> float:
        """计算外资流向情绪"""
        flow = fund_data.get('foreign_flow', 0)
        return max(-100, min(100, flow))
    
    def _calculate_smart_money_flow(self, fund_data: Dict[str, Any]) -> float:
        """计算聪明资金流向情绪"""
        flow = fund_data.get('smart_money_flow', 0)
        return max(-100, min(100, flow))
    
    def conduct_sentiment_survey(self, survey_data: Dict[str, Any] = None) -> Dict[str, float]:
        """
        进行市场参与者情绪调查
        
        Args:
            survey_data: 调查数据
            
        Returns:
            调查情绪指标
        """
        if survey_data is None:
            survey_data = self._generate_mock_survey_data()
        
        # 计算各项调查指标
        investor_confidence = self._calculate_investor_confidence(survey_data)
        analyst_optimism = self._calculate_analyst_optimism(survey_data)
        consumer_sentiment = self._calculate_consumer_sentiment(survey_data)
        ceo_confidence = self._calculate_ceo_confidence(survey_data)
        
        return {
            'investor_confidence': investor_confidence,
            'analyst_optimism': analyst_optimism,
            'consumer_sentiment': consumer_sentiment,
            'ceo_confidence': ceo_confidence,
            'overall_survey_sentiment': (investor_confidence + analyst_optimism + 
                                       consumer_sentiment + ceo_confidence) / 4
        }
    
    def _generate_mock_survey_data(self) -> Dict[str, Any]:
        """生成模拟调查数据"""
        return {
            'investor_survey': np.random.uniform(0, 100),
            'analyst_survey': np.random.uniform(0, 100),
            'consumer_survey': np.random.uniform(0, 100),
            'ceo_survey': np.random.uniform(0, 100)
        }
    
    def _calculate_investor_confidence(self, survey_data: Dict[str, Any]) -> float:
        """计算投资者信心指数"""
        return survey_data.get('investor_survey', 50)
    
    def _calculate_analyst_optimism(self, survey_data: Dict[str, Any]) -> float:
        """计算分析师乐观指数"""
        return survey_data.get('analyst_survey', 50)
    
    def _calculate_consumer_sentiment(self, survey_data: Dict[str, Any]) -> float:
        """计算消费者情绪指数"""
        return survey_data.get('consumer_survey', 50)
    
    def _calculate_ceo_confidence(self, survey_data: Dict[str, Any]) -> float:
        """计算CEO信心指数"""
        return survey_data.get('ceo_survey', 50)
    
    def calculate_comprehensive_sentiment_score(self, 
                                               social_sentiment: List[SentimentData] = None,
                                               news_sentiment: List[SentimentData] = None,
                                               fear_greed_index: FearGreedIndex = None,
                                               fund_flow_sentiment: Dict[str, float] = None,
                                               survey_sentiment: Dict[str, float] = None) -> Dict[str, Any]:
        """
        计算综合情绪评分
        
        Args:
            social_sentiment: 社交媒体情绪数据
            news_sentiment: 新闻情绪数据
            fear_greed_index: 恐惧贪婪指数
            fund_flow_sentiment: 资金流向情绪
            survey_sentiment: 调查情绪
            
        Returns:
            综合情绪评分结果
        """
        # 计算各维度情绪分数
        scores = {}
        
        # 1. 社交媒体情绪 (0-100)
        if social_sentiment:
            social_score = np.mean([data.sentiment_score for data in social_sentiment]) * 50 + 50
            scores['social_media'] = max(0, min(100, social_score))
        else:
            scores['social_media'] = 50.0
        
        # 2. 新闻情绪 (0-100)
        if news_sentiment:
            news_score = np.mean([data.sentiment_score for data in news_sentiment]) * 50 + 50
            scores['news'] = max(0, min(100, news_score))
        else:
            scores['news'] = 50.0
        
        # 3. 恐惧贪婪指数 (0-100)
        if fear_greed_index:
            scores['fear_greed'] = fear_greed_index.value
        else:
            scores['fear_greed'] = 50.0
        
        # 4. 资金流向情绪 (0-100)
        if fund_flow_sentiment:
            fund_score = fund_flow_sentiment.get('overall_fund_sentiment', 0) + 50
            scores['fund_flow'] = max(0, min(100, fund_score))
        else:
            scores['fund_flow'] = 50.0
        
        # 5. 调查情绪 (0-100)
        if survey_sentiment:
            survey_score = survey_sentiment.get('overall_survey_sentiment', 50)
            scores['survey'] = survey_score
        else:
            scores['survey'] = 50.0
        
        # 计算加权综合分数
        comprehensive_score = sum(scores[key] * self.weights[key] for key in scores)
        
        # 情绪强度分析
        score_variance = np.var(list(scores.values()))
        sentiment_intensity = min(100, score_variance * 2)  # 分数差异越大，情绪强度越高
        
        # 情绪方向分析
        sentiment_direction = "积极" if comprehensive_score > 55 else "消极" if comprehensive_score < 45 else "中性"
        
        # 风险等级评估
        if comprehensive_score < 20:
            risk_level = "极度恐慌"
        elif comprehensive_score < 35:
            risk_level = "恐慌"
        elif comprehensive_score < 45:
            risk_level = "谨慎"
        elif comprehensive_score < 55:
            risk_level = "中性"
        elif comprehensive_score < 70:
            risk_level = "乐观"
        else:
            risk_level = "极度乐观"
        
        return {
            'timestamp': datetime.now(),
            'comprehensive_score': comprehensive_score,
            'component_scores': scores,
            'weights': self.weights,
            'sentiment_direction': sentiment_direction,
            'sentiment_intensity': sentiment_intensity,
            'risk_level': risk_level,
            'confidence_level': min(100, len(social_sentiment or []) + len(news_sentiment or [])) / 10
        }
    
    def analyze_sentiment_price_correlation(self, sentiment_data: List[Dict], 
                                          price_data: List[float],
                                          time_lags: List[int] = None) -> Dict[str, float]:
        """
        分析情绪与价格的相关性
        
        Args:
            sentiment_data: 情绪数据序列
            price_data: 价格数据序列
            time_lags: 时间滞后列表
            
        Returns:
            相关性分析结果
        """
        if time_lags is None:
            time_lags = [0, 1, 3, 5, 10]  # 0天、1天、3天、5天、10天滞后
        
        correlations = {}
        
        # 确保数据长度一致
        min_length = min(len(sentiment_data), len(price_data))
        sentiment_series = sentiment_data[:min_length]
        price_series = price_data[:min_length]
        
        for lag in time_lags:
            if lag == 0:
                # 同时间相关性
                correlation = np.corrcoef(sentiment_series, price_series)[0, 1]
            else:
                # 滞后相关性
                if len(sentiment_series) > lag and len(price_series) > lag:
                    sentiment_lagged = sentiment_series[:-lag]
                    price_current = price_series[lag:]
                    correlation = np.corrcoef(sentiment_lagged, price_current)[0, 1]
                else:
                    correlation = 0.0
            
            correlations[f'lag_{lag}_days'] = correlation if not np.isnan(correlation) else 0.0
        
        # 计算最佳滞后时间
        best_lag = max(correlations, key=lambda k: abs(correlations[k]))
        
        return {
            'correlations': correlations,
            'best_lag': best_lag,
            'best_correlation': correlations[best_lag],
            'sentiment_predictive_power': max(abs(c) for c in correlations.values())
        }
    
    def generate_sentiment_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        生成情绪分析报告
        
        Args:
            analysis_results: 分析结果字典
            
        Returns:
            格式化的报告字符串
        """
        report = []
        report.append("=" * 60)
        report.append("市场情绪分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 综合评分
        comprehensive_score = analysis_results.get('comprehensive_score', 0)
        report.append(f"综合情绪评分: {comprehensive_score:.2f}/100")
        report.append(f"情绪方向: {analysis_results.get('sentiment_direction', '未知')}")
        report.append(f"情绪强度: {analysis_results.get('sentiment_intensity', 0):.2f}")
        report.append(f"风险等级: {analysis_results.get('risk_level', '未知')}")
        report.append("")
        
        # 各维度评分
        report.append("各维度情绪评分:")
        component_scores = analysis_results.get('component_scores', {})
        for component, score in component_scores.items():
            weight = self.weights.get(component, 0)
            report.append(f"  {component}: {score:.2f}/100 (权重: {weight:.1%})")
        report.append("")
        
        # 置信度
        confidence = analysis_results.get('confidence_level', 0)
        report.append(f"分析置信度: {confidence:.1f}%")
        report.append("")
        
        # 投资建议
        report.append("情绪分析投资建议:")
        if comprehensive_score < 30:
            report.append("  - 市场极度恐慌，可能是买入机会")
            report.append("  - 建议分批建仓，关注优质资产")
        elif comprehensive_score < 45:
            report.append("  - 市场偏恐慌，建议谨慎操作")
            report.append("  - 可小仓位试探，等待明确信号")
        elif comprehensive_score < 55:
            report.append("  - 市场情绪中性，建议观望")
            report.append("  - 等待更明确的方向信号")
        elif comprehensive_score < 70:
            report.append("  - 市场偏乐观，可适度参与")
            report.append("  - 注意控制风险，设置止损")
        else:
            report.append("  - 市场极度乐观，需警惕回调")
            report.append("  - 建议减仓或获利了结")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    async def run_comprehensive_analysis(self, keywords: List[str], 
                                       market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        运行综合情绪分析
        
        Args:
            keywords: 分析关键词
            market_data: 市场数据
            
        Returns:
            综合分析结果
        """
        self.logger.info(f"开始综合情绪分析，关键词: {keywords}")
        
        # 并行收集各类数据
        tasks = [
            self.collect_social_media_sentiment(keywords),
            self.analyze_news_sentiment(keywords),
            asyncio.create_task(asyncio.to_thread(self.calculate_fear_greed_index, market_data)),
            asyncio.create_task(asyncio.to_thread(self.analyze_fund_flow_sentiment)),
            asyncio.create_task(asyncio.to_thread(self.conduct_sentiment_survey))
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            social_sentiment = results[0] if not isinstance(results[0], Exception) else []
            news_sentiment = results[1] if not isinstance(results[1], Exception) else []
            fear_greed_index = results[2] if not isinstance(results[2], Exception) else None
            fund_flow_sentiment = results[3] if not isinstance(results[3], Exception) else {}
            survey_sentiment = results[4] if not isinstance(results[4], Exception) else {}
            
            # 计算综合评分
            comprehensive_result = self.calculate_comprehensive_sentiment_score(
                social_sentiment=social_sentiment,
                news_sentiment=news_sentiment,
                fear_greed_index=fear_greed_index,
                fund_flow_sentiment=fund_flow_sentiment,
                survey_sentiment=survey_sentiment
            )
            
            # 保存历史数据
            self.sentiment_history.append(comprehensive_result)
            if fear_greed_index:
                self.fear_greed_history.append(fear_greed_index)
            
            # 限制历史数据大小
            if len(self.sentiment_history) > 100:
                self.sentiment_history = self.sentiment_history[-100:]
            if len(self.fear_greed_history) > 100:
                self.fear_greed_history = self.fear_greed_history[-100:]
            
            self.logger.info("综合情绪分析完成")
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"综合分析失败: {e}")
            raise
    
    def get_sentiment_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        获取情绪趋势分析
        
        Args:
            days: 分析天数
            
        Returns:
            趋势分析结果
        """
        if not self.sentiment_history:
            return {'error': '无历史数据'}
        
        # 筛选最近的数据
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_data = [item for item in self.sentiment_history 
                      if item['timestamp'] >= cutoff_time]
        
        if len(recent_data) < 2:
            return {'error': '数据不足'}
        
        # 计算趋势
        scores = [item['comprehensive_score'] for item in recent_data]
        timestamps = [item['timestamp'] for item in recent_data]
        
        # 简单线性趋势
        x = np.arange(len(scores))
        trend_slope = np.polyfit(x, scores, 1)[0]
        
        # 波动率
        volatility = np.std(scores)
        
        # 极值分析
        max_score = max(scores)
        min_score = min(scores)
        max_time = timestamps[scores.index(max_score)]
        min_time = timestamps[scores.index(min_score)]
        
        # 趋势判断
        if trend_slope > 1:
            trend_direction = "明显上升"
        elif trend_slope > 0.1:
            trend_direction = "轻微上升"
        elif trend_slope < -1:
            trend_direction = "明显下降"
        elif trend_slope < -0.1:
            trend_direction = "轻微下降"
        else:
            trend_direction = "横盘整理"
        
        return {
            'trend_direction': trend_direction,
            'trend_slope': trend_slope,
            'volatility': volatility,
            'max_score': max_score,
            'min_score': min_score,
            'max_time': max_time,
            'min_time': min_time,
            'current_score': scores[-1],
            'data_points': len(recent_data)
        }


# 示例使用和测试函数
async def main():
    """主函数示例"""
    # 初始化分析器
    analyzer = SentimentAnalyzer()
    
    # 分析关键词
    keywords = ["比特币", "以太坊", "股票市场", "投资"]
    
    try:
        # 运行综合分析
        result = await analyzer.run_comprehensive_analysis(keywords)
        
        # 生成报告
        report = analyzer.generate_sentiment_report(result)
        print(report)
        
        # 获取趋势分析
        trends = analyzer.get_sentiment_trends()
        print("\n情绪趋势分析:")
        print(f"趋势方向: {trends.get('trend_direction', 'N/A')}")
        print(f"趋势斜率: {trends.get('trend_slope', 0):.3f}")
        print(f"波动率: {trends.get('volatility', 0):.2f}")
        
    except Exception as e:
        print(f"分析失败: {e}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行示例
    asyncio.run(main())