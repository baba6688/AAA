#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1市场感知引擎
===============

智能量化交易系统的核心感知引擎，实现多维度市场状态感知、评估和预测。

主要功能：
1. 实时市场状态感知和评估
2. 多维度市场特征提取
3. 市场强度和活跃度评估
4. 市场阶段识别（牛市、熊市、震荡市）
5. 市场情绪感知和量化
6. 市场风险感知和评估
7. 市场机会识别

技术特点：
- 集成A1-A9数据源
- 使用机器学习算法进行市场状态分类
- 实时感知和历史分析
- 市场状态评分和预警机制
- 可视化市场感知结果

Author: B1 Market Perception Engine
Date: 2025-11-05
Version: 1.0.0
"""

import asyncio
import numpy as np
import pandas as pd
import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import threading
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 机器学习库
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# 尝试导入深度学习库
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow未安装，将使用传统机器学习算法")

# 集成A1-A9数据源
try:
    from A.A1.MarketDataCollector import MarketDataCollector, MarketData
    from A.A2.EconomicIndicatorMonitor import EconomicIndicatorMonitor
    from A.A3.NewsEventProcessor import NewsEventProcessor
    from A.A4.ExchangeConnector import ExchangeConnector
    from A.A5.MacroEconomicAnalyzer import MacroEconomicAnalyzer
    from A.A6.GeopoliticalMonitor import GeopoliticalMonitor
    from A.A7.TechnicalIndicatorCalculator import TechnicalIndicatorCalculator
    from A.A8.SentimentAnalyzer import SentimentAnalyzer
    from A.A9.EnvironmentStateAggregator import EnvironmentStateAggregator, MarketEnvironment
    DATA_SOURCES_AVAILABLE = True
except ImportError as e:
    DATA_SOURCES_AVAILABLE = False
    logging.warning(f"数据源导入失败: {e}，将使用模拟数据")

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_perception.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """市场阶段枚举"""
    BULL_MARKET = "牛市"
    BEAR_MARKET = "熊市"
    SIDEWAYS = "震荡市"
    HIGH_VOLATILITY = "高波动市"
    LOW_VOLATILITY = "低波动市"
    TRANSITION = "过渡期"
    UNKNOWN = "未知"


class RiskLevel(Enum):
    """风险等级枚举"""
    VERY_LOW = "极低风险"
    LOW = "低风险"
    MEDIUM = "中等风险"
    HIGH = "高风险"
    VERY_HIGH = "极高风险"
    EXTREME = "极端风险"


class OpportunityType(Enum):
    """机会类型枚举"""
    TRENDING_UP = "上涨趋势机会"
    TRENDING_DOWN = "下跌趋势机会"
    BREAKOUT = "突破机会"
    REVERSAL = "反转机会"
    MOMENTUM = "动量机会"
    VALUE = "价值机会"
    ARBITRAGE = "套利机会"
    NONE = "无明显机会"


@dataclass
class MarketFeatures:
    """市场特征数据类"""
    timestamp: datetime
    price_features: Dict[str, float] = field(default_factory=dict)
    volume_features: Dict[str, float] = field(default_factory=dict)
    technical_features: Dict[str, float] = field(default_factory=dict)
    sentiment_features: Dict[str, float] = field(default_factory=dict)
    economic_features: Dict[str, float] = field(default_factory=dict)
    news_features: Dict[str, float] = field(default_factory=dict)
    risk_features: Dict[str, float] = field(default_factory=dict)
    macro_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketPerceptionResult:
    """市场感知结果数据类"""
    timestamp: datetime
    market_phase: MarketPhase
    confidence: float
    market_score: float  # -100到100的综合评分
    risk_level: RiskLevel
    opportunity_type: OpportunityType
    key_signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    features: MarketFeatures = None
    prediction_horizon: int = 24  # 预测时间窗口（小时）


@dataclass
class AlertConfig:
    """预警配置"""
    enable_alerts: bool = True
    risk_threshold: float = 70.0
    opportunity_threshold: float = 60.0
    volatility_threshold: float = 0.03
    sentiment_threshold: float = 0.8
    notification_methods: List[str] = field(default_factory=lambda: ["log", "file"])


class MarketPerceptionEngine:
    """市场感知引擎主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化市场感知引擎
        
        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 数据源
        self.data_sources = {}
        self._initialize_data_sources()
        
        # 机器学习模型
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
        # 实时数据缓存
        self.realtime_data = deque(maxlen=1000)
        self.historical_data = deque(maxlen=10000)
        
        # 特征工程
        self.feature_extractors = {}
        self._initialize_feature_extractors()
        
        # 预警系统
        self.alert_config = AlertConfig(**self.config.get('alerts', {}))
        self.active_alerts = []
        
        # 可视化
        self.viz_cache = {}
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 状态
        self.is_running = False
        self.last_update = None
        
        self.logger.info("市场感知引擎初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'model_config': {
                'market_phase_classifier': {
                    'algorithm': 'random_forest',
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'risk_assessor': {
                    'algorithm': 'gradient_boosting',
                    'n_estimators': 50,
                    'learning_rate': 0.1
                }
            },
            'feature_config': {
                'use_technical_indicators': True,
                'use_sentiment_analysis': True,
                'use_economic_indicators': True,
                'use_news_analysis': True,
                'use_macro_factors': True
            },
            'alerts': {
                'enable_alerts': True,
                'risk_threshold': 70.0,
                'opportunity_threshold': 60.0,
                'volatility_threshold': 0.03,
                'sentiment_threshold': 0.8
            },
            'update_interval': 60,  # 秒
            'prediction_horizon': 24,  # 小时
            'lookback_periods': 252  # 历史数据回看期
        }
    
    def _initialize_data_sources(self):
        """初始化数据源"""
        if not DATA_SOURCES_AVAILABLE:
            self.logger.warning("使用模拟数据源")
            return
        
        try:
            # A1: 市场数据采集器
            self.data_sources['market_data'] = MarketDataCollector()
            
            # A2: 经济指标监控器
            self.data_sources['economic_indicators'] = EconomicIndicatorMonitor()
            
            # A3: 新闻事件处理器
            self.data_sources['news_events'] = NewsEventProcessor()
            
            # A4: 交易所连接器
            self.data_sources['exchange_data'] = ExchangeConnector()
            
            # A5: 宏观经济分析器
            self.data_sources['macro_analysis'] = MacroEconomicAnalyzer()
            
            # A6: 地缘政治监控器
            self.data_sources['geopolitical'] = GeopoliticalMonitor()
            
            # A7: 技术指标计算器
            self.data_sources['technical_indicators'] = TechnicalIndicatorCalculator()
            
            # A8: 情感分析器
            self.data_sources['sentiment_analysis'] = SentimentAnalyzer()
            
            # A9: 环境状态聚合器
            self.data_sources['environment_aggregator'] = EnvironmentStateAggregator()
            
            self.logger.info("数据源初始化完成")
            
        except Exception as e:
            self.logger.error(f"数据源初始化失败: {e}")
            self.data_sources = {}
    
    def _initialize_feature_extractors(self):
        """初始化特征提取器"""
        self.feature_extractors = {
            'price_features': self._extract_price_features,
            'volume_features': self._extract_volume_features,
            'technical_features': self._extract_technical_features,
            'sentiment_features': self._extract_sentiment_features,
            'economic_features': self._extract_economic_features,
            'news_features': self._extract_news_features,
            'risk_features': self._extract_risk_features,
            'macro_features': self._extract_macro_features
        }
    
    def start_real_time_monitoring(self):
        """启动实时监控"""
        if self.is_running:
            self.logger.warning("监控已在运行中")
            return
        
        self.is_running = True
        self.logger.info("启动市场感知实时监控")
        
        # 启动数据收集线程
        self.data_collection_thread = threading.Thread(target=self._data_collection_loop)
        self.data_collection_thread.daemon = True
        self.data_collection_thread.start()
        
        # 启动特征提取线程
        self.feature_extraction_thread = threading.Thread(target=self._feature_extraction_loop)
        self.feature_extraction_thread.daemon = True
        self.feature_extraction_thread.start()
        
        # 启动模型预测线程
        self.prediction_thread = threading.Thread(target=self._prediction_loop)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
    
    def stop_real_time_monitoring(self):
        """停止实时监控"""
        self.is_running = False
        self.logger.info("停止市场感知实时监控")
    
    def _data_collection_loop(self):
        """数据收集循环"""
        while self.is_running:
            try:
                current_data = self._collect_current_data()
                if current_data:
                    with self.lock:
                        self.realtime_data.append(current_data)
                        self.historical_data.append(current_data)
                
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                self.logger.error(f"数据收集错误: {e}")
                time.sleep(5)
    
    def _feature_extraction_loop(self):
        """特征提取循环"""
        while self.is_running:
            try:
                with self.lock:
                    if len(self.realtime_data) > 0:
                        latest_data = self.realtime_data[-1]
                        features = self._extract_all_features(latest_data)
                        if features:
                            latest_data['features'] = features
                
                time.sleep(self.config['update_interval'] // 2)
                
            except Exception as e:
                self.logger.error(f"特征提取错误: {e}")
                time.sleep(5)
    
    def _prediction_loop(self):
        """预测循环"""
        while self.is_running:
            try:
                with self.lock:
                    if len(self.historical_data) >= 50:  # 需要足够的历史数据
                        result = self._perform_market_perception()
                        if result:
                            self._check_alerts(result)
                            self.last_update = datetime.now()
                
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                self.logger.error(f"市场感知预测错误: {e}")
                time.sleep(10)
    
    def _collect_current_data(self) -> Dict[str, Any]:
        """收集当前数据"""
        data = {
            'timestamp': datetime.now(),
            'market_data': {},
            'economic_data': {},
            'news_data': {},
            'sentiment_data': {},
            'technical_data': {}
        }
        
        if not DATA_SOURCES_AVAILABLE:
            # 使用模拟数据
            return self._generate_mock_data()
        
        try:
            # 收集市场数据
            if 'market_data' in self.data_sources:
                market_data = self._collect_market_data()
                data['market_data'] = market_data
            
            # 收集经济数据
            if 'economic_indicators' in self.data_sources:
                economic_data = self._collect_economic_data()
                data['economic_data'] = economic_data
            
            # 收集新闻数据
            if 'news_events' in self.data_sources:
                news_data = self._collect_news_data()
                data['news_data'] = news_data
            
            # 收集情感数据
            if 'sentiment_analysis' in self.data_sources:
                sentiment_data = self._collect_sentiment_data()
                data['sentiment_data'] = sentiment_data
            
            # 收集技术数据
            if 'technical_indicators' in self.data_sources:
                technical_data = self._collect_technical_data()
                data['technical_data'] = technical_data
            
            return data
            
        except Exception as e:
            self.logger.error(f"数据收集失败: {e}")
            return None
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """生成模拟数据用于测试"""
        import random
        
        base_time = datetime.now()
        
        return {
            'timestamp': base_time,
            'market_data': {
                'price': 50000 + random.uniform(-1000, 1000),
                'volume': random.uniform(1000000, 10000000),
                'volatility': random.uniform(0.01, 0.05),
                'rsi': random.uniform(20, 80),
                'macd': random.uniform(-100, 100),
                'bb_position': random.uniform(0, 1)
            },
            'economic_data': {
                'gdp_growth': random.uniform(-0.02, 0.04),
                'inflation_rate': random.uniform(0.01, 0.05),
                'unemployment_rate': random.uniform(0.03, 0.10),
                'interest_rate': random.uniform(0.01, 0.08)
            },
            'news_data': {
                'sentiment_score': random.uniform(-1, 1),
                'news_volume': random.randint(10, 100),
                'importance_score': random.uniform(0, 1)
            },
            'sentiment_data': {
                'fear_greed_index': random.uniform(0, 100),
                'social_sentiment': random.uniform(-1, 1),
                'institutional_sentiment': random.uniform(-1, 1)
            },
            'technical_data': {
                'trend_strength': random.uniform(0, 1),
                'momentum_score': random.uniform(-1, 1),
                'support_resistance': random.uniform(0, 1)
            }
        }
    
    def _collect_market_data(self) -> Dict[str, Any]:
        """收集市场数据"""
        try:
            if 'market_data' in self.data_sources:
                # 实际实现会调用真实的数据源
                return {
                    'price': 50000.0,  # 示例价格
                    'volume': 5000000.0,
                    'volatility': 0.025,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"市场数据收集失败: {e}")
        
        return {}
    
    def _collect_economic_data(self) -> Dict[str, Any]:
        """收集经济数据"""
        try:
            if 'economic_indicators' in self.data_sources:
                return {
                    'gdp_growth': 0.025,
                    'inflation_rate': 0.032,
                    'unemployment_rate': 0.055,
                    'interest_rate': 0.045,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"经济数据收集失败: {e}")
        
        return {}
    
    def _collect_news_data(self) -> Dict[str, Any]:
        """收集新闻数据"""
        try:
            if 'news_events' in self.data_sources:
                return {
                    'sentiment_score': 0.2,
                    'news_volume': 45,
                    'importance_score': 0.7,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"新闻数据收集失败: {e}")
        
        return {}
    
    def _collect_sentiment_data(self) -> Dict[str, Any]:
        """收集情感数据"""
        try:
            if 'sentiment_analysis' in self.data_sources:
                return {
                    'fear_greed_index': 65.0,
                    'social_sentiment': 0.3,
                    'institutional_sentiment': 0.1,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"情感数据收集失败: {e}")
        
        return {}
    
    def _collect_technical_data(self) -> Dict[str, Any]:
        """收集技术数据"""
        try:
            if 'technical_indicators' in self.data_sources:
                return {
                    'rsi': 55.0,
                    'macd': 150.0,
                    'bb_position': 0.6,
                    'trend_strength': 0.7,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"技术数据收集失败: {e}")
        
        return {}
    
    def _extract_all_features(self, data: Dict[str, Any]) -> MarketFeatures:
        """提取所有特征"""
        if not data:
            return None
        
        features = MarketFeatures(timestamp=data['timestamp'])
        
        try:
            # 提取各类特征
            for feature_type, extractor in self.feature_extractors.items():
                if feature_type in data:
                    extracted = extractor(data[feature_type])
                    setattr(features, feature_type.replace('_features', ''), extracted)
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            return None
    
    def _extract_price_features(self, price_data: Dict[str, Any]) -> Dict[str, float]:
        """提取价格特征"""
        features = {}
        
        try:
            if 'price' in price_data:
                features['current_price'] = price_data['price']
                features['price_momentum'] = price_data.get('price_momentum', 0.0)
                features['price_volatility'] = price_data.get('volatility', 0.0)
            
            if 'volume' in price_data:
                features['volume'] = price_data['volume']
                features['volume_momentum'] = price_data.get('volume_momentum', 0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"价格特征提取失败: {e}")
            return {}
    
    def _extract_volume_features(self, volume_data: Dict[str, Any]) -> Dict[str, float]:
        """提取成交量特征"""
        features = {}
        
        try:
            if 'volume' in volume_data:
                features['volume'] = volume_data['volume']
                features['volume_ratio'] = volume_data.get('volume_ratio', 1.0)
                features['volume_trend'] = volume_data.get('volume_trend', 0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"成交量特征提取失败: {e}")
            return {}
    
    def _extract_technical_features(self, technical_data: Dict[str, Any]) -> Dict[str, float]:
        """提取技术指标特征"""
        features = {}
        
        try:
            # RSI特征
            if 'rsi' in technical_data:
                features['rsi'] = technical_data['rsi']
                features['rsi_oversold'] = 1.0 if technical_data['rsi'] < 30 else 0.0
                features['rsi_overbought'] = 1.0 if technical_data['rsi'] > 70 else 0.0
            
            # MACD特征
            if 'macd' in technical_data:
                features['macd'] = technical_data['macd']
                features['macd_signal'] = technical_data.get('macd_signal', 0.0)
                features['macd_histogram'] = technical_data.get('macd_histogram', 0.0)
            
            # 布林带特征
            if 'bb_position' in technical_data:
                features['bb_position'] = technical_data['bb_position']
                features['bb_squeeze'] = 1.0 if technical_data['bb_position'] < 0.1 else 0.0
                features['bb_expansion'] = 1.0 if technical_data['bb_position'] > 0.9 else 0.0
            
            # 趋势特征
            if 'trend_strength' in technical_data:
                features['trend_strength'] = technical_data['trend_strength']
                features['trend_direction'] = technical_data.get('trend_direction', 0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"技术特征提取失败: {e}")
            return {}
    
    def _extract_sentiment_features(self, sentiment_data: Dict[str, Any]) -> Dict[str, float]:
        """提取情感特征"""
        features = {}
        
        try:
            # 恐惧贪婪指数
            if 'fear_greed_index' in sentiment_data:
                fgi = sentiment_data['fear_greed_index']
                features['fear_greed_index'] = fgi
                features['extreme_fear'] = 1.0 if fgi < 25 else 0.0
                features['extreme_greed'] = 1.0 if fgi > 75 else 0.0
            
            # 社交情感
            if 'social_sentiment' in sentiment_data:
                features['social_sentiment'] = sentiment_data['social_sentiment']
                features['sentiment_momentum'] = sentiment_data.get('sentiment_momentum', 0.0)
            
            # 机构情感
            if 'institutional_sentiment' in sentiment_data:
                features['institutional_sentiment'] = sentiment_data['institutional_sentiment']
            
            return features
            
        except Exception as e:
            self.logger.error(f"情感特征提取失败: {e}")
            return {}
    
    def _extract_economic_features(self, economic_data: Dict[str, Any]) -> Dict[str, float]:
        """提取经济特征"""
        features = {}
        
        try:
            if 'gdp_growth' in economic_data:
                features['gdp_growth'] = economic_data['gdp_growth']
                features['gdp_acceleration'] = economic_data.get('gdp_acceleration', 0.0)
            
            if 'inflation_rate' in economic_data:
                features['inflation_rate'] = economic_data['inflation_rate']
                features['inflation_trend'] = economic_data.get('inflation_trend', 0.0)
            
            if 'unemployment_rate' in economic_data:
                features['unemployment_rate'] = economic_data['unemployment_rate']
                features['employment_trend'] = economic_data.get('employment_trend', 0.0)
            
            if 'interest_rate' in economic_data:
                features['interest_rate'] = economic_data['interest_rate']
                features['rate_change'] = economic_data.get('rate_change', 0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"经济特征提取失败: {e}")
            return {}
    
    def _extract_news_features(self, news_data: Dict[str, Any]) -> Dict[str, float]:
        """提取新闻特征"""
        features = {}
        
        try:
            if 'sentiment_score' in news_data:
                features['news_sentiment'] = news_data['sentiment_score']
                features['sentiment_volatility'] = news_data.get('sentiment_volatility', 0.0)
            
            if 'news_volume' in news_data:
                features['news_volume'] = news_data['news_volume']
                features['news_intensity'] = news_data.get('news_intensity', 0.0)
            
            if 'importance_score' in news_data:
                features['news_importance'] = news_data['importance_score']
                features['high_impact_news'] = 1.0 if news_data['importance_score'] > 0.8 else 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"新闻特征提取失败: {e}")
            return {}
    
    def _extract_risk_features(self, risk_data: Dict[str, Any]) -> Dict[str, float]:
        """提取风险特征"""
        features = {}
        
        try:
            # 波动率特征
            if 'volatility' in risk_data:
                features['volatility'] = risk_data['volatility']
                features['volatility_percentile'] = risk_data.get('volatility_percentile', 50.0)
                features['volatility_regime'] = risk_data.get('volatility_regime', 0.0)
            
            # 回撤特征
            if 'drawdown' in risk_data:
                features['max_drawdown'] = risk_data['drawdown']
                features['drawdown_duration'] = risk_data.get('drawdown_duration', 0.0)
            
            # 相关性特征
            if 'correlation' in risk_data:
                features['market_correlation'] = risk_data['correlation']
                features['correlation_stability'] = risk_data.get('correlation_stability', 0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"风险特征提取失败: {e}")
            return {}
    
    def _extract_macro_features(self, macro_data: Dict[str, Any]) -> Dict[str, float]:
        """提取宏观特征"""
        features = {}
        
        try:
            # 地缘政治风险
            if 'geopolitical_risk' in macro_data:
                features['geopolitical_risk'] = macro_data['geopolitical_risk']
                features['risk_escalation'] = macro_data.get('risk_escalation', 0.0)
            
            # 政策环境
            if 'policy_environment' in macro_data:
                features['policy_uncertainty'] = macro_data['policy_environment']
                features['policy_impact'] = macro_data.get('policy_impact', 0.0)
            
            # 市场结构
            if 'market_structure' in macro_data:
                features['liquidity_score'] = macro_data['market_structure']
                features['market_depth'] = macro_data.get('market_depth', 0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"宏观特征提取失败: {e}")
            return {}
    
    def _perform_market_perception(self) -> Optional[MarketPerceptionResult]:
        """执行市场感知分析"""
        try:
            with self.lock:
                if len(self.historical_data) < 50:
                    return None
                
                # 准备特征数据
                features_df = self._prepare_features_dataframe()
                if features_df is None or features_df.empty:
                    return None
                
                # 市场阶段分类
                market_phase = self._classify_market_phase(features_df)
                
                # 风险评估
                risk_level = self._assess_market_risk(features_df)
                
                # 机会识别
                opportunity_type = self._identify_opportunities(features_df)
                
                # 综合评分
                market_score = self._calculate_market_score(features_df)
                
                # 置信度评估
                confidence = self._calculate_confidence(features_df)
                
                # 关键信号
                key_signals = self._extract_key_signals(features_df)
                
                # 预警检查
                warnings = self._generate_warnings(features_df, market_score, risk_level)
                
                result = MarketPerceptionResult(
                    timestamp=datetime.now(),
                    market_phase=market_phase,
                    confidence=confidence,
                    market_score=market_score,
                    risk_level=risk_level,
                    opportunity_type=opportunity_type,
                    key_signals=key_signals,
                    warnings=warnings,
                    features=self._get_latest_features()
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"市场感知分析失败: {e}")
            return None
    
    def _prepare_features_dataframe(self) -> Optional[pd.DataFrame]:
        """准备特征数据框"""
        try:
            features_list = []
            
            for data_point in list(self.historical_data):
                if 'features' in data_point and data_point['features']:
                    features = data_point['features']
                    row = {'timestamp': data_point['timestamp']}
                    
                    # 展平所有特征
                    for attr_name in ['price_features', 'volume_features', 'technical_features',
                                    'sentiment_features', 'economic_features', 'news_features',
                                    'risk_features', 'macro_features']:
                        if hasattr(features, attr_name):
                            feature_dict = getattr(features, attr_name)
                            if isinstance(feature_dict, dict):
                                for key, value in feature_dict.items():
                                    row[f"{attr_name}_{key}"] = value
                    
                    features_list.append(row)
            
            if not features_list:
                return None
            
            df = pd.DataFrame(features_list)
            df.set_index('timestamp', inplace=True)
            
            # 处理缺失值
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            
            # 确保有足够的数据
            if len(df) < 20:
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"特征数据准备失败: {e}")
            return None
    
    def _classify_market_phase(self, features_df: pd.DataFrame) -> MarketPhase:
        """分类市场阶段"""
        try:
            # 简化的市场阶段分类逻辑
            latest_features = features_df.iloc[-1]
            
            # 基于多个指标判断市场阶段
            price_momentum = latest_features.get('technical_features_trend_strength', 0)
            volatility = latest_features.get('risk_features_volatility', 0.02)
            sentiment = latest_features.get('sentiment_features_fear_greed_index', 50)
            
            # 分类逻辑
            if price_momentum > 0.6 and sentiment > 60:
                return MarketPhase.BULL_MARKET
            elif price_momentum < 0.4 and sentiment < 40:
                return MarketPhase.BEAR_MARKET
            elif volatility > 0.04:
                return MarketPhase.HIGH_VOLATILITY
            elif volatility < 0.015:
                return MarketPhase.LOW_VOLATILITY
            else:
                return MarketPhase.SIDEWAYS
                
        except Exception as e:
            self.logger.error(f"市场阶段分类失败: {e}")
            return MarketPhase.UNKNOWN
    
    def _assess_market_risk(self, features_df: pd.DataFrame) -> RiskLevel:
        """评估市场风险"""
        try:
            latest_features = features_df.iloc[-1]
            
            # 风险指标
            volatility = latest_features.get('risk_features_volatility', 0.02)
            correlation = latest_features.get('risk_features_market_correlation', 0.5)
            geopolitical_risk = latest_features.get('macro_features_geopolitical_risk', 0.0)
            
            # 综合风险评分
            risk_score = (
                volatility * 1000 +  # 波动率权重
                correlation * 50 +   # 相关性权重
                geopolitical_risk * 30  # 地缘政治风险权重
            )
            
            if risk_score < 20:
                return RiskLevel.VERY_LOW
            elif risk_score < 40:
                return RiskLevel.LOW
            elif risk_score < 60:
                return RiskLevel.MEDIUM
            elif risk_score < 80:
                return RiskLevel.HIGH
            elif risk_score < 100:
                return RiskLevel.VERY_HIGH
            else:
                return RiskLevel.EXTREME
                
        except Exception as e:
            self.logger.error(f"风险评估失败: {e}")
            return RiskLevel.MEDIUM
    
    def _identify_opportunities(self, features_df: pd.DataFrame) -> OpportunityType:
        """识别市场机会"""
        try:
            latest_features = features_df.iloc[-1]
            
            # 机会指标
            trend_strength = latest_features.get('technical_features_trend_strength', 0.5)
            momentum = latest_features.get('technical_features_momentum_score', 0.0)
            sentiment = latest_features.get('sentiment_features_social_sentiment', 0.0)
            price_position = latest_features.get('technical_features_bb_position', 0.5)
            
            # 机会识别逻辑
            if trend_strength > 0.7 and momentum > 0.3:
                return OpportunityType.TRENDING_UP
            elif trend_strength > 0.7 and momentum < -0.3:
                return OpportunityType.TRENDING_DOWN
            elif abs(momentum) > 0.5:
                return OpportunityType.MOMENTUM
            elif price_position < 0.1 or price_position > 0.9:
                return OpportunityType.REVERSAL
            else:
                return OpportunityType.NONE
                
        except Exception as e:
            self.logger.error(f"机会识别失败: {e}")
            return OpportunityType.NONE
    
    def _calculate_market_score(self, features_df: pd.DataFrame) -> float:
        """计算市场综合评分"""
        try:
            latest_features = features_df.iloc[-1]
            
            # 各项评分组件
            trend_score = latest_features.get('technical_features_trend_strength', 0.5) * 25
            sentiment_score = (latest_features.get('sentiment_features_fear_greed_index', 50) - 50) / 2
            volatility_penalty = max(0, (latest_features.get('risk_features_volatility', 0.02) - 0.02) * 500)
            momentum_score = latest_features.get('technical_features_momentum_score', 0.0) * 15
            
            # 综合评分
            market_score = trend_score + sentiment_score + momentum_score - volatility_penalty
            
            # 限制在-100到100范围内
            return max(-100, min(100, market_score))
            
        except Exception as e:
            self.logger.error(f"市场评分计算失败: {e}")
            return 0.0
    
    def _calculate_confidence(self, features_df: pd.DataFrame) -> float:
        """计算预测置信度"""
        try:
            # 基于数据质量和一致性计算置信度
            data_quality = min(1.0, len(features_df) / 100)  # 数据量置信度
            
            # 特征一致性
            if len(features_df) > 10:
                feature_std = features_df.std().mean()
                consistency = max(0, 1 - feature_std)  # 标准差越小，一致性越高
            else:
                consistency = 0.5
            
            # 综合置信度
            confidence = (data_quality + consistency) / 2
            return confidence
            
        except Exception as e:
            self.logger.error(f"置信度计算失败: {e}")
            return 0.5
    
    def _extract_key_signals(self, features_df: pd.DataFrame) -> List[str]:
        """提取关键信号"""
        signals = []
        
        try:
            latest_features = features_df.iloc[-1]
            
            # 技术信号
            if latest_features.get('technical_features_rsi_oversold', 0) > 0.5:
                signals.append("RSI超卖信号")
            
            if latest_features.get('technical_features_rsi_overbought', 0) > 0.5:
                signals.append("RSI超买信号")
            
            if latest_features.get('technical_features_bb_squeeze', 0) > 0.5:
                signals.append("布林带收缩，可能突破")
            
            if latest_features.get('technical_features_bb_expansion', 0) > 0.5:
                signals.append("布林带扩张，波动性增加")
            
            # 情感信号
            if latest_features.get('sentiment_features_extreme_fear', 0) > 0.5:
                signals.append("极度恐慌情绪")
            
            if latest_features.get('sentiment_features_extreme_greed', 0) > 0.5:
                signals.append("极度贪婪情绪")
            
            # 风险信号
            if latest_features.get('risk_features_volatility', 0) > 0.04:
                signals.append("高波动性警告")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"关键信号提取失败: {e}")
            return []
    
    def _generate_warnings(self, features_df: pd.DataFrame, market_score: float, risk_level: RiskLevel) -> List[str]:
        """生成预警信息"""
        warnings = []
        
        try:
            # 市场评分预警
            if market_score > 80:
                warnings.append("市场过热警告 - 评分过高")
            elif market_score < -80:
                warnings.append("市场过冷警告 - 评分过低")
            
            # 风险预警
            if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
                warnings.append(f"高风险警告 - 当前风险等级: {risk_level.value}")
            
            # 波动性预警
            latest_features = features_df.iloc[-1]
            volatility = latest_features.get('risk_features_volatility', 0.02)
            if volatility > 0.05:
                warnings.append("极高波动性警告")
            elif volatility > 0.03:
                warnings.append("高波动性警告")
            
            # 情感极端预警
            fgi = latest_features.get('sentiment_features_fear_greed_index', 50)
            if fgi < 20:
                warnings.append("极度恐慌情绪预警")
            elif fgi > 80:
                warnings.append("极度贪婪情绪预警")
            
            return warnings
            
        except Exception as e:
            self.logger.error(f"预警生成失败: {e}")
            return []
    
    def _get_latest_features(self) -> Optional[MarketFeatures]:
        """获取最新特征"""
        try:
            with self.lock:
                for data_point in reversed(self.historical_data):
                    if 'features' in data_point and data_point['features']:
                        return data_point['features']
            return None
        except Exception as e:
            self.logger.error(f"获取最新特征失败: {e}")
            return None
    
    def _check_alerts(self, result: MarketPerceptionResult):
        """检查预警条件"""
        if not self.alert_config.enable_alerts:
            return
        
        try:
            # 风险预警
            if result.market_score > self.alert_config.risk_threshold:
                self._trigger_alert("HIGH_SCORE", f"市场评分过高: {result.market_score:.2f}")
            
            # 机会预警
            if result.market_score < -self.alert_config.opportunity_threshold:
                self._trigger_alert("OPPORTUNITY", f"市场机会评分: {result.market_score:.2f}")
            
            # 波动性预警
            if result.features and hasattr(result.features, 'risk_features'):
                volatility = result.features.risk_features.get('volatility', 0)
                if volatility > self.alert_config.volatility_threshold:
                    self._trigger_alert("HIGH_VOLATILITY", f"波动性过高: {volatility:.4f}")
            
            # 情感预警
            if result.features and hasattr(result.features, 'sentiment_features'):
                fgi = result.features.sentiment_features.get('fear_greed_index', 50)
                if fgi > 90 or fgi < 10:
                    self._trigger_alert("EXTREME_SENTIMENT", f"极端情感指数: {fgi}")
            
        except Exception as e:
            self.logger.error(f"预警检查失败: {e}")
    
    def _trigger_alert(self, alert_type: str, message: str):
        """触发预警"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'level': 'WARNING'
        }
        
        self.active_alerts.append(alert)
        
        # 保持最近100条预警
        if len(self.active_alerts) > 100:
            self.active_alerts = self.active_alerts[-100:]
        
        # 记录日志
        self.logger.warning(f"市场预警 [{alert_type}]: {message}")
        
        # 保存到文件
        if "file" in self.alert_config.notification_methods:
            self._save_alert_to_file(alert)
    
    def _save_alert_to_file(self, alert: Dict[str, Any]):
        """保存预警到文件"""
        try:
            alert_file = Path("market_alerts.json")
            alerts = []
            
            if alert_file.exists():
                with open(alert_file, 'r', encoding='utf-8') as f:
                    alerts = json.load(f)
            
            alerts.append(alert)
            
            # 只保留最近1000条预警
            if len(alerts) > 1000:
                alerts = alerts[-1000:]
            
            with open(alert_file, 'w', encoding='utf-8') as f:
                json.dump(alerts, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"预警保存失败: {e}")
    
    def get_current_perception(self) -> Optional[MarketPerceptionResult]:
        """获取当前市场感知结果"""
        try:
            with self.lock:
                if len(self.historical_data) < 10:
                    return None
                
                return self._perform_market_perception()
                
        except Exception as e:
            self.logger.error(f"获取当前感知结果失败: {e}")
            return None
    
    def get_historical_perception(self, hours: int = 24) -> List[MarketPerceptionResult]:
        """获取历史市场感知结果"""
        try:
            with self.lock:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                historical_results = []
                
                for data_point in self.historical_data:
                    if data_point['timestamp'] >= cutoff_time:
                        if 'perception_result' in data_point:
                            historical_results.append(data_point['perception_result'])
                
                return historical_results
                
        except Exception as e:
            self.logger.error(f"获取历史感知结果失败: {e}")
            return []
    
    def train_market_phase_classifier(self, training_data: pd.DataFrame, labels: pd.Series):
        """训练市场阶段分类器"""
        try:
            self.logger.info("开始训练市场阶段分类器")
            
            # 数据预处理
            X_train, X_test, y_train, y_test = train_test_split(
                training_data, labels, test_size=0.2, random_state=42
            )
            
            # 特征标准化
            self.scalers['market_phase'] = StandardScaler()
            X_train_scaled = self.scalers['market_phase'].fit_transform(X_train)
            X_test_scaled = self.scalers['market_phase'].transform(X_test)
            
            # 特征选择
            self.feature_selectors['market_phase'] = SelectKBest(f_classif, k=20)
            X_train_selected = self.feature_selectors['market_phase'].fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selectors['market_phase'].transform(X_test_scaled)
            
            # 模型训练
            self.models['market_phase'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.models['market_phase'].fit(X_train_selected, y_train)
            
            # 模型评估
            train_score = self.models['market_phase'].score(X_train_selected, y_train)
            test_score = self.models['market_phase'].score(X_test_selected, y_test)
            
            self.logger.info(f"市场阶段分类器训练完成 - 训练准确率: {train_score:.3f}, 测试准确率: {test_score:.3f}")
            
            return {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'feature_importance': dict(zip(
                    training_data.columns[self.feature_selectors['market_phase'].get_support()],
                    self.models['market_phase'].feature_importances_
                ))
            }
            
        except Exception as e:
            self.logger.error(f"市场阶段分类器训练失败: {e}")
            return None
    
    def visualize_market_perception(self, result: MarketPerceptionResult, save_path: str = None):
        """可视化市场感知结果"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'市场感知分析结果 - {result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16)
            
            # 1. 市场阶段饼图
            ax1 = axes[0, 0]
            phase_counts = {phase.value: 1 for phase in MarketPhase}
            phase_counts[result.market_phase.value] = 1
            colors = ['lightblue' if phase != result.market_phase.value else 'red' for phase in phase_counts.keys()]
            ax1.pie(phase_counts.values(), labels=phase_counts.keys(), autopct='%1.1f%%', colors=colors)
            ax1.set_title(f'当前市场阶段: {result.market_phase.value}')
            
            # 2. 市场评分仪表盘
            ax2 = axes[0, 1]
            score_ranges = ['极差(-100~-80)', '差(-80~-40)', '中性(-40~40)', '好(40~80)', '极好(80~100)']
            score_colors = ['darkred', 'red', 'yellow', 'lightgreen', 'green']
            
            # 创建评分条形图
            bars = ax2.barh(score_ranges, [20, 40, 80, 40, 20], color=score_colors, alpha=0.7)
            ax2.axvline(x=result.market_score, color='black', linewidth=3, label=f'当前评分: {result.market_score:.1f}')
            ax2.set_xlim(-100, 100)
            ax2.set_xlabel('市场评分')
            ax2.set_title('市场综合评分')
            ax2.legend()
            
            # 3. 风险等级雷达图
            ax3 = axes[0, 2]
            risk_levels = ['极低', '低', '中等', '高', '极高', '极端']
            risk_values = [1, 2, 3, 4, 5, 6]
            current_risk = list(RiskLevel).index(result.risk_level) + 1
            
            angles = np.linspace(0, 2 * np.pi, len(risk_levels), endpoint=False).tolist()
            values = [1] * len(risk_levels)
            values[current_risk - 1] = 2
            
            ax3 = plt.subplot(2, 3, 3, projection='polar')
            ax3.plot(angles, values, 'o-', linewidth=2, color='red')
            ax3.fill(angles, values, alpha=0.25, color='red')
            ax3.set_xticks(angles)
            ax3.set_xticklabels(risk_levels)
            ax3.set_ylim(0, 2)
            ax3.set_title(f'风险等级: {result.risk_level.value}')
            
            # 4. 机会类型
            ax4 = axes[1, 0]
            opportunity_types = [op.value for op in OpportunityType]
            opportunity_scores = [50 if op != result.opportunity_type.value else 100 for op in opportunity_types]
            
            bars = ax4.bar(range(len(opportunity_types)), opportunity_scores, 
                          color=['green' if score == 100 else 'lightgray' for score in opportunity_scores])
            ax4.set_xticks(range(len(opportunity_types)))
            ax4.set_xticklabels([op[:4] for op in opportunity_types], rotation=45)
            ax4.set_ylabel('机会评分')
            ax4.set_title(f'当前机会类型: {result.opportunity_type.value}')
            
            # 5. 置信度
            ax5 = axes[1, 1]
            confidence_data = [result.confidence, 1 - result.confidence]
            confidence_labels = ['置信度', '不确定性']
            colors = ['green', 'lightcoral']
            
            ax5.pie(confidence_data, labels=confidence_labels, autopct='%1.1f%%', colors=colors)
            ax5.set_title(f'预测置信度: {result.confidence:.1%}')
            
            # 6. 关键信号和预警
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # 关键信号
            signals_text = "关键信号:\n" + "\n".join([f"• {signal}" for signal in result.key_signals])
            ax6.text(0.1, 0.8, signals_text, transform=ax6.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            # 预警信息
            warnings_text = "预警信息:\n" + "\n".join([f"⚠ {warning}" for warning in result.warnings])
            ax6.text(0.1, 0.4, warnings_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
            ax6.set_title('信号与预警')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"市场感知可视化结果已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"市场感知可视化失败: {e}")
    
    def generate_perception_report(self, result: MarketPerceptionResult) -> str:
        """生成市场感知报告"""
        try:
            report = f"""
市场感知分析报告
================
分析时间: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
预测时间窗口: {result.prediction_horizon}小时

一、市场阶段分析
当前阶段: {result.market_phase.value}
阶段置信度: {result.confidence:.1%}

二、市场综合评分
综合评分: {result.market_score:.2f}/100
评分解释: {self._interpret_market_score(result.market_score)}

三、风险评估
风险等级: {result.risk_level.value}
风险描述: {self._interpret_risk_level(result.risk_level)}

四、机会识别
机会类型: {result.opportunity_type.value}
机会描述: {self._interpret_opportunity_type(result.opportunity_type)}

五、关键信号
{chr(10).join([f"• {signal}" for signal in result.key_signals]) if result.key_signals else "无关键信号"}

六、风险预警
{chr(10).join([f"⚠ {warning}" for warning in result.warnings]) if result.warnings else "无预警信息"}

七、投资建议
{self._generate_investment_advice(result)}

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
市场感知引擎 v1.0.0
            """
            
            return report.strip()
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            return "报告生成失败"
    
    def _interpret_market_score(self, score: float) -> str:
        """解释市场评分"""
        if score >= 80:
            return "市场表现极佳，存在过热风险"
        elif score >= 60:
            return "市场表现良好，偏向乐观"
        elif score >= 40:
            return "市场表现中性，方向不明确"
        elif score >= 20:
            return "市场表现偏弱，需要谨慎"
        elif score >= -20:
            return "市场表现疲软，偏向悲观"
        elif score >= -40:
            return "市场表现较差，存在机会"
        elif score >= -60:
            return "市场表现很差，可能超跌"
        else:
            return "市场表现极差，可能触底"
    
    def _interpret_risk_level(self, risk_level: RiskLevel) -> str:
        """解释风险等级"""
        interpretations = {
            RiskLevel.VERY_LOW: "市场风险极低，适合积极投资",
            RiskLevel.LOW: "市场风险较低，可以适度投资",
            RiskLevel.MEDIUM: "市场风险中等，需要平衡配置",
            RiskLevel.HIGH: "市场风险较高，建议谨慎投资",
            RiskLevel.VERY_HIGH: "市场风险很高，建议减少仓位",
            RiskLevel.EXTREME: "市场风险极高，建议观望或减仓"
        }
        return interpretations.get(risk_level, "风险等级未知")
    
    def _interpret_opportunity_type(self, opportunity_type: OpportunityType) -> str:
        """解释机会类型"""
        interpretations = {
            OpportunityType.TRENDING_UP: "市场处于上升趋势，适合做多",
            OpportunityType.TRENDING_DOWN: "市场处于下降趋势，适合做空或避险",
            OpportunityType.BREAKOUT: "市场可能出现突破，关注支撑阻力位",
            OpportunityType.REVERSAL: "市场可能出现反转，关注超买超卖",
            OpportunityType.MOMENTUM: "市场存在动量机会，跟随趋势",
            OpportunityType.VALUE: "市场存在价值机会，关注估值修复",
            OpportunityType.ARBITRAGE: "市场存在套利机会，关注价差交易",
            OpportunityType.NONE: "当前无明显机会，建议观望"
        }
        return interpretations.get(opportunity_type, "机会类型未知")
    
    def _generate_investment_advice(self, result: MarketPerceptionResult) -> str:
        """生成投资建议"""
        advice_parts = []
        
        # 基于市场评分给出建议
        if result.market_score > 70:
            advice_parts.append("• 市场过热，建议获利了结或减少仓位")
        elif result.market_score < -70:
            advice_parts.append("• 市场超跌，可能存在抄底机会")
        
        # 基于风险等级给出建议
        if result.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
            advice_parts.append("• 当前风险较高，建议降低仓位或使用对冲策略")
        
        # 基于机会类型给出建议
        if result.opportunity_type == OpportunityType.TRENDING_UP:
            advice_parts.append("• 趋势向上，可考虑顺势做多")
        elif result.opportunity_type == OpportunityType.TRENDING_DOWN:
            advice_parts.append("• 趋势向下，可考虑做空或避险")
        elif result.opportunity_type == OpportunityType.REVERSAL:
            advice_parts.append("• 可能出现反转，可考虑逆向操作")
        
        # 基于置信度给出建议
        if result.confidence < 0.6:
            advice_parts.append("• 当前预测置信度较低，建议谨慎操作")
        
        if not advice_parts:
            advice_parts.append("• 当前市场状态中性，建议维持现有策略")
        
        return "\n".join(advice_parts)
    
    def export_perception_data(self, file_path: str):
        """导出感知数据"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'current_result': asdict(self.get_current_perception()) if self.get_current_perception() else None,
                'historical_data': [
                    {
                        'timestamp': data['timestamp'].isoformat(),
                        'data': data
                    }
                    for data in list(self.historical_data)[-100:]  # 最近100条记录
                ],
                'active_alerts': self.active_alerts[-50:],  # 最近50条预警
                'model_status': {
                    'models_trained': list(self.models.keys()),
                    'scalers_available': list(self.scalers.keys()),
                    'feature_selectors_available': list(self.feature_selectors.keys())
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"感知数据已导出到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"数据导出失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            status = {
                'is_running': self.is_running,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'data_points_count': len(self.historical_data),
                'realtime_data_count': len(self.realtime_data),
                'active_alerts_count': len(self.active_alerts),
                'models_status': {
                    model_name: 'trained' if model_name in self.models else 'not_trained'
                    for model_name in ['market_phase', 'risk_assessor']
                },
                'data_sources_status': {
                    name: 'available' if source else 'unavailable'
                    for name, source in self.data_sources.items()
                },
                'system_health': 'healthy' if self.is_running and self.last_update else 'unhealthy'
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {'error': str(e)}


# 示例使用和测试函数
def demo_market_perception_engine():
    """演示市场感知引擎的使用"""
    print("=== 市场感知引擎演示 ===")
    
    # 创建引擎实例
    engine = MarketPerceptionEngine()
    
    # 启动实时监控
    print("启动市场感知引擎...")
    engine.start_real_time_monitoring()
    
    try:
        # 等待一些数据
        print("等待数据收集...")
        time.sleep(10)
        
        # 获取当前感知结果
        current_result = engine.get_current_perception()
        if current_result:
            print(f"\n当前市场感知结果:")
            print(f"市场阶段: {current_result.market_phase.value}")
            print(f"市场评分: {current_result.market_score:.2f}")
            print(f"风险等级: {current_result.risk_level.value}")
            print(f"机会类型: {current_result.opportunity_type.value}")
            print(f"置信度: {current_result.confidence:.1%}")
            
            # 生成报告
            report = engine.generate_perception_report(current_result)
            print(f"\n市场感知报告:\n{report}")
            
            # 可视化
            engine.visualize_market_perception(current_result, "market_perception_demo.png")
        
        # 获取系统状态
        status = engine.get_system_status()
        print(f"\n系统状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    finally:
        # 停止监控
        engine.stop_real_time_monitoring()
        print("市场感知引擎已停止")


if __name__ == "__main__":
    # 运行演示
    demo_market_perception_engine()