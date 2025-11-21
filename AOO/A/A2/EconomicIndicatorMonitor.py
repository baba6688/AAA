#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2经济指标监控器
Economic Indicator Monitor

功能包括：
1. 主要经济指标数据获取（GDP、CPI、失业率等）
2. 央行政策利率监控
3. 汇率数据获取
4. 大宗商品价格监控
5. 债券收益率曲线
6. 经济数据预警系统

数据源：
- yfinance: 美股、指数、ETF、商品等
- pandas_datareader: 美联储、欧央行等官方数据
- FRED API: 美联储经济数据
- 中国央行数据


创建时间: 2025-11-05
"""

import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import requests
import json
import datetime as dt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

@dataclass
class EconomicIndicator:
    """经济指标数据类"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    change: Optional[float] = None
    change_percent: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None

@dataclass
class AlertRule:
    """预警规则类"""
    indicator: str
    condition: str  # 'above', 'below', 'change'
    threshold: float
    message: str
    enabled: bool = True

class EconomicIndicatorMonitor:
    """经济指标监控器主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化经济指标监控器
        
        Args:
            config: 配置字典，包含API密钥、刷新频率等
        """
        self.config = config or {}
        self.data_cache = {}
        self.alert_rules = []
        self.alerts = []
        self.last_update = {}
        
        # 设置更新频率（秒）
        self.update_interval = self.config.get('update_interval', 300)  # 5分钟
        
        # 初始化线程锁
        self.lock = threading.Lock()
        
        # 创建数据目录
        self.data_dir = Path("economic_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化预警规则
        self._init_alert_rules()
        
        logger.info("经济指标监控器初始化完成")
    
    def _init_alert_rules(self):
        """初始化预警规则"""
        # 美联储利率预警
        self.alert_rules.append(AlertRule(
            indicator="FED_RATE",
            condition="above",
            threshold=5.0,
            message="美联储利率突破5%，可能影响股市和债市"
        ))
        
        # 失业率预警
        self.alert_rules.append(AlertRule(
            indicator="UNEMPLOYMENT_RATE",
            condition="above",
            threshold=6.0,
            message="失业率超过6%，经济可能陷入衰退"
        ))
        
        # CPI通胀预警
        self.alert_rules.append(AlertRule(
            indicator="CPI",
            condition="above",
            threshold=3.0,
            message="CPI通胀率超过3%，央行可能收紧政策"
        ))
        
        # 美元指数预警
        self.alert_rules.append(AlertRule(
            indicator="DXY",
            condition="above",
            threshold=105.0,
            message="美元指数强势，可能对新兴市场造成压力"
        ))
        
        # 黄金价格预警
        self.alert_rules.append(AlertRule(
            indicator="GOLD",
            condition="above",
            threshold=2100.0,
            message="黄金价格创新高，市场避险情绪升温"
        ))
        
        logger.info(f"初始化了 {len(self.alert_rules)} 条预警规则")
    
    def get_fed_data(self) -> Dict[str, EconomicIndicator]:
        """获取美联储经济数据"""
        logger.info("获取美联储经济数据...")
        indicators = {}
        
        try:
            # 获取联邦基金利率
            fed_rate = pdr.get_data_fred('FEDFUNDS', start='2020-01-01')
            latest_rate = fed_rate['FEDFUNDS'].iloc[-1]
            previous_rate = fed_rate['FEDFUNDS'].iloc[-2]
            change = latest_rate - previous_rate
            
            indicators['FED_RATE'] = EconomicIndicator(
                name="美联储联邦基金利率",
                value=latest_rate,
                unit="%",
                timestamp=datetime.now(),
                source="FRED",
                change=change,
                change_percent=(change/previous_rate)*100 if previous_rate != 0 else 0
            )
            
            # 获取失业率
            unemployment = pdr.get_data_fred('UNRATE', start='2020-01-01')
            latest_unemployment = unemployment['UNRATE'].iloc[-1]
            previous_unemployment = unemployment['UNRATE'].iloc[-2]
            unemployment_change = latest_unemployment - previous_unemployment
            
            indicators['UNEMPLOYMENT_RATE'] = EconomicIndicator(
                name="美国失业率",
                value=latest_unemployment,
                unit="%",
                timestamp=datetime.now(),
                source="FRED",
                change=unemployment_change,
                change_percent=(unemployment_change/previous_unemployment)*100 if previous_unemployment != 0 else 0
            )
            
            # 获取CPI
            cpi = pdr.get_data_fred('CPIAUCSL', start='2020-01-01')
            latest_cpi = cpi['CPIAUCSL'].iloc[-1]
            previous_cpi = cpi['CPIAUCSL'].iloc[-2]
            cpi_change = ((latest_cpi - previous_cpi) / previous_cpi) * 100
            
            indicators['CPI'] = EconomicIndicator(
                name="美国CPI通胀率",
                value=cpi_change,
                unit="%",
                timestamp=datetime.now(),
                source="FRED",
                change=cpi_change,
                change_percent=cpi_change
            )
            
            # 获取GDP增长率
            gdp = pdr.get_data_fred('GDP', start='2020-01-01')
            latest_gdp = gdp['GDP'].iloc[-1]
            previous_gdp = gdp['GDP'].iloc[-2]
            gdp_change = ((latest_gdp - previous_gdp) / previous_gdp) * 100
            
            indicators['GDP_GROWTH'] = EconomicIndicator(
                name="美国GDP增长率",
                value=gdp_change,
                unit="%",
                timestamp=datetime.now(),
                source="FRED",
                change=gdp_change,
                change_percent=gdp_change
            )
            
            logger.info("美联储数据获取成功")
            
        except Exception as e:
            logger.error(f"获取美联储数据失败: {e}")
        
        return indicators
    
    def get_central_bank_rates(self) -> Dict[str, EconomicIndicator]:
        """获取各国央行利率"""
        logger.info("获取央行利率数据...")
        indicators = {}
        
        try:
            # 美联储利率
            fed_funds = pdr.get_data_fred('FEDFUNDS', start='2020-01-01')
            indicators['FED_RATE'] = EconomicIndicator(
                name="美联储联邦基金利率",
                value=fed_funds['FEDFUNDS'].iloc[-1],
                unit="%",
                timestamp=datetime.now(),
                source="FRED"
            )
            
            # 欧央行利率
            ecb_rate = pdr.get_data_fred('ECBDFR', start='2020-01-01')
            indicators['ECB_RATE'] = EconomicIndicator(
                name="欧央行存款利率",
                value=ecb_rate['ECBDFR'].iloc[-1],
                unit="%",
                timestamp=datetime.now(),
                source="FRED"
            )
            
            # 日本央行利率
            boj_rate = pdr.get_data_fred('BOJDFR', start='2020-01-01')
            indicators['BOJ_RATE'] = EconomicIndicator(
                name="日本央行政策利率",
                value=boj_rate['BOJDFR'].iloc[-1],
                unit="%",
                timestamp=datetime.now(),
                source="FRED"
            )
            
            # 英国央行利率
            bank_rate = pdr.get_data_fred('BANKREALLRATE', start='2020-01-01')
            indicators['BOE_RATE'] = EconomicIndicator(
                name="英国央行利率",
                value=bank_rate['BANKREALLRATE'].iloc[-1],
                unit="%",
                timestamp=datetime.now(),
                source="FRED"
            )
            
            logger.info("央行利率数据获取成功")
            
        except Exception as e:
            logger.error(f"获取央行利率数据失败: {e}")
        
        return indicators
    
    def get_fx_rates(self) -> Dict[str, EconomicIndicator]:
        """获取汇率数据"""
        logger.info("获取汇率数据...")
        indicators = {}
        
        try:
            # 美元指数
            dxy = yf.Ticker("DX-Y.NYB")
            dxy_data = dxy.history(period="5d")
            if not dxy_data.empty:
                latest_dxy = dxy_data['Close'].iloc[-1]
                previous_dxy = dxy_data['Close'].iloc[-2]
                dxy_change = latest_dxy - previous_dxy
                
                indicators['DXY'] = EconomicIndicator(
                    name="美元指数",
                    value=latest_dxy,
                    unit="",
                    timestamp=datetime.now(),
                    source="Yahoo Finance",
                    change=dxy_change,
                    change_percent=(dxy_change/previous_dxy)*100 if previous_dxy != 0 else 0
                )
            
            # EUR/USD
            eurusd = yf.Ticker("EURUSD=X")
            eurusd_data = eurusd.history(period="5d")
            if not eurusd_data.empty:
                latest_eurusd = eurusd_data['Close'].iloc[-1]
                previous_eurusd = eurusd_data['Close'].iloc[-2]
                eurusd_change = latest_eurusd - previous_eurusd
                
                indicators['EURUSD'] = EconomicIndicator(
                    name="EUR/USD",
                    value=latest_eurusd,
                    unit="",
                    timestamp=datetime.now(),
                    source="Yahoo Finance",
                    change=eurusd_change,
                    change_percent=(eurusd_change/previous_eurusd)*100 if previous_eurusd != 0 else 0
                )
            
            # USD/CNY
            usdcny = yf.Ticker("USDCNY=X")
            usdcny_data = usdcny.history(period="5d")
            if not usdcny_data.empty:
                latest_usdcny = usdcny_data['Close'].iloc[-1]
                previous_usdcny = usdcny_data['Close'].iloc[-2]
                usdcny_change = latest_usdcny - previous_usdcny
                
                indicators['USDCNY'] = EconomicIndicator(
                    name="USD/CNY",
                    value=latest_usdcny,
                    unit="",
                    timestamp=datetime.now(),
                    source="Yahoo Finance",
                    change=usdcny_change,
                    change_percent=(usdcny_change/previous_usdcny)*100 if previous_usdcny != 0 else 0
                )
            
            # GBP/USD
            gbpusd = yf.Ticker("GBPUSD=X")
            gbpusd_data = gbpusd.history(period="5d")
            if not gbpusd_data.empty:
                latest_gbpusd = gbpusd_data['Close'].iloc[-1]
                previous_gbpusd = gbpusd_data['Close'].iloc[-2]
                gbpusd_change = latest_gbpusd - previous_gbpusd
                
                indicators['GBPUSD'] = EconomicIndicator(
                    name="GBP/USD",
                    value=latest_gbpusd,
                    unit="",
                    timestamp=datetime.now(),
                    source="Yahoo Finance",
                    change=gbpusd_change,
                    change_percent=(gbpusd_change/previous_gbpusd)*100 if previous_gbpusd != 0 else 0
                )
            
            logger.info("汇率数据获取成功")
            
        except Exception as e:
            logger.error(f"获取汇率数据失败: {e}")
        
        return indicators
    
    def get_commodity_prices(self) -> Dict[str, EconomicIndicator]:
        """获取大宗商品价格"""
        logger.info("获取大宗商品价格...")
        indicators = {}
        
        try:
            # 黄金
            gold = yf.Ticker("GC=F")
            gold_data = gold.history(period="5d")
            if not gold_data.empty:
                latest_gold = gold_data['Close'].iloc[-1]
                previous_gold = gold_data['Close'].iloc[-2]
                gold_change = latest_gold - previous_gold
                
                indicators['GOLD'] = EconomicIndicator(
                    name="黄金期货价格",
                    value=latest_gold,
                    unit="USD/盎司",
                    timestamp=datetime.now(),
                    source="Yahoo Finance",
                    change=gold_change,
                    change_percent=(gold_change/previous_gold)*100 if previous_gold != 0 else 0
                )
            
            # 白银
            silver = yf.Ticker("SI=F")
            silver_data = silver.history(period="5d")
            if not silver_data.empty:
                latest_silver = silver_data['Close'].iloc[-1]
                previous_silver = silver_data['Close'].iloc[-2]
                silver_change = latest_silver - previous_silver
                
                indicators['SILVER'] = EconomicIndicator(
                    name="白银期货价格",
                    value=latest_silver,
                    unit="USD/盎司",
                    timestamp=datetime.now(),
                    source="Yahoo Finance",
                    change=silver_change,
                    change_percent=(silver_change/previous_silver)*100 if previous_silver != 0 else 0
                )
            
            # 原油
            oil = yf.Ticker("CL=F")
            oil_data = oil.history(period="5d")
            if not oil_data.empty:
                latest_oil = oil_data['Close'].iloc[-1]
                previous_oil = oil_data['Close'].iloc[-2]
                oil_change = latest_oil - previous_oil
                
                indicators['OIL'] = EconomicIndicator(
                    name="原油期货价格",
                    value=latest_oil,
                    unit="USD/桶",
                    timestamp=datetime.now(),
                    source="Yahoo Finance",
                    change=oil_change,
                    change_percent=(oil_change/previous_oil)*100 if previous_oil != 0 else 0
                )
            
            # 铜
            copper = yf.Ticker("HG=F")
            copper_data = copper.history(period="5d")
            if not copper_data.empty:
                latest_copper = copper_data['Close'].iloc[-1]
                previous_copper = copper_data['Close'].iloc[-2]
                copper_change = latest_copper - previous_copper
                
                indicators['COPPER'] = EconomicIndicator(
                    name="铜期货价格",
                    value=latest_copper,
                    unit="USD/磅",
                    timestamp=datetime.now(),
                    source="Yahoo Finance",
                    change=copper_change,
                    change_percent=(copper_change/previous_copper)*100 if previous_copper != 0 else 0
                )
            
            # 天然气
            natural_gas = yf.Ticker("NG=F")
            ng_data = natural_gas.history(period="5d")
            if not ng_data.empty:
                latest_ng = ng_data['Close'].iloc[-1]
                previous_ng = ng_data['Close'].iloc[-2]
                ng_change = latest_ng - previous_ng
                
                indicators['NATURAL_GAS'] = EconomicIndicator(
                    name="天然气期货价格",
                    value=latest_ng,
                    unit="USD/MMBtu",
                    timestamp=datetime.now(),
                    source="Yahoo Finance",
                    change=ng_change,
                    change_percent=(ng_change/previous_ng)*100 if previous_ng != 0 else 0
                )
            
            logger.info("大宗商品价格获取成功")
            
        except Exception as e:
            logger.error(f"获取大宗商品价格失败: {e}")
        
        return indicators
    
    def get_bond_yields(self) -> Dict[str, EconomicIndicator]:
        """获取债券收益率曲线"""
        logger.info("获取债券收益率数据...")
        indicators = {}
        
        try:
            # 美国10年期国债收益率
            treasury_10y = pdr.get_data_fred('DGS10', start='2020-01-01')
            latest_10y = treasury_10y['DGS10'].iloc[-1]
            previous_10y = treasury_10y['DGS10'].iloc[-2]
            yield_change = latest_10y - previous_10y
            
            indicators['TREASURY_10Y'] = EconomicIndicator(
                name="美国10年期国债收益率",
                value=latest_10y,
                unit="%",
                timestamp=datetime.now(),
                source="FRED",
                change=yield_change,
                change_percent=(yield_change/previous_10y)*100 if previous_10y != 0 else 0
            )
            
            # 美国2年期国债收益率
            treasury_2y = pdr.get_data_fred('DGS2', start='2020-01-01')
            latest_2y = treasury_2y['DGS2'].iloc[-1]
            previous_2y = treasury_2y['DGS2'].iloc[-2]
            yield_2y_change = latest_2y - previous_2y
            
            indicators['TREASURY_2Y'] = EconomicIndicator(
                name="美国2年期国债收益率",
                value=latest_2y,
                unit="%",
                timestamp=datetime.now(),
                source="FRED",
                change=yield_2y_change,
                change_percent=(yield_2y_change/previous_2y)*100 if previous_2y != 0 else 0
            )
            
            # 美国30年期国债收益率
            treasury_30y = pdr.get_data_fred('DGS30', start='2020-01-01')
            latest_30y = treasury_30y['DGS30'].iloc[-1]
            previous_30y = treasury_30y['DGS30'].iloc[-2]
            yield_30y_change = latest_30y - previous_30y
            
            indicators['TREASURY_30Y'] = EconomicIndicator(
                name="美国30年期国债收益率",
                value=latest_30y,
                unit="%",
                timestamp=datetime.now(),
                source="FRED",
                change=yield_30y_change,
                change_percent=(yield_30y_change/previous_30y)*100 if previous_30y != 0 else 0
            )
            
            # 计算收益率曲线斜率（10年-2年）
            yield_curve_slope = latest_10y - latest_2y
            indicators['YIELD_CURVE_SLOPE'] = EconomicIndicator(
                name="收益率曲线斜率(10Y-2Y)",
                value=yield_curve_slope,
                unit="%",
                timestamp=datetime.now(),
                source="FRED"
            )
            
            logger.info("债券收益率数据获取成功")
            
        except Exception as e:
            logger.error(f"获取债券收益率数据失败: {e}")
        
        return indicators
    
    def check_alerts(self, indicators: Dict[str, EconomicIndicator]) -> List[str]:
        """检查预警规则"""
        alerts = []
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            if rule.indicator not in indicators:
                continue
                
            indicator = indicators[rule.indicator]
            
            if rule.condition == 'above' and indicator.value > rule.threshold:
                alerts.append(f"🚨 {rule.message} (当前值: {indicator.value:.2f}{indicator.unit})")
            elif rule.condition == 'below' and indicator.value < rule.threshold:
                alerts.append(f"🚨 {rule.message} (当前值: {indicator.value:.2f}{indicator.unit})")
            elif rule.condition == 'change' and abs(indicator.change or 0) > rule.threshold:
                alerts.append(f"🚨 {rule.message} (变化: {indicator.change:.2f}{indicator.unit})")
        
        return alerts
    
    def update_all_data(self) -> Dict[str, Any]:
        """更新所有经济数据"""
        logger.info("开始更新所有经济数据...")
        
        all_indicators = {}
        
        # 并行获取数据
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(self.get_fed_data): 'fed_data',
                executor.submit(self.get_central_bank_rates): 'central_bank_rates',
                executor.submit(self.get_fx_rates): 'fx_rates',
                executor.submit(self.get_commodity_prices): 'commodity_prices',
                executor.submit(self.get_bond_yields): 'bond_yields'
            }
            
            for future in as_completed(futures):
                try:
                    data = future.result()
                    category = futures[future]
                    all_indicators.update(data)
                    logger.info(f"{category} 数据更新完成")
                except Exception as e:
                    logger.error(f"数据更新失败 {futures[future]}: {e}")
        
        # 检查预警
        alerts = self.check_alerts(all_indicators)
        
        # 保存数据
        with self.lock:
            self.data_cache = all_indicators
            self.alerts = alerts
            self.last_update = datetime.now()
        
        # 保存到文件
        self._save_data_to_file(all_indicators)
        
        logger.info(f"数据更新完成，共获取 {len(all_indicators)} 个指标")
        if alerts:
            logger.warning(f"触发 {len(alerts)} 条预警")
        
        return {
            'indicators': all_indicators,
            'alerts': alerts,
            'last_update': self.last_update,
            'total_indicators': len(all_indicators)
        }
    
    def _save_data_to_file(self, indicators: Dict[str, EconomicIndicator]):
        """保存数据到文件"""
        try:
            # 保存最新数据
            latest_data = {}
            for key, indicator in indicators.items():
                latest_data[key] = {
                    'name': indicator.name,
                    'value': indicator.value,
                    'unit': indicator.unit,
                    'timestamp': indicator.timestamp.isoformat(),
                    'source': indicator.source,
                    'change': indicator.change,
                    'change_percent': indicator.change_percent
                }
            
            # 保存为JSON
            with open(self.data_dir / 'latest_indicators.json', 'w', encoding='utf-8') as f:
                json.dump(latest_data, f, ensure_ascii=False, indent=2)
            
            # 保存预警
            with open(self.data_dir / 'alerts.json', 'w', encoding='utf-8') as f:
                json.dump(self.alerts, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
    
    def generate_dashboard(self, save_path: Optional[str] = None) -> str:
        """生成经济指标监控仪表板"""
        logger.info("生成经济指标监控仪表板...")
        
        if not self.data_cache:
            logger.warning("没有数据缓存，请先调用 update_all_data()")
            return ""
        
        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('经济指标监控仪表板', fontsize=16, fontweight='bold')
        
        # 1. 央行利率对比
        ax1 = axes[0, 0]
        central_bank_rates = ['FED_RATE', 'ECB_RATE', 'BOJ_RATE', 'BOE_RATE']
        rate_names = ['美联储', '欧央行', '日本央行', '英国央行']
        rate_values = []
        
        for rate in central_bank_rates:
            if rate in self.data_cache:
                rate_values.append(self.data_cache[rate].value)
            else:
                rate_values.append(0)
        
        bars = ax1.bar(rate_names, rate_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('央行政策利率对比 (%)')
        ax1.set_ylabel('利率 (%)')
        
        # 添加数值标签
        for bar, value in zip(bars, rate_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}%', ha='center', va='bottom')
        
        # 2. 主要汇率
        ax2 = axes[0, 1]
        fx_rates = ['DXY', 'EURUSD', 'USDCNY', 'GBPUSD']
        fx_names = ['美元指数', 'EUR/USD', 'USD/CNY', 'GBP/USD']
        fx_values = []
        
        for fx in fx_rates:
            if fx in self.data_cache:
                fx_values.append(self.data_cache[fx].value)
            else:
                fx_values.append(0)
        
        bars = ax2.bar(fx_names, fx_values, color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
        ax2.set_title('主要汇率')
        ax2.set_ylabel('汇率')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # 3. 大宗商品价格
        ax3 = axes[1, 0]
        commodities = ['GOLD', 'SILVER', 'OIL', 'COPPER']
        commodity_names = ['黄金', '白银', '原油', '铜']
        commodity_values = []
        
        for commodity in commodities:
            if commodity in self.data_cache:
                commodity_values.append(self.data_cache[commodity].value)
            else:
                commodity_values.append(0)
        
        bars = ax3.bar(commodity_names, commodity_values, color=['#bcbd22', '#17becf', '#ff9896', '#c5b0d5'])
        ax3.set_title('大宗商品价格')
        ax3.set_ylabel('价格')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # 4. 债券收益率曲线
        ax4 = axes[1, 1]
        bonds = ['TREASURY_2Y', 'TREASURY_10Y', 'TREASURY_30Y']
        bond_names = ['2年期', '10年期', '30年期']
        bond_values = []
        
        for bond in bonds:
            if bond in self.data_cache:
                bond_values.append(self.data_cache[bond].value)
            else:
                bond_values.append(0)
        
        ax4.plot(bond_names, bond_values, marker='o', linewidth=2, markersize=8)
        ax4.set_title('美国国债收益率曲线 (%)')
        ax4.set_ylabel('收益率 (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. 经济指标雷达图
        ax5 = axes[2, 0]
        economic_indicators = ['UNEMPLOYMENT_RATE', 'CPI', 'GDP_GROWTH']
        indicator_names = ['失业率', 'CPI通胀', 'GDP增长']
        indicator_values = []
        
        for indicator in economic_indicators:
            if indicator in self.data_cache:
                indicator_values.append(self.data_cache[indicator].value)
            else:
                indicator_values.append(0)
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(indicator_names), endpoint=False).tolist()
        indicator_values += indicator_values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax5.plot(angles, indicator_values, 'o-', linewidth=2)
        ax5.fill(angles, indicator_values, alpha=0.25)
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(indicator_names)
        ax5.set_title('主要经济指标')
        ax5.grid(True)
        
        # 6. 预警信息
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        if self.alerts:
            alert_text = "🚨 当前预警:\n\n"
            for i, alert in enumerate(self.alerts, 1):
                alert_text += f"{i}. {alert}\n"
        else:
            alert_text = "✅ 暂无预警"
        
        ax6.text(0.1, 0.9, alert_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = str(self.data_dir / 'economic_dashboard.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"仪表板已保存到: {save_path}")
        return save_path
    
    def get_summary_report(self) -> str:
        """生成经济指标摘要报告"""
        if not self.data_cache:
            return "暂无数据，请先更新数据"
        
        report = []
        report.append("=" * 60)
        report.append("经济指标监控摘要报告")
        report.append("=" * 60)
        report.append(f"更新时间: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"指标总数: {len(self.data_cache)}")
        report.append("")
        
        # 分类显示指标
        categories = {
            '央行利率': ['FED_RATE', 'ECB_RATE', 'BOJ_RATE', 'BOE_RATE'],
            '汇率': ['DXY', 'EURUSD', 'USDCNY', 'GBPUSD'],
            '大宗商品': ['GOLD', 'SILVER', 'OIL', 'COPPER', 'NATURAL_GAS'],
            '债券': ['TREASURY_2Y', 'TREASURY_10Y', 'TREASURY_30Y'],
            '经济指标': ['UNEMPLOYMENT_RATE', 'CPI', 'GDP_GROWTH']
        }
        
        for category, indicators in categories.items():
            report.append(f"📊 {category}:")
            report.append("-" * 30)
            
            for indicator in indicators:
                if indicator in self.data_cache:
                    data = self.data_cache[indicator]
                    change_str = ""
                    if data.change is not None:
                        change_str = f" ({data.change:+.2f}, {data.change_percent:+.2f}%)"
                    
                    report.append(f"  {data.name}: {data.value:.2f}{data.unit}{change_str}")
            
            report.append("")
        
        # 预警信息
        if self.alerts:
            report.append("🚨 预警信息:")
            report.append("-" * 30)
            for i, alert in enumerate(self.alerts, 1):
                report.append(f"  {i}. {alert}")
        else:
            report.append("✅ 暂无预警")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def start_monitoring(self, interval: int = 300):
        """启动实时监控"""
        logger.info(f"启动经济指标实时监控，间隔: {interval}秒")
        
        def monitor_loop():
            while True:
                try:
                    self.update_all_data()
                    
                    # 如果有预警，立即显示
                    if self.alerts:
                        print("\n🚨 发现预警:")
                        for alert in self.alerts:
                            print(f"  - {alert}")
                        print()
                    
                    # 生成仪表板
                    self.generate_dashboard()
                    
                    # 显示摘要
                    print(self.get_summary_report())
                    
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    logger.info("监控已停止")
                    break
                except Exception as e:
                    logger.error(f"监控过程中发生错误: {e}")
                    time.sleep(60)  # 错误时等待1分钟再重试
        
        # 在新线程中运行监控
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        return monitor_thread

def main():
    """主函数 - 演示用法"""
    # 创建监控器实例
    monitor = EconomicIndicatorMonitor()
    
    print("经济指标监控器启动...")
    print("1. 更新所有数据")
    print("2. 生成仪表板")
    print("3. 显示摘要报告")
    print("4. 启动实时监控")
    print("5. 退出")
    
    while True:
        choice = input("\n请选择操作 (1-5): ").strip()
        
        if choice == '1':
            print("正在更新数据...")
            result = monitor.update_all_data()
            print(f"✅ 数据更新完成，共获取 {result['total_indicators']} 个指标")
            if result['alerts']:
                print(f"🚨 发现 {len(result['alerts'])} 条预警")
        
        elif choice == '2':
            dashboard_path = monitor.generate_dashboard()
            print(f"✅ 仪表板已生成: {dashboard_path}")
        
        elif choice == '3':
            print(monitor.get_summary_report())
        
        elif choice == '4':
            interval = input("请输入监控间隔(秒，默认300): ").strip()
            interval = int(interval) if interval.isdigit() else 300
            monitor.start_monitoring(interval)
            print("✅ 实时监控已启动，按 Ctrl+C 停止")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n监控已停止")
        
        elif choice == '5':
            print("再见!")
            break
        
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()