import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import os

class DataLoader:
    """数据加载器 - 历史数据管理核心模块"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据库配置
        self.db_path = config.get('database_url', 'sqlite:///aoo_system.db').replace('sqlite:///', '')
        self._init_database()
        
        # OKX配置
        self.okx_config = config.get('okx', {})
        self.base_url = self.okx_config.get('base_url', 'https://www.okx.com')
        
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建K线数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kline_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建ticker数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ticker_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    last_price REAL NOT NULL,
                    volume_24h REAL NOT NULL,
                    price_change_24h REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_symbol_time ON kline_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_symbol_time ON ticker_data(symbol, timestamp)')
            
            conn.commit()
            conn.close()
            self.logger.info("数据库初始化完成")
            
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            
    def load_historical_data(self, symbol: str, timeframe: str = '1m', 
                           start_date: str = None, end_date: str = None,
                           limit: int = 1000) -> pd.DataFrame:
        """加载历史K线数据"""
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            # 首先尝试从数据库加载
            df = self._load_from_database(symbol, timeframe, start_date, end_date)
            
            if df.empty:
                # 从API加载
                df = self._load_from_api(symbol, timeframe, start_date, end_date, limit)
                # 保存到数据库
                if not df.empty:
                    self._save_to_database(symbol, timeframe, df)
                    
            return df
            
        except Exception as e:
            self.logger.error(f"加载历史数据失败: {e}")
            return pd.DataFrame()
            
    def _load_from_database(self, symbol: str, timeframe: str, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """从数据库加载数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, open, high, low, close, volume 
                FROM kline_data 
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(query, conn, 
                                 params=[symbol, start_date, end_date],
                                 parse_dates=['timestamp'])
            
            conn.close()
            return df
            
        except Exception as e:
            self.logger.error(f"从数据库加载数据失败: {e}")
            return pd.DataFrame()
            
    def _load_from_api(self, symbol: str, timeframe: str, 
                      start_date: str, end_date: str, limit: int) -> pd.DataFrame:
        """从OKX API加载数据"""
        try:
            url = f"{self.base_url}/api/v5/market/history-candles"
            
            params = {
                'instId': symbol,
                'bar': timeframe,
                'after': start_date,
                'before': end_date,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('code') == '0':
                candles = data.get('data', [])
                return self._parse_candle_data(candles, symbol)
            else:
                self.logger.error(f"API请求失败: {data}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"从API加载数据失败: {e}")
            return pd.DataFrame()
            
    def _parse_candle_data(self, candles: List, symbol: str) -> pd.DataFrame:
        """解析K线数据"""
        if not candles:
            return pd.DataFrame()
            
        data = []
        for candle in candles:
            data.append({
                'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
                'symbol': symbol,
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            })
            
        return pd.DataFrame(data)
        
    def _save_to_database(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """保存数据到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 检查是否已存在数据
            for _, row in df.iterrows():
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM kline_data 
                    WHERE symbol = ? AND timestamp = ?
                ''', (symbol, row['timestamp']))
                
                if cursor.fetchone() is None:
                    cursor.execute('''
                        INSERT INTO kline_data 
                        (symbol, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, row['timestamp'], row['open'], row['high'], 
                         row['low'], row['close'], row['volume']))
            
            conn.commit()
            conn.close()
            self.logger.info(f"保存 {symbol} 数据到数据库完成")
            
        except Exception as e:
            self.logger.error(f"保存数据到数据库失败: {e}")
            
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        try:
            url = f"{self.base_url}/api/v5/market/ticker"
            params = {'instId': symbol}
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('code') == '0':
                ticker = data.get('data', [{}])[0]
                return float(ticker.get('last', 0))
            else:
                self.logger.error(f"获取最新价格失败: {data}")
                return None
                
        except Exception as e:
            self.logger.error(f"获取最新价格异常: {e}")
            return None
            
    def get_multiple_symbols_data(self, symbols: List[str], 
                                timeframe: str = '1m') -> Dict[str, pd.DataFrame]:
        """获取多个交易对的数据"""
        result = {}
        
        for symbol in symbols:
            df = self.load_historical_data(symbol, timeframe)
            if not df.empty:
                result[symbol] = df
                
        return result