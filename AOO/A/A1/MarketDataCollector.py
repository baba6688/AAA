"""
A1市场数据采集器
支持多交易所实时数据采集、清洗、缓存和存储
"""

import asyncio
import json
import logging
import time
import gzip
import pickle
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import ccxt.async_support as ccxt
from ccxt.async_support.base.exchange import Exchange
import websockets
import aiohttp
import numpy as np
from pathlib import Path


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_data_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """标准化市场数据结构"""
    exchange: str
    symbol: str
    timestamp: int
    data_type: str  # 'kline', 'orderbook', 'trade'
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    def compress(self) -> bytes:
        """数据压缩"""
        return gzip.compress(json.dumps(self.to_dict()).encode('utf-8'))
    
    @classmethod
    def from_compressed(cls, data: bytes) -> 'MarketData':
        """从压缩数据恢复"""
        decompressed = gzip.decompress(data).decode('utf-8')
        return cls(**json.loads(decompressed))


@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: str
    api_key: str = ""
    secret: str = ""
    sandbox: bool = False
    rate_limit: int = 100  # 请求限制 (ms)
    max_retries: int = 3
    timeout: int = 30


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_kline_data(data: Dict[str, Any]) -> bool:
        """验证K线数据"""
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_orderbook_data(data: Dict[str, Any]) -> bool:
        """验证深度数据"""
        required_fields = ['bids', 'asks']
        if not all(field in data for field in required_fields):
            return False
        
        # 验证价格和数量格式
        for bid in data['bids'][:10]:  # 只检查前10条
            if not isinstance(bid, list) or len(bid) != 2:
                return False
            try:
                float(bid[0]), float(bid[1])
            except (ValueError, TypeError):
                return False
        
        for ask in data['asks'][:10]:  # 只检查前10条
            if not isinstance(ask, list) or len(ask) != 2:
                return False
            try:
                float(ask[0]), float(ask[1])
            except (ValueError, TypeError):
                return False
        
        return True
    
    @staticmethod
    def validate_trade_data(data: Dict[str, Any]) -> bool:
        """验证成交数据"""
        required_fields = ['id', 'price', 'amount', 'side', 'timestamp']
        return all(field in data for field in required_fields)


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_size))
        self.lock = threading.Lock()
    
    def add_data(self, key: str, data: MarketData) -> None:
        """添加数据到缓存"""
        with self.lock:
            self.cache[key].append(data)
    
    def get_recent_data(self, key: str, count: int = 100) -> List[MarketData]:
        """获取最近数据"""
        with self.lock:
            cache_data = list(self.cache[key])
            return cache_data[-count:] if len(cache_data) >= count else cache_data
    
    def get_data_by_time_range(self, key: str, start_time: int, end_time: int) -> List[MarketData]:
        """按时间范围获取数据"""
        with self.lock:
            return [data for data in self.cache[key] 
                   if start_time <= data.timestamp <= end_time]


class DataStorage:
    """数据存储管理器"""
    
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建市场数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                data_type TEXT NOT NULL,
                data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(exchange, symbol, timestamp, data_type)
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exchange_symbol ON market_data (exchange, symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON market_data (data_type)')
        
        conn.commit()
        conn.close()
    
    def store_data(self, market_data: MarketData) -> bool:
        """存储市场数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            compressed_data = market_data.compress()
            cursor.execute('''
                INSERT OR REPLACE INTO market_data 
                (exchange, symbol, timestamp, data_type, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                market_data.exchange,
                market_data.symbol,
                market_data.timestamp,
                market_data.data_type,
                compressed_data
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"数据存储失败: {e}")
            return False
    
    def batch_store_data(self, data_list: List[MarketData]) -> int:
        """批量存储数据"""
        success_count = 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for data in data_list:
                compressed_data = data.compress()
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (exchange, symbol, timestamp, data_type, data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    data.exchange,
                    data.symbol,
                    data.timestamp,
                    data.data_type,
                    compressed_data
                ))
                success_count += 1
            
            conn.commit()
        except Exception as e:
            logger.error(f"批量数据存储失败: {e}")
        finally:
            conn.close()
        
        return success_count
    
    def get_data_by_time_range(self, exchange: str, symbol: str, 
                              data_type: str, start_time: int, 
                              end_time: int) -> List[MarketData]:
        """按时间范围查询数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data FROM market_data 
            WHERE exchange = ? AND symbol = ? AND data_type = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (exchange, symbol, data_type, start_time, end_time))
        
        results = []
        for row in cursor.fetchall():
            try:
                data = MarketData.from_compressed(row[0])
                results.append(data)
            except Exception as e:
                logger.error(f"数据解压失败: {e}")
        
        conn.close()
        return results


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False
    
    async def connect_exchange(self, exchange_name: str, url: str, 
                              symbols: List[str], data_types: List[str]) -> None:
        """连接交易所WebSocket"""
        try:
            async with websockets.connect(url) as websocket:
                self.connections[exchange_name] = websocket
                logger.info(f"已连接到 {exchange_name} WebSocket")
                
                # 订阅数据流
                await self.subscribe_to_data(websocket, exchange_name, symbols, data_types)
                
                # 监听数据
                async for message in websocket:
                    await self.handle_message(exchange_name, message)
                    
        except Exception as e:
            logger.error(f"{exchange_name} WebSocket连接失败: {e}")
            await self.reconnect_exchange(exchange_name, url, symbols, data_types)
    
    async def subscribe_to_data(self, websocket: websockets.WebSocketServerProtocol,
                               exchange_name: str, symbols: List[str], 
                               data_types: List[str]) -> None:
        """订阅数据流"""
        # 根据不同交易所的WebSocket协议实现订阅逻辑
        if exchange_name == 'binance':
            for symbol in symbols:
                if 'kline' in data_types:
                    await websocket.send(json.dumps({
                        "method": "SUBSCRIBE",
                        "params": [f"{symbol.lower()}@kline_1m"],
                        "id": int(time.time())
                    }))
                if 'trade' in data_types:
                    await websocket.send(json.dumps({
                        "method": "SUBSCRIBE",
                        "params": [f"{symbol.lower()}@trade"],
                        "id": int(time.time())
                    }))
        
        logger.info(f"已订阅 {exchange_name} 数据流: {symbols} - {data_types}")
    
    async def handle_message(self, exchange_name: str, message: str) -> None:
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            
            # 标准化数据格式
            market_data = self.normalize_websocket_data(exchange_name, data)
            
            if market_data:
                # 通知订阅者
                for callback in self.subscribers[exchange_name]:
                    await callback(market_data)
                    
        except Exception as e:
            logger.error(f"处理WebSocket消息失败: {e}")
    
    def normalize_websocket_data(self, exchange_name: str, data: Dict[str, Any]) -> Optional[MarketData]:
        """标准化WebSocket数据"""
        try:
            if 'kline' in str(data):
                # K线数据处理
                kline_data = data.get('k', {})
                return MarketData(
                    exchange=exchange_name,
                    symbol=data.get('s', ''),
                    timestamp=int(kline_data.get('t', 0)),
                    data_type='kline',
                    data={
                        'open': float(kline_data.get('o', 0)),
                        'high': float(kline_data.get('h', 0)),
                        'low': float(kline_data.get('l', 0)),
                        'close': float(kline_data.get('c', 0)),
                        'volume': float(kline_data.get('v', 0))
                    }
                )
            
            elif 'e' in data and data['e'] == 'trade':
                # 成交数据处理
                return MarketData(
                    exchange=exchange_name,
                    symbol=data.get('s', ''),
                    timestamp=int(data.get('T', 0)),
                    data_type='trade',
                    data={
                        'id': data.get('t', ''),
                        'price': float(data.get('p', 0)),
                        'amount': float(data.get('q', 0)),
                        'side': 'buy' if data.get('m', False) else 'sell'
                    }
                )
            
        except Exception as e:
            logger.error(f"数据标准化失败: {e}")
        
        return None
    
    async def reconnect_exchange(self, exchange_name: str, url: str, 
                                symbols: List[str], data_types: List[str]) -> None:
        """重连交易所"""
        logger.info(f"尝试重连 {exchange_name}...")
        await asyncio.sleep(5)  # 等待5秒后重连
        await self.connect_exchange(exchange_name, url, symbols, data_types)
    
    def subscribe(self, exchange_name: str, callback: Callable) -> None:
        """订阅数据回调"""
        self.subscribers[exchange_name].append(callback)
    
    def start(self):
        """启动WebSocket管理器"""
        self.running = True
    
    def stop(self):
        """停止WebSocket管理器"""
        self.running = False
        for connection in self.connections.values():
            if not connection.closed:
                asyncio.create_task(connection.close())


class MarketDataCollector:
    """市场数据采集器主类"""
    
    def __init__(self, config: Dict[str, ExchangeConfig]):
        self.config = config
        self.exchanges: Dict[str, Exchange] = {}
        self.data_cache = DataCache()
        self.data_storage = DataStorage()
        self.websocket_manager = WebSocketManager()
        self.validator = DataValidator()
        self.running = False
        
        # 线程池用于批量处理
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 数据处理队列
        self.data_queue = asyncio.Queue(maxsize=1000)
        self.batch_data_buffer = []
        self.last_batch_time = time.time()
    
    async def initialize_exchanges(self) -> None:
        """初始化交易所连接"""
        for name, config in self.config.items():
            try:
                exchange_class = getattr(ccxt, name.lower())
                exchange = exchange_class({
                    'apiKey': config.api_key,
                    'secret': config.secret,
                    'sandbox': config.sandbox,
                    'enableRateLimit': True,
                    'rateLimit': config.rate_limit,
                    'timeout': config.timeout * 1000,
                })
                
                # 测试连接
                await exchange.load_markets()
                self.exchanges[name] = exchange
                logger.info(f"成功初始化 {name} 交易所连接")
                
            except Exception as e:
                logger.error(f"初始化 {name} 交易所失败: {e}")
    
    async def collect_kline_data(self, exchange_name: str, symbol: str, 
                                timeframe: str = '1m', limit: int = 500) -> List[MarketData]:
        """采集K线数据"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"交易所 {exchange_name} 未初始化")
            
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            market_data_list = []
            for ohlcv_data in ohlcv:
                market_data = MarketData(
                    exchange=exchange_name,
                    symbol=symbol,
                    timestamp=int(ohlcv_data[0]),
                    data_type='kline',
                    data={
                        'timestamp': int(ohlcv_data[0]),
                        'open': float(ohlcv_data[1]),
                        'high': float(ohlcv_data[2]),
                        'low': float(ohlcv_data[3]),
                        'close': float(ohlcv_data[4]),
                        'volume': float(ohlcv_data[5])
                    }
                )
                
                if self.validator.validate_kline_data(market_data.data):
                    market_data_list.append(market_data)
            
            logger.info(f"获取 {exchange_name} {symbol} K线数据: {len(market_data_list)} 条")
            return market_data_list
            
        except Exception as e:
            logger.error(f"采集K线数据失败 ({exchange_name} {symbol}): {e}")
            return []
    
    async def collect_orderbook_data(self, exchange_name: str, symbol: str, 
                                   limit: int = 20) -> Optional[MarketData]:
        """采集深度数据"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"交易所 {exchange_name} 未初始化")
            
            orderbook = await exchange.fetch_order_book(symbol, limit)
            
            market_data = MarketData(
                exchange=exchange_name,
                symbol=symbol,
                timestamp=int(time.time() * 1000),
                data_type='orderbook',
                data={
                    'bids': [[float(price), float(amount)] for price, amount in orderbook['bids']],
                    'asks': [[float(price), float(amount)] for price, amount in orderbook['asks']],
                    'timestamp': orderbook.get('timestamp', int(time.time() * 1000))
                }
            )
            
            if self.validator.validate_orderbook_data(market_data.data):
                logger.info(f"获取 {exchange_name} {symbol} 深度数据")
                return market_data
            else:
                logger.warning(f"深度数据验证失败: {exchange_name} {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"采集深度数据失败 ({exchange_name} {symbol}): {e}")
            return None
    
    async def collect_trade_data(self, exchange_name: str, symbol: str, 
                               since: Optional[int] = None, limit: int = 100) -> List[MarketData]:
        """采集成交数据"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"交易所 {exchange_name} 未初始化")
            
            trades = await exchange.fetch_trades(symbol, since, limit)
            
            market_data_list = []
            for trade in trades:
                market_data = MarketData(
                    exchange=exchange_name,
                    symbol=symbol,
                    timestamp=int(trade['timestamp']),
                    data_type='trade',
                    data={
                        'id': trade['id'],
                        'price': float(trade['price']),
                        'amount': float(trade['amount']),
                        'side': trade['side'],
                        'timestamp': int(trade['timestamp'])
                    }
                )
                
                if self.validator.validate_trade_data(market_data.data):
                    market_data_list.append(market_data)
            
            logger.info(f"获取 {exchange_name} {symbol} 成交数据: {len(market_data_list)} 条")
            return market_data_list
            
        except Exception as e:
            logger.error(f"采集成交数据失败 ({exchange_name} {symbol}): {e}")
            return []
    
    async def start_websocket_streams(self, exchange_configs: Dict[str, Dict]) -> None:
        """启动WebSocket实时数据流"""
        self.websocket_manager.start()
        
        tasks = []
        for exchange_name, config in exchange_configs.items():
            if exchange_name in self.exchanges:
                url = config.get('websocket_url')
                symbols = config.get('symbols', [])
                data_types = config.get('data_types', ['kline', 'trade'])
                
                if url and symbols:
                    task = asyncio.create_task(
                        self.websocket_manager.connect_exchange(
                            exchange_name, url, symbols, data_types
                        )
                    )
                    tasks.append(task)
        
        # 启动数据处理任务
        process_task = asyncio.create_task(self.process_data_stream())
        tasks.append(process_task)
        
        await asyncio.gather(*tasks)
    
    async def process_data_stream(self) -> None:
        """处理数据流"""
        while self.running:
            try:
                # 批量处理数据
                current_time = time.time()
                if (current_time - self.last_batch_time > 5 or 
                    len(self.batch_data_buffer) >= 100):
                    
                    if self.batch_data_buffer:
                        # 批量存储数据
                        success_count = self.data_storage.batch_store_data(self.batch_data_buffer)
                        logger.info(f"批量存储数据: {success_count}/{len(self.batch_data_buffer)} 条")
                        
                        # 添加到缓存
                        for data in self.batch_data_buffer:
                            cache_key = f"{data.exchange}_{data.symbol}_{data.data_type}"
                            self.data_cache.add_data(cache_key, data)
                        
                        self.batch_data_buffer.clear()
                        self.last_batch_time = current_time
                
                # 等待新数据
                try:
                    data = await asyncio.wait_for(self.data_queue.get(), timeout=1.0)
                    self.batch_data_buffer.append(data)
                except asyncio.TimeoutError:
                    continue
                    
            except Exception as e:
                logger.error(f"数据流处理失败: {e}")
                await asyncio.sleep(1)
    
    async def data_callback(self, market_data: MarketData) -> None:
        """数据回调处理"""
        try:
            # 验证数据
            if market_data.data_type == 'kline':
                if not self.validator.validate_kline_data(market_data.data):
                    return
            elif market_data.data_type == 'orderbook':
                if not self.validator.validate_orderbook_data(market_data.data):
                    return
            elif market_data.data_type == 'trade':
                if not self.validator.validate_trade_data(market_data.data):
                    return
            
            # 添加到处理队列
            await self.data_queue.put(market_data)
            
        except Exception as e:
            logger.error(f"数据回调处理失败: {e}")
    
    async def start_collection(self, symbols: List[str], 
                             data_types: List[str] = ['kline', 'orderbook', 'trade'],
                             websocket_enabled: bool = True) -> None:
        """开始数据采集"""
        self.running = True
        
        try:
            # 初始化交易所
            await self.initialize_exchanges()
            
            # 启动WebSocket流
            if websocket_enabled:
                exchange_configs = {
                    'binance': {
                        'websocket_url': 'wss://stream.binance.com:9443/ws/stream',
                        'symbols': symbols,
                        'data_types': data_types
                    }
                    # 可以添加更多交易所配置
                }
                
                # 注册数据回调
                for exchange_name in exchange_configs.keys():
                    if exchange_name in self.exchanges:
                        self.websocket_manager.subscribe(exchange_name, self.data_callback)
                
                await self.start_websocket_streams(exchange_configs)
            else:
                # 定时采集模式
                await self.start_periodic_collection(symbols, data_types)
                
        except Exception as e:
            logger.error(f"启动数据采集失败: {e}")
    
    async def start_periodic_collection(self, symbols: List[str], 
                                      data_types: List[str]) -> None:
        """定时采集模式"""
        while self.running:
            try:
                tasks = []
                
                for exchange_name, exchange in self.exchanges.items():
                    for symbol in symbols:
                        if 'kline' in data_types:
                            task = asyncio.create_task(
                                self.collect_kline_data(exchange_name, symbol)
                            )
                            tasks.append(task)
                        
                        if 'orderbook' in data_types:
                            task = asyncio.create_task(
                                self.collect_orderbook_data(exchange_name, symbol)
                            )
                            tasks.append(task)
                        
                        if 'trade' in data_types:
                            task = asyncio.create_task(
                                self.collect_trade_data(exchange_name, symbol)
                            )
                            tasks.append(task)
                
                # 等待所有任务完成
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 处理结果
                    for result in results:
                        if isinstance(result, list):
                            for market_data in result:
                                await self.data_queue.put(market_data)
                        elif isinstance(result, MarketData):
                            await self.data_queue.put(result)
                
                # 等待下次采集 (1分钟)
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"定时采集失败: {e}")
                await asyncio.sleep(10)
    
    def stop_collection(self) -> None:
        """停止数据采集"""
        self.running = False
        self.websocket_manager.stop()
        self.executor.shutdown(wait=True)
        
        # 关闭交易所连接
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'close'):
                asyncio.create_task(exchange.close())
        
        logger.info("数据采集已停止")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取采集统计信息"""
        stats = {
            'active_exchanges': len(self.exchanges),
            'cache_size': sum(len(cache) for cache in self.data_cache.cache.values()),
            'database_path': self.data_storage.db_path,
            'running': self.running
        }
        return stats


# 使用示例
async def main():
    """主函数示例"""
    # 配置交易所
    config = {
        'binance': ExchangeConfig(
            name='binance',
            api_key='your_api_key',
            secret='your_secret',
            sandbox=False,
            rate_limit=50,
            max_retries=3
        ),
        'okx': ExchangeConfig(
            name='okx',
            api_key='your_api_key',
            secret='your_secret',
            sandbox=False,
            rate_limit=100,
            max_retries=3
        )
    }
    
    # 创建采集器
    collector = MarketDataCollector(config)
    
    try:
        # 开始采集
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        await collector.start_collection(symbols, ['kline', 'orderbook', 'trade'])
        
        # 运行一段时间
        await asyncio.sleep(300)  # 运行5分钟
        
    except KeyboardInterrupt:
        logger.info("接收到停止信号")
    finally:
        collector.stop_collection()
        
        # 显示统计信息
        stats = collector.get_statistics()
        logger.info(f"采集统计: {stats}")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())