#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4交易所连接器
多交易所连接管理系统

功能特性：
1. 统一交易所API接口
2. 连接池管理和负载均衡
3. API限流和频率控制
4. 交易所状态监控
5. 自动故障转移
6. 数据源切换机制

支持交易所：币安(Binance)、OKX、火币(Huobi)、Gate.io
"""

import asyncio
import aiohttp
import time
import json
import logging
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import queue
import ssl
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    websocket = None

import ssl as ssl_module
import urllib.parse
import base64
import uuid


# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ExchangeStatus(Enum):
    """交易所状态枚举"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class ExchangeType(Enum):
    """交易所类型枚举"""
    BINANCE = "binance"
    OKX = "okx"
    HUOBI = "huobi"
    GATE = "gate"


@dataclass
class APIKey:
    """API密钥配置"""
    api_key: str
    secret_key: str
    passphrase: Optional[str] = None  # 部分交易所需要
    testnet: bool = False


@dataclass
class ExchangeConfig:
    """交易所配置"""
    name: ExchangeType
    base_url: str
    ws_url: str
    api_key: Optional[APIKey] = None
    rate_limit: int = 1200  # 每分钟请求限制
    weight_limit: int = 1200  # 权重限制
    timeout: int = 30
    max_retries: int = 3
    priority: int = 1  # 优先级，数字越小优先级越高


@dataclass
class HealthStatus:
    """健康状态信息"""
    exchange: ExchangeType
    status: ExchangeStatus
    latency: float
    last_check: datetime
    consecutive_failures: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 100.0


class RateLimiter:
    """API限流器"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """获取请求许可"""
        with self.lock:
            now = time.time()
            
            # 清理过期的请求记录
            while self.requests and now - self.requests[0] > self.time_window:
                self.requests.popleft()
            
            # 检查是否超过限制
            if len(self.requests) >= self.max_requests:
                return False
            
            # 记录请求
            self.requests.append(now)
            return True
    
    def wait_time(self) -> float:
        """获取需要等待的时间"""
        with self.lock:
            now = time.time()
            if not self.requests:
                return 0.0
            
            oldest_request = self.requests[0]
            wait_time = self.time_window - (now - oldest_request)
            return max(0.0, wait_time)


class ConnectionPool:
    """连接池管理"""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.active_connections = 0
        self.connection_queue = queue.Queue(maxsize=max_connections)
        self.lock = threading.Lock()
    
    async def get_connection(self, timeout: float = 30.0) -> Optional[aiohttp.ClientSession]:
        """获取连接"""
        try:
            async with aiohttp.ClientSession() as session:
                return session
        except Exception as e:
            logger.error(f"获取连接失败: {e}")
            return None
    
    def release_connection(self, connection):
        """释放连接"""
        pass  # aiohttp自动管理连接


class ExchangeConnector:
    """交易所连接器基类"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.health_status = HealthStatus(
            exchange=config.name,
            status=ExchangeStatus.OFFLINE,
            latency=0.0,
            last_check=datetime.now()
        )
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.connection_pool = ConnectionPool()
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[Any] = None
        self.subscribers = defaultdict(set)
        
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """建立连接"""
        try:
            connector = aiohttp.TCPConnector(
                limit=self.config.max_retries,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            self.health_status.status = ExchangeStatus.ONLINE
            logger.info(f"{self.config.name.value} 连接成功")
            
        except Exception as e:
            self.health_status.status = ExchangeStatus.OFFLINE
            logger.error(f"{self.config.name.value} 连接失败: {e}")
            raise
    
    async def disconnect(self):
        """断开连接"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.ws_connection:
            self.ws_connection.close()
            self.ws_connection = None
        
        self.health_status.status = ExchangeStatus.OFFLINE
        logger.info(f"{self.config.name.value} 连接已断开")
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            start_time = time.time()
            
            # 简单的API调用检查连接状态
            async with self.session.get(f"{self.config.base_url}/api/v3/ping") as response:
                latency = time.time() - start_time
                
                if response.status == 200:
                    self.health_status.latency = latency
                    self.health_status.last_check = datetime.now()
                    self.health_status.status = ExchangeStatus.ONLINE
                    return True
                else:
                    self.health_status.status = ExchangeStatus.DEGRADED
                    return False
                    
        except Exception as e:
            self.health_status.status = ExchangeStatus.OFFLINE
            logger.error(f"{self.config.name.value} 健康检查失败: {e}")
            return False
    
    def _sign_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> tuple[Dict[str, str], str]:
        """签名请求"""
        headers = {
            'X-MBX-APIKEY': self.config.api_key.api_key if self.config.api_key else '',
            'Content-Type': 'application/json'
        }
        
        # 构建查询字符串
        query_string = ''
        if params:
            query_string = urllib.parse.urlencode(params)
        
        if data:
            if query_string:
                query_string += '&'
            query_string += urllib.parse.urlencode(data)
        
        # 生成签名
        if query_string and self.config.api_key:
            signature = hmac.new(
                self.config.api_key.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            query_string += f'&signature={signature}'
        
        return headers, query_string
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """发起API请求"""
        if not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            logger.warning(f"{self.config.name.value} 触发限流，等待 {wait_time:.2f} 秒")
            await asyncio.sleep(wait_time)
        
        # 确保session存在
        if self.session is None:
            logger.error(f"{self.config.name.value} session未初始化")
            raise Exception("Session not initialized")
        
        try:
            headers, query_string = self._sign_request(method, endpoint, params, data)
            url = f"{self.config.base_url}{endpoint}"
            
            if query_string:
                if method.upper() == 'GET':
                    url += f"?{query_string}"
                else:
                    data = query_string
            
            start_time = time.time()
            
            # 使用async with 确保session正确管理
            async with self.session.request(method, url, headers=headers, data=data) as response:
                latency = time.time() - start_time
                self.health_status.latency = latency
                self.health_status.total_requests += 1
                
                if response.status == 200:
                    result = await response.json()
                    self.health_status.consecutive_failures = 0
                    return result
                else:
                    self.health_status.failed_requests += 1
                    self.health_status.consecutive_failures += 1
                    error_text = await response.text()
                    logger.error(f"{self.config.name.value} API请求失败: {response.status} - {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
                    
        except Exception as e:
            self.health_status.failed_requests += 1
            self.health_status.consecutive_failures += 1
            logger.error(f"{self.config.name.value} 请求异常: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Dict:
        """获取行情数据"""
        return await self._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """获取订单簿"""
        return await self._make_request('GET', '/api/v3/depth', {'symbol': symbol, 'limit': limit})
    
    async def get_trades(self, symbol: str, limit: int = 500) -> Dict:
        """获取交易记录"""
        return await self._make_request('GET', '/api/v3/trades', {'symbol': symbol, 'limit': limit})
    
    async def get_exchange_info(self) -> Dict:
        """获取交易所信息"""
        return await self._make_request('GET', '/api/v3/exchangeInfo')
    
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict:
        """下单"""
        data = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        if price:
            data['price'] = price
        
        return await self._make_request('POST', '/api/v3/order', data=data)
    
    async def get_account_info(self) -> Dict:
        """获取账户信息"""
        return await self._make_request('GET', '/api/v3/account')


class BinanceConnector(ExchangeConnector):
    """币安交易所连接器"""
    
    def __init__(self, config: ExchangeConfig):
        if config.base_url == "":
            if config.testnet:
                config.base_url = "https://testnet.binance.vision"
                config.ws_url = "wss://testnet.binance.vision/ws"
            else:
                config.base_url = "https://api.binance.com"
                config.ws_url = "wss://stream.binance.com:9443/ws"
        
        super().__init__(config)
    
    async def get_ticker(self, symbol: str) -> Dict:
        """获取行情数据 (Binance格式)"""
        try:
            # Binance直接使用标准格式，不需要转换
            logger.debug(f"🔍 Binance get_ticker收到参数: symbol='{symbol}'")
            
            # 防御性检查
            if not symbol:
                logger.error(f"❌ Binance ticker: 符号参数为空")
                return {'symbol': symbol, 'price': 0, 'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip()
            if not symbol_str:
                return {'symbol': symbol, 'price': 0, 'error': 'Symbol parameter is empty after conversion'}
            
            logger.debug(f"🚀 Binance最终API参数: symbol='{symbol_str}'")
            
            response = await self._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol_str})
            
            if isinstance(response, dict) and 'price' in response:
                return {
                    'symbol': symbol,
                    'price': float(response['price']),
                    'timestamp': int(time.time() * 1000)
                }
            
            return {'symbol': symbol, 'price': 0, 'error': 'No data'}
            
        except Exception as e:
            logger.error(f"Binance获取行情数据失败: {e}")
            return {'symbol': symbol, 'price': 0, 'error': str(e)}
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """获取订单簿数据 (Binance格式)"""
        try:
            if not symbol:
                return {'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip()
            if not symbol_str:
                return {'error': 'Symbol parameter is empty after conversion'}
            
            # Binance直接使用标准格式
            response = await self._make_request('GET', '/api/v3/depth', {
                'symbol': symbol_str, 
                'limit': limit
            })
            
            if isinstance(response, dict) and 'bids' in response:
                return {
                    'symbol': symbol,
                    'bids': [[float(bid[0]), float(bid[1])] for bid in response.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in response.get('asks', [])],
                    'timestamp': int(time.time() * 1000)
                }
            
            return {'error': 'No data'}
            
        except Exception as e:
            logger.error(f"Binance获取订单簿失败: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    async def get_trades(self, symbol: str, limit: int = 500) -> Dict:
        """获取交易记录 (Binance格式)"""
        try:
            if not symbol:
                return {'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip()
            if not symbol_str:
                return {'error': 'Symbol parameter is empty after conversion'}
            
            response = await self._make_request('GET', '/api/v3/trades', {
                'symbol': symbol_str, 
                'limit': limit
            })
            
            if isinstance(response, list):
                return {
                    'symbol': symbol,
                    'trades': [
                        {
                            'price': float(trade['price']),
                            'quantity': float(trade['qty']),
                            'timestamp': int(trade['time']),
                            'side': 'buy' if trade['isBuyerMaker'] else 'sell'
                        }
                        for trade in response
                    ]
                }
            
            return {'error': 'No data'}
            
        except Exception as e:
            logger.error(f"Binance获取交易记录失败: {e}")
            return {'symbol': symbol, 'error': str(e)}


class OKXConnector(ExchangeConnector):
    """OKX交易所连接器"""
    
    def __init__(self, config: ExchangeConfig):
        # 如果base_url为空，则根据testnet设置默认URL
        # 优先使用.env文件中的配置
        if config.base_url == "":
            if config.testnet:
                config.base_url = "https://www.oucnyi.com"  # 纸张交易使用中国域名
                config.ws_url = "wss://wspap.oucnyi.com:8443/ws/v5/public"
            else:
                config.base_url = "https://www.oucnyi.com"  # 实盘交易也使用中国域名
                config.ws_url = "wss://ws.oucnyi.com:8443/ws/v5/public"
        
        super().__init__(config)
    
    def _sign_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None):
        """OKX签名请求"""
        headers = {
            'Content-Type': 'application/json',
            'OK-ACCESS-KEY': self.config.api_key.api_key if self.config.api_key else '',
            'OK-ACCESS-SIGN': '',
            'OK-ACCESS-TIMESTAMP': '',
            'OK-ACCESS-PASSPHRASE': self.config.api_key.passphrase if self.config.api_key and self.config.api_key.passphrase else ''
        }
        
        query_string = ""
        
        if not self.config.api_key:
            return headers, query_string
        
        # 构建签名字符串
        timestamp = datetime.utcnow().isoformat() + 'Z'
        sign_string = timestamp + method.upper() + endpoint
        
        if params:
            sign_string += '?' + urllib.parse.urlencode(params)
        if data:
            sign_string += json.dumps(data)
        
        # 生成签名
        signature = base64.b64encode(hmac.new(
            self.config.api_key.secret_key.encode('utf-8'),
            sign_string.encode('utf-8'),
            hashlib.sha256
        ).digest()).decode()
        
        headers['OK-ACCESS-SIGN'] = signature
        headers['OK-ACCESS-TIMESTAMP'] = timestamp
        
        return headers, query_string
    
    async def get_ticker(self, symbol: str) -> Dict:
        """获取行情数据"""
        try:
            # 🔍 调试输出：检查输入参数
            logger.debug(f"🔍 get_ticker收到参数: symbol='{symbol}', type={type(symbol)}")
            
            # 检查symbol是否为空或无效
            if not symbol:
                logger.error(f"❌ 符号参数为空: symbol={repr(symbol)}")
                return {'symbol': symbol, 'price': 0, 'error': 'Symbol parameter is empty'}
            
            # 确保symbol是字符串类型
            symbol_str = str(symbol).strip()
            if not symbol_str:
                logger.error(f"❌ 符号参数转换后为空: original={repr(symbol)}")
                return {'symbol': symbol, 'price': 0, 'error': 'Symbol parameter is empty after conversion'}
            
            # 转换符号格式为OKX格式（如：BTCUSDT -> BTC-USDT）
            # 如果已经是OKX格式（包含连字符但不重复连字符），直接使用
            if '-' in symbol_str and '--' not in symbol_str:
                okx_symbol = symbol_str
                logger.debug(f"✅ 已是OKX格式: {symbol_str} -> {okx_symbol}")
            # 如果符合标准格式（字母 + USDT）
            elif len(symbol_str) >= 6 and symbol_str.endswith('USDT'):
                base = symbol_str[:-4]
                okx_symbol = f"{base}-USDT"
                logger.debug(f"✅ 标准格式转换: {symbol_str} -> {okx_symbol}")
            else:
                # 其他情况使用替换
                okx_symbol = symbol_str.replace('/', '-')
                logger.debug(f"⚠️ 使用替换: {symbol_str} -> {okx_symbol}")
            
            # 再次检查转换后的参数
            if not okx_symbol or not okx_symbol.strip():
                logger.error(f"❌ 转换后符号为空: original='{symbol_str}', converted='{okx_symbol}'")
                return {'symbol': symbol, 'price': 0, 'error': 'Converted symbol is empty'}
            
            logger.debug(f"🚀 最终API参数: instId='{okx_symbol}'")
            
            # 确保参数不为空字典
            if not okx_symbol:
                logger.error(f"❌ 最终OKX符号为空: {okx_symbol}")
                return {'symbol': symbol, 'price': 0, 'error': 'Final OKX symbol is empty'}
            
            response = await self._make_request('GET', '/api/v5/market/ticker', {'instId': okx_symbol})
            
            # 检查响应格式并标准化
            if isinstance(response, dict):
                if 'data' in response and response['data']:
                    data = response['data'][0]  # OKX返回数组格式
                    return {
                        'symbol': symbol,
                        'price': float(data.get('last', data.get('price', 0))),
                        'bid': float(data.get('bidPx', 0)),
                        'ask': float(data.get('askPx', 0)),
                        'volume': float(data.get('vol24h', 0)),
                        'timestamp': int(time.time() * 1000)
                    }
            
            return {'symbol': symbol, 'price': 0, 'error': 'No data'}
            
        except Exception as e:
            logger.error(f"OKX获取行情数据失败: {e}")
            return {'symbol': symbol, 'price': 0, 'error': str(e)}

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """获取订单簿数据 (OKX格式)"""
        try:
            # 防御性检查
            if not symbol:
                logger.error(f"❌ get_orderbook: 符号参数为空: {repr(symbol)}")
                return {'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip()
            if not symbol_str:
                return {'error': 'Symbol parameter is empty after conversion'}
            
            # 转换符号格式
            if '-' in symbol_str and '--' not in symbol_str:
                okx_symbol = symbol_str  # 已经是OKX格式
            elif len(symbol_str) >= 6 and symbol_str.endswith('USDT'):
                base = symbol_str[:-4]
                okx_symbol = f"{base}-USDT"
            else:
                okx_symbol = symbol_str.replace('/', '-')
            
            if not okx_symbol:
                logger.error(f"❌ get_orderbook: 转换后符号为空")
                return {'error': 'Converted symbol is empty'}
            
            response = await self._make_request('GET', '/api/v5/market/books', {'instId': okx_symbol, 'sz': str(limit)})
            
            if response.get('code') != '0':
                return {'error': response.get('msg', '获取订单簿失败')}
            
            data = response.get('data', [])
            if not data:
                return {'error': '未获取到订单簿数据'}
            
            orderbook_data = data[0]
            return {
                'symbol': symbol,
                'bids': [[float(bid[0]), float(bid[1])] for bid in orderbook_data.get('bids', [])],
                'asks': [[float(ask[0]), float(ask[1])] for ask in orderbook_data.get('asks', [])],
                'timestamp': int(orderbook_data.get('ts', 0))
            }
            
        except Exception as e:
            logger.error(f"OKX获取订单簿失败: {e}")
            return {'symbol': symbol, 'error': str(e)}

    async def get_trades(self, symbol: str, limit: int = 500) -> Dict:
        """获取交易记录 (OKX格式)"""
        try:
            # 防御性检查
            if not symbol:
                logger.error(f"❌ get_trades: 符号参数为空: {repr(symbol)}")
                return {'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip()
            if not symbol_str:
                return {'error': 'Symbol parameter is empty after conversion'}
            
            # 转换符号格式
            if '-' in symbol_str and '--' not in symbol_str:
                okx_symbol = symbol_str  # 已经是OKX格式
            elif len(symbol_str) >= 6 and symbol_str.endswith('USDT'):
                base = symbol_str[:-4]
                okx_symbol = f"{base}-USDT"
            else:
                okx_symbol = symbol_str.replace('/', '-')
            
            if not okx_symbol:
                logger.error(f"❌ get_trades: 转换后符号为空")
                return {'error': 'Converted symbol is empty'}
            
            response = await self._make_request('GET', '/api/v5/market/trades', {'instId': okx_symbol, 'limit': str(limit)})
            
            if response.get('code') != '0':
                return {'error': response.get('msg', '获取交易记录失败')}
            
            data = response.get('data', [])
            return {
                'symbol': symbol,
                'trades': [
                    {
                        'price': float(trade[1]),
                        'quantity': float(trade[2]),
                        'timestamp': int(trade[0]),
                        'side': trade[4] if len(trade) > 4 else 'unknown'
                    }
                    for trade in data
                ]
            }
            
        except Exception as e:
            logger.error(f"OKX获取交易记录失败: {e}")
            return {'symbol': symbol, 'error': str(e)}

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict:
        """下单 (OKX格式)"""
        try:
            # 防御性检查
            if not symbol:
                logger.error(f"❌ place_order: 符号参数为空: {repr(symbol)}")
                return {'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip()
            if not symbol_str:
                return {'error': 'Symbol parameter is empty after conversion'}
            
            # 转换符号格式
            if '-' in symbol_str and '--' not in symbol_str:
                okx_symbol = symbol_str  # 已经是OKX格式
            elif len(symbol_str) >= 6 and symbol_str.endswith('USDT'):
                base = symbol_str[:-4]
                okx_symbol = f"{base}-USDT"
            else:
                okx_symbol = symbol_str.replace('/', '-')
            
            if not okx_symbol:
                logger.error(f"❌ place_order: 转换后符号为空")
                return {'error': 'Converted symbol is empty'}
            
            order_data = {
                'instId': okx_symbol,
                'tdMode': 'cash',  # 现金交易模式
                'side': side.lower(),
                'ordType': 'limit' if price else 'market'
            }
            
            # 设置数量和价格
            order_data['sz'] = str(quantity)
            if price:
                order_data['px'] = str(price)
            
            response = await self._make_request('POST', '/api/v5/trade/order', order_data)
            
            if response.get('code') != '0':
                error_msg = response.get('msg', '下单失败')
                logger.error(f"OKX下单失败: {error_msg}")
                return {'error': error_msg}
            
            data = response.get('data', [])
            if data:
                result = data[0]
                return {
                    'order_id': result.get('ordId'),
                    'symbol': symbol,
                    'status': 'filled' if result.get('state') == 'filled' else 'pending'
                }
            else:
                return {'error': '下单响应数据为空'}
                
        except Exception as e:
            logger.error(f"OKX下单失败: {e}")
            return {'symbol': symbol, 'error': str(e)}


class HuobiConnector(ExchangeConnector):
    """火币交易所连接器"""
    
    def __init__(self, config: ExchangeConfig):
        if config.base_url == "":
            config.base_url = "https://api.huobi.pro"
            config.ws_url = "wss://api.huobi.pro/ws"
        
        super().__init__(config)
    
    async def get_ticker(self, symbol: str) -> Dict:
        """获取行情数据 (火币格式)"""
        try:
            logger.debug(f"🔍 Huobi get_ticker收到参数: symbol='{symbol}'")
            
            # 防御性检查
            if not symbol:
                logger.error(f"❌ Huobi ticker: 符号参数为空")
                return {'symbol': symbol, 'price': 0, 'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip().upper()
            if not symbol_str:
                return {'symbol': symbol, 'price': 0, 'error': 'Symbol parameter is empty after conversion'}
            
            # 火币使用小写符号格式
            huobi_symbol = symbol_str.lower()
            
            logger.debug(f"🚀 Huobi最终API参数: symbol='{huobi_symbol}'")
            
            response = await self._make_request('GET', '/market/detail/merged', {'symbol': huobi_symbol})
            
            if isinstance(response, dict) and 'tick' in response:
                tick_data = response['tick']
                return {
                    'symbol': symbol,
                    'price': float(tick_data.get('close', 0)),
                    'bid': float(tick_data.get('bid', [0, 0])[0]),
                    'ask': float(tick_data.get('ask', [0, 0])[0]),
                    'volume': float(tick_data.get('vol', 0)),
                    'timestamp': int(time.time() * 1000)
                }
            
            return {'symbol': symbol, 'price': 0, 'error': 'No data'}
            
        except Exception as e:
            logger.error(f"火币获取行情数据失败: {e}")
            return {'symbol': symbol, 'price': 0, 'error': str(e)}
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """获取订单簿数据 (火币格式)"""
        try:
            if not symbol:
                return {'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip().lower()
            if not symbol_str:
                return {'error': 'Symbol parameter is empty after conversion'}
            
            response = await self._make_request('GET', '/market/depth', {
                'symbol': symbol_str, 
                'type': 'step1',
                'depth': min(limit, 150)  # 火币限制最多150
            })
            
            if isinstance(response, dict) and 'tick' in response:
                tick_data = response['tick']
                return {
                    'symbol': symbol,
                    'bids': [[float(bid[0]), float(bid[1])] for bid in tick_data.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in tick_data.get('asks', [])],
                    'timestamp': int(time.time() * 1000)
                }
            
            return {'error': 'No data'}
            
        except Exception as e:
            logger.error(f"火币获取订单簿失败: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    async def get_trades(self, symbol: str, limit: int = 500) -> Dict:
        """获取交易记录 (火币格式)"""
        try:
            if not symbol:
                return {'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip().lower()
            if not symbol_str:
                return {'error': 'Symbol parameter is empty after conversion'}
            
            response = await self._make_request('GET', '/market/trade', {
                'symbol': symbol_str
            })
            
            if isinstance(response, dict) and 'tick' in response:
                trade_data = response['tick']
                return {
                    'symbol': symbol,
                    'trades': [
                        {
                            'price': float(trade['price']),
                            'quantity': float(trade['amount']),
                            'timestamp': int(trade['id']),
                            'side': trade['direction']
                        }
                        for trade in trade_data.get('data', [])[:limit]
                    ]
                }
            
            return {'error': 'No data'}
            
        except Exception as e:
            logger.error(f"火币获取交易记录失败: {e}")
            return {'symbol': symbol, 'error': str(e)}


class GateConnector(ExchangeConnector):
    """Gate.io交易所连接器"""
    
    def __init__(self, config: ExchangeConfig):
        if config.base_url == "":
            config.base_url = "https://api.gateio.ws"
            config.ws_url = "wss://api.gateio.ws/ws/v4/"
        
        super().__init__(config)
    
    async def get_ticker(self, symbol: str) -> Dict:
        """获取行情数据 (Gate.io格式)"""
        try:
            logger.debug(f"🔍 Gate.io get_ticker收到参数: symbol='{symbol}'")
            
            # 防御性检查
            if not symbol:
                logger.error(f"❌ Gate.io ticker: 符号参数为空")
                return {'symbol': symbol, 'price': 0, 'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip().upper()
            if not symbol_str:
                return {'symbol': symbol, 'price': 0, 'error': 'Symbol parameter is empty after conversion'}
            
            # Gate.io使用下划线格式
            gate_symbol = f"{symbol_str[:-4]}_{symbol_str[-4:]}" if symbol_str.endswith('USDT') else symbol_str
            
            logger.debug(f"🚀 Gate.io最终API参数: currency_pair='{gate_symbol}'")
            
            response = await self._make_request('GET', '/api/v4/spot/tickers', {'currency_pair': gate_symbol})
            
            if isinstance(response, list) and response:
                ticker_data = response[0]
                return {
                    'symbol': symbol,
                    'price': float(ticker_data.get('highest_bid', 0)),
                    'bid': float(ticker_data.get('highest_bid', 0)),
                    'ask': float(ticker_data.get('lowest_ask', 0)),
                    'volume': float(ticker_data.get('base_volume', 0)),
                    'timestamp': int(time.time() * 1000)
                }
            
            return {'symbol': symbol, 'price': 0, 'error': 'No data'}
            
        except Exception as e:
            logger.error(f"Gate.io获取行情数据失败: {e}")
            return {'symbol': symbol, 'price': 0, 'error': str(e)}
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """获取订单簿数据 (Gate.io格式)"""
        try:
            if not symbol:
                return {'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip().upper()
            if not symbol_str:
                return {'error': 'Symbol parameter is empty after conversion'}
            
            gate_symbol = f"{symbol_str[:-4]}_{symbol_str[-4:]}" if symbol_str.endswith('USDT') else symbol_str
            
            response = await self._make_request('GET', '/api/v4/spot/order_book', {
                'currency_pair': gate_symbol, 
                'limit': min(limit, 100)  # Gate.io限制最多100
            })
            
            if isinstance(response, dict) and 'bids' in response:
                return {
                    'symbol': symbol,
                    'bids': [[float(bid[0]), float(bid[1])] for bid in response.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in response.get('asks', [])],
                    'timestamp': int(time.time() * 1000)
                }
            
            return {'error': 'No data'}
            
        except Exception as e:
            logger.error(f"Gate.io获取订单簿失败: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    async def get_trades(self, symbol: str, limit: int = 500) -> Dict:
        """获取交易记录 (Gate.io格式)"""
        try:
            if not symbol:
                return {'error': 'Symbol parameter is empty'}
            
            symbol_str = str(symbol).strip().upper()
            if not symbol_str:
                return {'error': 'Symbol parameter is empty after conversion'}
            
            gate_symbol = f"{symbol_str[:-4]}_{symbol_str[-4:]}" if symbol_str.endswith('USDT') else symbol_str
            
            response = await self._make_request('GET', '/api/v4/spot/trades', {
                'currency_pair': gate_symbol,
                'limit': min(limit, 1000)  # Gate.io限制最多1000
            })
            
            if isinstance(response, list):
                return {
                    'symbol': symbol,
                    'trades': [
                        {
                            'price': float(trade['price']),
                            'quantity': float(trade['amount']),
                            'timestamp': int(trade['time_us']) // 1000,
                            'side': trade['side']
                        }
                        for trade in response[:limit]
                    ]
                }
            
            return {'error': 'No data'}
            
        except Exception as e:
            logger.error(f"Gate.io获取交易记录失败: {e}")
            return {'symbol': symbol, 'error': str(e)}


class ExchangeManager:
    """交易所管理器"""
    
    def __init__(self):
        self.connectors: Dict[ExchangeType, ExchangeConnector] = {}
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
    def add_exchange(self, config: ExchangeConfig) -> ExchangeConnector:
        """添加交易所配置"""
        connector = self._create_connector(config)
        self.connectors[config.name] = connector
        logger.info(f"添加交易所: {config.name.value}")
        return connector
    
    def _create_connector(self, config: ExchangeConfig) -> ExchangeConnector:
        """创建交易所连接器"""
        connector_map = {
            ExchangeType.BINANCE: BinanceConnector,
            ExchangeType.OKX: OKXConnector,
            ExchangeType.HUOBI: HuobiConnector,
            ExchangeType.GATE: GateConnector,
        }
        
        connector_class = connector_map.get(config.name)
        if not connector_class:
            raise ValueError(f"不支持的交易所类型: {config.name}")
        
        return connector_class(config)
    
    async def connect_all(self):
        """连接所有交易所"""
        tasks = []
        for connector in self.connectors.values():
            tasks.append(connector.connect())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 启动健康监控
        self.is_running = True
        self.health_monitor_task = asyncio.create_task(self._health_monitor())
    
    async def disconnect_all(self):
        """断开所有连接"""
        self.is_running = False
        
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        tasks = []
        for connector in self.connectors.values():
            tasks.append(connector.disconnect())
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _health_monitor(self):
        """健康监控任务"""
        while self.is_running:
            try:
                for connector in self.connectors.values():
                    await connector.health_check()
                
                # 检查是否需要故障转移
                await self._check_failover()
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康监控异常: {e}")
                await asyncio.sleep(5)
    
    async def _check_failover(self):
        """检查故障转移"""
        # 找出可用的交易所
        available_exchanges = [
            connector for connector in self.connectors.values()
            if connector.health_status.status == ExchangeStatus.ONLINE
        ]
        
        if not available_exchanges:
            logger.warning("没有可用的交易所连接")
            return
        
        # 按优先级和延迟排序
        available_exchanges.sort(key=lambda x: (x.config.priority, x.health_status.latency))
        
        # 设置主交易所（优先级最高的）
        primary_exchange = available_exchanges[0]
        logger.info(f"主交易所设置为: {primary_exchange.config.name.value}")
    
    async def get_best_exchange(self, symbol: str) -> Optional[ExchangeConnector]:
        """获取最佳交易所连接"""
        available_exchanges = [
            connector for connector in self.connectors.values()
            if connector.health_status.status == ExchangeStatus.ONLINE
        ]
        
        if not available_exchanges:
            return None
        
        # 按延迟排序，选择最快的
        available_exchanges.sort(key=lambda x: x.health_status.latency)
        return available_exchanges[0]
    
    async def get_ticker_from_all(self, symbol: str) -> Dict[ExchangeType, Dict]:
        """从所有交易所获取行情数据"""
        results = {}
        tasks = []
        
        for exchange_type, connector in self.connectors.items():
            if connector.health_status.status == ExchangeStatus.ONLINE:
                task = asyncio.create_task(self._safe_get_ticker(connector, symbol))
                tasks.append((exchange_type, task))
        
        for exchange_type, task in tasks:
            try:
                result = await task
                results[exchange_type] = result
            except Exception as e:
                logger.error(f"从 {exchange_type.value} 获取 {symbol} 行情失败: {e}")
                results[exchange_type] = None
        
        return results
    
    async def _safe_get_ticker(self, connector: ExchangeConnector, symbol: str) -> Dict:
        """安全获取行情数据"""
        return await connector.get_ticker(symbol)
    
    async def aggregate_orderbook(self, symbol: str) -> Dict:
        """聚合订单簿数据"""
        results = await self.get_ticker_from_all(symbol)
        
        # 这里可以实现更复杂的聚合逻辑
        # 例如：价格加权平均、成交量加权等
        aggregated_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'exchanges': results,
            'best_bid': None,
            'best_ask': None,
            'spread': None
        }
        
        # 简单的最佳买卖价计算
        bids = []
        asks = []
        
        for exchange_type, data in results.items():
            if data and 'bid' in data:
                bids.append(data['bid'])
            if data and 'ask' in data:
                asks.append(data['ask'])
        
        if bids:
            aggregated_data['best_bid'] = max(bids)
        if asks:
            aggregated_data['best_ask'] = min(asks)
        
        if aggregated_data['best_bid'] and aggregated_data['best_ask']:
            aggregated_data['spread'] = aggregated_data['best_ask'] - aggregated_data['best_bid']
        
        return aggregated_data
    
    def get_health_status(self) -> Dict[ExchangeType, HealthStatus]:
        """获取所有交易所健康状态"""
        return {
            exchange_type: connector.health_status
            for exchange_type, connector in self.connectors.items()
        }
    
    async def sync_trading_pairs(self) -> Dict[str, List[ExchangeType]]:
        """同步交易对信息"""
        trading_pairs = defaultdict(list)
        
        for exchange_type, connector in self.connectors.items():
            if connector.health_status.status == ExchangeStatus.ONLINE:
                try:
                    exchange_info = await connector.get_exchange_info()
                    symbols = exchange_info.get('symbols', [])
                    
                    for symbol_info in symbols:
                        symbol = symbol_info.get('symbol')
                        if symbol:
                            trading_pairs[symbol].append(exchange_type)
                            
                except Exception as e:
                    logger.error(f"同步 {exchange_type.value} 交易对失败: {e}")
        
        return dict(trading_pairs)


class DataAggregator:
    """数据聚合器"""
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.market_data_cache = {}
        self.data_callbacks = defaultdict(list)
    
    async def start_market_data_stream(self, symbols: List[str]):
        """启动市场数据流"""
        for symbol in symbols:
            asyncio.create_task(self._market_data_worker(symbol))
    
    async def _market_data_worker(self, symbol: str):
        """市场数据工作线程"""
        while True:
            try:
                # 获取聚合数据
                aggregated_data = await self.exchange_manager.aggregate_orderbook(symbol)
                self.market_data_cache[symbol] = aggregated_data
                
                # 调用回调函数
                for callback in self.data_callbacks[symbol]:
                    try:
                        await callback(aggregated_data)
                    except Exception as e:
                        logger.error(f"数据回调异常: {e}")
                
                await asyncio.sleep(1)  # 1秒更新一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"市场数据工作线程异常: {e}")
                await asyncio.sleep(5)
    
    def subscribe_market_data(self, symbol: str, callback: Callable):
        """订阅市场数据"""
        self.data_callbacks[symbol].append(callback)
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """获取市场数据"""
        return self.market_data_cache.get(symbol)


# 使用示例和测试代码
async def main():
    """主函数示例"""
    # 创建交易所管理器
    manager = ExchangeManager()
    
    # 配置交易所
    binance_config = ExchangeConfig(
        name=ExchangeType.BINANCE,
        base_url="",
        ws_url="",
        api_key=APIKey(
            api_key="your_binance_api_key",
            secret_key="your_binance_secret_key"
        ),
        priority=1
    )
    
    okx_config = ExchangeConfig(
        name=ExchangeType.OKX,
        base_url="",
        ws_url="",
        api_key=APIKey(
            api_key="your_okx_api_key",
            secret_key="your_okx_secret_key",
            passphrase="your_okx_passphrase"
        ),
        priority=2
    )
    
    # 添加交易所
    manager.add_exchange(binance_config)
    manager.add_exchange(okx_config)
    
    # 连接所有交易所
    await manager.connect_all()
    
    try:
        # 获取最佳交易所
        best_exchange = await manager.get_best_exchange("BTCUSDT")
        if best_exchange:
            print(f"最佳交易所: {best_exchange.config.name.value}")
            
            # 获取行情数据
            ticker = await best_exchange.get_ticker("BTCUSDT")
            print(f"行情数据: {ticker}")
        
        # 从所有交易所获取数据
        all_data = await manager.get_ticker_from_all("BTCUSDT")
        print(f"所有交易所数据: {all_data}")
        
        # 聚合订单簿
        orderbook = await manager.aggregate_orderbook("BTCUSDT")
        print(f"聚合订单簿: {orderbook}")
        
        # 获取健康状态
        health_status = manager.get_health_status()
        for exchange, status in health_status.items():
            print(f"{exchange.value}: {status.status.value}, 延迟: {status.latency:.2f}ms")
        
        # 同步交易对
        trading_pairs = await manager.sync_trading_pairs()
        print(f"交易对数量: {len(trading_pairs)}")
        
        # 创建数据聚合器
        aggregator = DataAggregator(manager)
        
        # 订阅市场数据
        def market_data_callback(data):
            print(f"收到市场数据: {data['symbol']}, 最佳买价: {data.get('best_bid')}")
        
        aggregator.subscribe_market_data("BTCUSDT", market_data_callback)
        
        # 启动市场数据流
        await aggregator.start_market_data_stream(["BTCUSDT", "ETHUSDT"])
        
        # 保持运行
        await asyncio.sleep(10)
        
    finally:
        # 断开所有连接
        await manager.disconnect_all()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())