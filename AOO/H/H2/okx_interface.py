"""
OKX交易所接口
与OKX交易所API直接交互的低层接口，提供完整的交易功能
支持REST API和WebSocket，包含错误处理和重试机制
"""

import os
import logging
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import json
import asyncio
import aiohttp
from urllib.parse import urlencode
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

# 导入配置管理器
try:
    from config_manager import ConfigManager, get_global_config_manager
except ImportError:
    # 备用导入路径
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_manager import ConfigManager, get_global_config_manager


class OKXOrderType(Enum):
    """OKX订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"
    FOK = "fok"
    IOC = "ioc"


class OKXOrderSide(Enum):
    """OKX订单方向"""
    BUY = "buy"
    SELL = "sell"


class OKXPositionSide(Enum):
    """OKX持仓方向"""
    LONG = "long"
    SHORT = "short"
    NET = "net"


@dataclass
class OKXOrder:
    """OKX订单数据类"""
    order_id: str
    client_oid: str
    symbol: str
    side: OKXOrderSide
    order_type: OKXOrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    state: str = "live"
    filled_quantity: Decimal = Decimal('0')
    average_price: Decimal = Decimal('0')
    fee: Decimal = Decimal('0')
    fee_currency: str = ""
    create_time: str = ""
    update_time: str = ""
    leverage: str = "1"


@dataclass
class OKXPosition:
    """OKX持仓数据类"""
    symbol: str
    position_side: OKXPositionSide
    quantity: Decimal
    available_quantity: Decimal
    average_price: Decimal
    leverage: str
    unrealized_pnl: Decimal
    margin: Decimal
    margin_ratio: Decimal
    liquidation_price: Decimal


@dataclass
class OKXBalance:
    """OKX余额数据类"""
    currency: str
    total_balance: Decimal
    available_balance: Decimal
    frozen_balance: Decimal


class OKXInterface:
    """OKX交易所接口 - 完整版实现"""
    
    def __init__(self, config: Dict = None, sandbox: bool = False):
        self.config = config or {}
        self.sandbox = sandbox
        self.logger = self._setup_logging()
        
        # API配置
        self.api_key = self.config.get('api_key', os.getenv('OKX_API_KEY', ''))
        self.secret_key = self.config.get('secret_key', os.getenv('OKX_API_SECRET', ''))
        self.passphrase = self.config.get('passphrase', os.getenv('OKX_PASSPHRASE', ''))
        
        # 基础URL
        if sandbox:
            self.base_url = self.config.get('base_url', 'https://www.okx.com') + '/api/v5'
        else:
            self.base_url = self.config.get('base_url', 'https://www.okx.com') + '/api/v5'
        
        # 交易对配置
        self.trading_pairs = self.config.get('trading_pairs', ['BTC-USDT', 'ETH-USDT', 'ADA-USDT'])
        
        # 请求配置
        self.timeout = self.config.get('timeout', 30)
        self.retry_count = self.config.get('retry_count', 3)
        self.retry_delay = self.config.get('retry_delay', 1)
        
        # 会话管理
        self._session = None
        self._session_lock = threading.RLock()
        
        # 请求限制
        self.rate_limits = self.config.get('rate_limits', {
            'requests_per_second': 10,
            'order_interval_ms': 100
        })
        self._last_request_time = 0
        self._request_times = deque(maxlen=100)
        self._rate_limit_lock = threading.RLock()
        
        # 缓存
        self._ticker_cache = {}
        self._balance_cache = {}
        self._positions_cache = {}
        self._cache_ttl = 5  # 5秒缓存
        
        # 性能统计
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'start_time': time.time()
        }
        
        # 订单簿
        self._orderbooks = {}
        
        self.logger.info(f"OKX接口初始化完成 - 模式: {'沙盒' if sandbox else '实盘'}")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('OKXInterface')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建会话"""
        with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
            return self._session
    
    async def _ensure_rate_limit(self):
        """确保遵守速率限制"""
        with self._rate_limit_lock:
            current_time = time.time()
            
            # 检查每秒请求限制
            one_second_ago = current_time - 1
            recent_requests = [t for t in self._request_times if t > one_second_ago]
            
            if len(recent_requests) >= self.rate_limits['requests_per_second']:
                sleep_time = 1.0 - (current_time - recent_requests[0])
                if sleep_time > 0:
                    self._stats['rate_limit_hits'] += 1
                    await asyncio.sleep(sleep_time)
            
            # 更新请求时间
            self._request_times.append(current_time)
            self._last_request_time = current_time
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """生成API签名"""
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, 'utf-8'),
            bytes(message, 'utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    def _get_timestamp(self) -> str:
        """获取ISO格式时间戳"""
        return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                           body: Dict = None, signed: bool = False) -> Dict[str, Any]:
        """发送API请求"""
        # 确保速率限制
        await self._ensure_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AOO-Trading-System/1.0'
        }
        
        # 添加签名头
        if signed:
            timestamp = self._get_timestamp()
            signature = self._generate_signature(timestamp, method, endpoint, json.dumps(body) if body else "")
            
            headers.update({
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': signature,
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self.passphrase
            })
        
        # 准备请求参数
        request_params = None
        if method.upper() == 'GET' and params:
            request_params = params
        elif body:
            request_data = json.dumps(body)
        else:
            request_data = None
        
        session = await self._get_session()
        
        # 重试逻辑
        for attempt in range(self.retry_count):
            try:
                self._stats['total_requests'] += 1
                
                async with session.request(
                    method=method,
                    url=url,
                    params=request_params,
                    data=request_data,
                    headers=headers
                ) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        result = json.loads(response_text)
                        
                        if result.get('code') == '0':
                            self._stats['successful_requests'] += 1
                            return result.get('data', [])
                        else:
                            error_msg = result.get('msg', 'Unknown error')
                            self.logger.error(f"API错误: {error_msg}")
                            raise Exception(f"OKX API Error: {error_msg}")
                    else:
                        self.logger.error(f"HTTP错误 {response.status}: {response_text}")
                        if attempt < self.retry_count - 1:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        raise Exception(f"HTTP {response.status}: {response_text}")
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"请求超时 (尝试 {attempt + 1}/{self.retry_count})")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise Exception("请求超时")
            except Exception as e:
                self.logger.error(f"请求异常 (尝试 {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise
        
        self._stats['failed_requests'] += 1
        raise Exception("所有重试尝试都失败了")
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """获取行情数据"""
        # 检查缓存
        cache_key = f"ticker_{symbol}"
        if cache_key in self._ticker_cache:
            cached_data, timestamp = self._ticker_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
        
        try:
            endpoint = f"/market/ticker"
            params = {'instId': symbol}
            
            data = await self._make_request('GET', endpoint, params=params)
            
            if data and len(data) > 0:
                ticker_data = data[0]
                result = {
                    'symbol': symbol,
                    'last': Decimal(ticker_data.get('last', '0')),
                    'bid': Decimal(ticker_data.get('bidPx', '0')),
                    'ask': Decimal(ticker_data.get('askPx', '0')),
                    'high': Decimal(ticker_data.get('high24h', '0')),
                    'low': Decimal(ticker_data.get('low24h', '0')),
                    'volume': Decimal(ticker_data.get('vol24h', '0')),
                    'timestamp': time.time()
                }
                
                # 更新缓存
                self._ticker_cache[cache_key] = (result, time.time())
                return result
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"获取行情数据失败 {symbol}: {e}")
            return {}
    
    async def create_market_order(self, symbol: str, side: str, quantity: Decimal, 
                                 client_oid: str = None) -> Dict[str, Any]:
        """创建市价订单"""
        try:
            endpoint = "/trade/order"
            
            order_data = {
                'instId': symbol,
                'tdMode': 'cash',  # 现货交易
                'side': 'buy' if side.lower() == 'buy' else 'sell',
                'ordType': 'market',
                'sz': str(quantity)
            }
            
            if client_oid:
                order_data['clOrdId'] = client_oid
            
            result = await self._make_request('POST', endpoint, body=order_data, signed=True)
            
            if result and len(result) > 0:
                order_info = result[0]
                return {
                    'order_id': order_info.get('ordId'),
                    'client_oid': order_info.get('clOrdId'),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'status': 'created',
                    'timestamp': time.time()
                }
            else:
                return {'error': '订单创建失败'}
                
        except Exception as e:
            self.logger.error(f"创建市价订单失败 {symbol} {side} {quantity}: {e}")
            return {'error': str(e)}
    
    async def create_limit_order(self, symbol: str, side: str, quantity: Decimal, 
                                price: Decimal, client_oid: str = None) -> Dict[str, Any]:
        """创建限价订单"""
        try:
            endpoint = "/trade/order"
            
            order_data = {
                'instId': symbol,
                'tdMode': 'cash',
                'side': 'buy' if side.lower() == 'buy' else 'sell',
                'ordType': 'limit',
                'sz': str(quantity),
                'px': str(price)
            }
            
            if client_oid:
                order_data['clOrdId'] = client_oid
            
            result = await self._make_request('POST', endpoint, body=order_data, signed=True)
            
            if result and len(result) > 0:
                order_info = result[0]
                return {
                    'order_id': order_info.get('ordId'),
                    'client_oid': order_info.get('clOrdId'),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'status': 'created',
                    'timestamp': time.time()
                }
            else:
                return {'error': '限价订单创建失败'}
                
        except Exception as e:
            self.logger.error(f"创建限价订单失败 {symbol} {side} {quantity} @ {price}: {e}")
            return {'error': str(e)}
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        try:
            endpoint = "/trade/cancel-order"
            
            cancel_data = {
                'instId': symbol,
                'ordId': order_id
            }
            
            result = await self._make_request('POST', endpoint, body=cancel_data, signed=True)
            
            if result and len(result) > 0:
                cancel_info = result[0]
                return cancel_info.get('sCode') == '0'
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"取消订单失败 {order_id} {symbol}: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """获取订单状态"""
        try:
            endpoint = "/trade/order"
            params = {
                'instId': symbol,
                'ordId': order_id
            }
            
            result = await self._make_request('GET', endpoint, params=params, signed=True)
            
            if result and len(result) > 0:
                order_info = result[0]
                return {
                    'order_id': order_info.get('ordId'),
                    'symbol': symbol,
                    'side': order_info.get('side'),
                    'order_type': order_info.get('ordType'),
                    'quantity': Decimal(order_info.get('sz', '0')),
                    'filled_quantity': Decimal(order_info.get('fillSz', '0')),
                    'price': Decimal(order_info.get('px', '0')),
                    'average_price': Decimal(order_info.get('avgPx', '0')),
                    'state': order_info.get('state'),
                    'create_time': order_info.get('cTime'),
                    'update_time': order_info.get('uTime')
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"获取订单状态失败 {order_id} {symbol}: {e}")
            return {}
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """获取未成交订单"""
        try:
            endpoint = "/trade/orders-pending"
            params = {}
            if symbol:
                params['instId'] = symbol
            
            result = await self._make_request('GET', endpoint, params=params, signed=True)
            
            orders = []
            if result:
                for order_info in result:
                    order = {
                        'order_id': order_info.get('ordId'),
                        'symbol': order_info.get('instId'),
                        'side': order_info.get('side'),
                        'order_type': order_info.get('ordType'),
                        'quantity': Decimal(order_info.get('sz', '0')),
                        'filled_quantity': Decimal(order_info.get('fillSz', '0')),
                        'price': Decimal(order_info.get('px', '0')),
                        'state': order_info.get('state'),
                        'create_time': order_info.get('cTime')
                    }
                    orders.append(order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"获取未成交订单失败: {e}")
            return []
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        # 检查缓存
        cache_key = "account_balance"
        if cache_key in self._balance_cache:
            cached_data, timestamp = self._balance_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
        
        try:
            endpoint = "/asset/balances"
            result = await self._make_request('GET', endpoint, signed=True)
            
            balance_data = {
                'total': {},
                'free': {},
                'used': {}
            }
            
            if result:
                for currency_balance in result:
                    currency = currency_balance.get('ccy')
                    total = Decimal(currency_balance.get('bal', '0'))
                    available = Decimal(currency_balance.get('availBal', '0'))
                    frozen = Decimal(currency_balance.get('frozenBal', '0'))
                    
                    balance_data['total'][currency] = total
                    balance_data['free'][currency] = available
                    balance_data['used'][currency] = frozen
            
            # 更新缓存
            self._balance_cache[cache_key] = (balance_data, time.time())
            return balance_data
            
        except Exception as e:
            self.logger.error(f"获取账户余额失败: {e}")
            return {'total': {}, 'free': {}, 'used': {}}
    
    async def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        try:
            endpoint = "/account/positions"
            params = {}
            if symbol:
                params['instId'] = symbol
            
            result = await self._make_request('GET', endpoint, params=params, signed=True)
            
            positions = []
            if result:
                for position_info in result:
                    position = {
                        'symbol': position_info.get('instId'),
                        'position_side': position_info.get('posSide', 'net'),
                        'quantity': Decimal(position_info.get('pos', '0')),
                        'available_quantity': Decimal(position_info.get('availPos', '0')),
                        'average_price': Decimal(position_info.get('avgPx', '0')),
                        'leverage': position_info.get('lever', '1'),
                        'unrealized_pnl': Decimal(position_info.get('upl', '0')),
                        'margin': Decimal(position_info.get('margin', '0')),
                        'liquidation_price': Decimal(position_info.get('liqPx', '0'))
                    }
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"获取持仓信息失败: {e}")
            return []
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """获取订单簿"""
        try:
            endpoint = "/market/books"
            params = {
                'instId': symbol,
                'sz': str(depth)
            }
            
            result = await self._make_request('GET', endpoint, params=params)
            
            if result and len(result) > 0:
                orderbook_data = result[0]
                return {
                    'symbol': symbol,
                    'bids': [[Decimal(price), Decimal(quantity)] for price, quantity, _ in orderbook_data.get('bids', [])],
                    'asks': [[Decimal(price), Decimal(quantity)] for price, quantity, _ in orderbook_data.get('asks', [])],
                    'timestamp': time.time()
                }
            else:
                return {'bids': [], 'asks': []}
                
        except Exception as e:
            self.logger.error(f"获取订单簿失败 {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        uptime = time.time() - self._stats['start_time']
        
        return {
            'total_requests': self._stats['total_requests'],
            'successful_requests': self._stats['successful_requests'],
            'failed_requests': self._stats['failed_requests'],
            'rate_limit_hits': self._stats['rate_limit_hits'],
            'uptime': uptime,
            'requests_per_second': self._stats['total_requests'] / uptime if uptime > 0 else 0,
            'success_rate': (self._stats['successful_requests'] / self._stats['total_requests'] 
                           if self._stats['total_requests'] > 0 else 0)
        }
    
    async def close(self):
        """关闭连接"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def __del__(self):
        """析构函数"""
        if self._session and not self._session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except:
                pass


# 使用示例
if __name__ == "__main__":
    async def test_okx_interface():
        config = {
            'api_key': 'your_api_key',
            'secret_key': 'your_secret_key',
            'passphrase': 'your_passphrase',
            'trading_pairs': ['BTC-USDT', 'ETH-USDT']
        }
        
        interface = OKXInterface(config, sandbox=True)
        
        try:
            # 测试获取行情
            ticker = await interface.get_ticker('BTC-USDT')
            print("BTC行情:", ticker)
            
            # 测试获取余额
            balance = await interface.get_account_balance()
            print("账户余额:", balance)
            
            # 测试获取统计
            stats = interface.get_performance_stats()
            print("性能统计:", stats)
            
        finally:
            await interface.close()
    
    asyncio.run(test_okx_interface())