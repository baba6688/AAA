import asyncio
import websocket
import json
import logging
from typing import Dict, Callable, List
import pandas as pd
from datetime import datetime

class MarketDataConsumer:
    """市场数据消费者 - 实时数据接入核心模块"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ws = None
        self.callbacks = []
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        
        # OKX配置
        self.okx_config = config.get('okx', {})
        self.base_url = self.okx_config.get('base_url', 'https://www.okx.com')
        self.ws_url = self.okx_config.get('ws_url', 'wss://ws.okx.com:8443/ws/v5/public')
        
        # 订阅的交易对
        self.trading_pairs = config.get('trading_pairs', ['BTC-USDT', 'ETH-USDT'])
        
    def add_callback(self, callback: Callable):
        """添加数据回调函数"""
        self.callbacks.append(callback)
        
    async def connect(self):
        """连接WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # 在单独线程中运行WebSocket
            await asyncio.get_event_loop().run_in_executor(
                None, self.ws.run_forever
            )
            
        except Exception as e:
            self.logger.error(f"WebSocket连接失败: {e}")
            
    def _on_open(self, ws):
        """WebSocket连接打开"""
        self.is_connected = True
        self.logger.info("WebSocket连接已建立")
        
        # 订阅行情数据
        self._subscribe_market_data()
        
    def _on_message(self, ws, message):
        """处理接收到的消息"""
        try:
            data = json.loads(message)
            
            # 处理不同的消息类型
            if 'event' in data:
                self._handle_control_message(data)
            else:
                self._handle_market_data(data)
                
        except Exception as e:
            self.logger.error(f"消息处理错误: {e}")
            
    def _on_error(self, ws, error):
        """处理错误"""
        self.logger.error(f"WebSocket错误: {error}")
        self.is_connected = False
        
    def _on_close(self, ws, close_status_code, close_msg):
        """连接关闭"""
        self.logger.info("WebSocket连接已关闭")
        self.is_connected = False
        
    def _subscribe_market_data(self):
        """订阅市场数据"""
        subscriptions = []
        
        for pair in self.trading_pairs:
            # 订阅ticker数据
            subscriptions.append({
                "op": "subscribe",
                "args": [{"channel": "tickers", "instId": pair}]
            })
            
            # 订阅K线数据
            subscriptions.append({
                "op": "subscribe", 
                "args": [{"channel": "candle1m", "instId": pair}]
            })
            
            # 订阅深度数据
            subscriptions.append({
                "op": "subscribe",
                "args": [{"channel": "books5", "instId": pair}]
            })
        
        for sub in subscriptions:
            self.ws.send(json.dumps(sub))
            self.logger.info(f"订阅市场数据: {sub}")
            
    def _handle_control_message(self, data: Dict):
        """处理控制消息"""
        event = data.get('event')
        if event == 'subscribe':
            self.logger.info(f"订阅成功: {data.get('arg')}")
        elif event == 'error':
            self.logger.error(f"订阅错误: {data}")
            
    def _handle_market_data(self, data: Dict):
        """处理市场数据"""
        try:
            arg = data.get('arg', {})
            channel = arg.get('channel')
            inst_id = arg.get('instId')
            
            # 根据频道类型处理数据
            if channel == 'tickers':
                self._process_ticker_data(data, inst_id)
            elif channel == 'candle1m':
                self._process_candle_data(data, inst_id)
            elif channel == 'books5':
                self._process_depth_data(data, inst_id)
                
        except Exception as e:
            self.logger.error(f"市场数据处理错误: {e}")
            
    def _process_ticker_data(self, data: Dict, inst_id: str):
        """处理ticker数据"""
        ticker_data = data.get('data', [{}])[0]
        
        processed_data = {
            'timestamp': datetime.now(),
            'symbol': inst_id,
            'last_price': float(ticker_data.get('last', 0)),
            'bid_price': float(ticker_data.get('bidPx', 0)),
            'ask_price': float(ticker_data.get('askPx', 0)),
            'volume_24h': float(ticker_data.get('vol24h', 0)),
            'price_change_24h': float(ticker_data.get('change24h', 0)),
            'price_change_percent_24h': float(ticker_data.get('changePercent24h', 0))
        }
        
        # 通知回调函数
        for callback in self.callbacks:
            try:
                callback('ticker', processed_data)
            except Exception as e:
                self.logger.error(f"回调函数执行错误: {e}")
                
    def _process_candle_data(self, data: Dict, inst_id: str):
        """处理K线数据"""
        candle_data = data.get('data', [{}])[0]
        
        processed_data = {
            'timestamp': datetime.fromtimestamp(int(candle_data[0]) / 1000),
            'symbol': inst_id,
            'open': float(candle_data[1]),
            'high': float(candle_data[2]),
            'low': float(candle_data[3]),
            'close': float(candle_data[4]),
            'volume': float(candle_data[5])
        }
        
        for callback in self.callbacks:
            try:
                callback('candle', processed_data)
            except Exception as e:
                self.logger.error(f"回调函数执行错误: {e}")
                
    def _process_depth_data(self, data: Dict, inst_id: str):
        """处理深度数据"""
        depth_data = data.get('data', [{}])[0]
        
        processed_data = {
            'timestamp': datetime.now(),
            'symbol': inst_id,
            'bids': [(float(bid[0]), float(bid[1])) for bid in depth_data.get('bids', [])[:5]],
            'asks': [(float(ask[0]), float(ask[1])) for ask in depth_data.get('asks', [])[:5]]
        }
        
        for callback in self.callbacks:
            try:
                callback('depth', processed_data)
            except Exception as e:
                self.logger.error(f"回调函数执行错误: {e}")
                
    async def start(self):
        """启动数据消费"""
        self.logger.info("启动市场数据消费者")
        await self.connect()
        
    async def stop(self):
        """停止数据消费"""
        if self.ws:
            self.ws.close()
        self.is_connected = False
        self.logger.info("市场数据消费者已停止")