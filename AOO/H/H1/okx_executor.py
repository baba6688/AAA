import os
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

# 导入同级和下级模块
try:
    from .H2.okx_interface import OKXInterface
    from .H2.okx_simulator import OKXSimulator
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from H.H2.okx_interface import OKXInterface
    from H.H2.okx_simulator import OKXSimulator


class OKXExecutor:
    """
    OKX执行器 - 统一交易执行管理
    负责实盘和模拟盘的交易执行、风险控制和订单管理
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化OKX执行器
        
        Args:
            config: 配置字典，包含交易参数和风险控制设置
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # 交易模式
        self.trading_mode = self.config.get('trading_mode', 'simulation')  # 'live' 或 'simulation'
        
        # 初始化接口
        self.live_interface = None
        self.sim_interface = None
        self._init_interfaces()
        
        # 执行统计
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': 0.0,
            'last_execution': None
        }
        
        self.logger.info(f"OKX执行器初始化完成 - 模式: {self.trading_mode}")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('OKXExecutor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _init_interfaces(self):
        """初始化交易接口"""
        # 构建配置
        live_config = self._build_live_config()
        sim_config = self._build_sim_config()
        
        # 初始化模拟接口
        try:
            self.sim_interface = OKXSimulator(sim_config)
            self.logger.info("模拟交易接口初始化成功")
        except Exception as e:
            self.logger.error(f"模拟交易接口初始化失败: {e}")
        
        # 初始化实盘接口（如果配置了API密钥）
        if live_config.get('api_key') and live_config.get('secret_key'):
            try:
                self.live_interface = OKXInterface(live_config, sandbox=False)
                self.logger.info("实盘交易接口初始化成功")
            except Exception as e:
                self.logger.error(f"实盘交易接口初始化失败: {e}")
        else:
            self.logger.warning("实盘API密钥未配置，仅使用模拟模式")
    
    def _build_live_config(self) -> Dict:
        """构建实盘配置"""
        return {
            'api_key': os.getenv('OKX_API_KEY', ''),
            'secret_key': os.getenv('OKX_API_SECRET', ''),
            'passphrase': os.getenv('OKX_PASSPHRASE', ''),
            'base_url': os.getenv('OKX_BASE_URL', 'https://www.okx.com'),
            'trading_pairs': self.config.get('trading_pairs', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']),
            'rate_limits': self.config.get('rate_limits', {
                'requests_per_second': 10,
                'order_interval_ms': 100
            })
        }
    
    def _build_sim_config(self) -> Dict:
        """构建模拟盘配置"""
        return {
            'api_key': os.getenv('PAPER_API_KEY', ''),
            'secret_key': os.getenv('PAPER_SECRET_KEY', ''),
            'passphrase': os.getenv('PAPER_PASSPHRASE', ''),
            'base_url': os.getenv('PAPER_BASE_URL', 'https://www.okx.com'),
            'trading_pairs': self.config.get('trading_pairs', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']),
            'initial_balance': self.config.get('initial_balance', {
                'USDT': 10000.0,
                'BTC': 0.0,
                'ETH': 0.0,
                'ADA': 0.0
            })
        }
    
    def switch_mode(self, mode: str) -> bool:
        """
        切换交易模式
        
        Args:
            mode: 'live' 或 'simulation'
            
        Returns:
            bool: 切换是否成功
        """
        if mode not in ['live', 'simulation']:
            self.logger.error(f"不支持的交易模式: {mode}")
            return False
        
        if mode == 'live' and not self.live_interface:
            self.logger.error("无法切换到实盘模式：实盘API未配置或初始化失败")
            return False
        
        old_mode = self.trading_mode
        self.trading_mode = mode
        self.logger.info(f"交易模式已切换: {old_mode} -> {mode}")
        return True
    
    def get_current_interface(self):
        """获取当前交易模式对应的接口"""
        if self.trading_mode == 'live' and self.live_interface:
            return self.live_interface
        else:
            return self.sim_interface
    
    def execute_trade(self, trade_signal: Dict) -> Dict:
        """
        执行交易信号
        
        Args:
            trade_signal: 交易信号字典，包含以下字段:
                - symbol: 交易对，如 'BTC/USDT'
                - action: 操作类型，'buy' 或 'sell'
                - quantity: 数量
                - order_type: 订单类型，'market' 或 'limit'
                - price: 限价单价格（对于市价单可省略）
                - strategy: 策略名称
                - metadata: 其他元数据
                
        Returns:
            Dict: 执行结果
        """
        try:
            # 验证交易信号
            validation_result = self._validate_trade_signal(trade_signal)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'timestamp': datetime.now()
                }
            
            symbol = trade_signal['symbol']
            action = trade_signal['action']
            quantity = trade_signal['quantity']
            order_type = trade_signal.get('order_type', 'market')
            price = trade_signal.get('price')
            strategy = trade_signal.get('strategy', 'unknown')
            
            # 风险控制检查
            risk_check = self._risk_control_check(symbol, action, quantity, price)
            if not risk_check['passed']:
                return {
                    'success': False,
                    'error': risk_check['reason'],
                    'timestamp': datetime.now()
                }
            
            # 执行交易
            interface = self.get_current_interface()
            if order_type == 'market':
                result = interface.create_market_order(symbol, action, quantity)
            elif order_type == 'limit':
                if not price:
                    return {
                        'success': False,
                        'error': '限价单必须指定价格',
                        'timestamp': datetime.now()
                    }
                result = interface.create_limit_order(symbol, action, quantity, price)
            else:
                return {
                    'success': False,
                    'error': f'不支持的订单类型: {order_type}',
                    'timestamp': datetime.now()
                }
            
            # 处理执行结果
            execution_result = self._process_execution_result(
                result, symbol, action, quantity, order_type, price, strategy
            )
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"执行交易异常: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _validate_trade_signal(self, trade_signal: Dict) -> Dict:
        """验证交易信号"""
        required_fields = ['symbol', 'action', 'quantity']
        
        for field in required_fields:
            if field not in trade_signal:
                return {
                    'valid': False,
                    'error': f'交易信号缺少必要字段: {field}'
                }
        
        if trade_signal['action'] not in ['buy', 'sell']:
            return {
                'valid': False,
                'error': f'无效的操作类型: {trade_signal["action"]}'
            }
        
        if trade_signal['quantity'] <= 0:
            return {
                'valid': False,
                'error': f'无效的数量: {trade_signal["quantity"]}'
            }
        
        order_type = trade_signal.get('order_type', 'market')
        if order_type not in ['market', 'limit']:
            return {
                'valid': False,
                'error': f'无效的订单类型: {order_type}'
            }
        
        if order_type == 'limit' and 'price' not in trade_signal:
            return {
                'valid': False,
                'error': '限价单必须指定价格'
            }
        
        return {'valid': True}
    
    def _risk_control_check(self, symbol: str, action: str, quantity: float, 
                           price: float = None) -> Dict:
        """
        风险控制检查
        
        Returns:
            Dict: {'passed': bool, 'reason': str}
        """
        try:
            interface = self.get_current_interface()
            
            # 获取账户余额
            balance = interface.get_account_balance()
            if not balance:
                return {'passed': False, 'reason': '无法获取账户余额'}
            
            # 解析交易对
            if '/' in symbol:
                base_currency, quote_currency = symbol.split('/')
            else:
                # 默认使用USDT作为报价货币
                base_currency = symbol.replace('USDT', '')
                quote_currency = 'USDT'
            
            # 资金检查
            if action == 'buy':
                # 买入：检查报价货币余额
                if price:
                    required_funds = quantity * price
                else:
                    # 市价单，使用当前卖一价估算
                    ticker = interface.get_ticker(symbol)
                    if not ticker or 'ask' not in ticker:
                        return {'passed': False, 'reason': '无法获取行情数据'}
                    required_funds = quantity * ticker['ask']
                
                available_quote = balance['free'].get(quote_currency, 0)
                if available_quote < required_funds:
                    return {
                        'passed': False, 
                        'reason': f'资金不足: 需要 {required_funds:.2f} {quote_currency}, 可用 {available_quote:.2f}'
                    }
            
            elif action == 'sell':
                # 卖出：检查基础货币余额
                available_base = balance['free'].get(base_currency, 0)
                if available_base < quantity:
                    return {
                        'passed': False,
                        'reason': f'资产不足: 需要 {quantity} {base_currency}, 可用 {available_base}'
                    }
            
            # 仓位限制检查
            positions = interface.get_positions(symbol)
            current_position = sum(pos.get('amount', 0) for pos in positions)
            
            max_position = self.config.get('risk', {}).get('max_position_size', 10)
            if action == 'buy' and current_position + quantity > max_position:
                return {
                    'passed': False,
                    'reason': f'超过最大持仓限制: 当前 {current_position}, 请求 {quantity}, 最大 {max_position}'
                }
            
            # 单笔交易风险检查
            max_risk_per_trade = self.config.get('risk', {}).get('max_risk_per_trade', 0.02)
            account_value = sum(balance['total'].values())
            if account_value > 0:
                trade_value = quantity * (price or self._get_current_price(symbol))
                risk_ratio = trade_value / account_value
                if risk_ratio > max_risk_per_trade:
                    return {
                        'passed': False,
                        'reason': f'单笔交易风险过高: {risk_ratio:.2%}, 最大允许 {max_risk_per_trade:.2%}'
                    }
            
            return {'passed': True, 'reason': '风险检查通过'}
            
        except Exception as e:
            self.logger.error(f"风险检查异常: {e}")
            return {'passed': False, 'reason': f'风险检查异常: {str(e)}'}
    
    def _get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            interface = self.get_current_interface()
            ticker = interface.get_ticker(symbol)
            return ticker.get('last', 0) if ticker else 0
        except:
            return 0
    
    def _process_execution_result(self, result: Dict, symbol: str, action: str, 
                                quantity: float, order_type: str, price: float, 
                                strategy: str) -> Dict:
        """处理执行结果"""
        # 更新执行统计
        self.execution_stats['total_orders'] += 1
        self.execution_stats['last_execution'] = datetime.now()
        
        if result and result.get('order_id'):
            self.execution_stats['successful_orders'] += 1
            self.execution_stats['total_volume'] += quantity
            
            execution_result = {
                'success': True,
                'order_id': result.get('order_id'),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'executed_price': result.get('price'),
                'status': result.get('status', 'unknown'),
                'strategy': strategy,
                'timestamp': datetime.now(),
                'mode': self.trading_mode
            }
            
            self.logger.info(
                f"交易执行成功: {strategy} {action} {quantity} {symbol} "
                f"@{result.get('price')} [{order_type}]"
            )
            
            return execution_result
        else:
            self.execution_stats['failed_orders'] += 1
            
            error_result = {
                'success': False,
                'error': '订单创建失败',
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'strategy': strategy,
                'timestamp': datetime.now(),
                'mode': self.trading_mode
            }
            
            self.logger.error(
                f"交易执行失败: {strategy} {action} {quantity} {symbol} [{order_type}]"
            )
            
            return error_result
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            symbol: 交易对
            
        Returns:
            Dict: 取消结果
        """
        try:
            interface = self.get_current_interface()
            success = interface.cancel_order(order_id, symbol)
            
            result = {
                'success': success,
                'order_id': order_id,
                'symbol': symbol,
                'timestamp': datetime.now()
            }
            
            if success:
                self.logger.info(f"订单取消成功: {order_id} {symbol}")
            else:
                self.logger.error(f"订单取消失败: {order_id} {symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"取消订单异常: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id,
                'symbol': symbol,
                'timestamp': datetime.now()
            }
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """获取订单状态"""
        try:
            interface = self.get_current_interface()
            return interface.get_order_status(order_id, symbol)
        except Exception as e:
            self.logger.error(f"获取订单状态异常: {e}")
            return {}
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """获取未成交订单列表"""
        try:
            interface = self.get_current_interface()
            return interface.get_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"获取未成交订单异常: {e}")
            return []
    
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """获取持仓信息"""
        try:
            interface = self.get_current_interface()
            return interface.get_positions(symbol)
        except Exception as e:
            self.logger.error(f"获取持仓信息异常: {e}")
            return []
    
    def get_account_balance(self) -> Dict:
        """获取账户余额"""
        try:
            interface = self.get_current_interface()
            return interface.get_account_balance()
        except Exception as e:
            self.logger.error(f"获取账户余额异常: {e}")
            return {}
    
    def get_execution_stats(self) -> Dict:
        """获取执行统计"""
        stats = self.execution_stats.copy()
        if stats['total_orders'] > 0:
            stats['success_rate'] = stats['successful_orders'] / stats['total_orders']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_simulator(self, initial_balance: Dict = None):
        """重置模拟器（仅模拟模式有效）"""
        if self.sim_interface and hasattr(self.sim_interface, 'reset_simulator'):
            self.sim_interface.reset_simulator(initial_balance)
            self.logger.info("模拟器已重置")
        else:
            self.logger.warning("重置模拟器仅适用于模拟模式")


# 使用示例
if __name__ == "__main__":
    # 测试代码
    config = {
        'trading_mode': 'simulation',
        'trading_pairs': ['BTC/USDT', 'ETH/USDT'],
        'risk': {
            'max_position_size': 5,
            'max_risk_per_trade': 0.02
        }
    }
    
    executor = OKXExecutor(config)
    
    # 示例交易信号
    trade_signal = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'quantity': 0.001,
        'order_type': 'market',
        'strategy': 'test_strategy'
    }
    
    result = executor.execute_trade(trade_signal)
    print("交易结果:", result)
    
    balance = executor.get_account_balance()
    print("账户余额:", balance)
    
    stats = executor.get_execution_stats()
    print("执行统计:", stats)