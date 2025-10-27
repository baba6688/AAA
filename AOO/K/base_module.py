"""
基础模块类
定义所有可发现模块的基类，支持依赖注入、生命周期管理和配置绑定
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from enum import Enum

class ModuleState(Enum):
    """模块状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DESTROYED = "destroyed"
    ERROR = "error"

class ServiceType(Enum):
    """服务类型枚举"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

class BaseModule(ABC):
    """
    基础模块抽象基类
    所有可自动发现的模块都应继承自此基类
    """
    
    # 类属性 - 用于自动发现和配置
    _aoo_discoverable = True
    _aoo_priority = 0  # 优先级，数值越大优先级越高
    _aoo_service_type = ServiceType.SINGLETON
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础模块
        
        Args:
            config: 模块配置字典
        """
        self._config = config or {}
        self._state = ModuleState.UNINITIALIZED
        self._dependencies = {}
        self._logger = logging.getLogger(f"AOO.{self.__class__.__name__}")
        
        # 模块元数据
        self._module_name = self.__class__.__name__
        self._module_version = "1.0.0"
        self._module_description = ""
        
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """初始化模块元数据"""
        # 从类属性获取元数据
        if hasattr(self.__class__, '_aoo_module_name'):
            self._module_name = getattr(self.__class__, '_aoo_module_name')
        if hasattr(self.__class__, '_aoo_version'):
            self._module_version = getattr(self.__class__, '_aoo_version')
        if hasattr(self.__class__, '_aoo_description'):
            self._module_description = getattr(self.__class__, '_aoo_description')
    
    @property
    def module_name(self) -> str:
        """获取模块名称"""
        return self._module_name
    
    @property
    def module_version(self) -> str:
        """获取模块版本"""
        return self._module_version
    
    @property
    def module_description(self) -> str:
        """获取模块描述"""
        return self._module_description
    
    @property
    def state(self) -> ModuleState:
        """获取模块状态"""
        return self._state
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取模块配置"""
        return self._config.copy()
    
    def set_dependency(self, name: str, instance: Any):
        """
        设置依赖实例
        
        Args:
            name: 依赖名称
            instance: 依赖实例
        """
        self._dependencies[name] = instance
    
    def get_dependency(self, name: str) -> Any:
        """
        获取依赖实例
        
        Args:
            name: 依赖名称
            
        Returns:
            依赖实例，如果不存在则返回None
        """
        return self._dependencies.get(name)
    
    def has_dependency(self, name: str) -> bool:
        """
        检查是否存在指定依赖
        
        Args:
            name: 依赖名称
            
        Returns:
            bool: 是否存在
        """
        return name in self._dependencies
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化模块（抽象方法）
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """
        启动模块（抽象方法）
        
        Returns:
            bool: 启动是否成功
        """
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """
        停止模块（抽象方法）
        
        Returns:
            bool: 停止是否成功
        """
        pass
    
    async def destroy(self):
        """
        销毁模块，释放资源
        子类可以重写此方法以进行资源清理
        """
        self._state = ModuleState.DESTROYED
        self._dependencies.clear()
        self._logger.info(f"模块 {self._module_name} 已销毁")
    
    def get_config_section(self) -> Optional[str]:
        """
        获取配置章节名
        子类可以重写此方法以指定配置章节
        
        Returns:
            str: 配置章节名，返回None表示不使用特定章节
        """
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict: 健康状态信息
        """
        return {
            "module": self._module_name,
            "version": self._module_version,
            "state": self._state.value,
            "healthy": self._state in [ModuleState.INITIALIZED, ModuleState.RUNNING],
            "dependencies": list(self._dependencies.keys()),
            "timestamp": self._get_timestamp()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取模块指标
        
        Returns:
            Dict: 模块指标数据
        """
        return {
            "module": self._module_name,
            "state": self._state.value,
            "dependency_count": len(self._dependencies),
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _set_state(self, new_state: ModuleState):
        """设置模块状态"""
        old_state = self._state
        self._state = new_state
        self._logger.debug(f"模块状态变更: {old_state.value} -> {new_state.value}")
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self._module_name}(v{self._module_version})[{self._state.value}]"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"<{self.__class__.__name__} name={self._module_name} version={self._module_version} state={self._state.value}>"


class DiscoverableModule(BaseModule):
    """
    可发现模块基类
    继承此类的模块会被自动扫描和注册
    """
    
    _aoo_discoverable = True
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._discovery_tags = set()
    
    def add_discovery_tag(self, tag: str):
        """添加发现标签"""
        self._discovery_tags.add(tag)
    
    def remove_discovery_tag(self, tag: str):
        """移除发现标签"""
        self._discovery_tags.discard(tag)
    
    def has_discovery_tag(self, tag: str) -> bool:
        """检查是否包含指定标签"""
        return tag in self._discovery_tags
    
    @property
    def discovery_tags(self) -> set:
        """获取所有发现标签"""
        return self._discovery_tags.copy()


class FactoryModule(DiscoverableModule):
    """
    工厂模块基类
    用于创建和管理其他模块实例的工厂模块
    """
    
    _aoo_service_type = ServiceType.SINGLETON
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._created_instances = {}
        self._instance_counter = 0
    
    async def create_instance(self, instance_type: Type, **kwargs) -> Any:
        """
        创建指定类型的实例
        
        Args:
            instance_type: 实例类型
            **kwargs: 创建参数
            
        Returns:
            创建的实例
        """
        try:
            instance_id = self._generate_instance_id()
            instance = instance_type(**kwargs)
            self._created_instances[instance_id] = instance
            self._logger.info(f"创建实例: {instance_type.__name__}[{instance_id}]")
            return instance
        except Exception as e:
            self._logger.error(f"创建实例失败 {instance_type.__name__}: {e}")
            raise
    
    async def destroy_instance(self, instance_id: str):
        """销毁指定实例"""
        if instance_id in self._created_instances:
            instance = self._created_instances.pop(instance_id)
            if hasattr(instance, 'destroy'):
                await instance.destroy()
            self._logger.info(f"销毁实例: {instance_id}")
    
    def _generate_instance_id(self) -> str:
        """生成实例ID"""
        self._instance_counter += 1
        return f"inst_{self._instance_counter}_{id(self)}"
    
    async def destroy(self):
        """销毁工厂，清理所有实例"""
        for instance_id in list(self._created_instances.keys()):
            await self.destroy_instance(instance_id)
        await super().destroy()


class ServiceModule(DiscoverableModule):
    """
    服务模块基类
    提供具体业务服务的模块基类
    """
    
    _aoo_service_type = ServiceType.SINGLETON
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._service_endpoints = {}
        self._service_health = {}
    
    def register_endpoint(self, endpoint: str, handler: callable):
        """
        注册服务端点
        
        Args:
            endpoint: 端点路径
            handler: 处理函数
        """
        self._service_endpoints[endpoint] = handler
        self._service_health[endpoint] = True
        self._logger.info(f"注册服务端点: {endpoint}")
    
    def unregister_endpoint(self, endpoint: str):
        """注销服务端点"""
        if endpoint in self._service_endpoints:
            del self._service_endpoints[endpoint]
            del self._service_health[endpoint]
            self._logger.info(f"注销服务端点: {endpoint}")
    
    async def call_endpoint(self, endpoint: str, *args, **kwargs) -> Any:
        """
        调用服务端点
        
        Args:
            endpoint: 端点路径
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            端点调用结果
        """
        if endpoint not in self._service_endpoints:
            raise ValueError(f"服务端点不存在: {endpoint}")
        
        handler = self._service_endpoints[endpoint]
        try:
            result = await handler(*args, **kwargs) if hasattr(handler, '__await__') else handler(*args, **kwargs)
            self._service_health[endpoint] = True
            return result
        except Exception as e:
            self._service_health[endpoint] = False
            self._logger.error(f"服务端点调用失败 {endpoint}: {e}")
            raise
    
    def get_service_endpoints(self) -> List[str]:
        """获取所有服务端点"""
        return list(self._service_endpoints.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查（包含端点健康状态）"""
        base_health = super().health_check()
        base_health.update({
            "service_endpoints": self.get_service_endpoints(),
            "endpoint_health": self._service_health.copy()
        })
        return base_health


class TradingModule(ServiceModule):
    """
    交易模块基类
    专门用于量化交易的模块基类
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._trading_enabled = False
        self._market_data = {}
        self._positions = {}
        self._orders = {}
    
    async def initialize(self) -> bool:
        """初始化交易模块"""
        self._set_state(ModuleState.INITIALIZING)
        try:
            # 初始化交易接口
            await self._initialize_trading_interface()
            self._set_state(ModuleState.INITIALIZED)
            self._logger.info("交易模块初始化完成")
            return True
        except Exception as e:
            self._set_state(ModuleState.ERROR)
            self._logger.error(f"交易模块初始化失败: {e}")
            return False
    
    async def start(self) -> bool:
        """启动交易模块"""
        self._set_state(ModuleState.STARTING)
        try:
            self._trading_enabled = True
            await self._start_market_data()
            self._set_state(ModuleState.RUNNING)
            self._logger.info("交易模块启动完成")
            return True
        except Exception as e:
            self._set_state(ModuleState.ERROR)
            self._logger.error(f"交易模块启动失败: {e}")
            return False
    
    async def stop(self) -> bool:
        """停止交易模块"""
        self._set_state(ModuleState.STOPPING)
        try:
            self._trading_enabled = False
            await self._stop_market_data()
            await self._cancel_pending_orders()
            self._set_state(ModuleState.STOPPED)
            self._logger.info("交易模块停止完成")
            return True
        except Exception as e:
            self._set_state(ModuleState.ERROR)
            self._logger.error(f"交易模块停止失败: {e}")
            return False
    
    @abstractmethod
    async def _initialize_trading_interface(self):
        """初始化交易接口（抽象方法）"""
        pass
    
    @abstractmethod
    async def _start_market_data(self):
        """启动市场数据（抽象方法）"""
        pass
    
    @abstractmethod
    async def _stop_market_data(self):
        """停止市场数据（抽象方法）"""
        pass
    
    @abstractmethod
    async def _cancel_pending_orders(self):
        """取消挂单（抽象方法）"""
        pass
    
    async def place_order(self, symbol: str, order_type: str, quantity: float, price: float = None) -> str:
        """
        下单
        
        Args:
            symbol: 交易对
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单需要）
            
        Returns:
            str: 订单ID
        """
        if not self._trading_enabled:
            raise RuntimeError("交易模块未启动")
        
        order_id = self._generate_order_id()
        self._orders[order_id] = {
            'symbol': symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'status': 'pending',
            'timestamp': self._get_timestamp()
        }
        
        self._logger.info(f"下单: {symbol} {order_type} {quantity} @ {price}")
        return order_id
    
    def _generate_order_id(self) -> str:
        """生成订单ID"""
        import uuid
        return f"order_{uuid.uuid4().hex[:8]}"
    
    def get_positions(self) -> Dict[str, float]:
        """获取持仓"""
        return self._positions.copy()
    
    def get_orders(self) -> Dict[str, Dict]:
        """获取订单"""
        return self._orders.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查（包含交易状态）"""
        base_health = super().health_check()
        base_health.update({
            "trading_enabled": self._trading_enabled,
            "position_count": len(self._positions),
            "order_count": len(self._orders),
            "market_data_sources": len(self._market_data)
        })
        return base_health


# 模块装饰器，用于标记可发现类
def discoverable(priority: int = 0, service_type: ServiceType = ServiceType.SINGLETON):
    """
    可发现类装饰器
    
    Args:
        priority: 优先级
        service_type: 服务类型
    """
    def decorator(cls):
        cls._aoo_discoverable = True
        cls._aoo_priority = priority
        cls._aoo_service_type = service_type
        return cls
    return decorator


def trading_module(config_section: str = None):
    """
    交易模块装饰器
    
    Args:
        config_section: 配置章节名
    """
    def decorator(cls):
        cls._aoo_discoverable = True
        cls._aoo_service_type = ServiceType.SINGLETON
        if config_section:
            cls.get_config_section = lambda self: config_section
        return cls
    return decorator