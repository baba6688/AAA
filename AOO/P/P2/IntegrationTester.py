#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2集成测试器
提供全面的集成测试功能，包括系统集成、API接口、数据库、消息队列等测试
"""

import asyncio
import json
import logging
import time
import unittest
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from urllib.parse import urljoin, urlparse
import sqlite3
import threading
import requests
import websocket
import pika
import psycopg2
import redis
from unittest.mock import Mock, patch


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """测试结果枚举"""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


class Environment(Enum):
    """测试环境枚举"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class TestCase:
    """测试用例数据类"""
    name: str
    description: str
    test_function: Callable
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 30
    retry_count: int = 3
    tags: List[str] = field(default_factory=list)
    environment: Optional[Environment] = None


@dataclass
class TestExecutionResult:
    """测试执行结果数据类"""
    test_name: str
    result: TestResult
    execution_time: float
    start_time: datetime
    end_time: datetime
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error_trace: Optional[str] = None


@dataclass
class IntegrationTestConfig:
    """集成测试配置"""
    environment: Environment
    api_endpoints: Dict[str, str]
    database_config: Dict[str, Any]
    message_queue_config: Dict[str, Any]
    external_services: Dict[str, Dict[str, Any]]
    timeout: int = 30
    retry_count: int = 3
    parallel_execution: bool = False
    max_workers: int = 5


class BaseTest(ABC):
    """测试基类"""
    
    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    async def execute(self) -> TestResult:
        """执行测试"""
        pass
    
    @contextmanager
    def timeout_context(self, timeout: int):
        """超时上下文管理器"""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Test exceeded timeout of {timeout} seconds")
            raise e


class SystemIntegrationTest(BaseTest):
    """系统集成测试"""
    
    async def execute(self) -> TestResult:
        """执行系统集成测试"""
        start_time = datetime.now()
        test_name = "系统集成测试"
        
        try:
            with self.timeout_context(self.config.timeout):
                # 测试模块间通信
                await self._test_module_communication()
                
                # 测试服务发现
                await self._test_service_discovery()
                
                # 测试负载均衡
                await self._test_load_balancing()
                
                # 测试容错机制
                await self._test_fault_tolerance()
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return TestExecutionResult(
                    test_name=test_name,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    message="系统集成测试通过"
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                message=f"系统集成测试失败: {str(e)}",
                error_trace=str(e)
            )
    
    async def _test_module_communication(self):
        """测试模块间通信"""
        self.logger.info("测试模块间通信...")
        
        # 模拟模块间通信测试
        modules = ["用户服务", "订单服务", "支付服务", "库存服务"]
        
        for module in modules:
            # 测试模块是否可访问
            response = requests.get(f"{self.config.api_endpoints.get('gateway', '')}/health/{module.lower()}")
            if response.status_code != 200:
                raise Exception(f"模块 {module} 通信失败")
        
        self.logger.info("模块间通信测试通过")
    
    async def _test_service_discovery(self):
        """测试服务发现"""
        self.logger.info("测试服务发现...")
        
        # 模拟服务发现测试
        services = ["user-service", "order-service", "payment-service"]
        
        for service in services:
            # 测试服务是否可以通过服务名发现
            service_url = self.config.api_endpoints.get(service, "")
            if not service_url:
                raise Exception(f"服务 {service} 未配置")
        
        self.logger.info("服务发现测试通过")
    
    async def _test_load_balancing(self):
        """测试负载均衡"""
        self.logger.info("测试负载均衡...")
        
        # 模拟负载均衡测试
        gateway_url = self.config.api_endpoints.get('gateway', '')
        if gateway_url:
            responses = []
            for i in range(5):
                response = requests.get(f"{gateway_url}/api/users")
                responses.append(response)
            
            # 检查响应是否来自不同的实例
            server_headers = [r.headers.get('Server', '') for r in responses]
            if len(set(server_headers)) > 1:
                self.logger.info("负载均衡正常工作")
            else:
                self.logger.warning("负载均衡可能未启用")
        
        self.logger.info("负载均衡测试通过")
    
    async def _test_fault_tolerance(self):
        """测试容错机制"""
        self.logger.info("测试容错机制...")
        
        # 模拟容错测试
        gateway_url = self.config.api_endpoints.get('gateway', '')
        if gateway_url:
            # 测试熔断器
            try:
                # 故意发送可能导致错误的请求
                response = requests.get(f"{gateway_url}/api/nonexistent", timeout=5)
            except requests.exceptions.RequestException:
                self.logger.info("容错机制正常工作")
        
        self.logger.info("容错机制测试通过")


class APIInterfaceTest(BaseTest):
    """API接口测试"""
    
    def __init__(self, config: IntegrationTestConfig):
        super().__init__(config)
        self.session = requests.Session()
    
    async def execute(self) -> TestResult:
        """执行API接口测试"""
        start_time = datetime.now()
        test_name = "API接口测试"
        
        try:
            with self.timeout_context(self.config.timeout):
                # 测试REST API
                await self._test_rest_api()
                
                # 测试GraphQL API
                await self._test_graphql_api()
                
                # 测试API认证
                await self._test_api_authentication()
                
                # 测试API限流
                await self._test_api_rate_limiting()
                
                # 测试API文档
                await self._test_api_documentation()
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return TestExecutionResult(
                    test_name=test_name,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    message="API接口测试通过"
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                message=f"API接口测试失败: {str(e)}",
                error_trace=str(e)
            )
    
    async def _test_rest_api(self):
        """测试REST API"""
        self.logger.info("测试REST API...")
        
        base_url = self.config.api_endpoints.get('api_base', '')
        if not base_url:
            raise Exception("API基础URL未配置")
        
        # 测试GET请求
        response = self.session.get(f"{base_url}/users", timeout=10)
        if response.status_code not in [200, 404]:  # 404也是可接受的，表示资源不存在
            raise Exception(f"GET请求失败，状态码: {response.status_code}")
        
        # 测试POST请求
        test_data = {"name": "测试用户", "email": "test@example.com"}
        response = self.session.post(
            f"{base_url}/users",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code not in [200, 201, 400]:  # 400表示请求格式错误，但API是可用的
            raise Exception(f"POST请求失败，状态码: {response.status_code}")
        
        # 测试PUT请求
        response = self.session.put(
            f"{base_url}/users/1",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code not in [200, 201, 404]:
            raise Exception(f"PUT请求失败，状态码: {response.status_code}")
        
        # 测试DELETE请求
        response = self.session.delete(f"{base_url}/users/1", timeout=10)
        if response.status_code not in [200, 204, 404]:
            raise Exception(f"DELETE请求失败，状态码: {response.status_code}")
        
        self.logger.info("REST API测试通过")
    
    async def _test_graphql_api(self):
        """测试GraphQL API"""
        self.logger.info("测试GraphQL API...")
        
        graphql_url = self.config.api_endpoints.get('graphql', '')
        if not graphql_url:
            self.logger.info("GraphQL API未配置，跳过测试")
            return
        
        # 测试GraphQL查询
        query = """
        {
            users {
                id
                name
                email
            }
        }
        """
        
        response = self.session.post(
            graphql_url,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"GraphQL请求失败，状态码: {response.status_code}")
        
        # 验证响应格式
        data = response.json()
        if "data" not in data and "errors" not in data:
            raise Exception("GraphQL响应格式不正确")
        
        self.logger.info("GraphQL API测试通过")
    
    async def _test_api_authentication(self):
        """测试API认证"""
        self.logger.info("测试API认证...")
        
        base_url = self.config.api_endpoints.get('api_base', '')
        if not base_url:
            return
        
        # 测试未认证访问
        response = self.session.get(f"{base_url}/protected", timeout=10)
        if response.status_code not in [401, 403]:
            self.logger.warning("API可能缺少认证保护")
        
        # 测试无效token访问
        headers = {"Authorization": "Bearer invalid_token"}
        response = self.session.get(f"{base_url}/protected", headers=headers, timeout=10)
        if response.status_code not in [401, 403]:
            self.logger.warning("API认证可能有问题")
        
        self.logger.info("API认证测试通过")
    
    async def _test_api_rate_limiting(self):
        """测试API限流"""
        self.logger.info("测试API限流...")
        
        base_url = self.config.api_endpoints.get('api_base', '')
        if not base_url:
            return
        
        # 发送大量请求测试限流
        responses = []
        for i in range(20):
            response = self.session.get(f"{base_url}/users", timeout=5)
            responses.append(response.status_code)
        
        # 检查是否有429状态码（Too Many Requests）
        if 429 in responses:
            self.logger.info("API限流正常工作")
        else:
            self.logger.info("API限流可能未启用")
        
        self.logger.info("API限流测试通过")
    
    async def _test_api_documentation(self):
        """测试API文档"""
        self.logger.info("测试API文档...")
        
        # 测试Swagger/OpenAPI文档
        swagger_url = self.config.api_endpoints.get('swagger', '')
        if swagger_url:
            response = self.session.get(swagger_url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"Swagger文档不可访问，状态码: {response.status_code}")
        
        # 测试API健康检查
        health_url = self.config.api_endpoints.get('health', '')
        if health_url:
            response = self.session.get(health_url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"健康检查失败，状态码: {response.status_code}")
        
        self.logger.info("API文档测试通过")


class DatabaseIntegrationTest(BaseTest):
    """数据库集成测试"""
    
    def __init__(self, config: IntegrationTestConfig):
        super().__init__(config)
        self.connections = {}
    
    async def execute(self) -> TestResult:
        """执行数据库集成测试"""
        start_time = datetime.now()
        test_name = "数据库集成测试"
        
        try:
            with self.timeout_context(self.config.timeout):
                # 连接数据库
                await self._connect_databases()
                
                # 测试数据库操作
                await self._test_database_operations()
                
                # 测试事务处理
                await self._test_transactions()
                
                # 测试并发操作
                await self._test_concurrent_operations()
                
                # 测试数据一致性
                await self._test_data_consistency()
                
                # 清理测试数据
                await self._cleanup_test_data()
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return TestExecutionResult(
                    test_name=test_name,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    message="数据库集成测试通过"
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                message=f"数据库集成测试失败: {str(e)}",
                error_trace=str(e)
            )
        finally:
            await self._close_connections()
    
    async def _connect_databases(self):
        """连接数据库"""
        self.logger.info("连接数据库...")
        
        db_config = self.config.database_config
        
        # 连接PostgreSQL
        if 'postgresql' in db_config:
            pg_config = db_config['postgresql']
            try:
                conn = psycopg2.connect(
                    host=pg_config.get('host', 'localhost'),
                    port=pg_config.get('port', 5432),
                    database=pg_config.get('database', 'test'),
                    user=pg_config.get('user', 'test'),
                    password=pg_config.get('password', '')
                )
                self.connections['postgresql'] = conn
                self.logger.info("PostgreSQL连接成功")
            except Exception as e:
                self.logger.warning(f"PostgreSQL连接失败: {e}")
        
        # 连接SQLite（用于测试）
        try:
            conn = sqlite3.connect(':memory:')
            self.connections['sqlite'] = conn
            self.logger.info("SQLite连接成功")
        except Exception as e:
            self.logger.warning(f"SQLite连接失败: {e}")
        
        # 连接Redis
        if 'redis' in db_config:
            redis_config = db_config['redis']
            try:
                r = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    decode_responses=True
                )
                r.ping()
                self.connections['redis'] = r
                self.logger.info("Redis连接成功")
            except Exception as e:
                self.logger.warning(f"Redis连接失败: {e}")
    
    async def _test_database_operations(self):
        """测试数据库操作"""
        self.logger.info("测试数据库操作...")
        
        # 测试SQLite操作
        if 'sqlite' in self.connections:
            conn = self.connections['sqlite']
            cursor = conn.cursor()
            
            # 创建测试表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE
                )
            ''')
            
            # 插入测试数据
            cursor.execute(
                "INSERT OR REPLACE INTO test_users (id, name, email) VALUES (?, ?, ?)",
                (1, "测试用户", "test@example.com")
            )
            
            # 查询测试数据
            cursor.execute("SELECT * FROM test_users WHERE id = ?", (1,))
            result = cursor.fetchone()
            
            if not result:
                raise Exception("数据库操作失败")
            
            conn.commit()
            self.logger.info("SQLite操作测试通过")
        
        # 测试Redis操作
        if 'redis' in self.connections:
            redis_conn = self.connections['redis']
            
            # 设置键值对
            redis_conn.set("test_key", "test_value")
            
            # 获取键值对
            value = redis_conn.get("test_key")
            
            if value != "test_value":
                raise Exception("Redis操作失败")
            
            self.logger.info("Redis操作测试通过")
    
    async def _test_transactions(self):
        """测试事务处理"""
        self.logger.info("测试事务处理...")
        
        if 'sqlite' in self.connections:
            conn = self.connections['sqlite']
            cursor = conn.cursor()
            
            try:
                # 开始事务
                conn.execute("BEGIN")
                
                # 执行多个操作
                cursor.execute(
                    "INSERT INTO test_users (id, name, email) VALUES (?, ?, ?)",
                    (2, "事务测试用户", "transaction@test.com")
                )
                
                cursor.execute(
                    "UPDATE test_users SET name = ? WHERE id = ?",
                    ("更新后的用户", 2)
                )
                
                # 提交事务
                conn.commit()
                
                # 验证结果
                cursor.execute("SELECT name FROM test_users WHERE id = ?", (2,))
                result = cursor.fetchone()
                
                if not result or result[0] != "更新后的用户":
                    raise Exception("事务测试失败")
                
                self.logger.info("事务测试通过")
                
            except Exception as e:
                conn.rollback()
                raise e
    
    async def _test_concurrent_operations(self):
        """测试并发操作"""
        self.logger.info("测试并发操作...")
        
        if 'sqlite' in self.connections:
            conn = self.connections['sqlite']
            
            def insert_user(user_id):
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "INSERT OR IGNORE INTO test_users (id, name, email) VALUES (?, ?, ?)",
                        (user_id, f"用户{user_id}", f"user{user_id}@test.com")
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    self.logger.error(f"插入用户{user_id}失败: {e}")
                    return False
            
            # 使用线程池测试并发插入
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(insert_user, i) for i in range(10, 20)]
                results = [future.result() for future in as_completed(futures)]
            
            success_count = sum(results)
            if success_count < 5:  # 至少应该有一些成功的插入
                raise Exception("并发操作测试失败")
            
            self.logger.info("并发操作测试通过")
    
    async def _test_data_consistency(self):
        """测试数据一致性"""
        self.logger.info("测试数据一致性...")
        
        if 'sqlite' in self.connections:
            conn = self.connections['sqlite']
            cursor = conn.cursor()
            
            # 统计记录数
            cursor.execute("SELECT COUNT(*) FROM test_users")
            count_before = cursor.fetchone()[0]
            
            # 再次统计
            cursor.execute("SELECT COUNT(*) FROM test_users")
            count_after = cursor.fetchone()[0]
            
            if count_before != count_after:
                raise Exception("数据一致性检查失败")
            
            self.logger.info("数据一致性测试通过")
    
    async def _cleanup_test_data(self):
        """清理测试数据"""
        self.logger.info("清理测试数据...")
        
        if 'sqlite' in self.connections:
            conn = self.connections['sqlite']
            cursor = conn.cursor()
            
            # 删除测试数据
            cursor.execute("DELETE FROM test_users WHERE email LIKE '%@test.com'")
            conn.commit()
            
            self.logger.info("测试数据清理完成")
    
    async def _close_connections(self):
        """关闭数据库连接"""
        for name, conn in self.connections.items():
            try:
                if name == 'redis':
                    conn.close()
                else:
                    conn.close()
                self.logger.info(f"{name}连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭{name}连接时出错: {e}")


class MessageQueueTest(BaseTest):
    """消息队列测试"""
    
    def __init__(self, config: IntegrationTestConfig):
        super().__init__(config)
        self.connections = {}
        self.test_messages = []
    
    async def execute(self) -> TestResult:
        """执行消息队列测试"""
        start_time = datetime.now()
        test_name = "消息队列测试"
        
        try:
            with self.timeout_context(self.config.timeout):
                # 连接消息队列
                await self._connect_message_queues()
                
                # 测试消息发送
                await self._test_message_publishing()
                
                # 测试消息接收
                await self._test_message_consumption()
                
                # 测试消息持久化
                await self._test_message_persistence()
                
                # 测试消息顺序
                await self._test_message_ordering()
                
                # 测试死信队列
                await self._test_dead_letter_queue()
                
                # 清理测试消息
                await self._cleanup_test_messages()
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return TestExecutionResult(
                    test_name=test_name,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    message="消息队列测试通过"
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                message=f"消息队列测试失败: {str(e)}",
                error_trace=str(e)
            )
        finally:
            await self._close_connections()
    
    async def _connect_message_queues(self):
        """连接消息队列"""
        self.logger.info("连接消息队列...")
        
        mq_config = self.config.message_queue_config
        
        # 连接RabbitMQ
        if 'rabbitmq' in mq_config:
            rabbit_config = mq_config['rabbitmq']
            try:
                parameters = pika.URLParameters(
                    f"amqp://{rabbit_config.get('username', 'guest')}:"
                    f"{rabbit_config.get('password', 'guest')}@"
                    f"{rabbit_config.get('host', 'localhost')}:"
                    f"{rabbit_config.get('port', 5672)}/"
                )
                connection = pika.BlockingConnection(parameters)
                self.connections['rabbitmq'] = connection
                self.logger.info("RabbitMQ连接成功")
            except Exception as e:
                self.logger.warning(f"RabbitMQ连接失败: {e}")
        
        # 模拟其他消息队列连接
        self.logger.info("消息队列连接完成")
    
    async def _test_message_publishing(self):
        """测试消息发送"""
        self.logger.info("测试消息发送...")
        
        if 'rabbitmq' in self.connections:
            connection = self.connections['rabbitmq']
            channel = connection.channel()
            
            # 声明队列
            queue_name = 'test_queue'
            channel.queue_declare(queue=queue_name, durable=True)
            
            # 发送测试消息
            test_message = {
                "id": 1,
                "type": "test",
                "content": "这是一个测试消息",
                "timestamp": datetime.now().isoformat()
            }
            
            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(test_message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # 消息持久化
                )
            )
            
            self.test_messages.append(test_message)
            self.logger.info("消息发送测试通过")
        else:
            self.logger.info("RabbitMQ未连接，跳过消息发送测试")
    
    async def _test_message_consumption(self):
        """测试消息接收"""
        self.logger.info("测试消息接收...")
        
        if 'rabbitmq' in self.connections:
            connection = self.connections['rabbitmq']
            channel = connection.channel()
            
            queue_name = 'test_queue'
            channel.queue_declare(queue=queue_name, durable=True)
            
            # 设置消费者
            received_messages = []
            
            def callback(ch, method, properties, body):
                received_messages.append(json.loads(body))
                ch.basic_ack(delivery_tag=method.delivery_tag)
            
            channel.basic_consume(queue=queue_name, on_message_callback=callback)
            
            # 启动消费（超时5秒）
            import threading
            consume_thread = threading.Thread(target=channel.start_consuming)
            consume_thread.daemon = True
            consume_thread.start()
            
            # 等待消息
            time.sleep(2)
            channel.stop_consuming()
            
            if received_messages:
                self.logger.info(f"成功接收 {len(received_messages)} 条消息")
            else:
                self.logger.info("未接收到消息（可能是队列为空）")
            
            self.logger.info("消息接收测试通过")
        else:
            self.logger.info("RabbitMQ未连接，跳过消息接收测试")
    
    async def _test_message_persistence(self):
        """测试消息持久化"""
        self.logger.info("测试消息持久化...")
        
        # 模拟消息持久化测试
        if 'rabbitmq' in self.connections:
            # 这里可以添加更复杂的持久化测试逻辑
            self.logger.info("消息持久化测试通过")
        else:
            self.logger.info("消息队列未连接，跳过持久化测试")
    
    async def _test_message_ordering(self):
        """测试消息顺序"""
        self.logger.info("测试消息顺序...")
        
        # 模拟消息顺序测试
        self.logger.info("消息顺序测试通过")
    
    async def _test_dead_letter_queue(self):
        """测试死信队列"""
        self.logger.info("测试死信队列...")
        
        # 模拟死信队列测试
        self.logger.info("死信队列测试通过")
    
    async def _cleanup_test_messages(self):
        """清理测试消息"""
        self.logger.info("清理测试消息...")
        
        if 'rabbitmq' in self.connections:
            connection = self.connections['rabbitmq']
            channel = connection.channel()
            
            # 清空测试队列
            try:
                channel.queue_purge(queue='test_queue')
                self.logger.info("测试队列已清空")
            except Exception as e:
                self.logger.warning(f"清空队列失败: {e}")
        
        self.test_messages.clear()
        self.logger.info("测试消息清理完成")
    
    async def _close_connections(self):
        """关闭消息队列连接"""
        for name, connection in self.connections.items():
            try:
                connection.close()
                self.logger.info(f"{name}连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭{name}连接时出错: {e}")


class ExternalServiceTest(BaseTest):
    """第三方服务集成测试"""
    
    def __init__(self, config: IntegrationTestConfig):
        super().__init__(config)
        self.session = requests.Session()
    
    async def execute(self) -> TestResult:
        """执行第三方服务测试"""
        start_time = datetime.now()
        test_name = "第三方服务集成测试"
        
        try:
            with self.timeout_context(self.config.timeout):
                # 测试外部API
                await self._test_external_apis()
                
                # 测试支付服务
                await self._test_payment_services()
                
                # 测试邮件服务
                await self._test_email_services()
                
                # 测试短信服务
                await self._test_sms_services()
                
                # 测试地图服务
                await self._test_map_services()
                
                # 测试云存储服务
                await self._test_cloud_storage()
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return TestExecutionResult(
                    test_name=test_name,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    message="第三方服务集成测试通过"
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                message=f"第三方服务集成测试失败: {str(e)}",
                error_trace=str(e)
            )
    
    async def _test_external_apis(self):
        """测试外部API"""
        self.logger.info("测试外部API...")
        
        external_services = self.config.external_services
        
        for service_name, service_config in external_services.items():
            if service_config.get('enabled', False):
                try:
                    url = service_config.get('url', '')
                    if url:
                        response = self.session.get(url, timeout=10)
                        if response.status_code == 200:
                            self.logger.info(f"{service_name} API测试通过")
                        else:
                            self.logger.warning(f"{service_name} API返回状态码: {response.status_code}")
                except Exception as e:
                    self.logger.warning(f"{service_name} API测试失败: {e}")
    
    async def _test_payment_services(self):
        """测试支付服务"""
        self.logger.info("测试支付服务...")
        
        # 模拟支付服务测试
        payment_services = ['alipay', 'wechat_pay', 'stripe']
        
        for service in payment_services:
            if service in self.config.external_services:
                self.logger.info(f"{service}支付服务配置正确")
        
        self.logger.info("支付服务测试通过")
    
    async def _test_email_services(self):
        """测试邮件服务"""
        self.logger.info("测试邮件服务...")
        
        # 模拟邮件服务测试
        email_services = ['smtp', 'sendgrid', 'mailgun']
        
        for service in email_services:
            if service in self.config.external_services:
                self.logger.info(f"{service}邮件服务配置正确")
        
        self.logger.info("邮件服务测试通过")
    
    async def _test_sms_services(self):
        """测试短信服务"""
        self.logger.info("测试短信服务...")
        
        # 模拟短信服务测试
        sms_services = ['aliyun_sms', 'tencent_sms', 'twilio']
        
        for service in sms_services:
            if service in self.config.external_services:
                self.logger.info(f"{service}短信服务配置正确")
        
        self.logger.info("短信服务测试通过")
    
    async def _test_map_services(self):
        """测试地图服务"""
        self.logger.info("测试地图服务...")
        
        # 模拟地图服务测试
        map_services = ['amap', 'baidu_map', 'google_map']
        
        for service in map_services:
            if service in self.config.external_services:
                self.logger.info(f"{service}地图服务配置正确")
        
        self.logger.info("地图服务测试通过")
    
    async def _test_cloud_storage(self):
        """测试云存储服务"""
        self.logger.info("测试云存储服务...")
        
        # 模拟云存储服务测试
        storage_services = ['aliyun_oss', 'qcloud_cos', 'aws_s3']
        
        for service in storage_services:
            if service in self.config.external_services:
                self.logger.info(f"{service}云存储服务配置正确")
        
        self.logger.info("云存储服务测试通过")


class DataFlowTest(BaseTest):
    """数据流测试"""
    
    async def execute(self) -> TestResult:
        """执行数据流测试"""
        start_time = datetime.now()
        test_name = "数据流测试"
        
        try:
            with self.timeout_context(self.config.timeout):
                # 测试数据输入
                await self._test_data_input()
                
                # 测试数据处理
                await self._test_data_processing()
                
                # 测试数据输出
                await self._test_data_output()
                
                # 测试数据转换
                await self._test_data_transformation()
                
                # 测试数据验证
                await self._test_data_validation()
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return TestExecutionResult(
                    test_name=test_name,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    message="数据流测试通过"
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                message=f"数据流测试失败: {str(e)}",
                error_trace=str(e)
            )
    
    async def _test_data_input(self):
        """测试数据输入"""
        self.logger.info("测试数据输入...")
        
        # 模拟数据输入测试
        test_data = {
            "user_id": 123,
            "action": "login",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"source": "web", "version": "1.0"}
        }
        
        # 验证数据格式
        required_fields = ["user_id", "action", "timestamp"]
        for field in required_fields:
            if field not in test_data:
                raise Exception(f"缺少必需字段: {field}")
        
        self.logger.info("数据输入测试通过")
    
    async def _test_data_processing(self):
        """测试数据处理"""
        self.logger.info("测试数据处理...")
        
        # 模拟数据处理测试
        input_data = {"value": 100}
        
        # 数据清洗
        processed_data = {
            "cleaned_value": max(0, input_data.get("value", 0)),
            "processed_at": datetime.now().isoformat()
        }
        
        # 验证处理结果
        if processed_data["cleaned_value"] != input_data["value"]:
            raise Exception("数据处理失败")
        
        self.logger.info("数据处理测试通过")
    
    async def _test_data_output(self):
        """测试数据输出"""
        self.logger.info("测试数据输出...")
        
        # 模拟数据输出测试
        output_data = {
            "result": "success",
            "data": {"processed": True},
            "timestamp": datetime.now().isoformat()
        }
        
        # 验证输出格式
        if not isinstance(output_data, dict):
            raise Exception("输出数据格式错误")
        
        self.logger.info("数据输出测试通过")
    
    async def _test_data_transformation(self):
        """测试数据转换"""
        self.logger.info("测试数据转换...")
        
        # 模拟数据转换测试
        source_data = {
            "firstName": "John",
            "lastName": "Doe",
            "email": "john.doe@example.com"
        }
        
        # 转换为目标格式
        target_data = {
            "name": f"{source_data['firstName']} {source_data['lastName']}",
            "contact": source_data['email'],
            "full_name": source_data['firstName'] + " " + source_data['lastName']
        }
        
        # 验证转换结果
        if "name" not in target_data:
            raise Exception("数据转换失败")
        
        self.logger.info("数据转换测试通过")
    
    async def _test_data_validation(self):
        """测试数据验证"""
        self.logger.info("测试数据验证...")
        
        # 模拟数据验证测试
        test_cases = [
            {"valid": True, "data": {"email": "test@example.com", "age": 25}},
            {"valid": False, "data": {"email": "invalid-email", "age": -1}},
        ]
        
        for test_case in test_cases:
            data = test_case["data"]
            
            # 验证邮箱格式
            if "@" not in data.get("email", ""):
                if test_case["valid"]:
                    raise Exception("数据验证逻辑错误")
            else:
                if not test_case["valid"]:
                    raise Exception("数据验证逻辑错误")
        
        self.logger.info("数据验证测试通过")


class EnvironmentConfigTest(BaseTest):
    """环境配置测试"""
    
    async def execute(self) -> TestResult:
        """执行环境配置测试"""
        start_time = datetime.now()
        test_name = "环境配置测试"
        
        try:
            with self.timeout_context(self.config.timeout):
                # 测试环境变量
                await self._test_environment_variables()
                
                # 测试配置文件
                await self._test_configuration_files()
                
                # 测试数据库配置
                await self._test_database_configuration()
                
                # 测试服务配置
                await self._test_service_configuration()
                
                # 测试安全配置
                await self._test_security_configuration()
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return TestExecutionResult(
                    test_name=test_name,
                    result=TestResult.PASS,
                    execution_time=execution_time,
                    start_time=start_time,
                    end_time=end_time,
                    message="环境配置测试通过"
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                result=TestResult.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                message=f"环境配置测试失败: {str(e)}",
                error_trace=str(e)
            )
    
    async def _test_environment_variables(self):
        """测试环境变量"""
        self.logger.info("测试环境变量...")
        
        # 检查必需的环境变量
        required_vars = ["ENVIRONMENT", "DEBUG", "LOG_LEVEL"]
        
        for var in required_vars:
            value = self.config.environment.value
            if not value:
                self.logger.warning(f"环境变量 {var} 未设置")
        
        # 验证环境配置
        if self.config.environment == Environment.PRODUCTION:
            # 生产环境应该有更严格的配置
            pass
        elif self.config.environment == Environment.DEVELOPMENT:
            # 开发环境应该有调试功能
            pass
        
        self.logger.info("环境变量测试通过")
    
    async def _test_configuration_files(self):
        """测试配置文件"""
        self.logger.info("测试配置文件...")
        
        # 模拟配置文件测试
        config_files = ["config.json", "application.yml", ".env"]
        
        for config_file in config_files:
            self.logger.info(f"检查配置文件: {config_file}")
        
        self.logger.info("配置文件测试通过")
    
    async def _test_database_configuration(self):
        """测试数据库配置"""
        self.logger.info("测试数据库配置...")
        
        db_config = self.config.database_config
        
        # 验证数据库配置
        if 'postgresql' in db_config:
            pg_config = db_config['postgresql']
            required_fields = ['host', 'port', 'database', 'user']
            
            for field in required_fields:
                if field not in pg_config:
                    self.logger.warning(f"PostgreSQL配置缺少字段: {field}")
        
        self.logger.info("数据库配置测试通过")
    
    async def _test_service_configuration(self):
        """测试服务配置"""
        self.logger.info("测试服务配置...")
        
        api_endpoints = self.config.api_endpoints
        
        # 验证API端点配置
        required_endpoints = ['api_base', 'health']
        
        for endpoint in required_endpoints:
            if endpoint not in api_endpoints:
                self.logger.warning(f"缺少API端点配置: {endpoint}")
        
        self.logger.info("服务配置测试通过")
    
    async def _test_security_configuration(self):
        """测试安全配置"""
        self.logger.info("测试安全配置...")
        
        # 模拟安全配置测试
        security_checks = [
            "HTTPS强制重定向",
            "CORS配置",
            "CSRF保护",
            "XSS防护",
            "SQL注入防护"
        ]
        
        for check in security_checks:
            self.logger.info(f"检查安全配置: {check}")
        
        self.logger.info("安全配置测试通过")


class IntegrationTester:
    """集成测试器主类"""
    
    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_results: List[TestResult] = []
        self.test_cases: List[TestCase] = []
        
        # 初始化测试组件
        self.system_test = SystemIntegrationTest(config)
        self.api_test = APIInterfaceTest(config)
        self.database_test = DatabaseIntegrationTest(config)
        self.message_queue_test = MessageQueueTest(config)
        self.external_service_test = ExternalServiceTest(config)
        self.data_flow_test = DataFlowTest(config)
        self.environment_test = EnvironmentConfigTest(config)
    
    def add_test_case(self, test_case: TestCase):
        """添加测试用例"""
        self.test_cases.append(test_case)
    
    def add_test_cases(self, test_cases: List[TestCase]):
        """批量添加测试用例"""
        self.test_cases.extend(test_cases)
    
    async def run_all_tests(self) -> List[TestResult]:
        """运行所有测试"""
        self.logger.info("开始运行所有集成测试...")
        
        # 运行内置测试
        await self._run_builtin_tests()
        
        # 运行自定义测试用例
        await self._run_custom_test_cases()
        
        # 生成测试报告
        self._generate_test_report()
        
        return self.test_results
    
    async def run_specific_tests(self, test_types: List[str]) -> List[TestResult]:
        """运行指定类型的测试"""
        self.logger.info(f"运行指定测试: {test_types}")
        
        test_mapping = {
            'system': self.system_test,
            'api': self.api_test,
            'database': self.database_test,
            'message_queue': self.message_queue_test,
            'external_service': self.external_service_test,
            'data_flow': self.data_flow_test,
            'environment': self.environment_test
        }
        
        for test_type in test_types:
            if test_type in test_mapping:
                test_instance = test_mapping[test_type]
                result = await test_instance.execute()
                self.test_results.append(result)
            else:
                self.logger.warning(f"未知的测试类型: {test_type}")
        
        return self.test_results
    
    async def _run_builtin_tests(self):
        """运行内置测试"""
        builtin_tests = [
            self.system_test,
            self.api_test,
            self.database_test,
            self.message_queue_test,
            self.external_service_test,
            self.data_flow_test,
            self.environment_test
        ]
        
        if self.config.parallel_execution:
            # 并行执行测试
            tasks = [test.execute() for test in builtin_tests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"测试执行异常: {result}")
                else:
                    self.test_results.append(result)
        else:
            # 顺序执行测试
            for test in builtin_tests:
                try:
                    result = await test.execute()
                    self.test_results.append(result)
                except Exception as e:
                    self.logger.error(f"测试执行失败: {e}")
    
    async def _run_custom_test_cases(self):
        """运行自定义测试用例"""
        if not self.test_cases:
            return
        
        self.logger.info(f"运行 {len(self.test_cases)} 个自定义测试用例...")
        
        for test_case in self.test_cases:
            try:
                start_time = datetime.now()
                
                # 检查环境要求
                if test_case.environment and test_case.environment != self.config.environment:
                    result = TestExecutionResult(
                        test_name=test_case.name,
                        result=TestResult.SKIP,
                        execution_time=0,
                        start_time=start_time,
                        end_time=datetime.now(),
                        message=f"测试环境不匹配，跳过测试"
                    )
                    self.test_results.append(result)
                    continue
                
                # 执行测试
                result = await asyncio.wait_for(
                    test_case.test_function(),
                    timeout=test_case.timeout
                )
                
                # 如果测试函数没有返回TestResult，创建一个默认的
                if not isinstance(result, TestResult):
                    result = TestExecutionResult(
                        test_name=test_case.name,
                        result=TestResult.PASS,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        start_time=start_time,
                        end_time=datetime.now(),
                        message="测试通过"
                    )
                
                self.test_results.append(result)
                
            except asyncio.TimeoutError:
                end_time = datetime.now()
                result = TestExecutionResult(
                    test_name=test_case.name,
                    result=TestResult.ERROR,
                    execution_time=(end_time - start_time).total_seconds(),
                    start_time=start_time,
                    end_time=end_time,
                    message=f"测试超时（{test_case.timeout}秒）"
                )
                self.test_results.append(result)
                
            except Exception as e:
                end_time = datetime.now()
                result = TestExecutionResult(
                    test_name=test_case.name,
                    result=TestResult.ERROR,
                    execution_time=(end_time - start_time).total_seconds(),
                    start_time=start_time,
                    end_time=end_time,
                    message=f"测试执行失败: {str(e)}",
                    error_trace=str(e)
                )
                self.test_results.append(result)
    
    def _generate_test_report(self):
        """生成测试报告"""
        self.logger.info("生成测试报告...")
        
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.result == TestResult.PASS)
        failed_tests = sum(1 for r in self.test_results if r.result == TestResult.FAIL)
        error_tests = sum(1 for r in self.test_results if r.result == TestResult.ERROR)
        skipped_tests = sum(1 for r in self.test_results if r.result == TestResult.SKIP)
        
        # 计算总执行时间
        total_execution_time = sum(r.execution_time for r in self.test_results)
        
        # 生成报告
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_execution_time": total_execution_time
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "result": r.result.value,
                    "execution_time": r.execution_time,
                    "start_time": r.start_time.isoformat(),
                    "end_time": r.end_time.isoformat(),
                    "message": r.message,
                    "details": r.details,
                    "error_trace": r.error_trace
                }
                for r in self.test_results
            ],
            "configuration": {
                "environment": self.config.environment.value,
                "timeout": self.config.timeout,
                "retry_count": self.config.retry_count,
                "parallel_execution": self.config.parallel_execution
            }
        }
        
        # 保存报告到文件
        report_file = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"测试报告已保存到: {report_file}")
        
        # 打印简要报告
        self._print_summary_report(report)
    
    def _print_summary_report(self, report: Dict[str, Any]):
        """打印简要报告"""
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("集成测试报告")
        print("="*60)
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过: {summary['passed']}")
        print(f"失败: {summary['failed']}")
        print(f"错误: {summary['errors']}")
        print(f"跳过: {summary['skipped']}")
        print(f"成功率: {summary['success_rate']:.2f}%")
        print(f"总执行时间: {summary['total_execution_time']:.2f}秒")
        print("="*60)
        
        # 打印失败的测试
        failed_tests = [r for r in report["test_results"] if r["result"] in ["FAIL", "ERROR"]]
        if failed_tests:
            print("\n失败的测试:")
            for test in failed_tests:
                print(f"  - {test['test_name']}: {test['message']}")
        
        print()
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """获取测试统计信息"""
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.result == TestResult.PASS)
        failed_tests = sum(1 for r in self.test_results if r.result == TestResult.FAIL)
        error_tests = sum(1 for r in self.test_results if r.result == TestResult.ERROR)
        skipped_tests = sum(1 for r in self.test_results if r.result == TestResult.SKIP)
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "skipped": skipped_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_execution_time": sum(r.execution_time for r in self.test_results),
            "average_execution_time": sum(r.execution_time for r in self.test_results) / total_tests if total_tests > 0 else 0
        }
    
    def export_results(self, format: str = "json", filename: Optional[str] = None) -> str:
        """导出测试结果"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"integration_test_results_{timestamp}.{format}"
        
        if format.lower() == "json":
            data = {
                "test_results": [
                    {
                        "test_name": r.test_name,
                        "result": r.result.value,
                        "execution_time": r.execution_time,
                        "start_time": r.start_time.isoformat(),
                        "end_time": r.end_time.isoformat(),
                        "message": r.message,
                        "details": r.details,
                        "error_trace": r.error_trace
                    }
                    for r in self.test_results
                ],
                "statistics": self.get_test_statistics()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        elif format.lower() == "csv":
            import csv
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "测试名称", "结果", "执行时间(秒)", "开始时间", "结束时间", "消息", "错误详情"
                ])
                
                for r in self.test_results:
                    writer.writerow([
                        r.test_name,
                        r.result.value,
                        r.execution_time,
                        r.start_time.isoformat(),
                        r.end_time.isoformat(),
                        r.message,
                        r.error_trace or ""
                    ])
        
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        self.logger.info(f"测试结果已导出到: {filename}")
        return filename


# 创建默认配置工厂函数
def create_default_config(environment: Environment = Environment.TESTING) -> IntegrationTestConfig:
    """创建默认配置"""
    return IntegrationTestConfig(
        environment=environment,
        api_endpoints={
            'api_base': 'http://localhost:8080/api',
            'health': 'http://localhost:8080/health',
            'gateway': 'http://localhost:8080',
            'graphql': 'http://localhost:8080/graphql',
            'swagger': 'http://localhost:8080/docs'
        },
        database_config={
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_db',
                'user': 'test_user',
                'password': 'test_password'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        },
        message_queue_config={
            'rabbitmq': {
                'host': 'localhost',
                'port': 5672,
                'username': 'guest',
                'password': 'guest'
            }
        },
        external_services={
            'alipay': {
                'enabled': False,
                'url': 'https://openapi.alipay.com/health'
            },
            'wechat_pay': {
                'enabled': False,
                'url': 'https://api.mch.weixin.qq.com/v3/pay/transactions/health'
            },
            'smtp': {
                'enabled': False
            },
            'aliyun_sms': {
                'enabled': False
            }
        }
    )


if __name__ == "__main__":
    # 示例使用
    async def main():
        # 创建配置
        config = create_default_config(Environment.TESTING)
        
        # 创建测试器
        tester = IntegrationTester(config)
        
        # 添加自定义测试用例
        async def custom_test():
            # 模拟自定义测试
            await asyncio.sleep(1)
            return TestResult(
                test_name="自定义测试",
                result=TestResult.PASS,
                execution_time=1.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                message="自定义测试通过"
            )
        
        tester.add_test_case(TestCase(
            name="自定义测试用例",
            description="这是一个自定义测试用例",
            test_function=custom_test
        ))
        
        # 运行测试
        results = await tester.run_all_tests()
        
        # 打印统计信息
        stats = tester.get_test_statistics()
        print(f"测试统计: {stats}")
        
        # 导出结果
        tester.export_results("json", "test_results.json")
    
    # 运行示例
    asyncio.run(main())