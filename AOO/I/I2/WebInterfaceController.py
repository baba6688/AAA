"""
Web接口控制器模块

提供完整的Web接口控制功能，包括：
- WebSocket连接管理
- HTTP请求处理
- 实时数据推送
- 会话管理
- 跨域支持
- 文件上传处理
- 静态资源服务
- 负载均衡支持
- Web安全防护


版本: 1.0.0
创建时间: 2025-11-05
"""

import asyncio
import json
import logging
import os
import hashlib
import hmac
import time
import uuid
import weakref
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Set
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass, field
from collections import defaultdict, deque
import mimetypes
import base64
import ssl

# Web框架相关导入
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, UploadFile, File
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.exceptions import RequestValidationError
    import uvicorn
    FASTRPI_AVAILABLE = True
except ImportError:
    FASTRPI_AVAILABLE = False

try:
    import websockets
    import aiofiles
    import aiohttp
    from aiohttp import web, WSMsgType
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# 安全相关导入
try:
    import jwt
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# 配置类
@dataclass
class WebConfig:
    """Web接口控制器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_connections: int = 1000
    connection_timeout: int = 30
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    static_dir: str = "static"
    upload_dir: str = "uploads"
    session_timeout: int = 3600  # 1小时
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 1分钟
    enable_ssl: bool = False
    ssl_cert: str = ""
    ssl_key: str = ""
    jwt_secret: str = "default-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    enable_csrf: bool = True
    enable_rate_limiting: bool = True
    enable_logging: bool = True


# 数据模型
@dataclass
class ConnectionInfo:
    """连接信息"""
    connection_id: str
    client_id: str
    websocket: Optional[WebSocket] = None
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    user_agent: str = ""
    ip_address: str = ""
    session_data: Dict[str, Any] = field(default_factory=dict)
    is_authenticated: bool = False
    permissions: Set[str] = field(default_factory=set)


@dataclass
class SessionData:
    """会话数据"""
    session_id: str
    user_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    is_active: bool = True


@dataclass
class RateLimitEntry:
    """速率限制条目"""
    count: int = 0
    window_start: float = field(default_factory=time.time)


class SecurityManager:
    """安全管理器"""
    
    def __init__(self, config: WebConfig):
        self.config = config
        self.blocked_ips: Set[str] = set()
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.blocked_until: Dict[str, float] = {}
    
    def validate_request(self, request: Request) -> bool:
        """验证请求安全性"""
        client_ip = self.get_client_ip(request)
        
        # 检查IP是否被阻止
        if client_ip in self.blocked_ips:
            return False
        
        # 检查失败尝试次数
        if client_ip in self.failed_attempts:
            if self.failed_attempts[client_ip] >= 5:  # 5次失败后阻止
                if client_ip not in self.blocked_until:
                    self.blocked_until[client_ip] = time.time() + 3600  # 阻止1小时
                    self.blocked_ips.add(client_ip)
                return False
        
        return True
    
    def get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """检查速率限制"""
        if not self.config.enable_rate_limiting:
            return True
        
        current_time = time.time()
        
        # 这里应该使用Redis或内存缓存来存储速率限制数据
        # 为了简化，这里使用内存存储
        if not hasattr(self, '_rate_limits'):
            self._rate_limits = {}
        
        if client_ip not in self._rate_limits:
            self._rate_limits[client_ip] = RateLimitEntry()
        
        entry = self._rate_limits[client_ip]
        
        # 重置窗口
        if current_time - entry.window_start >= self.config.rate_limit_window:
            entry.count = 0
            entry.window_start = current_time
        
        # 检查限制
        if entry.count >= self.config.rate_limit_requests:
            return False
        
        entry.count += 1
        return True
    
    def validate_csrf_token(self, request: Request) -> bool:
        """验证CSRF令牌"""
        if not self.config.enable_csrf:
            return True
        
        # 简化实现，实际应该验证令牌
        return True
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """哈希密码"""
        if salt is None:
            salt = os.urandom(32)
        
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return pwdhash, salt
    
    def verify_password(self, password: str, hashed: bytes, salt: bytes) -> bool:
        """验证密码"""
        return hmac.compare_digest(
            hashed,
            hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        )


class SessionManager:
    """会话管理器"""
    
    def __init__(self, config: WebConfig):
        self.config = config
        self.sessions: Dict[str, SessionData] = {}
        self.session_locks = defaultdict(asyncio.Lock)
    
    async def create_session(self, user_id: Optional[str] = None) -> SessionData:
        """创建会话"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(seconds=self.config.session_timeout)
        
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at
        )
        
        self.sessions[session_id] = session
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session and session.is_active and datetime.now() < session.expires_at:
            session.last_accessed = datetime.now()
            return session
        elif session:
            await self.delete_session(session_id)
        return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """更新会话数据"""
        session = await self.get_session(session_id)
        if session:
            session.data.update(data)
            session.last_accessed = datetime.now()
            return True
        return False
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if not session.is_active or session.expires_at <= current_time
        ]
        
        for session_id in expired_sessions:
            await self.delete_session(session_id)


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.servers = []
        self.current_index = 0
        self.connection_counts = defaultdict(int)
    
    def add_server(self, host: str, port: int, weight: int = 1):
        """添加服务器"""
        self.servers.append({
            'host': host,
            'port': port,
            'weight': weight,
            'active': True
        })
    
    def get_next_server(self) -> Optional[Dict[str, Any]]:
        """获取下一个服务器（轮询算法）"""
        active_servers = [s for s in self.servers if s['active']]
        if not active_servers:
            return None
        
        server = active_servers[self.current_index % len(active_servers)]
        self.current_index += 1
        return server
    
    def get_least_connections_server(self) -> Optional[Dict[str, Any]]:
        """获取最少连接的服务器"""
        active_servers = [s for s in self.servers if s['active']]
        if not active_servers:
            return None
        
        return min(active_servers, key=lambda s: self.connection_counts.get(f"{s['host']}:{s['port']}", 0))
    
    def update_connection_count(self, server_key: str, delta: int):
        """更新连接计数"""
        self.connection_counts[server_key] += delta
        if self.connection_counts[server_key] < 0:
            self.connection_counts[server_key] = 0


class WebInterfaceController:
    """
    Web接口控制器
    
    提供完整的Web接口控制功能，包括WebSocket连接管理、HTTP请求处理、
    实时数据推送、会话管理、跨域支持、文件上传处理、静态资源服务、
    负载均衡支持和Web安全防护。
    """
    
    def __init__(self, config: WebConfig = None):
        """
        初始化Web接口控制器
        
        Args:
            config: Web配置对象
        """
        self.config = config or WebConfig()
        self.logger = self._setup_logging()
        
        # 核心组件
        self.security_manager = SecurityManager(self.config)
        self.session_manager = SessionManager(self.config)
        self.load_balancer = LoadBalancer()
        
        # 连接管理
        self.connections: Dict[str, ConnectionInfo] = {}
        self.connection_locks = defaultdict(asyncio.Lock)
        
        # 数据推送
        self.subscribers: Dict[str, Set[str]] = defaultdict(set)  # topic -> connection_ids
        self.data_queue = asyncio.Queue()
        
        # 统计信息
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_requests': 0,
            'total_websockets': 0,
            'files_uploaded': 0,
            'start_time': datetime.now()
        }
        
        # Web框架实例
        self.app = None
        self.websocket_app = None
        
        # 初始化
        self._initialize_directories()
        self._setup_frameworks()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        if not self.config.enable_logging:
            return logging.getLogger('dummy')
        
        logger = logging.getLogger('WebInterfaceController')
        logger.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_directories(self):
        """初始化目录"""
        Path(self.config.static_dir).mkdir(exist_ok=True)
        Path(self.config.upload_dir).mkdir(exist_ok=True)
    
    def _setup_frameworks(self):
        """设置Web框架"""
        if FASTRPI_AVAILABLE:
            self._setup_fastapi()
        elif AIOHTTP_AVAILABLE:
            self._setup_aiohttp()
        else:
            self.logger.warning("未安装FastAPI或aiohttp，使用基础HTTP服务器")
    
    def _setup_fastapi(self):
        """设置FastAPI框架"""
        self.app = FastAPI(
            title="Web接口控制器",
            description="提供WebSocket连接管理、HTTP请求处理、实时数据推送等功能",
            version="1.0.0",
            debug=self.config.debug
        )
        
        # 添加CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 注册路由
        self._register_fastapi_routes()
    
    def _setup_aiohttp(self):
        """设置aiohttp框架"""
        self.websocket_app = web.Application()
        self.websocket_app.router.add_get('/ws', self._websocket_handler)
        self.websocket_app.router.add_post('/upload', self._upload_handler)
        self.websocket_app.router.add_get('/static/{path:path}', self._static_handler)
    
    def _register_fastapi_routes(self):
        """注册FastAPI路由"""
        if not self.app:
            return
        
        # HTTP路由
        @self.app.get("/")
        async def root():
            return {"message": "Web接口控制器运行中", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats
            }
        
        @self.app.get("/stats")
        async def get_stats():
            return self.stats
        
        @self.app.post("/login")
        async def login(request: Request):
            """用户登录"""
            try:
                data = await request.json()
                username = data.get("username")
                password = data.get("password")
                
                # 简化认证逻辑
                if username == "admin" and password == "admin":
                    session = await self.session_manager.create_session(user_id=username)
                    return {
                        "success": True,
                        "session_id": session.session_id,
                        "message": "登录成功"
                    }
                else:
                    return {"success": False, "message": "用户名或密码错误"}
            
            except Exception as e:
                self.logger.error(f"登录错误: {e}")
                raise HTTPException(status_code=400, detail="登录失败")
        
        @self.app.post("/logout")
        async def logout(request: Request):
            """用户登出"""
            try:
                data = await request.json()
                session_id = data.get("session_id")
                
                if await self.session_manager.delete_session(session_id):
                    return {"success": True, "message": "登出成功"}
                else:
                    return {"success": False, "message": "会话不存在"}
            
            except Exception as e:
                self.logger.error(f"登出错误: {e}")
                raise HTTPException(status_code=400, detail="登出失败")
        
        @self.app.get("/session/{session_id}")
        async def get_session_info(session_id: str):
            """获取会话信息"""
            session = await self.session_manager.get_session(session_id)
            if session:
                return {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "created_at": session.created_at.isoformat(),
                    "last_accessed": session.last_accessed.isoformat(),
                    "expires_at": session.expires_at.isoformat(),
                    "data": session.data
                }
            else:
                raise HTTPException(status_code=404, detail="会话不存在")
        
        @self.app.post("/upload")
        async def upload_file(file: UploadFile = File(...)):
            """文件上传"""
            try:
                # 检查文件大小
                content = await file.read()
                if len(content) > self.config.max_file_size:
                    raise HTTPException(status_code=413, detail="文件过大")
                
                # 保存文件
                file_path = Path(self.config.upload_dir) / file.filename
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content)
                
                self.stats['files_uploaded'] += 1
                self.logger.info(f"文件上传成功: {file.filename}")
                
                return {
                    "success": True,
                    "filename": file.filename,
                    "size": len(content),
                    "message": "文件上传成功"
                }
            
            except Exception as e:
                self.logger.error(f"文件上传错误: {e}")
                raise HTTPException(status_code=500, detail="文件上传失败")
        
        @self.app.get("/download/{filename}")
        async def download_file(filename: str):
            """文件下载"""
            file_path = Path(self.config.upload_dir) / filename
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="文件不存在")
            
            return FileResponse(
                path=file_path,
                filename=filename,
                media_type='application/octet-stream'
            )
        
        @self.app.get("/static/{path:path}")
        async def serve_static(path: str):
            """静态文件服务"""
            file_path = Path(self.config.static_dir) / path
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="文件不存在")
            
            return FileResponse(path=file_path)
        
        # WebSocket路由
        @self.app.websocket("/ws/{connection_id}")
        async def websocket_endpoint(websocket: WebSocket, connection_id: str):
            """WebSocket连接端点"""
            await self._handle_websocket_connection(websocket, connection_id)
    
    async def _handle_websocket_connection(self, websocket: WebSocket, connection_id: str):
        """处理WebSocket连接"""
        try:
            await websocket.accept()
            
            # 创建连接信息
            connection_info = ConnectionInfo(
                connection_id=connection_id,
                client_id=connection_id,
                websocket=websocket,
                ip_address=websocket.client.host if websocket.client else "unknown"
            )
            
            # 注册连接
            async with self.connection_locks[connection_id]:
                self.connections[connection_id] = connection_info
                self.stats['active_connections'] += 1
                self.stats['total_websockets'] += 1
            
            self.logger.info(f"WebSocket连接建立: {connection_id}")
            
            # 发送连接确认
            await websocket.send_json({
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # 处理消息
            while True:
                try:
                    # 接收消息
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    # 更新最后活动时间
                    connection_info.last_activity = datetime.now()
                    
                    # 处理不同类型的消息
                    await self._process_websocket_message(connection_id, message_data)
                
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "无效的JSON格式"
                    })
                except Exception as e:
                    self.logger.error(f"WebSocket消息处理错误: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "消息处理失败"
                    })
        
        except Exception as e:
            self.logger.error(f"WebSocket连接错误: {e}")
        
        finally:
            # 清理连接
            await self._cleanup_connection(connection_id)
    
    async def _process_websocket_message(self, connection_id: str, message_data: Dict[str, Any]):
        """处理WebSocket消息"""
        message_type = message_data.get("type")
        
        if message_type == "ping":
            # 心跳检测
            await self._send_to_connection(connection_id, {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            })
        
        elif message_type == "subscribe":
            # 订阅主题
            topic = message_data.get("topic")
            if topic:
                self.subscribers[topic].add(connection_id)
                await self._send_to_connection(connection_id, {
                    "type": "subscribed",
                    "topic": topic
                })
        
        elif message_type == "unsubscribe":
            # 取消订阅
            topic = message_data.get("topic")
            if topic:
                self.subscribers[topic].discard(connection_id)
                await self._send_to_connection(connection_id, {
                    "type": "unsubscribed",
                    "topic": topic
                })
        
        elif message_type == "broadcast":
            # 广播消息
            topic = message_data.get("topic")
            data = message_data.get("data")
            if topic and data:
                await self.broadcast_to_topic(topic, data)
        
        elif message_type == "get_stats":
            # 获取统计信息
            await self._send_to_connection(connection_id, {
                "type": "stats",
                "data": self.stats
            })
        
        else:
            # 未知消息类型
            await self._send_to_connection(connection_id, {
                "type": "error",
                "message": f"未知消息类型: {message_type}"
            })
    
    async def _send_to_connection(self, connection_id: str, data: Dict[str, Any]):
        """发送数据到指定连接"""
        connection_info = self.connections.get(connection_id)
        if connection_info and connection_info.websocket:
            try:
                await connection_info.websocket.send_json(data)
            except Exception as e:
                self.logger.error(f"发送消息到连接 {connection_id} 失败: {e}")
                await self._cleanup_connection(connection_id)
    
    async def broadcast_to_topic(self, topic: str, data: Any):
        """向主题的所有订阅者广播数据"""
        message = {
            "type": "broadcast",
            "topic": topic,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected_connections = []
        
        for connection_id in self.subscribers[topic]:
            try:
                await self._send_to_connection(connection_id, message)
            except Exception as e:
                self.logger.error(f"广播到连接 {connection_id} 失败: {e}")
                disconnected_connections.append(connection_id)
        
        # 清理断开的连接
        for connection_id in disconnected_connections:
            self.subscribers[topic].discard(connection_id)
            await self._cleanup_connection(connection_id)
    
    async def _cleanup_connection(self, connection_id: str):
        """清理连接"""
        async with self.connection_locks[connection_id]:
            if connection_id in self.connections:
                connection_info = self.connections[connection_id]
                
                # 从所有主题中移除
                for topic_subscribers in self.subscribers.values():
                    topic_subscribers.discard(connection_id)
                
                # 关闭WebSocket
                if connection_info.websocket:
                    try:
                        await connection_info.websocket.close()
                    except:
                        pass
                
                # 移除连接
                del self.connections[connection_id]
                self.stats['active_connections'] -= 1
                
                self.logger.info(f"WebSocket连接清理: {connection_id}")
    
    async def _websocket_handler(self, request):
        """aiohttp WebSocket处理器"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        connection_id = str(uuid.uuid4())
        self.logger.info(f"WebSocket连接建立: {connection_id}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._process_websocket_message(connection_id, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            "type": "error",
                            "message": "无效的JSON格式"
                        }))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f"WebSocket错误: {ws.exception()}")
                    break
        except Exception as e:
            self.logger.error(f"WebSocket处理错误: {e}")
        finally:
            await self._cleanup_connection(connection_id)
        
        return ws
    
    async def _upload_handler(self, request):
        """文件上传处理器"""
        try:
            reader = await request.multipart()
            
            while True:
                part = await reader.next()
                if part is None:
                    break
                
                if part.name == 'file':
                    content = await part.read()
                    
                    if len(content) > self.config.max_file_size:
                        return web.json_response({
                            "success": False,
                            "message": "文件过大"
                        }, status=413)
                    
                    file_path = Path(self.config.upload_dir) / part.filename
                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(content)
                    
                    self.stats['files_uploaded'] += 1
                    
                    return web.json_response({
                        "success": True,
                        "filename": part.filename,
                        "size": len(content)
                    })
            
            return web.json_response({
                "success": False,
                "message": "未找到文件"
            }, status=400)
        
        except Exception as e:
            self.logger.error(f"文件上传错误: {e}")
            return web.json_response({
                "success": False,
                "message": "上传失败"
            }, status=500)
    
    async def _static_handler(self, request):
        """静态文件处理器"""
        try:
            path = request.match_info['path']
            file_path = Path(self.config.static_dir) / path
            
            if not file_path.exists():
                return web.json_response({
                    "error": "文件不存在"
                }, status=404)
            
            return web.FileResponse(file_path)
        
        except Exception as e:
            self.logger.error(f"静态文件服务错误: {e}")
            return web.json_response({
                "error": "服务错误"
            }, status=500)
    
    async def start_server(self):
        """启动服务器"""
        try:
            if FASTRPI_AVAILABLE and self.app:
                # 使用FastAPI
                config = uvicorn.Config(
                    self.app,
                    host=self.config.host,
                    port=self.config.port,
                    debug=self.config.debug,
                    ssl_keyfile=self.config.ssl_key if self.config.enable_ssl else None,
                    ssl_certfile=self.config.ssl_cert if self.config.enable_ssl else None
                )
                server = uvicorn.Server(config)
                await server.serve()
            
            elif AIOHTTP_AVAILABLE and self.websocket_app:
                # 使用aiohttp
                runner = web.AppRunner(self.websocket_app)
                await runner.setup()
                
                site = web.TCPSite(
                    runner,
                    self.config.host,
                    self.config.port,
                    ssl_context=ssl.create_default_context() if self.config.enable_ssl else None
                )
                await site.start()
                
                self.logger.info(f"服务器启动在 {self.config.host}:{self.config.port}")
                
                # 保持服务器运行
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    pass
                finally:
                    await runner.cleanup()
            
            else:
                # 基础HTTP服务器
                self.logger.warning("使用基础HTTP服务器，功能有限")
                await self._start_basic_server()
        
        except Exception as e:
            self.logger.error(f"服务器启动失败: {e}")
            raise
    
    async def _start_basic_server(self):
        """启动基础HTTP服务器"""
        import http.server
        import socketserver
        
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=self.config.static_dir, **kwargs)
        
        with socketserver.TCPServer((self.config.host, self.config.port), CustomHandler) as httpd:
            self.logger.info(f"基础HTTP服务器启动在 {self.config.host}:{self.config.port}")
            httpd.serve_forever()
    
    async def stop_server(self):
        """停止服务器"""
        self.logger.info("正在停止服务器...")
        
        # 关闭所有WebSocket连接
        for connection_id in list(self.connections.keys()):
            await self._cleanup_connection(connection_id)
        
        # 清理过期会话
        await self.session_manager.cleanup_expired_sessions()
        
        self.logger.info("服务器已停止")
    
    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return len(self.connections)
    
    def get_server_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        uptime = datetime.now() - self.stats['start_time']
        
        return {
            **self.stats,
            'uptime_seconds': uptime.total_seconds(),
            'active_connections': len(self.connections),
            'subscribers': {topic: len(connections) for topic, connections in self.subscribers.items()}
        }
    
    async def push_data_to_client(self, connection_id: str, data: Any):
        """向客户端推送数据"""
        await self._send_to_connection(connection_id, {
            "type": "push_data",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    async def broadcast_system_message(self, message: str, level: str = "info"):
        """广播系统消息"""
        await self.broadcast_to_topic("system", {
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat()
        })


# 测试和示例代码
async def test_websocket_controller():
    """测试Web接口控制器"""
    config = WebConfig(
        host="localhost",
        port=8000,
        debug=True,
        cors_origins=["http://localhost:3000"],
        max_file_size=10 * 1024 * 1024  # 10MB
    )
    
    controller = WebInterfaceController(config)
    
    # 启动服务器
    try:
        await controller.start_server()
    except KeyboardInterrupt:
        await controller.stop_server()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_websocket_controller())