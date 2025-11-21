"""
V4模块 - 模型部署器模块

这个模块提供了一个完整的模型部署系统，支持多种部署格式、
容器化部署、负载均衡、版本管理和监控等功能。

主要功能：
1. 模型打包和序列化
2. 多种部署格式支持（ONNX、TensorRT、PMML等）
3. 服务器部署和负载均衡
4. 容器化部署支持
5. API服务生成
6. 模型版本管理
7. 部署状态监控
8. 回滚和更新机制
9. 部署日志和错误处理
"""

from .ModelDeployer import (
    # 核心枚举和类
    DeploymentFormat,
    DeploymentStatus,
    ServerType,
    DeploymentConfig,
    ModelDeployer,
    
    # 便利函数
    create_model_deployer
)

__all__ = [
    'DeploymentFormat',
    'DeploymentStatus', 
    'ServerType',
    'DeploymentConfig',
    'ModelDeployer',
    'create_model_deployer'
]

__version__ = '1.0.0'

# 便利函数
def create_model_deployer(config: DeploymentConfig = None, **kwargs):
    """
    创建模型部署器实例
    
    Args:
        config: 部署配置对象
        **kwargs: 部署器参数
    
    Returns:
        ModelDeployer: 模型部署器实例
    
    Examples:
        from V4 import create_model_deployer, DeploymentConfig
        
        # 使用默认配置创建部署器
        deployer = create_model_deployer()
        
        # 使用自定义配置创建部署器
        config = DeploymentConfig(
            format=DeploymentFormat.ONNX,
            server_type=ServerType.FLASK,
            containerized=True
        )
        deployer = create_model_deployer(config=config)
    """
    return ModelDeployer(config=config, **kwargs)

# 部署策略
class DeploymentStrategies:
    """部署策略枚举"""
    BLUE_GREEN = "blue_green"  # 蓝绿部署
    CANARY = "canary"  # 金丝雀部署
    ROLLING = "rolling"  # 滚动部署
    RECREATE = "recreate"  # 重新创建部署

# 环境类型
class EnvironmentTypes:
    """环境类型枚举"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

# 快速开始指南
QUICK_START = """
V4模型部署器快速开始：

1. 创建部署器：
   from V4 import create_model_deployer
   deployer = create_model_deployer()

2. 配置部署参数：
   # 加载模型
   model = load_your_model()
   
   # 配置部署设置
   config = DeploymentConfig(
       format=DeploymentFormat.ONNX,
       server_type=ServerType.FLASK,
       port=8080
   )

3. 部署模型：
   # 部署到本地
   deployment = deployer.deploy(model, config)
   
   # 部署到Docker容器
   deployment = deployer.deploy_to_container(model, config)

4. 监控部署状态：
   status = deployer.get_deployment_status(deployment.id)
   print(f"部署状态: {status.status}")

5. 更新模型：
   new_model = load_new_model()
   updated_deployment = deployer.update_model(deployment.id, new_model)

6. 回滚到上一版本：
   deployer.rollback_deployment(deployment.id)
"""

# 部署检查装饰器
def validate_deployment(func):
    """部署验证装饰器"""
    def wrapper(*args, **kwargs):
        import os
        import json
        
        # 检查必要的配置
        config = None
        for arg in args:
            if hasattr(arg, 'format'):
                config = arg
                break
        
        if config:
            # 验证部署格式
            if config.format not in DeploymentFormat:
                raise ValueError(f"不支持的部署格式: {config.format}")
            
            # 验证服务器类型
            if config.server_type not in ServerType:
                raise ValueError(f"不支持的服务器类型: {config.server_type}")
            
            # 检查端口是否被占用
            if hasattr(config, 'port'):
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', config.port))
                if result == 0:
                    print(f"警告: 端口 {config.port} 已被占用")
                sock.close()
        
        return func(*args, **kwargs)
    
    return wrapper

# 资源监控装饰器
def monitor_deployment_resources(func):
    """部署资源监控装饰器"""
    def wrapper(*args, **kwargs):
        import psutil
        import time
        
        # 获取初始资源状态
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # 获取最终资源状态
        final_memory = psutil.virtual_memory().percent
        final_cpu = psutil.cpu_percent()
        
        print(f"部署函数 {func.__name__} 资源使用:")
        print(f"  执行时间: {end_time - start_time:.3f}秒")
        print(f"  CPU使用率: {initial_cpu}% → {final_cpu}%")
        print(f"  内存使用率: {initial_memory}% → {final_memory}%")
        
        return result
    
    return wrapper

print("V4模型部署器已加载")