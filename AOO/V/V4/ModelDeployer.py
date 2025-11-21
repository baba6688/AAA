"""
V4模型部署器 - 完整的模型部署解决方案

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


版本: 1.0.0
日期: 2025-11-05
"""

import os
import json
import pickle
import logging
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import zipfile
import tempfile
import yaml
import requests
import docker
from flask import Flask, request, jsonify, Response
import joblib
import numpy as np


class DeploymentFormat(Enum):
    """支持的部署格式枚举"""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    PMML = "pmml"
    TORCHSCRIPT = "torchscript"
    TENSORFLOW_SAVED_MODEL = "tensorflow_saved_model"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


class DeploymentStatus(Enum):
    """部署状态枚举"""
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"
    UPDATING = "updating"
    ROLLING_BACK = "rolling_back"


class ServerType(Enum):
    """服务器类型枚举"""
    FLASK = "flask"
    FASTAPI = "fastapi"
    TORCH_SERVE = "torch_serve"
    TENSORFLOW_SERVING = "tensorflow_serving"
    SELdon = "seldon"
    KSERVE = "kserve"


@dataclass
class ModelMetadata:
    """模型元数据类"""
    model_id: str
    model_name: str
    model_version: str
    format: DeploymentFormat
    created_at: datetime
    framework: str
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    input_types: Optional[List[str]] = None
    output_types: Optional[List[str]] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    postprocessing_config: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None


@dataclass
class DeploymentConfig:
    """部署配置类"""
    deployment_id: str
    model_id: str
    server_type: ServerType
    format: DeploymentFormat
    replicas: int = 1
    resources: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, str]] = None
    ports: Optional[List[int]] = None
    health_check: Optional[Dict[str, Any]] = None
    load_balancer_config: Optional[Dict[str, Any]] = None
    auto_scaling: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    rollback_config: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentStatusInfo:
    """部署状态信息类"""
    deployment_id: str
    status: DeploymentStatus
    created_at: datetime
    updated_at: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    logs: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None
    health_status: Optional[str] = None


class ModelSerializer:
    """模型序列化器类"""
    
    @staticmethod
    def serialize_model(model: Any, format: DeploymentFormat, 
                       output_path: str, metadata: ModelMetadata) -> bool:
        """
        序列化模型到指定格式
        
        Args:
            model: 要序列化的模型
            format: 部署格式
            output_path: 输出路径
            metadata: 模型元数据
            
        Returns:
            bool: 是否成功
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if format == DeploymentFormat.PICKLE:
                return ModelSerializer._serialize_pickle(model, output_path, metadata)
            elif format == DeploymentFormat.JOBLIB:
                return ModelSerializer._serialize_joblib(model, output_path, metadata)
            elif format == DeploymentFormat.ONNX:
                return ModelSerializer._serialize_onnx(model, output_path, metadata)
            elif format == DeploymentFormat.TENSORRT:
                return ModelSerializer._serialize_tensorrt(model, output_path, metadata)
            elif format == DeploymentFormat.PMML:
                return ModelSerializer._serialize_pmml(model, output_path, metadata)
            elif format == DeploymentFormat.TORCHSCRIPT:
                return ModelSerializer._serialize_torchscript(model, output_path, metadata)
            elif format == DeploymentFormat.TENSORFLOW_SAVED_MODEL:
                return ModelSerializer._serialize_tensorflow(model, output_path, metadata)
            elif format == DeploymentFormat.XGBOOST:
                return ModelSerializer._serialize_xgboost(model, output_path, metadata)
            elif format == DeploymentFormat.LIGHTGBM:
                return ModelSerializer._serialize_lightgbm(model, output_path, metadata)
            elif format == DeploymentFormat.CATBOOST:
                return ModelSerializer._serialize_catboost(model, output_path, metadata)
            else:
                raise ValueError(f"不支持的格式: {format}")
                
        except Exception as e:
            logging.error(f"模型序列化失败: {e}")
            return False
    
    @staticmethod
    def _serialize_pickle(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """Pickle格式序列化"""
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 保存元数据
        metadata_path = output_path + '.meta'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, default=str, indent=2, ensure_ascii=False)
        
        return True
    
    @staticmethod
    def _serialize_joblib(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """Joblib格式序列化"""
        joblib.dump(model, output_path)
        
        # 保存元数据
        metadata_path = output_path + '.meta'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, default=str, indent=2, ensure_ascii=False)
        
        return True
    
    @staticmethod
    def _serialize_onnx(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """ONNX格式序列化"""
        try:
            import torch
            import torch.onnx
            
            if hasattr(model, 'eval'):
                model.eval()
            
            # 创建一个示例输入
            dummy_input = torch.randn(metadata.input_shape)
            
            # 导出到ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 保存元数据
            metadata_path = output_path + '.meta'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, default=str, indent=2, ensure_ascii=False)
            
            return True
            
        except ImportError:
            logging.warning("PyTorch未安装，无法导出ONNX格式")
            return False
        except Exception as e:
            logging.error(f"ONNX导出失败: {e}")
            return False
    
    @staticmethod
    def _serialize_tensorrt(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """TensorRT格式序列化"""
        try:
            import tensorrt as trt
            
            # 这里需要根据具体模型类型实现TensorRT转换逻辑
            # 简化实现，实际中需要更复杂的转换流程
            logging.info("TensorRT序列化需要额外的转换步骤")
            return False
            
        except ImportError:
            logging.warning("TensorRT未安装")
            return False
        except Exception as e:
            logging.error(f"TensorRT序列化失败: {e}")
            return False
    
    @staticmethod
    def _serialize_pmml(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """PMML格式序列化"""
        try:
            from sklearn2pmml import PMMLPipeline
            from sklearn2pmml import sklearn2pmml
            
            if hasattr(model, 'predict'):
                # 假设是sklearn模型
                pipeline = PMMLPipeline([('classifier', model)])
                sklearn2pmml(pipeline, output_path)
                
                # 保存元数据
                metadata_path = output_path + '.meta'
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(metadata), f, default=str, indent=2, ensure_ascii=False)
                
                return True
            else:
                logging.warning("模型不支持PMML格式")
                return False
                
        except ImportError:
            logging.warning("sklearn2pmml未安装，无法导出PMML格式")
            return False
        except Exception as e:
            logging.error(f"PMML导出失败: {e}")
            return False
    
    @staticmethod
    def _serialize_torchscript(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """TorchScript格式序列化"""
        try:
            import torch
            
            if hasattr(model, 'eval'):
                model.eval()
            
            # 转换为TorchScript
            example_input = torch.randn(metadata.input_shape)
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(output_path)
            
            # 保存元数据
            metadata_path = output_path + '.meta'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, default=str, indent=2, ensure_ascii=False)
            
            return True
            
        except ImportError:
            logging.warning("PyTorch未安装，无法导出TorchScript格式")
            return False
        except Exception as e:
            logging.error(f"TorchScript导出失败: {e}")
            return False
    
    @staticmethod
    def _serialize_tensorflow(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """TensorFlow SavedModel格式序列化"""
        try:
            import tensorflow as tf
            
            if hasattr(model, 'save'):
                model.save(output_path, save_format='tf')
                
                # 保存元数据
                metadata_path = os.path.join(output_path, 'metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(metadata), f, default=str, indent=2, ensure_ascii=False)
                
                return True
            else:
                logging.warning("模型不支持TensorFlow SavedModel格式")
                return False
                
        except ImportError:
            logging.warning("TensorFlow未安装，无法导出SavedModel格式")
            return False
        except Exception as e:
            logging.error(f"TensorFlow SavedModel导出失败: {e}")
            return False
    
    @staticmethod
    def _serialize_xgboost(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """XGBoost格式序列化"""
        try:
            import xgboost as xgb
            
            if hasattr(model, 'save_model'):
                model.save_model(output_path)
                
                # 保存元数据
                metadata_path = output_path + '.meta'
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(metadata), f, default=str, indent=2, ensure_ascii=False)
                
                return True
            else:
                logging.warning("模型不是有效的XGBoost模型")
                return False
                
        except ImportError:
            logging.warning("XGBoost未安装")
            return False
        except Exception as e:
            logging.error(f"XGBoost序列化失败: {e}")
            return False
    
    @staticmethod
    def _serialize_lightgbm(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """LightGBM格式序列化"""
        try:
            import lightgbm as lgb
            
            if hasattr(model, 'save_model'):
                model.save_model(output_path)
                
                # 保存元数据
                metadata_path = output_path + '.meta'
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(metadata), f, default=str, indent=2, ensure_ascii=False)
                
                return True
            else:
                logging.warning("模型不是有效的LightGBM模型")
                return False
                
        except ImportError:
            logging.warning("LightGBM未安装")
            return False
        except Exception as e:
            logging.error(f"LightGBM序列化失败: {e}")
            return False
    
    @staticmethod
    def _serialize_catboost(model: Any, output_path: str, metadata: ModelMetadata) -> bool:
        """CatBoost格式序列化"""
        try:
            import catboost as cb
            
            if hasattr(model, 'save_model'):
                model.save_model(output_path)
                
                # 保存元数据
                metadata_path = output_path + '.meta'
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(metadata), f, default=str, indent=2, ensure_ascii=False)
                
                return True
            else:
                logging.warning("模型不是有效的CatBoost模型")
                return False
                
        except ImportError:
            logging.warning("CatBoost未安装")
            return False
        except Exception as e:
            logging.error(f"CatBoost序列化失败: {e}")
            return False


class APIServiceGenerator:
    """API服务生成器类"""
    
    def __init__(self, base_port: int = 8000):
        self.base_port = base_port
        self.services = {}
    
    def generate_flask_service(self, model_path: str, metadata: ModelMetadata, 
                             port: Optional[int] = None) -> str:
        """
        生成Flask API服务
        
        Args:
            model_path: 模型路径
            metadata: 模型元数据
            port: 端口号
            
        Returns:
            str: 服务代码
        """
        if port is None:
            port = self.base_port
        
        service_code = f'''
"""
Flask API服务 - 自动生成的模型服务
模型: {metadata.model_name} v{metadata.model_version}
格式: {metadata.format.value}
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from datetime import datetime

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载模型
try:
    model = joblib.load("{model_path}")
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {{e}}")
    model = None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({{
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    }})

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        if model is None:
            return jsonify({{'error': '模型未加载'}}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({{'error': '无效的输入数据'}}), 400
        
        # 提取特征
        features = data.get('features', [])
        if not features:
            return jsonify({{'error': '缺少特征数据'}}), 400
        
        # 转换为numpy数组
        input_array = np.array(features)
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)
        
        # 进行预测
        predictions = model.predict(input_array)
        
        # 返回结果
        result = {{
            'predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat(),
            'model_version': '{metadata.model_version}'
        }}
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"预测失败: {{e}}")
        return jsonify({{'error': f'预测失败: {{str(e)}}'}}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """模型信息接口"""
    return jsonify({{
        'model_name': '{metadata.model_name}',
        'model_version': '{metadata.model_version}',
        'format': '{metadata.format.value}',
        'framework': '{metadata.framework}',
        'input_shape': {metadata.input_shape},
        'output_shape': {metadata.output_shape}
    }})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port={port}, debug=False)
'''
        return service_code
    
    def generate_fastapi_service(self, model_path: str, metadata: ModelMetadata, 
                               port: Optional[int] = None) -> str:
        """
        生成FastAPI服务
        
        Args:
            model_path: 模型路径
            metadata: 模型元数据
            port: 端口号
            
        Returns:
            str: 服务代码
        """
        if port is None:
            port = self.base_port + 1
        
        service_code = f'''
"""
FastAPI服务 - 自动生成的模型服务
模型: {metadata.model_name} v{metadata.model_version}
格式: {metadata.format.value}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Any
from datetime import datetime
import uvicorn

app = FastAPI(title="{metadata.model_name}", 
              description=f"{{metadata.model_name}} v{{metadata.model_version}} API",
              version="{metadata.model_version}")

# 加载模型
try:
    model = joblib.load("{model_path}")
except Exception as e:
    model = None
    print(f"模型加载失败: {{e}}")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    predictions: List[Any]
    timestamp: str
    model_version: str

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {{
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    }}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """预测接口"""
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 转换为numpy数组
        input_array = np.array(request.features)
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)
        
        # 进行预测
        predictions = model.predict(input_array)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            timestamp=datetime.now().isoformat(),
            model_version="{metadata.model_version}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {{str(e)}}")

@app.get("/model/info")
async def model_info():
    """模型信息接口"""
    return {{
        'model_name': '{metadata.model_name}',
        'model_version': '{metadata.model_version}',
        'format': '{metadata.format.value}',
        'framework': '{metadata.framework}',
        'input_shape': {metadata.input_shape},
        'output_shape': {metadata.output_shape}
    }}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port={port})
'''
        return service_code


class ContainerDeployer:
    """容器化部署器类"""
    
    def __init__(self, docker_client: Optional[docker.DockerClient] = None):
        self.docker_client = docker_client or docker.from_env()
    
    def create_dockerfile(self, model_path: str, service_code: str, 
                         requirements: List[str], output_dir: str) -> str:
        """
        创建Dockerfile
        
        Args:
            model_path: 模型文件路径
            service_code: 服务代码
            requirements: Python依赖列表
            output_dir: 输出目录
            
        Returns:
            str: Dockerfile路径
        """
        dockerfile_path = os.path.join(output_dir, 'Dockerfile')
        
        # 生成requirements.txt
        requirements_path = os.path.join(output_dir, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        # 生成Dockerfile
        dockerfile_content = f'''
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制模型文件和服务代码
COPY {os.path.basename(model_path)} ./model/
COPY service.py ./

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "service.py"]
'''
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # 保存服务代码
        service_path = os.path.join(output_dir, 'service.py')
        with open(service_path, 'w') as f:
            f.write(service_code)
        
        return dockerfile_path
    
    def build_image(self, dockerfile_dir: str, image_name: str, 
                   image_tag: str = 'latest') -> bool:
        """
        构建Docker镜像
        
        Args:
            dockerfile_dir: Dockerfile目录
            image_name: 镜像名称
            image_tag: 镜像标签
            
        Returns:
            bool: 是否成功
        """
        try:
            full_image_name = f"{image_name}:{image_tag}"
            
            # 构建镜像
            self.docker_client.images.build(
                path=dockerfile_dir,
                tag=full_image_name,
                rm=True
            )
            
            logging.info(f"Docker镜像构建成功: {full_image_name}")
            return True
            
        except Exception as e:
            logging.error(f"Docker镜像构建失败: {e}")
            return False
    
    def run_container(self, image_name: str, container_name: str, 
                     ports: Dict[int, int], environment: Optional[Dict[str, str]] = None) -> str:
        """
        运行Docker容器
        
        Args:
            image_name: 镜像名称
            container_name: 容器名称
            ports: 端口映射
            environment: 环境变量
            
        Returns:
            str: 容器ID
        """
        try:
            container = self.docker_client.containers.run(
                image=image_name,
                name=container_name,
                ports=ports,
                environment=environment,
                detach=True,
                remove=False
            )
            
            logging.info(f"容器启动成功: {container_name}")
            return container.id
            
        except Exception as e:
            logging.error(f"容器启动失败: {e}")
            return None


class LoadBalancer:
    """负载均衡器类"""
    
    def __init__(self, algorithm: str = 'round_robin'):
        self.algorithm = algorithm
        self.servers = []
        self.current_index = 0
    
    def add_server(self, host: str, port: int, weight: int = 1) -> None:
        """添加服务器"""
        self.servers.append({
            'host': host,
            'port': port,
            'weight': weight,
            'active': True
        })
    
    def remove_server(self, host: str, port: int) -> bool:
        """移除服务器"""
        for server in self.servers:
            if server['host'] == host and server['port'] == port:
                self.servers.remove(server)
                return True
        return False
    
    def get_next_server(self) -> Optional[Dict[str, Any]]:
        """获取下一个服务器"""
        if not self.servers:
            return None
        
        active_servers = [s for s in self.servers if s['active']]
        if not active_servers:
            return None
        
        if self.algorithm == 'round_robin':
            server = active_servers[self.current_index % len(active_servers)]
            self.current_index += 1
            return server
        elif self.algorithm == 'random':
            import random
            return random.choice(active_servers)
        elif self.algorithm == 'weighted_round_robin':
            # 简化实现，实际中需要更复杂的加权轮询算法
            return active_servers[0]
        else:
            return active_servers[0]


class ModelVersionManager:
    """模型版本管理器类"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.versions_file = os.path.join(storage_path, 'versions.json')
        self._ensure_storage_dir()
        self._load_versions()
    
    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _load_versions(self):
        """加载版本信息"""
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r', encoding='utf-8') as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def _save_versions(self):
        """保存版本信息"""
        with open(self.versions_file, 'w', encoding='utf-8') as f:
            json.dump(self.versions, f, indent=2, ensure_ascii=False)
    
    def register_model(self, metadata: ModelMetadata, model_path: str) -> str:
        """注册模型"""
        model_id = metadata.model_id
        
        self.versions[model_id] = {
            'metadata': asdict(metadata),
            'model_path': model_path,
            'created_at': metadata.created_at.isoformat(),
            'status': 'active'
        }
        
        self._save_versions()
        logging.info(f"模型注册成功: {model_id}")
        return model_id
    
    def create_version(self, model_id: str, new_version: str, 
                      model_path: str, metadata: ModelMetadata) -> bool:
        """创建新版本"""
        if model_id not in self.versions:
            logging.error(f"模型不存在: {model_id}")
            return False
        
        # 标记旧版本为inactive
        for version_info in self.versions[model_id].values():
            if isinstance(version_info, dict) and 'status' in version_info:
                version_info['status'] = 'inactive'
        
        # 添加新版本
        version_key = f"v{new_version}"
        self.versions[model_id][version_key] = {
            'metadata': asdict(metadata),
            'model_path': model_path,
            'created_at': metadata.created_at.isoformat(),
            'status': 'active'
        }
        
        self._save_versions()
        logging.info(f"模型版本创建成功: {model_id} {version_key}")
        return True
    
    def get_active_version(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取活跃版本"""
        if model_id not in self.versions:
            return None
        
        for version_key, version_info in self.versions[model_id].items():
            if isinstance(version_info, dict) and version_info.get('status') == 'active':
                return version_info
        
        return None
    
    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """列出所有版本"""
        if model_id not in self.versions:
            return []
        
        versions = []
        for version_key, version_info in self.versions[model_id].items():
            if isinstance(version_info, dict):
                versions.append({
                    'version': version_key,
                    'status': version_info.get('status'),
                    'created_at': version_info.get('created_at'),
                    'metadata': version_info.get('metadata', {})
                })
        
        return versions
    
    def rollback_version(self, model_id: str, target_version: str) -> bool:
        """回滚到指定版本"""
        if model_id not in self.versions:
            return False
        
        version_key = f"v{target_version}" if not target_version.startswith('v') else target_version
        
        if version_key not in self.versions[model_id]:
            return False
        
        # 标记所有版本为inactive
        for version_info in self.versions[model_id].values():
            if isinstance(version_info, dict):
                version_info['status'] = 'inactive'
        
        # 激活目标版本
        self.versions[model_id][version_key]['status'] = 'active'
        
        self._save_versions()
        logging.info(f"模型回滚成功: {model_id} -> {version_key}")
        return True


class DeploymentMonitor:
    """部署监控器类"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.deployments = {}
        self.monitoring_thread = None
        self.running = False
    
    def start_monitoring(self):
        """开始监控"""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop)
            self.monitoring_thread.start()
            logging.info("部署监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logging.info("部署监控已停止")
    
    def add_deployment(self, deployment_id: str, config: DeploymentConfig):
        """添加部署监控"""
        self.deployments[deployment_id] = {
            'config': config,
            'status_info': DeploymentStatusInfo(
                deployment_id=deployment_id,
                status=DeploymentStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            'last_check': None,
            'metrics': {}
        }
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                for deployment_id, deployment_info in self.deployments.items():
                    self._check_deployment_status(deployment_id, deployment_info)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"监控循环错误: {e}")
    
    def _check_deployment_status(self, deployment_id: str, deployment_info: Dict[str, Any]):
        """检查部署状态"""
        try:
            config = deployment_info['config']
            status_info = deployment_info['status_info']
            
            # 检查容器状态（如果使用容器化部署）
            if hasattr(self, '_check_container_status'):
                self._check_container_status(deployment_id, deployment_info)
            
            # 检查API健康状态
            if hasattr(self, '_check_api_health'):
                self._check_api_health(deployment_id, deployment_info)
            
            # 更新检查时间
            deployment_info['last_check'] = datetime.now()
            status_info.updated_at = datetime.now()
            
        except Exception as e:
            logging.error(f"检查部署状态失败 {deployment_id}: {e}")
    
    def _check_container_status(self, deployment_id: str, deployment_info: Dict[str, Any]):
        """检查容器状态"""
        # 这里可以实现容器状态检查逻辑
        pass
    
    def _check_api_health(self, deployment_id: str, deployment_info: Dict[str, Any]):
        """检查API健康状态"""
        config = deployment_info['config']
        
        # 假设API地址格式
        api_url = f"http://localhost:{config.ports[0] if config.ports else 8000}/health"
        
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                deployment_info['status_info'].health_status = 'healthy'
            else:
                deployment_info['status_info'].health_status = 'unhealthy'
        except Exception:
            deployment_info['status_info'].health_status = 'unreachable'


class ModelDeployer:
    """V4模型部署器主类
    
    这是一个完整的模型部署解决方案，支持多种部署格式、
    容器化部署、负载均衡、版本管理和监控等功能。
    """
    
    def __init__(self, storage_path: str = "./model_storage", 
                 base_port: int = 8000, log_level: str = "INFO"):
        """
        初始化模型部署器
        
        Args:
            storage_path: 模型存储路径
            base_port: 基础端口号
            log_level: 日志级别
        """
        # 设置日志
        self._setup_logging(log_level)
        
        # 初始化组件
        self.storage_path = storage_path
        self.base_port = base_port
        self.current_port = base_port
        
        # 创建组件实例
        self.serializer = ModelSerializer()
        self.api_generator = APIServiceGenerator(base_port)
        self.container_deployer = ContainerDeployer()
        self.version_manager = ModelVersionManager(storage_path)
        self.monitor = DeploymentMonitor()
        
        # 部署管理
        self.deployments = {}
        self.load_balancer = LoadBalancer()
        
        # 启动监控
        self.monitor.start_monitoring()
        
        logging.info("V4模型部署器初始化完成")
    
    def _setup_logging(self, log_level: str):
        """设置日志配置"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_deployer.log'),
                logging.StreamHandler()
            ]
        )
    
    def package_model(self, model: Any, model_name: str, 
                     format: DeploymentFormat, 
                     framework: str,
                     input_shape: Optional[List[int]] = None,
                     output_shape: Optional[List[int]] = None,
                     tags: Optional[List[str]] = None) -> Optional[str]:
        """
        打包模型
        
        Args:
            model: 要打包的模型
            model_name: 模型名称
            format: 部署格式
            framework: 框架名称
            input_shape: 输入形状
            output_shape: 输出形状
            tags: 标签列表
            
        Returns:
            str: 模型ID，失败返回None
        """
        try:
            # 生成模型ID和版本
            model_id = str(uuid.uuid4())
            model_version = "1.0.0"
            
            # 创建模型元数据
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_version=model_version,
                format=format,
                created_at=datetime.now(),
                framework=framework,
                input_shape=input_shape,
                output_shape=output_shape,
                tags=tags or []
            )
            
            # 创建模型文件路径
            model_filename = f"{model_name}_{model_version}.{format.value}"
            model_path = os.path.join(self.storage_path, model_filename)
            
            # 序列化模型
            if self.serializer.serialize_model(model, format, model_path, metadata):
                # 注册模型版本
                self.version_manager.register_model(metadata, model_path)
                
                logging.info(f"模型打包成功: {model_id}")
                return model_id
            else:
                logging.error(f"模型序列化失败: {model_name}")
                return None
                
        except Exception as e:
            logging.error(f"模型打包失败: {e}")
            return None
    
    def deploy_model(self, model_id: str, server_type: ServerType,
                    replicas: int = 1, environment: Optional[Dict[str, str]] = None,
                    auto_scaling: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        部署模型
        
        Args:
            model_id: 模型ID
            server_type: 服务器类型
            replicas: 副本数量
            environment: 环境变量
            auto_scaling: 自动缩放配置
            
        Returns:
            str: 部署ID，失败返回None
        """
        try:
            # 获取模型信息
            active_version = self.version_manager.get_active_version(model_id)
            if not active_version:
                logging.error(f"模型版本不存在: {model_id}")
                return None
            
            metadata_dict = active_version['metadata']
            metadata = ModelMetadata(**metadata_dict)
            
            # 生成部署ID
            deployment_id = str(uuid.uuid4())
            
            # 创建部署配置
            config = DeploymentConfig(
                deployment_id=deployment_id,
                model_id=model_id,
                server_type=server_type,
                format=metadata.format,
                replicas=replicas,
                environment=environment or {},
                ports=[self.current_port + len(self.deployments)],
                auto_scaling=auto_scaling
            )
            
            # 添加到监控
            self.monitor.add_deployment(deployment_id, config)
            
            # 创建部署目录
            deploy_dir = os.path.join(self.storage_path, "deployments", deployment_id)
            os.makedirs(deploy_dir, exist_ok=True)
            
            # 生成API服务
            service_code = self._generate_service_code(active_version['model_path'], metadata, server_type)
            
            # 创建容器化部署
            if server_type in [ServerType.FLASK, ServerType.FASTAPI]:
                container_id = self._deploy_container(deployment_id, config, service_code, deploy_dir)
                if container_id:
                    config.environment['CONTAINER_ID'] = container_id
            
            # 添加到负载均衡器
            for i in range(replicas):
                host = "localhost"
                port = config.ports[0] + i
                self.load_balancer.add_server(host, port)
            
            # 保存部署信息
            self.deployments[deployment_id] = {
                'config': config,
                'service_code': service_code,
                'deploy_dir': deploy_dir,
                'container_id': container_id if 'container_id' in locals() else None
            }
            
            # 更新部署状态
            self.monitor.deployments[deployment_id]['status_info'].status = DeploymentStatus.RUNNING
            self.monitor.deployments[deployment_id]['status_info'].start_time = datetime.now()
            
            logging.info(f"模型部署成功: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logging.error(f"模型部署失败: {e}")
            return None
    
    def _generate_service_code(self, model_path: str, metadata: ModelMetadata, 
                             server_type: ServerType) -> str:
        """生成服务代码"""
        if server_type == ServerType.FLASK:
            return self.api_generator.generate_flask_service(model_path, metadata)
        elif server_type == ServerType.FASTAPI:
            return self.api_generator.generate_fastapi_service(model_path, metadata)
        else:
            raise ValueError(f"不支持的服务器类型: {server_type}")
    
    def _deploy_container(self, deployment_id: str, config: DeploymentConfig, 
                         service_code: str, deploy_dir: str) -> Optional[str]:
        """部署容器"""
        try:
            # 生成requirements
            requirements = self._generate_requirements(config.format)
            
            # 创建Dockerfile
            dockerfile_path = self.container_deployer.create_dockerfile(
                config.model_id, service_code, requirements, deploy_dir
            )
            
            # 构建镜像
            image_name = f"model_{deployment_id}"
            if self.container_deployer.build_image(deploy_dir, image_name):
                # 运行容器
                ports = {8000: config.ports[0]}
                container_name = f"model_{deployment_id}"
                container_id = self.container_deployer.run_container(
                    image_name, container_name, ports, config.environment
                )
                return container_id
            
            return None
            
        except Exception as e:
            logging.error(f"容器部署失败: {e}")
            return None
    
    def _generate_requirements(self, format: DeploymentFormat) -> List[str]:
        """生成依赖列表"""
        base_requirements = [
            "flask==2.3.3",
            "fastapi==0.103.1",
            "uvicorn==0.23.2",
            "numpy==1.24.3",
            "joblib==1.3.2"
        ]
        
        if format == DeploymentFormat.ONNX:
            base_requirements.extend([
                "onnx==1.14.0",
                "onnxruntime==1.15.1",
                "torch==2.0.1"
            ])
        elif format == DeploymentFormat.TENSORRT:
            base_requirements.extend([
                "tensorrt==8.6.1",
                "pycuda==2022.1"
            ])
        elif format == DeploymentFormat.PMML:
            base_requirements.extend([
                "sklearn2pmml==0.87.1",
                "jpmml-sklearn==1.8.12"
            ])
        elif format == DeploymentFormat.TORCHSCRIPT:
            base_requirements.extend([
                "torch==2.0.1"
            ])
        elif format == DeploymentFormat.TENSORFLOW_SAVED_MODEL:
            base_requirements.extend([
                "tensorflow==2.13.0"
            ])
        elif format == DeploymentFormat.XGBOOST:
            base_requirements.extend([
                "xgboost==1.7.6"
            ])
        elif format == DeploymentFormat.LIGHTGBM:
            base_requirements.extend([
                "lightgbm==4.0.0"
            ])
        elif format == DeploymentFormat.CATBOOST:
            base_requirements.extend([
                "catboost==1.2"
            ])
        
        return base_requirements
    
    def update_model(self, model_id: str, new_model: Any, 
                    new_version: Optional[str] = None) -> bool:
        """
        更新模型
        
        Args:
            model_id: 模型ID
            new_model: 新模型
            new_version: 新版本号
            
        Returns:
            bool: 是否成功
        """
        try:
            # 获取当前模型信息
            active_version = self.version_manager.get_active_version(model_id)
            if not active_version:
                logging.error(f"模型不存在: {model_id}")
                return False
            
            old_metadata = ModelMetadata(**active_version['metadata'])
            
            # 生成新版本号
            if new_version is None:
                # 自动递增版本号
                old_version_parts = old_metadata.model_version.split('.')
                major, minor, patch = map(int, old_version_parts)
                new_version = f"{major}.{minor}.{patch + 1}"
            
            # 创建新元数据
            new_metadata = ModelMetadata(
                model_id=model_id,
                model_name=old_metadata.model_name,
                model_version=new_version,
                format=old_metadata.format,
                created_at=datetime.now(),
                framework=old_metadata.framework,
                input_shape=old_metadata.input_shape,
                output_shape=old_metadata.output_shape,
                tags=old_metadata.tags
            )
            
            # 创建新模型文件路径
            model_filename = f"{old_metadata.model_name}_{new_version}.{old_metadata.format.value}"
            model_path = os.path.join(self.storage_path, model_filename)
            
            # 序列化新模型
            if self.serializer.serialize_model(new_model, old_metadata.format, model_path, new_metadata):
                # 创建新版本
                if self.version_manager.create_version(model_id, new_version, model_path, new_metadata):
                    # 重新部署
                    self._redeploy_model(model_id)
                    logging.info(f"模型更新成功: {model_id} -> v{new_version}")
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"模型更新失败: {e}")
            return False
    
    def _redeploy_model(self, model_id: str):
        """重新部署模型"""
        # 找到该模型的所有部署
        for deployment_id, deployment_info in list(self.deployments.items()):
            if deployment_info['config'].model_id == model_id:
                # 停止旧部署
                self.stop_deployment(deployment_id)
                
                # 创建新部署
                config = deployment_info['config']
                new_deployment_id = self.deploy_model(
                    model_id, config.server_type, config.replicas, 
                    config.environment, config.auto_scaling
                )
                
                if new_deployment_id:
                    logging.info(f"模型重新部署成功: {model_id}")
    
    def rollback_model(self, model_id: str, target_version: str) -> bool:
        """
        回滚模型
        
        Args:
            model_id: 模型ID
            target_version: 目标版本
            
        Returns:
            bool: 是否成功
        """
        try:
            if self.version_manager.rollback_version(model_id, target_version):
                self._redeploy_model(model_id)
                logging.info(f"模型回滚成功: {model_id} -> v{target_version}")
                return True
            else:
                logging.error(f"模型回滚失败: {model_id}")
                return False
                
        except Exception as e:
            logging.error(f"模型回滚失败: {e}")
            return False
    
    def stop_deployment(self, deployment_id: str) -> bool:
        """
        停止部署
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            bool: 是否成功
        """
        try:
            if deployment_id not in self.deployments:
                logging.error(f"部署不存在: {deployment_id}")
                return False
            
            deployment_info = self.deployments[deployment_id]
            
            # 停止容器
            if deployment_info.get('container_id'):
                try:
                    container = self.container_deployer.docker_client.containers.get(
                        deployment_info['container_id']
                    )
                    container.stop()
                    container.remove()
                except Exception as e:
                    logging.warning(f"停止容器失败: {e}")
            
            # 从负载均衡器移除
            config = deployment_info['config']
            for port in config.ports:
                self.load_balancer.remove_server('localhost', port)
            
            # 更新状态
            if deployment_id in self.monitor.deployments:
                self.monitor.deployments[deployment_id]['status_info'].status = DeploymentStatus.STOPPED
                self.monitor.deployments[deployment_id]['status_info'].end_time = datetime.now()
            
            # 移除部署信息
            del self.deployments[deployment_id]
            
            logging.info(f"部署已停止: {deployment_id}")
            return True
            
        except Exception as e:
            logging.error(f"停止部署失败: {e}")
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatusInfo]:
        """
        获取部署状态
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            DeploymentStatusInfo: 部署状态信息
        """
        if deployment_id in self.monitor.deployments:
            return self.monitor.deployments[deployment_id]['status_info']
        return None
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """
        列出所有部署
        
        Returns:
            List[Dict[str, Any]]: 部署列表
        """
        deployments = []
        for deployment_id, deployment_info in self.deployments.items():
            status_info = self.get_deployment_status(deployment_id)
            deployments.append({
                'deployment_id': deployment_id,
                'config': asdict(deployment_info['config']),
                'status': status_info.status.value if status_info else 'unknown',
                'health_status': status_info.health_status if status_info else 'unknown',
                'created_at': status_info.created_at if status_info else None,
                'updated_at': status_info.updated_at if status_info else None
            })
        
        return deployments
    
    def get_model_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """
        获取模型版本列表
        
        Args:
            model_id: 模型ID
            
        Returns:
            List[Dict[str, Any]]: 版本列表
        """
        return self.version_manager.list_versions(model_id)
    
    def predict(self, model_id: str, features: List[float]) -> Optional[Any]:
        """
        直接预测（用于测试）
        
        Args:
            model_id: 模型ID
            features: 输入特征
            
        Returns:
            Any: 预测结果
        """
        try:
            active_version = self.version_manager.get_active_version(model_id)
            if not active_version:
                logging.error(f"模型版本不存在: {model_id}")
                return None
            
            model_path = active_version['model_path']
            
            # 加载模型
            if active_version['metadata']['format'] == DeploymentFormat.JOBLIB.value:
                model = joblib.load(model_path)
            elif active_version['metadata']['format'] == DeploymentFormat.PICKLE.value:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                logging.error(f"不支持的格式进行直接预测: {active_version['metadata']['format']}")
                return None
            
            # 进行预测
            input_array = np.array(features)
            if input_array.ndim == 1:
                input_array = input_array.reshape(1, -1)
            
            predictions = model.predict(input_array)
            return predictions[0] if len(predictions) == 1 else predictions
            
        except Exception as e:
            logging.error(f"预测失败: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        try:
            # 停止监控
            self.monitor.stop_monitoring()
            
            # 停止所有部署
            for deployment_id in list(self.deployments.keys()):
                self.stop_deployment(deployment_id)
            
            logging.info("模型部署器已清理")
            
        except Exception as e:
            logging.error(f"清理资源失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


# 测试用例
def test_model_deployer():
    """测试模型部署器"""
    print("开始测试V4模型部署器...")
    
    # 创建示例模型（简单的线性回归）
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    
    # 生成测试数据
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    model = LinearRegression()
    model.fit(X, y)
    
    # 测试部署器
    with ModelDeployer(storage_path="./test_storage", log_level="INFO") as deployer:
        try:
            # 1. 测试模型打包
            print("1. 测试模型打包...")
            model_id = deployer.package_model(
                model=model,
                model_name="test_linear_model",
                format=DeploymentFormat.JOBLIB,
                framework="sklearn",
                input_shape=[4],
                output_shape=[1],
                tags=["test", "linear"]
            )
            
            if model_id:
                print(f"   ✓ 模型打包成功，ID: {model_id}")
            else:
                print("   ✗ 模型打包失败")
                return
            
            # 2. 测试模型部署
            print("2. 测试模型部署...")
            deployment_id = deployer.deploy_model(
                model_id=model_id,
                server_type=ServerType.FLASK,
                replicas=1
            )
            
            if deployment_id:
                print(f"   ✓ 模型部署成功，部署ID: {deployment_id}")
            else:
                print("   ✗ 模型部署失败")
                return
            
            # 3. 测试直接预测
            print("3. 测试直接预测...")
            test_features = [1.0, 2.0, 3.0, 4.0]
            prediction = deployer.predict(model_id, test_features)
            
            if prediction is not None:
                print(f"   ✓ 预测成功，结果: {prediction}")
            else:
                print("   ✗ 预测失败")
            
            # 4. 测试状态查询
            print("4. 测试状态查询...")
            status = deployer.get_deployment_status(deployment_id)
            if status:
                print(f"   ✓ 部署状态: {status.status.value}")
            else:
                print("   ✗ 状态查询失败")
            
            # 5. 测试版本管理
            print("5. 测试版本管理...")
            versions = deployer.get_model_versions(model_id)
            print(f"   ✓ 模型版本列表: {len(versions)} 个版本")
            
            # 6. 测试部署列表
            print("6. 测试部署列表...")
            deployments = deployer.list_deployments()
            print(f"   ✓ 部署列表: {len(deployments)} 个部署")
            
            # 7. 测试模型更新
            print("7. 测试模型更新...")
            # 创建新模型（稍微不同的参数）
            new_model = LinearRegression()
            new_model.fit(X, y)
            
            if deployer.update_model(model_id, new_model):
                print("   ✓ 模型更新成功")
            else:
                print("   ✗ 模型更新失败")
            
            # 等待一段时间让部署完成
            time.sleep(5)
            
            # 8. 测试停止部署
            print("8. 测试停止部署...")
            if deployer.stop_deployment(deployment_id):
                print("   ✓ 部署停止成功")
            else:
                print("   ✗ 部署停止失败")
            
            print("\n所有测试完成！")
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 运行测试
    test_model_deployer()