#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R3模型备份器
一个完整的机器学习模型备份、版本管理和部署系统
"""

import os
import json
import shutil
import pickle
import hashlib
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

# 可选导入，核心功能不依赖这些库
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ModelStatus(Enum):
    """模型状态枚举"""
    DRAFT = "draft"           # 草稿
    TRAINING = "training"     # 训练中
    VALIDATED = "validated"   # 已验证
    DEPLOYED = "deployed"     # 已部署
    ARCHIVED = "archived"     # 已归档
    FAILED = "failed"         # 失败


class DeploymentStatus(Enum):
    """部署状态枚举"""
    PENDING = "pending"       # 待部署
    RUNNING = "running"       # 运行中
    SUCCESS = "success"       # 成功
    FAILED = "failed"         # 失败
    ROLLED_BACK = "rolled_back"  # 已回滚


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_name: str
    model_version: str
    model_type: str
    framework: str
    created_at: str
    created_by: str
    description: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    file_path: str
    file_hash: str
    file_size: int
    status: ModelStatus
    tags: List[str]
    parent_version: Optional[str] = None


@dataclass
class DeploymentRecord:
    """部署记录"""
    deployment_id: str
    model_name: str
    model_version: str
    deployed_at: str
    deployed_by: str
    environment: str
    status: DeploymentStatus
    endpoint: Optional[str] = None
    notes: str = ""


@dataclass
class ModelValidation:
    """模型验证结果"""
    validation_id: str
    model_name: str
    model_version: str
    validation_date: str
    validation_type: str
    test_data_path: str
    metrics: Dict[str, float]
    passed: bool
    notes: str = ""


class ModelBackup:
    """模型备份器主类"""
    
    def __init__(self, backup_dir: str = "./model_backups"):
        """
        初始化模型备份器
        
        Args:
            backup_dir: 备份目录路径
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.backup_dir / "models").mkdir(exist_ok=True)
        (self.backup_dir / "metadata").mkdir(exist_ok=True)
        (self.backup_dir / "deployments").mkdir(exist_ok=True)
        (self.backup_dir / "validations").mkdir(exist_ok=True)
        (self.backup_dir / "monitoring").mkdir(exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 加载现有元数据
        self.metadata_file = self.backup_dir / "metadata" / "models_registry.json"
        self.metadata = self._load_metadata()
        
    def _setup_logging(self):
        """设置日志"""
        log_dir = self.backup_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "model_backup.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_metadata(self) -> Dict[str, ModelMetadata]:
        """加载模型元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {k: ModelMetadata(**v) for k, v in data.items()}
        return {}
    
    def _json_serializer(self, obj):
        """自定义JSON编码器处理枚举类型"""
        if hasattr(obj, 'value'):  # 枚举类型
            return obj.value
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def _save_metadata(self):
        """保存模型元数据"""
        data = {k: asdict(v) for k, v in self.metadata.items()}
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=self._json_serializer)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def register_model(self, 
                      model_path: str, 
                      model_name: str, 
                      framework: str = "unknown",
                      model_type: str = "unknown",
                      description: str = "",
                      created_by: str = "system",
                      parameters: Dict[str, Any] = None,
                      metrics: Dict[str, float] = None,
                      tags: List[str] = None) -> str:
        """
        注册模型文件
        
        Args:
            model_path: 模型文件路径
            model_name: 模型名称
            framework: 模型框架
            model_type: 模型类型
            description: 模型描述
            created_by: 创建者
            parameters: 模型参数
            metrics: 模型指标
            tags: 标签
            
        Returns:
            模型ID
        """
        try:
            # 生成模型版本
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_version = f"v{timestamp}"
            model_id = f"{model_name}_{model_version}"
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 创建元数据
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_version=model_version,
                model_type=model_type,
                framework=framework,
                description=description,
                created_by=created_by,
                created_at=datetime.datetime.now(),
                parameters=parameters or {},
                metrics=metrics or {},
                tags=tags or [],
                status=ModelStatus.DRAFT
            )
            
            # 保存元数据到注册表
            self.metadata[model_id] = metadata
            self._save_metadata()
            
            self.logger.info(f"模型注册成功: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"注册模型失败: {e}")
            raise
    
    def backup_model(self, 
                    model: Any, 
                    model_name: str, 
                    model_version: str,
                    model_type: str = "unknown",
                    framework: str = "unknown",
                    description: str = "",
                    created_by: str = "system",
                    parameters: Dict[str, Any] = None,
                    metrics: Dict[str, float] = None,
                    tags: List[str] = None,
                    parent_version: str = None) -> str:
        """
        备份模型
        
        Args:
            model: 要备份的模型对象
            model_name: 模型名称
            model_version: 模型版本
            model_type: 模型类型
            framework: 框架名称
            description: 描述
            created_by: 创建者
            parameters: 模型参数
            metrics: 性能指标
            tags: 标签
            parent_version: 父版本
            
        Returns:
            模型ID
        """
        model_id = f"{model_name}_{model_version}"
        
        # 检查模型是否已存在
        if model_id in self.metadata:
            self.logger.warning(f"模型 {model_id} 已存在，将被覆盖")
        
        # 创建模型目录
        model_dir = self.backup_dir / "models" / model_name / model_version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型文件
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # 计算文件哈希和大小
        file_hash = self._calculate_file_hash(str(model_file))
        file_size = model_file.stat().st_size
        
        # 创建元数据
        metadata = ModelMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            framework=framework,
            created_at=datetime.datetime.now().isoformat(),
            created_by=created_by,
            description=description,
            parameters=parameters or {},
            metrics=metrics or {},
            file_path=str(model_file),
            file_hash=file_hash,
            file_size=file_size,
            status=ModelStatus.DRAFT,
            tags=tags or [],
            parent_version=parent_version
        )
        
        # 保存元数据
        self.metadata[model_id] = metadata
        self._save_metadata()
        
        self.logger.info(f"模型 {model_id} 备份成功")
        return model_id
    
    def load_model(self, model_name: str, model_version: str) -> Optional[Any]:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            模型对象，如果不存在则返回None
        """
        model_id = f"{model_name}_{model_version}"
        
        if model_id not in self.metadata:
            self.logger.error(f"模型 {model_id} 不存在")
            return None
        
        metadata = self.metadata[model_id]
        
        try:
            with open(metadata.file_path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info(f"模型 {model_id} 加载成功")
            return model
        except Exception as e:
            self.logger.error(f"加载模型 {model_id} 失败: {e}")
            return None
    
    def list_models(self, model_name: str = None) -> List[ModelMetadata]:
        """
        列出模型
        
        Args:
            model_name: 模型名称过滤
            
        Returns:
            模型元数据列表
        """
        models = list(self.metadata.values())
        
        if model_name:
            models = [m for m in models if m.model_name == model_name]
        
        # 按创建时间排序
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models
    
    def update_model_status(self, model_name: str, model_version: str, status: ModelStatus):
        """
        更新模型状态
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            status: 新状态
        """
        model_id = f"{model_name}_{model_version}"
        
        if model_id not in self.metadata:
            self.logger.error(f"模型 {model_id} 不存在")
            return
        
        self.metadata[model_id].status = status
        self._save_metadata()
        self.logger.info(f"模型 {model_id} 状态更新为 {status.value}")
    
    def validate_model(self, 
                      model_name: str, 
                      model_version: str,
                      validation_data: Any,
                      validation_type: str = "basic",
                      test_data_path: str = "") -> ModelValidation:
        """
        验证模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            validation_data: 验证数据
            validation_type: 验证类型
            test_data_path: 测试数据路径
            
        Returns:
            验证结果
        """
        model_id = f"{model_name}_{model_version}"
        
        if model_id not in self.metadata:
            raise ValueError(f"模型 {model_id} 不存在")
        
        model = self.load_model(model_name, model_version)
        if model is None:
            raise ValueError(f"无法加载模型 {model_id}")
        
        # 执行模型验证
        validation_result = self._perform_validation(model, validation_data, validation_type)
        
        # 创建验证记录
        validation_id = f"val_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        validation = ModelValidation(
            validation_id=validation_id,
            model_name=model_name,
            model_version=model_version,
            validation_date=datetime.datetime.now().isoformat(),
            validation_type=validation_type,
            test_data_path=test_data_path,
            metrics=validation_result['metrics'],
            passed=validation_result['passed'],
            notes=validation_result.get('notes', '')
        )
        
        # 保存验证记录
        validation_file = self.backup_dir / "validations" / f"{validation_id}.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(validation), f, ensure_ascii=False, indent=2, default=self._json_serializer)
        
        # 更新模型状态
        if validation_result['passed']:
            self.update_model_status(model_name, model_version, ModelStatus.VALIDATED)
        else:
            self.update_model_status(model_name, model_version, ModelStatus.FAILED)
        
        self.logger.info(f"模型 {model_id} 验证完成，结果: {'通过' if validation_result['passed'] else '失败'}")
        return validation
    
    def _perform_validation(self, model: Any, validation_data: Any, validation_type: str) -> Dict[str, Any]:
        """
        执行模型验证
        
        Args:
            model: 模型对象
            validation_data: 验证数据
            validation_type: 验证类型
            
        Returns:
            验证结果
        """
        # 基础验证逻辑
        if validation_type == "basic":
            try:
                # 尝试预测
                if hasattr(model, 'predict'):
                    predictions = model.predict(validation_data)
                    
                    # 基本检查
                    has_nan = False
                    if HAS_NUMPY and isinstance(predictions, np.ndarray):
                        has_nan = np.isnan(predictions).any()
                    elif hasattr(predictions, '__iter__') and not isinstance(predictions, str):
                        try:
                            has_nan = any(p != p for p in predictions)  # 检查NaN值
                        except:
                            has_nan = False
                    
                    metrics = {
                        'prediction_count': len(predictions),
                        'has_predictions': len(predictions) > 0,
                        'no_nan': not has_nan
                    }
                    
                    passed = metrics['has_predictions'] and metrics['no_nan']
                    
                    return {
                        'metrics': metrics,
                        'passed': passed,
                        'notes': '基础验证通过' if passed else '基础验证失败'
                    }
                else:
                    return {
                        'metrics': {'has_model': True},
                        'passed': True,
                        'notes': '模型对象验证通过'
                    }
            except Exception as e:
                return {
                    'metrics': {'error': str(e)},
                    'passed': False,
                    'notes': f'验证失败: {e}'
                }
        
        return {
            'metrics': {},
            'passed': False,
            'notes': f'不支持的验证类型: {validation_type}'
        }
    
    def deploy_model(self, 
                    model_name: str, 
                    model_version: str,
                    environment: str = "production",
                    deployed_by: str = "system",
                    endpoint: str = None,
                    notes: str = "") -> str:
        """
        部署模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            environment: 部署环境
            deployed_by: 部署者
            endpoint: 端点地址
            notes: 备注
            
        Returns:
            部署ID
        """
        model_id = f"{model_name}_{model_version}"
        
        if model_id not in self.metadata:
            raise ValueError(f"模型 {model_id} 不存在")
        
        metadata = self.metadata[model_id]
        
        # 检查模型状态
        if metadata.status != ModelStatus.VALIDATED:
            self.logger.warning(f"模型 {model_id} 状态为 {metadata.status.value}，建议先验证")
        
        # 创建部署记录
        deployment_id = f"dep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        deployment = DeploymentRecord(
            deployment_id=deployment_id,
            model_name=model_name,
            model_version=model_version,
            deployed_at=datetime.datetime.now().isoformat(),
            deployed_by=deployed_by,
            environment=environment,
            status=DeploymentStatus.SUCCESS,
            endpoint=endpoint,
            notes=notes
        )
        
        # 保存部署记录
        deployment_file = self.backup_dir / "deployments" / f"{deployment_id}.json"
        with open(deployment_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(deployment), f, ensure_ascii=False, indent=2, default=self._json_serializer)
        
        # 更新模型状态
        self.update_model_status(model_name, model_version, ModelStatus.DEPLOYED)
        
        self.logger.info(f"模型 {model_id} 部署成功，部署ID: {deployment_id}")
        return deployment_id
    
    def rollback_model(self, 
                      model_name: str, 
                      target_version: str,
                      environment: str = "production",
                      rolled_back_by: str = "system",
                      notes: str = "") -> bool:
        """
        回滚模型到指定版本
        
        Args:
            model_name: 模型名称
            target_version: 目标版本
            environment: 部署环境
            rolled_back_by: 回滚者
            notes: 备注
            
        Returns:
            是否成功
        """
        target_model_id = f"{model_name}_{target_version}"
        
        if target_model_id not in self.metadata:
            self.logger.error(f"目标模型 {target_model_id} 不存在")
            return False
        
        # 获取目标模型
        target_metadata = self.metadata[target_model_id]
        
        try:
            # 创建回滚部署记录
            rollback_deployment_id = f"rollback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            rollback_deployment = DeploymentRecord(
                deployment_id=rollback_deployment_id,
                model_name=model_name,
                model_version=target_version,
                deployed_at=datetime.datetime.now().isoformat(),
                deployed_by=rolled_back_by,
                environment=environment,
                status=DeploymentStatus.ROLLED_BACK,
                notes=f"回滚到版本 {target_version}. {notes}"
            )
            
            # 保存回滚记录
            rollback_file = self.backup_dir / "deployments" / f"{rollback_deployment_id}.json"
            with open(rollback_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(rollback_deployment), f, ensure_ascii=False, indent=2, default=self._json_serializer)
            
            # 更新模型状态
            self.update_model_status(model_name, target_version, ModelStatus.DEPLOYED)
            
            self.logger.info(f"模型 {model_name} 回滚到版本 {target_version} 成功")
            return True
            
        except Exception as e:
            self.logger.error(f"模型回滚失败: {e}")
            return False
    
    def compare_models(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        比较两个模型版本
        
        Args:
            model_name: 模型名称
            version1: 版本1
            version2: 版本2
            
        Returns:
            比较结果
        """
        model_id1 = f"{model_name}_{version1}"
        model_id2 = f"{model_name}_{version2}"
        
        if model_id1 not in self.metadata or model_id2 not in self.metadata:
            raise ValueError("指定的模型版本不存在")
        
        metadata1 = self.metadata[model_id1]
        metadata2 = self.metadata[model_id2]
        
        # 比较元数据
        comparison = {
            'model_name': model_name,
            'version1': version1,
            'version2': version2,
            'created_at1': metadata1.created_at,
            'created_at2': metadata2.created_at,
            'created_by1': metadata1.created_by,
            'created_by2': metadata2.created_by,
            'framework1': metadata1.framework,
            'framework2': metadata2.framework,
            'status1': metadata1.status.value,
            'status2': metadata2.status.value,
            'file_size1': metadata1.file_size,
            'file_size2': metadata2.file_size,
            'metrics_comparison': {},
            'parameters_comparison': {}
        }
        
        # 比较性能指标
        for metric in set(metadata1.metrics.keys()) | set(metadata2.metrics.keys()):
            val1 = metadata1.metrics.get(metric, 0)
            val2 = metadata2.metrics.get(metric, 0)
            comparison['metrics_comparison'][metric] = {
                'version1': val1,
                'version2': val2,
                'difference': val2 - val1,
                'improvement': ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf')
            }
        
        # 比较参数
        for param in set(metadata1.parameters.keys()) | set(metadata2.parameters.keys()):
            val1 = metadata1.parameters.get(param, None)
            val2 = metadata2.parameters.get(param, None)
            comparison['parameters_comparison'][param] = {
                'version1': val1,
                'version2': val2,
                'changed': val1 != val2
            }
        
        self.logger.info(f"模型 {model_name} 版本 {version1} 和 {version2} 比较完成")
        return comparison
    
    def monitor_model(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """
        监控模型性能
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            监控结果
        """
        model_id = f"{model_name}_{model_version}"
        
        if model_id not in self.metadata:
            raise ValueError(f"模型 {model_id} 不存在")
        
        metadata = self.metadata[model_id]
        
        # 获取相关部署记录
        deployment_files = list((self.backup_dir / "deployments").glob("*.json"))
        deployments = []
        
        for file in deployment_files:
            with open(file, 'r', encoding='utf-8') as f:
                dep_data = json.load(f)
                if (dep_data['model_name'] == model_name and 
                    dep_data['model_version'] == model_version):
                    deployments.append(dep_data)
        
        # 获取验证记录
        validation_files = list((self.backup_dir / "validations").glob("*.json"))
        validations = []
        
        for file in validation_files:
            with open(file, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
                if (val_data['model_name'] == model_name and 
                    val_data['model_version'] == model_version):
                    validations.append(val_data)
        
        # 生成监控报告
        monitoring_result = {
            'model_info': {
                'model_name': model_name,
                'model_version': model_version,
                'status': metadata.status.value,
                'created_at': metadata.created_at,
                'framework': metadata.framework,
                'tags': metadata.tags
            },
            'performance_metrics': metadata.metrics,
            'deployment_history': deployments,
            'validation_history': validations,
            'health_score': self._calculate_health_score(metadata, deployments, validations),
            'recommendations': self._generate_recommendations(metadata, deployments, validations)
        }
        
        # 保存监控结果
        monitor_file = self.backup_dir / "monitoring" / f"{model_id}_monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(monitor_file, 'w', encoding='utf-8') as f:
            json.dump(monitoring_result, f, ensure_ascii=False, indent=2, default=self._json_serializer)
        
        self.logger.info(f"模型 {model_id} 监控完成")
        return monitoring_result
    
    def _calculate_health_score(self, metadata: ModelMetadata, deployments: List, validations: List) -> float:
        """计算模型健康评分"""
        score = 100.0
        
        # 根据状态扣分
        status_penalties = {
            ModelStatus.DRAFT: 20,
            ModelStatus.TRAINING: 10,
            ModelStatus.VALIDATED: 0,
            ModelStatus.DEPLOYED: 0,
            ModelStatus.ARCHIVED: 30,
            ModelStatus.FAILED: 50
        }
        score -= status_penalties.get(metadata.status, 0)
        
        # 根据验证结果扣分
        failed_validations = sum(1 for v in validations if not v['passed'])
        score -= failed_validations * 10
        
        # 根据部署失败扣分
        failed_deployments = sum(1 for d in deployments if d['status'] == 'failed')
        score -= failed_deployments * 15
        
        return max(0.0, score)
    
    def _generate_recommendations(self, metadata: ModelMetadata, deployments: List, validations: List) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if metadata.status == ModelStatus.DRAFT:
            recommendations.append("模型处于草稿状态，建议进行验证")
        
        if metadata.status == ModelStatus.FAILED:
            recommendations.append("模型验证失败，建议检查模型质量和训练数据")
        
        if not validations:
            recommendations.append("模型尚未进行验证，建议执行验证")
        
        failed_validations = sum(1 for v in validations if not v['passed'])
        if failed_validations > 0:
            recommendations.append(f"模型有 {failed_validations} 次验证失败，建议重新训练")
        
        failed_deployments = sum(1 for d in deployments if d['status'] == 'failed')
        if failed_deployments > 0:
            recommendations.append(f"模型有 {failed_deployments} 次部署失败，建议检查部署环境")
        
        if not recommendations:
            recommendations.append("模型运行良好，无需特殊处理")
        
        return recommendations
    
    def delete_model(self, model_name: str, model_version: str) -> bool:
        """
        删除模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            是否成功
        """
        model_id = f"{model_name}_{model_version}"
        
        if model_id not in self.metadata:
            self.logger.error(f"模型 {model_id} 不存在")
            return False
        
        try:
            # 删除模型文件
            metadata = self.metadata[model_id]
            if os.path.exists(metadata.file_path):
                shutil.rmtree(os.path.dirname(metadata.file_path))
            
            # 从元数据中删除
            del self.metadata[model_id]
            self._save_metadata()
            
            self.logger.info(f"模型 {model_id} 删除成功")
            return True
            
        except Exception as e:
            self.logger.error(f"删除模型 {model_id} 失败: {e}")
            return False
    
    def get_model_info(self, model_name: str, model_version: str) -> Optional[Dict[str, Any]]:
        """
        获取模型详细信息
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            模型信息字典
        """
        model_id = f"{model_name}_{model_version}"
        
        if model_id not in self.metadata:
            return None
        
        metadata = self.metadata[model_id]
        info = asdict(metadata)
        
        # 处理枚举类型，转换为字符串值
        if 'status' in info:
            info['status'] = info['status'].value if hasattr(info['status'], 'value') else str(info['status'])
        
        return info
    
    def export_model_registry(self, export_path: str) -> bool:
        """
        导出模型注册表
        
        Args:
            export_path: 导出路径
            
        Returns:
            是否成功
        """
        try:
            export_data = {
                'export_time': datetime.datetime.now().isoformat(),
                'total_models': len(self.metadata),
                'models': {k: asdict(v) for k, v in self.metadata.items()}
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=self._json_serializer)
            
            self.logger.info(f"模型注册表导出成功: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出模型注册表失败: {e}")
            return False
    
    def import_model_registry(self, import_path: str, merge: bool = True) -> bool:
        """
        导入模型注册表
        
        Args:
            import_path: 导入路径
            merge: 是否合并现有数据
            
        Returns:
            是否成功
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if not merge:
                self.metadata = {}
            
            # 导入模型元数据
            for model_id, model_data in import_data['models'].items():
                if merge and model_id in self.metadata:
                    self.logger.warning(f"模型 {model_id} 已存在，将被覆盖")
                self.metadata[model_id] = ModelMetadata(**model_data)
            
            self._save_metadata()
            self.logger.info(f"模型注册表导入成功: {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导入模型注册表失败: {e}")
            return False