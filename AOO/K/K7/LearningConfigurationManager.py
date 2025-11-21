"""
K7学习配置管理器

这是一个全面的机器学习配置管理系统，支持模型配置、训练、验证、评估、
部署、更新和实验管理的完整生命周期。

作者: K7团队
版本: 1.0.0
创建日期: 2025-11-06
"""

import asyncio
import json
import logging
import os
import pickle
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Type, Generic, TypeVar
from collections import defaultdict, deque

# 可选依赖
try:
    import yaml
except ImportError:
    yaml = None

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    jsonschema = None
    ValidationError = Exception

try:
    import aiofiles
except ImportError:
    aiofiles = None

try:
    from pydantic import BaseModel, Field, validator, root_validator
except ImportError:
    from typing import Generic
    BaseModel = object
    Field = lambda **kwargs: lambda x: x
    validator = lambda **kwargs: lambda x: x
    root_validator = lambda **kwargs: lambda x: x

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('k7_learning_config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 类型定义
T = TypeVar('T')
ConfigType = TypeVar('ConfigType', bound='BaseConfiguration')
ModelType = TypeVar('ModelType', bound='MLModelConfiguration')


class ConfigurationError(Exception):
    """配置相关错误基类"""
    pass


class ValidationError(ConfigurationError):
    """配置验证错误"""
    pass


class DeploymentError(ConfigurationError):
    """部署相关错误"""
    pass


class ExperimentError(ConfigurationError):
    """实验相关错误"""
    pass


class AsyncProcessingError(ConfigurationError):
    """异步处理错误"""
    pass


class ConfigurationStatus(Enum):
    """配置状态枚举"""
    DRAFT = "draft"
    VALIDATED = "validated"
    ACTIVE = "active"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentStatus(Enum):
    """部署状态枚举"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class ExperimentStatus(Enum):
    """实验状态枚举"""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelTypeEnum(Enum):
    """模型类型枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"


class BaseConfiguration(object if BaseModel is object else BaseModel):
    """基础配置类"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="配置名称")
    description: Optional[str] = Field(None, description="配置描述")
    version: str = Field(default="1.0.0", description="配置版本")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    status: ConfigurationStatus = Field(default=ConfigurationStatus.DRAFT, description="配置状态")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ConfigurationStatus: lambda v: v.value
        }
    
    def update_timestamp(self):
        """更新时间戳"""
        self.updated_at = datetime.now()
    
    def validate_config(self) -> bool:
        """验证配置"""
        try:
            self.validate_configurations()
            self.status = ConfigurationStatus.VALIDATED
            return True
        except ValidationError as e:
            logger.error(f"配置验证失败: {e}")
            self.status = ConfigurationStatus.FAILED
            return False
    
    @abstractmethod
    def validate_configurations(self):
        """验证配置的具体实现"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return self.json()
    
    def save_to_file(self, filepath: Union[str, Path]):
        """保存到文件"""
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, indent=2, ensure_ascii=False, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'BaseConfiguration':
        """从文件加载"""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.parse_obj(data)


class MLModelConfiguration(BaseConfiguration):
    """机器学习模型配置"""
    
    model_type: ModelTypeEnum = Field(..., description="模型类型")
    model_architecture: Dict[str, Any] = Field(..., description="模型架构")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="超参数")
    input_spec: Dict[str, Any] = Field(..., description="输入规范")
    output_spec: Dict[str, Any] = Field(..., description="输出规范")
    dependencies: List[str] = Field(default_factory=list, description="依赖库")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="资源需求")
    model_size: Optional[int] = Field(None, description="模型大小(MB)")
    inference_time: Optional[float] = Field(None, description="推理时间(秒)")
    
    def validate_configurations(self):
        """验证模型配置"""
        if not self.model_architecture:
            raise ValidationError("模型架构不能为空")
        
        if not self.input_spec:
            raise ValidationError("输入规范不能为空")
        
        if not self.output_spec:
            raise ValidationError("输出规范不能为空")
        
        # 验证超参数类型
        for key, value in self.hyperparameters.items():
            if not isinstance(value, (int, float, str, bool, list)):
                raise ValidationError(f"超参数 {key} 的类型不支持: {type(value)}")
        
        # 验证模型类型
        if self.model_type not in ModelTypeEnum:
            raise ValidationError(f"不支持的模型类型: {self.model_type}")


class TrainingConfiguration(BaseConfiguration):
    """训练配置"""
    
    model_config_id: str = Field(..., description="关联的模型配置ID")
    training_data: Dict[str, Any] = Field(..., description="训练数据配置")
    validation_data: Dict[str, Any] = Field(default_factory=dict, description="验证数据配置")
    training_parameters: Dict[str, Any] = Field(..., description="训练参数")
    training_strategy: Dict[str, Any] = Field(default_factory=dict, description="训练策略")
    optimizer: Dict[str, Any] = Field(default_factory=dict, description="优化器配置")
    loss_function: Dict[str, Any] = Field(default_factory=dict, description="损失函数配置")
    learning_rate_schedule: Dict[str, Any] = Field(default_factory=dict, description="学习率调度")
    early_stopping: Dict[str, Any] = Field(default_factory=dict, description="早停配置")
    checkpoint_config: Dict[str, Any] = Field(default_factory=dict, description="检查点配置")
    distributed_training: Dict[str, Any] = Field(default_factory=dict, description="分布式训练配置")
    
    def validate_configurations(self):
        """验证训练配置"""
        if not self.training_data:
            raise ValidationError("训练数据配置不能为空")
        
        if not self.training_parameters:
            raise ValidationError("训练参数不能为空")
        
        # 验证必要的训练参数
        required_params = ['batch_size', 'epochs', 'learning_rate']
        for param in required_params:
            if param not in self.training_parameters:
                raise ValidationError(f"缺少必要的训练参数: {param}")
        
        # 验证数据类型
        batch_size = self.training_parameters.get('batch_size')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValidationError("批次大小必须是正整数")
        
        epochs = self.training_parameters.get('epochs')
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValidationError("训练轮数必须是正整数")
        
        learning_rate = self.training_parameters.get('learning_rate')
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValidationError("学习率必须是正数")


class ValidationConfiguration(BaseConfiguration):
    """验证配置"""
    
    training_config_id: str = Field(..., description="关联的训练配置ID")
    validation_data: Dict[str, Any] = Field(..., description="验证数据")
    validation_metrics: List[str] = Field(..., description="验证指标")
    validation_strategy: Dict[str, Any] = Field(default_factory=dict, description="验证策略")
    cross_validation: Dict[str, Any] = Field(default_factory=dict, description="交叉验证配置")
    validation_split: float = Field(default=0.2, description="验证集比例", ge=0.0, le=1.0)
    stratified_split: bool = Field(default=False, description="分层抽样")
    custom_metrics: Dict[str, Any] = Field(default_factory=dict, description="自定义指标")
    
    def validate_configurations(self):
        """验证验证配置"""
        if not self.validation_data:
            raise ValidationError("验证数据配置不能为空")
        
        if not self.validation_metrics:
            raise ValidationError("验证指标不能为空")
        
        # 验证验证指标
        valid_metrics = {
            'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
            'mse', 'mae', 'rmse', 'r2_score', 'custom'
        }
        for metric in self.validation_metrics:
            if metric not in valid_metrics:
                logger.warning(f"未知的验证指标: {metric}")
        
        # 验证验证集比例
        if not 0.0 <= self.validation_split <= 1.0:
            raise ValidationError("验证集比例必须在0到1之间")


class EvaluationConfiguration(BaseConfiguration):
    """模型评估配置"""
    
    model_config_id: str = Field(..., description="关联的模型配置ID")
    evaluation_data: Dict[str, Any] = Field(..., description="评估数据")
    evaluation_metrics: List[str] = Field(..., description="评估指标")
    evaluation_methods: Dict[str, Any] = Field(default_factory=dict, description="评估方法")
    evaluation_report: Dict[str, Any] = Field(default_factory=dict, description="评估报告配置")
    benchmark_comparison: Dict[str, Any] = Field(default_factory=dict, description="基准对比配置")
    performance_thresholds: Dict[str, float] = Field(default_factory=dict, description="性能阈值")
    evaluation_frequency: str = Field(default="daily", description="评估频率")
    
    def validate_configurations(self):
        """验证评估配置"""
        if not self.evaluation_data:
            raise ValidationError("评估数据配置不能为空")
        
        if not self.evaluation_metrics:
            raise ValidationError("评估指标不能为空")
        
        # 验证评估频率
        valid_frequencies = ['hourly', 'daily', 'weekly', 'monthly', 'on_demand']
        if self.evaluation_frequency not in valid_frequencies:
            raise ValidationError(f"不支持的评估频率: {self.evaluation_frequency}")


class DeploymentConfiguration(BaseConfiguration):
    """模型部署配置"""
    
    model_config_id: str = Field(..., description="关联的模型配置ID")
    deployment_environment: Dict[str, Any] = Field(..., description="部署环境")
    deployment_strategy: Dict[str, Any] = Field(default_factory=dict, description="部署策略")
    deployment_platform: str = Field(..., description="部署平台")
    scaling_config: Dict[str, Any] = Field(default_factory=dict, description="扩缩容配置")
    monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="监控配置")
    security_config: Dict[str, Any] = Field(default_factory=dict, description="安全配置")
    rollback_config: Dict[str, Any] = Field(default_factory=dict, description="回滚配置")
    resource_allocation: Dict[str, Any] = Field(default_factory=dict, description="资源分配")
    deployment_status: DeploymentStatus = Field(default=DeploymentStatus.PENDING, description="部署状态")
    
    def validate_configurations(self):
        """验证部署配置"""
        if not self.deployment_environment:
            raise ValidationError("部署环境配置不能为空")
        
        if not self.deployment_platform:
            raise ValidationError("部署平台不能为空")
        
        # 验证部署平台
        valid_platforms = ['kubernetes', 'docker', 'aws_sagemaker', 'azure_ml', 'gcp_ai_platform', 'local']
        if self.deployment_platform not in valid_platforms:
            raise ValidationError(f"不支持的部署平台: {self.deployment_platform}")


class ModelUpdateConfiguration(BaseConfiguration):
    """模型更新配置"""
    
    model_config_id: str = Field(..., description="关联的模型配置ID")
    update_strategy: Dict[str, Any] = Field(..., description="更新策略")
    version_management: Dict[str, Any] = Field(default_factory=dict, description="版本管理")
    rollback_mechanism: Dict[str, Any] = Field(default_factory=dict, description="回滚机制")
    update_frequency: str = Field(default="manual", description="更新频率")
    auto_update: bool = Field(default=False, description="自动更新")
    update_criteria: Dict[str, Any] = Field(default_factory=dict, description="更新条件")
    notification_config: Dict[str, Any] = Field(default_factory=dict, description="通知配置")
    
    def validate_configurations(self):
        """验证更新配置"""
        if not self.update_strategy:
            raise ValidationError("更新策略不能为空")
        
        # 验证更新频率
        valid_frequencies = ['manual', 'daily', 'weekly', 'monthly', 'on_performance_degradation']
        if self.update_frequency not in valid_frequencies:
            raise ValidationError(f"不支持的更新频率: {self.update_frequency}")


class ExperimentConfiguration(BaseConfiguration):
    """实验配置"""
    
    experiment_name: str = Field(..., description="实验名称")
    experiment_type: str = Field(..., description="实验类型")
    hypothesis: Optional[str] = Field(None, description="实验假设")
    variables: Dict[str, Any] = Field(default_factory=dict, description="实验变量")
    controls: Dict[str, Any] = Field(default_factory=dict, description="控制变量")
    metrics: List[str] = Field(default_factory=list, description="实验指标")
    experimental_design: Dict[str, Any] = Field(default_factory=dict, description="实验设计")
    sample_size: Optional[int] = Field(None, description="样本大小")
    statistical_power: Optional[float] = Field(None, description="统计功效", ge=0.0, le=1.0)
    alpha_level: float = Field(default=0.05, description="显著性水平", ge=0.0, le=1.0)
    experiment_status: ExperimentStatus = Field(default=ExperimentStatus.PLANNED, description="实验状态")
    results: Dict[str, Any] = Field(default_factory=dict, description="实验结果")
    
    def validate_configurations(self):
        """验证实验配置"""
        if not self.experiment_name:
            raise ValidationError("实验名称不能为空")
        
        if not self.experiment_type:
            raise ValidationError("实验类型不能为空")
        
        # 验证统计参数
        if self.statistical_power is not None:
            if not 0.0 <= self.statistical_power <= 1.0:
                raise ValidationError("统计功效必须在0到1之间")
        
        if not 0.0 <= self.alpha_level <= 1.0:
            raise ValidationError("显著性水平必须在0到1之间")
        
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValidationError("样本大小必须是正数")


class AsyncConfigurationProcessor:
    """异步配置处理器"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = asyncio.Queue()
        self.results = {}
        self._lock = asyncio.Lock()
    
    async def process_configurations_async(
        self,
        configurations: List[BaseConfiguration],
        process_func: Callable[[BaseConfiguration], Any],
        batch_size: int = 5
    ) -> Dict[str, Any]:
        """异步处理配置"""
        results = {}
        
        # 分批处理
        for i in range(0, len(configurations), batch_size):
            batch = configurations[i:i + batch_size]
            batch_results = await self._process_batch(batch, process_func)
            results.update(batch_results)
        
        return results
    
    async def _process_batch(
        self,
        batch: List[BaseConfiguration],
        process_func: Callable[[BaseConfiguration], Any]
    ) -> Dict[str, Any]:
        """处理一批配置"""
        tasks = []
        for config in batch:
            task = asyncio.create_task(self._process_single(config, process_func))
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for config, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.error(f"处理配置 {config.id} 时出错: {result}")
                results[config.id] = {"error": str(result)}
            else:
                results[config.id] = result
        
        return results
    
    async def _process_single(
        self,
        config: BaseConfiguration,
        process_func: Callable[[BaseConfiguration], Any]
    ) -> Any:
        """处理单个配置"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, process_func, config)
    
    async def validate_configurations_async(
        self,
        configurations: List[BaseConfiguration]
    ) -> Dict[str, bool]:
        """异步验证配置"""
        async def validate_single(config: BaseConfiguration) -> bool:
            try:
                return config.validate_config()
            except Exception as e:
                logger.error(f"验证配置 {config.id} 时出错: {e}")
                return False
        
        tasks = [validate_single(config) for config in configurations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        validation_results = {}
        for config, result in zip(configurations, results):
            if isinstance(result, Exception):
                validation_results[config.id] = False
            else:
                validation_results[config.id] = result
        
        return validation_results
    
    async def save_configurations_async(
        self,
        configurations: List[BaseConfiguration],
        output_dir: Union[str, Path]
    ) -> Dict[str, bool]:
        """异步保存配置"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        async def save_single(config: BaseConfiguration) -> bool:
            try:
                filepath = output_dir / f"{config.id}.json"
                if aiofiles is not None:
                    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                        await f.write(config.to_json())
                else:
                    # 同步保存作为后备
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(config.to_json())
                return True
            except Exception as e:
                logger.error(f"保存配置 {config.id} 时出错: {e}")
                return False
        
        tasks = [save_single(config) for config in configurations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        save_results = {}
        for config, result in zip(configurations, results):
            if isinstance(result, Exception):
                save_results[config.id] = False
            else:
                save_results[config.id] = result
        
        return save_results
    
    def shutdown(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)


class ConfigurationValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_json_schema(config_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """验证JSON模式"""
        if jsonschema is None:
            logger.warning("jsonschema未安装，跳过JSON模式验证")
            return True
        try:
            jsonschema.validate(config_data, schema)
            return True
        except ValidationError as e:
            logger.error(f"JSON模式验证失败: {e}")
            return False
    
    @staticmethod
    def validate_dependencies(dependencies: List[str]) -> bool:
        """验证依赖"""
        missing_deps = []
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.error(f"缺少依赖库: {missing_deps}")
            return False
        return True
    
    @staticmethod
    def validate_resource_requirements(requirements: Dict[str, Any]) -> bool:
        """验证资源需求"""
        required_resources = ['cpu', 'memory', 'storage']
        for resource in required_resources:
            if resource not in requirements:
                logger.warning(f"缺少资源需求: {resource}")
                return False
        
        # 验证资源值
        cpu = requirements.get('cpu')
        if cpu and (not isinstance(cpu, (int, float)) or cpu <= 0):
            logger.error("CPU需求必须是正数")
            return False
        
        memory = requirements.get('memory')
        if memory and (not isinstance(memory, (int, float)) or memory <= 0):
            logger.error("内存需求必须是正数")
            return False
        
        storage = requirements.get('storage')
        if storage and (not isinstance(storage, (int, float)) or storage <= 0):
            logger.error("存储需求必须是正数")
            return False
        
        return True
    
    @staticmethod
    def validate_hyperparameters(hyperparameters: Dict[str, Any]) -> bool:
        """验证超参数"""
        for key, value in hyperparameters.items():
            # 检查参数名
            if not isinstance(key, str) or not key:
                logger.error(f"超参数名无效: {key}")
                return False
            
            # 检查参数值类型
            if not isinstance(value, (int, float, str, bool, list, dict)):
                logger.error(f"超参数 {key} 的值类型不支持: {type(value)}")
                return False
            
            # 检查数值范围
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    logger.error(f"超参数 {key} 的值无效: {value}")
                    return False
        
        return True


class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self, storage_path: Union[str, Path] = "configs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 配置存储
        self._configurations: Dict[str, BaseConfiguration] = {}
        self._config_schemas: Dict[str, Dict[str, Any]] = {}
        self._validation_rules: Dict[str, Callable] = {}
        
        # 异步处理器
        self.async_processor = AsyncConfigurationProcessor()
        
        # 验证器
        self.validator = ConfigurationValidator()
        
        # 加载现有配置
        self._load_configurations()
        
        logger.info(f"配置管理器初始化完成，存储路径: {self.storage_path}")
    
    def register_configuration_type(
        self,
        config_type: Type[BaseConfiguration],
        schema: Optional[Dict[str, Any]] = None,
        validation_rule: Optional[Callable] = None
    ):
        """注册配置类型"""
        type_name = config_type.__name__
        self._config_schemas[type_name] = schema
        if validation_rule:
            self._validation_rules[type_name] = validation_rule
        logger.info(f"注册配置类型: {type_name}")
    
    def create_configuration(
        self,
        config_type: Type[ConfigType],
        **kwargs
    ) -> ConfigType:
        """创建配置"""
        try:
            config = config_type(**kwargs)
            config.validate_config()
            self._configurations[config.id] = config
            self._save_configuration(config)
            logger.info(f"创建配置成功: {config.id}")
            return config
        except Exception as e:
            logger.error(f"创建配置失败: {e}")
            raise ConfigurationError(f"创建配置失败: {e}")
    
    def get_configuration(self, config_id: str) -> Optional[BaseConfiguration]:
        """获取配置"""
        return self._configurations.get(config_id)
    
    def update_configuration(
        self,
        config_id: str,
        **updates
    ) -> Optional[BaseConfiguration]:
        """更新配置"""
        if config_id not in self._configurations:
            raise ConfigurationError(f"配置不存在: {config_id}")
        
        config = self._configurations[config_id]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.update_timestamp()
        config.validate_config()
        self._save_configuration(config)
        logger.info(f"更新配置成功: {config_id}")
        return config
    
    def delete_configuration(self, config_id: str) -> bool:
        """删除配置"""
        if config_id not in self._configurations:
            logger.warning(f"配置不存在: {config_id}")
            return False
        
        config = self._configurations[config_id]
        del self._configurations[config_id]
        
        # 删除文件
        config_file = self.storage_path / f"{config_id}.json"
        if config_file.exists():
            config_file.unlink()
        
        logger.info(f"删除配置成功: {config_id}")
        return True
    
    def list_configurations(
        self,
        config_type: Optional[Type[BaseConfiguration]] = None,
        status: Optional[ConfigurationStatus] = None
    ) -> List[BaseConfiguration]:
        """列出配置"""
        configs = list(self._configurations.values())
        
        if config_type:
            configs = [c for c in configs if isinstance(c, config_type)]
        
        if status:
            configs = [c for c in configs if c.status == status]
        
        return configs
    
    def search_configurations(
        self,
        query: str,
        fields: Optional[List[str]] = None
    ) -> List[BaseConfiguration]:
        """搜索配置"""
        if not fields:
            fields = ['name', 'description', 'metadata']
        
        results = []
        query_lower = query.lower()
        
        for config in self._configurations.values():
            for field in fields:
                if hasattr(config, field):
                    value = str(getattr(config, field)).lower()
                    if query_lower in value:
                        results.append(config)
                        break
        
        return results
    
    def validate_configuration(self, config: BaseConfiguration) -> bool:
        """验证配置"""
        try:
            # 基本验证
            config.validate_config()
            
            # JSON模式验证
            config_type_name = type(config).__name__
            if config_type_name in self._config_schemas:
                schema = self._config_schemas[config_type_name]
                if schema:
                    if not self.validator.validate_json_schema(config.dict(), schema):
                        return False
            
            # 自定义验证规则
            if config_type_name in self._validation_rules:
                rule = self._validation_rules[config_type_name]
                if not rule(config):
                    return False
            
            return True
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    async def validate_configurations_async(
        self,
        configurations: List[BaseConfiguration]
    ) -> Dict[str, bool]:
        """异步验证配置"""
        return await self.async_processor.validate_configurations_async(configurations)
    
    async def save_configurations_async(
        self,
        configurations: List[BaseConfiguration],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, bool]:
        """异步保存配置"""
        if output_dir is None:
            output_dir = self.storage_path
        return await self.async_processor.save_configurations_async(configurations, output_dir)
    
    def export_configurations(
        self,
        config_ids: List[str],
        output_file: Union[str, Path],
        format: str = "json"
    ) -> bool:
        """导出配置"""
        try:
            configs = [self._configurations[config_id] for config_id in config_ids 
                      if config_id in self._configurations]
            
            if format.lower() == "json":
                data = [config.dict() for config in configs]
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            elif format.lower() == "yaml":
                if yaml is None:
                    raise ValueError("yaml未安装，无法导出YAML格式")
                data = [config.dict() for config in configs]
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"导出配置成功: {len(configs)} 个配置")
            return True
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return False
    
    def import_configurations(
        self,
        input_file: Union[str, Path],
        config_type: Type[ConfigType]
    ) -> List[ConfigType]:
        """导入配置"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                if str(input_file).endswith('.yaml') or str(input_file).endswith('.yml'):
                    if yaml is None:
                        raise ValueError("yaml未安装，无法导入YAML文件")
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            configurations = []
            for item in data:
                config = config_type.parse_obj(item)
                config.validate_config()
                self._configurations[config.id] = config
                self._save_configuration(config)
                configurations.append(config)
            
            logger.info(f"导入配置成功: {len(configurations)} 个配置")
            return configurations
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            raise ConfigurationError(f"导入配置失败: {e}")
    
    def _load_configurations(self):
        """加载配置"""
        try:
            for config_file in self.storage_path.glob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 根据数据确定配置类型
                    config_type = self._determine_config_type(data)
                    if config_type:
                        config = config_type.parse_obj(data)
                        self._configurations[config.id] = config
                except Exception as e:
                    logger.error(f"加载配置文件失败 {config_file}: {e}")
            
            logger.info(f"加载配置完成: {len(self._configurations)} 个配置")
        except Exception as e:
            logger.error(f"加载配置时出错: {e}")
    
    def _determine_config_type(self, data: Dict[str, Any]) -> Optional[Type[BaseConfiguration]]:
        """根据数据确定配置类型"""
        type_mapping = {
            'model_type': MLModelConfiguration,
            'training_data': TrainingConfiguration,
            'validation_data': ValidationConfiguration,
            'evaluation_data': EvaluationConfiguration,
            'deployment_environment': DeploymentConfiguration,
            'update_strategy': ModelUpdateConfiguration,
            'experiment_name': ExperimentConfiguration
        }
        
        for key, config_type in type_mapping.items():
            if key in data:
                return config_type
        
        return None
    
    def _save_configuration(self, config: BaseConfiguration):
        """保存配置"""
        try:
            config_file = self.storage_path / f"{config.id}.json"
            config.save_to_file(config_file)
        except Exception as e:
            logger.error(f"保存配置失败 {config.id}: {e}")
    
    def get_configuration_statistics(self) -> Dict[str, Any]:
        """获取配置统计信息"""
        stats = {
            'total_configurations': len(self._configurations),
            'configurations_by_type': defaultdict(int),
            'configurations_by_status': defaultdict(int),
            'recent_activity': []
        }
        
        now = datetime.now()
        recent_threshold = now - timedelta(days=7)
        
        for config in self._configurations.values():
            config_type = type(config).__name__
            stats['configurations_by_type'][config_type] += 1
            stats['configurations_by_status'][config.status.value] += 1
            
            if config.updated_at > recent_threshold:
                stats['recent_activity'].append({
                    'id': config.id,
                    'name': config.name,
                    'type': config_type,
                    'updated_at': config.updated_at.isoformat()
                })
        
        return dict(stats)
    
    def cleanup_old_configurations(self, days: int = 30) -> int:
        """清理旧配置"""
        cutoff_date = datetime.now() - timedelta(days=days)
        old_configs = [
            config_id for config_id, config in self._configurations.items()
            if config.updated_at < cutoff_date and config.status == ConfigurationStatus.ARCHIVED
        ]
        
        for config_id in old_configs:
            self.delete_configuration(config_id)
        
        logger.info(f"清理旧配置: {len(old_configs)} 个")
        return len(old_configs)
    
    def shutdown(self):
        """关闭管理器"""
        self.async_processor.shutdown()
        logger.info("配置管理器已关闭")


class LearningConfigurationManager:
    """K7学习配置管理器主类"""
    
    def __init__(self, storage_path: Union[str, Path] = "k7_configs"):
        self.config_manager = ConfigurationManager(storage_path)
        self.model_configs: Dict[str, MLModelConfiguration] = {}
        self.training_configs: Dict[str, TrainingConfiguration] = {}
        self.validation_configs: Dict[str, ValidationConfiguration] = {}
        self.evaluation_configs: Dict[str, EvaluationConfiguration] = {}
        self.deployment_configs: Dict[str, DeploymentConfiguration] = {}
        self.update_configs: Dict[str, ModelUpdateConfiguration] = {}
        self.experiment_configs: Dict[str, ExperimentConfiguration] = {}
        
        # 异步处理器
        self.async_processor = AsyncConfigurationProcessor()
        
        # 初始化配置类型
        self._register_configuration_types()
        
        logger.info("K7学习配置管理器初始化完成")
    
    def _register_configuration_types(self):
        """注册配置类型"""
        self.config_manager.register_configuration_type(MLModelConfiguration)
        self.config_manager.register_configuration_type(TrainingConfiguration)
        self.config_manager.register_configuration_type(ValidationConfiguration)
        self.config_manager.register_configuration_type(EvaluationConfiguration)
        self.config_manager.register_configuration_type(DeploymentConfiguration)
        self.config_manager.register_configuration_type(ModelUpdateConfiguration)
        self.config_manager.register_configuration_type(ExperimentConfiguration)
    
    # 模型配置管理
    def create_model_configuration(
        self,
        name: str,
        model_type: ModelTypeEnum,
        model_architecture: Dict[str, Any],
        input_spec: Dict[str, Any],
        output_spec: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> MLModelConfiguration:
        """创建模型配置"""
        config = self.config_manager.create_configuration(
            MLModelConfiguration,
            name=name,
            model_type=model_type,
            model_architecture=model_architecture,
            input_spec=input_spec,
            output_spec=output_spec,
            hyperparameters=hyperparameters or {},
            description=description,
            **kwargs
        )
        
        self.model_configs[config.id] = config
        logger.info(f"创建模型配置: {config.name} ({config.id})")
        return config
    
    def get_model_configuration(self, config_id: str) -> Optional[MLModelConfiguration]:
        """获取模型配置"""
        return self.model_configs.get(config_id)
    
    def list_model_configurations(self, model_type: Optional[ModelTypeEnum] = None) -> List[MLModelConfiguration]:
        """列出模型配置"""
        configs = list(self.model_configs.values())
        if model_type:
            configs = [c for c in configs if c.model_type == model_type]
        return configs
    
    def update_model_configuration(
        self,
        config_id: str,
        **updates
    ) -> Optional[MLModelConfiguration]:
        """更新模型配置"""
        config = self.config_manager.update_configuration(config_id, **updates)
        if config:
            self.model_configs[config_id] = config
        return config
    
    def delete_model_configuration(self, config_id: str) -> bool:
        """删除模型配置"""
        if config_id in self.model_configs:
            del self.model_configs[config_id]
        return self.config_manager.delete_configuration(config_id)
    
    # 训练配置管理
    def create_training_configuration(
        self,
        model_config_id: str,
        training_data: Dict[str, Any],
        training_parameters: Dict[str, Any],
        name: str,
        validation_data: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> TrainingConfiguration:
        """创建训练配置"""
        if model_config_id not in self.model_configs:
            raise ConfigurationError(f"模型配置不存在: {model_config_id}")
        
        config = self.config_manager.create_configuration(
            TrainingConfiguration,
            model_config_id=model_config_id,
            training_data=training_data,
            training_parameters=training_parameters,
            validation_data=validation_data or {},
            name=name,
            description=description,
            **kwargs
        )
        
        self.training_configs[config.id] = config
        logger.info(f"创建训练配置: {config.name} ({config.id})")
        return config
    
    def get_training_configuration(self, config_id: str) -> Optional[TrainingConfiguration]:
        """获取训练配置"""
        return self.training_configs.get(config_id)
    
    def list_training_configurations(self, model_config_id: Optional[str] = None) -> List[TrainingConfiguration]:
        """列出训练配置"""
        configs = list(self.training_configs.values())
        if model_config_id:
            configs = [c for c in configs if c.model_config_id == model_config_id]
        return configs
    
    def update_training_configuration(
        self,
        config_id: str,
        **updates
    ) -> Optional[TrainingConfiguration]:
        """更新训练配置"""
        config = self.config_manager.update_configuration(config_id, **updates)
        if config:
            self.training_configs[config_id] = config
        return config
    
    def delete_training_configuration(self, config_id: str) -> bool:
        """删除训练配置"""
        if config_id in self.training_configs:
            del self.training_configs[config_id]
        return self.config_manager.delete_configuration(config_id)
    
    # 验证配置管理
    def create_validation_configuration(
        self,
        training_config_id: str,
        validation_data: Dict[str, Any],
        validation_metrics: List[str],
        name: str,
        description: Optional[str] = None,
        **kwargs
    ) -> ValidationConfiguration:
        """创建验证配置"""
        if training_config_id not in self.training_configs:
            raise ConfigurationError(f"训练配置不存在: {training_config_id}")
        
        config = self.config_manager.create_configuration(
            ValidationConfiguration,
            training_config_id=training_config_id,
            validation_data=validation_data,
            validation_metrics=validation_metrics,
            name=name,
            description=description,
            **kwargs
        )
        
        self.validation_configs[config.id] = config
        logger.info(f"创建验证配置: {config.name} ({config.id})")
        return config
    
    def get_validation_configuration(self, config_id: str) -> Optional[ValidationConfiguration]:
        """获取验证配置"""
        return self.validation_configs.get(config_id)
    
    def list_validation_configurations(self, training_config_id: Optional[str] = None) -> List[ValidationConfiguration]:
        """列出验证配置"""
        configs = list(self.validation_configs.values())
        if training_config_id:
            configs = [c for c in configs if c.training_config_id == training_config_id]
        return configs
    
    # 评估配置管理
    def create_evaluation_configuration(
        self,
        model_config_id: str,
        evaluation_data: Dict[str, Any],
        evaluation_metrics: List[str],
        name: str,
        description: Optional[str] = None,
        **kwargs
    ) -> EvaluationConfiguration:
        """创建评估配置"""
        if model_config_id not in self.model_configs:
            raise ConfigurationError(f"模型配置不存在: {model_config_id}")
        
        config = self.config_manager.create_configuration(
            EvaluationConfiguration,
            model_config_id=model_config_id,
            evaluation_data=evaluation_data,
            evaluation_metrics=evaluation_metrics,
            name=name,
            description=description,
            **kwargs
        )
        
        self.evaluation_configs[config.id] = config
        logger.info(f"创建评估配置: {config.name} ({config.id})")
        return config
    
    def get_evaluation_configuration(self, config_id: str) -> Optional[EvaluationConfiguration]:
        """获取评估配置"""
        return self.evaluation_configs.get(config_id)
    
    def list_evaluation_configurations(self, model_config_id: Optional[str] = None) -> List[EvaluationConfiguration]:
        """列出评估配置"""
        configs = list(self.evaluation_configs.values())
        if model_config_id:
            configs = [c for c in configs if c.model_config_id == model_config_id]
        return configs
    
    # 部署配置管理
    def create_deployment_configuration(
        self,
        model_config_id: str,
        deployment_environment: Dict[str, Any],
        deployment_platform: str,
        name: str,
        description: Optional[str] = None,
        **kwargs
    ) -> DeploymentConfiguration:
        """创建部署配置"""
        if model_config_id not in self.model_configs:
            raise ConfigurationError(f"模型配置不存在: {model_config_id}")
        
        config = self.config_manager.create_configuration(
            DeploymentConfiguration,
            model_config_id=model_config_id,
            deployment_environment=deployment_environment,
            deployment_platform=deployment_platform,
            name=name,
            description=description,
            **kwargs
        )
        
        self.deployment_configs[config.id] = config
        logger.info(f"创建部署配置: {config.name} ({config.id})")
        return config
    
    def get_deployment_configuration(self, config_id: str) -> Optional[DeploymentConfiguration]:
        """获取部署配置"""
        return self.deployment_configs.get(config_id)
    
    def list_deployment_configurations(self, model_config_id: Optional[str] = None) -> List[DeploymentConfiguration]:
        """列出部署配置"""
        configs = list(self.deployment_configs.values())
        if model_config_id:
            configs = [c for c in configs if c.model_config_id == model_config_id]
        return configs
    
    def update_deployment_status(
        self,
        config_id: str,
        status: DeploymentStatus,
        message: Optional[str] = None
    ) -> bool:
        """更新部署状态"""
        try:
            config = self.deployment_configs.get(config_id)
            if not config:
                logger.error(f"部署配置不存在: {config_id}")
                return False
            
            config.deployment_status = status
            config.update_timestamp()
            
            if message:
                config.metadata['status_message'] = message
            
            self.config_manager._save_configuration(config)
            logger.info(f"更新部署状态: {config_id} -> {status.value}")
            return True
        except Exception as e:
            logger.error(f"更新部署状态失败: {e}")
            return False
    
    # 模型更新配置管理
    def create_model_update_configuration(
        self,
        model_config_id: str,
        update_strategy: Dict[str, Any],
        name: str,
        description: Optional[str] = None,
        **kwargs
    ) -> ModelUpdateConfiguration:
        """创建模型更新配置"""
        if model_config_id not in self.model_configs:
            raise ConfigurationError(f"模型配置不存在: {model_config_id}")
        
        config = self.config_manager.create_configuration(
            ModelUpdateConfiguration,
            model_config_id=model_config_id,
            update_strategy=update_strategy,
            name=name,
            description=description,
            **kwargs
        )
        
        self.update_configs[config.id] = config
        logger.info(f"创建模型更新配置: {config.name} ({config.id})")
        return config
    
    def get_model_update_configuration(self, config_id: str) -> Optional[ModelUpdateConfiguration]:
        """获取模型更新配置"""
        return self.update_configs.get(config_id)
    
    def list_model_update_configurations(self, model_config_id: Optional[str] = None) -> List[ModelUpdateConfiguration]:
        """列出模型更新配置"""
        configs = list(self.update_configs.values())
        if model_config_id:
            configs = [c for c in configs if c.model_config_id == model_config_id]
        return configs
    
    # 实验配置管理
    def create_experiment_configuration(
        self,
        experiment_name: str,
        experiment_type: str,
        name: str,
        variables: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> ExperimentConfiguration:
        """创建实验配置"""
        config = self.config_manager.create_configuration(
            ExperimentConfiguration,
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            variables=variables or {},
            metrics=metrics or [],
            name=name,
            description=description,
            **kwargs
        )
        
        self.experiment_configs[config.id] = config
        logger.info(f"创建实验配置: {config.name} ({config.id})")
        return config
    
    def get_experiment_configuration(self, config_id: str) -> Optional[ExperimentConfiguration]:
        """获取实验配置"""
        return self.experiment_configs.get(config_id)
    
    def list_experiment_configurations(self, experiment_type: Optional[str] = None) -> List[ExperimentConfiguration]:
        """列出实验配置"""
        configs = list(self.experiment_configs.values())
        if experiment_type:
            configs = [c for c in configs if c.experiment_type == experiment_type]
        return configs
    
    def update_experiment_status(
        self,
        config_id: str,
        status: ExperimentStatus,
        results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新实验状态"""
        try:
            config = self.experiment_configs.get(config_id)
            if not config:
                logger.error(f"实验配置不存在: {config_id}")
                return False
            
            config.experiment_status = status
            config.update_timestamp()
            
            if results:
                config.results.update(results)
            
            self.config_manager._save_configuration(config)
            logger.info(f"更新实验状态: {config_id} -> {status.value}")
            return True
        except Exception as e:
            logger.error(f"更新实验状态失败: {e}")
            return False
    
    # 异步处理方法
    async def process_configurations_async(
        self,
        configurations: List[BaseConfiguration],
        process_type: str = "validate"
    ) -> Dict[str, Any]:
        """异步处理配置"""
        if process_type == "validate":
            return await self.config_manager.validate_configurations_async(configurations)
        elif process_type == "save":
            return await self.config_manager.save_configurations_async(configurations)
        else:
            raise AsyncProcessingError(f"不支持的处理类型: {process_type}")
    
    async def batch_create_configurations(
        self,
        configs_data: List[Dict[str, Any]],
        config_type: Type[BaseConfiguration]
    ) -> List[BaseConfiguration]:
        """批量创建配置"""
        configurations = []
        
        for config_data in configs_data:
            try:
                config = config_type(**config_data)
                config.validate_config()
                configurations.append(config)
            except Exception as e:
                logger.error(f"创建配置失败: {e}")
                continue
        
        # 异步保存
        if configurations:
            await self.config_manager.save_configurations_async(configurations)
        
        return configurations
    
    async def validate_all_configurations_async(self) -> Dict[str, bool]:
        """异步验证所有配置"""
        all_configs = (
            list(self.model_configs.values()) +
            list(self.training_configs.values()) +
            list(self.validation_configs.values()) +
            list(self.evaluation_configs.values()) +
            list(self.deployment_configs.values()) +
            list(self.update_configs.values()) +
            list(self.experiment_configs.values())
        )
        
        return await self.config_manager.validate_configurations_async(all_configs)
    
    # 搜索和查询方法
    def search_configurations(
        self,
        query: str,
        config_types: Optional[List[str]] = None
    ) -> Dict[str, List[BaseConfiguration]]:
        """搜索配置"""
        results = {}
        
        if not config_types or 'model' in config_types:
            results['model'] = self.config_manager.search_configurations(query, ['name', 'description'])
        
        if not config_types or 'training' in config_types:
            results['training'] = self.config_manager.search_configurations(query, ['name', 'description'])
        
        if not config_types or 'validation' in config_types:
            results['validation'] = self.config_manager.search_configurations(query, ['name', 'description'])
        
        if not config_types or 'evaluation' in config_types:
            results['evaluation'] = self.config_manager.search_configurations(query, ['name', 'description'])
        
        if not config_types or 'deployment' in config_types:
            results['deployment'] = self.config_manager.search_configurations(query, ['name', 'description'])
        
        if not config_types or 'update' in config_types:
            results['update'] = self.config_manager.search_configurations(query, ['name', 'description'])
        
        if not config_types or 'experiment' in config_types:
            results['experiment'] = self.config_manager.search_configurations(query, ['name', 'description'])
        
        return results
    
    def get_configuration_dependencies(self, config_id: str) -> Dict[str, List[str]]:
        """获取配置依赖关系"""
        dependencies = {}
        
        # 查找依赖此配置的其他配置
        for config in self.training_configs.values():
            if config.model_config_id == config_id:
                dependencies.setdefault('training', []).append(config.id)
        
        for config in self.validation_configs.values():
            if config.training_config_id == config_id:
                dependencies.setdefault('validation', []).append(config.id)
        
        for config in self.evaluation_configs.values():
            if config.model_config_id == config_id:
                dependencies.setdefault('evaluation', []).append(config.id)
        
        for config in self.deployment_configs.values():
            if config.model_config_id == config_id:
                dependencies.setdefault('deployment', []).append(config.id)
        
        for config in self.update_configs.values():
            if config.model_config_id == config_id:
                dependencies.setdefault('update', []).append(config.id)
        
        return dependencies
    
    # 统计和报告方法
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            'model_configurations': {
                'total': len(self.model_configs),
                'by_type': defaultdict(int),
                'by_status': defaultdict(int)
            },
            'training_configurations': {
                'total': len(self.training_configs),
                'by_status': defaultdict(int)
            },
            'validation_configurations': {
                'total': len(self.validation_configs),
                'by_status': defaultdict(int)
            },
            'evaluation_configurations': {
                'total': len(self.evaluation_configs),
                'by_status': defaultdict(int)
            },
            'deployment_configurations': {
                'total': len(self.deployment_configs),
                'by_status': defaultdict(int),
                'by_platform': defaultdict(int)
            },
            'update_configurations': {
                'total': len(self.update_configs),
                'by_status': defaultdict(int),
                'by_frequency': defaultdict(int)
            },
            'experiment_configurations': {
                'total': len(self.experiment_configs),
                'by_status': defaultdict(int),
                'by_type': defaultdict(int)
            }
        }
        
        # 统计模型配置
        for config in self.model_configs.values():
            model_type = config.model_type.value if hasattr(config.model_type, 'value') else str(config.model_type)
            stats['model_configurations']['by_type'][model_type] += 1
            status = config.status.value if hasattr(config.status, 'value') else str(config.status)
            stats['model_configurations']['by_status'][status] += 1
        
        # 统计训练配置
        for config in self.training_configs.values():
            stats['training_configurations']['by_status'][config.status.value] += 1
        
        # 统计验证配置
        for config in self.validation_configs.values():
            stats['validation_configurations']['by_status'][config.status.value] += 1
        
        # 统计评估配置
        for config in self.evaluation_configs.values():
            stats['evaluation_configurations']['by_status'][config.status.value] += 1
        
        # 统计部署配置
        for config in self.deployment_configs.values():
            stats['deployment_configurations']['by_status'][config.deployment_status.value] += 1
            stats['deployment_configurations']['by_platform'][config.deployment_platform] += 1
        
        # 统计更新配置
        for config in self.update_configs.values():
            stats['update_configurations']['by_status'][config.status.value] += 1
            stats['update_configurations']['by_frequency'][config.update_frequency] += 1
        
        # 统计实验配置
        for config in self.experiment_configs.values():
            stats['experiment_configurations']['by_status'][config.experiment_status.value] += 1
            stats['experiment_configurations']['by_type'][config.experiment_type] += 1
        
        # 转换defaultdict为dict
        for category in stats.values():
            for key, value in category.items():
                if isinstance(value, defaultdict):
                    category[key] = dict(value)
        
        return stats
    
    def generate_configuration_report(self, config_id: str) -> Dict[str, Any]:
        """生成配置报告"""
        config = None
        config_type = None
        
        # 查找配置
        for config_list, type_name in [
            (self.model_configs, 'model'),
            (self.training_configs, 'training'),
            (self.validation_configs, 'validation'),
            (self.evaluation_configs, 'evaluation'),
            (self.deployment_configs, 'deployment'),
            (self.update_configs, 'update'),
            (self.experiment_configs, 'experiment')
        ]:
            if config_id in config_list:
                config = config_list[config_id]
                config_type = type_name
                break
        
        if not config:
            raise ConfigurationError(f"配置不存在: {config_id}")
        
        report = {
            'configuration_id': config.id,
            'configuration_type': config_type,
            'name': config.name,
            'description': config.description,
            'version': config.version,
            'status': config.status.value,
            'created_at': config.created_at.isoformat(),
            'updated_at': config.updated_at.isoformat(),
            'metadata': config.metadata,
            'dependencies': self.get_configuration_dependencies(config_id),
            'configuration_data': config.dict()
        }
        
        return report
    
    # 配置生命周期管理
    def activate_configuration(self, config_id: str) -> bool:
        """激活配置"""
        try:
            config = self.config_manager.get_configuration(config_id)
            if not config:
                return False
            
            config.status = ConfigurationStatus.ACTIVE
            config.update_timestamp()
            self.config_manager._save_configuration(config)
            
            logger.info(f"激活配置: {config_id}")
            return True
        except Exception as e:
            logger.error(f"激活配置失败: {e}")
            return False
    
    def archive_configuration(self, config_id: str) -> bool:
        """归档配置"""
        try:
            config = self.config_manager.get_configuration(config_id)
            if not config:
                return False
            
            config.status = ConfigurationStatus.ARCHIVED
            config.update_timestamp()
            self.config_manager._save_configuration(config)
            
            logger.info(f"归档配置: {config_id}")
            return True
        except Exception as e:
            logger.error(f"归档配置失败: {e}")
            return False
    
    def clone_configuration(
        self,
        config_id: str,
        new_name: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseConfiguration]:
        """克隆配置"""
        try:
            original_config = self.config_manager.get_configuration(config_id)
            if not original_config:
                return None
            
            # 创建新配置
            config_data = original_config.dict()
            config_data['id'] = str(uuid.uuid4())
            config_data['name'] = new_name
            config_data['created_at'] = datetime.now()
            config_data['updated_at'] = datetime.now()
            config_data['status'] = ConfigurationStatus.DRAFT
            
            # 应用修改
            if modifications:
                config_data.update(modifications)
            
            # 创建新配置实例
            config_type = type(original_config)
            new_config = config_type(**config_data)
            
            # 保存配置
            self.config_manager._configurations[new_config.id] = new_config
            self.config_manager._save_configuration(new_config)
            
            # 更新相应的字典
            if isinstance(new_config, MLModelConfiguration):
                self.model_configs[new_config.id] = new_config
            elif isinstance(new_config, TrainingConfiguration):
                self.training_configs[new_config.id] = new_config
            elif isinstance(new_config, ValidationConfiguration):
                self.validation_configs[new_config.id] = new_config
            elif isinstance(new_config, EvaluationConfiguration):
                self.evaluation_configs[new_config.id] = new_config
            elif isinstance(new_config, DeploymentConfiguration):
                self.deployment_configs[new_config.id] = new_config
            elif isinstance(new_config, ModelUpdateConfiguration):
                self.update_configs[new_config.id] = new_config
            elif isinstance(new_config, ExperimentConfiguration):
                self.experiment_configs[new_config.id] = new_config
            
            logger.info(f"克隆配置: {config_id} -> {new_config.id}")
            return new_config
        except Exception as e:
            logger.error(f"克隆配置失败: {e}")
            return None
    
    # 导入导出方法
    def export_system_configurations(
        self,
        output_file: Union[str, Path],
        format: str = "json",
        include_metadata: bool = True
    ) -> bool:
        """导出系统配置"""
        try:
            all_configs = {
                'model_configurations': [config.dict() for config in self.model_configs.values()],
                'training_configurations': [config.dict() for config in self.training_configs.values()],
                'validation_configurations': [config.dict() for config in self.validation_configs.values()],
                'evaluation_configurations': [config.dict() for config in self.evaluation_configs.values()],
                'deployment_configurations': [config.dict() for config in self.deployment_configs.values()],
                'update_configurations': [config.dict() for config in self.update_configs.values()],
                'experiment_configurations': [config.dict() for config in self.experiment_configs.values()],
                'export_info': {
                    'export_time': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'total_configurations': sum(len(configs) for configs in all_configs.values() if isinstance(configs, list))
                }
            }
            
            if not include_metadata:
                # 移除元数据
                for config_type in all_configs:
                    if isinstance(all_configs[config_type], list):
                        for config in all_configs[config_type]:
                            if 'metadata' in config:
                                config['metadata'] = {}
            
            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_configs, f, indent=2, ensure_ascii=False, default=str)
            elif format.lower() == "yaml":
                if yaml is None:
                    raise ValueError("yaml未安装，无法导出YAML格式")
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(all_configs, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"导出系统配置成功: {output_file}")
            return True
        except Exception as e:
            logger.error(f"导出系统配置失败: {e}")
            return False
    
    def import_system_configurations(
        self,
        input_file: Union[str, Path],
        merge_strategy: str = "skip_existing"
    ) -> Dict[str, int]:
        """导入系统配置"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                if str(input_file).endswith('.yaml') or str(input_file).endswith('.yml'):
                    if yaml is None:
                        raise ValueError("yaml未安装，无法导入YAML文件")
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            import_stats = {
                'model_configurations': 0,
                'training_configurations': 0,
                'validation_configurations': 0,
                'evaluation_configurations': 0,
                'deployment_configurations': 0,
                'update_configurations': 0,
                'experiment_configurations': 0,
                'skipped': 0,
                'errors': 0
            }
            
            # 导入模型配置
            for config_data in data.get('model_configurations', []):
                try:
                    if config_data['id'] in self.model_configs and merge_strategy == "skip_existing":
                        import_stats['skipped'] += 1
                        continue
                    
                    config = MLModelConfiguration(**config_data)
                    self.model_configs[config.id] = config
                    self.config_manager._configurations[config.id] = config
                    import_stats['model_configurations'] += 1
                except Exception as e:
                    logger.error(f"导入模型配置失败: {e}")
                    import_stats['errors'] += 1
            
            # 导入训练配置
            for config_data in data.get('training_configurations', []):
                try:
                    if config_data['id'] in self.training_configs and merge_strategy == "skip_existing":
                        import_stats['skipped'] += 1
                        continue
                    
                    config = TrainingConfiguration(**config_data)
                    self.training_configs[config.id] = config
                    self.config_manager._configurations[config.id] = config
                    import_stats['training_configurations'] += 1
                except Exception as e:
                    logger.error(f"导入训练配置失败: {e}")
                    import_stats['errors'] += 1
            
            # 导入验证配置
            for config_data in data.get('validation_configurations', []):
                try:
                    if config_data['id'] in self.validation_configs and merge_strategy == "skip_existing":
                        import_stats['skipped'] += 1
                        continue
                    
                    config = ValidationConfiguration(**config_data)
                    self.validation_configs[config.id] = config
                    self.config_manager._configurations[config.id] = config
                    import_stats['validation_configurations'] += 1
                except Exception as e:
                    logger.error(f"导入验证配置失败: {e}")
                    import_stats['errors'] += 1
            
            # 导入评估配置
            for config_data in data.get('evaluation_configurations', []):
                try:
                    if config_data['id'] in self.evaluation_configs and merge_strategy == "skip_existing":
                        import_stats['skipped'] += 1
                        continue
                    
                    config = EvaluationConfiguration(**config_data)
                    self.evaluation_configs[config.id] = config
                    self.config_manager._configurations[config.id] = config
                    import_stats['evaluation_configurations'] += 1
                except Exception as e:
                    logger.error(f"导入评估配置失败: {e}")
                    import_stats['errors'] += 1
            
            # 导入部署配置
            for config_data in data.get('deployment_configurations', []):
                try:
                    if config_data['id'] in self.deployment_configs and merge_strategy == "skip_existing":
                        import_stats['skipped'] += 1
                        continue
                    
                    config = DeploymentConfiguration(**config_data)
                    self.deployment_configs[config.id] = config
                    self.config_manager._configurations[config.id] = config
                    import_stats['deployment_configurations'] += 1
                except Exception as e:
                    logger.error(f"导入部署配置失败: {e}")
                    import_stats['errors'] += 1
            
            # 导入更新配置
            for config_data in data.get('update_configurations', []):
                try:
                    if config_data['id'] in self.update_configs and merge_strategy == "skip_existing":
                        import_stats['skipped'] += 1
                        continue
                    
                    config = ModelUpdateConfiguration(**config_data)
                    self.update_configs[config.id] = config
                    self.config_manager._configurations[config.id] = config
                    import_stats['update_configurations'] += 1
                except Exception as e:
                    logger.error(f"导入更新配置失败: {e}")
                    import_stats['errors'] += 1
            
            # 导入实验配置
            for config_data in data.get('experiment_configurations', []):
                try:
                    if config_data['id'] in self.experiment_configs and merge_strategy == "skip_existing":
                        import_stats['skipped'] += 1
                        continue
                    
                    config = ExperimentConfiguration(**config_data)
                    self.experiment_configs[config.id] = config
                    self.config_manager._configurations[config.id] = config
                    import_stats['experiment_configurations'] += 1
                except Exception as e:
                    logger.error(f"导入实验配置失败: {e}")
                    import_stats['errors'] += 1
            
            logger.info(f"导入系统配置完成: {import_stats}")
            return import_stats
        except Exception as e:
            logger.error(f"导入系统配置失败: {e}")
            return {'errors': 1}
    
    # 系统维护方法
    def cleanup_system(self, days: int = 30) -> Dict[str, int]:
        """清理系统"""
        cleanup_stats = {
            'archived_configurations': 0,
            'orphaned_configurations': 0,
            'invalid_configurations': 0
        }
        
        # 清理归档配置
        cleanup_stats['archived_configurations'] = self.config_manager.cleanup_old_configurations(days)
        
        # 清理孤立配置
        orphaned_count = 0
        for config_id in list(self.config_manager._configurations.keys()):
            config = self.config_manager._configurations[config_id]
            if not self._is_configuration_referenced(config_id):
                orphaned_count += 1
                self.config_manager.delete_configuration(config_id)
        
        cleanup_stats['orphaned_configurations'] = orphaned_count
        
        # 清理无效配置
        invalid_count = 0
        for config_id in list(self.config_manager._configurations.keys()):
            config = self.config_manager._configurations[config_id]
            try:
                if not config.validate_config():
                    invalid_count += 1
                    self.config_manager.delete_configuration(config_id)
            except Exception:
                invalid_count += 1
                self.config_manager.delete_configuration(config_id)
        
        cleanup_stats['invalid_configurations'] = invalid_count
        
        logger.info(f"系统清理完成: {cleanup_stats}")
        return cleanup_stats
    
    def _is_configuration_referenced(self, config_id: str) -> bool:
        """检查配置是否被引用"""
        # 检查训练配置是否引用此模型配置
        for config in self.training_configs.values():
            if config.model_config_id == config_id:
                return True
        
        # 检查验证配置是否引用此训练配置
        for config in self.validation_configs.values():
            if config.training_config_id == config_id:
                return True
        
        # 检查评估配置是否引用此模型配置
        for config in self.evaluation_configs.values():
            if config.model_config_id == config_id:
                return True
        
        # 检查部署配置是否引用此模型配置
        for config in self.deployment_configs.values():
            if config.model_config_id == config_id:
                return True
        
        # 检查更新配置是否引用此模型配置
        for config in self.update_configs.values():
            if config.model_config_id == config_id:
                return True
        
        return False
    
    def validate_system_integrity(self) -> Dict[str, Any]:
        """验证系统完整性"""
        integrity_report = {
            'is_valid': True,
            'issues': [],
            'statistics': self.get_system_statistics()
        }
        
        # 检查配置引用完整性
        for config_id, config in self.training_configs.items():
            if config.model_config_id not in self.model_configs:
                integrity_report['issues'].append(f"训练配置 {config_id} 引用了不存在的模型配置 {config.model_config_id}")
                integrity_report['is_valid'] = False
        
        for config_id, config in self.validation_configs.items():
            if config.training_config_id not in self.training_configs:
                integrity_report['issues'].append(f"验证配置 {config_id} 引用了不存在的训练配置 {config.training_config_id}")
                integrity_report['is_valid'] = False
        
        for config_id, config in self.evaluation_configs.items():
            if config.model_config_id not in self.model_configs:
                integrity_report['issues'].append(f"评估配置 {config_id} 引用了不存在的模型配置 {config.model_config_id}")
                integrity_report['is_valid'] = False
        
        for config_id, config in self.deployment_configs.items():
            if config.model_config_id not in self.model_configs:
                integrity_report['issues'].append(f"部署配置 {config_id} 引用了不存在的模型配置 {config.model_config_id}")
                integrity_report['is_valid'] = False
        
        for config_id, config in self.update_configs.items():
            if config.model_config_id not in self.model_configs:
                integrity_report['issues'].append(f"更新配置 {config_id} 引用了不存在的模型配置 {config.model_config_id}")
                integrity_report['is_valid'] = False
        
        # 检查配置状态一致性
        for config_id, config in self.model_configs.items():
            if config.status == ConfigurationStatus.ACTIVE:
                # 检查是否有相应的训练、评估、部署配置
                has_training = any(tc.model_config_id == config_id for tc in self.training_configs.values())
                has_evaluation = any(ec.model_config_id == config_id for ec in self.evaluation_configs.values())
                
                if not has_training:
                    integrity_report['issues'].append(f"活跃模型配置 {config_id} 没有训练配置")
                
                if not has_evaluation:
                    integrity_report['issues'].append(f"活跃模型配置 {config_id} 没有评估配置")
        
        logger.info(f"系统完整性验证完成: {'通过' if integrity_report['is_valid'] else '失败'}")
        return integrity_report
    
    def shutdown(self):
        """关闭管理器"""
        self.async_processor.shutdown()
        self.config_manager.shutdown()
        logger.info("K7学习配置管理器已关闭")


# 使用示例和测试代码
def create_sample_configurations():
    """创建示例配置"""
    # 创建管理器
    manager = LearningConfigurationManager("sample_configs")
    
    # 创建模型配置
    model_config = manager.create_model_configuration(
        name="示例分类模型",
        model_type=ModelTypeEnum.CLASSIFICATION,
        model_architecture={
            "type": "neural_network",
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 10, "activation": "softmax"}
            ]
        },
        input_spec={
            "shape": [784],
            "dtype": "float32",
            "normalization": "min_max"
        },
        output_spec={
            "shape": [10],
            "dtype": "float32",
            "activation": "softmax"
        },
        hyperparameters={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "dropout_rate": 0.2
        },
        description="用于手写数字识别的神经网络模型"
    )
    
    # 创建训练配置
    training_config = manager.create_training_configuration(
        model_config_id=model_config.id,
        name="标准训练配置",
        training_data={
            "source": "mnist_dataset",
            "format": "numpy",
            "train_split": 0.8,
            "validation_split": 0.2,
            "preprocessing": {
                "normalize": True,
                "reshape": True
            }
        },
        training_parameters={
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "validation_freq": 1
        },
        optimizer={
            "type": "adam",
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999
        },
        early_stopping={
            "monitor": "val_loss",
            "patience": 10,
            "restore_best_weights": True
        },
        description="标准的模型训练配置"
    )
    
    # 创建验证配置
    validation_config = manager.create_validation_configuration(
        training_config_id=training_config.id,
        name="标准验证配置",
        validation_data={
            "source": "mnist_validation",
            "format": "numpy",
            "preprocessing": {
                "normalize": True,
                "reshape": True
            }
        },
        validation_metrics=["accuracy", "precision", "recall", "f1_score"],
        validation_strategy={
            "type": "k_fold",
            "k": 5,
            "random_state": 42
        },
        description="标准的模型验证配置"
    )
    
    # 创建评估配置
    evaluation_config = manager.create_evaluation_configuration(
        model_config_id=model_config.id,
        name="标准评估配置",
        evaluation_data={
            "source": "mnist_test",
            "format": "numpy",
            "preprocessing": {
                "normalize": True,
                "reshape": True
            }
        },
        evaluation_metrics=["accuracy", "precision", "recall", "f1_score", "auc_roc"],
        evaluation_methods={
            "confusion_matrix": True,
            "classification_report": True,
            "roc_curve": True
        },
        evaluation_report={
            "format": "html",
            "include_plots": True,
            "include_statistics": True
        },
        description="标准的模型评估配置"
    )
    
    # 创建部署配置
    deployment_config = manager.create_deployment_configuration(
        model_config_id=model_config.id,
        name="Kubernetes部署配置",
        deployment_environment={
            "namespace": "ml-models",
            "service_type": "ClusterIP",
            "replicas": 3,
            "resources": {
                "cpu": "500m",
                "memory": "1Gi",
                "storage": "5Gi"
            }
        },
        deployment_platform="kubernetes",
        scaling_config={
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu_utilization": 70
        },
        monitoring_config={
            "metrics_endpoint": "/metrics",
            "health_check_endpoint": "/health",
            "log_level": "INFO"
        },
        description="Kubernetes平台部署配置"
    )
    
    # 创建更新配置
    update_config = manager.create_model_update_configuration(
        model_config_id=model_config.id,
        name="自动更新配置",
        update_strategy={
            "type": "performance_based",
            "trigger_metrics": ["accuracy", "f1_score"],
            "improvement_threshold": 0.02,
            "rollback_on_degradation": True
        },
        version_management={
            "strategy": "semantic_versioning",
            "keep_versions": 5,
            "backup_before_update": True
        },
        rollback_mechanism={
            "automatic": True,
            "max_rollback_time": 300,
            "health_check_timeout": 60
        },
        update_frequency="on_performance_degradation",
        auto_update=True,
        description="基于性能指标的自动模型更新配置"
    )
    
    # 创建实验配置
    experiment_config = manager.create_experiment_configuration(
        experiment_name="学习率对模型性能的影响",
        experiment_type="hyperparameter_optimization",
        hypothesis="较低的学习率将提高模型的泛化能力",
        variables={
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64],
            "optimizer": ["adam", "sgd", "rmsprop"]
        },
        controls={
            "epochs": 50,
            "validation_split": 0.2,
            "random_seed": 42
        },
        metrics=["accuracy", "val_accuracy", "loss", "val_loss"],
        experimental_design={
            "type": "factorial",
            "replicates": 3,
            "randomization": True
        },
        sample_size=10000,
        statistical_power=0.8,
        description="研究不同学习率设置对模型性能的影响"
    )
    
    return manager, {
        'model': model_config,
        'training': training_config,
        'validation': validation_config,
        'evaluation': evaluation_config,
        'deployment': deployment_config,
        'update': update_config,
        'experiment': experiment_config
    }


async def demonstrate_async_processing():
    """演示异步处理功能"""
    print("=== 异步处理演示 ===")
    
    # 创建示例配置
    manager, configs = create_sample_configurations()
    
    # 异步验证所有配置
    print("异步验证所有配置...")
    validation_results = await manager.validate_all_configurations_async()
    print(f"验证结果: {validation_results}")
    
    # 异步批量处理配置
    all_configs = list(configs.values())
    print("异步处理配置...")
    processing_results = await manager.process_configurations_async(all_configs, "validate")
    print(f"处理结果: {processing_results}")
    
    # 异步保存配置
    print("异步保存配置...")
    save_results = await manager.save_configurations_async(all_configs, "async_configs")
    print(f"保存结果: {save_results}")
    
    # 关闭管理器
    manager.shutdown()


def demonstrate_configuration_management():
    """演示配置管理功能"""
    print("=== 配置管理演示 ===")
    
    # 创建示例配置
    manager, configs = create_sample_configurations()
    
    # 获取系统统计信息
    stats = manager.get_system_statistics()
    print(f"系统统计信息: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 搜索配置
    search_results = manager.search_configurations("分类")
    print(f"搜索结果: {json.dumps({k: len(v) for k, v in search_results.items()}, indent=2)}")
    
    # 生成配置报告
    model_config = configs['model']
    report = manager.generate_configuration_report(model_config.id)
    print(f"配置报告: {json.dumps({k: v for k, v in report.items() if k != 'configuration_data'}, indent=2)}")
    
    # 验证系统完整性
    integrity = manager.validate_system_integrity()
    print(f"系统完整性: {'通过' if integrity['is_valid'] else '失败'}")
    if integrity['issues']:
        print(f"问题: {integrity['issues']}")
    
    # 克隆配置
    cloned_config = manager.clone_configuration(
        model_config.id,
        "克隆的分类模型",
        {"description": "这是克隆的模型配置"}
    )
    if cloned_config:
        print(f"克隆配置成功: {cloned_config.id}")
    
    # 导出配置
    export_success = manager.export_system_configurations("exported_configs.json")
    print(f"导出配置: {'成功' if export_success else '失败'}")
    
    # 关闭管理器
    manager.shutdown()


def demonstrate_lifecycle_management():
    """演示生命周期管理"""
    print("=== 生命周期管理演示 ===")
    
    # 创建示例配置
    manager, configs = create_sample_configurations()
    
    # 激活配置
    model_config = configs['model']
    activation_success = manager.activate_configuration(model_config.id)
    print(f"激活配置: {'成功' if activation_success else '失败'}")
    
    # 更新实验状态
    experiment_config = configs['experiment']
    status_update = manager.update_experiment_status(
        experiment_config.id,
        ExperimentStatus.RUNNING,
        {"start_time": datetime.now().isoformat()}
    )
    print(f"更新实验状态: {'成功' if status_update else '失败'}")
    
    # 更新部署状态
    deployment_config = configs['deployment']
    deployment_status = manager.update_deployment_status(
        deployment_config.id,
        DeploymentStatus.DEPLOYING,
        "开始部署过程"
    )
    print(f"更新部署状态: {'成功' if deployment_status else '失败'}")
    
    # 归档配置
    archive_success = manager.archive_configuration(experiment_config.id)
    print(f"归档配置: {'成功' if archive_success else '失败'}")
    
    # 清理系统
    cleanup_stats = manager.cleanup_system()
    print(f"清理结果: {cleanup_stats}")
    
    # 关闭管理器
    manager.shutdown()


if __name__ == "__main__":
    print("K7学习配置管理器演示")
    print("=" * 50)
    
    try:
        # 演示配置管理
        demonstrate_configuration_management()
        
        print("\n" + "=" * 50)
        
        # 演示生命周期管理
        demonstrate_lifecycle_management()
        
        print("\n" + "=" * 50)
        
        # 演示异步处理
        asyncio.run(demonstrate_async_processing())
        
    except Exception as e:
        print(f"演示过程中出错: {e}")
        logger.error(f"演示失败: {e}")

# 扩展配置类型
class MonitoringConfiguration(BaseConfiguration):
    """监控配置"""
    
    model_config_id: str = Field(..., description="关联的模型配置ID")
    monitoring_metrics: List[str] = Field(..., description="监控指标")
    monitoring_frequency: str = Field(default="real_time", description="监控频率")
    alert_thresholds: Dict[str, float] = Field(default_factory=dict, description="告警阈值")
    notification_channels: List[str] = Field(default_factory=list, description="通知渠道")
    data_collection_config: Dict[str, Any] = Field(default_factory=dict, description="数据收集配置")
    visualization_config: Dict[str, Any] = Field(default_factory=dict, description="可视化配置")
    
    def validate_configurations(self):
        """验证监控配置"""
        if not self.monitoring_metrics:
            raise ValidationError("监控指标不能为空")
        
        valid_frequencies = ['real_time', 'hourly', 'daily', 'weekly']
        if self.monitoring_frequency not in valid_frequencies:
            raise ValidationError(f"不支持的监控频率: {self.monitoring_frequency}")


class DataPipelineConfiguration(BaseConfiguration):
    """数据管道配置"""
    
    pipeline_name: str = Field(..., description="管道名称")
    data_sources: List[Dict[str, Any]] = Field(..., description="数据源")
    data_transformations: List[Dict[str, Any]] = Field(default_factory=list, description="数据转换")
    data_validation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="数据验证规则")
    data_storage_config: Dict[str, Any] = Field(..., description="数据存储配置")
    scheduling_config: Dict[str, Any] = Field(default_factory=dict, description="调度配置")
    quality_checks: List[str] = Field(default_factory=list, description="质量检查")
    
    def validate_configurations(self):
        """验证数据管道配置"""
        if not self.data_sources:
            raise ValidationError("数据源不能为空")
        
        if not self.data_storage_config:
            raise ValidationError("数据存储配置不能为空")


class ABTestConfiguration(BaseConfiguration):
    """A/B测试配置"""
    
    test_name: str = Field(..., description="测试名称")
    test_hypothesis: str = Field(..., description="测试假设")
    control_group_config: Dict[str, Any] = Field(..., description="对照组配置")
    treatment_group_config: Dict[str, Any] = Field(..., description="实验组配置")
    test_metrics: List[str] = Field(..., description="测试指标")
    sample_size_calculation: Dict[str, Any] = Field(default_factory=dict, description="样本大小计算")
    statistical_config: Dict[str, Any] = Field(default_factory=dict, description="统计配置")
    test_duration: int = Field(..., description="测试持续时间(天)")
    traffic_allocation: Dict[str, float] = Field(default_factory=dict, description="流量分配")
    
    def validate_configurations(self):
        """验证A/B测试配置"""
        if not self.test_name:
            raise ValidationError("测试名称不能为空")
        
        if not self.test_hypothesis:
            raise ValidationError("测试假设不能为空")
        
        if not self.test_metrics:
            raise ValidationError("测试指标不能为空")
        
        # 验证流量分配
        total_allocation = sum(self.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.001:
            raise ValidationError("流量分配总和必须为1.0")


class ModelRegistryConfiguration(BaseConfiguration):
    """模型注册表配置"""
    
    registry_name: str = Field(..., description="注册表名称")
    model_storage_backend: str = Field(..., description="模型存储后端")
    version_control_config: Dict[str, Any] = Field(default_factory=dict, description="版本控制配置")
    metadata_storage_config: Dict[str, Any] = Field(default_factory=dict, description="元数据存储配置")
    access_control_config: Dict[str, Any] = Field(default_factory=dict, description="访问控制配置")
    model_lineage_tracking: bool = Field(default=True, description="模型血缘跟踪")
    model_validation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="模型验证规则")
    
    def validate_configurations(self):
        """验证模型注册表配置"""
        if not self.registry_name:
            raise ValidationError("注册表名称不能为空")
        
        valid_backends = ['local', 's3', 'gcs', 'azure_blob', 'hdfs', 'database']
        if self.model_storage_backend not in valid_backends:
            raise ValidationError(f"不支持的存储后端: {self.model_storage_backend}")


class FeatureStoreConfiguration(BaseConfiguration):
    """特征存储配置"""
    
    feature_store_name: str = Field(..., description="特征存储名称")
    feature_definitions: List[Dict[str, Any]] = Field(..., description="特征定义")
    data_sources: List[Dict[str, Any]] = Field(..., description="数据源")
    transformation_functions: List[Dict[str, Any]] = Field(default_factory=list, description="转换函数")
    storage_config: Dict[str, Any] = Field(..., description="存储配置")
    serving_config: Dict[str, Any] = Field(default_factory=dict, description="服务配置")
    monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="监控配置")
    
    def validate_configurations(self):
        """验证特征存储配置"""
        if not self.feature_store_name:
            raise ValidationError("特征存储名称不能为空")
        
        if not self.feature_definitions:
            raise ValidationError("特征定义不能为空")
        
        if not self.data_sources:
            raise ValidationError("数据源不能为空")


# 高级配置管理器扩展
class AdvancedConfigurationManager(ConfigurationManager):
    """高级配置管理器"""
    
    def __init__(self, storage_path: Union[str, Path] = "advanced_configs"):
        super().__init__(storage_path)
        self.configuration_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.configuration_tags: Dict[str, Set[str]] = defaultdict(set)
        self.configuration_relationships: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
    def add_configuration_tag(self, config_id: str, tag: str):
        """添加配置标签"""
        if config_id in self._configurations:
            self.configuration_tags[config_id].add(tag)
            logger.info(f"为配置 {config_id} 添加标签: {tag}")
    
    def remove_configuration_tag(self, config_id: str, tag: str):
        """移除配置标签"""
        if config_id in self._configurations:
            self.configuration_tags[config_id].discard(tag)
            logger.info(f"从配置 {config_id} 移除标签: {tag}")
    
    def get_configurations_by_tag(self, tag: str) -> List[BaseConfiguration]:
        """根据标签获取配置"""
        config_ids = {config_id for config_id, tags in self.configuration_tags.items() if tag in tags}
        return [self._configurations[config_id] for config_id in config_ids if config_id in self._configurations]
    
    def add_configuration_relationship(
        self,
        source_config_id: str,
        target_config_id: str,
        relationship_type: str
    ):
        """添加配置关系"""
        if source_config_id in self._configurations and target_config_id in self._configurations:
            self.configuration_relationships[source_config_id][relationship_type].append(target_config_id)
            logger.info(f"添加配置关系: {source_config_id} -> {target_config_id} ({relationship_type})")
    
    def get_configuration_dependencies(self, config_id: str) -> Dict[str, List[str]]:
        """获取配置依赖关系"""
        return dict(self.configuration_relationships.get(config_id, {}))
    
    def get_dependent_configurations(self, config_id: str) -> Dict[str, List[str]]:
        """获取依赖此配置的其他配置"""
        dependents = defaultdict(list)
        for source_id, relationships in self.configuration_relationships.items():
            for rel_type, target_ids in relationships.items():
                if config_id in target_ids:
                    dependents[rel_type].append(source_id)
        return dict(dependents)
    
    def save_configuration_version(self, config_id: str, version_note: str = ""):
        """保存配置版本"""
        if config_id in self._configurations:
            config = self._configurations[config_id]
            version_data = {
                'version': len(self.configuration_history[config_id]) + 1,
                'timestamp': datetime.now().isoformat(),
                'configuration': config.dict(),
                'note': version_note
            }
            self.configuration_history[config_id].append(version_data)
            logger.info(f"保存配置版本: {config_id} v{version_data['version']}")
    
    def get_configuration_history(self, config_id: str) -> List[Dict[str, Any]]:
        """获取配置历史"""
        return self.configuration_history.get(config_id, [])
    
    def restore_configuration_version(self, config_id: str, version: int) -> bool:
        """恢复配置版本"""
        if config_id in self.configuration_history:
            history = self.configuration_history[config_id]
            if 1 <= version <= len(history):
                version_data = history[version - 1]
                config_data = version_data['configuration']
                config_type = self._determine_config_type(config_data)
                if config_type:
                    config = config_type(**config_data)
                    self._configurations[config_id] = config
                    self._save_configuration(config)
                    logger.info(f"恢复配置版本: {config_id} v{version}")
                    return True
        return False
    
    def compare_configurations(self, config_id1: str, config_id2: str) -> Dict[str, Any]:
        """比较两个配置"""
        config1 = self._configurations.get(config_id1)
        config2 = self._configurations.get(config_id2)
        
        if not config1 or not config2:
            raise ConfigurationError("配置不存在")
        
        comparison = {
            'config1_id': config_id1,
            'config2_id': config_id2,
            'config1_name': config1.name,
            'config2_name': config2.name,
            'differences': [],
            'similarities': []
        }
        
        # 比较基本属性
        for field in ['name', 'description', 'version', 'status']:
            value1 = getattr(config1, field)
            value2 = getattr(config2, field)
            if value1 != value2:
                comparison['differences'].append({
                    'field': field,
                    'config1_value': value1,
                    'config2_value': value2
                })
            else:
                comparison['similarities'].append(field)
        
        # 比较配置数据
        data1 = config1.dict()
        data2 = config2.dict()
        
        for key in data1:
            if key in data2:
                if data1[key] != data2[key]:
                    comparison['differences'].append({
                        'field': key,
                        'config1_value': data1[key],
                        'config2_value': data2[key]
                    })
        
        return comparison


# 配置模板系统
class ConfigurationTemplate:
    """配置模板"""
    
    def __init__(self, template_name: str, template_data: Dict[str, Any]):
        self.template_name = template_name
        self.template_data = template_data
        self.created_at = datetime.now()
        self.variables = self._extract_variables(template_data)
    
    def _extract_variables(self, data: Any) -> Set[str]:
        """提取模板变量"""
        variables = set()
        
        def extract_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.startswith('${') and key.endswith('}'):
                        variables.add(key[2:-1])
                    else:
                        extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
            elif isinstance(obj, str):
                import re
                matches = re.findall(r'\$\{([^}]+)\}', obj)
                variables.update(matches)
        
        extract_recursive(data)
        return variables
    
    def render(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """渲染模板"""
        rendered_data = json.loads(json.dumps(self.template_data))
        
        def render_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.startswith('${') and key.endswith('}'):
                        var_name = key[2:-1]
                        obj[key] = variables.get(var_name, key)
                    else:
                        render_recursive(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    render_recursive(item)
            elif isinstance(obj, str):
                import re
                def replace_var(match):
                    var_name = match.group(1)
                    return str(variables.get(var_name, match.group(0)))
                
                obj = re.sub(r'\$\{([^}]+)\}', replace_var, obj)
                return obj
        
        render_recursive(rendered_data)
        return rendered_data
    
    def validate_variables(self, variables: Dict[str, Any]) -> bool:
        """验证变量"""
        missing_vars = self.variables - set(variables.keys())
        if missing_vars:
            logger.error(f"缺少模板变量: {missing_vars}")
            return False
        return True


class TemplateManager:
    """模板管理器"""
    
    def __init__(self):
        self.templates: Dict[str, ConfigurationTemplate] = {}
    
    def create_template(self, template_name: str, template_data: Dict[str, Any]) -> ConfigurationTemplate:
        """创建模板"""
        template = ConfigurationTemplate(template_name, template_data)
        self.templates[template_name] = template
        logger.info(f"创建模板: {template_name}")
        return template
    
    def get_template(self, template_name: str) -> Optional[ConfigurationTemplate]:
        """获取模板"""
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self.templates.keys())
    
    def delete_template(self, template_name: str) -> bool:
        """删除模板"""
        if template_name in self.templates:
            del self.templates[template_name]
            logger.info(f"删除模板: {template_name}")
            return True
        return False
    
    def render_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        config_type: Type[BaseConfiguration]
    ) -> BaseConfiguration:
        """渲染模板并创建配置"""
        template = self.get_template(template_name)
        if not template:
            raise ConfigurationError(f"模板不存在: {template_name}")
        
        if not template.validate_variables(variables):
            raise ValidationError("模板变量验证失败")
        
        rendered_data = template.render(variables)
        config = config_type(**rendered_data)
        config.validate_config()
        
        return config


# 配置验证器扩展
class AdvancedConfigurationValidator:
    """高级配置验证器"""
    
    @staticmethod
    def validate_model_architecture(architecture: Dict[str, Any]) -> bool:
        """验证模型架构"""
        required_keys = ['type', 'layers']
        for key in required_keys:
            if key not in architecture:
                logger.error(f"模型架构缺少必要字段: {key}")
                return False
        
        # 验证层级结构
        layers = architecture.get('layers', [])
        if not isinstance(layers, list) or not layers:
            logger.error("模型层级必须是非空列表")
            return False
        
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict):
                logger.error(f"层级 {i} 必须是字典")
                return False
            
            if 'type' not in layer:
                logger.error(f"层级 {i} 缺少类型定义")
                return False
        
        return True
    
    @staticmethod
    def validate_data_schema(data_schema: Dict[str, Any]) -> bool:
        """验证数据模式"""
        required_fields = ['columns', 'data_types', 'constraints']
        for field in required_fields:
            if field not in data_schema:
                logger.error(f"数据模式缺少字段: {field}")
                return False
        
        columns = data_schema['columns']
        data_types = data_schema['data_types']
        
        if not isinstance(columns, list) or not columns:
            logger.error("列定义必须是非空列表")
            return False
        
        if not isinstance(data_types, dict):
            logger.error("数据类型定义必须是字典")
            return False
        
        # 验证列与类型的对应关系
        for column in columns:
            if column not in data_types:
                logger.error(f"列 {column} 缺少类型定义")
                return False
        
        return True
    
    @staticmethod
    def validate_performance_requirements(requirements: Dict[str, Any]) -> bool:
        """验证性能要求"""
        performance_metrics = ['latency', 'throughput', 'accuracy', 'memory_usage']
        
        for metric in performance_metrics:
            if metric in requirements:
                value = requirements[metric]
                if not isinstance(value, (int, float)) or value < 0:
                    logger.error(f"性能指标 {metric} 必须是非负数")
                    return False
        
        return True
    
    @staticmethod
    def validate_security_config(security_config: Dict[str, Any]) -> bool:
        """验证安全配置"""
        security_aspects = ['authentication', 'authorization', 'encryption', 'audit']
        
        for aspect in security_aspects:
            if aspect not in security_config:
                logger.warning(f"安全配置缺少方面: {aspect}")
        
        # 验证加密配置
        if 'encryption' in security_config:
            encryption_config = security_config['encryption']
            if 'algorithm' not in encryption_config:
                logger.error("加密配置缺少算法定义")
                return False
            
            valid_algorithms = ['aes256', 'rsa', 'ecc', 'chacha20_poly1305']
            if encryption_config['algorithm'] not in valid_algorithms:
                logger.error(f"不支持的加密算法: {encryption_config['algorithm']}")
                return False
        
        return True


# 配置分析器
class ConfigurationAnalyzer:
    """配置分析器"""
    
    def __init__(self, manager: LearningConfigurationManager):
        self.manager = manager
    
    def analyze_configuration_complexity(self, config: BaseConfiguration) -> Dict[str, Any]:
        """分析配置复杂度"""
        analysis = {
            'complexity_score': 0,
            'complexity_level': 'simple',
            'complexity_factors': [],
            'recommendations': []
        }
        
        config_dict = config.dict()
        
        # 分析嵌套层级
        max_depth = self._calculate_max_depth(config_dict)
        if max_depth > 5:
            analysis['complexity_factors'].append(f"深层嵌套结构 (深度: {max_depth})")
            analysis['complexity_score'] += max_depth * 2
        
        # 分析配置项数量
        config_items = self._count_config_items(config_dict)
        if config_items > 50:
            analysis['complexity_factors'].append(f"大量配置项 ({config_items})")
            analysis['complexity_score'] += config_items // 10
        
        # 分析数据类型多样性
        data_types = self._analyze_data_types(config_dict)
        if len(data_types) > 5:
            analysis['complexity_factors'].append(f"多样数据类型 ({len(data_types)})")
            analysis['complexity_score'] += len(data_types)
        
        # 确定复杂度等级
        if analysis['complexity_score'] < 10:
            analysis['complexity_level'] = 'simple'
        elif analysis['complexity_score'] < 25:
            analysis['complexity_level'] = 'moderate'
        elif analysis['complexity_score'] < 50:
            analysis['complexity_level'] = 'complex'
        else:
            analysis['complexity_level'] = 'very_complex'
        
        # 生成建议
        if max_depth > 5:
            analysis['recommendations'].append("考虑简化嵌套结构")
        
        if config_items > 50:
            analysis['recommendations'].append("考虑将配置分解为更小的部分")
        
        if len(data_types) > 5:
            analysis['recommendations'].append("考虑标准化数据类型")
        
        return analysis
    
    def _calculate_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """计算最大深度"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_max_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_max_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _count_config_items(self, obj: Any) -> int:
        """计算配置项数量"""
        if isinstance(obj, dict):
            return len(obj) + sum(self._count_config_items(v) for v in obj.values())
        elif isinstance(obj, list):
            return sum(self._count_config_items(item) for item in obj)
        else:
            return 1
    
    def _analyze_data_types(self, obj: Any) -> Set[str]:
        """分析数据类型"""
        data_types = set()
        
        def analyze_recursive(item):
            if isinstance(item, dict):
                for value in item.values():
                    analyze_recursive(value)
            elif isinstance(item, list):
                for element in item:
                    analyze_recursive(element)
            else:
                data_types.add(type(item).__name__)
        
        analyze_recursive(obj)
        return data_types
    
    def find_configuration_issues(self, config: BaseConfiguration) -> List[Dict[str, Any]]:
        """查找配置问题"""
        issues = []
        config_dict = config.dict()
        
        # 检查空值
        empty_values = self._find_empty_values(config_dict)
        if empty_values:
            issues.append({
                'type': 'empty_values',
                'severity': 'warning',
                'description': f"发现 {len(empty_values)} 个空值字段",
                'details': empty_values
            })
        
        # 检查重复值
        duplicate_values = self._find_duplicate_values(config_dict)
        if duplicate_values:
            issues.append({
                'type': 'duplicate_values',
                'severity': 'info',
                'description': f"发现 {len(duplicate_values)} 个重复值",
                'details': duplicate_values
            })
        
        # 检查不一致的类型
        type_inconsistencies = self._find_type_inconsistencies(config_dict)
        if type_inconsistencies:
            issues.append({
                'type': 'type_inconsistencies',
                'severity': 'error',
                'description': f"发现 {len(type_inconsistencies)} 个类型不一致",
                'details': type_inconsistencies
            })
        
        return issues
    
    def _find_empty_values(self, obj: Any, path: str = "") -> List[str]:
        """查找空值"""
        empty_paths = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if value is None or value == "" or (isinstance(value, (list, dict)) and not value):
                    empty_paths.append(current_path)
                else:
                    empty_paths.extend(self._find_empty_values(value, current_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                empty_paths.extend(self._find_empty_values(item, current_path))
        
        return empty_paths
    
    def _find_duplicate_values(self, obj: Any) -> List[Dict[str, Any]]:
        """查找重复值"""
        value_counts = defaultdict(int)
        duplicates = []
        
        def count_values(item, path=""):
            if isinstance(item, dict):
                for key, value in item.items():
                    current_path = f"{path}.{key}" if path else key
                    if not isinstance(value, (dict, list)):
                        value_counts[str(value)] += 1
                    count_values(value, current_path)
            elif isinstance(item, list):
                for i, element in enumerate(item):
                    current_path = f"{path}[{i}]"
                    if not isinstance(element, (dict, list)):
                        value_counts[str(element)] += 1
                    count_values(element, current_path)
        
        count_values(obj)
        
        for value, count in value_counts.items():
            if count > 1:
                duplicates.append({
                    'value': value,
                    'count': count
                })
        
        return duplicates
    
    def _find_type_inconsistencies(self, obj: Any) -> List[Dict[str, Any]]:
        """查找类型不一致"""
        type_info = defaultdict(list)
        inconsistencies = []
        
        def collect_types(item, path=""):
            if isinstance(item, dict):
                for key, value in item.items():
                    current_path = f"{path}.{key}" if path else key
                    if not isinstance(value, (dict, list)):
                        type_info[key].append((type(item).__name__, current_path))
                    collect_types(value, current_path)
            elif isinstance(item, list):
                for i, element in enumerate(item):
                    current_path = f"{path}[{i}]"
                    if not isinstance(element, (dict, list)):
                        type_info['list_element'].append((type(element).__name__, current_path))
                    collect_types(element, current_path)
        
        collect_types(obj)
        
        for field, types in type_info.items():
            type_names = [t[0] for t in types]
            if len(set(type_names)) > 1:
                inconsistencies.append({
                    'field': field,
                    'types': types
                })
        
        return inconsistencies


# 性能监控器
class ConfigurationPerformanceMonitor:
    """配置性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.thresholds: Dict[str, float] = {}
    
    def record_operation(self, operation: str, duration: float, config_id: str = None):
        """记录操作性能"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration': duration,
            'config_id': config_id
        }
        self.metrics[operation].append(metric)
        
        # 检查阈值
        if operation in self.thresholds:
            if duration > self.thresholds[operation]:
                logger.warning(f"操作 {operation} 性能超阈值: {duration}s > {self.thresholds[operation]}s")
    
    def set_threshold(self, operation: str, threshold: float):
        """设置性能阈值"""
        self.thresholds[operation] = threshold
    
    def get_performance_stats(self, operation: str) -> Dict[str, Any]:
        """获取性能统计"""
        if operation not in self.metrics:
            return {}
        
        metrics = self.metrics[operation]
        durations = [m['duration'] for m in metrics]
        
        if not durations:
            return {}
        
        if np is not None:
            return {
                'operation': operation,
                'total_operations': len(durations),
                'avg_duration': np.mean(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'std_duration': np.std(durations),
                'p50_duration': np.percentile(durations, 50),
                'p95_duration': np.percentile(durations, 95),
                'p99_duration': np.percentile(durations, 99)
            }
        else:
            # 使用Python内置函数作为后备
            durations_sorted = sorted(durations)
            n = len(durations_sorted)
            return {
                'operation': operation,
                'total_operations': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'std_duration': (sum((x - sum(durations) / len(durations)) ** 2 for x in durations) / len(durations)) ** 0.5,
                'p50_duration': durations_sorted[n // 2],
                'p95_duration': durations_sorted[int(n * 0.95)],
                'p99_duration': durations_sorted[int(n * 0.99)]
            }
    
    def get_recent_performance(self, operation: str, minutes: int = 60) -> List[Dict[str, Any]]:
        """获取最近性能数据"""
        if operation not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = []
        
        for metric in self.metrics[operation]:
            metric_time = datetime.fromisoformat(metric['timestamp'])
            if metric_time >= cutoff_time:
                recent_metrics.append(metric)
        
        return recent_metrics


# 配置优化器
class ConfigurationOptimizer:
    """配置优化器"""
    
    def __init__(self, manager: LearningConfigurationManager):
        self.manager = manager
    
    def optimize_configuration(self, config: BaseConfiguration) -> Dict[str, Any]:
        """优化配置"""
        optimization_report = {
            'original_config_id': config.id,
            'optimizations_applied': [],
            'performance_improvements': {},
            'recommendations': []
        }
        
        config_dict = config.dict()
        
        # 优化数据格式
        optimized_dict = self._optimize_data_formats(config_dict)
        if optimized_dict != config_dict:
            optimization_report['optimizations_applied'].append('data_format_optimization')
            optimization_report['performance_improvements']['data_size_reduction'] = True
        
        # 优化参数设置
        optimized_params = self._optimize_parameters(config_dict)
        if optimized_params != config_dict:
            optimization_report['optimizations_applied'].append('parameter_optimization')
            optimization_report['performance_improvements']['parameter_efficiency'] = True
        
        # 生成优化建议
        recommendations = self._generate_optimization_recommendations(config)
        optimization_report['recommendations'] = recommendations
        
        return optimization_report
    
    def _optimize_data_formats(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """优化数据格式"""
        optimized = json.loads(json.dumps(config_dict))
        
        # 转换长字符串为更紧凑的格式
        def optimize_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and len(value) > 100:
                        # 考虑使用更紧凑的格式
                        pass
                    else:
                        optimize_recursive(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    optimize_recursive(item)
        
        optimize_recursive(optimized)
        return optimized
    
    def _optimize_parameters(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """优化参数"""
        optimized = json.loads(json.dumps(config_dict))
        
        # 优化超参数范围
        if 'hyperparameters' in optimized:
            hp = optimized['hyperparameters']
            for key, value in hp.items():
                if isinstance(value, float) and value != int(value):
                    # 考虑使用整数如果精度允许
                    if abs(value - round(value)) < 1e-10:
                        hp[key] = int(round(value))
        
        return optimized
    
    def _generate_optimization_recommendations(self, config: BaseConfiguration) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        config_dict = config.dict()
        
        # 检查配置大小
        config_size = len(json.dumps(config_dict))
        if config_size > 10000:  # 10KB
            recommendations.append("配置较大，考虑分解为多个子配置")
        
        # 检查嵌套深度
        max_depth = self._calculate_max_depth(config_dict)
        if max_depth > 5:
            recommendations.append("嵌套层级较深，考虑扁平化结构")
        
        # 检查重复配置
        repeated_sections = self._find_repeated_sections(config_dict)
        if repeated_sections:
            recommendations.append(f"发现重复配置片段: {repeated_sections}")
        
        return recommendations
    
    def _calculate_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """计算最大深度"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_max_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_max_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _find_repeated_sections(self, obj: Any) -> List[str]:
        """查找重复配置片段"""
        # 简化的重复检测
        sections = []
        obj_str = json.dumps(obj, sort_keys=True)
        
        # 这里可以实现更复杂的重复检测逻辑
        return sections


# 扩展学习配置管理器
class ExtendedLearningConfigurationManager(LearningConfigurationManager):
    """扩展的学习配置管理器"""
    
    def __init__(self, storage_path: Union[str, Path] = "extended_configs"):
        super().__init__(storage_path)
        storage_path_obj = Path(storage_path)
        self.advanced_manager = AdvancedConfigurationManager(storage_path_obj / "advanced")
        self.template_manager = TemplateManager()
        self.analyzer = ConfigurationAnalyzer(self)
        self.performance_monitor = ConfigurationPerformanceMonitor()
        self.optimizer = ConfigurationOptimizer(self)
        
        # 注册扩展配置类型
        self._register_extended_configurations()
        
        logger.info("扩展学习配置管理器初始化完成")
    
    def _register_extended_configurations(self):
        """注册扩展配置类型"""
        self.config_manager.register_configuration_type(MonitoringConfiguration)
        self.config_manager.register_configuration_type(DataPipelineConfiguration)
        self.config_manager.register_configuration_type(ABTestConfiguration)
        self.config_manager.register_configuration_type(ModelRegistryConfiguration)
        self.config_manager.register_configuration_type(FeatureStoreConfiguration)
    
    # 扩展配置创建方法
    def create_monitoring_configuration(
        self,
        model_config_id: str,
        monitoring_metrics: List[str],
        name: str,
        monitoring_frequency: str = "real_time",
        description: Optional[str] = None,
        **kwargs
    ) -> MonitoringConfiguration:
        """创建监控配置"""
        if model_config_id not in self.model_configs:
            raise ConfigurationError(f"模型配置不存在: {model_config_id}")
        
        config = self.config_manager.create_configuration(
            MonitoringConfiguration,
            model_config_id=model_config_id,
            monitoring_metrics=monitoring_metrics,
            monitoring_frequency=monitoring_frequency,
            name=name,
            description=description,
            **kwargs
        )
        
        logger.info(f"创建监控配置: {config.name} ({config.id})")
        return config
    
    def create_data_pipeline_configuration(
        self,
        pipeline_name: str,
        data_sources: List[Dict[str, Any]],
        data_storage_config: Dict[str, Any],
        name: str,
        description: Optional[str] = None,
        **kwargs
    ) -> DataPipelineConfiguration:
        """创建数据管道配置"""
        config = self.config_manager.create_configuration(
            DataPipelineConfiguration,
            pipeline_name=pipeline_name,
            data_sources=data_sources,
            data_storage_config=data_storage_config,
            name=name,
            description=description,
            **kwargs
        )
        
        logger.info(f"创建数据管道配置: {config.name} ({config.id})")
        return config
    
    def create_ab_test_configuration(
        self,
        test_name: str,
        test_hypothesis: str,
        control_group_config: Dict[str, Any],
        treatment_group_config: Dict[str, Any],
        test_metrics: List[str],
        test_duration: int,
        name: str,
        description: Optional[str] = None,
        **kwargs
    ) -> ABTestConfiguration:
        """创建A/B测试配置"""
        config = self.config_manager.create_configuration(
            ABTestConfiguration,
            test_name=test_name,
            test_hypothesis=test_hypothesis,
            control_group_config=control_group_config,
            treatment_group_config=treatment_group_config,
            test_metrics=test_metrics,
            test_duration=test_duration,
            name=name,
            description=description,
            **kwargs
        )
        
        logger.info(f"创建A/B测试配置: {config.name} ({config.id})")
        return config
    
    def create_model_registry_configuration(
        self,
        registry_name: str,
        model_storage_backend: str,
        name: str,
        description: Optional[str] = None,
        **kwargs
    ) -> ModelRegistryConfiguration:
        """创建模型注册表配置"""
        config = self.config_manager.create_configuration(
            ModelRegistryConfiguration,
            registry_name=registry_name,
            model_storage_backend=model_storage_backend,
            name=name,
            description=description,
            **kwargs
        )
        
        logger.info(f"创建模型注册表配置: {config.name} ({config.id})")
        return config
    
    def create_feature_store_configuration(
        self,
        feature_store_name: str,
        feature_definitions: List[Dict[str, Any]],
        data_sources: List[Dict[str, Any]],
        name: str,
        description: Optional[str] = None,
        **kwargs
    ) -> FeatureStoreConfiguration:
        """创建特征存储配置"""
        config = self.config_manager.create_configuration(
            FeatureStoreConfiguration,
            feature_store_name=feature_store_name,
            feature_definitions=feature_definitions,
            data_sources=data_sources,
            name=name,
            description=description,
            **kwargs
        )
        
        logger.info(f"创建特征存储配置: {config.name} ({config.id})")
        return config
    
    # 高级分析功能
    def analyze_configuration(self, config_id: str) -> Dict[str, Any]:
        """分析配置"""
        config = self.config_manager.get_configuration(config_id)
        if not config:
            raise ConfigurationError(f"配置不存在: {config_id}")
        
        # 复杂度分析
        complexity_analysis = self.analyzer.analyze_configuration_complexity(config)
        
        # 问题检测
        issues = self.analyzer.find_configuration_issues(config)
        
        # 性能优化
        optimization_report = self.optimizer.optimize_configuration(config)
        
        analysis_report = {
            'config_id': config_id,
            'config_name': config.name,
            'analysis_timestamp': datetime.now().isoformat(),
            'complexity_analysis': complexity_analysis,
            'issues_found': issues,
            'optimization_report': optimization_report,
            'overall_score': self._calculate_overall_score(complexity_analysis, issues)
        }
        
        return analysis_report
    
    def _calculate_overall_score(self, complexity_analysis: Dict[str, Any], issues: List[Dict[str, Any]]) -> float:
        """计算总体评分"""
        score = 100.0
        
        # 根据复杂度扣分
        complexity_penalty = min(complexity_analysis['complexity_score'] * 2, 50)
        score -= complexity_penalty
        
        # 根据问题扣分
        for issue in issues:
            if issue['severity'] == 'error':
                score -= 20
            elif issue['severity'] == 'warning':
                score -= 10
            elif issue['severity'] == 'info':
                score -= 5
        
        return max(0.0, score)
    
    # 模板功能
    def create_configuration_from_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        config_type: Type[BaseConfiguration],
        name: str,
        description: Optional[str] = None
    ) -> BaseConfiguration:
        """从模板创建配置"""
        config = self.template_manager.render_template(template_name, variables, config_type)
        config.name = name
        if description:
            config.description = description
        
        config.validate_config()
        self.config_manager._configurations[config.id] = config
        self.config_manager._save_configuration(config)
        
        logger.info(f"从模板创建配置: {name} ({config.id})")
        return config
    
    def save_as_template(self, config_id: str, template_name: str) -> ConfigurationTemplate:
        """将配置保存为模板"""
        config = self.config_manager.get_configuration(config_id)
        if not config:
            raise ConfigurationError(f"配置不存在: {config_id}")
        
        template_data = config.dict()
        template = self.template_manager.create_template(template_name, template_data)
        
        logger.info(f"保存配置为模板: {template_name}")
        return template
    
    # 批量操作
    def batch_optimize_configurations(self, config_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量优化配置"""
        optimization_results = {}
        
        for config_id in config_ids:
            try:
                config = self.config_manager.get_configuration(config_id)
                if config:
                    result = self.optimizer.optimize_configuration(config)
                    optimization_results[config_id] = result
                else:
                    optimization_results[config_id] = {'error': '配置不存在'}
            except Exception as e:
                optimization_results[config_id] = {'error': str(e)}
        
        return optimization_results
    
    def batch_analyze_configurations(self, config_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量分析配置"""
        analysis_results = {}
        
        for config_id in config_ids:
            try:
                result = self.analyze_configuration(config_id)
                analysis_results[config_id] = result
            except Exception as e:
                analysis_results[config_id] = {'error': str(e)}
        
        return analysis_results
    
    # 性能监控
    def monitor_configuration_performance(self, config_id: str, operation: str, duration: float):
        """监控配置性能"""
        self.performance_monitor.record_operation(operation, duration, config_id)
    
    def get_performance_report(self, operation: str) -> Dict[str, Any]:
        """获取性能报告"""
        return self.performance_monitor.get_performance_stats(operation)
    
    def set_performance_threshold(self, operation: str, threshold: float):
        """设置性能阈值"""
        self.performance_monitor.set_threshold(operation, threshold)


# 演示扩展功能
def demonstrate_extended_features():
    """演示扩展功能"""
    print("=== 扩展功能演示 ===")
    
    # 创建扩展管理器
    manager = ExtendedLearningConfigurationManager("extended_demo_configs")
    
    try:
        # 创建监控配置
        model_config = manager.create_model_configuration(
            name="扩展演示模型",
            model_type=ModelTypeEnum.CLASSIFICATION,
            model_architecture={"type": "simple", "layers": []},
            input_spec={"shape": [10]},
            output_spec={"shape": [2]}
        )
        
        monitoring_config = manager.create_monitoring_configuration(
            model_config_id=model_config.id,
            monitoring_metrics=["accuracy", "latency", "memory_usage"],
            name="模型监控配置",
            monitoring_frequency="real_time"
        )
        
        # 创建数据管道配置
        pipeline_config = manager.create_data_pipeline_configuration(
            pipeline_name="数据处理管道",
            data_sources=[
                {"type": "database", "connection": "postgresql://..."},
                {"type": "api", "endpoint": "https://api.example.com/data"}
            ],
            data_storage_config={
                "backend": "s3",
                "bucket": "ml-data-pipeline",
                "format": "parquet"
            },
            name="数据管道配置"
        )
        
        # 创建A/B测试配置
        ab_test_config = manager.create_ab_test_configuration(
            test_name="模型版本对比测试",
            test_hypothesis="新模型版本将提高准确率",
            control_group_config={"model_version": "v1.0"},
            treatment_group_config={"model_version": "v2.0"},
            test_metrics=["accuracy", "precision", "recall"],
            test_duration=30,
            name="A/B测试配置"
        )
        
        # 分析配置
        print("分析模型配置...")
        analysis = manager.analyze_configuration(model_config.id)
        print(f"配置分析完成，评分: {analysis['overall_score']}")
        
        # 创建模板
        print("创建配置模板...")
        template = manager.save_as_template(model_config.id, "classification_model_template")
        print(f"模板变量: {template.variables}")
        
        # 从模板创建配置
        print("从模板创建配置...")
        new_config = manager.create_configuration_from_template(
            template_name="classification_model_template",
            variables={
                "name": "模板生成模型",
                "description": "从模板生成的新模型"
            },
            config_type=MLModelConfiguration,
            name="模板生成模型"
        )
        print(f"从模板创建配置: {new_config.id}")
        
        # 批量操作
        config_ids = [model_config.id, monitoring_config.id, pipeline_config.id]
        
        print("批量分析配置...")
        batch_analysis = manager.batch_analyze_configurations(config_ids)
        print(f"批量分析完成: {len(batch_analysis)} 个配置")
        
        print("批量优化配置...")
        batch_optimization = manager.batch_optimize_configurations(config_ids)
        print(f"批量优化完成: {len(batch_optimization)} 个配置")
        
        # 性能监控
        print("性能监控演示...")
        manager.monitor_configuration_performance(model_config.id, "validation", 1.23)
        manager.monitor_configuration_performance(model_config.id, "training", 45.67)
        
        performance_report = manager.get_performance_report("validation")
        print(f"验证性能报告: {performance_report}")
        
        manager.set_performance_threshold("validation", 2.0)
        
    finally:
        manager.shutdown()


# 完整系统演示
def run_complete_system_demo():
    """运行完整系统演示"""
    print("K7学习配置管理器完整系统演示")
    print("=" * 60)
    
    try:
        # 基础功能演示
        demonstrate_configuration_management()
        
        print("\n" + "=" * 60)
        
        # 生命周期管理演示
        demonstrate_lifecycle_management()
        
        print("\n" + "=" * 60)
        
        # 异步处理演示
        asyncio.run(demonstrate_async_processing())
        
        print("\n" + "=" * 60)
        
        # 扩展功能演示
        demonstrate_extended_features()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        
    except Exception as e:
        print(f"演示过程中出错: {e}")
        logger.error(f"演示失败: {e}")


if __name__ == "__main__":
    run_complete_system_demo()