"""
V1模块 - 模型训练器模块

这个模块包含了V1版本的模型训练器类，提供统一的模型训练接口，
支持自动数据预处理、交叉验证训练、早停机制、分布式训练、
超参数调优、模型性能监控、训练进度跟踪和错误处理恢复等功能。
"""

from .ModelTrainer import (
    # 核心类和枚举
    TrainingConfig,
    TrainingResult,
    ModelTrainer,
    
    # 便利函数
    create_model_trainer
)

__all__ = [
    'TrainingConfig',
    'TrainingResult',
    'ModelTrainer',
    'create_model_trainer'
]

__version__ = '1.0.0'

# 便利函数
def create_model_trainer(config: TrainingConfig = None, **kwargs):
    """
    创建模型训练器实例
    
    Args:
        config: 训练配置对象
        **kwargs: 训练器参数
    
    Returns:
        ModelTrainer: 模型训练器实例
    
    Examples:
        from V1 import create_model_trainer, TrainingConfig
        
        # 使用默认配置创建训练器
        trainer = create_model_trainer()
        
        # 使用自定义配置创建训练器
        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            early_stopping=True,
            distributed=True
        )
        trainer = create_model_trainer(config=config)
    """
    return ModelTrainer(config=config, **kwargs)

# 快速开始指南
QUICK_START = """
V1模型训练器快速开始：

1. 创建训练器：
   from V1 import create_model_trainer, TrainingConfig
   trainer = create_model_trainer()

2. 配置训练参数：
   config = TrainingConfig(
       epochs=100,
       batch_size=32,
       learning_rate=0.001,
       early_stopping=True
   )
   trainer = create_model_trainer(config=config)

3. 训练模型：
   # 准备数据
   X_train, y_train, X_val, y_val = load_your_data()
   
   # 训练模型
   result = trainer.train(X_train, y_train, X_val, y_val)
   
4. 获取训练结果：
   print(f"最佳模型路径: {result['best_model_path']}")
   print(f"训练历史: {result['history']}")

5. 保存模型：
   trainer.save_model('my_model.pkl')
"""

print("V1模型训练器已加载")