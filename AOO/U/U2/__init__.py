"""
U2模块 - 深度学习算法库
=======================

这是一个完整的深度学习算法库模块，提供各种神经网络架构和训练方法。
支持从基础神经网络到高级架构如Transformer、GAN、VAE等。

主要功能：
1. 神经网络基础架构
2. 卷积神经网络(CNN)
3. 循环神经网络(RNN/LSTM/GRU)
4. 注意力机制和Transformer
5. 自编码器
6. 生成对抗网络(GAN)
7. 变分自编码器(VAE)
8. 深度强化学习
9. 模型压缩和量化
10. 训练和评估工具

版本: 1.0.0
创建时间: 2025-11-14
作者: U2团队
"""

# 从DLAlgorithmLibrary模块导入所有核心类
from .DLAlgorithmLibrary import (
    # 1. 基础神经网络
    BaseNeuralNetwork,
    
    # 2. 卷积神经网络
    ConvolutionalNeuralNetwork,
    
    # 3. 循环神经网络
    RecurrentNeuralNetwork,
    
    # 4. 注意力机制和Transformer
    MultiHeadAttention,
    TransformerBlock,
    Transformer,
    
    # 5. 自编码器
    Autoencoder,
    
    # 6. 生成对抗网络
    Generator,
    Discriminator,
    GAN,
    
    # 7. 变分自编码器
    VariationalAutoencoder,
    
    # 8. 深度强化学习
    ReplayBuffer,
    DeepQNetwork,
    DQNAgent,
    
    # 9. 模型压缩和量化
    ModelPruner,
    ModelQuantizer,
    
    # 10. 训练和评估工具
    ModelTrainer,
    
    # 11. 主算法库类
    DLAlgorithmLibrary
)

# 定义模块的导出列表，控制from module import * 的行为
__all__ = [
    # 基础组件
    'BaseNeuralNetwork',
    
    # 神经网络架构
    'ConvolutionalNeuralNetwork',
    'RecurrentNeuralNetwork',
    
    # 注意力机制和Transformer
    'MultiHeadAttention',
    'TransformerBlock', 
    'Transformer',
    
    # 自编码器相关
    'Autoencoder',
    'VariationalAutoencoder',
    
    # 生成对抗网络
    'Generator',
    'Discriminator',
    'GAN',
    
    # 深度强化学习
    'ReplayBuffer',
    'DeepQNetwork',
    'DQNAgent',
    
    # 模型优化
    'ModelPruner',
    'ModelQuantizer',
    
    # 训练工具
    'ModelTrainer',
    
    # 主库类
    'DLAlgorithmLibrary'
]

# 模块版本信息
__version__ = "1.0.0"
__author__ = "U2团队"
__email__ = "u2-team@example.com"

# 模块描述
__doc__ = """
U2深度学习算法库模块

这是一个完整的深度学习算法库，提供了：

神经网络基础：
- BaseNeuralNetwork: 基础全连接神经网络
- ConvolutionalNeuralNetwork: 卷积神经网络
- RecurrentNeuralNetwork: 循环神经网络(LSTM/GRU)

高级架构：
- MultiHeadAttention: 多头注意力机制
- TransformerBlock: Transformer块
- Transformer: 完整的Transformer模型

生成模型：
- Autoencoder: 自编码器
- VariationalAutoencoder: 变分自编码器(VAE)
- Generator: GAN生成器
- Discriminator: GAN判别器
- GAN: 生成对抗网络

强化学习：
- ReplayBuffer: 经验回放缓冲区
- DeepQNetwork: 深度Q网络
- DQNAgent: DQN智能体

模型优化：
- ModelPruner: 模型剪枝器
- ModelQuantizer: 模型量化器

训练工具：
- ModelTrainer: 模型训练器

主库类：
- DLAlgorithmLibrary: 整合所有功能的算法库主类

使用示例：
    from U.U2 import BaseNeuralNetwork, ModelTrainer, DLAlgorithmLibrary
    
    # 创建基础神经网络
    model = BaseNeuralNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=5)
    
    # 使用算法库
    dl_lib = DLAlgorithmLibrary()
    model = dl_lib.create_base_network("my_model", 10, [64, 32], 5)
"""

# 便捷导入函数
def get_library_info():
    """获取库信息"""
    return {
        'name': 'U2深度学习算法库',
        'version': __version__,
        'author': __author__,
        'classes_count': len(__all__),
        'classes': __all__
    }

def list_available_models():
    """列出所有可用的模型类"""
    return {
        '基础网络': ['BaseNeuralNetwork'],
        '卷积网络': ['ConvolutionalNeuralNetwork'],
        '循环网络': ['RecurrentNeuralNetwork'],
        '注意力机制': ['MultiHeadAttention', 'TransformerBlock', 'Transformer'],
        '自编码器': ['Autoencoder', 'VariationalAutoencoder'],
        '生成模型': ['Generator', 'Discriminator', 'GAN'],
        '强化学习': ['ReplayBuffer', 'DeepQNetwork', 'DQNAgent'],
        '模型优化': ['ModelPruner', 'ModelQuantizer'],
        '训练工具': ['ModelTrainer'],
        '主库': ['DLAlgorithmLibrary']
    }

# 模块初始化信息
def _init_module():
    """模块初始化时的提示信息"""
    import sys
    
    print("=" * 60)
    print("U2深度学习算法库 v{}".format(__version__))
    print("作者: {}".format(__author__))
    print("=" * 60)
    print("可用组件数量: {}".format(len(__all__)))
    print()
    print("主要功能:")
    print("✓ 基础神经网络架构")
    print("✓ 卷积神经网络(CNN)")
    print("✓ 循环神经网络(RNN/LSTM/GRU)")
    print("✓ Transformer和注意力机制")
    print("✓ 自编码器和变分自编码器")
    print("✓ 生成对抗网络(GAN)")
    print("✓ 深度强化学习(DQN)")
    print("✓ 模型压缩(剪枝和量化)")
    print("✓ 训练和评估工具")
    print()
    print("快速开始:")
    print("  from U.U2 import BaseNeuralNetwork, DLAlgorithmLibrary")
    print("  model = BaseNeuralNetwork(10, [64, 32], 5)")
    print("  dl_lib = DLAlgorithmLibrary()")
    print("=" * 60)
    
    return True

# 在模块导入时显示初始化信息
try:
    _init_module()
except Exception:
    # 静默处理初始化错误，避免影响导入
    pass

# 导出模块元信息
__all__.extend([
    '__version__',
    '__author__', 
    '__email__',
    '__doc__',
    'get_library_info',
    'list_available_models'
])