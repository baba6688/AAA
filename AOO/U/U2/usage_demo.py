#!/usr/bin/env python3
"""
U2模块使用演示脚本
================

展示如何使用U2模块的导出接口
"""

def demo_import():
    """演示导入方式"""
    print("=" * 60)
    print("U2模块导入演示")
    print("=" * 60)
    
    print("# 方式1: 导入整个模块")
    print("import U.U2")
    print()
    
    print("# 方式2: 从模块导入特定类")
    print("from U.U2 import BaseNeuralNetwork, ConvolutionalNeuralNetwork")
    print()
    
    print("# 方式3: 导入所有核心类")
    print("from U.U2 import (")
    print("    BaseNeuralNetwork,")
    print("    ConvolutionalNeuralNetwork,") 
    print("    RecurrentNeuralNetwork,")
    print("    MultiHeadAttention,")
    print("    Transformer,")
    print("    Autoencoder,")
    print("    GAN,")
    print("    VariationalAutoencoder,")
    print("    DQNAgent,")
    print("    ModelTrainer,")
    print("    DLAlgorithmLibrary")
    print(")")
    print()
    
    print("# 方式4: 使用__all__控制导入")
    print("from U.U2 import *")
    print("print(dir())  # 查看所有可用类")
    print()

def demo_usage():
    """演示基本使用"""
    print("=" * 60)
    print("基本使用演示")
    print("=" * 60)
    
    print("# 导入主库类")
    print("from U.U2 import DLAlgorithmLibrary")
    print()
    
    print("# 创建库实例")
    print("dl_lib = DLAlgorithmLibrary()")
    print()
    
    print("# 创建各种模型")
    print("# 1. 基础神经网络")
    print("model = dl_lib.create_base_network(")
    print("    name='my_network',")
    print("    input_dim=100,")
    print("    hidden_dims=[128, 64, 32],")
    print("    output_dim=10")
    print(")")
    print()
    
    print("# 2. 卷积神经网络")
    print("conv_configs = [")
    print("    {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},")
    print("    {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}")
    print("]")
    print("cnn = dl_lib.create_cnn(")
    print("    name='my_cnn',")
    print("    input_channels=3,")
    print("    conv_configs=conv_configs,")
    print("    fc_dims=[128, 64],")
    print("    output_dim=10")
    print(")")
    print()
    
    print("# 3. 自编码器")
    print("autoencoder = dl_lib.create_autoencoder(")
    print("    name='my_autoencoder',")
    print("    input_dim=784,")
    print("    latent_dim=32")
    print(")")
    print()
    
    print("# 4. GAN")
    print("gan = dl_lib.create_gan(")
    print("    name='my_gan',")
    print("    latent_dim=100,")
    print("    hidden_dims=[256, 512],")
    print("    output_dim=784")
    print(")")
    print()
    
    print("# 5. DQN智能体")
    print("dqn = dl_lib.create_dqn_agent(")
    print("    name='my_dqn',")
    print("    state_dim=4,")
    print("    action_dim=2")
    print(")")
    print()

def demo_direct_usage():
    """演示直接使用类"""
    print("=" * 60)
    print("直接使用类演示")
    print("=" * 60)
    
    print("# 直接导入并创建模型")
    print("from U.U2 import BaseNeuralNetwork, Autoencoder, GAN")
    print()
    
    print("# 注意：这些类需要PyTorch环境才能实际运行")
    print("# 以下是代码示例：")
    print()
    
    print("# 基础神经网络")
    print("model = BaseNeuralNetwork(")
    print("    input_dim=10,")
    print("    hidden_dims=[64, 32],")
    print("    output_dim=5,")
    print("    activation='relu',")
    print("    dropout_rate=0.1")
    print(")")
    print()
    
    print("# 自编码器")
    print("autoencoder = Autoencoder(")
    print("    input_dim=784,")
    print("    latent_dim=32,")
    print("    hidden_dims=[512, 256, 128]")
    print(")")
    print()
    
    print("# GAN")
    print("gan = GAN(")
    print("    latent_dim=100,")
    print("    hidden_dims=[256, 512],")
    print("    output_dim=784")
    print(")")
    print()

def demo_utility_functions():
    """演示工具函数"""
    print("=" * 60)
    print("工具函数演示")
    print("=" * 60)
    
    print("# 获取库信息")
    print("from U.U2 import get_library_info")
    print("info = get_library_info()")
    print("print(f'库名称: {info[\"name\"]}')")
    print("print(f'版本: {info[\"version\"]}')")
    print("print(f'组件数量: {info[\"classes_count\"]}')")
    print()
    
    print("# 列出可用模型")
    print("from U.U2 import list_available_models")
    print("models = list_available_models()")
    print("for category, model_list in models.items():")
    print("    print(f'{category}: {model_list}')")
    print()

def demo_training_tools():
    """演示训练工具"""
    print("=" * 60)
    print("训练工具演示")
    print("=" * 60)
    
    print("# 使用ModelTrainer训练模型")
    print("from U.U2 import ModelTrainer")
    print()
    
    print("# 注意：需要实际的PyTorch环境和数据")
    print("# 以下是代码示例：")
    print()
    
    print("trainer = ModelTrainer(")
    print("    model=my_model,")
    print("    optimizer=None,  # 使用默认Adam优化器")
    print("    criterion=None,  # 使用默认CrossEntropyLoss")
    print("    device=None      # 自动选择设备")
    print(")")
    print()
    
    print("# 设置学习率调度器")
    print("trainer.set_scheduler('step', step_size=30, gamma=0.1)")
    print()
    
    print("# 训练模型")
    print("history = trainer.train(")
    print("    train_loader=train_loader,")
    print("    val_loader=val_loader,")
    print("    epochs=100,")
    print("    save_path='model.pth',")
    print("    early_stopping_patience=10")
    print(")")
    print()
    
    print("# 绘制训练历史")
    print("trainer.plot_training_history('training_curves.png')")
    print()

def demo_model_compression():
    """演示模型压缩"""
    print("=" * 60)
    print("模型压缩演示")
    print("=" * 60)
    
    print("# 使用ModelPruner剪枝模型")
    print("from U.U2 import ModelPruner")
    print()
    
    print("pruner = ModelPruner(")
    print("    model=my_model,")
    print("    pruning_ratio=0.5,  # 剪除50%参数")
    print("    method='magnitude'   # 幅度剪枝")
    print(")")
    print()
    
    print("pruner.prune_model()")
    print("stats = pruner.get_pruning_stats()")
    print("print(f'剪枝比例: {stats[\"actual_pruning_ratio\"]:.2%}')")
    print()
    
    print("# 使用ModelQuantizer量化模型")
    print("from U.U2 import ModelQuantizer")
    print()
    
    print("quantizer = ModelQuantizer(")
    print("    model=my_model,")
    print("    bit_width=8,           # 8位量化")
    print("    quantization_type='dynamic'  # 动态量化")
    print(")")
    print()
    
    print("quantized_model = quantizer.dynamic_quantize()")
    print("stats = quantizer.get_quantization_stats()")
    print("print(f'压缩比: {stats[\"compression_ratio\"]:.2f}x')")
    print()

def main():
    """主函数"""
    print("U2深度学习算法库使用演示")
    print("=" * 60)
    print("此演示展示了U2模块的各种导入和使用方式")
    print("注意：实际运行需要安装PyTorch等依赖库")
    print("=" * 60)
    print()
    
    demo_import()
    demo_usage()
    demo_direct_usage()
    demo_utility_functions()
    demo_training_tools()
    demo_model_compression()
    
    print("=" * 60)
    print("演示完成")
    print("=" * 60)
    print("更多信息请查看U2模块的文档和源码")
    print("模块位置: /workspace/U/U2/")
    print("- __init__.py: 导出接口定义")
    print("- DLAlgorithmLibrary.py: 主要实现")

if __name__ == "__main__":
    main()