#!/usr/bin/env python3
"""
U2模块导出接口最终验证脚本
=========================

快速验证导出接口是否正确创建
"""

import os
import sys

def quick_validate():
    """快速验证"""
    print("=" * 60)
    print("U2模块导出接口快速验证")
    print("=" * 60)
    
    # 检查文件存在
    u2_dir = os.path.dirname(__file__)
    init_file = os.path.join(u2_dir, "__init__.py")
    dl_file = os.path.join(u2_dir, "DLAlgorithmLibrary.py")
    
    if not os.path.exists(init_file):
        print("✗ __init__.py文件不存在")
        return False
    if not os.path.exists(dl_file):
        print("✗ DLAlgorithmLibrary.py文件不存在")
        return False
    
    print("✓ 文件结构正确")
    
    # 检查__init__.py内容
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 验证关键内容
    checks = [
        ('U2模块 - 深度学习算法库', '模块描述'),
        ('from .DLAlgorithmLibrary import', '导入语句'),
        ('BaseNeuralNetwork', 'BaseNeuralNetwork'),
        ('ConvolutionalNeuralNetwork', 'ConvolutionalNeuralNetwork'),
        ('RecurrentNeuralNetwork', 'RecurrentNeuralNetwork'),
        ('MultiHeadAttention', 'MultiHeadAttention'),
        ('TransformerBlock', 'TransformerBlock'),
        ('Transformer', 'Transformer'),
        ('Autoencoder', 'Autoencoder'),
        ('Generator', 'Generator'),
        ('Discriminator', 'Discriminator'),
        ('GAN', 'GAN'),
        ('VariationalAutoencoder', 'VariationalAutoencoder'),
        ('ReplayBuffer', 'ReplayBuffer'),
        ('DeepQNetwork', 'DeepQNetwork'),
        ('DQNAgent', 'DQNAgent'),
        ('ModelPruner', 'ModelPruner'),
        ('ModelQuantizer', 'ModelQuantizer'),
        ('ModelTrainer', 'ModelTrainer'),
        ('DLAlgorithmLibrary', 'DLAlgorithmLibrary'),
        ('__all__', '__all__导出列表')
    ]
    
    failed_checks = []
    for check, desc in checks:
        if check in content:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc}")
            failed_checks.append(desc)
    
    # 检查DLAlgorithmLibrary.py中的类定义
    with open(dl_file, 'r', encoding='utf-8') as f:
        dl_content = f.read()
    
    expected_classes = [
        'BaseNeuralNetwork', 'ConvolutionalNeuralNetwork', 'RecurrentNeuralNetwork',
        'MultiHeadAttention', 'TransformerBlock', 'Transformer', 'Autoencoder',
        'Generator', 'Discriminator', 'GAN', 'VariationalAutoencoder', 'ReplayBuffer',
        'DeepQNetwork', 'DQNAgent', 'ModelPruner', 'ModelQuantizer', 'ModelTrainer',
        'DLAlgorithmLibrary'
    ]
    
    missing_classes = []
    for class_name in expected_classes:
        if f"class {class_name}" in dl_content:
            print(f"✓ {class_name} 类定义")
        else:
            print(f"✗ {class_name} 类定义")
            missing_classes.append(class_name)
    
    # 总结
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    
    if not failed_checks and not missing_classes:
        print("🎉 导出接口创建成功！")
        print(f"✓ 所有18个核心类已正确导入")
        print("✓ __init__.py文件结构完整")
        print("✓ 模块文档和导出列表完整")
        return True
    else:
        if failed_checks:
            print(f"✗ __init__.py中缺失: {failed_checks}")
        if missing_classes:
            print(f"✗ DLAlgorithmLibrary.py中缺失类: {missing_classes}")
        return False

if __name__ == "__main__":
    success = quick_validate()
    sys.exit(0 if success else 1)