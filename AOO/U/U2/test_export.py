#!/usr/bin/env python3
"""
U2模块导出接口测试脚本
====================

测试所有18个类是否能够正确导入和实例化
"""

import sys
import os
import traceback

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试所有类的导入"""
    print("=" * 60)
    print("测试U2模块导出接口")
    print("=" * 60)
    
    try:
        # 测试导入所有类
        from U.U2 import (
            BaseNeuralNetwork,
            ConvolutionalNeuralNetwork,
            RecurrentNeuralNetwork,
            MultiHeadAttention,
            TransformerBlock,
            Transformer,
            Autoencoder,
            Generator,
            Discriminator,
            GAN,
            VariationalAutoencoder,
            ReplayBuffer,
            DeepQNetwork,
            DQNAgent,
            ModelPruner,
            ModelQuantizer,
            ModelTrainer,
            DLAlgorithmLibrary
        )
        
        print("✓ 所有18个类导入成功！")
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ 发生未知错误: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试基本功能")
    print("=" * 60)
    
    try:
        from U.U2 import BaseNeuralNetwork, DLAlgorithmLibrary
        
        # 测试基础神经网络创建
        model = BaseNeuralNetwork(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=5,
            activation='relu',
            dropout_rate=0.1
        )
        print(f"✓ BaseNeuralNetwork创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试前向传播
        import torch
        x = torch.randn(5, 10)
        output = model(x)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
        # 测试主库类
        dl_lib = DLAlgorithmLibrary()
        print(f"✓ DLAlgorithmLibrary初始化成功，使用设备: {dl_lib.device}")
        
        # 测试通过库创建模型
        model2 = dl_lib.create_base_network(
            name="test_model",
            input_dim=20,
            hidden_dims=[128, 64],
            output_dim=10
        )
        print(f"✓ 通过库创建模型成功，名称: test_model")
        
        # 测试模型信息获取
        info = dl_lib.get_model_info("test_model")
        print(f"✓ 模型信息获取成功: {info['model_name']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        traceback.print_exc()
        return False

def test_cnn_functionality():
    """测试CNN功能"""
    print("\n" + "=" * 60)
    print("测试CNN功能")
    print("=" * 60)
    
    try:
        from U.U2 import ConvolutionalNeuralNetwork
        
        # 创建CNN
        conv_configs = [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
        
        model = ConvolutionalNeuralNetwork(
            input_channels=1,
            conv_configs=conv_configs,
            fc_dims=[128, 64],
            output_dim=10,
            pool_type='max',
            dropout_rate=0.2
        )
        
        print(f"✓ CNN创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试前向传播
        import torch
        x = torch.randn(4, 1, 28, 28)  # batch_size=4, channels=1, height=28, width=28
        output = model(x)
        print(f"✓ CNN前向传播成功，输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ CNN测试失败: {e}")
        traceback.print_exc()
        return False

def test_autoencoder_functionality():
    """测试自编码器功能"""
    print("\n" + "=" * 60)
    print("测试自编码器功能")
    print("=" * 60)
    
    try:
        from U.U2 import Autoencoder, VariationalAutoencoder
        
        # 测试标准自编码器
        autoencoder = Autoencoder(input_dim=784, latent_dim=32, hidden_dims=[512, 256, 128])
        print(f"✓ Autoencoder创建成功，参数数量: {sum(p.numel() for p in autoencoder.parameters())}")
        
        # 测试前向传播
        import torch
        x = torch.randn(5, 784)
        reconstructed, latent = autoencoder(x)
        print(f"✓ Autoencoder前向传播成功，重构形状: {reconstructed.shape}, 潜在表示形状: {latent.shape}")
        
        # 测试变分自编码器
        vae = VariationalAutoencoder(input_dim=784, latent_dim=32, hidden_dims=[512, 256, 128])
        print(f"✓ VAE创建成功，参数数量: {sum(p.numel() for p in vae.parameters())}")
        
        # 测试VAE前向传播
        reconstructed, mu, logvar = vae(x)
        print(f"✓ VAE前向传播成功，重构形状: {reconstructed.shape}, 均值形状: {mu.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 自编码器测试失败: {e}")
        traceback.print_exc()
        return False

def test_gan_functionality():
    """测试GAN功能"""
    print("\n" + "=" * 60)
    print("测试GAN功能")
    print("=" * 60)
    
    try:
        from U.U2 import GAN
        
        # 创建GAN
        gan = GAN(latent_dim=100, hidden_dims=[256, 512], output_dim=784)
        print(f"✓ GAN创建成功，参数数量: {sum(p.numel() for p in gan.parameters())}")
        
        # 测试生成器
        import torch
        z = torch.randn(5, 100)
        generated = gan.generate(z)
        print(f"✓ GAN生成器测试成功，生成样本形状: {generated.shape}")
        
        # 测试判别器
        real_samples = torch.randn(5, 784)
        fake_samples = generated
        real_scores = gan.discriminate(real_samples)
        fake_scores = gan.discriminate(fake_samples)
        print(f"✓ GAN判别器测试成功，真实样本分数: {real_scores.mean().item():.4f}")
        print(f"  生成样本分数: {fake_scores.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ GAN测试失败: {e}")
        traceback.print_exc()
        return False

def test_dqn_functionality():
    """测试DQN功能"""
    print("\n" + "=" * 60)
    print("测试DQN功能")
    print("=" * 60)
    
    try:
        from U.U2 import DQNAgent
        
        # 创建DQN智能体
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dims=[64, 32],
            gamma=0.99,
            epsilon=1.0,
            batch_size=32
        )
        print(f"✓ DQN智能体创建成功")
        
        # 测试动作选择
        import numpy as np
        state = np.random.random(4)
        action = agent.select_action(state, training=False)
        print(f"✓ 动作选择成功，选择动作: {action}")
        
        # 测试训练（添加一些经验）
        for _ in range(10):
            state = np.random.random(4)
            action = np.random.randint(0, 2)
            reward = np.random.random()
            next_state = np.random.random(4)
            done = np.random.random() > 0.8
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # 训练一次
        loss = agent.train()
        print(f"✓ DQN训练成功，损失: {loss:.4f}")
        
        # 获取智能体信息
        info = agent.get_model_info()
        print(f"✓ DQN信息获取成功: {info['agent_type']}")
        
        return True
        
    except Exception as e:
        print(f"✗ DQN测试失败: {e}")
        traceback.print_exc()
        return False

def test_module_info():
    """测试模块信息函数"""
    print("\n" + "=" * 60)
    print("测试模块信息函数")
    print("=" * 60)
    
    try:
        from U.U2 import get_library_info, list_available_models
        
        # 测试库信息获取
        info = get_library_info()
        print(f"✓ 库信息获取成功:")
        print(f"  名称: {info['name']}")
        print(f"  版本: {info['version']}")
        print(f"  组件数量: {info['classes_count']}")
        
        # 测试模型列表获取
        models = list_available_models()
        print(f"✓ 可用模型列表获取成功:")
        for category, model_list in models.items():
            print(f"  {category}: {len(model_list)}个组件")
        
        return True
        
    except Exception as e:
        print(f"✗ 模块信息测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始U2模块导出接口测试\n")
    
    # 执行所有测试
    tests = [
        ("导入测试", test_imports),
        ("基本功能测试", test_basic_functionality),
        ("CNN功能测试", test_cnn_functionality),
        ("自编码器功能测试", test_autoencoder_functionality),
        ("GAN功能测试", test_gan_functionality),
        ("DQN功能测试", test_dqn_functionality),
        ("模块信息测试", test_module_info)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 通过")
            else:
                failed += 1
                print(f"✗ {test_name} 失败")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} 异常: {e}")
    
    # 测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"总测试数: {len(tests)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"成功率: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！U2模块导出接口创建成功！")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查相关功能。")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)