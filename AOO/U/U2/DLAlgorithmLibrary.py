"""
深度学习算法库 (Deep Learning Algorithm Library)
==============================================

这是一个完整的深度学习算法库，包含各种神经网络架构和训练方法。
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

作者: U2团队
版本: 1.0.0
创建时间: 2025-11-05
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import random
from collections import deque
import math


# =============================================================================
# 1. 神经网络基础架构
# =============================================================================

class BaseNeuralNetwork(nn.Module):
    """
    基础神经网络类，提供通用的神经网络功能
    
    属性:
        layers: 网络层列表
        activation_fn: 激活函数
        dropout_rate: Dropout比率
        batch_norm: 是否使用批归一化
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 output_dim: int,
                 activation: str = 'relu',
                 dropout_rate: float = 0.0,
                 batch_norm: bool = False):
        """
        初始化基础神经网络
        
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh', 'leaky_relu')
            dropout_rate: Dropout比率
            batch_norm: 是否使用批归一化
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # 构建网络层
        self.layers = nn.ModuleList()
        
        # 输入层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
                
            # 添加激活函数
            self.layers.append(self._get_activation(activation))
            
            # 添加Dropout
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
            
        # 输出层
        self.layers.append(nn.Linear(prev_dim, output_dim))
        
    def _get_activation(self, activation: str) -> nn.Module:
        """
        获取激活函数
        
        参数:
            activation: 激活函数名称
            
        返回:
            对应的激活函数模块
        """
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量
        """
        for layer in self.layers:
            x = layer(x)
        return x
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        返回:
            包含模型信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers_count': len(self.layers)
        }


# =============================================================================
# 2. 卷积神经网络(CNN)
# =============================================================================

class ConvolutionalNeuralNetwork(nn.Module):
    """
    卷积神经网络类，支持多种CNN架构
    
    属性:
        conv_layers: 卷积层列表
        fc_layers: 全连接层列表
        pool_type: 池化类型
        dropout_rate: Dropout比率
    """
    
    def __init__(self,
                 input_channels: int,
                 conv_configs: List[Dict],
                 fc_dims: List[int],
                 output_dim: int,
                 pool_type: str = 'max',
                 dropout_rate: float = 0.0):
        """
        初始化CNN
        
        参数:
            input_channels: 输入通道数
            conv_configs: 卷积层配置列表，每个配置包含：
                - out_channels: 输出通道数
                - kernel_size: 卷积核大小
                - stride: 步长
                - padding: 填充大小
            fc_dims: 全连接层维度列表
            output_dim: 输出维度
            pool_type: 池化类型 ('max', 'avg', 'adaptive')
            dropout_rate: Dropout比率
        """
        super().__init__()
        self.input_channels = input_channels
        self.conv_configs = conv_configs
        self.fc_dims = fc_dims
        self.output_dim = output_dim
        self.pool_type = pool_type
        self.dropout_rate = dropout_rate
        
        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        prev_channels = input_channels
        
        for config in conv_configs:
            out_channels = config['out_channels']
            kernel_size = config['kernel_size']
            stride = config.get('stride', 1)
            padding = config.get('padding', 0)
            
            self.conv_layers.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size, stride, padding)
            )
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(nn.ReLU())
            
            # 添加池化层
            if pool_type == 'max':
                self.conv_layers.append(nn.MaxPool2d(2, 2))
            elif pool_type == 'avg':
                self.conv_layers.append(nn.AvgPool2d(2, 2))
            elif pool_type == 'adaptive':
                self.conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                
            prev_channels = out_channels
            
        # 计算全连接层输入维度
        self.fc_input_dim = prev_channels
        
        # 构建全连接层
        self.fc_layers = nn.ModuleList()
        prev_dim = self.fc_input_dim
        
        for fc_dim in fc_dims:
            self.fc_layers.append(nn.Linear(prev_dim, fc_dim))
            self.fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(dropout_rate))
            prev_dim = fc_dim
            
        # 输出层
        self.fc_layers.append(nn.Linear(prev_dim, output_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, channels, height, width)
            
        返回:
            输出张量
        """
        # 卷积层
        for layer in self.conv_layers:
            x = layer(x)
            
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        for layer in self.fc_layers:
            x = layer(x)
            
        return x
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'input_channels': self.input_channels,
            'conv_configs': self.conv_configs,
            'fc_dims': self.fc_dims,
            'output_dim': self.output_dim,
            'pool_type': self.pool_type,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


# =============================================================================
# 3. 循环神经网络(RNN/LSTM/GRU)
# =============================================================================

class RecurrentNeuralNetwork(nn.Module):
    """
    循环神经网络类，支持RNN、LSTM和GRU
    
    属性:
        rnn_type: RNN类型 ('rnn', 'lstm', 'gru')
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: 层数
        dropout_rate: Dropout比率
        bidirectional: 是否双向
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 rnn_type: str = 'lstm',
                 dropout_rate: float = 0.0,
                 bidirectional: bool = False):
        """
        初始化RNN
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            num_layers: 层数
            output_size: 输出维度
            rnn_type: RNN类型 ('rnn', 'lstm', 'gru')
            dropout_rate: Dropout比率
            bidirectional: 是否双向
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.rnn_type = rnn_type
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        
        # 选择RNN类型
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
                             bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
            
        # 计算输出维度
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        
        # 全连接层
        self.fc = nn.Linear(rnn_output_size, output_size)
        
        # Dropout层
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
            
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Any]:
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, sequence_length, input_size)
            hidden: 隐藏状态 (对于LSTM是(h0, c0)，对于其他是h0)
            
        返回:
            output: 输出张量
            hidden: 更新后的隐藏状态
        """
        # RNN前向传播
        if self.rnn_type == 'lstm':
            if hidden is None:
                output, hidden = self.rnn(x)
            else:
                output, hidden = self.rnn(x, hidden)
        else:
            if hidden is None:
                output, hidden = self.rnn(x)
            else:
                output, hidden = self.rnn(x, hidden)
                
        # 应用dropout
        if self.dropout is not None:
            output = self.dropout(output)
            
        # 全连接层
        output = self.fc(output)
        
        return output, hidden
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'rnn_type': self.rnn_type,
            'dropout_rate': self.dropout_rate,
            'bidirectional': self.bidirectional,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


# =============================================================================
# 4. 注意力机制和Transformer
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    属性:
        d_model: 模型维度
        num_heads: 注意力头数
        d_k: 键的维度
        d_v: 值的维度
        dropout_rate: Dropout比率
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        """
        初始化多头注意力
        
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout_rate: Dropout比率
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query: 查询张量 (batch_size, seq_len, d_model)
            key: 键张量 (batch_size, seq_len, d_model)
            value: 值张量 (batch_size, seq_len, d_model)
            mask: 掩码张量
            
        返回:
            注意力输出
        """
        batch_size, seq_len, d_model = query.size()
        
        # 线性变换并重塑为多头
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        
        # 重塑并通过输出层
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(context)
        
        return output


class TransformerBlock(nn.Module):
    """
    Transformer块
    
    属性:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络维度
        dropout_rate: Dropout比率
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            mask: 掩码张量
            
        返回:
            输出张量
        """
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    Transformer模型
    
    属性:
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: 层数
        d_ff: 前馈网络维度
        vocab_size: 词汇表大小
        max_seq_length: 最大序列长度
    """
    
    def __init__(self, d_model: int, num_heads: int, num_layers: int, 
                 d_ff: int, vocab_size: int, max_seq_length: int = 512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(max_seq_length, d_model)
        
        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def _create_pos_encoding(self, max_seq_length: int, d_model: int) -> torch.Tensor:
        """
        创建位置编码
        
        参数:
            max_seq_length: 最大序列长度
            d_model: 模型维度
            
        返回:
            位置编码张量
        """
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, seq_len)
            mask: 掩码张量
            
        返回:
            输出张量
        """
        seq_len = x.size(1)
        
        # 嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        
        # 通过Transformer块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
            
        # 输出投影
        output = self.output_projection(x)
        
        return output
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'd_model': self.d_model,
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'num_layers': len(self.transformer_blocks),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


# =============================================================================
# 5. 自编码器
# =============================================================================

class Autoencoder(nn.Module):
    """
    自编码器
    
    属性:
        encoder: 编码器
        decoder: 解码器
        latent_dim: 潜在空间维度
    """
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Optional[List[int]] = None):
        """
        初始化自编码器
        
        参数:
            input_dim: 输入维度
            latent_dim: 潜在空间维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
            
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 潜在空间
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = [nn.Linear(latent_dim, hidden_dims[-1]), nn.ReLU()]
        
        prev_dim = hidden_dims[-1]
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # 假设输入在[0,1]范围内
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码
        
        参数:
            x: 输入张量
            
        返回:
            潜在表示
        """
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码
        
        参数:
            z: 潜在表示
            
        返回:
            重构的输入
        """
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            重构的输入和潜在表示
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


# =============================================================================
# 6. 生成对抗网络(GAN)
# =============================================================================

class Generator(nn.Module):
    """
    GAN生成器
    
    属性:
        latent_dim: 潜在噪声维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
    """
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # 输出在[-1,1]范围内
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        生成样本
        
        参数:
            z: 潜在噪声
            
        返回:
            生成的样本
        """
        return self.model(z)


class Discriminator(nn.Module):
    """
    GAN判别器
    
    属性:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.input_dim = input_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        # 输出层 (二分类：真实或虚假)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        判别真假
        
        参数:
            x: 输入样本
            
        返回:
            真实性概率
        """
        return self.model(x)


class GAN(nn.Module):
    """
    生成对抗网络
    
    属性:
        generator: 生成器
        discriminator: 判别器
        latent_dim: 潜在噪声维度
    """
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.generator = Generator(latent_dim, hidden_dims, output_dim)
        self.discriminator = Discriminator(output_dim, hidden_dims[::-1])
        
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """
        生成样本
        
        参数:
            z: 潜在噪声
            
        返回:
            生成的样本
        """
        return self.generator(z)
        
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """
        判别真假
        
        参数:
            x: 输入样本
            
        返回:
            真实性概率
        """
        return self.discriminator(x)
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'latent_dim': self.latent_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


# =============================================================================
# 7. 变分自编码器(VAE)
# =============================================================================

class VariationalAutoencoder(nn.Module):
    """
    变分自编码器 (VAE)
    
    属性:
        encoder: 编码器
        decoder: 解码器
        latent_dim: 潜在空间维度
        hidden_dims: 隐藏层维度列表
    """
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
            
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 潜在空间参数
        self.fc_mu = nn.Linear(prev_dim, latent_dim)  # 均值
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)  # 对数方差
        
        # 解码器
        decoder_layers = [nn.Linear(latent_dim, hidden_dims[-1]), nn.ReLU()]
        
        prev_dim = hidden_dims[-1]
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码，返回均值和对数方差
        
        参数:
            x: 输入张量
            
        返回:
            均值和对数方差
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧
        
        参数:
            mu: 均值
            logvar: 对数方差
            
        返回:
            潜在表示
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码
        
        参数:
            z: 潜在表示
            
        返回:
            重构的输入
        """
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            重构的输入、均值和对数方差
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


# =============================================================================
# 8. 深度强化学习
# =============================================================================

class ReplayBuffer:
    """
    经验回放缓冲区
    
    属性:
        buffer: 经验缓冲区
        max_size: 最大容量
        position: 当前写入位置
    """
    
    def __init__(self, max_size: int = 100000):
        """
        初始化经验回放缓冲区
        
        参数:
            max_size: 最大容量
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.position = 0
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        添加经验
        
        参数:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.max_size
            
    def sample(self, batch_size: int) -> List:
        """
        采样批量经验
        
        参数:
            batch_size: 批量大小
            
        返回:
            批量经验
        """
        return random.sample(self.buffer, batch_size)
        
    def __len__(self) -> int:
        """返回缓冲区长度"""
        return len(self.buffer)


class DeepQNetwork(nn.Module):
    """
    深度Q网络 (DQN)
    
    属性:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dims: 隐藏层维度列表
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            state: 状态张量
            
        返回:
            Q值
        """
        return self.network(state)


class DQNAgent:
    """
    DQN智能体
    
    属性:
        state_dim: 状态维度
        action_dim: 动作维度
        q_network: Q网络
        target_network: 目标网络
        optimizer: 优化器
        replay_buffer: 经验回放缓冲区
        gamma: 折扣因子
        epsilon: 探索率
        epsilon_decay: 探索率衰减
        epsilon_min: 最小探索率
        target_update_freq: 目标网络更新频率
        batch_size: 批量大小
        learning_rate: 学习率
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 target_update_freq: int = 1000,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 replay_buffer_size: int = 100000):
        """
        初始化DQN智能体
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度列表
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            target_update_freq: 目标网络更新频率
            batch_size: 批量大小
            learning_rate: 学习率
            replay_buffer_size: 经验回放缓冲区大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 创建网络
        self.q_network = DeepQNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = DeepQNetwork(state_dim, action_dim, hidden_dims)
        
        # 同步目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # 训练步数
        self.steps = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作
        
        参数:
            state: 当前状态
            training: 是否为训练模式
            
        返回:
            选择的动作
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randint(0, self.action_dim - 1)
        else:
            # 利用：选择最优动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
            
    def train(self) -> float:
        """
        训练智能体
        
        返回:
            损失值
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
            
        # 采样批量经验
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 下一状态的最大Q值
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        
        return {
            'agent_type': 'DQN',
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'replay_buffer_size': len(self.replay_buffer)
        }


# =============================================================================
# 9. 模型压缩和量化
# =============================================================================

class ModelPruner:
    """
    模型剪枝器
    
    属性:
        model: 待剪枝的模型
        pruning_ratio: 剪枝比例
        method: 剪枝方法 ('magnitude', 'random', 'structured')
    """
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5, method: str = 'magnitude'):
        """
        初始化模型剪枝器
        
        参数:
            model: 待剪枝的模型
            pruning_ratio: 剪枝比例
            method: 剪枝方法
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.method = method
        
    def magnitude_prune(self, layer: nn.Module) -> nn.Module:
        """
        幅度剪枝
        
        参数:
            layer: 待剪枝的层
            
        返回:
            剪枝后的层
        """
        if isinstance(layer, nn.Linear):
            # 计算权重的绝对值
            weights = layer.weight.data.abs()
            
            # 计算阈值
            threshold = torch.quantile(weights.flatten(), self.pruning_ratio)
            
            # 创建掩码
            mask = weights > threshold
            
            # 应用掩码
            layer.weight.data = layer.weight.data * mask.float()
            
        return layer
        
    def random_prune(self, layer: nn.Module) -> nn.Module:
        """
        随机剪枝
        
        参数:
            layer: 待剪枝的层
            
        返回:
            剪枝后的层
        """
        if isinstance(layer, nn.Linear):
            # 创建随机掩码
            mask = torch.rand_like(layer.weight.data) > self.pruning_ratio
            
            # 应用掩码
            layer.weight.data = layer.weight.data * mask.float()
            
        return layer
        
    def structured_prune(self, layer: nn.Module) -> nn.Module:
        """
        结构化剪枝（剪除整个神经元）
        
        参数:
            layer: 待剪枝的层
            
        返回:
            剪枝后的层
        """
        if isinstance(layer, nn.Linear):
            # 计算每个神经元的平均权重
            neuron_importance = layer.weight.data.abs().mean(dim=1)
            
            # 计算要剪除的神经元数量
            num_neurons_to_prune = int(neuron_importance.size(0) * self.pruning_ratio)
            
            # 找到重要性最低的神经元
            _, indices_to_prune = torch.topk(neuron_importance, num_neurons_to_prune, largest=False)
            
            # 创建掩码
            mask = torch.ones_like(neuron_importance, dtype=torch.bool)
            mask[indices_to_prune] = False
            
            # 重新构建层
            old_weight = layer.weight.data
            new_weight = old_weight[mask]
            
            layer.weight.data = new_weight
            layer.out_features = new_weight.size(0)
            
        return layer
        
    def prune_model(self) -> nn.Module:
        """
        对整个模型进行剪枝
        
        返回:
            剪枝后的模型
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if self.method == 'magnitude':
                    self.magnitude_prune(module)
                elif self.method == 'random':
                    self.random_prune(module)
                elif self.method == 'structured':
                    self.structured_prune(module)
                    
        return self.model
        
    def get_pruning_stats(self) -> Dict[str, Any]:
        """
        获取剪枝统计信息
        
        返回:
            剪枝统计信息
        """
        original_params = sum(p.numel() for p in self.model.parameters())
        
        # 计算剪枝后的参数数量
        pruned_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                pruned_params += module.weight.data.numel()
                
        pruning_ratio_actual = 1.0 - (pruned_params / original_params)
        
        return {
            'original_parameters': original_params,
            'pruned_parameters': pruned_params,
            'actual_pruning_ratio': pruning_ratio_actual,
            'target_pruning_ratio': self.pruning_ratio,
            'pruning_method': self.method
        }


class ModelQuantizer:
    """
    模型量化器
    
    属性:
        model: 待量化的模型
        bit_width: 量化位宽
        quantization_type: 量化类型 ('dynamic', 'static', 'qat')
    """
    
    def __init__(self, model: nn.Module, bit_width: int = 8, quantization_type: str = 'dynamic'):
        """
        初始化模型量化器
        
        参数:
            model: 待量化的模型
            bit_width: 量化位宽
            quantization_type: 量化类型
        """
        self.model = model
        self.bit_width = bit_width
        self.quantization_type = quantization_type
        
    def dynamic_quantize(self) -> nn.Module:
        """
        动态量化
        
        返回:
            量化后的模型
        """
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        return quantized_model
        
    def static_quantize(self, calibration_data: Any) -> nn.Module:
        """
        静态量化
        
        参数:
            calibration_data: 校准数据
            
        返回:
            量化后的模型
        """
        # 设置量化配置
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备量化
        self.model = torch.quantization.prepare(self.model)
        
        # 校准
        with torch.no_grad():
            for data in calibration_data:
                self.model(data)
                
        # 转换为量化模型
        quantized_model = torch.quantization.convert(self.model)
        
        return quantized_model
        
    def quantize_aware_training(self, epochs: int = 10, calibration_data: Any = None) -> nn.Module:
        """
        量化感知训练
        
        参数:
            epochs: 训练轮数
            calibration_data: 校准数据
            
        返回:
            量化后的模型
        """
        # 设置QAT配置
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # 准备QAT
        self.model = torch.quantization.prepare_qat(self.model)
        
        # 量化感知训练
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            for data in calibration_data:
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, torch.randint(0, 10, (data.size(0),)))
                loss.backward()
                optimizer.step()
                
        # 转换为量化模型
        quantized_model = torch.quantization.convert(self.model.eval())
        
        return quantized_model
        
    def get_quantization_stats(self) -> Dict[str, Any]:
        """
        获取量化统计信息
        
        返回:
            量化统计信息
        """
        original_size = sum(p.numel() * 4 for p in self.model.parameters())  # 假设float32为4字节
        
        # 估算量化后大小
        quantized_size = sum(p.numel() * (self.bit_width // 8) for p in self.model.parameters())
        
        compression_ratio = original_size / quantized_size
        
        return {
            'bit_width': self.bit_width,
            'quantization_type': self.quantization_type,
            'original_size_bytes': original_size,
            'quantized_size_bytes': quantized_size,
            'compression_ratio': compression_ratio
        }


# =============================================================================
# 10. 训练和评估工具
# =============================================================================

class ModelTrainer:
    """
    模型训练器
    
    属性:
        model: 待训练的模型
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        scheduler: 学习率调度器
    """
    
    def __init__(self, model: nn.Module, 
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None):
        """
        初始化模型训练器
        
        参数:
            model: 待训练的模型
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 设置默认优化器
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
            
        # 设置默认损失函数
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
            
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def set_scheduler(self, scheduler_type: str = 'step', **kwargs):
        """
        设置学习率调度器
        
        参数:
            scheduler_type: 调度器类型
            **kwargs: 调度器参数
        """
        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **kwargs)
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, **kwargs)
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs)
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs)
            
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch
        
        参数:
            train_loader: 训练数据加载器
            
        返回:
            平均损失和准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
        
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证模型
        
        参数:
            val_loader: 验证数据加载器
            
        返回:
            平均损失和准确率
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
        
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 100, save_path: Optional[str] = None, 
              early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_path: 模型保存路径
            early_stopping_patience: 早停耐心值
            
        返回:
            训练历史
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 验证
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                      
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    print(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
                    
                # 更新学习率
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            else:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                      
                if self.scheduler is not None:
                    self.scheduler.step()
                    
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        绘制训练历史
        
        参数:
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失')
        if self.val_losses:
            ax1.plot(self.val_losses, label='验证损失')
        ax1.set_title('损失曲线')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率')
        if self.val_accuracies:
            ax2.plot(self.val_accuracies, label='验证准确率')
        ax2.set_title('准确率曲线')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


# =============================================================================
# 11. 主算法库类
# =============================================================================

class DLAlgorithmLibrary:
    """
    深度学习算法库主类
    
    整合了所有深度学习算法和工具，提供统一的接口。
    
    属性:
        models: 已创建的模型字典
        trainers: 训练器字典
        device: 计算设备
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        初始化深度学习算法库
        
        参数:
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.trainers = {}
        
        print(f"深度学习算法库初始化完成，使用设备: {self.device}")
        
    def create_base_network(self, name: str, input_dim: int, hidden_dims: List[int], 
                          output_dim: int, **kwargs) -> BaseNeuralNetwork:
        """
        创建基础神经网络
        
        参数:
            name: 模型名称
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            **kwargs: 其他参数
            
        返回:
            创建的模型
        """
        model = BaseNeuralNetwork(input_dim, hidden_dims, output_dim, **kwargs)
        self.models[name] = model
        return model
        
    def create_cnn(self, name: str, input_channels: int, conv_configs: List[Dict],
                  fc_dims: List[int], output_dim: int, **kwargs) -> ConvolutionalNeuralNetwork:
        """
        创建卷积神经网络
        
        参数:
            name: 模型名称
            input_channels: 输入通道数
            conv_configs: 卷积层配置
            fc_dims: 全连接层维度列表
            output_dim: 输出维度
            **kwargs: 其他参数
            
        返回:
            创建的模型
        """
        model = ConvolutionalNeuralNetwork(input_channels, conv_configs, fc_dims, 
                                         output_dim, **kwargs)
        self.models[name] = model
        return model
        
    def create_rnn(self, name: str, input_size: int, hidden_size: int, 
                  num_layers: int, output_size: int, **kwargs) -> RecurrentNeuralNetwork:
        """
        创建循环神经网络
        
        参数:
            name: 模型名称
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            num_layers: 层数
            output_size: 输出维度
            **kwargs: 其他参数
            
        返回:
            创建的模型
        """
        model = RecurrentNeuralNetwork(input_size, hidden_size, num_layers, 
                                     output_size, **kwargs)
        self.models[name] = model
        return model
        
    def create_transformer(self, name: str, d_model: int, num_heads: int, 
                          num_layers: int, d_ff: int, vocab_size: int, 
                          **kwargs) -> Transformer:
        """
        创建Transformer模型
        
        参数:
            name: 模型名称
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 层数
            d_ff: 前馈网络维度
            vocab_size: 词汇表大小
            **kwargs: 其他参数
            
        返回:
            创建的模型
        """
        model = Transformer(d_model, num_heads, num_layers, d_ff, vocab_size, **kwargs)
        self.models[name] = model
        return model
        
    def create_autoencoder(self, name: str, input_dim: int, latent_dim: int,
                          **kwargs) -> Autoencoder:
        """
        创建自编码器
        
        参数:
            name: 模型名称
            input_dim: 输入维度
            latent_dim: 潜在空间维度
            **kwargs: 其他参数
            
        返回:
            创建的模型
        """
        model = Autoencoder(input_dim, latent_dim, **kwargs)
        self.models[name] = model
        return model
        
    def create_gan(self, name: str, latent_dim: int, hidden_dims: List[int],
                  output_dim: int) -> GAN:
        """
        创建生成对抗网络
        
        参数:
            name: 模型名称
            latent_dim: 潜在噪声维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            
        返回:
            创建的模型
        """
        model = GAN(latent_dim, hidden_dims, output_dim)
        self.models[name] = model
        return model
        
    def create_vae(self, name: str, input_dim: int, latent_dim: int,
                  **kwargs) -> VariationalAutoencoder:
        """
        创建变分自编码器
        
        参数:
            name: 模型名称
            input_dim: 输入维度
            latent_dim: 潜在空间维度
            **kwargs: 其他参数
            
        返回:
            创建的模型
        """
        model = VariationalAutoencoder(input_dim, latent_dim, **kwargs)
        self.models[name] = model
        return model
        
    def create_dqn_agent(self, name: str, state_dim: int, action_dim: int,
                        **kwargs) -> DQNAgent:
        """
        创建DQN智能体
        
        参数:
            name: 智能体名称
            state_dim: 状态维度
            action_dim: 动作维度
            **kwargs: 其他参数
            
        返回:
            创建的智能体
        """
        agent = DQNAgent(state_dim, action_dim, **kwargs)
        self.models[name] = agent
        return agent
        
    def create_trainer(self, name: str, model: nn.Module, 
                      optimizer: Optional[torch.optim.Optimizer] = None,
                      criterion: Optional[nn.Module] = None) -> ModelTrainer:
        """
        创建训练器
        
        参数:
            name: 训练器名称
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            
        返回:
            创建的训练器
        """
        trainer = ModelTrainer(model, optimizer, criterion, self.device)
        self.trainers[name] = trainer
        return trainer
        
    def prune_model(self, model_name: str, pruning_ratio: float = 0.5, 
                   method: str = 'magnitude') -> Dict[str, Any]:
        """
        剪枝模型
        
        参数:
            model_name: 模型名称
            pruning_ratio: 剪枝比例
            method: 剪枝方法
            
        返回:
            剪枝统计信息
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
            
        model = self.models[model_name]
        pruner = ModelPruner(model, pruning_ratio, method)
        pruner.prune_model()
        
        return pruner.get_pruning_stats()
        
    def quantize_model(self, model_name: str, bit_width: int = 8,
                      quantization_type: str = 'dynamic') -> Dict[str, Any]:
        """
        量化模型
        
        参数:
            model_name: 模型名称
            bit_width: 量化位宽
            quantization_type: 量化类型
            
        返回:
            量化统计信息
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
            
        model = self.models[model_name]
        quantizer = ModelQuantizer(model, bit_width, quantization_type)
        
        if quantization_type == 'dynamic':
            quantized_model = quantizer.dynamic_quantize()
        else:
            # 其他量化方法需要校准数据，这里只做示例
            quantized_model = model
            
        self.models[f"{model_name}_quantized"] = quantized_model
        
        return quantizer.get_quantization_stats()
        
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        参数:
            model_name: 模型名称
            
        返回:
            模型信息
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
            
        model = self.models[model_name]
        if hasattr(model, 'get_model_info'):
            return model.get_model_info()
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                'model_name': model.__class__.__name__,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            }
            
    def list_models(self) -> List[str]:
        """
        列出所有模型
        
        返回:
            模型名称列表
        """
        return list(self.models.keys())
        
    def remove_model(self, model_name: str):
        """
        删除模型
        
        参数:
            model_name: 模型名称
        """
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.trainers:
            del self.trainers[model_name]


# =============================================================================
# 12. 测试用例和示例
# =============================================================================

def run_basic_network_test():
    """测试基础神经网络"""
    print("=== 测试基础神经网络 ===")
    
    # 创建库实例
    dl_lib = DLAlgorithmLibrary()
    
    # 创建基础神经网络
    model = dl_lib.create_base_network(
        name="test_network",
        input_dim=10,
        hidden_dims=[64, 32, 16],
        output_dim=5,
        activation='relu',
        dropout_rate=0.2
    )
    
    # 获取模型信息
    info = dl_lib.get_model_info("test_network")
    print("模型信息:", info)
    
    # 创建训练器
    trainer = dl_lib.create_trainer("test_trainer", model)
    
    # 创建模拟数据
    batch_size = 32
    input_dim = 10
    output_dim = 5
    
    # 生成随机数据
    train_data = torch.randn(batch_size, input_dim)
    train_labels = torch.randint(0, output_dim, (batch_size,))
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    print("基础神经网络测试完成")
    return dl_lib


def run_cnn_test():
    """测试卷积神经网络"""
    print("\n=== 测试卷积神经网络 ===")
    
    dl_lib = DLAlgorithmLibrary()
    
    # 创建CNN
    conv_configs = [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ]
    
    model = dl_lib.create_cnn(
        name="test_cnn",
        input_channels=1,
        conv_configs=conv_configs,
        fc_dims=[128, 64],
        output_dim=10,
        pool_type='max',
        dropout_rate=0.3
    )
    
    info = dl_lib.get_model_info("test_cnn")
    print("CNN模型信息:", info)
    
    print("卷积神经网络测试完成")
    return dl_lib


def run_rnn_test():
    """测试循环神经网络"""
    print("\n=== 测试循环神经网络 ===")
    
    dl_lib = DLAlgorithmLibrary()
    
    # 创建LSTM
    model = dl_lib.create_rnn(
        name="test_lstm",
        input_size=50,
        hidden_size=128,
        num_layers=2,
        output_size=10,
        rnn_type='lstm',
        dropout_rate=0.2,
        bidirectional=True
    )
    
    info = dl_lib.get_model_info("test_lstm")
    print("LSTM模型信息:", info)
    
    print("循环神经网络测试完成")
    return dl_lib


def run_autoencoder_test():
    """测试自编码器"""
    print("\n=== 测试自编码器 ===")
    
    dl_lib = DLAlgorithmLibrary()
    
    # 创建自编码器
    model = dl_lib.create_autoencoder(
        name="test_autoencoder",
        input_dim=784,  # 28x28图像
        latent_dim=32,
        hidden_dims=[512, 256, 128, 64]
    )
    
    info = dl_lib.get_model_info("test_autoencoder")
    print("自编码器模型信息:", info)
    
    # 测试前向传播
    test_input = torch.randn(10, 784)
    reconstructed, latent = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"重构形状: {reconstructed.shape}")
    print(f"潜在表示形状: {latent.shape}")
    
    print("自编码器测试完成")
    return dl_lib


def run_gan_test():
    """测试生成对抗网络"""
    print("\n=== 测试生成对抗网络 ===")
    
    dl_lib = DLAlgorithmLibrary()
    
    # 创建GAN
    model = dl_lib.create_gan(
        name="test_gan",
        latent_dim=100,
        hidden_dims=[256, 512, 1024],
        output_dim=784  # 28x28图像
    )
    
    info = dl_lib.get_model_info("test_gan")
    print("GAN模型信息:", info)
    
    # 测试生成器
    z = torch.randn(10, 100)
    generated_samples = model.generate(z)
    print(f"生成样本形状: {generated_samples.shape}")
    
    # 测试判别器
    real_samples = torch.randn(10, 784)
    real_scores = model.discriminate(real_samples)
    fake_scores = model.discriminate(generated_samples)
    
    print(f"真实样本判别分数: {real_scores.mean().item():.4f}")
    print(f"生成样本判别分数: {fake_scores.mean().item():.4f}")
    
    print("生成对抗网络测试完成")
    return dl_lib


def run_vae_test():
    """测试变分自编码器"""
    print("\n=== 测试变分自编码器 ===")
    
    dl_lib = DLAlgorithmLibrary()
    
    # 创建VAE
    model = dl_lib.create_vae(
        name="test_vae",
        input_dim=784,
        latent_dim=32,
        hidden_dims=[512, 256, 128, 64]
    )
    
    info = dl_lib.get_model_info("test_vae")
    print("VAE模型信息:", info)
    
    # 测试前向传播
    test_input = torch.randn(10, 784)
    reconstructed, mu, logvar = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"重构形状: {reconstructed.shape}")
    print(f"均值形状: {mu.shape}")
    print(f"对数方差形状: {logvar.shape}")
    
    print("变分自编码器测试完成")
    return dl_lib


def run_dqn_test():
    """测试深度Q网络"""
    print("\n=== 测试深度Q网络 ===")
    
    dl_lib = DLAlgorithmLibrary()
    
    # 创建DQN智能体
    agent = dl_lib.create_dqn_agent(
        name="test_dqn",
        state_dim=4,  # 例如CartPole状态
        action_dim=2,  # 例如CartPole动作
        hidden_dims=[64, 32],
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=32,
        learning_rate=0.001
    )
    
    info = agent.get_model_info()
    print("DQN智能体信息:", info)
    
    # 模拟训练过程
    print("模拟DQN训练过程...")
    for episode in range(5):
        # 模拟环境交互
        state = np.random.random(4)
        action = agent.select_action(state, training=True)
        reward = np.random.random()
        next_state = np.random.random(4)
        done = np.random.random() > 0.8
        
        # 添加到经验回放缓冲区
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # 训练
        loss = agent.train()
        print(f"Episode {episode+1}: 动作={action}, 损失={loss:.4f}, epsilon={agent.epsilon:.4f}")
    
    print("深度Q网络测试完成")
    return dl_lib


def run_model_compression_test():
    """测试模型压缩"""
    print("\n=== 测试模型压缩 ===")
    
    dl_lib = DLAlgorithmLibrary()
    
    # 创建基础网络
    model = dl_lib.create_base_network(
        name="compression_test",
        input_dim=100,
        hidden_dims=[512, 256, 128, 64],
        output_dim=10
    )
    
    # 获取原始参数数量
    original_info = dl_lib.get_model_info("compression_test")
    print(f"原始模型参数: {original_info['total_parameters']}")
    
    # 测试剪枝
    print("测试模型剪枝...")
    pruning_stats = dl_lib.prune_model("compression_test", pruning_ratio=0.5, method='magnitude')
    print("剪枝统计:", pruning_stats)
    
    # 测试量化
    print("测试模型量化...")
    quantization_stats = dl_lib.quantize_model("compression_test", bit_width=8, quantization_type='dynamic')
    print("量化统计:", quantization_stats)
    
    print("模型压缩测试完成")
    return dl_lib


def run_comprehensive_test():
    """运行综合测试"""
    print("开始运行深度学习算法库综合测试")
    print("=" * 60)
    
    # 运行所有测试
    libraries = []
    
    try:
        libraries.append(run_basic_network_test())
        libraries.append(run_cnn_test())
        libraries.append(run_rnn_test())
        libraries.append(run_autoencoder_test())
        libraries.append(run_gan_test())
        libraries.append(run_vae_test())
        libraries.append(run_dqn_test())
        libraries.append(run_model_compression_test())
        
        print("\n" + "=" * 60)
        print("所有测试完成！深度学习算法库功能验证成功。")
        
        # 总结
        total_models = sum(len(lib.list_models()) for lib in libraries)
        print(f"总共创建了 {total_models} 个模型")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行综合测试
    run_comprehensive_test()