#!/usr/bin/env python3
"""
U2模块导出接口验证脚本（简化版）
=============================

验证导入结构是否正确，不依赖torch库的实际安装
"""

import sys
import os
import ast

def test_file_structure():
    """测试文件结构"""
    print("=" * 60)
    print("测试U2模块文件结构")
    print("=" * 60)
    
    # 检查文件是否存在
    u2_dir = os.path.dirname(os.path.abspath(__file__))
    init_file = os.path.join(u2_dir, "__init__.py")
    dl_file = os.path.join(u2_dir, "DLAlgorithmLibrary.py")
    
    checks = [
        ("__init__.py文件", os.path.exists(init_file)),
        ("DLAlgorithmLibrary.py文件", os.path.exists(dl_file))
    ]
    
    for name, exists in checks:
        if exists:
            print(f"✓ {name} 存在")
        else:
            print(f"✗ {name} 不存在")
    
    return all(exists for _, exists in checks)

def test_init_file_syntax():
    """测试__init__.py文件语法"""
    print("\n" + "=" * 60)
    print("测试__init__.py文件语法")
    print("=" * 60)
    
    init_file = os.path.join(os.path.dirname(__file__), "__init__.py")
    
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析AST
        ast.parse(content)
        print("✓ __init__.py文件语法正确")
        
        # 检查关键内容
        checks = [
            ('"U2模块 - 深度学习算法库"', '模块描述'),
            ('from .DLAlgorithmLibrary import', '导入语句'),
            ('BaseNeuralNetwork', 'BaseNeuralNetwork类'),
            ('ConvolutionalNeuralNetwork', 'ConvolutionalNeuralNetwork类'),
            ('RecurrentNeuralNetwork', 'RecurrentNeuralNetwork类'),
            ('MultiHeadAttention', 'MultiHeadAttention类'),
            ('TransformerBlock', 'TransformerBlock类'),
            ('Transformer', 'Transformer类'),
            ('Autoencoder', 'Autoencoder类'),
            ('Generator', 'Generator类'),
            ('Discriminator', 'Discriminator类'),
            ('GAN', 'GAN类'),
            ('VariationalAutoencoder', 'VariationalAutoencoder类'),
            ('ReplayBuffer', 'ReplayBuffer类'),
            ('DeepQNetwork', 'DeepQNetwork类'),
            ('DQNAgent', 'DQNAgent类'),
            ('ModelPruner', 'ModelPruner类'),
            ('ModelQuantizer', 'ModelQuantizer类'),
            ('ModelTrainer', 'ModelTrainer类'),
            ('DLAlgorithmLibrary', 'DLAlgorithmLibrary类'),
            ('__all__', '__all__定义')
        ]
        
        for check, desc in checks:
            if check in content:
                print(f"✓ {desc} 存在")
            else:
                print(f"✗ {desc} 缺失")
                return False
        
        return True
        
    except SyntaxError as e:
        print(f"✗ __init__.py文件语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 读取文件时发生错误: {e}")
        return False

def test_all_class_definitions():
    """测试所有类是否在源文件中定义"""
    print("\n" + "=" * 60)
    print("测试类定义")
    print("=" * 60)
    
    dl_file = os.path.join(os.path.dirname(__file__), "DLAlgorithmLibrary.py")
    
    # 需要检查的18个类
    expected_classes = [
        'BaseNeuralNetwork',
        'ConvolutionalNeuralNetwork', 
        'RecurrentNeuralNetwork',
        'MultiHeadAttention',
        'TransformerBlock',
        'Transformer',
        'Autoencoder',
        'Generator',
        'Discriminator',
        'GAN',
        'VariationalAutoencoder',
        'ReplayBuffer',
        'DeepQNetwork',
        'DQNAgent',
        'ModelPruner',
        'ModelQuantizer',
        'ModelTrainer',
        'DLAlgorithmLibrary'
    ]
    
    try:
        with open(dl_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        found_classes = []
        for class_name in expected_classes:
            if f"class {class_name}(" in content:
                found_classes.append(class_name)
                print(f"✓ {class_name} 类定义存在")
            else:
                print(f"✗ {class_name} 类定义缺失")
        
        if len(found_classes) == len(expected_classes):
            print(f"\n✓ 所有 {len(expected_classes)} 个类都已定义")
            return True
        else:
            print(f"\n✗ 只找到 {len(found_classes)}/{len(expected_classes)} 个类")
            return False
            
    except Exception as e:
        print(f"✗ 读取源文件时发生错误: {e}")
        return False

def test_import_structure():
    """测试导入结构"""
    print("\n" + "=" * 60)
    print("测试导入结构")
    print("=" * 60)
    
    init_file = os.path.join(os.path.dirname(__file__), "__init__.py")
    
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST来检查导入结构
        tree = ast.parse(content)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == '.DLAlgorithmLibrary':
                    for alias in node.names:
                        imports.append(alias.name)
                    print(f"✓ 找到从.DLAlgorithmLibrary的导入: {imports}")
        
        # 检查是否包含了所有预期的类
        expected_classes = [
            'BaseNeuralNetwork', 'ConvolutionalNeuralNetwork', 'RecurrentNeuralNetwork',
            'MultiHeadAttention', 'TransformerBlock', 'Transformer', 'Autoencoder',
            'Generator', 'Discriminator', 'GAN', 'VariationalAutoencoder', 'ReplayBuffer',
            'DeepQNetwork', 'DQNAgent', 'ModelPruner', 'ModelQuantizer', 'ModelTrainer',
            'DLAlgorithmLibrary'
        ]
        
        missing = []
        for class_name in expected_classes:
            if class_name not in imports:
                missing.append(class_name)
        
        if not missing:
            print("✓ 所有18个类都已正确导入")
            return True
        else:
            print(f"✗ 缺失导入: {missing}")
            return False
            
    except Exception as e:
        print(f"✗ 分析导入结构时发生错误: {e}")
        return False

def test_documentation():
    """测试文档"""
    print("\n" + "=" * 60)
    print("测试文档完整性")
    print("=" * 60)
    
    init_file = os.path.join(os.path.dirname(__file__), "__init__.py")
    
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('模块描述', 'U2模块 - 深度学习算法库'),
            ('功能说明', '主要功能'),
            ('使用示例', '使用示例'),
            ('版本信息', '__version__'),
            ('作者信息', '__author__'),
            ('获取库信息函数', 'get_library_info'),
            ('列出模型函数', 'list_available_models'),
            ('模块初始化', '_init_module')
        ]
        
        all_found = True
        for desc, content_check in checks:
            if content_check in content:
                print(f"✓ {desc} 存在")
            else:
                print(f"✗ {desc} 缺失")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ 检查文档时发生错误: {e}")
        return False

def test_export_list():
    """测试导出列表"""
    print("\n" + "=" * 60)
    print("测试导出列表")
    print("=" * 60)
    
    init_file = os.path.join(os.path.dirname(__file__), "__init__.py")
    
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST来检查__all__列表
        tree = ast.parse(content)
        
        all_list = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            all_list = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
        
        if all_list:
            print(f"✓ 找到__all__列表，包含 {len(all_list)} 个项目")
            
            # 检查是否包含了所有核心类
            expected_core_classes = [
                'BaseNeuralNetwork', 'ConvolutionalNeuralNetwork', 'RecurrentNeuralNetwork',
                'MultiHeadAttention', 'TransformerBlock', 'Transformer', 'Autoencoder',
                'Generator', 'Discriminator', 'GAN', 'VariationalAutoencoder', 'ReplayBuffer',
                'DeepQNetwork', 'DQNAgent', 'ModelPruner', 'ModelQuantizer', 'ModelTrainer',
                'DLAlgorithmLibrary'
            ]
            
            missing = []
            for class_name in expected_core_classes:
                if class_name not in all_list:
                    missing.append(class_name)
            
            if not missing:
                print("✓ __all__列表包含所有18个核心类")
                return True
            else:
                print(f"✗ __all__列表缺失: {missing}")
                return False
        else:
            print("✗ 未找到__all__列表")
            return False
            
    except Exception as e:
        print(f"✗ 检查__all__列表时发生错误: {e}")
        return False

def main():
    """主验证函数"""
    print("开始U2模块导出接口验证（简化版）\n")
    
    # 执行所有验证测试
    tests = [
        ("文件结构测试", test_file_structure),
        ("__init__.py语法测试", test_init_file_syntax),
        ("类定义测试", test_all_class_definitions),
        ("导入结构测试", test_import_structure),
        ("文档测试", test_documentation),
        ("导出列表测试", test_export_list)
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
    
    # 验证总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    print(f"总验证项目: {len(tests)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"成功率: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有验证通过！U2模块导出接口创建成功！")
        print("\n导出接口包含:")
        print("✓ 18个核心类的完整导入")
        print("✓ 详细的模块文档和使用说明") 
        print("✓ 适当的__all__导出列表")
        print("✓ 模块信息和便捷函数")
        print("✓ 完整的初始化提示信息")
    else:
        print(f"\n⚠️  有 {failed} 个验证失败，请检查相关功能。")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)