#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X1模块导出接口验证脚本
用于验证所有8个核心类和便捷函数是否正确导出
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有导入是否正常"""
    print("=" * 60)
    print("X1模块导出接口验证")
    print("=" * 60)
    
    try:
        import X.X1
        print("✓ 模块导入成功")
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False
    
    return True

def test_core_classes():
    """测试8个核心类"""
    print("\n" + "-" * 40)
    print("测试核心类 (8个主要类)")
    print("-" * 40)
    
    from X.X1 import (
        MemoryCacheManager, CacheEntry, CacheStatistics,
        EvictionStrategy, LRUEvictionStrategy, TTLEvictionStrategy,
        SizeEvictionStrategy, CacheConfigs
    )
    
    classes = {
        'MemoryCacheManager': MemoryCacheManager,
        'CacheEntry': CacheEntry,
        'CacheStatistics': CacheStatistics,
        'EvictionStrategy': EvictionStrategy,
        'LRUEvictionStrategy': LRUEvictionStrategy,
        'TTLEvictionStrategy': TTLEvictionStrategy,
        'SizeEvictionStrategy': SizeEvictionStrategy,
        'CacheConfigs': CacheConfigs
    }
    
    for name, cls in classes.items():
        try:
            if cls:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name} - 为None")
        except Exception as e:
            print(f"  ✗ {name} - 异常: {e}")
    
    return classes

def test_convenience_functions():
    """测试便捷函数"""
    print("\n" + "-" * 40)
    print("测试便捷函数")
    print("-" * 40)
    
    from X.X1 import (
        create_cache_manager, get_global_cache, set_global_cache,
        get_cache, clear_global_cache, quick_cache
    )
    
    functions = {
        'create_cache_manager': create_cache_manager,
        'get_global_cache': get_global_cache,
        'set_global_cache': set_global_cache,
        'get_cache': get_cache,
        'clear_global_cache': clear_global_cache,
        'quick_cache': quick_cache
    }
    
    for name, func in functions.items():
        try:
            if callable(func):
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name} - 不可调用")
        except Exception as e:
            print(f"  ✗ {name} - 异常: {e}")
    
    return functions

def test_cache_configs():
    """测试预定义配置"""
    print("\n" + "-" * 40)
    print("测试预定义配置")
    print("-" * 40)
    
    from X.X1 import CacheConfigs
    
    configs = ['SMALL', 'MEDIUM', 'LARGE', 'PERSISTENT', 'UNLIMITED']
    
    for config_name in configs:
        try:
            if hasattr(CacheConfigs, config_name):
                config = getattr(CacheConfigs, config_name)
                print(f"  ✓ {config_name}: {config.get('max_size', 'unlimited')} entries")
            else:
                print(f"  ✗ {config_name} - 不存在")
        except Exception as e:
            print(f"  ✗ {config_name} - 异常: {e}")

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "-" * 40)
    print("测试基本功能")
    print("-" * 40)
    
    from X.X1 import MemoryCacheManager, CacheConfigs, quick_cache
    
    try:
        # 测试使用预定义配置创建缓存
        cache = MemoryCacheManager(**CacheConfigs.SMALL)
        print("  ✓ 使用预定义配置创建缓存")
        
        # 测试存储和获取
        cache.put("test_key", "test_value", ttl=60)
        value = cache.get("test_key")
        
        if value == "test_value":
            print("  ✓ 存储和获取数据")
        else:
            print(f"  ✗ 存储和获取数据 - 预期: test_value, 实际: {value}")
        
        # 测试统计信息
        stats = cache.get_statistics()
        if 'hit_rate' in stats:
            print("  ✓ 获取统计信息")
        else:
            print("  ✗ 获取统计信息 - 缺少字段")
        
        # 测试全局缓存
        from X.X1 import get_cache, clear_global_cache
        global_cache = get_cache()
        if global_cache is not None:
            print("  ✓ 全局缓存功能")
        else:
            print("  ✗ 全局缓存功能 - 返回None")
        
        # 测试quick_cache
        quick_cache_instance = quick_cache(max_size=100)
        if quick_cache_instance is not None:
            print("  ✓ 快速缓存创建")
        else:
            print("  ✗ 快速缓存创建 - 返回None")
        
        cache.close()
        
        return True
        
    except Exception as e:
        print(f"  ✗ 基本功能测试异常: {e}")
        return False

def test_all_export():
    """测试__all__导出列表"""
    print("\n" + "-" * 40)
    print("测试__all__导出列表")
    print("-" * 40)
    
    from X.X1 import __all__
    
    expected_exports = {
        'MemoryCacheManager', 'CacheEntry', 'CacheStatistics',
        'EvictionStrategy', 'LRUEvictionStrategy', 'TTLEvictionStrategy',
        'SizeEvictionStrategy', 'CacheConfigs',
        'create_cache_manager', 'get_global_cache', 'set_global_cache',
        'get_cache', 'clear_global_cache', 'quick_cache',
        '__version__', '__author__', '__email__', '__all__'
    }
    
    actual_exports = set(__all__)
    
    for export in expected_exports:
        if export in actual_exports:
            print(f"  ✓ {export}")
        else:
            print(f"  ✗ {export} - 未在__all__中")
    
    # 检查是否有额外的导出
    extra_exports = actual_exports - expected_exports
    if extra_exports:
        print(f"\n  额外导出: {extra_exports}")

def test_package_info():
    """测试包信息"""
    print("\n" + "-" * 40)
    print("测试包信息")
    print("-" * 40)
    
    from X.X1 import __version__, __author__, __email__
    
    print(f"  版本: {__version__}")
    print(f"  作者: {__author__}")
    print(f"  邮箱: {__email__}")
    
    if __version__ == "1.0.0":
        print("  ✓ 版本信息正确")
    else:
        print(f"  ✗ 版本信息错误: {__version__}")

def main():
    """主测试函数"""
    print("开始验证X1模块导出接口...")
    
    # 执行所有测试
    if not test_imports():
        return False
    
    test_core_classes()
    test_convenience_functions()
    test_cache_configs()
    test_basic_functionality()
    test_all_export()
    test_package_info()
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)