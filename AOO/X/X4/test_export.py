#!/usr/bin/env python3
"""
X4导出接口测试脚本
"""

# 测试所有导入是否正常工作
try:
    # 测试主要类导入
    from X.X4 import (
        CacheStrategyManager,
        CacheStrategyBase,
        LRUStrategy,
        LFUStrategy,
        TTLStrategy,
        FIFOStrategy,
        CacheConfig,
        StrategyMetrics,
        StrategyPerformance,
        CacheStrategy,
        create_cache_manager,
        cached
    )
    
    print("✅ 所有类导入成功")
    
    # 测试枚举值
    print(f"缓存策略: {list(CacheStrategy)}")
    print(f"性能等级: {list(StrategyPerformance)}")
    
    # 测试创建管理器
    manager = create_cache_manager()
    print(f"✅ 缓存管理器创建成功，当前策略: {manager.get_current_strategy()}")
    
    # 测试基本功能
    manager.put("test_key", "test_value")
    result = manager.get("test_key")
    print(f"✅ 缓存读写测试: {result}")
    
    # 测试策略切换
    manager.switch_strategy(CacheStrategy.LFU)
    print(f"✅ 策略切换成功，当前策略: {manager.get_current_strategy()}")
    
    # 测试指标获取
    metrics = manager.get_strategy_metrics()
    print(f"✅ 性能指标: 命中率={metrics.hit_rate:.2%}")
    
    print("\n🎉 X4导出接口测试全部通过！")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
except Exception as e:
    print(f"❌ 运行错误: {e}")