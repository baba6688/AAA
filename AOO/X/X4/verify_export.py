#!/usr/bin/env python3
"""
X4导出接口最终验证脚本
从X模块级别测试导入
"""

import sys
sys.path.insert(0, '/workspace')

def test_x4_export():
    """测试X4模块的导出接口"""
    
    print("🔍 验证X4模块导出接口...")
    
    try:
        # 测试从X.X4导入
        from X.X4 import (
            CacheStrategyManager,
            CacheStrategy,
            LRUStrategy,
            LFUStrategy,
            TTLStrategy,
            FIFOStrategy,
            CacheConfig,
            StrategyMetrics,
            StrategyPerformance,
            CacheStrategyBase,
            create_cache_manager,
            cached
        )
        
        print("✅ 所有类导入成功")
        
        # 验证主要功能
        manager = create_cache_manager(max_size=100)
        
        # 测试不同策略
        for strategy in [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.FIFO, CacheStrategy.TTL]:
            manager.switch_strategy(strategy)
            print(f"✅ 策略 {strategy.value} 切换成功")
        
        # 测试缓存操作
        manager.put("test", "value")
        result = manager.get("test")
        assert result == "value", "缓存读写测试失败"
        print("✅ 缓存读写功能正常")
        
        # 测试指标收集
        metrics = manager.get_strategy_metrics()
        print(f"✅ 指标收集正常: 命中次数={metrics.hits}")
        
        manager.cleanup()
        
        print("\n🎉 X4导出接口验证完成 - 全部功能正常！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    success = test_x4_export()
    sys.exit(0 if success else 1)