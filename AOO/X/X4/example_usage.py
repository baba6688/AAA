#!/usr/bin/env python3
"""
X4缓存策略管理器 - 完整使用示例
展示所有导出的功能和使用方法
"""

from X.X4 import (
    # 核心类
    CacheStrategyManager,
    CacheStrategyBase,
    
    # 策略实现
    LRUStrategy,
    LFUStrategy,
    TTLStrategy,
    FIFOStrategy,
    
    # 配置和指标
    CacheConfig,
    StrategyMetrics,
    StrategyPerformance,
    
    # 枚举
    CacheStrategy,
    
    # 工具函数
    create_cache_manager,
    cached,
    
    # 额外功能
    quick_start,
    run_benchmark,
    get_default_config,
    get_config_template,
    ERROR_CODES,
    get_error_message
)

def main():
    print("=" * 60)
    print("X4缓存策略管理器 - 完整功能展示")
    print("=" * 60)
    
    # 1. 使用工厂函数创建管理器
    print("\n1. 创建缓存管理器")
    print("-" * 30)
    
    # 默认配置
    manager1 = create_cache_manager()
    print(f"默认管理器 - 当前策略: {manager1.get_current_strategy()}")
    
    # 自定义配置
    config = CacheConfig(max_size=500, ttl=1800, enable_optimization=True)
    manager2 = CacheStrategyManager(config)
    print(f"自定义配置管理器 - 最大容量: {config.max_size}")
    
    # 使用配置模板
    web_config = get_config_template("web_cache")
    print(f"Web缓存配置模板: {web_config}")
    
    # 2. 基本缓存操作
    print("\n2. 基本缓存操作")
    print("-" * 30)
    
    manager = create_cache_manager()
    
    # 存储数据
    manager.put("user:1", {"name": "张三", "age": 25})
    manager.put("user:2", {"name": "李四", "age": 30})
    manager.put("session:abc", "active")
    
    # 读取数据
    user1 = manager.get("user:1")
    print(f"读取用户1: {user1}")
    
    # 删除数据
    removed = manager.remove("session:abc")
    print(f"删除session: {'成功' if removed else '失败'}")
    
    # 3. 策略管理
    print("\n3. 策略管理")
    print("-" * 30)
    
    # 查看可用策略
    strategies = manager.get_available_strategies()
    print(f"可用策略: {[s.value for s in strategies]}")
    
    # 切换策略
    current = manager.get_current_strategy()
    print(f"当前策略: {current.value}")
    
    # 切换到LFU
    manager.switch_strategy(CacheStrategy.LFU)
    print(f"切换后策略: {manager.get_current_strategy().value}")
    
    # 4. 性能监控
    print("\n4. 性能监控")
    print("-" * 30)
    
    # 执行一些操作来产生指标
    for i in range(10):
        manager.put(f"key_{i}", f"value_{i}")
    
    for i in range(15):  # 包含一些不存在的key
        manager.get(f"key_{i}")
    
    # 获取当前策略指标
    metrics = manager.get_strategy_metrics()
    print(f"当前策略指标:")
    print(f"  命中次数: {metrics.hits}")
    print(f"  未命中次数: {metrics.misses}")
    print(f"  命中率: {metrics.hit_rate:.2%}")
    print(f"  驱逐次数: {metrics.evictions}")
    print(f"  平均响应时间: {metrics.avg_response_time:.6f}秒")
    
    # 获取所有策略指标
    all_metrics = manager.get_all_metrics()
    print(f"\n所有策略指标:")
    for strategy, metrics in all_metrics.items():
        print(f"  {strategy}: 命中率={metrics.hit_rate:.2%}")
    
    # 5. 策略优化
    print("\n5. 策略优化")
    print("-" * 30)
    
    # 评估策略性能
    for strategy in strategies:
        performance = manager.optimize_strategy(strategy)
        print(f"{strategy.value}: {performance.value}")
    
    # 自动优化
    best_strategy = manager.auto_optimize()
    if best_strategy:
        print(f"自动优化建议切换到: {best_strategy.value}")
    
    # 6. 策略对比
    print("\n6. 策略对比分析")
    print("-" * 30)
    
    comparison = manager.get_strategy_comparison()
    for strategy, info in comparison.items():
        metrics = info['metrics']
        performance = info['performance']
        is_current = info['is_current']
        status = " (当前)" if is_current else ""
        print(f"{strategy}{status}:")
        print(f"  性能等级: {performance}")
        print(f"  命中率: {metrics['hit_rate']:.2%}")
        print(f"  平均响应时间: {metrics['avg_response_time']:.6f}秒")
    
    # 7. 使用统计
    print("\n7. 使用统计")
    print("-" * 30)
    
    stats = manager.get_usage_statistics()
    print(f"总体统计:")
    print(f"  总操作次数: {stats['total_operations']}")
    print(f"  总命中次数: {stats['total_hits']}")
    print(f"  总未命中次数: {stats['total_misses']}")
    print(f"  总体命中率: {stats['overall_hit_rate']:.2%}")
    print(f"  策略切换次数: {stats['strategy_switches']}")
    print(f"  当前策略: {stats['current_strategy']}")
    
    # 8. 配置管理
    print("\n8. 配置导入导出")
    print("-" * 30)
    
    # 导出配置
    config_json = manager.export_configuration()
    print(f"导出配置大小: {len(config_json)} 字符")
    
    # 创建新管理器并导入配置
    new_manager = CacheStrategyManager()
    import_success = new_manager.import_configuration(config_json)
    print(f"导入配置: {'成功' if import_success else '失败'}")
    
    # 9. 缓存装饰器
    print("\n9. 缓存装饰器使用")
    print("-" * 30)
    
    cache_manager = create_cache_manager()
    
    @cached(cache_manager, "calc:")
    def expensive_calculation(x, y):
        """模拟耗时计算"""
        import time
        time.sleep(0.1)  # 模拟计算时间
        return x * x + y * y
    
    # 第一次调用（缓存未命中）
    import time
    start = time.time()
    result1 = expensive_calculation(3, 4)
    time1 = time.time() - start
    print(f"第一次计算结果: {result1}, 耗时: {time1:.3f}秒")
    
    # 第二次调用（缓存命中）
    start = time.time()
    result2 = expensive_calculation(3, 4)
    time2 = time.time() - start
    print(f"第二次计算结果: {result2}, 耗时: {time2:.3f}秒")
    print(f"缓存加速: {time1/time2:.1f}倍")
    
    # 10. 性能基准测试
    print("\n10. 性能基准测试")
    print("-" * 30)
    
    benchmark_results = run_benchmark()
    for strategy, results in benchmark_results.items():
        print(f"{strategy}:")
        print(f"  插入时间: {results['insert_time']:.4f}秒")
        print(f"  读取时间: {results['read_time']:.4f}秒")
        print(f"  命中率: {results['hit_rate']:.2%}")
    
    # 11. 错误处理
    print("\n11. 错误处理示例")
    print("-" * 30)
    
    # 展示错误代码
    print(f"错误代码示例:")
    for name, code in ERROR_CODES.items():
        message = get_error_message(code)
        print(f"  {name} ({code}): {message}")
    
    # 12. 清理资源
    print("\n12. 清理资源")
    print("-" * 30)
    
    manager.cleanup()
    cache_manager.cleanup()
    print("✅ 所有缓存管理器已清理")
    
    print("\n" + "=" * 60)
    print("🎉 X4缓存策略管理器功能展示完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()