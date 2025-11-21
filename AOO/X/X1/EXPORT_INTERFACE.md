# X1内存缓存管理器导出接口文档

## 概述

X1模块提供了完整的内存缓存管理器导出接口，包含8个核心类和多个便捷函数，支持多种缓存策略、线程安全访问、内存监控等功能。

## 导出接口清单

### 核心类 (8个主要类)

1. **MemoryCacheManager** - 主要的内存缓存管理器
2. **CacheEntry** - 缓存条目数据结构
3. **CacheStatistics** - 缓存统计信息
4. **EvictionStrategy** - 缓存清理策略基类
5. **LRUEvictionStrategy** - LRU清理策略
6. **TTLEvictionStrategy** - TTL清理策略
7. **SizeEvictionStrategy** - 大小限制清理策略
8. **CacheConfigs** - 预定义的缓存配置类

### 便捷函数

- **create_cache_manager** - 创建缓存管理器
- **get_global_cache** - 获取全局缓存管理器实例
- **set_global_cache** - 设置全局缓存管理器实例
- **get_cache** - 获取或创建全局缓存管理器
- **clear_global_cache** - 清空全局缓存
- **quick_cache** - 快速创建简单缓存管理器

### 包信息

- **__version__** - 版本信息 (1.0.0)
- **__author__** - 作者信息 (X1 Team)
- **__email__** - 邮箱信息 (team@x1.com)

## 使用示例

### 基础导入
```python
from X.X1 import *

# 或者按需导入
from X.X1 import MemoryCacheManager, CacheEntry, CacheConfigs
```

### 创建缓存管理器
```python
# 使用便捷函数
cache = quick_cache(max_size=1000, default_ttl=300)

# 使用预定义配置
from X.X1 import MemoryCacheManager, CacheConfigs
cache = MemoryCacheManager(**CacheConfigs.MEDIUM)

# 直接实例化
cache = MemoryCacheManager(
    max_size=1000,
    max_memory_bytes=100*1024*1024,
    default_ttl=1800,
    eviction_strategy='lru',
    enable_persistence=True
)
```

### 缓存操作
```python
# 存储数据
cache.put("user:1", {"name": "张三", "age": 25})
cache.put("user:2", {"name": "李四", "age": 30}, ttl=60)

# 获取数据
user1 = cache.get("user:1")
user2 = cache.get("user:2", default={"name": "默认用户"})

# 删除数据
cache.delete("user:1")

# 检查是否存在
if "user:2" in cache:
    print("用户存在")
```

### 全局缓存管理
```python
from X.X1 import get_cache, clear_global_cache

# 获取全局缓存
cache = get_cache()

# 清空全局缓存
clear_global_cache()
```

### 统计信息
```python
# 获取统计信息
stats = cache.get_statistics()
print(f"命中率: {stats['hit_rate']:.2%}")
print(f"当前大小: {stats['current_size']}")
print(f"内存使用: {stats['current_memory_usage']} 字节")

# 获取内存使用情况
memory_info = cache.get_memory_usage()
print(f"内存使用率: {memory_info['usage_percentage']:.2f}%")
```

## 预定义配置

CacheConfigs类提供了5种预定义的缓存配置：

### SMALL - 小型缓存（适合会话数据）
- 最大条目数：100
- 最大内存：10MB
- 默认TTL：5分钟
- 清理间隔：1分钟

### MEDIUM - 中型缓存（适合应用缓存）
- 最大条目数：1000
- 最大内存：100MB
- 默认TTL：30分钟
- 清理间隔：5分钟

### LARGE - 大型缓存（适合数据缓存）
- 最大条目数：10000
- 最大内存：1GB
- 默认TTL：1小时
- 清理间隔：10分钟

### PERSISTENT - 持久化缓存（适合重要数据）
- 最大条目数：5000
- 最大内存：500MB
- 默认TTL：2小时
- 清理间隔：15分钟
- 启用持久化：True

### UNLIMITED - 无限制缓存（适合测试）
- 最大条目数：无限制
- 最大内存：无限制
- 默认TTL：永不过期
- 清理间隔：不自动清理

## 缓存策略

### LRU策略（最近最少使用）
```python
cache = MemoryCacheManager(eviction_strategy='lru')
```

### TTL策略（过期时间）
```python
cache = MemoryCacheManager(eviction_strategy='ttl')
```

### Size策略（大小限制）
```python
cache = MemoryCacheManager(eviction_strategy='size')
```

## 高级功能

### 条件获取
```python
# 获取缓存值，如果不存在则设置并返回
user = cache.get_or_set("user:3", lambda: {"name": "王五", "age": 35}, ttl=300)
```

### TTL管理
```python
# 获取剩余TTL
remaining = cache.get_ttl("user:1")

# 设置TTL
cache.set_ttl("user:1", 600)  # 10分钟
```

### 批量操作
```python
# 获取所有键
keys = cache.get_keys(pattern="user:*")

# 获取所有条目
entries = cache.get_entries()
```

### 数据持久化
```python
# 启用持久化
cache = MemoryCacheManager(
    enable_persistence=True,
    persistence_file="my_cache.pkl"
)

# 手动保存
cache.save_to_disk()

# 导入导出数据
json_data = cache.export_data(format='json')
cache.import_data(json_data, format='json')
```

## 线程安全

所有操作都是线程安全的，可以在多线程环境中使用：

```python
import threading
from X.X1 import MemoryCacheManager

cache = MemoryCacheManager()

def worker():
    for i in range(100):
        cache.put(f"key_{i}", f"value_{i}")

# 创建多个线程
threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()

for t in threads:
    t.join()
```

## 上下文管理器支持

```python
with MemoryCacheManager() as cache:
    cache.put("key", "value")
    # 缓存会自动保存和清理
```

## 错误处理

```python
try:
    cache = MemoryCacheManager()
    cache.put("key", "value")
except Exception as e:
    print(f"缓存操作错误: {e}")
finally:
    cache.close()  # 确保资源清理
```

## 性能监控

```python
# 监控统计
stats = cache.get_statistics()
print(f"命中率: {stats['hit_rate']:.2%}")
print(f"总请求数: {stats['total_requests']}")
print(f"清理次数: {stats['evictions']}")
print(f"过期次数: {stats['expirations']}")
print(f"运行时间: {stats['uptime']:.2f}秒")
```

## 版本信息

- **版本**: 1.0.0
- **作者**: X1 Team
- **邮箱**: team@x1.com
- **最后更新**: 2025-11-14

## 注意事项

1. 使用完毕后记得调用`close()`方法清理资源
2. 大型缓存建议启用监控功能进行性能跟踪
3. 持久化功能会在程序结束时自动保存
4. 在多线程环境中，所有操作都是线程安全的
5. 预定义配置可直接使用，也可根据需要调整参数

---

*本文档描述了X1内存缓存管理器的完整导出接口和使用方法。如有疑问请联系开发团队。*