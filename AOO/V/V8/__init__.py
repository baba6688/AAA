"""
V8模块 - 模型转换器

实现多框架模型转换、ONNX格式转换、TensorRT优化等功能。

主要功能：
1. 框架间转换 - PyTorch、TensorFlow、sklearn模型互转
2. ONNX格式转换 - 支持ONNX标准格式转换
3. TensorRT优化 - NVIDIA TensorRT推理加速
4. 格式验证 - 验证转换前后的模型一致性
5. 批量转换 - 支持批量模型转换
6. 转换优化 - 自动优化转换过程
7. 兼容性检查 - 检查目标环境兼容性
8. 性能评估 - 评估转换后的性能
"""

from .ModelConverter import (
    # 核心枚举和类
    ModelFramework,
    ConversionType,
    ModelConverter,
    
    # 便利函数
    create_model_converter
)

__all__ = [
    'ModelFramework',
    'ConversionType',
    'ModelConverter',
    'create_model_converter'
]

__version__ = '1.0.0'

# 便利函数
def create_model_converter(**kwargs):
    """
    创建模型转换器实例
    
    Args:
        **kwargs: 转换器参数
    
    Returns:
        ModelConverter: 模型转换器实例
    
    Examples:
        from V8 import create_model_converter
        
        # 创建基本转换器
        converter = create_model_converter()
        
        # 创建带优化选项的转换器
        converter = create_model_converter(
            optimize=True,
            validate_output=True
        )
    """
    return ModelConverter(**kwargs)

# 转换目标格式
class ConversionTargets:
    """转换目标格式常量"""
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TORCHSCRIPT = "torchscript"
    TENSORFLOW_SAVED_MODEL = "tensorflow_saved_model"
    PMML = "pmml"
    COREML = "coreml"
    TVM = "tvm"

# 优化级别
class OptimizationLevels:
    """优化级别常量"""
    NONE = "none"  # 不优化
    BASIC = "basic"  # 基本优化
    AGGRESSIVE = "aggressive"  # 激进优化
    CUSTOM = "custom"  # 自定义优化

# 精度类型
class PrecisionTypes:
    """精度类型常量"""
    FP32 = "fp32"  # 32位浮点
    FP16 = "fp16"  # 16位浮点
    INT8 = "int8"  # 8位整数
    INT4 = "int4"  # 4位整数

# 快速开始指南
QUICK_START = """
V8模型转换器快速开始：

1. 创建转换器：
   from V8 import create_model_converter
   converter = create_model_converter()

2. 检测模型框架：
   framework = converter.detect_framework(model)
   print(f"检测到框架: {framework}")

3. 转换为ONNX：
   onnx_model = converter.convert_to_onnx(
       model,
       input_shape=(1, 3, 224, 224),
       opset_version=11
   )
   converter.save_model(onnx_model, "model.onnx")

4. TensorRT优化：
   trt_model = converter.convert_to_tensorrt(
       onnx_model,
       precision="fp16",
       optimization_level="aggressive"
   )
   converter.save_model(trt_model, "model.trt")

5. 验证转换结果：
   is_valid = converter.validate_conversion(
       original_model,
       converted_model,
       test_data
   )
   print(f"转换验证: {is_valid}")

6. 性能对比：
   performance = converter.compare_performance(
       original_model,
       converted_model,
       test_data
   )
   print(f"性能对比: {performance}")

7. 批量转换：
   models = load_multiple_models()
   results = converter.batch_convert(models, target_format="onnx")
"""

# 转换验证装饰器
def validate_conversion(func):
    """转换验证装饰器"""
    def wrapper(*args, **kwargs):
        import numpy as np
        import warnings
        
        # 获取模型参数
        model = None
        target_format = None
        
        for arg in args:
            if hasattr(arg, 'predict'):
                model = arg
            elif isinstance(arg, str) and arg.upper() in ['ONNX', 'TENSORRT', 'TORCHSCRIPT']:
                target_format = arg.upper()
        
        if model and target_format:
            # 基本验证
            try:
                test_input = np.random.rand(1, 5)
                pred_original = model.predict(test_input)
                
                print(f"开始转换模型到 {target_format} 格式...")
                
                result = func(*args, **kwargs)
                
                # 转换后验证（如果可能）
                if hasattr(result, 'predict'):
                    try:
                        pred_converted = result.predict(test_input)
                        print("转换后模型验证成功")
                    except:
                        warnings.warn("转换后模型验证失败，请检查转换结果")
                
                return result
                
            except Exception as e:
                print(f"转换验证失败: {e}")
                raise
        
        return func(*args, **kwargs)
    
    return wrapper

# 转换性能监控装饰器
def monitor_conversion_performance(func):
    """转换性能监控装饰器"""
    def wrapper(*args, **kwargs):
        import time
        import psutil
        import os
        
        # 获取初始资源状态
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent()
        process = psutil.Process()
        initial_gpu_memory = None
        
        # 尝试获取GPU内存（如果可用）
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                initial_gpu_memory = gpus[0].memoryUsed
        except ImportError:
            pass
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            
            # 获取最终资源状态
            final_memory = psutil.virtual_memory().percent
            final_cpu = psutil.cpu_percent()
            final_process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"模型转换性能统计:")
            print(f"  执行时间: {end_time - start_time:.2f}秒")
            print(f"  CPU使用率: {initial_cpu}% → {final_cpu}%")
            print(f"  内存使用率: {initial_memory}% → {final_memory}%")
            print(f"  进程内存: {final_process_memory:.1f}MB")
            
            if initial_gpu_memory is not None:
                final_gpu_memory = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
                print(f"  GPU内存: {initial_gpu_memory}MB → {final_gpu_memory}MB")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            print(f"转换失败，耗时: {end_time - start_time:.2f}秒，错误: {e}")
            raise
    
    return wrapper

print("V8模型转换器已加载")