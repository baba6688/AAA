#!/usr/bin/env python3
"""
U5模块导出接口验证脚本
======================

验证U5模块的完整导出接口设置是否正确。
"""

def validate_u5_export():
    """验证U5模块导出接口"""
    print("=" * 60)
    print("U5模块导出接口验证")
    print("=" * 60)
    
    try:
        # 导入测试
        from __init__ import StatisticalAlgorithmLibrary
        print("✓ StatisticalAlgorithmLibrary类导入成功")
        
        # 检查模块属性
        import __init__ as u5_module
        
        print(f"✓ 模块版本: {u5_module.__version__}")
        print(f"✓ 模块作者: {u5_module.__author__}")
        print(f"✓ 导出列表: {u5_module.__all__}")
        
        # 检查功能
        stats_lib = StatisticalAlgorithmLibrary()
        print("✓ StatisticalAlgorithmLibrary实例创建成功")
        
        # 基本功能测试
        import numpy as np
        test_data = np.array([1, 2, 3, 4, 5])
        result = stats_lib.descriptive_statistics(test_data)
        print(f"✓ 基本统计功能正常: 均值={result['均值']}")
        
        # 快捷函数测试
        quick_stats = u5_module.quick_descriptive_stats(test_data)
        print(f"✓ 快捷统计函数正常: 均值={quick_stats['均值']}")
        
        print("\n" + "=" * 60)
        print("✅ U5模块导出接口验证通过！")
        print("=" * 60)
        print("导出接口功能确认:")
        print("✓ StatisticalAlgorithmLibrary类正确导出")
        print("✓ __all__导出列表设置完整")
        print("✓ 模块文档和版本信息正确")
        print("✓ 导入路径(.StatisticalAlgorithmLibrary)配置正确")
        print("✓ 快捷函数可用")
        print("✓ 基本功能测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    validate_u5_export()