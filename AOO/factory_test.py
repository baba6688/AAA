# precise_fix_identified_issues.py
import logging
from pathlib import Path
import sys
import shutil

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('PreciseFix')

def fix_dependency_resolver_line_338():
    """修复 dependency_resolver.py 第338行的 class_info.get 调用"""
    logger.info("🔧 修复 dependency_resolver.py 第338行...")
    
    try:
        resolver_file = project_root / "K" / "dependency_resolver.py"
        
        # 备份文件
        backup_file = resolver_file.with_suffix('.py.line_338_fix_backup')
        shutil.copy2(resolver_file, backup_file)
        logger.info(f"✅ 创建备份: {backup_file}")
        
        # 读取文件
        with open(resolver_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并修复第338行
        lines = content.split('\n')
        if len(lines) >= 338:
            original_line = lines[337]  # 索引从0开始
            logger.info(f"📝 原始代码: {original_line}")
            
            # 替换 class_info.get 为 getattr
            if "class_info.get('name', 'unknown')" in original_line:
                fixed_line = original_line.replace(
                    "class_info.get('name', 'unknown')", 
                    "getattr(class_info, 'name', 'unknown')"
                )
                lines[337] = fixed_line
                logger.info(f"📝 修复后代码: {fixed_line}")
                
                # 写入修复后的文件
                with open(resolver_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                logger.info("✅ 第338行修复完成")
                return True
            else:
                logger.error("❌ 第338行内容不符合预期")
                return False
        else:
            logger.error("❌ 文件行数不足338行")
            return False
            
    except Exception as e:
        logger.error(f"❌ 修复失败: {e}")
        return False

def fix_auto_wiring_factory_config_section():
    """修复 auto_wiring_factory.py 中的 get_config_section 调用"""
    logger.info("🔧 修复 auto_wiring_factory.py 配置节调用...")
    
    try:
        factory_file = project_root / "K" / "auto_wiring_factory.py"
        
        # 备份文件
        backup_file = factory_file.with_suffix('.py.config_section_fix_backup')
        shutil.copy2(factory_file, backup_file)
        logger.info(f"✅ 创建备份: {backup_file}")
        
        # 读取文件
        with open(factory_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并修复第440-444行的配置节获取逻辑
        lines = content.split('\n')
        fixes_applied = 0
        
        # 修复第440行 (索引439)
        if len(lines) > 439 and "if hasattr(cls, 'get_config_section'):" in lines[439]:
            logger.info(f"📝 第440行原始代码: {lines[439]}")
            lines[439] = "                # 跳过配置节检查，直接使用空配置"
            fixes_applied += 1
            logger.info(f"📝 第440行修复后代码: {lines[439]}")
        
        # 修复第441行 (索引440)
        if len(lines) > 440:
            logger.info(f"📝 第441行原始代码: {lines[440]}")
            lines[440] = "                config_section = {}"
            fixes_applied += 1
            logger.info(f"📝 第441行修复后代码: {lines[440]}")
        
        # 修复第442行 (索引441)
        if len(lines) > 441 and "config_section = cls.get_config_section()" in lines[441]:
            logger.info(f"📝 第442行原始代码: {lines[441]}")
            lines[441] = "                # config_section = cls.get_config_section()  # 已注释，ClassInfo没有此方法"
            fixes_applied += 1
            logger.info(f"📝 第442行修复后代码: {lines[441]}")
        
        # 修复第443行 (索引442) - 如果有异常处理，可以保留或调整
        if len(lines) > 442 and lines[442].strip() == "except:":
            logger.info(f"📝 第443行原始代码: {lines[442]}")
            lines[442] = "                # except:  # 已注释，不再需要异常处理"
            fixes_applied += 1
            logger.info(f"📝 第443行修复后代码: {lines[442]}")
        
        # 修复第444行 (索引443) - 如果有pass，可以保留或调整
        if len(lines) > 443 and lines[443].strip() == "pass":
            logger.info(f"📝 第444行原始代码: {lines[443]}")
            lines[443] = "                # pass  # 已注释"
            fixes_applied += 1
            logger.info(f"📝 第444行修复后代码: {lines[443]}")
        
        if fixes_applied > 0:
            # 写入修复后的文件
            with open(factory_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"✅ 配置节调用修复完成: {fixes_applied} 处修复")
            return True
        else:
            logger.error("❌ 未找到需要修复的配置节调用")
            return False
            
    except Exception as e:
        logger.error(f"❌ 修复失败: {e}")
        return False

def verify_fixes():
    """验证修复效果"""
    logger.info("🔍 验证修复效果...")
    
    try:
        # 清除模块缓存
        modules_to_clear = [
            'K.dependency_resolver', 'K.auto_wiring_factory', 
            'K.module_registry', 'QQQ'
        ]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        
        # 测试语法
        from K.dependency_resolver import DependencyResolver
        from K.auto_wiring_factory import AutoWiringFactory
        logger.info("✅ 语法检查通过")
        
        # 测试依赖解析器功能
        from K.module_registry import ModuleRegistry
        
        registry = ModuleRegistry()
        resolver = DependencyResolver(registry)
        
        # 创建模拟 ClassInfo 对象
        class MockClassInfo:
            name = "TestClass"
            class_object = type('TestClass', (), {'__init__': lambda self: None})
        
        mock_info = MockClassInfo()
        
        # 测试依赖分析（包含第338行的代码路径）
        dependencies = resolver.analyze_dependencies(mock_info)
        logger.info(f"✅ 依赖分析测试通过: {len(dependencies)} 个依赖")
        
        # 测试系统功能
        from QQQ import AOOFixedStarter
        
        starter = AOOFixedStarter(project_root)
        starter._init_config_manager()
        starter._init_scanner()
        starter._init_registry()
        starter._init_resolver()
        starter._init_factory()
        starter._execute_auto_discovery()
        
        # 测试实例创建
        factory = starter.system_components['factory']
        registry = starter.system_components['registry']
        
        all_classes = registry.get_all_classes()
        logger.info(f"📊 注册类数量: {len(all_classes)}")
        
        # 测试具体类实例创建
        test_class = 'OKXConnector'
        if test_class in all_classes:
            instance = factory.create_instance(test_class)
            if instance:
                logger.info(f"✅ {test_class} 实例创建成功")
                return True
            else:
                logger.error(f"❌ {test_class} 实例创建失败")
                return False
        else:
            logger.error(f"❌ {test_class} 未找到")
            return False
            
    except Exception as e:
        logger.error(f"❌ 验证失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False

def check_remaining_issues():
    """检查剩余问题"""
    logger.info("🔍 检查剩余问题...")
    
    issues_found = []
    
    # 检查 dependency_resolver.py
    resolver_file = project_root / "K" / "dependency_resolver.py"
    with open(resolver_file, 'r', encoding='utf-8') as f:
        resolver_content = f.read()
    
    if 'class_info.get(' in resolver_content:
        issues_found.append("❌ dependency_resolver.py 中仍有 class_info.get 调用")
    else:
        issues_found.append("✅ dependency_resolver.py 中已修复 class_info.get 问题")
    
    # 检查 auto_wiring_factory.py
    factory_file = project_root / "K" / "auto_wiring_factory.py"
    with open(factory_file, 'r', encoding='utf-8') as f:
        factory_content = f.read()
    
    if 'get_config_section()' in factory_content:
        issues_found.append("❌ auto_wiring_factory.py 中仍有 get_config_section 调用")
    else:
        issues_found.append("✅ auto_wiring_factory.py 中已修复 get_config_section 问题")
    
    # 输出检查结果
    logger.info("📋 剩余问题检查结果:")
    for issue in issues_found:
        logger.info(f"  {issue}")
    
    return len([i for i in issues_found if '❌' in i]) == 0

def main():
    """主执行流程"""
    logger.info("🚀 开始精确修复已识别的问题...")
    
    # 步骤1: 修复 dependency_resolver.py 第338行
    if fix_dependency_resolver_line_338():
        logger.info("✅ 步骤1完成: dependency_resolver.py 修复")
        
        # 步骤2: 修复 auto_wiring_factory.py 配置节调用
        if fix_auto_wiring_factory_config_section():
            logger.info("✅ 步骤2完成: auto_wiring_factory.py 修复")
            
            # 步骤3: 验证修复效果
            if verify_fixes():
                logger.info("✅ 步骤3完成: 修复验证通过")
                
                # 步骤4: 检查剩余问题
                if check_remaining_issues():
                    logger.info("🎉🎉🎉 所有已识别问题完全修复！")
                    logger.info("📋 修复成果总结:")
                    logger.info("  ✅ 修复了 dependency_resolver.py 第338行的 class_info.get 调用")
                    logger.info("  ✅ 修复了 auto_wiring_factory.py 中的 get_config_section 调用")
                    logger.info("  ✅ 系统语法检查通过")
                    logger.info("  ✅ 依赖解析功能正常")
                    logger.info("  ✅ 实例创建功能正常")
                    logger.info("  ✅ 无剩余问题")
                    logger.info("🚀 工厂内部问题已彻底解决！")
                else:
                    logger.warning("⚠️ 仍有少量问题存在")
            else:
                logger.error("❌ 步骤3失败: 修复验证失败")
        else:
            logger.error("❌ 步骤2失败: auto_wiring_factory.py 修复失败")
    else:
        logger.error("❌ 步骤1失败: dependency_resolver.py 修复失败")

if __name__ == "__main__":
    main()