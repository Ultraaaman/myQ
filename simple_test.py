#!/usr/bin/env python3
"""
简单的导入测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("开始测试导入...")

try:
    print("测试 1: 导入基础模块")
    import quantlib
    print("✅ quantlib 基础导入成功")
    
    print("测试 2: 导入 portfolio 模块")
    import quantlib.portfolio
    print("✅ quantlib.portfolio 导入成功")
    
    print("测试 3: 导入特定函数")
    from quantlib.portfolio import PortfolioManager
    print("✅ PortfolioManager 导入成功")
    
    print("测试 4: 导入执行器")
    from quantlib.portfolio import create_strategy_executor
    print("✅ create_strategy_executor 导入成功")
    
    print("测试 5: 测试策略模块")
    from quantlib.strategy import create_ma_cross_strategy
    print("✅ create_ma_cross_strategy 导入成功")
    
    print("\n🎉 所有基础导入测试通过!")
    
    # 创建一个简单的实例测试
    print("\n测试实例创建...")
    executor = create_strategy_executor("live", 100000)
    print(f"✅ 执行器创建成功: {type(executor)}")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()