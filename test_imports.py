#!/usr/bin/env python3
"""
测试导入问题的脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    print("1. 测试基础模块导入...")
    from quantlib.strategy.base import BaseStrategy, SignalType, TradingSignal
    print("✅ strategy.base 导入成功")
    
    print("2. 测试因子策略导入...")
    from quantlib.strategy.factor_strategies import FactorType, SingleFactorStrategy
    print("✅ strategy.factor_strategies 导入成功")
    
    print("3. 测试portfolio manager导入...")
    from quantlib.portfolio.manager import PortfolioManager, FactorPortfolioManager
    print("✅ portfolio.manager 导入成功")
    
    print("4. 测试strategy executor导入...")
    from quantlib.portfolio.strategy_executor import StrategyExecutor
    print("✅ portfolio.strategy_executor 导入成功")
    
    print("5. 测试顶层导入...")
    from quantlib.portfolio import create_strategy_executor
    print("✅ portfolio 顶层导入成功")
    
    from quantlib.strategy import create_factor_strategy
    print("✅ strategy 顶层导入成功")
    
    print("\n🎉 所有导入测试通过!")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()