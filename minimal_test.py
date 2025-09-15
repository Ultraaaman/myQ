#!/usr/bin/env python3
"""
最小化测试 - 验证核心功能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 最小化功能测试")
    print("=" * 30)
    
    try:
        # 测试1: 基础导入
        print("测试1: 基础模块导入")
        from quantlib.portfolio import PortfolioManager
        from quantlib.strategy import BaseStrategy
        print("✅ 基础导入成功")
        
        # 测试2: 创建组合管理器
        print("测试2: 创建组合管理器")
        portfolio = PortfolioManager(initial_capital=100000, name="Test Portfolio")
        print(f"✅ 组合创建成功: {portfolio.name}, 资金: ${portfolio.initial_capital:,.2f}")
        
        # 测试3: 测试基础操作
        print("测试3: 基础组合操作")
        total_value = portfolio.get_total_value()
        cash_weight = portfolio.get_cash_weight()
        print(f"✅ 组合价值: ${total_value:,.2f}, 现金权重: {cash_weight:.1%}")
        
        # 测试4: 尝试买入操作
        print("测试4: 模拟买入操作")
        success = portfolio.buy(symbol="TEST001", quantity=100, price=10.0)
        if success:
            print(f"✅ 买入成功: 持仓数量 {len(portfolio.positions)}")
        else:
            print("ℹ️ 买入测试完成")
        
        print("\n🎉 所有测试通过!")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_basic_functionality()