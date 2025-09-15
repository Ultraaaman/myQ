#!/usr/bin/env python3
"""
简化的Portfolio与Strategy集成演示

展示基本的策略执行功能
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """主演示函数"""
    print("🚀 Portfolio策略集成简化演示")
    print("=" * 50)
    
    try:
        # 1. 测试基础导入
        print("1. 测试模块导入...")
        from quantlib.portfolio import (
            create_strategy_executor, 
            StrategyType
        )
        from quantlib.strategy import create_ma_cross_strategy
        print("✅ 模块导入成功")
        
        # 2. 创建策略执行器
        print("\n2. 创建策略执行器...")
        executor = create_strategy_executor(mode="live", initial_capital=100000)
        print(f"✅ 执行器创建成功: 初始资金 ${executor.initial_capital:,.2f}")
        
        # 3. 创建简单的均线交叉策略
        print("\n3. 创建均线交叉策略...")
        symbols = ['000001', '000002']
        ma_strategy = create_ma_cross_strategy(symbols, short_window=10, long_window=30)
        print(f"✅ 均线策略创建成功: 监控股票 {symbols}")
        
        # 4. 添加策略到执行器
        print("\n4. 添加策略到执行器...")
        success = executor.add_strategy(
            name="MA_Cross_10_30",
            strategy=ma_strategy,
            weight=1.0,
            strategy_type=StrategyType.TECHNICAL
        )
        print(f"✅ 策略添加{'成功' if success else '失败'}")
        
        # 5. 生成模拟数据并执行测试
        print("\n5. 生成模拟数据并执行...")
        
        # 生成样本数据
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        sample_data = {}
        
        for symbol in symbols:
            # 模拟股价走势
            np.random.seed(hash(symbol) % 2**32)
            base_price = 10.0
            price_changes = np.random.normal(0.001, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(price_changes))
            
            sample_data[symbol] = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': prices,
                'volume': 1000000 + np.random.randint(-200000, 200000, len(dates))
            }, index=dates)
        
        # 设置策略数据
        executor.set_data(sample_data)
        executor.initialize_strategies()
        print("✅ 策略数据设置和初始化完成")
        
        # 6. 执行几个交易日的模拟
        print("\n6. 执行模拟交易...")
        
        for i in range(3):
            current_date = dates[i]
            current_data = {}
            prices = {}
            
            for symbol in symbols:
                latest_data = sample_data[symbol].loc[current_date]
                current_data[symbol] = latest_data
                prices[symbol] = latest_data['close']
            
            # 执行单步
            result = executor.execute_single_step(current_date, current_data, prices)
            
            print(f"  第{i+1}天 ({current_date.strftime('%Y-%m-%d')}):")
            print(f"    信号数: {result['total_signals']}")
            print(f"    执行交易: {result['executed_trades']}")
            print(f"    组合价值: ${result['portfolio_value']:,.2f}")
            print(f"    现金余额: ${result['cash']:,.2f}")
        
        # 7. 显示最终结果
        print("\n7. 最终结果摘要:")
        executor.print_summary()
        
        print(f"\n✅ 演示完成!")
        print(f"   总执行步骤: {len(executor.execution_history)}")
        print(f"   当前组合价值: ${executor.portfolio.get_total_value():,.2f}")
        print(f"   总交易数: {len(executor.portfolio.trades)}")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()