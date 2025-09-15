#!/usr/bin/env python3
"""
Portfolio与Strategy模块集成示例

展示如何使用统一的策略执行框架整合：
- Portfolio管理
- Strategy策略
- Backtest回测
- Factor因子投资

完整演示Live模式和Backtest模式的使用方法
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

# 导入所需模块
from quantlib.portfolio import (
    create_strategy_executor, 
    create_factor_executor,
    ExecutionMode,
    StrategyType
)
from quantlib.strategy import (
    create_ma_cross_strategy,
    create_factor_strategy,
    create_factor_multi_strategy,
    FactorType
)
# from quantlib.market_data import get_stock_data  # 暂时注释掉


def generate_sample_factor_data(symbols: list, periods: int = 252) -> dict:
    """生成示例因子数据"""
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
    
    # 生成价值因子数据 (模拟P/E比率倒数)
    np.random.seed(42)
    value_data = pd.DataFrame(
        np.random.normal(0.05, 0.02, (periods, len(symbols))),
        index=dates,
        columns=symbols
    )
    
    # 生成动量因子数据 (模拟20日收益率)
    momentum_data = pd.DataFrame(
        np.random.normal(0.02, 0.05, (periods, len(symbols))),
        index=dates,
        columns=symbols
    )
    
    # 生成质量因子数据 (模拟ROE)
    quality_data = pd.DataFrame(
        np.random.normal(0.15, 0.05, (periods, len(symbols))),
        index=dates,
        columns=symbols
    )
    
    return {
        FactorType.VALUE: value_data,
        FactorType.MOMENTUM: momentum_data,
        FactorType.QUALITY: quality_data
    }


def demo_basic_strategy_execution():
    """基础策略执行演示"""
    print("=== 基础策略执行演示 ===")
    
    # 创建策略执行器 (Live模式)
    executor = create_strategy_executor(mode="live", initial_capital=100000)
    
    # 添加技术分析策略
    symbols = ['000001', '000002', '600519']
    ma_strategy = create_ma_cross_strategy(symbols, short_window=10, long_window=30)
    
    executor.add_strategy(
        name="MA_Cross_10_30",
        strategy=ma_strategy,
        weight=1.0,
        strategy_type=StrategyType.TECHNICAL
    )
    
    print(f"✅ 已添加策略: {list(executor.strategies.keys())}")
    
    # 生成模拟数据进行单步执行
    current_time = datetime.now()
    current_data = {}
    prices = {}
    
    for symbol in symbols:
        # 模拟当前市场数据
        current_data[symbol] = pd.Series({
            'open': 10.0 + np.random.normal(0, 0.5),
            'high': 10.5 + np.random.normal(0, 0.5), 
            'low': 9.5 + np.random.normal(0, 0.5),
            'close': 10.0 + np.random.normal(0, 0.5),
            'volume': 1000000 + np.random.randint(-200000, 200000)
        })
        prices[symbol] = current_data[symbol]['close']
    
    # 执行单个时间步
    result = executor.execute_single_step(current_time, current_data, prices)
    
    print(f"执行结果:")
    print(f"  总信号数: {result['total_signals']}")
    print(f"  执行交易数: {result['executed_trades']}")
    print(f"  组合价值: ${result['portfolio_value']:,.2f}")
    print(f"  现金余额: ${result['cash']:,.2f}")
    
    # 打印执行器摘要
    executor.print_summary()


def demo_factor_investment():
    """因子投资演示"""
    print("\n=== 因子投资策略演示 ===")
    
    # 创建因子策略执行器
    executor = create_factor_executor(initial_capital=200000, mode="live")
    
    symbols = ['000001', '000002', '000858', '600519', '600036']
    
    # 生成示例因子数据
    factor_data = generate_sample_factor_data(symbols, periods=100)
    
    # 创建单因子策略（价值因子）
    value_strategy = create_factor_strategy(
        factor_type=FactorType.VALUE,
        symbols=symbols,
        factor_data=factor_data[FactorType.VALUE],
        long_pct=0.4,  # 买入排名前40%的股票
        short_pct=0.2   # 卖出排名后20%的股票
    )
    
    # 创建多因子策略
    multi_factor_strategy = create_factor_multi_strategy(
        symbols=symbols,
        factor_data=factor_data,
        factor_weights={
            FactorType.VALUE: 0.4,
            FactorType.MOMENTUM: 0.3, 
            FactorType.QUALITY: 0.3
        }
    )
    
    # 添加策略到执行器
    executor.add_strategy("Value_Factor", value_strategy, weight=0.6, strategy_type=StrategyType.FACTOR)
    executor.add_strategy("Multi_Factor", multi_factor_strategy, weight=0.4, strategy_type=StrategyType.FACTOR)
    
    print(f"✅ 已添加因子策略: {list(executor.strategies.keys())}")
    
    # 设置数据并初始化
    sample_data = {}
    for symbol in symbols:
        # 生成样本股价数据
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        sample_data[symbol] = pd.DataFrame({
            'open': 10 + np.cumsum(np.random.normal(0, 0.02, 100)),
            'high': 10.2 + np.cumsum(np.random.normal(0, 0.02, 100)),
            'low': 9.8 + np.cumsum(np.random.normal(0, 0.02, 100)),
            'close': 10 + np.cumsum(np.random.normal(0, 0.02, 100)),
            'volume': 1000000 + np.random.randint(-200000, 200000, 100)
        }, index=dates)
    
    executor.set_data(sample_data)
    executor.initialize_strategies()
    
    print("✅ 策略已初始化")
    
    # 模拟几个时间步的执行
    for i in range(5):
        current_time = datetime.now() - timedelta(days=4-i)
        current_data = {}
        prices = {}
        
        for symbol in symbols:
            latest_data = sample_data[symbol].iloc[95+i]  # 获取最近的数据
            current_data[symbol] = latest_data
            prices[symbol] = latest_data['close']
        
        result = executor.execute_single_step(current_time, current_data, prices)
        print(f"第{i+1}步 - 信号数: {result['total_signals']}, 执行数: {result['executed_trades']}, 组合价值: ${result['portfolio_value']:,.2f}")
    
    # 获取因子归因分析
    if hasattr(executor.portfolio, 'get_factor_attribution'):
        attribution = executor.portfolio.get_factor_attribution()
        print(f"\n📊 因子归因分析:")
        for factor, contribution in attribution.items():
            print(f"  {factor}: {contribution:.2%}")


def demo_backtest_integration():
    """回测集成演示"""
    print("\n=== 回测集成演示 ===")
    
    # 创建回测模式执行器
    executor = create_strategy_executor(mode="backtest", initial_capital=100000)
    
    symbols = ['000001', '000002']
    
    # 添加多个不同类型的策略
    ma_short = create_ma_cross_strategy(symbols, short_window=5, long_window=20)
    ma_long = create_ma_cross_strategy(symbols, short_window=20, long_window=60)
    
    executor.add_strategy("MA_Short", ma_short, weight=0.6, strategy_type=StrategyType.TECHNICAL)
    executor.add_strategy("MA_Long", ma_long, weight=0.4, strategy_type=StrategyType.TECHNICAL)
    
    print(f"✅ 已添加回测策略: {list(executor.strategies.keys())}")
    
    # 生成回测数据
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    backtest_data = {}
    
    for symbol in symbols:
        # 生成模拟的股价走势
        np.random.seed(42)
        price_changes = np.random.normal(0.001, 0.02, len(dates))
        prices = 10 * np.exp(np.cumsum(price_changes))
        
        backtest_data[symbol] = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': 1000000 + np.random.randint(-200000, 200000, len(dates))
        }, index=dates)
    
    # 运行回测
    print("🔄 正在运行回测...")
    backtest_result = executor.run_backtest(backtest_data)
    
    print("📊 回测结果:")
    print(f"  最终组合价值: ${backtest_result['final_portfolio_value']:,.2f}")
    print(f"  总收益率: {backtest_result['total_return']:.2%}")
    print(f"  策略数量: {backtest_result['strategies_count']}")
    print(f"  总交易数: {backtest_result['total_trades']}")
    
    # 显示各策略的表现
    if backtest_result['detailed_results']:
        print("\n📈 各策略详细表现:")
        for strategy_name, result in backtest_result['detailed_results'].items():
            print(f"  {strategy_name}:")
            print(f"    收益率: {result.get('total_return_pct', 'N/A')}")
            print(f"    说明: {result.get('note', '')}")
    
    # 打印最终摘要
    executor.print_summary()


def demo_comprehensive_workflow():
    """综合工作流程演示"""
    print("\n=== 综合工作流程演示 ===")
    print("展示从策略开发到回测到实盘的完整流程")
    
    symbols = ['000001', '000858', '600519']
    
    # 1. 策略开发阶段 - 创建多种策略
    print("\n1️⃣ 策略开发阶段")
    
    # 技术策略
    ma_strategy = create_ma_cross_strategy(symbols, 20, 60)
    
    # 因子策略  
    factor_data = generate_sample_factor_data(symbols, 200)
    factor_strategy = create_factor_multi_strategy(
        symbols,
        factor_data,
        {FactorType.VALUE: 0.5, FactorType.MOMENTUM: 0.3, FactorType.QUALITY: 0.2}
    )
    
    print("✅ 策略开发完成")
    
    # 2. 回测阶段
    print("\n2️⃣ 回测验证阶段")
    
    backtest_executor = create_strategy_executor("backtest", 200000)
    backtest_executor.add_strategy("MA_Strategy", ma_strategy, 0.5, StrategyType.TECHNICAL)
    backtest_executor.add_strategy("Factor_Strategy", factor_strategy, 0.5, StrategyType.FACTOR)
    
    # 生成回测数据
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    test_data = {}
    
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)  # 每个股票使用不同的随机种子
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 年化约12.6%的收益，50%的波动率
        prices = 10 * np.exp(np.cumsum(returns))
        
        test_data[symbol] = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.003, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.008, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.008, len(dates)))),
            'close': prices,
            'volume': 1500000 + np.random.randint(-500000, 500000, len(dates))
        }, index=dates)
    
    backtest_result = backtest_executor.run_backtest(test_data)
    
    print(f"✅ 回测完成 - 总收益率: {backtest_result['total_return']:.2%}")
    
    # 3. 实盘部署准备
    print("\n3️⃣ 实盘部署准备")
    
    if backtest_result['total_return'] > 0.1:  # 如果回测收益率超过10%
        print("📈 回测表现良好，准备实盘部署")
        
        # 创建实盘执行器
        live_executor = create_strategy_executor("live", 100000)
        
        # 部署经过回测验证的策略
        live_executor.add_strategy("MA_Strategy", ma_strategy, 0.5, StrategyType.TECHNICAL)
        live_executor.add_strategy("Factor_Strategy", factor_strategy, 0.5, StrategyType.FACTOR)
        
        print("✅ 实盘执行器已配置")
        print(f"   策略数量: {len(live_executor.strategies)}")
        print(f"   初始资金: ${live_executor.initial_capital:,.2f}")
        
        # 模拟实盘运行的前几步
        print("\n🔄 模拟实盘运行...")
        
        for day in range(3):
            current_time = datetime.now() - timedelta(days=2-day)
            current_data = {}
            prices = {}
            
            # 模拟实时数据
            for symbol in symbols:
                base_price = 10 + np.random.normal(0, 2)
                current_data[symbol] = pd.Series({
                    'open': base_price * (1 + np.random.normal(0, 0.01)),
                    'high': base_price * (1 + np.abs(np.random.normal(0, 0.015))),
                    'low': base_price * (1 - np.abs(np.random.normal(0, 0.015))),
                    'close': base_price,
                    'volume': 1000000 + np.random.randint(-300000, 300000)
                })
                prices[symbol] = current_data[symbol]['close']
            
            result = live_executor.execute_single_step(current_time, current_data, prices)
            print(f"   第{day+1}天: 信号{result['total_signals']}个, 执行{result['executed_trades']}个, 价值${result['portfolio_value']:,.2f}")
        
        live_executor.print_summary()
        
    else:
        print("⚠️  回测表现不佳，需要优化策略")
    
    print("\n🎉 综合工作流程演示完成!")


def main():
    """主函数 - 运行所有演示"""
    print("Portfolio策略集成框架完整演示")
    print("=" * 60)
    
    try:
        # 基础策略执行
        demo_basic_strategy_execution()
        
        # 因子投资演示
        demo_factor_investment()
        
        # 回测集成演示
        demo_backtest_integration()
        
        # 综合工作流程演示
        demo_comprehensive_workflow()
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()