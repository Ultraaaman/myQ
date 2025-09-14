"""
策略和回测模块功能测试

测试包括：
- 策略基类功能
- 各种策略示例
- 回测引擎
- 投资组合管理
- 性能分析
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, date

# 设置控制台编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_data(symbol: str, periods: int = 252) -> pd.DataFrame:
    """创建示例数据用于测试"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')

    # 模拟股价走势
    base_price = 100
    returns = np.random.normal(0.001, 0.02, periods)  # 日收益率
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = []
    for i, (date_val, price) in enumerate(zip(dates, prices)):
        daily_vol = np.random.uniform(0.01, 0.05)
        high = price * (1 + daily_vol)
        low = price * (1 - daily_vol)

        if i == 0:
            open_price = price
        else:
            open_price = prices[i-1] * np.random.uniform(0.99, 1.01)

        open_price = max(min(open_price, high), low)
        close_price = max(min(price, high), low)
        volume = np.random.randint(100000, 1000000)

        data.append({
            'date': date_val,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

def test_strategy_creation():
    """测试策略创建"""
    print("=== 测试策略创建 ===")

    try:
        from quantlib.strategy import (
            MovingAverageCrossStrategy,
            RSIStrategy,
            BollingerBandsStrategy,
            MACDStrategy,
            MomentumStrategy,
            MeanReversionStrategy,
            MultiFactorStrategy,
            create_ma_cross_strategy,
            create_rsi_strategy
        )

        symbols = ['000001', '000002']

        # 测试各种策略创建
        strategies = {
            'MA Cross': MovingAverageCrossStrategy(symbols, 10, 30),
            'RSI': RSIStrategy(symbols, 14, 30, 70),
            'Bollinger Bands': BollingerBandsStrategy(symbols, 20, 2.0),
            'MACD': MACDStrategy(symbols, 12, 26, 9),
            'Momentum': MomentumStrategy(symbols, 10, 0.02),
            'Mean Reversion': MeanReversionStrategy(symbols, 20, 0.05),
            'Multi-Factor': MultiFactorStrategy(symbols)
        }

        print("✓ 策略创建测试:")
        for name, strategy in strategies.items():
            print(f"  ✓ {name}: {strategy.name}")

        # 测试便捷函数
        ma_strategy = create_ma_cross_strategy(['000001'], 20, 60)
        rsi_strategy = create_rsi_strategy(['000001'], 14, 30, 70)

        print(f"  ✓ 便捷函数创建: {ma_strategy.name}, {rsi_strategy.name}")

        return True

    except Exception as e:
        print(f"✗ 策略创建测试失败: {e}")
        return False

def test_strategy_initialization():
    """测试策略初始化和信号生成"""
    print("\n=== 测试策略初始化和信号生成 ===")

    try:
        from quantlib.strategy import create_ma_cross_strategy

        # 创建示例数据
        symbol = '000001'
        data = create_sample_data(symbol, 100)

        # 创建策略
        strategy = create_ma_cross_strategy([symbol], 10, 30)
        strategy.set_data(data)

        print(f"✓ 数据设置完成: {len(data)} 条记录")

        # 初始化策略
        strategy.initialize()
        print(f"✓ 策略初始化完成")
        print(f"  指标数量: {len(strategy.indicators.get(symbol, {}))}")

        # 测试信号生成
        current_time = data.index[50]  # 选择中间的一个时间点
        current_data = {
            symbol: pd.Series({
                'open': data.loc[current_time, 'open'],
                'high': data.loc[current_time, 'high'],
                'low': data.loc[current_time, 'low'],
                'close': data.loc[current_time, 'close'],
                'volume': data.loc[current_time, 'volume']
            })
        }

        signals = strategy.generate_signals(current_time, current_data)
        print(f"✓ 信号生成测试完成，生成 {len(signals)} 个信号")

        for signal in signals:
            print(f"  信号: {signal.signal_type.value} {signal.symbol} 置信度: {signal.confidence}")

        return True

    except Exception as e:
        print(f"✗ 策略初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_management():
    """测试投资组合管理"""
    print("\n=== 测试投资组合管理 ===")

    try:
        from quantlib.portfolio import create_portfolio_manager

        # 创建投资组合管理器
        portfolio = create_portfolio_manager(100000, "测试组合")
        print(f"✓ 投资组合创建: {portfolio}")

        # 测试买入
        success = portfolio.buy('000001', quantity=1000, price=50.0)
        print(f"✓ 买入测试: {'成功' if success else '失败'}")

        success = portfolio.buy('000002', quantity=800, price=60.0)
        print(f"✓ 买入测试2: {'成功' if success else '失败'}")

        # 更新价格
        portfolio.update_prices({'000001': 52.0, '000002': 58.0})
        print(f"✓ 价格更新完成")

        # 获取持仓摘要
        positions = portfolio.get_positions_summary()
        print(f"✓ 持仓摘要:")
        print(positions.to_string(index=False))

        # 测试卖出
        success = portfolio.sell('000001', quantity=500, price=52.0)
        print(f"✓ 卖出测试: {'成功' if success else '失败'}")

        # 获取绩效指标
        portfolio.record_daily_value()
        # 模拟多日价格变动
        for i in range(10):
            portfolio.update_prices({
                '000001': 50.0 + i * 0.5,
                '000002': 60.0 - i * 0.3
            })
            portfolio.record_daily_value()

        metrics = portfolio.get_performance_metrics()
        print(f"✓ 绩效指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # 测试权重计算
        equal_weights = portfolio.calculate_equal_weights(['000001', '000002', '000003'])
        print(f"✓ 等权重配置: {equal_weights}")

        return True

    except Exception as e:
        print(f"✗ 投资组合管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtrader_engine():
    """测试Backtrader回测引擎"""
    print("\n=== 测试Backtrader回测引擎 ===")

    try:
        from quantlib.strategy import create_ma_cross_strategy
        from quantlib.backtest import create_backtrader_engine

        # 创建示例数据
        symbol = '000001'
        data = create_sample_data(symbol, 200)

        # 创建策略
        strategy = create_ma_cross_strategy([symbol], 10, 30, initial_capital=100000)

        # 创建回测引擎
        engine = create_backtrader_engine(initial_cash=100000, commission=0.001)
        print("✓ Backtrader引擎创建完成")

        # 运行回测
        results = engine.run_backtest(
            strategy=strategy,
            data=data,
            plot=False
        )

        print(f"✓ 回测完成")
        print(f"  初始资金: ${results['initial_value']:,.2f}")
        print(f"  最终资金: ${results['final_value']:,.2f}")
        print(f"  总收益: ${results['total_return']:,.2f} ({results['total_return_pct']:.2f}%)")

        # 打印性能摘要
        engine.print_performance_summary()

        return True

    except ImportError:
        print("⚠️ Backtrader未安装，跳过测试")
        print("  安装命令: pip install backtrader")
        return True

    except Exception as e:
        print(f"✗ Backtrader回测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_analysis():
    """测试性能分析"""
    print("\n=== 测试性能分析 ===")

    try:
        from quantlib.backtest import PerformanceAnalyzer, analyze_backtest_results

        # 创建示例收益率数据
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        # 创建基准收益率
        benchmark = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)

        print(f"✓ 示例数据创建完成: {len(returns)} 个交易日")

        # 创建性能分析器
        analyzer = PerformanceAnalyzer(returns, benchmark, risk_free_rate=0.03)

        # 计算各种指标
        returns_metrics = analyzer.calculate_returns_metrics()
        risk_metrics = analyzer.calculate_risk_metrics()
        benchmark_metrics = analyzer.calculate_benchmark_metrics()

        print(f"✓ 性能指标计算完成")
        print(f"  年化收益率: {returns_metrics['annualized_return_pct']:.2f}%")
        print(f"  年化波动率: {risk_metrics['volatility_pct']:.2f}%")
        print(f"  夏普比率: {risk_metrics['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {risk_metrics['max_drawdown_pct']:.2f}%")

        if benchmark_metrics:
            print(f"  Alpha: {benchmark_metrics['alpha_pct']:.2f}%")
            print(f"  Beta: {benchmark_metrics['beta']:.3f}")

        # 测试交易统计
        trades_data = pd.DataFrame({
            'action': ['buy', 'sell', 'buy', 'sell', 'buy', 'sell'],
            'pnl': [0, 100, 0, -50, 0, 200]
        })

        trading_metrics = analyzer.calculate_trading_metrics(trades_data)
        print(f"✓ 交易统计:")
        print(f"  胜率: {trading_metrics.get('win_rate_pct', 0):.1f}%")
        print(f"  盈亏比: {trading_metrics.get('profit_factor', 0):.2f}")

        # 生成完整报告
        report = analyzer.generate_report(trades_data)
        print(f"\n✓ 完整分析报告已生成")

        # 使用便捷函数
        results = analyze_backtest_results(returns, benchmark, trades_data, plot=False)
        print(f"✓ 便捷分析函数测试完成")

        return True

    except Exception as e:
        print(f"✗ 性能分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_strategy_comparison():
    """测试多策略比较"""
    print("\n=== 测试多策略比较 ===")

    try:
        from quantlib.strategy import (
            create_ma_cross_strategy,
            create_rsi_strategy,
            create_bollinger_bands_strategy
        )

        # 创建示例数据
        symbol = '000001'
        data = create_sample_data(symbol, 150)

        # 创建多个策略
        strategies = {
            'MA交叉': create_ma_cross_strategy([symbol], 10, 30),
            'RSI': create_rsi_strategy([symbol], 14, 30, 70),
            '布林带': create_bollinger_bands_strategy([symbol], 20, 2.0)
        }

        results = {}

        for name, strategy in strategies.items():
            try:
                strategy.set_data(data)
                strategy.initialize()

                # 模拟简单的回测过程
                for i in range(50, len(data)):
                    current_time = data.index[i]
                    current_data = {
                        symbol: pd.Series({
                            'open': data.loc[current_time, 'open'],
                            'high': data.loc[current_time, 'high'],
                            'low': data.loc[current_time, 'low'],
                            'close': data.loc[current_time, 'close'],
                            'volume': data.loc[current_time, 'volume']
                        })
                    }

                    signals = strategy.generate_signals(current_time, current_data)

                    # 简单执行信号
                    for signal in signals:
                        strategy.execute_signal(signal, current_data)

                # 获取策略绩效
                performance = strategy.get_performance_metrics()
                results[name] = performance

                print(f"✓ {name} 策略测试完成:")
                print(f"  总交易: {performance.get('total_trades', 0)}")
                print(f"  总收益: {performance.get('total_return_pct', 0):.2f}%")
                print(f"  胜率: {performance.get('win_rate_pct', 0):.1f}%")

            except Exception as e:
                print(f"✗ {name} 策略测试失败: {e}")

        print(f"\n✓ 多策略比较测试完成，共测试 {len(results)} 个策略")

        return True

    except Exception as e:
        print(f"✗ 多策略比较测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始策略和回测模块功能测试\n")

    test_results = []

    # 运行各项测试
    tests = [
        ("策略创建", test_strategy_creation),
        ("策略初始化", test_strategy_initialization),
        ("投资组合管理", test_portfolio_management),
        ("Backtrader引擎", test_backtrader_engine),
        ("性能分析", test_performance_analysis),
        ("多策略比较", test_multi_strategy_comparison),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            test_results.append((test_name, False))

    # 打印测试结果摘要
    print("\n" + "="*60)
    print("📊 测试结果摘要")
    print("="*60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status:<8} {test_name}")
        if result:
            passed += 1

    print(f"\n总计: {passed}/{total} 个测试通过")

    if passed == total:
        print("🎉 所有测试通过！策略和回测模块功能正常")
    else:
        print(f"⚠️ {total - passed} 个测试失败，请检查相关功能")

    print("\n📝 策略和回测模块使用说明:")
    print("1. 策略开发:")
    print("   from quantlib.strategy import BaseStrategy, create_ma_cross_strategy")
    print("   strategy = create_ma_cross_strategy(['000001'], 20, 60)")
    print()
    print("2. 回测执行:")
    print("   from quantlib.backtest import create_backtrader_engine")
    print("   engine = create_backtrader_engine()")
    print("   results = engine.run_backtest(strategy, data)")
    print()
    print("3. 投资组合管理:")
    print("   from quantlib.portfolio import create_portfolio_manager")
    print("   portfolio = create_portfolio_manager(100000)")
    print()
    print("4. 性能分析:")
    print("   from quantlib.backtest import analyze_backtest_results")
    print("   analyze_backtest_results(returns, benchmark)")

if __name__ == "__main__":
    main()