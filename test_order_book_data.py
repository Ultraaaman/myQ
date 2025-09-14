"""
测试订单簿数据功能
"""
import sys
import os
import pandas as pd

# 设置控制台编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("=== 测试订单簿数据功能 ===\n")

    # 导入相关模块
    from quantlib.market_data import (
        get_order_book, get_tick_data, get_intraday_data,
        analyze_order_book, analyze_tick_data,
        OrderBookAnalyzer, TickDataAnalyzer
    )

    test_symbol = "000001"  # 平安银行
    print(f"测试股票: {test_symbol} (平安银行)")

    # 1. 测试订单簿数据获取
    print(f"\n--- 测试订单簿数据 (五档行情) ---")
    try:
        order_book = get_order_book(test_symbol, market='CN')
        if order_book:
            print(f"✓ 成功获取订单簿数据")
            print(f"  时间戳: {order_book['timestamp']}")
            print(f"  买盘档位数: {len(order_book['bids'])}")
            print(f"  卖盘档位数: {len(order_book['asks'])}")
            print(f"  买卖价差: {order_book['spread']:.4f}")
            print(f"  中间价: {order_book['mid_price']:.4f}")

            # 显示买卖盘口信息
            if order_book['bids']:
                print(f"  最佳买价: {order_book['bids'][0]['price']:.4f} ({order_book['bids'][0]['volume']} 手)")
            if order_book['asks']:
                print(f"  最佳卖价: {order_book['asks'][0]['price']:.4f} ({order_book['asks'][0]['volume']} 手)")

            # 详细分析
            print(f"\n--- 订单簿详细分析 ---")
            analyzer = OrderBookAnalyzer(order_book)

            # 市场深度指标
            depth_metrics = analyzer.get_market_depth_metrics()
            print(f"  总买盘量: {depth_metrics['total_bid_volume']:,} 手")
            print(f"  总卖盘量: {depth_metrics['total_ask_volume']:,} 手")
            print(f"  买卖失衡: {depth_metrics['imbalance_ratio']:.2%}")

            # 价差分析
            spread_metrics = analyzer.calculate_spread_metrics()
            print(f"  相对价差: {spread_metrics['spread_relative']:.2f}%")
            print(f"  价差(基点): {spread_metrics['spread_bps']:.1f}")

            # 流动性指标
            liquidity_metrics = analyzer.calculate_liquidity_metrics()
            print(f"  流动性比率: {liquidity_metrics.get('liquidity_ratio', 0):.2f}")
            print(f"  市场冲击(1%): {liquidity_metrics.get('market_impact_1pct', 0):.1f}bp")

        else:
            print(f"✗ 未获取到订单簿数据")
    except Exception as e:
        print(f"✗ 订单簿数据测试失败: {e}")

    # 2. 测试逐笔交易数据
    print(f"\n--- 测试逐笔交易数据 ---")
    try:
        tick_data = get_tick_data(test_symbol, market='CN')
        if tick_data is not None and not tick_data.empty:
            print(f"✓ 成功获取逐笔数据: {len(tick_data)} 条记录")
            print(f"  数据列: {list(tick_data.columns)}")

            if 'time' in tick_data.columns:
                print(f"  时间范围: {tick_data['time'].min()} 到 {tick_data['time'].max()}")

            # 基本统计
            if 'volume' in tick_data.columns:
                print(f"  总成交量: {tick_data['volume'].sum():,} 手")
                print(f"  平均单笔量: {tick_data['volume'].mean():.0f} 手")

            if 'side' in tick_data.columns:
                buy_count = len(tick_data[tick_data['side'] == 'buy'])
                sell_count = len(tick_data[tick_data['side'] == 'sell'])
                print(f"  买盘笔数: {buy_count}, 卖盘笔数: {sell_count}")

            # 逐笔数据分析
            print(f"\n--- 逐笔数据分析 ---")
            tick_analyzer = TickDataAnalyzer(tick_data)

            # VWAP计算
            vwap = tick_analyzer.calculate_vwap()
            print(f"  VWAP: {vwap:.4f}")

            # 订单流分析
            order_flow = tick_analyzer.analyze_order_flow()
            print(f"  总交易笔数: {order_flow['total_trades']}")
            print(f"  买盘交易: {order_flow['buy_trades']}")
            print(f"  卖盘交易: {order_flow['sell_trades']}")
            print(f"  订单流失衡: {order_flow['order_flow_imbalance']:.2%}")

            # 大额交易检测
            large_trades = tick_analyzer.detect_large_trades()
            if not large_trades.empty:
                print(f"  大额交易数量: {len(large_trades)}")
                print(f"  最大单笔: {large_trades['volume'].max():,} 手")

        else:
            print(f"✗ 未获取到逐笔数据")
    except Exception as e:
        print(f"✗ 逐笔数据测试失败: {e}")

    # 3. 测试盘中交易明细
    print(f"\n--- 测试盘中交易明细 ---")
    try:
        intraday_data = get_intraday_data(test_symbol, market='CN')
        if intraday_data is not None and not intraday_data.empty:
            print(f"✓ 成功获取盘中数据: {len(intraday_data)} 条记录")
            print(f"  数据列: {list(intraday_data.columns)}")
        else:
            print(f"✗ 未获取到盘中数据")
    except Exception as e:
        print(f"✗ 盘中数据测试失败: {e}")

    # 4. 测试便捷分析函数
    print(f"\n--- 测试便捷分析函数 ---")
    try:
        # 订单簿分析
        print("订单簿分析:")
        ob_analysis = analyze_order_book(test_symbol)
        if ob_analysis:
            print("✓ 订单簿分析成功")
            # 显示分析报告的一部分
            report_lines = ob_analysis['report'].split('\n')[:10]
            for line in report_lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print("✗ 订单簿分析失败")

        # 逐笔数据分析
        print("\n逐笔数据分析:")
        tick_analysis = analyze_tick_data(test_symbol)
        if tick_analysis:
            print("✓ 逐笔数据分析成功")
            print(f"  VWAP: {tick_analysis['vwap']:.4f}")
            flow_analysis = tick_analysis['order_flow_analysis']
            print(f"  总交易: {flow_analysis['total_trades']} 笔")
            print(f"  买卖比: {flow_analysis['buy_trades']}:{flow_analysis['sell_trades']}")
        else:
            print("✗ 逐笔数据分析失败")

    except Exception as e:
        print(f"✗ 便捷分析函数测试失败: {e}")

    print(f"\n=== 订单簿数据功能测试完成 ===")

    # 使用说明
    print(f"\n📝 订单簿数据功能说明:")
    print("1. 数据类型:")
    print("   • 订单簿数据: get_order_book() - 五档买卖盘口")
    print("   • 逐笔数据: get_tick_data() - 每笔交易明细")
    print("   • 盘中数据: get_intraday_data() - 盘中交易明细")
    print()
    print("2. 分析功能:")
    print("   • 市场深度分析: 买卖盘量、价差、流动性")
    print("   • 订单流分析: 买卖压力、VWAP、大额交易")
    print("   • 微观结构指标: 市场冲击、价格改善等")
    print()
    print("3. 使用示例:")
    print("   from quantlib.market_data import get_order_book, analyze_order_book")
    print("   order_book = get_order_book('000001')")
    print("   analysis = analyze_order_book('000001')")
    print("   print(analysis['report'])  # 显示详细分析报告")

except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("请确保安装akshare: pip install akshare")

except Exception as e:
    print(f"✗ 运行错误: {e}")
    import traceback
    traceback.print_exc()