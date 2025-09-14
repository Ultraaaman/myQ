"""
测试统一接口的分钟级数据功能
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
    print("=== 测试统一market_data接口的分钟级数据功能 ===\n")

    # 测试统一接口
    from quantlib.market_data import get_stock_data, get_a_share_minute_data, get_multiple_a_share_minute_data

    test_symbol = "000001"  # 平安银行
    print(f"测试股票: {test_symbol} (平安银行)")

    # 1. 测试统一接口获取分钟级数据
    print(f"\n--- 测试统一接口 get_stock_data ---")
    try:
        data = get_stock_data(test_symbol, market='CN', interval='1min')
        if data is not None and not data.empty:
            print(f"✓ 成功获取1分钟数据: {len(data)} 条记录")
            print(f"  时间范围: {data['date'].min()} 到 {data['date'].max()}")
            print(f"  数据列: {list(data.columns)}")
            print(f"  最新价格: {data['close'].iloc[-1]:.2f}")
        else:
            print(f"✗ 未获取到数据")
    except Exception as e:
        print(f"✗ 获取失败: {e}")

    # 2. 测试专用分钟级函数
    print(f"\n--- 测试专用函数 get_a_share_minute_data ---")
    try:
        data = get_a_share_minute_data(test_symbol, interval='1min')
        if data is not None and not data.empty:
            print(f"✓ 成功获取1分钟数据: {len(data)} 条记录")
            print(f"  时间范围: {data['date'].min()} 到 {data['date'].max()}")
        else:
            print(f"✗ 未获取到数据")
    except Exception as e:
        print(f"✗ 获取失败: {e}")

    # 3. 测试批量获取
    print(f"\n--- 测试批量获取 get_multiple_a_share_minute_data ---")
    test_symbols = ["000001", "000002"]
    try:
        batch_data = get_multiple_a_share_minute_data(test_symbols, interval="5min")
        print(f"✓ 批量获取完成")
        for symbol, data in batch_data.items():
            if data is not None and not data.empty:
                print(f"  {symbol}: {len(data)} 条5分钟数据")
            else:
                print(f"  {symbol}: 无数据")
    except Exception as e:
        print(f"✗ 批量获取失败: {e}")

    # 4. 测试向后兼容性
    print(f"\n--- 测试向后兼容性 (technical模块) ---")
    try:
        from quantlib.technical import get_a_share_minute_data as tech_minute_data
        data = tech_minute_data(test_symbol, interval='5min')
        if data is not None and not data.empty:
            print(f"✓ technical模块兼容性正常: {len(data)} 条记录")
        else:
            print(f"✗ technical模块返回空数据")
    except Exception as e:
        print(f"✗ technical模块兼容性测试失败: {e}")

    # 5. 测试notebook中的用法
    print(f"\n--- 测试Notebook兼容性 ---")
    print("现在在notebook中可以直接使用:")
    print("  from quantlib.market_data import get_stock_data")
    print("  data = get_stock_data(symbol, market='CN', interval='5min')")
    print()
    print("或者使用专用函数:")
    print("  from quantlib.market_data import get_a_share_minute_data")
    print("  data = get_a_share_minute_data(symbol, interval='5min')")

    print(f"\n=== 统一接口分钟级数据测试完成 ===")

    # 使用说明
    print(f"\n📝 统一接口使用说明:")
    print("1. 主要接口: quantlib.market_data.get_stock_data(symbol, market='CN', interval='5min')")
    print("2. 专用函数: quantlib.market_data.get_a_share_minute_data(symbol, interval='5min')")
    print("3. 批量获取: quantlib.market_data.get_multiple_a_share_minute_data(symbols, interval='5min')")
    print("4. 支持周期: 1min, 5min, 15min, 30min, 60min (或简写 1m, 5m, 15m, 30m, 60m)")
    print("5. 向后兼容: quantlib.technical模块仍可使用，但会显示弃用警告")

except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("请确保安装akshare: pip install akshare")

except Exception as e:
    print(f"✗ 运行错误: {e}")
    import traceback
    traceback.print_exc()