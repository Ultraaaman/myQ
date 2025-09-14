"""
测试A股分钟级数据获取功能
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
    from quantlib.technical import get_a_share_minute_data, get_multiple_a_share_minute_data, get_a_share_data

    print("=== 测试A股分钟级数据获取 ===\n")

    # 测试单只股票的分钟级数据
    test_symbol = "000001"  # 平安银行
    print(f"测试股票: {test_symbol} (平安银行)")

    # 只测试一个间隔以减少输出
    for interval in ["1min"]:
        print(f"\n--- 测试 {interval} 数据 ---")
        try:
            data = get_a_share_minute_data(test_symbol, interval=interval)
            if data is not None and not data.empty:
                print(f"✓ 成功获取 {interval} 数据")
                print(f"  数据条数: {len(data)}")
                print(f"  时间范围: {data['date'].min()} 到 {data['date'].max()}")
                print(f"  数据列: {list(data.columns)}")
                print(f"  最新价格: {data['close'].iloc[-1]:.2f}")
            else:
                print(f"✗ 未获取到 {interval} 数据")
        except Exception as e:
            print(f"✗ 获取 {interval} 数据失败: {e}")

    # 测试日线数据对比
    print(f"\n--- 对比日线数据 ---")
    try:
        daily_data = get_a_share_data(test_symbol, period="1mo")  # 获取1个月日线数据
        if daily_data is not None and not daily_data.empty:
            print(f"✓ 成功获取日线数据")
            print(f"  日线数据条数: {len(daily_data)}")
            print(f"  日线时间范围: {daily_data['date'].min()} 到 {daily_data['date'].max()}")
            print(f"  最新收盘价: {daily_data['close'].iloc[-1]:.2f}")
        else:
            print(f"✗ 未获取到日线数据")
    except Exception as e:
        print(f"✗ 获取日线数据失败: {e}")

    # 测试批量获取分钟级数据
    print(f"\n--- 测试批量获取分钟级数据 ---")
    test_symbols = ["000001", "000002", "600519"]  # 平安银行, 万科A, 贵州茅台
    try:
        batch_data = get_multiple_a_share_minute_data(test_symbols, interval="5min")
        print(f"✓ 批量获取完成")
        for symbol, data in batch_data.items():
            if data is not None:
                print(f"  {symbol}: {len(data)} 条5分钟数据")
            else:
                print(f"  {symbol}: 无数据")
    except Exception as e:
        print(f"✗ 批量获取失败: {e}")

    # 测试notebook兼容性
    print(f"\n--- 测试Notebook兼容性 ---")
    print("为了在notebook中使用分钟级数据，建议修改代码如下：")
    print()
    print("替换这行代码：")
    print("  data = get_stock_data(symbol, market='CN', period=period, interval='1m')")
    print()
    print("改为：")
    print("  from quantlib.technical import get_a_share_minute_data")
    print("  data = get_a_share_minute_data(symbol, interval='1min')")
    print()
    print("或者批量获取：")
    print("  from quantlib.technical import get_multiple_a_share_minute_data")
    print("  stock_data = get_multiple_a_share_minute_data(list(stocks.keys()), interval='5min')")

    print(f"\n=== 分钟级数据功能测试完成 ===")

    # 使用说明
    print(f"\n📝 使用说明:")
    print("1. 分钟级数据支持周期: 1min, 5min, 15min, 30min, 60min")
    print("2. 分钟级数据受限于akshare，只能获取近5个交易日的数据")
    print("3. 数据包含开、高、低、收、量等字段")
    print("4. 示例代码:")
    print("   from quantlib.technical import get_a_share_minute_data")
    print("   data = get_a_share_minute_data('000001', interval='5min')")

except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("请确保安装akshare: pip install akshare")

except Exception as e:
    print(f"✗ 运行错误: {e}")
    import traceback
    traceback.print_exc()