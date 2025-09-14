"""
直接测试akshare分钟级数据API
"""
import sys
import os

# 设置控制台编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import akshare as ak

    print("=== 直接测试akshare分钟级数据API ===")

    # 测试不同的股票代码格式
    test_symbols = ['sz000001', 'sh600519', 'sz000002']

    for symbol in test_symbols:
        print(f"\n--- 测试股票 {symbol} ---")
        try:
            # 获取1分钟数据
            data = ak.stock_zh_a_minute(symbol=symbol, period='1', adjust='qfq')

            if data is not None and not data.empty:
                print(f"✓ 成功获取数据，共 {len(data)} 条记录")
                print(f"  列名: {list(data.columns)}")
                print(f"  前5行:")
                print(data.head())
                print(f"  数据类型:")
                print(data.dtypes)

                # 检查索引
                print(f"  索引类型: {type(data.index)}")
                print(f"  索引名称: {data.index.name}")
                if not data.empty:
                    print(f"  第一条时间: {data.index[0] if hasattr(data.index, '__getitem__') else '无法获取'}")
                    print(f"  最后一条时间: {data.index[-1] if hasattr(data.index, '__getitem__') else '无法获取'}")
                break  # 成功获取一个就停止
            else:
                print(f"✗ 获取数据为空")

        except Exception as e:
            print(f"✗ 获取失败: {e}")

    print("\n=== 测试完成 ===")

except ImportError:
    print("✗ 未安装akshare库: pip install akshare")
except Exception as e:
    print(f"✗ 运行错误: {e}")
    import traceback
    traceback.print_exc()