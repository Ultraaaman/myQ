"""
检查akshare可用的API接口
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
    print(f"AKShare版本: {ak.__version__}")
    print()

    # 检查订单簿相关API
    print("=== 检查订单簿相关API ===")
    order_book_apis = [
        'stock_bid_ask_em',
        'stock_zh_a_spot_em',
        'stock_zh_a_tick_163',
        'stock_zh_a_tick_tx',
        'stock_intraday_em',
        'stock_intraday_sina',
        'stock_zh_a_minute'
    ]

    for api_name in order_book_apis:
        if hasattr(ak, api_name):
            print(f"✓ {api_name} - 可用")
        else:
            print(f"✗ {api_name} - 不可用")

    # 尝试获取实时行情数据看看列结构
    print(f"\n=== 测试实时行情数据结构 ===")
    try:
        spot_data = ak.stock_zh_a_spot_em()
        if not spot_data.empty:
            print(f"✓ 实时行情数据获取成功")
            print(f"  数据条数: {len(spot_data)}")
            print(f"  列名: {list(spot_data.columns)}")

            # 查看是否包含盘口信息
            sample_row = spot_data.iloc[0]
            print(f"  示例数据: {sample_row.to_dict()}")
        else:
            print(f"✗ 实时行情数据为空")
    except Exception as e:
        print(f"✗ 实时行情数据获取失败: {e}")

    # 尝试获取分钟级数据
    print(f"\n=== 测试分钟级数据 ===")
    try:
        minute_data = ak.stock_zh_a_minute(symbol="sz000001", period="1", adjust="qfq")
        if not minute_data.empty:
            print(f"✓ 分钟级数据获取成功")
            print(f"  数据条数: {len(minute_data)}")
            print(f"  列名: {list(minute_data.columns)}")
            print(f"  前3行:")
            print(minute_data.head(3))
        else:
            print(f"✗ 分钟级数据为空")
    except Exception as e:
        print(f"✗ 分钟级数据获取失败: {e}")

except ImportError:
    print("AKShare未安装: pip install akshare")
except Exception as e:
    print(f"检查失败: {e}")
    import traceback
    traceback.print_exc()