#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试market_data模块的数据获取功能
"""

import sys
import os
sys.path.append(r'D:\projects\q\myQ')

def test_market_data():
    """测试market_data模块"""
    print("测试 market_data 模块...")

    try:
        from quantlib.market_data import MarketDataManager, DataProviderFactory, get_stock_data
        print("✓ 成功导入 market_data 模块")

        # 检查支持的市场
        supported_markets = DataProviderFactory.get_supported_markets()
        print(f"✓ 支持的市场: {supported_markets}")

        # 测试1: 使用 get_stock_data 函数
        print("\n=== 测试1: 使用 get_stock_data 函数 ===")
        data1 = get_stock_data('601899', market='CN', period='1mo')
        if data1 is not None:
            print(f"✓ get_stock_data 成功，数据形状: {data1.shape}")
            print(f"列名: {list(data1.columns)}")
            print("前3行数据:")
            print(data1.head(3))
        else:
            print("✗ get_stock_data 返回 None")

        # 测试2: 使用 MarketDataManager
        print("\n=== 测试2: 使用 MarketDataManager ===")
        manager = MarketDataManager()

        # 尝试不同的市场标识符
        for market in ['CN', 'A股', 'CHINA']:
            if market in supported_markets:
                print(f"\n尝试市场: {market}")
                try:
                    data2 = manager.get_stock_data('601899', market=market, period='1mo', interval='1d')
                    if data2 is not None:
                        print(f"✓ 使用 {market} 成功获取数据，形状: {data2.shape}")
                        print(f"时间范围: {data2.index.min()} 到 {data2.index.max()}")
                        print("前3行数据:")
                        print(data2.head(3))
                        break
                    else:
                        print(f"✗ 使用 {market} 获取数据失败，返回 None")
                except Exception as e:
                    print(f"✗ 使用 {market} 出错: {e}")
            else:
                print(f"× 不支持市场: {market}")

        # 测试3: 测试你提到的调用方式
        print("\n=== 测试3: 测试你提到的调用方式 ===")
        try:
            data3 = get_stock_data('601899', market='CN', period='1day')
            if data3 is not None:
                print(f"✓ 你的调用方式成功，数据形状: {data3.shape}")
            else:
                print("✗ 你的调用方式返回 None")
        except Exception as e:
            print(f"✗ 你的调用方式出错: {e}")

        # 测试4: 尝试不同的时间周期格式
        print("\n=== 测试4: 尝试不同的时间周期格式 ===")
        period_options = ['1day', '1d', '1mo', '30d', '3mo']

        for period in period_options:
            try:
                print(f"\n测试 period='{period}':")
                data4 = get_stock_data('601899', market='CN', period=period)
                if data4 is not None:
                    print(f"✓ period='{period}' 成功，数据形状: {data4.shape}")
                else:
                    print(f"✗ period='{period}' 返回 None")
            except Exception as e:
                print(f"✗ period='{period}' 出错: {e}")

    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("请检查 quantlib.market_data 模块是否正确安装")
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_market_data()