#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的market_data模块
"""

import sys
sys.path.append(r'D:\projects\q\myQ')

def test_fixed_market_data():
    """测试修复后的market_data模块"""
    print("测试修复后的 market_data 模块...")

    try:
        from quantlib.market_data import MarketDataManager, DataProviderFactory, get_stock_data
        print("✓ 成功导入 market_data 模块")

        # 检查支持的市场
        supported_markets = DataProviderFactory.get_supported_markets()
        print(f"✓ 支持的市场: {supported_markets}")

        # 测试1: 使用 MarketDataManager
        print("\n=== 测试1: 使用 MarketDataManager ===")
        manager = MarketDataManager()

        # 测试不同的interval参数（之前出错的地方）
        test_cases = [
            ('CN', '1mo', '1d'),
            ('CN', '3mo', '1d'),
            ('A股', '1mo', '1d'),
            ('CHINA', '1mo', '1d'),
        ]

        for market, period, interval in test_cases:
            if market in supported_markets:
                print(f"\n测试: market='{market}', period='{period}', interval='{interval}'")
                try:
                    data = manager.get_stock_data('601899', market=market, period=period, interval=interval)
                    if data is not None and len(data) > 0:
                        print(f"✓ 成功获取数据，形状: {data.shape}")
                        print(f"  列名: {list(data.columns)}")
                        print(f"  时间范围: {data['date'].min()} 到 {data['date'].max()}" if 'date' in data.columns else "  无日期列")

                        # 检查数据格式
                        if hasattr(data, 'index') and hasattr(data.index, 'name'):
                            print(f"  索引类型: {type(data.index)}, 索引名: {data.index.name}")

                        print("  前3行:")
                        print(data.head(3))
                        break  # 成功了就不用测试其他市场了
                    else:
                        print(f"✗ 返回空数据")
                except Exception as e:
                    print(f"✗ 失败: {e}")
            else:
                print(f"× 不支持市场: {market}")

        # 测试2: 使用便捷函数
        print("\n=== 测试2: 使用便捷函数 get_stock_data ===")
        try:
            data2 = get_stock_data('601899', market='CN', period='1mo')
            if data2 is not None:
                print(f"✓ 便捷函数成功，数据形状: {data2.shape}")
                print(f"  列名: {list(data2.columns)}")
                print("  前3行:")
                print(data2.head(3))
            else:
                print("✗ 便捷函数返回 None")
        except Exception as e:
            print(f"✗ 便捷函数失败: {e}")

        # 测试3: 测试你原来失败的调用方式
        print("\n=== 测试3: 测试原来失败的调用方式 ===")
        try:
            # 这是你原来失败的调用
            data3 = get_stock_data('601899', market='CN', period='1day')
            if data3 is not None:
                print(f"✓ 原来的调用方式现在成功了，数据形状: {data3.shape}")
            else:
                print("✗ 原来的调用方式仍然失败")
        except Exception as e:
            print(f"✗ 原来的调用方式仍然出错: {e}")

        # 测试4: 测试现在应该正确工作的调用方式
        print("\n=== 测试4: 测试正确的调用方式 ===")
        correct_calls = [
            ("get_stock_data('601899', market='CN', period='1mo')", lambda: get_stock_data('601899', market='CN', period='1mo')),
            ("manager.get_stock_data('601899', market='CN', period='1mo', interval='1d')", lambda: manager.get_stock_data('601899', market='CN', period='1mo', interval='1d')),
        ]

        for desc, func in correct_calls:
            try:
                print(f"\n测试: {desc}")
                result = func()
                if result is not None:
                    print(f"✓ 成功，数据形状: {result.shape}")
                else:
                    print("✗ 返回 None")
            except Exception as e:
                print(f"✗ 失败: {e}")

    except ImportError as e:
        print(f"✗ 导入失败: {e}")
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_fixed_market_data()