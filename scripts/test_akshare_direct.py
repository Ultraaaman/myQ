#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试akshare接口
"""

import pandas as pd
from datetime import datetime, timedelta

def test_akshare_basic():
    """测试基础akshare功能"""
    print("=" * 60)
    print("测试 Akshare 基础功能")
    print("=" * 60)

    try:
        import akshare as ak
        print("✓ 成功导入 akshare")
        print(f"akshare 版本: {ak.__version__}")
    except ImportError as e:
        print(f"✗ 导入 akshare 失败: {e}")
        print("请安装: pip install akshare")
        return

    # 测试1: 获取股票基本信息
    print("\n" + "=" * 50)
    print("测试1: 获取股票基本信息")
    print("=" * 50)

    try:
        # 获取A股股票列表
        stock_list = ak.stock_info_a_code_name()
        print(f"✓ 获取A股股票列表成功，共 {len(stock_list)} 只股票")

        # 查找紫金矿业
        zjky = stock_list[stock_list['code'] == '601899']
        if not zjky.empty:
            print(f"✓ 找到紫金矿业: {zjky.iloc[0]['name']} ({zjky.iloc[0]['code']})")
        else:
            print("✗ 未找到紫金矿业 601899")

    except Exception as e:
        print(f"✗ 获取股票列表失败: {e}")

    # 测试2: 获取紫金矿业历史数据 - 使用不同的API
    print("\n" + "=" * 50)
    print("测试2: 获取紫金矿业历史数据")
    print("=" * 50)

    # 计算日期范围
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')

    print(f"获取时间范围: {start_date} 到 {end_date}")

    # 方法1: stock_zh_a_hist (推荐的新API)
    try:
        print("\n--- 方法1: ak.stock_zh_a_hist ---")
        data1 = ak.stock_zh_a_hist(symbol="601899", period="daily", start_date=start_date, end_date=end_date, adjust="")
        if data1 is not None and not data1.empty:
            print(f"✓ stock_zh_a_hist 成功，数据形状: {data1.shape}")
            print(f"列名: {list(data1.columns)}")
            print("前3行数据:")
            print(data1.head(3))
            print(f"最新3行数据:")
            print(data1.tail(3))
        else:
            print("✗ stock_zh_a_hist 返回空数据")
    except Exception as e:
        print(f"✗ stock_zh_a_hist 失败: {e}")

    # 方法2: stock_zh_a_daily (如果存在)
    try:
        print("\n--- 方法2: ak.stock_zh_a_daily ---")
        if hasattr(ak, 'stock_zh_a_daily'):
            data2 = ak.stock_zh_a_daily(symbol="sh601899", start_date=start_date, end_date=end_date)
            if data2 is not None and not data2.empty:
                print(f"✓ stock_zh_a_daily 成功，数据形状: {data2.shape}")
                print(f"列名: {list(data2.columns)}")
                print("前3行数据:")
                print(data2.head(3))
            else:
                print("✗ stock_zh_a_daily 返回空数据")
        else:
            print("× stock_zh_a_daily 方法不存在")
    except Exception as e:
        print(f"✗ stock_zh_a_daily 失败: {e}")

    # 方法3: stock_individual_info_em (实时数据)
    try:
        print("\n--- 方法3: ak.stock_individual_info_em (实时数据) ---")
        realtime_data = ak.stock_individual_info_em(symbol="601899")
        if realtime_data is not None and not realtime_data.empty:
            print(f"✓ 获取实时数据成功")
            print("实时数据:")
            print(realtime_data)
        else:
            print("✗ 获取实时数据失败")
    except Exception as e:
        print(f"✗ 获取实时数据失败: {e}")

    # 测试3: 尝试不同的时间周期参数
    print("\n" + "=" * 50)
    print("测试3: 尝试不同的时间周期参数")
    print("=" * 50)

    period_options = ["daily", "weekly", "monthly"]
    adjust_options = ["", "qfq", "hfq"]  # 不复权、前复权、后复权

    for period in period_options:
        for adjust in adjust_options:
            try:
                adjust_name = {"": "不复权", "qfq": "前复权", "hfq": "后复权"}[adjust]
                print(f"\n测试 period='{period}', adjust='{adjust}' ({adjust_name})")

                data = ak.stock_zh_a_hist(
                    symbol="601899",
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )

                if data is not None and not data.empty:
                    print(f"✓ 成功获取 {period}/{adjust_name} 数据，形状: {data.shape}")
                    print(f"  最新价格: {data.iloc[-1]['收盘'] if '收盘' in data.columns else 'N/A'}")
                else:
                    print(f"✗ {period}/{adjust_name} 返回空数据")

            except Exception as e:
                print(f"✗ {period}/{adjust_name} 失败: {e}")

    # 测试4: 检查API文档建议的用法
    print("\n" + "=" * 50)
    print("测试4: 检查最新API用法")
    print("=" * 50)

    try:
        # 最简单的调用方式
        print("使用最简单的调用方式...")
        simple_data = ak.stock_zh_a_hist(symbol="601899")
        if simple_data is not None and not simple_data.empty:
            print(f"✓ 简单调用成功，数据形状: {simple_data.shape}")
            print("最新5行数据:")
            print(simple_data.tail(5))

            # 检查数据质量
            print(f"\n数据质量检查:")
            print(f"  时间范围: {simple_data.index[0]} 到 {simple_data.index[-1]}")
            print(f"  是否有缺失值: {simple_data.isnull().any().any()}")
            if '收盘' in simple_data.columns:
                print(f"  价格范围: {simple_data['收盘'].min():.2f} - {simple_data['收盘'].max():.2f}")
        else:
            print("✗ 简单调用失败")

    except Exception as e:
        print(f"✗ 简单调用出错: {e}")

    # 测试5: 网络和数据源状态检查
    print("\n" + "=" * 50)
    print("测试5: 网络和数据源状态检查")
    print("=" * 50)

    try:
        # 测试获取指数数据，验证网络连接
        print("测试获取上证指数数据...")
        index_data = ak.stock_zh_index_daily(symbol="sh000001")
        if index_data is not None and not index_data.empty:
            print(f"✓ 网络连接正常，获取到上证指数数据: {len(index_data)} 条")
        else:
            print("✗ 网络可能有问题，无法获取指数数据")
    except Exception as e:
        print(f"✗ 网络测试失败: {e}")

def test_data_format_conversion():
    """测试数据格式转换"""
    print("\n" + "=" * 60)
    print("测试数据格式转换")
    print("=" * 60)

    try:
        import akshare as ak

        # 获取数据
        data = ak.stock_zh_a_hist(symbol="601899", period="daily")

        if data is not None and not data.empty:
            print(f"原始数据格式:")
            print(f"  索引类型: {type(data.index)}")
            print(f"  索引名称: {data.index.name}")
            print(f"  列名: {list(data.columns)}")
            print(f"  数据类型:\n{data.dtypes}")

            # 尝试标准化为OHLCV格式
            print("\n尝试标准化为OHLCV格式...")

            # 重命名列
            column_mapping = {
                '开盘': 'Open',
                '最高': 'High',
                '最低': 'Low',
                '收盘': 'Close',
                '成交量': 'Volume',
                '成交额': 'Amount'
            }

            standardized_data = data.rename(columns=column_mapping)

            # 检查是否需要重置索引
            if standardized_data.index.name == '日期':
                standardized_data = standardized_data.reset_index()
                standardized_data = standardized_data.rename(columns={'日期': 'Date'})

            print("✓ 标准化成功")
            print(f"  新列名: {list(standardized_data.columns)}")
            print("  标准化后数据预览:")
            print(standardized_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].head(3))

        else:
            print("✗ 无法获取数据进行格式转换测试")

    except Exception as e:
        print(f"✗ 格式转换测试失败: {e}")

if __name__ == "__main__":
    test_akshare_basic()
    test_data_format_conversion()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("如果以上测试大部分失败，可能的原因:")
    print("1. akshare版本过旧，请更新: pip install akshare --upgrade")
    print("2. 网络连接问题，akshare需要访问外部数据源")
    print("3. 数据源API变更，需要查看akshare最新文档")
    print("4. 某些功能需要VIP权限或有访问频率限制")