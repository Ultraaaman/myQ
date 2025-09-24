#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于新闻评分的交易策略测试
- 测试强新闻评分买入，次日卖出的策略
- 分析第二天最高价涨幅分布
- 优化卖出价格设定
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
sys.path.append('D:/projects/q/myQ')
from quantlib.market_data import get_stock_data
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data():
    """加载新闻数据并获取对应的股票价格数据"""
    try:
        # 加载新闻情感分析数据
        news_path = "D:/projects/q/myQ/scripts/news_scores_result_1y_yanjin.csv"
        news_df = pd.read_csv(news_path, encoding='utf-8-sig')
        print(f"加载新闻数据: {len(news_df)} 条")

        # 转换日期格式
        news_df['date'] = pd.to_datetime(news_df['original_date'])

        # 判断新闻发布是否在交易时间内，并按日期聚合
        def is_trading_hours(time_str):
            """判断是否在交易时间内 (9:30-15:00)"""
            try:
                dt = pd.to_datetime(time_str)
                hour_minute = dt.hour + dt.minute / 60.0
                return 9.5 <= hour_minute <= 15.0
            except:
                return False

        # 添加交易时间标记
        news_df['is_trading_hours'] = news_df['original_date'].apply(is_trading_hours)

        # 按日期聚合新闻数据，同时记录是否有交易时间内的新闻
        daily_news = news_df.groupby(news_df['date'].dt.date).agg({
            'overall_score': ['mean', 'max', 'count'],
            'sentiment': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral',
            'direct_impact_score': 'mean',
            'indirect_impact_score': 'mean',
            'is_trading_hours': 'any'  # 只要有一条新闻在交易时间内就标记为True
        }).reset_index()

        # 展平列名
        daily_news.columns = ['date', 'score_mean', 'score_max', 'news_count',
                              'sentiment_mode', 'direct_impact_mean', 'indirect_impact_mean',
                              'has_trading_hours_news']
        daily_news['date'] = pd.to_datetime(daily_news['date'])

        print(f"按日聚合后: {len(daily_news)} 天")

        # 获取股票价格数据
        print("获取股票价格数据...")

        # 确定日期范围
        start_date = daily_news['date'].min().strftime('%Y-%m-%d')
        end_date = daily_news['date'].max().strftime('%Y-%m-%d')
        print(f"新闻数据时间范围: {start_date} 到 {end_date}")

        # 获取紫金矿业股价 (使用1年多的数据保证覆盖新闻时间范围)
        stock_data = get_stock_data("601899", market="CN", period="2y", interval="1d")

        if stock_data.empty:
            print("未获取到股价数据")
            return None

        print(f"获取股价数据: {len(stock_data)} 条")
        print(f"股价数据列名: {stock_data.columns.tolist()}")
        print(f"股价数据前3行:")
        print(stock_data.head(3))

        # 合并数据
        stock_data = stock_data.reset_index()
        stock_data['date'] = pd.to_datetime(stock_data['date'])

        # 筛选股价数据到新闻时间范围
        news_start = daily_news['date'].min()
        news_end = daily_news['date'].max()
        stock_data = stock_data[(stock_data['date'] >= news_start - pd.Timedelta(days=30)) &
                                (stock_data['date'] <= news_end + pd.Timedelta(days=10))]

        print(f"筛选后股价数据: {len(stock_data)} 条")

        # 外连接合并，确保所有新闻日期都有对应股价
        merged_df = pd.merge(daily_news, stock_data, on='date', how='left')

        print(f"合并后数据列名: {merged_df.columns.tolist()}")

        # 先检查有哪些价格相关的列，然后向前填充
        price_cols = [col for col in merged_df.columns if col in ['open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume']]
        if price_cols:
            print(f"发现价格列: {price_cols}")
            merged_df[price_cols] = merged_df[price_cols].fillna(method='ffill')
            main_price_col = price_cols[0] if price_cols else None
        else:
            # 如果没有标准列名，尝试其他可能的列名
            print("未找到标准价格列名，检查所有列...")
            main_price_col = None
            for col in merged_df.columns:
                if any(keyword in col.lower() for keyword in ['open', '开盘', 'price']):
                    main_price_col = col
                    break

        # 过滤掉没有股价数据的记录
        if main_price_col:
            merged_df = merged_df.dropna(subset=[main_price_col])
        else:
            print("未找到主要价格列，跳过过滤")

        print(f"合并后数据: {len(merged_df)} 条")

        # 计算买入卖出价格数据
        merged_df = merged_df.sort_values('date').reset_index(drop=True)

        # 动态查找价格列
        price_mapping = {}
        for std_name, possible_names in [
            ('open', ['open', 'Open', '开盘价', 'opening_price']),
            ('high', ['high', 'High', '最高价', 'highest_price']),
            ('low', ['low', 'Low', '最低价', 'lowest_price']),
            ('close', ['close', 'Close', '收盘价', 'closing_price'])
        ]:
            for col in merged_df.columns:
                if any(name in col for name in possible_names):
                    price_mapping[std_name] = col
                    break

        print(f"价格列映射: {price_mapping}")

        # 计算买入价格和卖出价格（基于新闻发布时间）
        if 'open' in price_mapping and 'close' in price_mapping and 'high' in price_mapping and 'low' in price_mapping:
            # 买入价格逻辑
            merged_df['buy_price'] = merged_df.apply(lambda row:
                row[price_mapping['close']] if row['has_trading_hours_news']
                else merged_df.loc[merged_df.index.get_loc(row.name) + 1, price_mapping['open']]
                if row.name + 1 < len(merged_df) else None, axis=1)

            # 卖出价格逻辑
            merged_df['sell_high'] = merged_df.apply(lambda row:
                row[price_mapping['high']] if row['has_trading_hours_news']
                else merged_df.loc[merged_df.index.get_loc(row.name) + 1, price_mapping['high']]
                if row.name + 1 < len(merged_df) else None, axis=1)

            merged_df['sell_low'] = merged_df.apply(lambda row:
                row[price_mapping['low']] if row['has_trading_hours_news']
                else merged_df.loc[merged_df.index.get_loc(row.name) + 1, price_mapping['low']]
                if row.name + 1 < len(merged_df) else None, axis=1)

            merged_df['sell_close'] = merged_df.apply(lambda row:
                row[price_mapping['close']] if row['has_trading_hours_news']
                else merged_df.loc[merged_df.index.get_loc(row.name) + 1, price_mapping['close']]
                if row.name + 1 < len(merged_df) else None, axis=1)

            print(f"交易时间内新闻: {merged_df['has_trading_hours_news'].sum()} 天")
            print(f"交易时间外新闻: {(~merged_df['has_trading_hours_news']).sum()} 天")
        else:
            print("缺少必要的价格列，无法计算买卖价格")

        return merged_df

    except Exception as e:
        print(f"加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_next_day_high_returns(df):
    """分析基于真实买入逻辑的涨幅分布统计"""
    print("\n=== 基于真实买入时机的涨幅分布分析 ===")

    # 计算基于买入价格的收益率
    df = df.copy()

    if 'buy_price' not in df.columns or 'sell_high' not in df.columns:
        print(f"缺少必要的价格列。可用列: {df.columns.tolist()}")
        return None

    # 计算基于真实买入价格的收益率
    df['high_return'] = (df['sell_high'] - df['buy_price']) / df['buy_price']
    df['low_return'] = (df['sell_low'] - df['buy_price']) / df['buy_price']
    df['close_return'] = (df['sell_close'] - df['buy_price']) / df['buy_price']

    # 过滤有效数据
    valid_data = df.dropna(subset=['high_return'])
    returns = valid_data['high_return'] * 100  # 转换为百分比

    print(f"样本数: {len(returns)}")
    print(f"最大涨幅: {returns.max():.2f}%")
    print(f"最小涨幅: {returns.min():.2f}%")
    print(f"平均涨幅: {returns.mean():.2f}%")
    print(f"中位数涨幅: {returns.median():.2f}%")
    print(f"标准差: {returns.std():.2f}%")

    # 分位数统计
    print("\n分位数统计:")
    for p in [10, 25, 50, 75, 80, 90, 95]:
        print(f"{p}%分位数: {np.percentile(returns, p):.2f}%")

    # 胜率统计（不同价格目标）
    print("\n不同卖出价格的胜率:")
    for target in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        win_rate = (returns >= target).mean() * 100
        print(f">{target}%涨幅胜率: {win_rate:.1f}%")

    # 可视化
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(returns, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
    plt.axvline(returns.median(), color='green', linestyle='--', label=f'Median: {returns.median():.2f}%')
    plt.xlabel('Next Day High Return (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Next Day High Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.boxplot(returns)
    plt.ylabel('Next Day High Return (%)')
    plt.title('Box Plot of Returns')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    # 累积分布
    sorted_returns = np.sort(returns)
    cumulative = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    plt.plot(sorted_returns, cumulative)
    plt.axvline(1.0, color='red', linestyle='--', label='1% Target')
    plt.axvline(2.0, color='orange', linestyle='--', label='2% Target')
    plt.xlabel('Next Day High Return (%)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    # 胜率vs目标价格
    targets = np.arange(0.1, 5.1, 0.1)
    win_rates = [(returns >= t).mean() * 100 for t in targets]
    plt.plot(targets, win_rates, 'b-', linewidth=2)
    plt.axhline(50, color='red', linestyle='--', label='50% Win Rate')
    plt.axvline(1.0, color='green', linestyle='--', label='1% Target')
    plt.xlabel('Target Return (%)')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate vs Target Return')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return returns

def test_news_score_strategy(df, score_threshold_percentile=80, target_return=1.0):
    """测试基于新闻评分的交易策略"""
    print(f"\n=== 新闻评分策略测试 (阈值: 前{100-score_threshold_percentile}%, 目标: {target_return}%) ===")

    df = df.copy()

    # 使用 score_mean 作为评分列
    score_col = 'score_mean'
    print(f"使用评分列: {score_col}")

    # 找到开盘价列
    open_col = None
    for col in df.columns:
        if col in ['open', 'Open', '开盘价']:
            open_col = col
            break

    if open_col is None:
        print(f"未找到开盘价列。可用列: {df.columns.tolist()}")
        return None

    # 确保使用基于真实买入价格的收益率
    if 'buy_price' in df.columns and 'sell_high' in df.columns:
        df['high_return'] = (df['sell_high'] - df['buy_price']) / df['buy_price'] * 100
    if 'buy_price' in df.columns and 'sell_close' in df.columns:
        df['close_return'] = (df['sell_close'] - df['buy_price']) / df['buy_price'] * 100
    if 'buy_price' in df.columns and 'sell_low' in df.columns:
        df['low_return'] = (df['sell_low'] - df['buy_price']) / df['buy_price'] * 100

    # 过滤有效数据
    valid_df = df.dropna(subset=[score_col, 'high_return'])

    # 计算阈值
    threshold = np.percentile(valid_df[score_col], score_threshold_percentile)
    print(f"评分阈值: {threshold:.3f}")

    # 筛选强评分信号
    signals = valid_df[valid_df[score_col] >= threshold].copy()

    print(f"总样本数: {len(valid_df)}")
    print(f"强评分信号数: {len(signals)}")
    print(f"信号比例: {len(signals)/len(valid_df)*100:.1f}%")

    if len(signals) == 0:
        print("没有符合条件的信号")
        return None

    # 策略统计
    high_returns = signals['high_return']
    close_returns = signals['close_return']

    print(f"\n基于最高价的收益:")
    print(f"平均收益: {high_returns.mean():.2f}%")
    print(f"胜率(>{target_return}%): {(high_returns >= target_return).mean()*100:.1f}%")
    print(f"最大收益: {high_returns.max():.2f}%")
    print(f"最小收益: {high_returns.min():.2f}%")

    print(f"\n基于收盘价的收益:")
    print(f"平均收益: {close_returns.mean():.2f}%")
    print(f"胜率(>0%): {(close_returns > 0).mean()*100:.1f}%")
    print(f"最大收益: {close_returns.max():.2f}%")
    print(f"最小收益: {close_returns.min():.2f}%")

    # 不同目标价格的胜率
    print(f"\n不同目标价格胜率:")
    for target in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        win_rate = (high_returns >= target).mean() * 100
        print(f">{target}%: {win_rate:.1f}%")

    return signals

def calculate_optimal_thresholds_and_returns(df, score_threshold_percentile=80):
    """计算最优止盈止损阈值并计算总收益率"""
    print(f"\n=== 最优阈值分析和总收益计算 ===")

    df = df.copy()

    # 确保计算了基于真实买入价格的收益率列
    if 'buy_price' in df.columns and 'sell_high' in df.columns:
        df['high_return'] = (df['sell_high'] - df['buy_price']) / df['buy_price'] * 100
    if 'buy_price' in df.columns and 'sell_low' in df.columns:
        df['low_return'] = (df['sell_low'] - df['buy_price']) / df['buy_price'] * 100
    if 'buy_price' in df.columns and 'sell_close' in df.columns:
        df['close_return'] = (df['sell_close'] - df['buy_price']) / df['buy_price'] * 100

    print(f"数据列名: {df.columns.tolist()}")

    # 使用 score_mean 作为评分列
    score_col = 'score_mean'

    # 计算评分阈值
    threshold = np.percentile(df[score_col].dropna(), score_threshold_percentile)
    signals = df[df[score_col] >= threshold].copy()

    if len(signals) == 0:
        print("没有符合条件的信号")
        return None

    print(f"强评分信号数: {len(signals)}")
    print(f"评分阈值: {threshold:.3f}")

    # 检查必要列是否存在
    if 'high_return' not in signals.columns:
        print("错误：缺少 high_return 列")
        return None

    # 计算各种收益率
    high_returns = signals['high_return'].dropna()
    low_returns = signals['low_return'].dropna() if 'low_return' in signals.columns else pd.Series([])

    print(f"\n=== 第二天涨跌幅统计 (基于开盘价买入) ===")
    print(f"最高价收益统计:")
    print(f"  平均: {high_returns.mean():.2f}%, 中位数: {high_returns.median():.2f}%")
    print(f"  最大: {high_returns.max():.2f}%, 最小: {high_returns.min():.2f}%")
    print(f"  标准差: {high_returns.std():.2f}%")

    print(f"\n最低价收益统计 (用于止损分析):")
    print(f"  平均: {low_returns.mean():.2f}%, 中位数: {low_returns.median():.2f}%")
    print(f"  最大: {low_returns.max():.2f}%, 最小: {low_returns.min():.2f}%")

    # 止盈阈值分析
    print(f"\n=== 止盈阈值分析 ===")
    profit_targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for target in profit_targets:
        win_rate = (high_returns >= target).mean() * 100
        print(f">{target:>4.1f}%止盈胜率: {win_rate:>6.1f}%")

    # 止损阈值分析
    print(f"\n=== 止损阈值分析 ===")
    stop_losses = [-1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0]
    for stop in stop_losses:
        loss_rate = (low_returns <= stop).mean() * 100
        print(f"<{stop:>5.1f}%止损触发率: {loss_rate:>6.1f}%")

    # 计算不同策略的总收益率
    print(f"\n=== 不同策略总收益率 ===")

    # 计算考虑止损的策略收益率
    def calculate_strategy_with_stop_loss(profit_target, stop_loss=-2.0):
        strategy_returns = []
        profit_count = 0
        loss_count = 0
        neutral_count = 0
        both_conditions = 0  # 既能止盈又会触发止损的交易数

        for i, row in signals.iterrows():
            high_ret = row['high_return']
            low_ret = row['low_return']

            if pd.isna(high_ret) or pd.isna(low_ret):
                continue

            # 检查是否既能止盈又会止损（这是关键！）
            can_profit = high_ret >= profit_target
            can_loss = low_ret <= stop_loss

            if can_profit and can_loss:
                both_conditions += 1

            # 策略逻辑：优先止盈，达不到止盈再考虑止损
            if high_ret >= profit_target:  # 达到止盈目标
                actual_return = profit_target
                profit_count += 1
            elif low_ret <= stop_loss:  # 触发止损
                actual_return = stop_loss
                loss_count += 1
            else:  # 正常结算：动态小止盈策略
                mini_profit = 0.3  # 0.3%小止盈目标
                if high_ret >= mini_profit:
                    actual_return = mini_profit  # 达到小止盈就卖出
                else:
                    actual_return = row.get('close_return', high_ret * 0.7)  # 收盘价或打折卖出
                neutral_count += 1

            strategy_returns.append(actual_return)

        print(f"\n{profit_target}%止盈 {stop_loss}%止损策略分析:")
        print(f"  止盈次数: {profit_count}, 止损次数: {loss_count}, 正常: {neutral_count}")
        print(f"  既能止盈又会触发止损的交易: {both_conditions} 次")

        return np.array(strategy_returns)

    # 计算动态小止盈基准策略
    def calculate_mini_profit_baseline(profit_target):
        baseline_returns = []
        for i, row in signals.iterrows():
            high_ret = row['high_return']
            close_ret = row.get('close_return', high_ret * 0.7)

            if pd.isna(high_ret):
                continue

            if high_ret >= profit_target:
                actual_return = profit_target  # 达到大止盈
            elif high_ret >= 0.3:  # 达到小止盈0.3%
                actual_return = 0.3
            else:
                actual_return = close_ret  # 收盘价

            baseline_returns.append(actual_return)
        return np.array(baseline_returns)

    # 计算买入持有不动策略
    def calculate_buy_and_hold_returns(signals, hold_days=1):
        """计算买入持有策略的收益率"""
        hold_returns = []

        for i, row in signals.iterrows():
            buy_price = row['buy_price']

            if pd.isna(buy_price):
                continue

            # 寻找持有期结束时的价格
            current_date = row['date']
            target_date = current_date + pd.Timedelta(days=hold_days)

            # 在原始数据中找到目标日期或最接近的交易日
            future_data = df[df['date'] >= target_date].head(1)

            if not future_data.empty:
                # 找到对应的收盘价列
                close_col = None
                for col in future_data.columns:
                    if 'close' in col.lower() or '收盘' in col:
                        close_col = col
                        break

                if close_col is not None:
                    sell_price = future_data.iloc[0][close_col]
                    hold_return = (sell_price - buy_price) / buy_price * 100
                    hold_returns.append(hold_return)
                else:
                    # 如果找不到收盘价，用当天收盘价代替
                    hold_return = row.get('close_return', 0)
                    hold_returns.append(hold_return)
            else:
                # 如果找不到未来数据，用当天收盘价代替
                hold_return = row.get('close_return', 0)
                hold_returns.append(hold_return)

        return np.array(hold_returns)

    # 计算整个新闻期间长期持有策略
    def calculate_long_term_hold_strategy(signals):
        """计算整个新闻采集期间的长期持有策略 - 返回单一总收益率"""
        if len(signals) == 0:
            return np.array([])

        # 找到第一次买入的价格和最后一次卖出的价格
        valid_signals = signals.dropna(subset=['buy_price'])

        if len(valid_signals) == 0:
            return np.array([])

        # 找到收盘价列
        close_col = None
        for col in df.columns:
            if 'close' in col.lower() or '收盘' in col:
                close_col = col
                break

        if close_col is None:
            return np.array([])

        # 平均成本法 - 每次强新闻都买入固定金额
        total_investment = 0
        total_shares = 0
        buy_dates = []

        for i, row in valid_signals.iterrows():
            buy_price = row['buy_price']
            investment = 1000  # 假设每次投入1000元
            shares = investment / buy_price

            total_investment += investment
            total_shares += shares
            buy_dates.append(row['date'])

        if total_shares == 0:
            return np.array([])

        avg_cost = total_investment / total_shares

        # 使用最后一个交易日的收盘价作为卖出价
        last_date = df['date'].max()
        last_close_price = df[df['date'] == last_date][close_col].iloc[0]

        # 计算长期持有总收益率
        long_term_return = (last_close_price - avg_cost) / avg_cost * 100

        print(f"\n=== 长期持有策略分析 ===")
        print(f"买入次数: {len(valid_signals)} 次")
        print(f"总投资: {total_investment:.2f} 元")
        print(f"平均买入成本: {avg_cost:.2f} 元")
        print(f"最终卖出价格: {last_close_price:.2f} 元")
        print(f"持有期: {buy_dates[0].strftime('%Y-%m-%d')} 到 {last_date.strftime('%Y-%m-%d')}")
        print(f"长期持有总收益: {long_term_return:.2f}%")

        # 返回单一总收益率，重复填充以匹配其他策略的格式
        return np.full(len(valid_signals), long_term_return)

    strategies = [
        ("理想策略(卖在最高点)", high_returns),
        ("保守策略(1%止盈,-5%止损)", calculate_strategy_with_stop_loss(1.0, -5.0)),
        ("中等策略(2%止盈,-5%止损)", calculate_strategy_with_stop_loss(2.0, -5.0)),
        ("激进策略(3%止盈,-5%止损)", calculate_strategy_with_stop_loss(3.0, -5.0)),
        ("对比-2%紧止损(2%止盈)", calculate_strategy_with_stop_loss(2.0, -2.0)),
        ("动态小止盈基准(1%止盈)", calculate_mini_profit_baseline(1.0)),
        ("动态小止盈基准(2%止盈)", calculate_mini_profit_baseline(2.0)),
        ("买入持有1天", calculate_buy_and_hold_returns(signals, 1)),
        ("买入持有3天", calculate_buy_and_hold_returns(signals, 3)),
        ("买入持有5天", calculate_buy_and_hold_returns(signals, 5)),
        ("长期持有(整个新闻期间)", calculate_long_term_hold_strategy(signals))
    ]

    for name, returns in strategies:
        if len(returns) == 0:
            continue

        # 修复：对于长期持有策略，不应该用复利计算
        if "长期持有" in name:
            # 长期持有策略的收益率就是单一总收益率，不需要复利计算
            total_return = returns.mean() / 100  # 转换为小数
        else:
            # 其他策略用复利计算（每次独立交易）
            total_return = (1 + returns/100).prod() - 1
        avg_return = returns.mean()
        win_rate = (returns > 0).mean() * 100
        max_loss = returns.min()

        print(f"\n{name}:")
        print(f"  总收益率: {total_return*100:>8.2f}%")
        print(f"  平均收益: {avg_return:>8.2f}%")
        print(f"  胜率: {win_rate:>12.1f}%")
        print(f"  最大亏损: {max_loss:>8.2f}%")
        print(f"  交易次数: {len(returns):>8d} 次")

    # 推荐阈值
    print(f"\n=== 推荐策略 ===")
    # 基于分析结果的推荐：宽松止损防范极端风险
    profit_75 = np.percentile(high_returns, 75)

    print(f"推荐止盈: {profit_75:.1f}% (75分位数)")
    print(f"推荐止损: -5.0% (防范极端风险，避免过度干扰正常交易)")

    # 使用新的推荐参数
    recommended_stop_loss = -5.0

    # 调试信息：检查数据分布
    print(f"\n调试信息:")
    print(f"最高价收益范围: {high_returns.min():.2f}% 到 {high_returns.max():.2f}%")
    print(f"最低价收益范围: {low_returns.min():.2f}% 到 {low_returns.max():.2f}%")
    print(f"能达到{profit_75:.1f}%止盈的交易: {(high_returns >= profit_75).sum()} / {len(high_returns)}")
    print(f"会触发{recommended_stop_loss:.1f}%止损的交易: {(low_returns <= recommended_stop_loss).sum()} / {len(low_returns)}")

    # 计算推荐策略的收益率
    recommended_returns = []
    profit_count = 0
    loss_count = 0
    neutral_count = 0
    for i, row in signals.iterrows():
        high_ret = row['high_return']
        low_ret = row['low_return']

        if pd.isna(high_ret) or pd.isna(low_ret):
            continue

        # 修正策略逻辑：优先检查是否能止盈，止盈优先于止损
        if high_ret >= profit_75:  # 能达到止盈目标
            actual_return = profit_75
            profit_count += 1
        elif low_ret <= recommended_stop_loss:  # 达不到止盈但触发止损
            actual_return = recommended_stop_loss
            loss_count += 1
        else:  # 既没止盈也没止损，动态小止盈策略
            mini_profit = 0.3  # 0.3%小止盈目标
            if high_ret >= mini_profit:
                actual_return = mini_profit  # 达到小止盈就卖出
            else:
                actual_return = row.get('close_return', high_ret * 0.7)  # 收盘价或打折卖出
            neutral_count += 1

        recommended_returns.append(actual_return)

    if recommended_returns:
        recommended_returns = np.array(recommended_returns)
        total_return = (1 + recommended_returns/100).prod() - 1
        avg_return = recommended_returns.mean()
        win_rate = (recommended_returns > 0).mean() * 100

        print(f"\n推荐策略表现:")
        print(f"  总收益率: {total_return*100:>8.2f}%")
        print(f"  平均收益: {avg_return:>8.2f}%")
        print(f"  胜率: {win_rate:>12.1f}%")
        print(f"  交易次数: {len(recommended_returns):>8d} 次")
        print(f"  其中: 止盈{profit_count}次, 止损{loss_count}次, 正常{neutral_count}次")

    return {
        'signals': signals,
        'high_returns': high_returns,
        'low_returns': low_returns,
        'recommended_profit_target': profit_75,
        'recommended_stop_loss': recommended_stop_loss
    }

def main():
    """主函数"""
    print("=== 基于新闻评分的交易策略测试 ===")

    # 加载数据
    df = load_and_merge_data()
    if df is None:
        return

    print(f"数据列名: {df.columns.tolist()}")
    print(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"平均新闻评分: {df['score_mean'].mean():.2f}")
    print(f"评分范围: {df['score_mean'].min():.2f} 到 {df['score_mean'].max():.2f}")

    # 分析第二天最高价涨幅分布
    returns = analyze_next_day_high_returns(df)

    # 测试不同阈值的策略
    print("\n" + "="*50)
    for percentile in [70, 80, 90, 95]:
        test_news_score_strategy(df, score_threshold_percentile=percentile, target_return=1.0)

    # 计算最优阈值和总收益率
    print("\n" + "="*60)
    result = calculate_optimal_thresholds_and_returns(df, score_threshold_percentile=80)

    print("\n=== 结论建议 ===")
    if returns is not None:
        print(f"基于{len(returns)}个样本的分析:")
        print(f"- 建议卖出目标: {np.percentile(returns, 75):.1f}% (75分位数)")
        print(f"- 保守目标: {np.percentile(returns, 60):.1f}% (60分位数)")
        print(f"- 1%目标胜率: {(returns >= 1.0).mean()*100:.1f}%")
        print(f"- 2%目标胜率: {(returns >= 2.0).mean()*100:.1f}%")

if __name__ == "__main__":
    main()