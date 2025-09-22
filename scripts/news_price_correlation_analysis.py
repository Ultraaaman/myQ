#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻评分与未来收益率相关性分析
==============================

本脚本分析新闻情感评分与未来股票收益率的相关性，包括：
1. 读取新闻评分数据
2. 获取对应时间的股价数据
3. 计算未来1天、2天、3天、5天收益率
4. 数据对齐和聚合
5. 可视化分析
6. 相关性统计分析

主要改进：
- 重点关注未来收益率而非当日股价变化
- 分析不同时间周期的预测效果
- 提供基于统计显著性的投资建议

作者: Claude Code
日期: 2025-09-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os
from scipy.stats import pearsonr, spearmanr
from scipy import stats

# 添加项目路径
sys.path.append(r'D:\projects\q\myQ')

# 导入 market_data 模块
try:
    from quantlib.market_data import get_stock_data, MarketDataManager
    MARKET_DATA_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入 quantlib.market_data 模块: {e}")
    print("将使用模拟数据进行演示...")
    MARKET_DATA_AVAILABLE = False

# 设置图表样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def load_news_data(file_path):
    """
    读取新闻评分数据

    Args:
        file_path (str): 新闻评分CSV文件路径

    Returns:
        pd.DataFrame: 处理后的新闻数据
    """
    print("=" * 50)
    print("步骤 1: 读取新闻评分数据")
    print("=" * 50)

    try:
        news_df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"✓ 成功读取数据，形状: {news_df.shape}")

        # 转换时间格式
        news_df['date'] = pd.to_datetime(news_df['original_date'])
        news_df = news_df.sort_values('date')

        print(f"✓ 时间范围: {news_df['date'].min().date()} 到 {news_df['date'].max().date()}")

        # 显示更详细的统计信息
        print(f"✓ 数据质量检查:")
        print(f"  - overall_score 范围: {news_df['overall_score'].min()} 到 {news_df['overall_score'].max()}")
        print(f"  - overall_score 均值: {news_df['overall_score'].mean():.2f}")

        if 'sentiment' in news_df.columns:
            print(f"  - 情绪分布: {dict(news_df['sentiment'].value_counts().head(3))}")

        if 'certainty' in news_df.columns:
            print(f"  - 确定性均值: {news_df['certainty'].mean():.2f}")

        if 'action_suggestion' in news_df.columns:
            print(f"  - 行动建议分布: {dict(news_df['action_suggestion'].value_counts().head(3))}")

        return news_df

    except Exception as e:
        print(f"✗ 读取新闻数据失败: {e}")
        return None

def aggregate_daily_scores(news_df):
    """
    按日期聚合新闻评分（增强版）

    Args:
        news_df (pd.DataFrame): 原始新闻数据

    Returns:
        pd.DataFrame: 日度聚合的评分数据
    """
    print("\n" + "=" * 50)
    print("步骤 2: 按日期聚合新闻评分")
    print("=" * 50)

    # 构建聚合字典
    agg_dict = {
        'overall_score': ['mean', 'sum', 'count', 'std', 'min', 'max'],
        'sentiment': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral'
    }

    # 添加其他可用列的聚合
    if 'direct_impact_score' in news_df.columns:
        agg_dict['direct_impact_score'] = ['mean', 'std']

    if 'indirect_impact_score' in news_df.columns:
        agg_dict['indirect_impact_score'] = ['mean', 'std']

    if 'certainty' in news_df.columns:
        agg_dict['certainty'] = ['mean', 'std']

    if 'action_suggestion' in news_df.columns:
        agg_dict['action_suggestion'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral'

    # 按日期聚合
    daily_scores = news_df.groupby(news_df['date'].dt.date).agg(agg_dict).round(3)

    # 展平列名
    new_columns = []
    for col in daily_scores.columns:
        if isinstance(col, tuple):
            if col[1] == '<lambda>':
                new_columns.append(f"dominant_{col[0]}")
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col)

    daily_scores.columns = new_columns
    daily_scores.reset_index(inplace=True)
    daily_scores['date'] = pd.to_datetime(daily_scores['date'])

    # 填充缺失值
    numeric_cols = daily_scores.select_dtypes(include=[np.number]).columns
    daily_scores[numeric_cols] = daily_scores[numeric_cols].fillna(0)

    # 计算组合指标
    if 'direct_impact_score_mean' in daily_scores.columns and 'indirect_impact_score_mean' in daily_scores.columns:
        daily_scores['combined_impact'] = (daily_scores['direct_impact_score_mean'] + daily_scores['indirect_impact_score_mean']) / 2

    if 'certainty_mean' in daily_scores.columns:
        # 确定性加权的综合评分
        daily_scores['weighted_score'] = daily_scores['overall_score_mean'] * daily_scores['certainty_mean']

    print(f"✓ 聚合完成，共 {len(daily_scores)} 个交易日")
    print(f"✓ 聚合维度: {len(daily_scores.columns)} 个指标")
    print(f"✓ 主要指标:")

    key_metrics = ['overall_score_mean', 'overall_score_std', 'news_count', 'dominant_sentiment']
    available_metrics = [col for col in key_metrics if col in daily_scores.columns]

    if available_metrics:
        print(daily_scores[['date'] + available_metrics].head())

    return daily_scores

def get_stock_price_data(daily_scores):
    """
    获取股价数据

    Args:
        daily_scores (pd.DataFrame): 日度新闻评分数据

    Returns:
        pd.DataFrame: 股价数据
    """
    print("\n" + "=" * 50)
    print("步骤 3: 获取股价数据")
    print("=" * 50)

    if MARKET_DATA_AVAILABLE:
        try:
            from quantlib.market_data import DataProviderFactory

            # 检查支持的市场
            supported_markets = DataProviderFactory.get_supported_markets()
            print(f"支持的市场: {supported_markets}")

            # 初始化数据管理器
            data_manager = MarketDataManager()

            # 扩展时间范围
            start_date = daily_scores['date'].min() - timedelta(days=5)
            end_date = daily_scores['date'].max() + timedelta(days=5)

            print(f"获取股价数据时间范围: {start_date.date()} 到 {end_date.date()}")

            # 尝试不同的市场标识符和时间周期获取紫金矿业(601899)股价数据
            market_options = ['CN', 'A股', 'CHINA']
            period_options = ['3mo', '1mo', '2mo', '90d', '60d']
            stock_data = None

            for market in market_options:
                if market in supported_markets:
                    print(f"尝试使用市场标识符: {market}")
                    for period in period_options:
                        try:
                            print(f"  - 尝试时间周期: {period}")
                            stock_data = data_manager.get_stock_data('601899', market=market, period=period, interval='1d')
                            if stock_data is not None and len(stock_data) > 0:
                                print(f"✓ 使用 {market}, period={period} 成功获取股价数据，形状: {stock_data.shape}")
                                break
                        except Exception as e:
                            print(f"    × period={period} 失败: {e}")
                            continue

                    if stock_data is not None and len(stock_data) > 0:
                        break
                else:
                    print(f"× 不支持市场标识符: {market}")

            # 如果上述方法都失败，尝试使用便捷函数
            if stock_data is None:
                try:
                    from quantlib.market_data import get_stock_data
                    print("尝试使用便捷函数 get_stock_data...")
                    stock_data = get_stock_data('601899', market='CN', period='1mo')
                    if stock_data is not None:
                        print(f"✓ 便捷函数成功获取数据，形状: {stock_data.shape}")
                except Exception as e:
                    print(f"× 便捷函数也失败: {e}")

            if stock_data is not None and len(stock_data) > 0:
                # 根据akshare测试结果，处理数据格式
                stock_data_clean = stock_data.copy()

                # 检查是否已经有date列
                if 'date' not in stock_data_clean.columns:
                    # 如果没有date列，检查索引或其他日期列
                    if hasattr(stock_data_clean.index, 'date'):
                        stock_data_clean['date'] = pd.to_datetime(stock_data_clean.index.date)
                    elif '日期' in stock_data_clean.columns:
                        stock_data_clean['date'] = pd.to_datetime(stock_data_clean['日期'])
                    else:
                        # 假设索引就是日期
                        stock_data_clean['date'] = pd.to_datetime(stock_data_clean.index)
                else:
                    # 确保date列是datetime格式
                    stock_data_clean['date'] = pd.to_datetime(stock_data_clean['date'])

                # 确保有必要的OHLCV列（处理中英文列名）
                column_mapping = {
                    '开盘': 'Open',
                    '最高': 'High',
                    '最低': 'Low',
                    '收盘': 'Close',
                    '成交量': 'Volume',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }

                for old_col, new_col in column_mapping.items():
                    if old_col in stock_data_clean.columns and new_col not in stock_data_clean.columns:
                        stock_data_clean[new_col] = stock_data_clean[old_col]

                print(f"✓ 数据处理完成，时间范围: {stock_data_clean['date'].min().date()} 到 {stock_data_clean['date'].max().date()}")
                print(f"✓ 数据列: {[col for col in stock_data_clean.columns if col in ['date', 'Open', 'High', 'Low', 'Close', 'Volume']]}")
                return stock_data_clean
            else:
                print("✗ 所有市场标识符都获取数据失败，使用模拟数据")
                return create_simulated_stock_data(daily_scores)

        except Exception as e:
            print(f"✗ 获取股价数据时出错: {e}")
            print("使用模拟数据进行演示...")
            return create_simulated_stock_data(daily_scores)
    else:
        print("quantlib.market_data 模块不可用，使用模拟数据进行演示...")
        return create_simulated_stock_data(daily_scores)

def create_simulated_stock_data(daily_scores):
    """
    创建模拟股价数据用于演示

    Args:
        daily_scores (pd.DataFrame): 日度新闻评分数据

    Returns:
        pd.DataFrame: 模拟股价数据
    """
    print("创建模拟股价数据...")

    np.random.seed(42)
    dates = pd.date_range(
        start=daily_scores['date'].min() - timedelta(days=2),
        end=daily_scores['date'].max() + timedelta(days=2),
        freq='D'
    )

    simulated_prices = []
    base_price = 25.0  # 紫金矿业大概价格

    for i, date in enumerate(dates):
        if i == 0:
            price = base_price
        else:
            # 随机游走 + 一些趋势
            change = np.random.normal(0, 0.015)  # 1.5%的日波动
            price = max(simulated_prices[-1] * (1 + change), 1.0)  # 确保价格大于0
        simulated_prices.append(price)

    # 创建完整的OHLCV数据
    stock_data_sim = pd.DataFrame({
        'date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in simulated_prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in simulated_prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in simulated_prices],
        'Close': simulated_prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })

    # 确保OHLC关系正确
    for i in range(len(stock_data_sim)):
        high = max(stock_data_sim.loc[i, 'Open'], stock_data_sim.loc[i, 'Close'], stock_data_sim.loc[i, 'High'])
        low = min(stock_data_sim.loc[i, 'Open'], stock_data_sim.loc[i, 'Close'], stock_data_sim.loc[i, 'Low'])
        stock_data_sim.loc[i, 'High'] = high
        stock_data_sim.loc[i, 'Low'] = low

    print(f"✓ 模拟数据创建完成，形状: {stock_data_sim.shape}")
    return stock_data_sim

def merge_data(daily_scores, stock_data):
    """
    合并新闻评分和股价数据

    Args:
        daily_scores (pd.DataFrame): 日度新闻评分数据
        stock_data (pd.DataFrame): 股价数据

    Returns:
        pd.DataFrame: 合并后的数据
    """
    print("\n" + "=" * 50)
    print("步骤 4: 数据对齐和合并")
    print("=" * 50)

    # 确定要合并的股价列
    stock_columns = ['date']
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in stock_data.columns:
            stock_columns.append(col)

    print(f"✓ 准备合并的股价列: {stock_columns}")

    # 合并数据
    merged_data = pd.merge(
        daily_scores,
        stock_data[stock_columns],
        on='date',
        how='inner'
    )

    print(f"✓ 合并后数据形状: {merged_data.shape}")

    if len(merged_data) > 0:
        # 计算价格变化率和未来收益率
        merged_data = merged_data.sort_values('date').reset_index(drop=True)
        merged_data['price_change'] = merged_data['Close'].pct_change() * 100

        # 计算未来1天、2天、3天、5天的收益率
        merged_data['future_return_1d'] = (merged_data['Close'].shift(-1) / merged_data['Close'] - 1) * 100
        merged_data['future_return_2d'] = (merged_data['Close'].shift(-2) / merged_data['Close'] - 1) * 100
        merged_data['future_return_3d'] = (merged_data['Close'].shift(-3) / merged_data['Close'] - 1) * 100
        merged_data['future_return_5d'] = (merged_data['Close'].shift(-5) / merged_data['Close'] - 1) * 100

        # 保留原有的price_change_next以向后兼容
        merged_data['price_change_next'] = merged_data['future_return_1d']
        merged_data['volume_change'] = merged_data['Volume'].pct_change() * 100

        print(f"✓ 时间范围: {merged_data['date'].min().date()} 到 {merged_data['date'].max().date()}")
        print("\n合并数据预览:")

        # 动态确定评分列名
        score_col = 'overall_score_mean' if 'overall_score_mean' in merged_data.columns else 'score_mean'
        preview_cols = ['date', score_col, 'Close', 'future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_5d']
        available_cols = [col for col in preview_cols if col in merged_data.columns]

        if len(available_cols) >= 4:
            print("包含未来收益率的数据预览:")
            print(merged_data[available_cols].head())
        else:
            print("列名不匹配，显示所有可用列:")
            print(f"可用列: {list(merged_data.columns)}")
            if 'date' in merged_data.columns and 'Close' in merged_data.columns:
                print(merged_data[['date', 'Close'] + [col for col in ['future_return_1d', 'future_return_2d'] if col in merged_data.columns]].head())

    return merged_data

def plot_time_series(merged_data):
    """
    绘制增强的时间序列图

    Args:
        merged_data (pd.DataFrame): 合并后的数据
    """
    print("\n" + "=" * 50)
    print("步骤 5: 绘制时间序列分析图")
    print("=" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Overall Score Time Series
    ax1 = axes[0, 0]
    ax1.plot(merged_data['date'], merged_data['overall_score_mean'],
             'b-o', linewidth=2.5, markersize=6, alpha=0.8, label='Overall Score')

    # Add standard deviation fill
    if 'overall_score_std' in merged_data.columns:
        ax1.fill_between(merged_data['date'],
                         merged_data['overall_score_mean'] - merged_data['overall_score_std'],
                         merged_data['overall_score_mean'] + merged_data['overall_score_std'],
                         alpha=0.2, color='blue', label='±1 Std Dev')

    ax1.set_title('Overall Score Time Series', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Overall Score', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Direct vs Indirect Impact
    ax2 = axes[0, 1]
    if 'direct_impact_score_mean' in merged_data.columns and 'indirect_impact_score_mean' in merged_data.columns:
        ax2.plot(merged_data['date'], merged_data['direct_impact_score_mean'],
                 'g-o', linewidth=2, markersize=5, alpha=0.8, label='Direct Impact')
        ax2.plot(merged_data['date'], merged_data['indirect_impact_score_mean'],
                 'orange', linewidth=2, markersize=5, alpha=0.8, label='Indirect Impact')
        ax2.set_title('Direct vs Indirect Impact', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Impact Score', fontsize=12)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Direct/Indirect Impact\nData Not Available',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Direct vs Indirect Impact', fontsize=14, fontweight='bold')

    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: News Score vs Future Returns
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()

    # Left axis: News Score
    line1 = ax3.plot(merged_data['date'], merged_data['overall_score_mean'],
                     'b-o', linewidth=2, markersize=4, label='News Score', alpha=0.8)
    ax3.set_ylabel('News Score', color='blue', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.grid(True, alpha=0.3)

    # Right axis: 1-Day Future Return
    if 'future_return_1d' in merged_data.columns:
        line2 = ax3_twin.plot(merged_data['date'], merged_data['future_return_1d'],
                              'r-s', linewidth=2, markersize=4, label='1-Day Future Return', alpha=0.8)
        ax3_twin.set_ylabel('1-Day Future Return (%)', color='red', fontsize=12)
        ax3_twin.tick_params(axis='y', labelcolor='red')
        ax3_twin.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    ax3.set_title('News Score vs 1-Day Future Return', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)

    # Combine legends
    lines = line1 + (line2 if 'future_return_1d' in merged_data.columns else [])
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    # Plot 4: Future Returns Comparison
    ax4 = axes[1, 1]

    future_return_cols = ['future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_5d']
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['1-Day', '2-Day', '3-Day', '5-Day']

    plotted_any = False
    for i, (col, color, label) in enumerate(zip(future_return_cols, colors, labels)):
        if col in merged_data.columns:
            # Calculate 3-day moving average for smoothing
            ma_data = merged_data[col].rolling(window=3, min_periods=1).mean()
            ax4.plot(merged_data['date'], ma_data,
                     color=color, linewidth=2, alpha=0.8, label=f'{label} Future Return', marker='o', markersize=3)
            plotted_any = True

    if plotted_any:
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Future Returns Comparison (3-Day MA)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Future Return (%)', fontsize=12)
        ax4.legend(fontsize=10)
    else:
        # Fallback: Show daily news count
        count_col = 'overall_score_count' if 'overall_score_count' in merged_data.columns else 'news_count'
        if count_col in merged_data.columns:
            ax4.bar(merged_data['date'], merged_data[count_col],
                    alpha=0.6, color='green', width=0.8)
            ax4.set_title('Daily News Count', fontsize=14, fontweight='bold')
            ax4.set_ylabel('News Count', fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'No Future Return Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Future Returns Comparison', fontsize=14, fontweight='bold')

    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def plot_correlation_analysis(merged_data):
    """
    绘制增强的相关性分析图

    Args:
        merged_data (pd.DataFrame): 合并后的数据
    """
    print("\n" + "=" * 50)
    print("步骤 6: 绘制相关性分析图")
    print("=" * 50)

    fig = plt.figure(figsize=(20, 16))

    # Plot 1: Overall Score vs 1-Day Future Return
    ax1 = plt.subplot(3, 4, 1)
    score_col = 'overall_score_mean' if 'overall_score_mean' in merged_data.columns else 'score_mean'
    valid_data = merged_data.dropna(subset=[score_col, 'future_return_1d'])
    if len(valid_data) > 1:
        ax1.scatter(valid_data[score_col], valid_data['future_return_1d'],
                   alpha=0.7, s=50, c='blue', edgecolors='navy', linewidth=0.5)
        if len(valid_data) > 2:
            z = np.polyfit(valid_data[score_col], valid_data['future_return_1d'], 1)
            p = np.poly1d(z)
            ax1.plot(valid_data[score_col], p(valid_data[score_col]),
                    "r--", alpha=0.8, linewidth=2)
        ax1.set_xlabel('Overall Score', fontsize=9)
        ax1.set_ylabel('1-Day Return %', fontsize=9)
        ax1.set_title('Score vs 1-Day Future Return', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot 2: Overall Score vs 2-Day Future Return
    ax2 = plt.subplot(3, 4, 2)
    valid_data_2d = merged_data.dropna(subset=[score_col, 'future_return_2d'])
    if len(valid_data_2d) > 1:
        ax2.scatter(valid_data_2d[score_col], valid_data_2d['future_return_2d'],
                   alpha=0.7, s=50, c='green', edgecolors='forestgreen', linewidth=0.5)
        if len(valid_data_2d) > 2:
            z = np.polyfit(valid_data_2d[score_col], valid_data_2d['future_return_2d'], 1)
            p = np.poly1d(z)
            ax2.plot(valid_data_2d[score_col], p(valid_data_2d[score_col]),
                    "r--", alpha=0.8, linewidth=2)
        ax2.set_xlabel('Overall Score', fontsize=9)
        ax2.set_ylabel('2-Day Return %', fontsize=9)
        ax2.set_title('Score vs 2-Day Future Return', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot 3: Overall Score vs 3-Day Future Return
    ax3 = plt.subplot(3, 4, 3)
    valid_data_3d = merged_data.dropna(subset=[score_col, 'future_return_3d'])
    if len(valid_data_3d) > 1:
        ax3.scatter(valid_data_3d[score_col], valid_data_3d['future_return_3d'],
                   alpha=0.7, s=50, c='orange', edgecolors='orangered', linewidth=0.5)
        if len(valid_data_3d) > 2:
            z = np.polyfit(valid_data_3d[score_col], valid_data_3d['future_return_3d'], 1)
            p = np.poly1d(z)
            ax3.plot(valid_data_3d[score_col], p(valid_data_3d[score_col]),
                    "r--", alpha=0.8, linewidth=2)
        ax3.set_xlabel('Overall Score', fontsize=9)
        ax3.set_ylabel('3-Day Return %', fontsize=9)
        ax3.set_title('Score vs 3-Day Future Return', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot 4: Overall Score vs 5-Day Future Return
    ax4 = plt.subplot(3, 4, 4)
    valid_data_5d = merged_data.dropna(subset=[score_col, 'future_return_5d'])
    if len(valid_data_5d) > 1:
        ax4.scatter(valid_data_5d[score_col], valid_data_5d['future_return_5d'],
                   alpha=0.7, s=50, c='purple', edgecolors='indigo', linewidth=0.5)
        if len(valid_data_5d) > 2:
            z = np.polyfit(valid_data_5d[score_col], valid_data_5d['future_return_5d'], 1)
            p = np.poly1d(z)
            ax4.plot(valid_data_5d[score_col], p(valid_data_5d[score_col]),
                    "r--", alpha=0.8, linewidth=2)
        ax4.set_xlabel('Overall Score', fontsize=9)
        ax4.set_ylabel('5-Day Return %', fontsize=9)
        ax4.set_title('Score vs 5-Day Future Return', fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot 5: Direct Impact vs 1-Day Future Return
    ax5 = plt.subplot(3, 4, 5)
    if 'direct_impact_score_mean' in merged_data.columns:
        valid_direct = merged_data.dropna(subset=['direct_impact_score_mean', 'future_return_1d'])
        if len(valid_direct) > 1:
            ax5.scatter(valid_direct['direct_impact_score_mean'], valid_direct['future_return_1d'],
                       alpha=0.7, s=50, c='red', edgecolors='crimson', linewidth=0.5)
            if len(valid_direct) > 2:
                z = np.polyfit(valid_direct['direct_impact_score_mean'], valid_direct['future_return_1d'], 1)
                p = np.poly1d(z)
                ax5.plot(valid_direct['direct_impact_score_mean'], p(valid_direct['direct_impact_score_mean']),
                        "r--", alpha=0.8, linewidth=2)
            ax5.set_xlabel('Direct Impact', fontsize=9)
            ax5.set_ylabel('1-Day Return %', fontsize=9)
            ax5.set_title('Direct Impact vs 1-Day Return', fontsize=10, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No Direct Impact Data', ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'No Direct Impact Data', ha='center', va='center', transform=ax5.transAxes)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Indirect Impact vs 1-Day Future Return
    ax6 = plt.subplot(3, 4, 6)
    if 'indirect_impact_score_mean' in merged_data.columns:
        valid_indirect = merged_data.dropna(subset=['indirect_impact_score_mean', 'future_return_1d'])
        if len(valid_indirect) > 1:
            ax6.scatter(valid_indirect['indirect_impact_score_mean'], valid_indirect['future_return_1d'],
                       alpha=0.7, s=50, c='brown', edgecolors='saddlebrown', linewidth=0.5)
            if len(valid_indirect) > 2:
                z = np.polyfit(valid_indirect['indirect_impact_score_mean'], valid_indirect['future_return_1d'], 1)
                p = np.poly1d(z)
                ax6.plot(valid_indirect['indirect_impact_score_mean'], p(valid_indirect['indirect_impact_score_mean']),
                        "r--", alpha=0.8, linewidth=2)
            ax6.set_xlabel('Indirect Impact', fontsize=9)
            ax6.set_ylabel('1-Day Return %', fontsize=9)
            ax6.set_title('Indirect Impact vs 1-Day Return', fontsize=10, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No Indirect Impact Data', ha='center', va='center', transform=ax6.transAxes)
    else:
        ax6.text(0.5, 0.5, 'No Indirect Impact Data', ha='center', va='center', transform=ax6.transAxes)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Certainty vs 1-Day Future Return
    ax7 = plt.subplot(3, 4, 7)
    if 'certainty_mean' in merged_data.columns:
        valid_certainty = merged_data.dropna(subset=['certainty_mean', 'future_return_1d'])
        if len(valid_certainty) > 1:
            ax7.scatter(valid_certainty['certainty_mean'], valid_certainty['future_return_1d'],
                       alpha=0.7, s=50, c='teal', edgecolors='darkslategray', linewidth=0.5)
            if len(valid_certainty) > 2:
                z = np.polyfit(valid_certainty['certainty_mean'], valid_certainty['future_return_1d'], 1)
                p = np.poly1d(z)
                ax7.plot(valid_certainty['certainty_mean'], p(valid_certainty['certainty_mean']),
                        "r--", alpha=0.8, linewidth=2)
            ax7.set_xlabel('Certainty', fontsize=9)
            ax7.set_ylabel('1-Day Return %', fontsize=9)
            ax7.set_title('Certainty vs 1-Day Return', fontsize=10, fontweight='bold')
        else:
            ax7.text(0.5, 0.5, 'No Certainty Data', ha='center', va='center', transform=ax7.transAxes)
    else:
        ax7.text(0.5, 0.5, 'No Certainty Data', ha='center', va='center', transform=ax7.transAxes)
    ax7.grid(True, alpha=0.3)

    # Plot 8: Weighted Score vs 1-Day Future Return
    ax8 = plt.subplot(3, 4, 8)
    if 'weighted_score' in merged_data.columns:
        valid_weighted = merged_data.dropna(subset=['weighted_score', 'future_return_1d'])
        if len(valid_weighted) > 1:
            ax8.scatter(valid_weighted['weighted_score'], valid_weighted['future_return_1d'],
                       alpha=0.7, s=50, c='darkgoldenrod', edgecolors='saddlebrown', linewidth=0.5)
            if len(valid_weighted) > 2:
                z = np.polyfit(valid_weighted['weighted_score'], valid_weighted['future_return_1d'], 1)
                p = np.poly1d(z)
                ax8.plot(valid_weighted['weighted_score'], p(valid_weighted['weighted_score']),
                        "r--", alpha=0.8, linewidth=2)
            ax8.set_xlabel('Weighted Score', fontsize=9)
            ax8.set_ylabel('1-Day Return %', fontsize=9)
            ax8.set_title('Weighted Score vs 1-Day Return', fontsize=10, fontweight='bold')
        else:
            ax8.text(0.5, 0.5, 'No Weighted Score Data', ha='center', va='center', transform=ax8.transAxes)
    else:
        ax8.text(0.5, 0.5, 'No Weighted Score Data', ha='center', va='center', transform=ax8.transAxes)
    ax8.grid(True, alpha=0.3)

    # 图9-12: 相关性热力图（合并成一个大图）
    ax_heatmap = plt.subplot(3, 4, (9, 12))

    # 构建相关性矩阵的列
    correlation_columns = []
    column_mapping = {}

    # 基础列
    if score_col in merged_data.columns:
        correlation_columns.append(score_col)
        column_mapping[score_col] = 'Overall Score'

    if 'direct_impact_score_mean' in merged_data.columns:
        correlation_columns.append('direct_impact_score_mean')
        column_mapping['direct_impact_score_mean'] = 'Direct Impact'

    if 'indirect_impact_score_mean' in merged_data.columns:
        correlation_columns.append('indirect_impact_score_mean')
        column_mapping['indirect_impact_score_mean'] = 'Indirect Impact'

    if 'certainty_mean' in merged_data.columns:
        correlation_columns.append('certainty_mean')
        column_mapping['certainty_mean'] = 'Certainty'

    if 'weighted_score' in merged_data.columns:
        correlation_columns.append('weighted_score')
        column_mapping['weighted_score'] = 'Weighted Score'

    # 未来收益率列
    future_return_columns = ['future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_5d']
    for col in future_return_columns:
        if col in merged_data.columns:
            correlation_columns.append(col)
            if col == 'future_return_1d':
                column_mapping[col] = '1-Day Return'
            elif col == 'future_return_2d':
                column_mapping[col] = '2-Day Return'
            elif col == 'future_return_3d':
                column_mapping[col] = '3-Day Return'
            elif col == 'future_return_5d':
                column_mapping[col] = '5-Day Return'

    # 添加收盘价作为参考
    if 'Close' in merged_data.columns:
        correlation_columns.append('Close')
        column_mapping['Close'] = 'Close Price'

    if len(correlation_columns) > 2:
        correlation_data = merged_data[correlation_columns].corr()
        correlation_data.rename(columns=column_mapping, index=column_mapping, inplace=True)

        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, ax=ax_heatmap, cbar_kws={'shrink': 0.8},
                    fmt='.3f', linewidths=0.5)
        ax_heatmap.set_title('News-Future Return Correlation Matrix', fontsize=12, fontweight='bold')
    else:
        ax_heatmap.text(0.5, 0.5, 'Insufficient Data for Correlation Matrix',
                       ha='center', va='center', transform=ax_heatmap.transAxes)

    plt.tight_layout()
    plt.show()

def plot_ic_analysis(merged_data):
    """
    绘制IC系数分析图表

    Args:
        merged_data (pd.DataFrame): 合并后的数据
    """
    print("\n" + "=" * 50)
    print("步骤 6.5: IC系数可视化分析")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    score_col = 'overall_score_mean' if 'overall_score_mean' in merged_data.columns else 'score_mean'
    future_return_cols = ['future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_5d']
    colors = ['blue', 'green', 'orange', 'red']
    periods = ['1-Day', '2-Day', '3-Day', '5-Day']

    # 计算每个时间周期的IC值
    ic_data = []
    for i, (return_col, color, period) in enumerate(zip(future_return_cols, colors, periods)):
        if return_col in merged_data.columns and score_col in merged_data.columns:
            ic_metrics = calculate_ic_metrics(merged_data[score_col], merged_data[return_col])
            ic_data.append({
                'period': period,
                'normal_ic': ic_metrics['normal_ic'],
                'rank_ic': ic_metrics['rank_ic'],
                'rank_p': ic_metrics['rank_ic_pvalue'],
                'sample_size': ic_metrics['sample_size'],
                'color': color
            })

    # 图1: IC值对比柱状图
    ax1 = axes[0, 0]
    if ic_data:
        periods_list = [item['period'] for item in ic_data]
        normal_ics = [item['normal_ic'] if not np.isnan(item['normal_ic']) else 0 for item in ic_data]
        rank_ics = [item['rank_ic'] if not np.isnan(item['rank_ic']) else 0 for item in ic_data]

        x = np.arange(len(periods_list))
        width = 0.35

        bars1 = ax1.bar(x - width/2, normal_ics, width, label='Normal IC', alpha=0.8, color='lightblue')
        bars2 = ax1.bar(x + width/2, rank_ics, width, label='Rank IC', alpha=0.8, color='lightcoral')

        # 添加显著性标记
        for i, item in enumerate(ic_data):
            if item['rank_p'] < 0.05:
                ax1.text(i + width/2, item['rank_ic'] + 0.001, '**', ha='center', va='bottom', fontweight='bold')

        ax1.set_xlabel('Forecast Period', fontsize=12)
        ax1.set_ylabel('IC Coefficient', fontsize=12)
        ax1.set_title('IC Coefficient Comparison Across Periods', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(periods_list)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # 图2: IC绝对值对比（显示预测能力强度）
    ax2 = axes[0, 1]
    if ic_data:
        abs_normal_ics = [abs(item['normal_ic']) if not np.isnan(item['normal_ic']) else 0 for item in ic_data]
        abs_rank_ics = [abs(item['rank_ic']) if not np.isnan(item['rank_ic']) else 0 for item in ic_data]

        bars1 = ax2.bar(x - width/2, abs_normal_ics, width, label='|Normal IC|', alpha=0.8, color='skyblue')
        bars2 = ax2.bar(x + width/2, abs_rank_ics, width, label='|Rank IC|', alpha=0.8, color='salmon')

        ax2.set_xlabel('Forecast Period', fontsize=12)
        ax2.set_ylabel('IC Absolute Value', fontsize=12)
        ax2.set_title('Predictive Power Strength Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(periods_list)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add quality reference lines
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate Predictive Power')
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Strong Predictive Power')

    # Plot 3: IC Statistical Significance
    ax3 = axes[0, 2]
    if ic_data:
        p_values = [item['rank_p'] if not np.isnan(item['rank_p']) else 1 for item in ic_data]
        colors_sig = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]

        bars = ax3.bar(periods_list, [-np.log10(p) if p > 0 else 0 for p in p_values],
                      color=colors_sig, alpha=0.7)

        ax3.set_xlabel('Forecast Period', fontsize=12)
        ax3.set_ylabel('-log10(p-value)', fontsize=12)
        ax3.set_title('IC Statistical Significance', fontsize=14, fontweight='bold')
        ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax3.axhline(y=-np.log10(0.1), color='orange', linestyle='--', alpha=0.7, label='p=0.1')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: News Score Distribution
    ax4 = axes[1, 0]
    if score_col in merged_data.columns:
        valid_scores = merged_data[score_col].dropna()
        ax4.hist(valid_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_xlabel('News Score', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('News Score Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Add statistical information
        ax4.axvline(valid_scores.mean(), color='red', linestyle='--', alpha=0.8, label=f'Mean: {valid_scores.mean():.3f}')
        ax4.axvline(valid_scores.median(), color='blue', linestyle='--', alpha=0.8, label=f'Median: {valid_scores.median():.3f}')
        ax4.legend()

    # Plot 5: Future Return Distribution Comparison
    ax5 = axes[1, 1]
    for i, (return_col, color, period) in enumerate(zip(future_return_cols[:2], colors[:2], periods[:2])):
        if return_col in merged_data.columns:
            valid_returns = merged_data[return_col].dropna()
            ax5.hist(valid_returns, bins=15, alpha=0.6, color=color, label=f'{period} Return',
                    density=True)

    ax5.set_xlabel('Return (%)', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.set_title('Future Return Distribution Comparison', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axvline(0, color='black', linestyle='-', alpha=0.5)

    # Plot 6: IC Quality Comparison (Simplified)
    ax6 = axes[1, 2]
    if ic_data and len(ic_data) >= 2:
        # Select the best two periods for comparison
        sorted_ic = sorted(ic_data, key=lambda x: abs(x['rank_ic']) if not np.isnan(x['rank_ic']) else 0, reverse=True)
        if len(sorted_ic) >= 2:
            period1, period2 = sorted_ic[0], sorted_ic[1]

            # Create simplified quality comparison
            metrics = ['IC Strength', 'Significance', 'Sample Size']
            values1 = [
                min(abs(period1['rank_ic']) * 10, 1) if not np.isnan(period1['rank_ic']) else 0,  # IC strength (scaled by 10 for display)
                1 if period1['rank_p'] < 0.05 else 0.5 if period1['rank_p'] < 0.1 else 0,  # Significance
                min(period1['sample_size'] / 30, 1)  # Sample sufficiency
            ]
            values2 = [
                min(abs(period2['rank_ic']) * 10, 1) if not np.isnan(period2['rank_ic']) else 0,
                1 if period2['rank_p'] < 0.05 else 0.5 if period2['rank_p'] < 0.1 else 0,
                min(period2['sample_size'] / 30, 1)
            ]

            x = np.arange(len(metrics))
            width = 0.35

            bars1 = ax6.bar(x - width/2, values1, width, label=f'{period1["period"]}', alpha=0.7, color=period1['color'])
            bars2 = ax6.bar(x + width/2, values2, width, label=f'{period2["period"]}', alpha=0.7, color=period2['color'])

            ax6.set_xlabel('Quality Metrics', fontsize=12)
            ax6.set_ylabel('Normalized Score', fontsize=12)
            ax6.set_title('IC Quality Comparison', fontsize=14, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(metrics)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(0, 1.2)
        else:
            ax6.text(0.5, 0.5, 'Insufficient Data\nfor Quality Comparison', ha='center', va='center', transform=ax6.transAxes, fontsize=12)

    plt.tight_layout()
    plt.show()

def calculate_ic_metrics(factor_values, return_values):
    """
    计算信息系数(IC)相关指标

    Args:
        factor_values (pd.Series): 因子值序列
        return_values (pd.Series): 收益率序列

    Returns:
        dict: IC相关指标
    """
    # 移除缺失值
    valid_data = pd.DataFrame({'factor': factor_values, 'return': return_values}).dropna()

    if len(valid_data) < 3:
        return {
            'normal_ic': np.nan,
            'rank_ic': np.nan,
            'normal_ic_pvalue': np.nan,
            'rank_ic_pvalue': np.nan,
            'sample_size': len(valid_data)
        }

    try:
        # Normal IC (Pearson correlation)
        normal_ic, normal_p = pearsonr(valid_data['factor'], valid_data['return'])

        # Rank IC (Spearman correlation) - 更鲁棒
        rank_ic, rank_p = spearmanr(valid_data['factor'], valid_data['return'])

        return {
            'normal_ic': normal_ic,
            'rank_ic': rank_ic,
            'normal_ic_pvalue': normal_p,
            'rank_ic_pvalue': rank_p,
            'sample_size': len(valid_data)
        }
    except Exception as e:
        print(f"IC计算错误: {e}")
        return {
            'normal_ic': np.nan,
            'rank_ic': np.nan,
            'normal_ic_pvalue': np.nan,
            'rank_ic_pvalue': np.nan,
            'sample_size': len(valid_data)
        }

def calculate_ic_summary_stats(ic_values):
    """
    计算IC汇总统计指标

    Args:
        ic_values (list): IC值列表

    Returns:
        dict: IC汇总统计
    """
    if not ic_values or len([x for x in ic_values if not np.isnan(x)]) < 2:
        return {
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'ic_ir': np.nan,
            'ic_win_rate': np.nan,
            'ic_abs_mean': np.nan
        }

    ic_array = np.array([x for x in ic_values if not np.isnan(x)])

    ic_mean = np.mean(ic_array)
    ic_std = np.std(ic_array, ddof=1)
    ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan
    ic_win_rate = np.sum(ic_array > 0) / len(ic_array)
    ic_abs_mean = np.mean(np.abs(ic_array))

    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'ic_win_rate': ic_win_rate,
        'ic_abs_mean': ic_abs_mean
    }

def interpret_ic_quality(ic_ir, ic_mean, ic_win_rate):
    """
    解释IC质量

    Args:
        ic_ir (float): IC信息比率
        ic_mean (float): IC均值
        ic_win_rate (float): IC胜率

    Returns:
        str: IC质量评价
    """
    if np.isnan(ic_ir) or np.isnan(ic_mean):
        return "数据不足"

    # IC_IR评价标准（常用标准）
    if abs(ic_ir) >= 0.5:
        ir_quality = "优秀"
    elif abs(ic_ir) >= 0.3:
        ir_quality = "良好"
    elif abs(ic_ir) >= 0.15:
        ir_quality = "一般"
    else:
        ir_quality = "较差"

    # IC均值评价
    if abs(ic_mean) >= 0.1:
        mean_quality = "强"
    elif abs(ic_mean) >= 0.05:
        mean_quality = "中等"
    elif abs(ic_mean) >= 0.02:
        mean_quality = "弱"
    else:
        mean_quality = "很弱"

    # 胜率评价
    if ic_win_rate >= 0.6:
        win_quality = "高"
    elif ic_win_rate >= 0.5:
        win_quality = "中等"
    else:
        win_quality = "低"

    return f"{ir_quality}(IR={ic_ir:.3f}), {mean_quality}预测力(IC={ic_mean:.3f}), {win_quality}胜率({ic_win_rate:.1%})"

def calculate_correlation_statistics(merged_data):
    """
    计算增强的相关性和IC统计

    Args:
        merged_data (pd.DataFrame): 合并后的数据

    Returns:
        dict: 相关性和IC分析结果
    """
    print("\n" + "=" * 50)
    print("步骤 7: IC系数与相关性统计分析")
    print("=" * 50)

    results = {}

    # 确定主要评分列
    score_col = 'overall_score_mean' if 'overall_score_mean' in merged_data.columns else 'score_mean'

    # 移除缺失值 (包含所有未来收益率列)
    future_return_cols = ['future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_5d']
    analysis_cols = [score_col] + [col for col in future_return_cols if col in merged_data.columns]
    analysis_data = merged_data.dropna(subset=analysis_cols)

    if len(analysis_data) >= 3:  # 至少需要3个数据点进行相关性分析

        print(f"1. 综合评分与未来收益率IC分析:")
        try:
            # 1. 综合评分与未来各期收益率相关性和IC分析
            results['overall_score'] = {}

            future_periods = {
                'future_return_1d': '1天',
                'future_return_2d': '2天',
                'future_return_3d': '3天',
                'future_return_5d': '5天'
            }

            for return_col, period_name in future_periods.items():
                if return_col in analysis_data.columns:
                    valid_data = analysis_data.dropna(subset=[score_col, return_col])
                    if len(valid_data) >= 3:
                        # 计算传统相关性
                        corr_val, p_val = pearsonr(valid_data[score_col], valid_data[return_col])

                        # 计算IC指标
                        ic_metrics = calculate_ic_metrics(valid_data[score_col], valid_data[return_col])

                        results['overall_score'][return_col] = {
                            'correlation': corr_val,
                            'p_value': p_val,
                            'sample_size': len(valid_data),
                            'ic_metrics': ic_metrics
                        }

                        significance = "**显著**" if p_val < 0.05 else "不显著"
                        rank_ic_sig = "**显著**" if ic_metrics['rank_ic_pvalue'] < 0.05 else "不显著"

                        print(f"   综合评分 vs 未来{period_name}收益率:")
                        print(f"     相关系数: r = {corr_val:.4f}, p = {p_val:.4f} ({significance})")
                        print(f"     Normal IC: {ic_metrics['normal_ic']:.4f}, Rank IC: {ic_metrics['rank_ic']:.4f} ({rank_ic_sig})")

        except Exception as e:
            print(f"   综合评分IC分析计算失败: {e}")

        # 2. 直接影响与未来收益率相关性
        print(f"\n2. 直接影响与未来收益率相关性分析:")
        if 'direct_impact_score_mean' in merged_data.columns:
            try:
                results['direct_impact'] = {}
                for return_col, period_name in future_periods.items():
                    if return_col in analysis_data.columns:
                        direct_data = analysis_data.dropna(subset=['direct_impact_score_mean', return_col])
                        if len(direct_data) >= 3:
                            corr_val, p_val = pearsonr(direct_data['direct_impact_score_mean'], direct_data[return_col])
                            results['direct_impact'][return_col] = {
                                'correlation': corr_val,
                                'p_value': p_val,
                                'sample_size': len(direct_data)
                            }
                            significance = "**显著**" if p_val < 0.05 else "不显著"
                            print(f"   直接影响 vs 未来{period_name}收益率: r = {corr_val:.4f}, p = {p_val:.4f} ({significance}, n={len(direct_data)})")
                        else:
                            print(f"   直接影响 vs 未来{period_name}收益率: 数据不足")
            except Exception as e:
                print(f"   直接影响相关性计算失败: {e}")
        else:
            print("   无直接影响数据")

        # 3. 间接影响与未来收益率相关性
        print(f"\n3. 间接影响与未来收益率相关性分析:")
        if 'indirect_impact_score_mean' in merged_data.columns:
            try:
                results['indirect_impact'] = {}
                for return_col, period_name in future_periods.items():
                    if return_col in analysis_data.columns:
                        indirect_data = analysis_data.dropna(subset=['indirect_impact_score_mean', return_col])
                        if len(indirect_data) >= 3:
                            corr_val, p_val = pearsonr(indirect_data['indirect_impact_score_mean'], indirect_data[return_col])
                            results['indirect_impact'][return_col] = {
                                'correlation': corr_val,
                                'p_value': p_val,
                                'sample_size': len(indirect_data)
                            }
                            significance = "**显著**" if p_val < 0.05 else "不显著"
                            print(f"   间接影响 vs 未来{period_name}收益率: r = {corr_val:.4f}, p = {p_val:.4f} ({significance}, n={len(indirect_data)})")
                        else:
                            print(f"   间接影响 vs 未来{period_name}收益率: 数据不足")
            except Exception as e:
                print(f"   间接影响相关性计算失败: {e}")
        else:
            print("   无间接影响数据")

        # 4. 确定性与未来收益率相关性
        print(f"\n4. 确定性与未来收益率相关性分析:")
        if 'certainty_mean' in merged_data.columns:
            try:
                results['certainty'] = {}
                for return_col, period_name in future_periods.items():
                    if return_col in analysis_data.columns:
                        certainty_data = analysis_data.dropna(subset=['certainty_mean', return_col])
                        if len(certainty_data) >= 3:
                            corr_val, p_val = pearsonr(certainty_data['certainty_mean'], certainty_data[return_col])
                            results['certainty'][return_col] = {
                                'correlation': corr_val,
                                'p_value': p_val,
                                'sample_size': len(certainty_data)
                            }
                            significance = "**显著**" if p_val < 0.05 else "不显著"
                            print(f"   确定性 vs 未来{period_name}收益率: r = {corr_val:.4f}, p = {p_val:.4f} ({significance}, n={len(certainty_data)})")
                        else:
                            print(f"   确定性 vs 未来{period_name}收益率: 数据不足")
            except Exception as e:
                print(f"   确定性相关性计算失败: {e}")
        else:
            print("   无确定性数据")

        # 5. 确定性加权评分与未来收益率相关性
        print(f"\n5. 确定性加权评分与未来收益率相关性分析:")
        if 'weighted_score' in merged_data.columns:
            try:
                results['weighted_score'] = {}
                for return_col, period_name in future_periods.items():
                    if return_col in analysis_data.columns:
                        weighted_data = analysis_data.dropna(subset=['weighted_score', return_col])
                        if len(weighted_data) >= 3:
                            corr_val, p_val = pearsonr(weighted_data['weighted_score'], weighted_data[return_col])
                            results['weighted_score'][return_col] = {
                                'correlation': corr_val,
                                'p_value': p_val,
                                'sample_size': len(weighted_data)
                            }
                            significance = "**显著**" if p_val < 0.05 else "不显著"
                            print(f"   加权评分 vs 未来{period_name}收益率: r = {corr_val:.4f}, p = {p_val:.4f} ({significance}, n={len(weighted_data)})")
                        else:
                            print(f"   加权评分 vs 未来{period_name}收益率: 数据不足")
            except Exception as e:
                print(f"   加权评分相关性计算失败: {e}")
        else:
            print("   无加权评分数据")

        # 6. 相关性强度解释和比较
        def interpret_correlation(r):
            abs_r = abs(r)
            if abs_r < 0.1:
                return "几乎无相关"
            elif abs_r < 0.3:
                return "弱相关"
            elif abs_r < 0.5:
                return "中等相关"
            elif abs_r < 0.7:
                return "强相关"
            else:
                return "很强相关"

        print(f"\n6. IC系数综合分析:")
        # 收集所有IC值进行汇总分析
        all_ic_results = []

        for category, data in results.items():
            if isinstance(data, dict):
                for return_col, stats in data.items():
                    if isinstance(stats, dict) and 'ic_metrics' in stats:
                        period = return_col.replace('future_return_', '').replace('d', '天')
                        ic_metrics = stats['ic_metrics']
                        all_ic_results.append((
                            category, period,
                            ic_metrics['normal_ic'],
                            ic_metrics['rank_ic'],
                            ic_metrics['rank_ic_pvalue'],
                            stats['correlation']
                        ))

        if all_ic_results:
            # 按Rank IC绝对值排序（Rank IC更鲁棒）
            all_ic_results.sort(key=lambda x: abs(x[3]) if not np.isnan(x[3]) else 0, reverse=True)

            print("   === IC系数排序结果 (按Rank IC强度降序) ===")
            for i, (category, period, normal_ic, rank_ic, rank_p, corr) in enumerate(all_ic_results[:10]):
                direction = '正' if rank_ic > 0 else '负'
                significance = "**显著**" if rank_p < 0.05 else "不显著"
                print(f"   {i+1}. {category}_未来{period}:")
                print(f"      Rank IC: {rank_ic:.4f} ({direction}向, {significance})")
                print(f"      Normal IC: {normal_ic:.4f}, 相关系数: r={corr:.4f}")

            # IC质量汇总
            print(f"\n   === IC质量汇总评价 ===")

            # 按因子类别汇总IC
            factor_ic_summary = {}
            for category, period, normal_ic, rank_ic, rank_p, corr in all_ic_results:
                if category not in factor_ic_summary:
                    factor_ic_summary[category] = {
                        'normal_ics': [],
                        'rank_ics': [],
                        'periods': []
                    }
                factor_ic_summary[category]['normal_ics'].append(normal_ic)
                factor_ic_summary[category]['rank_ics'].append(rank_ic)
                factor_ic_summary[category]['periods'].append(period)

            for category, ic_data in factor_ic_summary.items():
                # 计算该因子的IC汇总统计
                rank_ic_stats = calculate_ic_summary_stats(ic_data['rank_ics'])
                normal_ic_stats = calculate_ic_summary_stats(ic_data['normal_ics'])

                print(f"   {category}因子IC汇总:")
                print(f"     Rank IC - 均值: {rank_ic_stats['ic_mean']:.4f}, IR: {rank_ic_stats['ic_ir']:.3f}, 胜率: {rank_ic_stats['ic_win_rate']:.1%}")
                print(f"     Normal IC - 均值: {normal_ic_stats['ic_mean']:.4f}, IR: {normal_ic_stats['ic_ir']:.3f}, 胜率: {normal_ic_stats['ic_win_rate']:.1%}")

                # IC质量评价
                quality_assessment = interpret_ic_quality(
                    rank_ic_stats['ic_ir'],
                    rank_ic_stats['ic_mean'],
                    rank_ic_stats['ic_win_rate']
                )
                print(f"     综合评价: {quality_assessment}")

        else:
            print("   无可用的IC分析数据")

        print(f"\n7. 传统相关性强度比较:")
        correlations = []

        for category, data in results.items():
            if isinstance(data, dict):
                for return_col, stats in data.items():
                    if isinstance(stats, dict) and 'correlation' in stats:
                        period = return_col.replace('future_return_', '').replace('d', '天')
                        correlations.append((category, period, stats['correlation'], stats['p_value']))

        # 按绝对值排序
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        print("   === 排序结果 (按相关性强度降序) ===")
        for i, (category, period, corr, p_val) in enumerate(correlations[:10]):  # 显示前10个最强相关性
            direction = '正' if corr > 0 else '负'
            strength = interpret_correlation(corr)
            significance = "**" if p_val < 0.05 else ""
            print(f"   {i+1}. {category}_未来{period}: r={corr:.4f} ({strength}, {direction}相关){significance}")

        # 8. 基于IC分析的投资建议
        print(f"\n8. 基于IC分析的量化投资建议:")

        if all_ic_results:
            best_ic_result = all_ic_results[0]  # 最强IC
            category, period, normal_ic, rank_ic, rank_p, corr = best_ic_result

            print(f"   • 最佳因子: {category} (预测未来{period}收益率)")
            print(f"   • Rank IC: {rank_ic:.4f} ({interpret_correlation(rank_ic)})")

            if rank_p < 0.05:
                print("   • IC统计显著性: **显著** (p < 0.05)")
                significance_level = "高"
            elif rank_p < 0.1:
                print("   • IC统计显著性: 边缘显著 (p < 0.1)")
                significance_level = "中"
            else:
                print("   • IC统计显著性: 不显著 (p ≥ 0.1)")
                significance_level = "低"

            # 基于IC值给出量化建议
            if abs(rank_ic) >= 0.05:
                direction = '正面' if rank_ic > 0 else '负面'
                print(f"   • 实用价值: {direction}新闻评分对未来{period}收益率有{'较强' if abs(rank_ic) >= 0.1 else '中等'}预测能力")

                # 根据IC质量给出具体建议
                if abs(rank_ic) >= 0.1 and rank_p < 0.05:
                    print("   • 策略建议: ★★★ 可作为主要选股因子")
                elif abs(rank_ic) >= 0.05 and rank_p < 0.1:
                    print("   • 策略建议: ★★☆ 可作为辅助选股因子")
                else:
                    print("   • 策略建议: ★☆☆ 仅供参考，需结合其他因子")

                # 时间周期优化建议
                if all_ic_results:
                    # 找出该因子在不同时间周期的表现
                    factor_performance = [(prd, ric, rp) for cat, prd, nic, ric, rp, cr in all_ic_results if cat == category]
                    if len(factor_performance) > 1:
                        print("   • 时间周期优化:")
                        best_period = max(factor_performance, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
                        print(f"     - 最优预测周期: 未来{best_period[0]} (Rank IC: {best_period[1]:.4f})")

                        # 显示所有周期表现
                        for prd, ric, rp in sorted(factor_performance, key=lambda x: abs(x[1]), reverse=True):
                            sig_mark = "**显著**" if rp < 0.05 else "不显著"
                            print(f"     - 未来{prd}: Rank IC = {ric:.4f} ({sig_mark})")

            else:
                print("   • 实用价值: 新闻评分的预测能力较弱")
                print("   • 策略建议: 不建议单独使用，需结合技术指标和基本面分析")

            # 因子组合建议
            if len(factor_ic_summary) > 1:
                print("   • 因子组合建议:")
                factor_rankings = []
                for cat, ic_data in factor_ic_summary.items():
                    rank_ic_stats = calculate_ic_summary_stats(ic_data['rank_ics'])
                    if not np.isnan(rank_ic_stats['ic_ir']):
                        factor_rankings.append((cat, rank_ic_stats['ic_ir'], rank_ic_stats['ic_mean']))

                factor_rankings.sort(key=lambda x: abs(x[1]), reverse=True)
                for i, (cat, ic_ir, ic_mean) in enumerate(factor_rankings[:3]):
                    print(f"     {i+1}. {cat}: IR={ic_ir:.3f}, IC均值={ic_mean:.4f}")

        else:
            print("   • 无可用的IC分析数据")
            print("   • 建议检查数据质量或增加样本量")

        results['sample_size'] = len(analysis_data)

    else:
        print(f"⚠️  数据样本过少 (仅{len(analysis_data)}个样本)，无法进行可靠的相关性分析")
        print("建议收集更多时间段的数据以获得更准确的分析结果")
        results['error'] = "样本量不足"

    return results

def save_results(merged_data, output_path):
    """
    保存分析结果

    Args:
        merged_data (pd.DataFrame): 分析结果数据
        output_path (str): 输出文件路径
    """
    print("\n" + "=" * 50)
    print("步骤 8: 保存分析结果")
    print("=" * 50)

    try:
        merged_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 分析结果已保存到: {output_path}")

        # 显示最终数据摘要
        print(f"\n=== 最终数据摘要 ===")
        print(f"总共分析了 {len(merged_data)} 个交易日的数据")
        print(f"时间范围: {merged_data['date'].min().date()} 到 {merged_data['date'].max().date()}")
        # 动态确定评分列名
        score_col = 'overall_score_mean' if 'overall_score_mean' in merged_data.columns else 'score_mean'
        if score_col in merged_data.columns:
            print(f"平均新闻评分: {merged_data[score_col].mean():.2f}")
        else:
            print(f"无法找到评分列，可用列: {list(merged_data.columns)}")
        print(f"平均股价: {merged_data['Close'].mean():.2f} 元")

        if 'price_change' in merged_data.columns:
            valid_changes = merged_data['price_change'].dropna()
            if len(valid_changes) > 0:
                print(f"平均日收益率: {valid_changes.mean():.2f}%")
                print(f"收益率波动率: {valid_changes.std():.2f}%")

    except Exception as e:
        print(f"✗ 保存失败: {e}")

def main():
    """
    主函数 - 执行完整的分析流程
    """
    print("🔍 新闻评分与未来收益率IC系数分析")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 文件路径配置
    news_data_path = r'E:\projects\myQ\scripts\news_scores_result.csv'
    output_path = r'E:\projects\myQ\scripts\news_price_analysis_result.csv'

    # 检查输入文件是否存在
    if not os.path.exists(news_data_path):
        print(f"✗ 错误: 找不到新闻数据文件 {news_data_path}")
        return

    try:
        # 步骤1: 读取新闻数据
        news_df = load_news_data(news_data_path)
        if news_df is None:
            return

        # 步骤2: 聚合日度评分
        daily_scores = aggregate_daily_scores(news_df)

        # 步骤3: 获取股价数据
        stock_data = get_stock_price_data(daily_scores)

        # 步骤4: 合并数据
        merged_data = merge_data(daily_scores, stock_data)

        if len(merged_data) == 0:
            print("✗ 错误: 合并后没有数据，无法进行分析")
            return

        # 步骤5: 时间序列可视化
        plot_time_series(merged_data)

        # 步骤6: 相关性可视化
        plot_correlation_analysis(merged_data)

        # 步骤6.5: IC系数可视化
        plot_ic_analysis(merged_data)

        # 步骤7: IC系数与相关性统计分析
        correlation_results = calculate_correlation_statistics(merged_data)

        # 步骤8: 保存结果
        save_results(merged_data, output_path)

        print("\n" + "=" * 60)
        print("🎉 分析完成！")
        print("=" * 60)
        print("主要输出:")
        print("1. 时间序列图表")
        print("2. 相关性分析图表")
        print("3. IC系数专业分析图表")
        print("4. IC与相关性统计分析结果")
        print(f"5. 量化投资建议")
        print(f"6. 数据文件: {output_path}")

    except Exception as e:
        print(f"✗ 分析过程中发生错误: {e}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()