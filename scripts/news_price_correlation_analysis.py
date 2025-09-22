#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻评分与股价相关性分析
==============================

本脚本分析新闻情感评分与股价变化的相关性，包括：
1. 读取新闻评分数据
2. 获取对应时间的股价数据
3. 数据对齐和聚合
4. 可视化分析
5. 相关性统计分析

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
        # 计算价格变化率
        merged_data = merged_data.sort_values('date').reset_index(drop=True)
        merged_data['price_change'] = merged_data['Close'].pct_change() * 100
        merged_data['price_change_next'] = (merged_data['Close'].shift(-1) / merged_data['Close'] - 1) * 100
        merged_data['volume_change'] = merged_data['Volume'].pct_change() * 100

        print(f"✓ 时间范围: {merged_data['date'].min().date()} 到 {merged_data['date'].max().date()}")
        print("\n合并数据预览:")

        # 动态确定评分列名
        score_col = 'overall_score_mean' if 'overall_score_mean' in merged_data.columns else 'score_mean'
        preview_cols = ['date', score_col, 'Close', 'price_change']
        available_cols = [col for col in preview_cols if col in merged_data.columns]

        if len(available_cols) >= 3:
            print(merged_data[available_cols].head())
        else:
            print("列名不匹配，显示所有可用列:")
            print(f"可用列: {list(merged_data.columns)}")
            print(merged_data[['date', 'Close']].head())

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

    # 图1: 综合评分时间序列
    ax1 = axes[0, 0]
    ax1.plot(merged_data['date'], merged_data['overall_score_mean'],
             'b-o', linewidth=2.5, markersize=6, alpha=0.8, label='Overall Score')

    # 添加标准差填充
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

    # 图2: 直接影响 vs 间接影响
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

    # 图3: 股价与评分对比
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()

    # 左轴：新闻评分
    line1 = ax3.plot(merged_data['date'], merged_data['overall_score_mean'],
                     'b-o', linewidth=2, markersize=5, label='News Score', alpha=0.8)
    ax3.set_ylabel('News Score', color='blue', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.grid(True, alpha=0.3)

    # 右轴：股价
    line2 = ax3_twin.plot(merged_data['date'], merged_data['Close'],
                          'r-s', linewidth=2, markersize=5, label='Close Price', alpha=0.8)
    ax3_twin.set_ylabel('Stock Price (CNY)', color='red', fontsize=12)
    ax3_twin.tick_params(axis='y', labelcolor='red')

    ax3.set_title('News Score vs Stock Price', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    # 图4: 确定性加权评分（如果可用）
    ax4 = axes[1, 1]
    if 'weighted_score' in merged_data.columns:
        ax4.plot(merged_data['date'], merged_data['weighted_score'],
                 'purple', linewidth=2.5, markersize=6, alpha=0.8, label='Weighted Score')
        ax4.plot(merged_data['date'], merged_data['overall_score_mean'],
                 'b--', linewidth=1.5, alpha=0.6, label='Original Score')
        ax4.set_title('Certainty-Weighted Score', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Weighted Score', fontsize=12)
        ax4.legend()
    else:
        # 显示新闻数量分布
        # 查找新闻数量列
        count_col = 'overall_score_count' if 'overall_score_count' in merged_data.columns else 'news_count'
        if count_col in merged_data.columns:
            ax4.bar(merged_data['date'], merged_data[count_col],
                    alpha=0.6, color='green', width=0.8)
        else:
            ax4.text(0.5, 0.5, 'No News Count Data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Daily News Count', fontsize=14, fontweight='bold')
        ax4.set_ylabel('News Count', fontsize=12)

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

    fig = plt.figure(figsize=(18, 14))

    # 图1: 综合评分 vs 当日股价变化
    ax1 = plt.subplot(3, 3, 1)
    score_col = 'overall_score_mean' if 'overall_score_mean' in merged_data.columns else 'score_mean'
    valid_data = merged_data.dropna(subset=[score_col, 'price_change'])
    if len(valid_data) > 1:
        ax1.scatter(valid_data[score_col], valid_data['price_change'],
                   alpha=0.7, s=60, c='blue', edgecolors='navy', linewidth=0.5)
        if len(valid_data) > 2:
            z = np.polyfit(valid_data[score_col], valid_data['price_change'], 1)
            p = np.poly1d(z)
            ax1.plot(valid_data[score_col], p(valid_data[score_col]),
                    "r--", alpha=0.8, linewidth=2)
        ax1.set_xlabel('Overall Score', fontsize=10)
        ax1.set_ylabel('Same Day Δ%', fontsize=10)
        ax1.set_title('Overall Score vs Same Day Price', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # 图2: 综合评分 vs 次日股价变化
    ax2 = plt.subplot(3, 3, 2)
    valid_data_next = merged_data.dropna(subset=[score_col, 'price_change_next'])
    if len(valid_data_next) > 1:
        ax2.scatter(valid_data_next[score_col], valid_data_next['price_change_next'],
                   alpha=0.7, s=60, c='green', edgecolors='forestgreen', linewidth=0.5)
        if len(valid_data_next) > 2:
            z = np.polyfit(valid_data_next[score_col], valid_data_next['price_change_next'], 1)
            p = np.poly1d(z)
            ax2.plot(valid_data_next[score_col], p(valid_data_next[score_col]),
                    "r--", alpha=0.8, linewidth=2)
        ax2.set_xlabel('Overall Score', fontsize=10)
        ax2.set_ylabel('Next Day Δ%', fontsize=10)
        ax2.set_title('Overall Score vs Next Day Price', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # 图3: 直接影响 vs 股价变化
    ax3 = plt.subplot(3, 3, 3)
    if 'direct_impact_score_mean' in merged_data.columns:
        valid_direct = merged_data.dropna(subset=['direct_impact_score_mean', 'price_change'])
        if len(valid_direct) > 1:
            ax3.scatter(valid_direct['direct_impact_score_mean'], valid_direct['price_change'],
                       alpha=0.7, s=60, c='orange', edgecolors='orangered', linewidth=0.5)
            if len(valid_direct) > 2:
                z = np.polyfit(valid_direct['direct_impact_score_mean'], valid_direct['price_change'], 1)
                p = np.poly1d(z)
                ax3.plot(valid_direct['direct_impact_score_mean'], p(valid_direct['direct_impact_score_mean']),
                        "r--", alpha=0.8, linewidth=2)
            ax3.set_xlabel('Direct Impact', fontsize=10)
            ax3.set_ylabel('Same Day Δ%', fontsize=10)
            ax3.set_title('Direct Impact vs Price Change', fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Direct Impact Data', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No Direct Impact Data', ha='center', va='center', transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)

    # 图4: 间接影响 vs 股价变化
    ax4 = plt.subplot(3, 3, 4)
    if 'indirect_impact_score_mean' in merged_data.columns:
        valid_indirect = merged_data.dropna(subset=['indirect_impact_score_mean', 'price_change'])
        if len(valid_indirect) > 1:
            ax4.scatter(valid_indirect['indirect_impact_score_mean'], valid_indirect['price_change'],
                       alpha=0.7, s=60, c='purple', edgecolors='indigo', linewidth=0.5)
            if len(valid_indirect) > 2:
                z = np.polyfit(valid_indirect['indirect_impact_score_mean'], valid_indirect['price_change'], 1)
                p = np.poly1d(z)
                ax4.plot(valid_indirect['indirect_impact_score_mean'], p(valid_indirect['indirect_impact_score_mean']),
                        "r--", alpha=0.8, linewidth=2)
            ax4.set_xlabel('Indirect Impact', fontsize=10)
            ax4.set_ylabel('Same Day Δ%', fontsize=10)
            ax4.set_title('Indirect Impact vs Price Change', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No Indirect Impact Data', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No Indirect Impact Data', ha='center', va='center', transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3)

    # 图5: 确定性 vs 股价变化
    ax5 = plt.subplot(3, 3, 5)
    if 'certainty_mean' in merged_data.columns:
        valid_certainty = merged_data.dropna(subset=['certainty_mean', 'price_change'])
        if len(valid_certainty) > 1:
            ax5.scatter(valid_certainty['certainty_mean'], valid_certainty['price_change'],
                       alpha=0.7, s=60, c='red', edgecolors='crimson', linewidth=0.5)
            if len(valid_certainty) > 2:
                z = np.polyfit(valid_certainty['certainty_mean'], valid_certainty['price_change'], 1)
                p = np.poly1d(z)
                ax5.plot(valid_certainty['certainty_mean'], p(valid_certainty['certainty_mean']),
                        "r--", alpha=0.8, linewidth=2)
            ax5.set_xlabel('Certainty', fontsize=10)
            ax5.set_ylabel('Same Day Δ%', fontsize=10)
            ax5.set_title('Certainty vs Price Change', fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No Certainty Data', ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'No Certainty Data', ha='center', va='center', transform=ax5.transAxes)
    ax5.grid(True, alpha=0.3)

    # 图6: 加权评分 vs 股价变化
    ax6 = plt.subplot(3, 3, 6)
    if 'weighted_score' in merged_data.columns:
        valid_weighted = merged_data.dropna(subset=['weighted_score', 'price_change'])
        if len(valid_weighted) > 1:
            ax6.scatter(valid_weighted['weighted_score'], valid_weighted['price_change'],
                       alpha=0.7, s=60, c='brown', edgecolors='saddlebrown', linewidth=0.5)
            if len(valid_weighted) > 2:
                z = np.polyfit(valid_weighted['weighted_score'], valid_weighted['price_change'], 1)
                p = np.poly1d(z)
                ax6.plot(valid_weighted['weighted_score'], p(valid_weighted['weighted_score']),
                        "r--", alpha=0.8, linewidth=2)
            ax6.set_xlabel('Weighted Score', fontsize=10)
            ax6.set_ylabel('Same Day Δ%', fontsize=10)
            ax6.set_title('Weighted Score vs Price Change', fontsize=12, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No Weighted Score Data', ha='center', va='center', transform=ax6.transAxes)
    else:
        ax6.text(0.5, 0.5, 'No Weighted Score Data', ha='center', va='center', transform=ax6.transAxes)
    ax6.grid(True, alpha=0.3)

    # 图7-9: 相关性热力图（合并成一个大图）
    ax_heatmap = plt.subplot(3, 3, (7, 9))

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

    # 股价相关列
    price_columns = ['Close', 'price_change', 'price_change_next']
    for col in price_columns:
        if col in merged_data.columns:
            correlation_columns.append(col)
            if col == 'Close':
                column_mapping[col] = 'Close Price'
            elif col == 'price_change':
                column_mapping[col] = 'Same Day Δ%'
            elif col == 'price_change_next':
                column_mapping[col] = 'Next Day Δ%'

    if len(correlation_columns) > 2:
        correlation_data = merged_data[correlation_columns].corr()
        correlation_data.rename(columns=column_mapping, index=column_mapping, inplace=True)

        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, ax=ax_heatmap, cbar_kws={'shrink': 0.8},
                    fmt='.3f', linewidths=0.5)
        ax_heatmap.set_title('Enhanced Correlation Matrix', fontsize=14, fontweight='bold')
    else:
        ax_heatmap.text(0.5, 0.5, 'Insufficient Data for Correlation Matrix',
                       ha='center', va='center', transform=ax_heatmap.transAxes)

    plt.tight_layout()
    plt.show()

def calculate_correlation_statistics(merged_data):
    """
    计算增强的相关性统计

    Args:
        merged_data (pd.DataFrame): 合并后的数据

    Returns:
        dict: 相关性分析结果
    """
    print("\n" + "=" * 50)
    print("步骤 7: 增强的相关性统计分析")
    print("=" * 50)

    results = {}

    # 确定主要评分列
    score_col = 'overall_score_mean' if 'overall_score_mean' in merged_data.columns else 'score_mean'

    # 移除缺失值
    analysis_data = merged_data.dropna(subset=[score_col, 'price_change', 'price_change_next'])

    if len(analysis_data) >= 3:  # 至少需要3个数据点进行相关性分析

        print(f"1. 综合评分相关性分析:")
        try:
            # 1. 综合评分相关性
            corr_current, p_value_current = pearsonr(analysis_data[score_col], analysis_data['price_change'])
            corr_next, p_value_next = pearsonr(analysis_data[score_col], analysis_data['price_change_next'])

            results['overall_score'] = {
                'current_day': {'correlation': corr_current, 'p_value': p_value_current},
                'next_day': {'correlation': corr_next, 'p_value': p_value_next}
            }

            print(f"   综合评分 vs 当日股价变化: r = {corr_current:.4f}, p-value = {p_value_current:.4f}")
            print(f"   综合评分 vs 次日股价变化: r = {corr_next:.4f}, p-value = {p_value_next:.4f}")

        except Exception as e:
            print(f"   综合评分相关性计算失败: {e}")

        # 2. 直接影响相关性
        print(f"\n2. 直接影响相关性分析:")
        if 'direct_impact_score_mean' in merged_data.columns:
            try:
                direct_data = analysis_data.dropna(subset=['direct_impact_score_mean'])
                if len(direct_data) >= 3:
                    corr_direct_current, p_direct_current = pearsonr(direct_data['direct_impact_score_mean'], direct_data['price_change'])
                    corr_direct_next, p_direct_next = pearsonr(direct_data['direct_impact_score_mean'], direct_data['price_change_next'])

                    results['direct_impact'] = {
                        'current_day': {'correlation': corr_direct_current, 'p_value': p_direct_current},
                        'next_day': {'correlation': corr_direct_next, 'p_value': p_direct_next}
                    }

                    print(f"   直接影响 vs 当日股价变化: r = {corr_direct_current:.4f}, p-value = {p_direct_current:.4f}")
                    print(f"   直接影响 vs 次日股价变化: r = {corr_direct_next:.4f}, p-value = {p_direct_next:.4f}")
                else:
                    print("   直接影响数据不足，无法计算相关性")
            except Exception as e:
                print(f"   直接影响相关性计算失败: {e}")
        else:
            print("   无直接影响数据")

        # 3. 间接影响相关性
        print(f"\n3. 间接影响相关性分析:")
        if 'indirect_impact_score_mean' in merged_data.columns:
            try:
                indirect_data = analysis_data.dropna(subset=['indirect_impact_score_mean'])
                if len(indirect_data) >= 3:
                    corr_indirect_current, p_indirect_current = pearsonr(indirect_data['indirect_impact_score_mean'], indirect_data['price_change'])
                    corr_indirect_next, p_indirect_next = pearsonr(indirect_data['indirect_impact_score_mean'], indirect_data['price_change_next'])

                    results['indirect_impact'] = {
                        'current_day': {'correlation': corr_indirect_current, 'p_value': p_indirect_current},
                        'next_day': {'correlation': corr_indirect_next, 'p_value': p_indirect_next}
                    }

                    print(f"   间接影响 vs 当日股价变化: r = {corr_indirect_current:.4f}, p-value = {p_indirect_current:.4f}")
                    print(f"   间接影响 vs 次日股价变化: r = {corr_indirect_next:.4f}, p-value = {p_indirect_next:.4f}")
                else:
                    print("   间接影响数据不足，无法计算相关性")
            except Exception as e:
                print(f"   间接影响相关性计算失败: {e}")
        else:
            print("   无间接影响数据")

        # 4. 确定性相关性
        print(f"\n4. 确定性相关性分析:")
        if 'certainty_mean' in merged_data.columns:
            try:
                certainty_data = analysis_data.dropna(subset=['certainty_mean'])
                if len(certainty_data) >= 3:
                    corr_certainty_current, p_certainty_current = pearsonr(certainty_data['certainty_mean'], certainty_data['price_change'])
                    corr_certainty_next, p_certainty_next = pearsonr(certainty_data['certainty_mean'], certainty_data['price_change_next'])

                    results['certainty'] = {
                        'current_day': {'correlation': corr_certainty_current, 'p_value': p_certainty_current},
                        'next_day': {'correlation': corr_certainty_next, 'p_value': p_certainty_next}
                    }

                    print(f"   确定性 vs 当日股价变化: r = {corr_certainty_current:.4f}, p-value = {p_certainty_current:.4f}")
                    print(f"   确定性 vs 次日股价变化: r = {corr_certainty_next:.4f}, p-value = {p_certainty_next:.4f}")
                else:
                    print("   确定性数据不足，无法计算相关性")
            except Exception as e:
                print(f"   确定性相关性计算失败: {e}")
        else:
            print("   无确定性数据")

        # 5. 加权评分相关性
        print(f"\n5. 确定性加权评分相关性分析:")
        if 'weighted_score' in merged_data.columns:
            try:
                weighted_data = analysis_data.dropna(subset=['weighted_score'])
                if len(weighted_data) >= 3:
                    corr_weighted_current, p_weighted_current = pearsonr(weighted_data['weighted_score'], weighted_data['price_change'])
                    corr_weighted_next, p_weighted_next = pearsonr(weighted_data['weighted_score'], weighted_data['price_change_next'])

                    results['weighted_score'] = {
                        'current_day': {'correlation': corr_weighted_current, 'p_value': p_weighted_current},
                        'next_day': {'correlation': corr_weighted_next, 'p_value': p_weighted_next}
                    }

                    print(f"   加权评分 vs 当日股价变化: r = {corr_weighted_current:.4f}, p-value = {p_weighted_current:.4f}")
                    print(f"   加权评分 vs 次日股价变化: r = {corr_weighted_next:.4f}, p-value = {p_weighted_next:.4f}")
                else:
                    print("   加权评分数据不足，无法计算相关性")
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

        print(f"\n6. 相关性强度比较:")
        correlations = []

        for category, data in results.items():
            if isinstance(data, dict) and 'current_day' in data:
                current_r = data['current_day']['correlation']
                next_r = data['next_day']['correlation']
                correlations.append((category, 'current_day', current_r))
                correlations.append((category, 'next_day', next_r))

        # 按绝对值排序
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        for i, (category, timing, corr) in enumerate(correlations[:5]):  # 显示前5个最强相关性
            direction = '正' if corr > 0 else '负'
            strength = interpret_correlation(corr)
            print(f"   {i+1}. {category}_{timing}: r={corr:.4f} ({strength}, {direction}相关)")

        # 7. 投资建议
        print(f"\n7. 基于增强分析的投资建议:")

        best_correlation = max(correlations, key=lambda x: abs(x[2])) if correlations else None

        if best_correlation and abs(best_correlation[2]) > 0.3:
            category, timing, corr = best_correlation
            print(f"   • 最强预测因子: {category} ({'当日' if timing == 'current_day' else '次日'})")
            if timing == 'next_day' and abs(corr) > 0.3:
                direction = '正面' if corr > 0 else '负面'
                print(f"   • {direction}新闻评分可能预示次日股价{'上涨' if corr > 0 else '下跌'}")
            else:
                print("   • 主要影响体现在当日股价反应")
        else:
            print("   • 新闻评分与股价变化的关系相对较弱")
            print("   • 建议结合其他技术指标进行综合分析")

        if 'weighted_score' in results and 'overall_score' in results:
            weighted_corr = abs(results['weighted_score']['next_day']['correlation'])
            overall_corr = abs(results['overall_score']['next_day']['correlation'])
            if weighted_corr > overall_corr * 1.1:
                print("   • 确定性加权评分预测效果更好，建议重点关注高确定性新闻")

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
    print("🔍 新闻评分与股价相关性分析")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 文件路径配置
    news_data_path = r'D:\projects\q\myQ\scripts\news_scores_result.csv'
    output_path = r'D:\projects\q\myQ\scripts\news_price_analysis_result.csv'

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

        # 步骤7: 统计分析
        correlation_results = calculate_correlation_statistics(merged_data)

        # 步骤8: 保存结果
        save_results(merged_data, output_path)

        print("\n" + "=" * 60)
        print("🎉 分析完成！")
        print("=" * 60)
        print("主要输出:")
        print("1. 时间序列图表")
        print("2. 相关性分析图表")
        print("3. 统计分析结果")
        print(f"4. 数据文件: {output_path}")

    except Exception as e:
        print(f"✗ 分析过程中发生错误: {e}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()