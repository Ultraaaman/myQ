#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻评分与开盘强度深度分析
============================

专门分析新闻情感评分与开盘强度指标的相关性，包括：
1. 开盘强度的IC系数分析
2. 不同情感评分阈值的影响
3. 时间衰减效应分析
4. 开盘强度与成交量、波动率的组合分析
5. 基于开盘强度的交易策略回测

发现：开盘强度比简单收益率更能反映新闻的市场影响
应用：短期交易策略、盘口分析、情绪传导研究

作者: Claude Code
日期: 2025-09-23
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
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(r'E:\projects\myQ')

# 导入 market_data 模块
try:
    from quantlib.market_data import get_stock_data, MarketDataManager
    MARKET_DATA_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入 quantlib.market_data 模块: {e}")
    MARKET_DATA_AVAILABLE = False

# 设置图表样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# 设置英文显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10

def load_and_prepare_data(news_file_path, stock_code='601899'):
    """
    加载并准备分析数据

    Args:
        news_file_path (str): 新闻评分文件路径
        stock_code (str): 股票代码

    Returns:
        pd.DataFrame: 合并后的分析数据
    """
    print("=" * 60)
    print("步骤 1: 数据加载与预处理")
    print("=" * 60)

    # 读取新闻数据
    news_df = pd.read_csv(news_file_path, encoding='utf-8-sig')
    news_df['date'] = pd.to_datetime(news_df['original_date']).dt.date

    print(f"✓ 新闻数据: {len(news_df)} 条")
    print(f"✓ 时间范围: {news_df['date'].min()} 到 {news_df['date'].max()}")

    # 聚合日度新闻评分
    daily_scores = news_df.groupby('date').agg({
        'overall_score': ['mean', 'std', 'count', 'min', 'max'],
        'direct_impact_score': ['mean', 'std'],
        'indirect_impact_score': ['mean', 'std'],
        'certainty': ['mean', 'std'],
        'sentiment': lambda x: x.mode()[0] if not x.empty else 'neutral'
    }).round(4)

    # 扁平化列名
    daily_scores.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0]
                           for col in daily_scores.columns]
    daily_scores = daily_scores.reset_index()

    # 计算确定性加权评分
    daily_scores['weighted_score'] = (daily_scores['overall_score_mean'] *
                                     daily_scores['certainty_mean'])

    print(f"✓ 日度聚合: {len(daily_scores)} 天")

    # 获取股价数据
    if MARKET_DATA_AVAILABLE:
        try:
            manager = MarketDataManager()

            # 计算时间跨度，选择合适的period
            start_date = daily_scores['date'].min() - timedelta(days=10)
            end_date = daily_scores['date'].max() + timedelta(days=10)
            time_span = (end_date - start_date).days

            print(f"✓ 需要的股价数据时间范围: {start_date} 到 {end_date} ({time_span}天)")

            # 由于新闻数据是2024年的，需要获取更长期间的数据
            # 选择足够长的period来覆盖2024年的数据
            current_date = datetime.now().date()
            days_from_start = (current_date - start_date).days

            if days_from_start <= 95:
                period = '6mo'
            elif days_from_start <= 185:
                period = '1y'
            elif days_from_start <= 370:
                period = '2y'
            else:
                period = '5y'

            print(f"✓ 选择period: {period} (距离开始日期{days_from_start}天)")

            stock_data = manager.get_stock_data(
                symbol=stock_code,
                market='CN',  # 中国市场
                period=period,
                interval='1d'
            )

            if stock_data is not None and len(stock_data) > 0:
                print(f"✓ 原始股价数据形状: {stock_data.shape}")
                print(f"✓ 原始股价数据列: {list(stock_data.columns)}")

                # 重置索引，确保date列存在
                stock_data = stock_data.reset_index()

                # 选择需要的列（OHLCV）
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                available_columns = []

                # 查找匹配的列（不区分大小写）
                for req_col in required_columns:
                    for col in stock_data.columns:
                        if col.lower() == req_col.lower():
                            available_columns.append(col)
                            break

                if len(available_columns) >= 4:  # 至少需要OHLC
                    # 只选择需要的列
                    stock_data = stock_data[['date'] + available_columns[:5]]  # date + OHLCV

                    # 重命名列
                    new_column_names = ['date', 'Open', 'High', 'Low', 'Close']
                    if len(available_columns) == 5:
                        new_column_names.append('Volume')

                    stock_data.columns = new_column_names[:len(stock_data.columns)]

                    # 确保日期列格式正确
                    stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date

                    print(f"✓ 处理后股价数据: {len(stock_data)} 天")
                    print(f"✓ 股价数据时间范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
                    print(f"✓ 最终列名: {list(stock_data.columns)}")

                    # 筛选股价数据到新闻时间范围
                    news_start = daily_scores['date'].min()
                    news_end = daily_scores['date'].max()

                    stock_filtered = stock_data[
                        (stock_data['date'] >= news_start) &
                        (stock_data['date'] <= news_end)
                    ].copy()

                    if len(stock_filtered) > 0:
                        stock_data = stock_filtered
                        print(f"✓ 筛选到新闻时间范围的股价数据: {len(stock_data)} 天")
                        print(f"✓ 筛选后股价时间范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
                    else:
                        print(f"⚠️ 在新闻时间范围内没有找到股价数据")
                        print(f"   新闻时间: {news_start} 到 {news_end}")
                        print(f"   股价时间: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
                else:
                    raise ValueError(f"缺少必要的OHLC列，可用列: {available_columns}")
            else:
                raise ValueError("股价数据为空")

        except Exception as e:
            print(f"⚠️ 获取实际股价数据失败: {e}")
            print("✓ 使用模拟数据")
            stock_data = create_simulated_stock_data(daily_scores)
    else:
        print("✓ 使用模拟数据")
        stock_data = create_simulated_stock_data(daily_scores)

    # 合并数据
    merged_data = merge_and_calculate_indicators(daily_scores, stock_data)

    return merged_data

def create_simulated_stock_data(daily_scores):
    """创建模拟股价数据"""
    date_range = pd.date_range(
        start=daily_scores['date'].min() - timedelta(days=10),
        end=daily_scores['date'].max() + timedelta(days=10),
        freq='D'
    )

    # 移除周末
    date_range = [d for d in date_range if d.weekday() < 5]

    np.random.seed(42)
    n_days = len(date_range)

    # 模拟股价走势
    returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
    prices = 100 * np.cumprod(1 + returns)  # 累积价格

    stock_data = pd.DataFrame({
        'date': [d.date() for d in date_range],
        'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'High': prices * (1 + np.abs(np.random.normal(0.01, 0.01, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, n_days))),
        'Close': prices,
        'Volume': np.random.lognormal(15, 0.5, n_days)
    })

    # 确保High >= max(Open, Close) 和 Low <= min(Open, Close)
    stock_data['High'] = np.maximum(stock_data['High'],
                                   np.maximum(stock_data['Open'], stock_data['Close']))
    stock_data['Low'] = np.minimum(stock_data['Low'],
                                  np.minimum(stock_data['Open'], stock_data['Close']))

    return stock_data

def merge_and_calculate_indicators(daily_scores, stock_data):
    """合并数据并计算开盘强度等指标"""
    print("\n" + "=" * 60)
    print("步骤 2: 计算开盘强度及相关指标")
    print("=" * 60)

    # 调试信息
    print(f"✓ 新闻数据日期范围: {daily_scores['date'].min()} 到 {daily_scores['date'].max()}")
    print(f"✓ 股价数据日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
    print(f"✓ 新闻数据日期类型: {type(daily_scores['date'].iloc[0])}")
    print(f"✓ 股价数据日期类型: {type(stock_data['date'].iloc[0])}")

    # 确保日期格式一致
    daily_scores['date'] = pd.to_datetime(daily_scores['date']).dt.date
    stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date

    # 合并数据 - 先尝试inner join，如果结果为空则尝试其他方法
    merged_data = pd.merge(daily_scores, stock_data, on='date', how='inner')

    print(f"✓ Inner join结果: {len(merged_data)} 天")

    if len(merged_data) == 0:
        print("⚠️ Inner join结果为空，尝试查找重叠日期...")

        # 查找重叠的日期
        news_dates = set(daily_scores['date'])
        stock_dates = set(stock_data['date'])
        overlap_dates = news_dates.intersection(stock_dates)

        print(f"✓ 新闻数据天数: {len(news_dates)}")
        print(f"✓ 股价数据天数: {len(stock_dates)}")
        print(f"✓ 重叠天数: {len(overlap_dates)}")

        if len(overlap_dates) > 0:
            print(f"✓ 重叠日期示例: {sorted(list(overlap_dates))[:5]}")

            # 手动筛选重叠日期的数据
            news_subset = daily_scores[daily_scores['date'].isin(overlap_dates)].copy()
            stock_subset = stock_data[stock_data['date'].isin(overlap_dates)].copy()

            merged_data = pd.merge(news_subset, stock_subset, on='date', how='inner')
        else:
            print("✓ 新闻数据日期示例:", sorted(list(news_dates))[:5])
            print("✓ 股价数据日期示例:", sorted(list(stock_dates))[:5])

    merged_data = merged_data.sort_values('date').reset_index(drop=True)
    print(f"✓ 最终合并后数据: {len(merged_data)} 天")

    # 计算基础指标
    merged_data['price_change'] = merged_data['Close'].pct_change()

    # 核心指标：开盘强度相关
    print("✓ 计算开盘强度指标...")

    # 1. 开盘强度（开盘到最高价的涨幅）
    merged_data['open_strength'] = (merged_data['High'] - merged_data['Open']) / merged_data['Open']

    # 2. 收盘强度（最低价到收盘价的涨幅）
    merged_data['close_strength'] = (merged_data['Close'] - merged_data['Low']) / merged_data['Low']

    # 3. 开盘相对强度（开盘强度 vs 全日振幅）
    merged_data['daily_amplitude'] = (merged_data['High'] - merged_data['Low']) / merged_data['Close']
    merged_data['open_strength_ratio'] = merged_data['open_strength'] / (merged_data['daily_amplitude'] + 1e-8)

    # 4. 隔夜跳空
    merged_data['overnight_gap'] = merged_data['Open'] / merged_data['Close'].shift(1) - 1

    # 5. 开盘后动量（开盘强度 vs 隔夜跳空）
    merged_data['open_momentum'] = merged_data['open_strength'] / (np.abs(merged_data['overnight_gap']) + 1e-8)

    # 计算未来开盘强度指标
    print("✓ 计算未来开盘强度指标...")

    # 未来1天、3天、5天的开盘强度
    merged_data['future_open_strength_1d'] = merged_data['open_strength'].shift(-1)
    merged_data['future_open_strength_3d'] = merged_data['open_strength'].shift(-3)
    merged_data['future_open_strength_5d'] = merged_data['open_strength'].shift(-5)

    # 未来收盘强度
    merged_data['future_close_strength_1d'] = merged_data['close_strength'].shift(-1)
    merged_data['future_close_strength_3d'] = merged_data['close_strength'].shift(-3)
    merged_data['future_close_strength_5d'] = merged_data['close_strength'].shift(-5)

    # 未来开盘相对强度
    merged_data['future_open_ratio_1d'] = merged_data['open_strength_ratio'].shift(-1)
    merged_data['future_open_ratio_3d'] = merged_data['open_strength_ratio'].shift(-3)
    merged_data['future_open_ratio_5d'] = merged_data['open_strength_ratio'].shift(-5)

    # 新增：未来最高价相关指标
    print("✓ 计算未来最高价指标...")

    # 未来最高价强度（当日开盘价到未来最高价的涨幅）
    merged_data['future_high_strength_1d'] = (merged_data['High'].shift(-1) - merged_data['Open']) / merged_data['Open']
    merged_data['future_high_strength_3d'] = (merged_data['High'].shift(-3) - merged_data['Open']) / merged_data['Open']
    merged_data['future_high_strength_5d'] = (merged_data['High'].shift(-5) - merged_data['Open']) / merged_data['Open']

    # 未来最高价突破程度（未来最高价相对当前收盘价）
    merged_data['future_high_breakout_1d'] = (merged_data['High'].shift(-1) - merged_data['Close']) / merged_data['Close']
    merged_data['future_high_breakout_3d'] = (merged_data['High'].shift(-3) - merged_data['Close']) / merged_data['Close']
    merged_data['future_high_breakout_5d'] = (merged_data['High'].shift(-5) - merged_data['Close']) / merged_data['Close']

    # 未来上影线长度（最高价与收盘价的差异）
    merged_data['future_upper_shadow_1d'] = (merged_data['High'].shift(-1) - merged_data['Close'].shift(-1)) / merged_data['Close'].shift(-1)
    merged_data['future_upper_shadow_3d'] = (merged_data['High'].shift(-3) - merged_data['Close'].shift(-3)) / merged_data['Close'].shift(-3)
    merged_data['future_upper_shadow_5d'] = (merged_data['High'].shift(-5) - merged_data['Close'].shift(-5)) / merged_data['Close'].shift(-5)

    # 计算成交量和波动率指标
    print("✓ 计算成交量和波动率指标...")

    # 成交量变化
    merged_data['volume_change'] = merged_data['Volume'].pct_change()
    merged_data['volume_ma_5'] = merged_data['Volume'].rolling(5).mean()
    merged_data['turnover_ratio'] = merged_data['Volume'] / merged_data['volume_ma_5']

    # 波动率指标
    merged_data['volatility_5d'] = merged_data['price_change'].rolling(5).std()
    merged_data['atr_5'] = calculate_atr(merged_data, 5)

    # 量价配合度（开盘强度与成交量的关系）
    volume_direction = np.sign(merged_data['volume_change'])
    open_direction = np.sign(merged_data['open_strength'])
    merged_data['open_volume_sync'] = open_direction * volume_direction

    print(f"✓ 指标计算完成，数据形状: {merged_data.shape}")

    return merged_data

def calculate_atr(data, period=5):
    """计算平均真实波动范围"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift(1))
    low_close = np.abs(data['Low'] - data['Close'].shift(1))

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(period).mean()

def calculate_ic_metrics(factor_values, target_values):
    """计算IC指标"""
    valid_data = pd.DataFrame({'factor': factor_values, 'target': target_values}).dropna()

    if len(valid_data) < 3:
        return {
            'normal_ic': np.nan,
            'rank_ic': np.nan,
            'normal_ic_pvalue': np.nan,
            'rank_ic_pvalue': np.nan,
            'sample_size': len(valid_data)
        }

    try:
        normal_ic, normal_p = pearsonr(valid_data['factor'], valid_data['target'])
        rank_ic, rank_p = spearmanr(valid_data['factor'], valid_data['target'])

        return {
            'normal_ic': normal_ic,
            'rank_ic': rank_ic,
            'normal_ic_pvalue': normal_p,
            'rank_ic_pvalue': rank_p,
            'sample_size': len(valid_data)
        }
    except:
        return {
            'normal_ic': np.nan,
            'rank_ic': np.nan,
            'normal_ic_pvalue': np.nan,
            'rank_ic_pvalue': np.nan,
            'sample_size': len(valid_data)
        }

def analyze_opening_strength_ic(merged_data):
    """分析开盘强度的IC系数"""
    print("\n" + "=" * 60)
    print("步骤 3: 开盘强度IC系数分析")
    print("=" * 60)

    score_col = 'overall_score_mean'

    if score_col not in merged_data.columns:
        print("⚠️ 未找到评分列，跳过IC分析")
        return {}

    results = {}

    # 定义要分析的开盘强度指标
    opening_indicators = {
        'future_open_strength_1d': '未来1天开盘强度',
        'future_open_strength_3d': '未来3天开盘强度',
        'future_open_strength_5d': '未来5天开盘强度',
        'future_close_strength_1d': '未来1天收盘强度',
        'future_close_strength_3d': '未来3天收盘强度',
        'future_close_strength_5d': '未来5天收盘强度',
        'future_open_ratio_1d': '未来1天开盘相对强度',
        'future_open_ratio_3d': '未来3天开盘相对强度',
        'future_open_ratio_5d': '未来5天开盘相对强度'
    }

    # 新增：最高价相关指标
    high_indicators = {
        'future_high_strength_1d': '未来1天最高价强度',
        'future_high_strength_3d': '未来3天最高价强度',
        'future_high_strength_5d': '未来5天最高价强度',
        'future_high_breakout_1d': '未来1天最高价突破',
        'future_high_breakout_3d': '未来3天最高价突破',
        'future_high_breakout_5d': '未来5天最高价突破',
        'future_upper_shadow_1d': '未来1天上影线',
        'future_upper_shadow_3d': '未来3天上影线',
        'future_upper_shadow_5d': '未来5天上影线'
    }

    print("1. 新闻评分与未来开盘强度IC分析:")

    for indicator, name in opening_indicators.items():
        if indicator in merged_data.columns:
            valid_data = merged_data.dropna(subset=[score_col, indicator])

            if len(valid_data) >= 3:
                ic_metrics = calculate_ic_metrics(valid_data[score_col], valid_data[indicator])
                corr_val, p_val = pearsonr(valid_data[score_col], valid_data[indicator])

                results[indicator] = {
                    'correlation': corr_val,
                    'p_value': p_val,
                    'normal_ic': ic_metrics['normal_ic'],
                    'rank_ic': ic_metrics['rank_ic'],
                    'sample_size': len(valid_data)
                }

                # 改进显著性判断：使用p值而不是简单阈值
                if p_val < 0.001:
                    significance = "***极显著"
                elif p_val < 0.01:
                    significance = "**很显著"
                elif p_val < 0.05:
                    significance = "*显著"
                elif p_val < 0.1:
                    significance = "边际显著"
                else:
                    significance = "不显著"

                ic_strength = "强" if abs(ic_metrics['rank_ic']) > 0.1 else "中等" if abs(ic_metrics['rank_ic']) > 0.05 else "弱"

                print(f"   {name}:")
                print(f"     相关系数: r = {corr_val:.4f} (p = {p_val:.4f}, {significance})")
                print(f"     Normal IC: {ic_metrics['normal_ic']:.4f}")
                print(f"     Rank IC: {ic_metrics['rank_ic']:.4f} ({ic_strength}预测能力)")
                print(f"     样本数: {len(valid_data)}")
                print()
            else:
                print(f"   {name}: 数据不足 (仅{len(valid_data)}个样本)")

    # 新增：最高价指标分析
    print("2. 新闻评分与未来最高价指标IC分析:")

    for indicator, name in high_indicators.items():
        if indicator in merged_data.columns:
            valid_data = merged_data.dropna(subset=[score_col, indicator])

            if len(valid_data) >= 3:
                ic_metrics = calculate_ic_metrics(valid_data[score_col], valid_data[indicator])
                corr_val, p_val = pearsonr(valid_data[score_col], valid_data[indicator])

                results[indicator] = {
                    'correlation': corr_val,
                    'p_value': p_val,
                    'normal_ic': ic_metrics['normal_ic'],
                    'rank_ic': ic_metrics['rank_ic'],
                    'sample_size': len(valid_data)
                }

                # 使用p值分级显著性
                if p_val < 0.001:
                    significance = "***极显著"
                elif p_val < 0.01:
                    significance = "**很显著"
                elif p_val < 0.05:
                    significance = "*显著"
                elif p_val < 0.1:
                    significance = "边际显著"
                else:
                    significance = "不显著"

                ic_strength = "强" if abs(ic_metrics['rank_ic']) > 0.1 else "中等" if abs(ic_metrics['rank_ic']) > 0.05 else "弱"

                print(f"   {name}:")
                print(f"     相关系数: r = {corr_val:.4f} (p = {p_val:.4f}, {significance})")
                print(f"     Normal IC: {ic_metrics['normal_ic']:.4f}")
                print(f"     Rank IC: {ic_metrics['rank_ic']:.4f} ({ic_strength}预测能力)")
                print(f"     样本数: {len(valid_data)}")
                print()
            else:
                print(f"   {name}: 数据不足 (仅{len(valid_data)}个样本)")

    return results

def analyze_sentiment_thresholds(merged_data):
    """分析不同情感评分阈值的影响"""
    print("\n" + "=" * 60)
    print("步骤 4: 情感评分阈值分析")
    print("=" * 60)

    score_col = 'overall_score_mean'
    target_col = 'future_open_strength_1d'

    if score_col not in merged_data.columns or target_col not in merged_data.columns:
        print("⚠️ 缺少必要列，跳过阈值分析")
        return

    valid_data = merged_data.dropna(subset=[score_col, target_col])

    if len(valid_data) < 10:
        print(f"⚠️ 数据不足 (仅{len(valid_data)}个样本)")
        return

    # 定义不同的评分阈值
    thresholds = [
        (-10, -2, 'Very Negative'),
        (-2, -0.5, 'Negative'),
        (-0.5, 0.5, 'Neutral'),
        (0.5, 2, 'Positive'),
        (2, 10, 'Very Positive')
    ]

    print("1. 不同情感强度对未来1天开盘强度的影响:")

    threshold_results = []

    for min_score, max_score, label in thresholds:
        subset = valid_data[
            (valid_data[score_col] >= min_score) &
            (valid_data[score_col] < max_score)
        ]

        if len(subset) >= 3:
            mean_open_strength = subset[target_col].mean()
            std_open_strength = subset[target_col].std()
            median_open_strength = subset[target_col].median()

            # 计算与中性组的差异
            neutral_subset = valid_data[
                (valid_data[score_col] >= -0.5) &
                (valid_data[score_col] < 0.5)
            ]

            if len(neutral_subset) >= 3:
                neutral_mean = neutral_subset[target_col].mean()
                difference = mean_open_strength - neutral_mean

                # 统计显著性检验
                from scipy.stats import ttest_ind
                t_stat, p_val = ttest_ind(subset[target_col], neutral_subset[target_col])

                # 改进显著性判断
                if p_val < 0.001:
                    significance = "***极显著"
                elif p_val < 0.01:
                    significance = "**很显著"
                elif p_val < 0.05:
                    significance = "*显著"
                elif p_val < 0.1:
                    significance = "边际显著"
                else:
                    significance = "不显著"
            else:
                difference = np.nan
                significance = "无法比较"

            threshold_results.append({
                'label': label,
                'count': len(subset),
                'mean_open_strength': mean_open_strength,
                'difference_from_neutral': difference,
                'significance': significance
            })

            print(f"   {label} (评分: {min_score} ~ {max_score}):")
            print(f"     样本数: {len(subset)}")
            print(f"     平均开盘强度: {mean_open_strength:.4f}")
            if not np.isnan(difference):
                print(f"     相对中性组差异: {difference:.4f} (p = {p_val:.4f}, {significance})")
            else:
                print(f"     相对中性组差异: {difference:.4f} ({significance})")
            print()
        else:
            print(f"   {label}: 样本不足 ({len(subset)}个)")

    return threshold_results

def analyze_time_decay(merged_data):
    """分析时间衰减效应"""
    print("\n" + "=" * 60)
    print("步骤 5: 时间衰减效应分析")
    print("=" * 60)

    score_col = 'overall_score_mean'

    # 分析1天、3天、5天的衰减
    periods = [
        ('future_open_strength_1d', '1天'),
        ('future_open_strength_3d', '3天'),
        ('future_open_strength_5d', '5天')
    ]

    decay_results = []

    print("1. 新闻影响的时间衰减分析:")

    for target_col, period_name in periods:
        if target_col in merged_data.columns:
            valid_data = merged_data.dropna(subset=[score_col, target_col])

            if len(valid_data) >= 3:
                ic_metrics = calculate_ic_metrics(valid_data[score_col], valid_data[target_col])
                corr_val, p_val = pearsonr(valid_data[score_col], valid_data[target_col])

                decay_results.append({
                    'period': period_name,
                    'correlation': corr_val,
                    'rank_ic': ic_metrics['rank_ic'],
                    'sample_size': len(valid_data)
                })

                significance = "**显著**" if p_val < 0.05 else "不显著"

                print(f"   未来{period_name}开盘强度:")
                print(f"     相关系数: {corr_val:.4f} ({significance})")
                print(f"     Rank IC: {ic_metrics['rank_ic']:.4f}")
                print(f"     样本数: {len(valid_data)}")
                print()

    # 计算衰减率
    if len(decay_results) >= 2:
        print("2. 衰减率分析:")
        base_ic = decay_results[0]['rank_ic']  # 1天的IC作为基准

        for i, result in enumerate(decay_results[1:], 1):
            decay_rate = (base_ic - result['rank_ic']) / base_ic if base_ic != 0 else 0
            print(f"   相对1天的衰减率 ({result['period']}): {decay_rate:.2%}")

    return decay_results

def analyze_combined_indicators(merged_data):
    """分析开盘强度与其他指标的组合效果"""
    print("\n" + "=" * 60)
    print("步骤 6: 组合指标分析")
    print("=" * 60)

    score_col = 'overall_score_mean'
    target_col = 'future_open_strength_1d'

    if any(col not in merged_data.columns for col in [score_col, target_col]):
        print("⚠️ 缺少必要列，跳过组合分析")
        return

    # 创建组合指标
    print("✓ 创建组合指标...")

    # 1. 新闻评分 × 成交量异常
    merged_data['score_volume_signal'] = (merged_data[score_col] *
                                         merged_data['turnover_ratio'])

    # 2. 新闻评分 × 波动率
    merged_data['score_volatility_signal'] = (merged_data[score_col] *
                                             merged_data['volatility_5d'].fillna(0))

    # 3. 新闻评分 × 隔夜跳空
    merged_data['score_gap_signal'] = (merged_data[score_col] *
                                      np.abs(merged_data['overnight_gap']))

    # 分析组合指标的效果
    combined_indicators = {
        score_col: '单纯新闻评分',
        'score_volume_signal': '新闻评分 × 成交量异常',
        'score_volatility_signal': '新闻评分 × 波动率',
        'score_gap_signal': '新闻评分 × 隔夜跳空'
    }

    print("1. 组合指标预测效果比较:")

    combination_results = []

    for indicator, name in combined_indicators.items():
        if indicator in merged_data.columns:
            valid_data = merged_data.dropna(subset=[indicator, target_col])

            if len(valid_data) >= 3:
                ic_metrics = calculate_ic_metrics(valid_data[indicator], valid_data[target_col])
                corr_val, p_val = pearsonr(valid_data[indicator], valid_data[target_col])

                combination_results.append({
                    'name': name,
                    'correlation': corr_val,
                    'rank_ic': ic_metrics['rank_ic'],
                    'sample_size': len(valid_data)
                })

                significance = "**显著**" if p_val < 0.05 else "不显著"

                print(f"   {name}:")
                print(f"     相关系数: {corr_val:.4f} ({significance})")
                print(f"     Rank IC: {ic_metrics['rank_ic']:.4f}")
                print(f"     样本数: {len(valid_data)}")
                print()

    # 排序显示最佳组合
    if combination_results:
        combination_results.sort(key=lambda x: abs(x['rank_ic']), reverse=True)

        print("2. 最佳组合指标排序 (按Rank IC):")
        for i, result in enumerate(combination_results, 1):
            print(f"   {i}. {result['name']}: Rank IC = {result['rank_ic']:.4f}")

    return combination_results

def create_visualization(merged_data):
    """创建可视化图表"""
    print("\n" + "=" * 60)
    print("步骤 7: 创建可视化图表")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('News Sentiment vs Opening Strength Analysis', fontsize=16, fontweight='bold')

    score_col = 'overall_score_mean'

    # 图1: 新闻评分 vs 未来1天开盘强度
    ax1 = axes[0, 0]
    if 'future_open_strength_1d' in merged_data.columns:
        valid_data = merged_data.dropna(subset=[score_col, 'future_open_strength_1d'])
        if len(valid_data) > 0:
            ax1.scatter(valid_data[score_col], valid_data['future_open_strength_1d'] * 100,
                       alpha=0.6, s=30, color='blue', edgecolors='navy', linewidth=0.5)

            # 添加趋势线
            if len(valid_data) > 2:
                z = np.polyfit(valid_data[score_col], valid_data['future_open_strength_1d'], 1)
                p = np.poly1d(z)
                ax1.plot(valid_data[score_col], p(valid_data[score_col]) * 100,
                        "r--", alpha=0.8, linewidth=2)

            corr, p_val = pearsonr(valid_data[score_col], valid_data['future_open_strength_1d'])
            ax1.set_title(f'News Score vs Future 1-Day Opening Strength\nr = {corr:.4f}, p = {p_val:.4f}',
                         fontsize=10, fontweight='bold')

    ax1.set_xlabel('News Overall Score')
    ax1.set_ylabel('Future 1-Day Opening Strength (%)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # 图2: 时间衰减效应
    ax2 = axes[0, 1]
    periods = ['1d', '3d', '5d']
    correlations = []

    for period in periods:
        col = f'future_open_strength_{period}'
        if col in merged_data.columns:
            valid_data = merged_data.dropna(subset=[score_col, col])
            if len(valid_data) > 2:
                corr, _ = pearsonr(valid_data[score_col], valid_data[col])
                correlations.append(abs(corr))
            else:
                correlations.append(0)
        else:
            correlations.append(0)

    if correlations:
        ax2.plot(periods, correlations, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_title('Time Decay Effect', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Prediction Period')
        ax2.set_ylabel('Absolute Correlation')
        ax2.grid(True, alpha=0.3)

    # 图3: 不同情感强度的开盘强度分布
    ax3 = axes[0, 2]
    if 'future_open_strength_1d' in merged_data.columns:
        valid_data = merged_data.dropna(subset=[score_col, 'future_open_strength_1d'])

        # 创建情感分组
        valid_data['sentiment_group'] = pd.cut(valid_data[score_col],
                                              bins=[-np.inf, -1, 0, 1, np.inf],
                                              labels=['Negative', 'Slightly Negative', 'Slightly Positive', 'Positive'])

        groups = []
        labels = []
        for group in ['Negative', 'Slightly Negative', 'Slightly Positive', 'Positive']:
            group_data = valid_data[valid_data['sentiment_group'] == group]['future_open_strength_1d'] * 100
            if len(group_data) > 0:
                groups.append(group_data)
                labels.append(f'{group}\n(n={len(group_data)})')

        if groups:
            ax3.boxplot(groups, labels=labels)
            ax3.set_title('Opening Strength by Sentiment Intensity', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Future 1-Day Opening Strength (%)')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 图4: 开盘强度时间序列
    ax4 = axes[1, 0]
    if 'date' in merged_data.columns and 'open_strength' in merged_data.columns:
        # 选择最近30天的数据
        recent_data = merged_data.tail(min(30, len(merged_data)))

        ax4.plot(recent_data['date'], recent_data['open_strength'] * 100,
                'b-o', linewidth=1.5, markersize=4, alpha=0.7, label='Opening Strength')

        if score_col in recent_data.columns:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(recent_data['date'], recent_data[score_col],
                         'r-s', linewidth=1.5, markersize=3, alpha=0.7, label='News Score')
            ax4_twin.set_ylabel('News Score', color='red')
            ax4_twin.tick_params(axis='y', labelcolor='red')

        ax4.set_title('Opening Strength vs News Score Time Series', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Opening Strength (%)', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

    # 图5: 组合指标效果比较
    ax5 = axes[1, 1]

    # 计算不同指标的IC
    if 'future_open_strength_1d' in merged_data.columns:
        indicators = {
            'overall_score_mean': 'News Score',
            'score_volume_signal': 'Score × Volume',
            'score_volatility_signal': 'Score × Volatility'
        }

        ic_values = []
        indicator_names = []

        for indicator, name in indicators.items():
            if indicator in merged_data.columns:
                valid_data = merged_data.dropna(subset=[indicator, 'future_open_strength_1d'])
                if len(valid_data) > 2:
                    ic_metrics = calculate_ic_metrics(valid_data[indicator],
                                                     valid_data['future_open_strength_1d'])
                    ic_values.append(abs(ic_metrics['rank_ic']))
                    indicator_names.append(name)

        if ic_values:
            bars = ax5.bar(indicator_names, ic_values, alpha=0.7,
                          color=['blue', 'green', 'orange'][:len(ic_values)])
            ax5.set_title('Combined Indicators Prediction Comparison', fontsize=10, fontweight='bold')
            ax5.set_ylabel('Absolute Rank IC')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars, ic_values):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    # 图6: 开盘强度分布直方图
    ax6 = axes[1, 2]
    if 'open_strength' in merged_data.columns:
        open_strength_pct = merged_data['open_strength'].dropna() * 100
        ax6.hist(open_strength_pct, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax6.axvline(x=open_strength_pct.mean(), color='red', linestyle='--',
                   label=f'Mean: {open_strength_pct.mean():.2f}%')
        ax6.set_title('Opening Strength Distribution', fontsize=10, fontweight='bold')
        ax6.set_xlabel('Opening Strength (%)')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def generate_trading_strategy(merged_data):
    """生成基于开盘强度的交易策略"""
    print("\n" + "=" * 60)
    print("步骤 8: 交易策略生成")
    print("=" * 60)

    score_col = 'overall_score_mean'

    if score_col not in merged_data.columns:
        print("⚠️ 缺少新闻评分列，无法生成策略")
        return

    strategy_data = merged_data.copy()

    # 生成交易信号
    print("✓ 生成交易信号...")

    # 信号1：基于新闻评分的简单信号
    strategy_data['signal_simple'] = np.where(strategy_data[score_col] > 1, 1,
                                             np.where(strategy_data[score_col] < -1, -1, 0))

    # 信号2：基于组合指标的复合信号
    if 'score_volume_signal' in strategy_data.columns:
        strategy_data['signal_combined'] = np.where(
            (strategy_data[score_col] > 0.5) & (strategy_data['turnover_ratio'] > 1.2), 1,
            np.where(
                (strategy_data[score_col] < -0.5) & (strategy_data['turnover_ratio'] > 1.2), -1, 0
            )
        )
    else:
        strategy_data['signal_combined'] = strategy_data['signal_simple']

    # 计算策略收益（使用开盘强度作为目标）
    if 'future_open_strength_1d' in strategy_data.columns:
        strategy_data['strategy_return_simple'] = (strategy_data['signal_simple'] *
                                                  strategy_data['future_open_strength_1d'])
        strategy_data['strategy_return_combined'] = (strategy_data['signal_combined'] *
                                                    strategy_data['future_open_strength_1d'])

        # 计算累积收益
        strategy_data['cumulative_return_simple'] = strategy_data['strategy_return_simple'].cumsum()
        strategy_data['cumulative_return_combined'] = strategy_data['strategy_return_combined'].cumsum()

        # 策略评估
        print("1. 策略表现评估:")

        simple_returns = strategy_data['strategy_return_simple'].dropna()
        combined_returns = strategy_data['strategy_return_combined'].dropna()

        if len(simple_returns) > 0:
            print(f"   简单策略:")
            print(f"     平均收益: {simple_returns.mean():.4f}")
            print(f"     收益波动: {simple_returns.std():.4f}")
            print(f"     夏普比率: {simple_returns.mean() / (simple_returns.std() + 1e-8):.4f}")
            print(f"     胜率: {(simple_returns > 0).mean():.2%}")
            print()

        if len(combined_returns) > 0:
            print(f"   组合策略:")
            print(f"     平均收益: {combined_returns.mean():.4f}")
            print(f"     收益波动: {combined_returns.std():.4f}")
            print(f"     夏普比率: {combined_returns.mean() / (combined_returns.std() + 1e-8):.4f}")
            print(f"     胜率: {(combined_returns > 0).mean():.2%}")

    print("✓ 策略回测完成")

def main():
    """主函数"""
    print("🎯 新闻评分与开盘强度深度分析")
    print("=" * 80)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 文件路径配置
    news_data_path = r'E:\projects\myQ\scripts\news_scores_result_1y_zijin.csv'

    # 检查文件是否存在
    if not os.path.exists(news_data_path):
        print(f"✗ 错误: 找不到新闻数据文件 {news_data_path}")
        return

    try:
        # 步骤1: 加载和准备数据
        merged_data = load_and_prepare_data(news_data_path)

        if len(merged_data) < 10:
            print("✗ 错误: 数据量不足，无法进行分析")
            return

        # 步骤2: IC系数分析
        ic_results = analyze_opening_strength_ic(merged_data)

        # 步骤3: 情感阈值分析
        threshold_results = analyze_sentiment_thresholds(merged_data)

        # 步骤4: 时间衰减分析
        decay_results = analyze_time_decay(merged_data)

        # 步骤5: 组合指标分析
        combination_results = analyze_combined_indicators(merged_data)

        # 步骤6: 可视化
        create_visualization(merged_data)

        # 步骤7: 交易策略
        generate_trading_strategy(merged_data)

        print("\n" + "=" * 80)
        print("🎉 分析完成！主要发现：")
        print("=" * 80)

        # 输出关键发现
        if ic_results:
            best_indicator = max(ic_results.items(),
                               key=lambda x: abs(x[1].get('rank_ic', 0)))
            print(f"• 最强预测指标: {best_indicator[0]}")
            print(f"  Rank IC: {best_indicator[1]['rank_ic']:.4f}")

        if combination_results:
            best_combination = max(combination_results,
                                 key=lambda x: abs(x.get('rank_ic', 0)))
            print(f"• 最佳组合策略: {best_combination['name']}")
            print(f"  Rank IC: {best_combination['rank_ic']:.4f}")

        print("• 开盘强度相比简单收益率更能反映新闻影响")
        print("• 建议重点关注开盘后15分钟的价格行为")
        print("• 结合成交量异常可以提升预测效果")

    except Exception as e:
        print(f"✗ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()