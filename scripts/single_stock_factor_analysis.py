#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单只股票综合因子分析系统
========================

结合新闻情绪因子和传统技术因子，分析与收益率、开盘强度的关系

功能：
1. 新闻情绪因子处理与聚合
2. 传统技术因子计算（动量、成交量、波动率）
3. 开盘强度因子计算
4. 因子有效性分析（IC、RankIC）
5. 因子组合与预测效果评估
6. 可视化分析报告

作者: Claude Code
日期: 2025-09-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
import sys
import os
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(r'D:\projects\q\myQ')

# 导入市场数据模块
try:
    from quantlib.market_data import MarketDataManager
    MARKET_DATA_AVAILABLE = True
    print("✓ 市场数据模块导入成功")
except ImportError as e:
    print(f"⚠️ 无法导入市场数据模块: {e}")
    MARKET_DATA_AVAILABLE = False

# 设置图表样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 10

class SingleStockFactorAnalyzer:
    def __init__(self, sentiment_file_path, stock_code='601899'):
        """
        初始化单股票因子分析器

        Args:
            sentiment_file_path (str): 情绪数据文件路径
            stock_code (str): 股票代码
        """
        self.sentiment_file_path = sentiment_file_path
        self.stock_code = stock_code
        self.stock_name = self._get_stock_name()

        print(f"🎯 初始化单股票因子分析器")
        print(f"   股票: {self.stock_name} ({self.stock_code})")
        print(f"   情绪数据: {sentiment_file_path}")

    def _get_stock_name(self):
        """根据股票代码获取股票名称"""
        stock_names = {
            '601899': '紫金矿业',
            '002847': '盐津铺子',
            '000001': '平安银行',
            # 可以继续添加其他股票
        }
        return stock_names.get(self.stock_code, f'股票_{self.stock_code}')

    def load_sentiment_data(self):
        """加载和预处理情绪数据"""
        print("\n" + "=" * 60)
        print("步骤 1: 加载情绪因子数据")
        print("=" * 60)

        # 读取情绪数据
        sentiment_df = pd.read_csv(self.sentiment_file_path, encoding='utf-8-sig')
        print(f"✓ 加载情绪数据: {len(sentiment_df)} 条记录")

        # 转换日期格式
        sentiment_df['date'] = pd.to_datetime(sentiment_df['original_date']).dt.date
        print(f"✓ 日期范围: {sentiment_df['date'].min()} 到 {sentiment_df['date'].max()}")

        # 聚合日度情绪数据
        daily_sentiment = sentiment_df.groupby('date').agg({
            'overall_score': ['mean', 'std', 'count', 'min', 'max', 'sum'],
            'direct_impact_score': ['mean', 'std'],
            'indirect_impact_score': ['mean', 'std'],
            'certainty': ['mean', 'std', 'min', 'max'],
            'sentiment': lambda x: x.mode()[0] if not x.empty else 'neutral'
        }).round(4)

        # 扁平化列名
        daily_sentiment.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0]
                                 for col in daily_sentiment.columns]
        daily_sentiment = daily_sentiment.reset_index()

        print(f"✓ 日度聚合: {len(daily_sentiment)} 天")

        # 计算情绪因子
        print("✓ 计算情绪衍生因子...")

        # 按日期排序
        daily_sentiment = daily_sentiment.sort_values('date').reset_index(drop=True)

        # 1. 基础情绪因子
        daily_sentiment['news_intensity'] = daily_sentiment['overall_score_count']  # 新闻强度
        daily_sentiment['sentiment_strength'] = abs(daily_sentiment['overall_score_mean'])  # 情绪强度
        daily_sentiment['weighted_sentiment'] = (daily_sentiment['overall_score_mean'] *
                                               daily_sentiment['certainty_mean'])  # 确定性加权情绪

        # 2. 情绪变化因子
        daily_sentiment['sentiment_change_1d'] = daily_sentiment['overall_score_mean'].diff()
        daily_sentiment['sentiment_change_3d'] = daily_sentiment['overall_score_mean'].diff(3)
        daily_sentiment['sentiment_momentum'] = daily_sentiment['overall_score_mean'].rolling(3).mean()
        daily_sentiment['sentiment_volatility'] = daily_sentiment['overall_score_mean'].rolling(5).std()

        # 3. 情绪极值因子
        daily_sentiment['sentiment_max_impact'] = daily_sentiment['overall_score_max']
        daily_sentiment['sentiment_min_impact'] = daily_sentiment['overall_score_min']
        daily_sentiment['sentiment_range'] = (daily_sentiment['overall_score_max'] -
                                            daily_sentiment['overall_score_min'])

        # 4. 情绪一致性因子
        daily_sentiment['sentiment_consistency'] = (1 - daily_sentiment['overall_score_std'].fillna(0))
        daily_sentiment['certainty_strength'] = daily_sentiment['certainty_mean']

        print(f"✓ 情绪因子计算完成")

        self.sentiment_data = daily_sentiment
        return daily_sentiment

    def load_market_data(self):
        """获取市场数据"""
        print("\n" + "=" * 60)
        print("步骤 2: 获取市场数据")
        print("=" * 60)

        if not MARKET_DATA_AVAILABLE:
            print("⚠️ 市场数据模块不可用，使用模拟数据")
            return self._create_simulated_market_data()

        try:
            manager = MarketDataManager()

            # 确定数据时间范围
            start_date = self.sentiment_data['date'].min() - timedelta(days=30)
            end_date = self.sentiment_data['date'].max() + timedelta(days=10)

            print(f"✓ 获取股价数据: {start_date} 到 {end_date}")

            # 选择合适的period
            days_span = (end_date - start_date).days
            if days_span <= 95:
                period = '6mo'
            elif days_span <= 185:
                period = '1y'
            else:
                period = '2y'

            print(f"✓ 使用period: {period}")

            # 获取股价数据
            stock_data = manager.get_stock_data(
                symbol=self.stock_code,
                market='CN',
                period=period,
                interval='1d'
            )

            if stock_data is not None and len(stock_data) > 0:
                stock_data = stock_data.reset_index()
                stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date

                # 重命名列名
                column_mapping = {}
                for col in stock_data.columns:
                    if col.lower() == 'open':
                        column_mapping[col] = 'Open'
                    elif col.lower() == 'high':
                        column_mapping[col] = 'High'
                    elif col.lower() == 'low':
                        column_mapping[col] = 'Low'
                    elif col.lower() == 'close':
                        column_mapping[col] = 'Close'
                    elif col.lower() == 'volume':
                        column_mapping[col] = 'Volume'

                stock_data = stock_data.rename(columns=column_mapping)

                print(f"✓ 股价数据: {len(stock_data)} 天")
                print(f"✓ 列名: {list(stock_data.columns)}")

                # 筛选到情绪数据时间范围
                sentiment_start = self.sentiment_data['date'].min()
                sentiment_end = self.sentiment_data['date'].max()

                stock_filtered = stock_data[
                    (stock_data['date'] >= sentiment_start) &
                    (stock_data['date'] <= sentiment_end)
                ].copy()

                if len(stock_filtered) > 0:
                    print(f"✓ 筛选后股价数据: {len(stock_filtered)} 天")
                    self.market_data = stock_filtered
                    return stock_filtered
                else:
                    print("⚠️ 筛选后无数据，使用模拟数据")
                    return self._create_simulated_market_data()
            else:
                print("⚠️ 未获取到股价数据，使用模拟数据")
                return self._create_simulated_market_data()

        except Exception as e:
            print(f"⚠️ 获取市场数据失败: {e}")
            return self._create_simulated_market_data()

    def _create_simulated_market_data(self):
        """创建模拟市场数据"""
        print("✓ 生成模拟市场数据...")

        start_date = self.sentiment_data['date'].min() - timedelta(days=10)
        end_date = self.sentiment_data['date'].max() + timedelta(days=5)

        # 生成交易日
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d.date() for d in date_range if d.weekday() < 5]

        np.random.seed(42)
        n_days = len(trading_days)

        # 模拟股价数据
        returns = np.random.normal(0.002, 0.025, n_days)
        prices = 15.0 * np.cumprod(1 + returns)  # 紫金矿业大概15元左右

        market_data = pd.DataFrame({
            'date': trading_days,
            'Open': prices * (1 + np.random.normal(0, 0.003, n_days)),
            'High': prices * (1 + np.abs(np.random.normal(0.01, 0.008, n_days))),
            'Low': prices * (1 - np.abs(np.random.normal(0.01, 0.008, n_days))),
            'Close': prices,
            'Volume': np.random.lognormal(16, 0.4, n_days)  # 调整成交量
        })

        # 确保价格逻辑正确
        market_data['High'] = np.maximum(market_data['High'],
                                       np.maximum(market_data['Open'], market_data['Close']))
        market_data['Low'] = np.minimum(market_data['Low'],
                                      np.minimum(market_data['Open'], market_data['Close']))

        print(f"✓ 模拟数据: {len(market_data)} 天")
        self.market_data = market_data
        return market_data

    def calculate_traditional_factors(self):
        """计算传统技术因子"""
        print("\n" + "=" * 60)
        print("步骤 3: 计算传统技术因子")
        print("=" * 60)

        market_data = self.market_data.copy()

        # 1. 价格因子
        print("✓ 计算价格因子...")
        market_data['price_change'] = market_data['Close'].pct_change()
        market_data['price_change_3d'] = market_data['Close'].pct_change(3)
        market_data['price_change_5d'] = market_data['Close'].pct_change(5)
        market_data['price_change_10d'] = market_data['Close'].pct_change(10)

        # 2. 增强动量因子
        print("✓ 计算动量因子...")
        # 短期动量
        market_data['momentum_3d'] = market_data['Close'].rolling(3).mean() / market_data['Close'].rolling(10).mean() - 1
        market_data['momentum_5d'] = market_data['Close'].rolling(5).mean() / market_data['Close'].rolling(20).mean() - 1
        market_data['momentum_10d'] = market_data['Close'].rolling(10).mean() / market_data['Close'].rolling(30).mean() - 1

        # 长期动量
        market_data['momentum_20d'] = market_data['Close'].rolling(20).mean() / market_data['Close'].rolling(60).mean() - 1
        market_data['momentum_60d'] = market_data['Close'].rolling(60).mean() / market_data['Close'].rolling(120).mean() - 1

        # 动量加速度（动量变化率）
        market_data['momentum_acceleration_3d'] = market_data['momentum_3d'] - market_data['momentum_3d'].shift(1)
        market_data['momentum_acceleration_5d'] = market_data['momentum_5d'] - market_data['momentum_5d'].shift(1)

        # 价格动量（简单收益率动量）
        market_data['price_momentum_5d'] = market_data['Close'] / market_data['Close'].shift(5) - 1
        market_data['price_momentum_20d'] = market_data['Close'] / market_data['Close'].shift(20) - 1
        market_data['price_momentum_60d'] = market_data['Close'] / market_data['Close'].shift(60) - 1

        # RSI (Relative Strength Index)
        delta = market_data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        market_data['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = market_data['Close'].ewm(span=12).mean()
        ema26 = market_data['Close'].ewm(span=26).mean()
        market_data['macd'] = ema12 - ema26
        market_data['macd_signal'] = market_data['macd'].ewm(span=9).mean()
        market_data['macd_histogram'] = market_data['macd'] - market_data['macd_signal']

        # 3. 增强成交量因子
        print("✓ 计算成交量因子...")
        market_data['volume_change'] = market_data['Volume'].pct_change()
        market_data['volume_ma_5'] = market_data['Volume'].rolling(5).mean()
        market_data['volume_ma_20'] = market_data['Volume'].rolling(20).mean()
        market_data['volume_ma_60'] = market_data['Volume'].rolling(60).mean()

        # 成交量比率（用于后续与新闻因子交互）
        market_data['volume_ratio_5d'] = market_data['Volume'] / (market_data['volume_ma_5'] + 1e-8)
        market_data['volume_ratio_20d'] = market_data['Volume'] / (market_data['volume_ma_20'] + 1e-8)
        market_data['volume_ratio_60d'] = market_data['Volume'] / (market_data['volume_ma_60'] + 1e-8)

        # 异常成交量（用于过滤新闻噪声）
        market_data['abnormal_volume_5d'] = (market_data['Volume'] - market_data['volume_ma_5']) / (market_data['volume_ma_5'] + 1e-8)
        market_data['abnormal_volume_20d'] = (market_data['Volume'] - market_data['volume_ma_20']) / (market_data['volume_ma_20'] + 1e-8)

        # 成交量动量
        market_data['volume_momentum_5d'] = market_data['volume_ma_5'] / market_data['volume_ma_20'] - 1
        market_data['volume_momentum_20d'] = market_data['volume_ma_20'] / market_data['volume_ma_60'] - 1

        # 量价配合度
        price_direction = np.sign(market_data['price_change'])
        volume_direction = np.sign(market_data['volume_change'])
        market_data['volume_price_sync'] = price_direction * volume_direction

        # OBV (On-Balance Volume) - 资金流向指标
        obv = np.zeros(len(market_data))
        for i in range(1, len(market_data)):
            if market_data['Close'].iloc[i] > market_data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] + market_data['Volume'].iloc[i]
            elif market_data['Close'].iloc[i] < market_data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] - market_data['Volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        market_data['obv'] = obv
        market_data['obv_ma_5'] = market_data['obv'].rolling(5).mean()
        market_data['obv_momentum'] = market_data['obv'] / (market_data['obv_ma_5'] + 1e-8) - 1

        # 5. 波动率因子
        print("✓ 计算波动率因子...")
        market_data['volatility_5d'] = market_data['price_change'].rolling(5).std()
        market_data['volatility_10d'] = market_data['price_change'].rolling(10).std()
        market_data['volatility_20d'] = market_data['price_change'].rolling(20).std()

        # ATR (Average True Range)
        high_low = market_data['High'] - market_data['Low']
        high_close = np.abs(market_data['High'] - market_data['Close'].shift(1))
        low_close = np.abs(market_data['Low'] - market_data['Close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        market_data['atr_5'] = true_range.rolling(5).mean()
        market_data['atr_10'] = true_range.rolling(10).mean()

        # 相对波动率
        market_data['relative_volatility'] = market_data['volatility_5d'] / market_data['volatility_20d']

        print(f"✓ 传统因子计算完成，数据形状: {market_data.shape}")

        self.market_data = market_data
        return market_data

    def merge_factors(self):
        """合并情绪因子和传统因子"""
        print("\n" + "=" * 60)
        print("步骤 4: 合并因子数据")
        print("=" * 60)

        # 合并数据
        merged_data = pd.merge(self.sentiment_data, self.market_data, on='date', how='inner')
        print(f"✓ 合并后数据: {len(merged_data)} 天")

        if len(merged_data) == 0:
            print("❌ 合并后数据为空，请检查日期匹配")
            return None

        # 计算未来收益率（目标变量）
        print("✓ 计算目标变量...")
        merged_data['future_return_1d'] = merged_data['Close'].shift(-1) / merged_data['Close'] - 1
        merged_data['future_return_3d'] = merged_data['Close'].shift(-3) / merged_data['Close'] - 1
        merged_data['future_return_5d'] = merged_data['Close'].shift(-5) / merged_data['Close'] - 1

        # 计算开盘表现（开盘相对前收盘的表现）
        merged_data['open_performance_1d'] = (merged_data['Open'].shift(-1) / merged_data['Close'] - 1)
        merged_data['open_performance_3d'] = (merged_data['Open'].shift(-3) / merged_data['Close'] - 1)

        # 计算新闻-成交量交互因子
        print("✓ 计算新闻-成交量交互因子...")

        # 1. 基础交互：新闻情绪 * 成交量异常
        merged_data['news_vol_interaction_5d'] = merged_data['overall_score_mean'] * merged_data['abnormal_volume_5d']
        merged_data['news_vol_interaction_20d'] = merged_data['overall_score_mean'] * merged_data['abnormal_volume_20d']

        # 2. 情绪强度过滤：只有在成交量放大时，正面/负面情绪才有效
        # 正面新闻 + 成交量放大 = 更可信的正面信号
        positive_sentiment = np.where(merged_data['overall_score_mean'] > 0, merged_data['overall_score_mean'], 0)
        negative_sentiment = np.where(merged_data['overall_score_mean'] < 0, merged_data['overall_score_mean'], 0)
        volume_amplification = np.where(merged_data['volume_ratio_5d'] > 1.2, merged_data['volume_ratio_5d'], 0)

        merged_data['filtered_positive_news'] = positive_sentiment * volume_amplification
        merged_data['filtered_negative_news'] = negative_sentiment * volume_amplification

        # 3. 基于确定性的过滤：高确定性新闻 * 资金响应
        merged_data['certainty_vol_factor'] = merged_data['certainty_mean'] * merged_data['obv_momentum']

        # 4. 新闻驱动的资金流入流出强度
        merged_data['news_money_flow'] = merged_data['sentiment_strength'] * merged_data['volume_price_sync']

        # 5. 综合新闻-成交量因子：考虑情绪方向、强度和资金响应
        merged_data['comprehensive_news_vol'] = (
            merged_data['overall_score_mean'] *
            merged_data['certainty_mean'] *
            merged_data['volume_ratio_20d'] *
            np.sign(merged_data['volume_change'])
        )

        print(f"✓ 目标变量计算完成")

        self.merged_data = merged_data
        return merged_data

    def calculate_factor_ic(self, factor_col, target_col, min_samples=10):
        """计算因子IC指标"""
        valid_data = self.merged_data.dropna(subset=[factor_col, target_col])

        if len(valid_data) < min_samples:
            return {
                'normal_ic': np.nan,
                'rank_ic': np.nan,
                'p_value': np.nan,
                'sample_size': len(valid_data)
            }

        try:
            # 计算相关系数
            normal_ic, p_value = pearsonr(valid_data[factor_col], valid_data[target_col])
            rank_ic, _ = spearmanr(valid_data[factor_col], valid_data[target_col])

            return {
                'normal_ic': normal_ic,
                'rank_ic': rank_ic,
                'p_value': p_value,
                'sample_size': len(valid_data)
            }
        except:
            return {
                'normal_ic': np.nan,
                'rank_ic': np.nan,
                'p_value': np.nan,
                'sample_size': len(valid_data)
            }

    def analyze_factor_effectiveness(self):
        """分析因子有效性"""
        print("\n" + "=" * 60)
        print("步骤 5: 因子有效性分析")
        print("=" * 60)

        # 定义因子组
        sentiment_factors = {
            'overall_score_mean': 'Average Sentiment Score',
            'sentiment_strength': 'Sentiment Strength',
            'weighted_sentiment': 'Certainty-Weighted Sentiment',
            'sentiment_change_1d': '1-day Sentiment Change',
            'sentiment_change_3d': '3-day Sentiment Change',
            'sentiment_momentum': 'Sentiment Momentum',
            'sentiment_volatility': 'Sentiment Volatility',
            'news_intensity': 'News Intensity',
            'sentiment_consistency': 'Sentiment Consistency'
        }

        traditional_factors = {
            'momentum_3d': '3-day Momentum',
            'momentum_5d': '5-day Momentum',
            'momentum_10d': '10-day Momentum',
            'momentum_20d': '20-day Momentum',
            'momentum_60d': '60-day Momentum',
            'momentum_acceleration_3d': '3-day Momentum Acceleration',
            'momentum_acceleration_5d': '5-day Momentum Acceleration',
            'price_momentum_5d': '5-day Price Momentum',
            'price_momentum_20d': '20-day Price Momentum',
            'price_momentum_60d': '60-day Price Momentum',
            'rsi': 'RSI',
            'macd': 'MACD',
            'macd_histogram': 'MACD Histogram',
            'volume_ratio_5d': '5-day Volume Ratio',
            'volume_ratio_20d': '20-day Volume Ratio',
            'abnormal_volume_5d': '5-day Abnormal Volume',
            'volume_momentum_5d': '5-day Volume Momentum',
            'volume_price_sync': 'Volume-Price Sync',
            'obv_momentum': 'OBV Momentum',
            'volatility_5d': '5-day Volatility',
            'relative_volatility': 'Relative Volatility'
        }

        # 新闻-成交量交互因子
        news_volume_factors = {
            'news_vol_interaction_5d': 'News-Volume Interaction 5d',
            'news_vol_interaction_20d': 'News-Volume Interaction 20d',
            'filtered_positive_news': 'Volume-Filtered Positive News',
            'filtered_negative_news': 'Volume-Filtered Negative News',
            'certainty_vol_factor': 'Certainty-Volume Factor',
            'news_money_flow': 'News-driven Money Flow',
            'comprehensive_news_vol': 'Comprehensive News-Volume Factor'
        }

        # 目标变量
        targets = {
            'future_return_1d': 'Future 1-day Return',
            'future_return_3d': 'Future 3-day Return',
            'future_return_5d': 'Future 5-day Return',
            'open_performance_1d': 'Next-day Open Performance',
            'open_performance_3d': '3-day Open Performance'
        }

        # 存储结果
        ic_results = {}

        # 分析情绪因子
        print("1. Sentiment Factor Effectiveness Analysis:")
        sentiment_results = []
        for factor, factor_name in sentiment_factors.items():
            if factor in self.merged_data.columns:
                factor_result = {'factor': factor_name, 'factor_code': factor}
                for target, target_name in targets.items():
                    if target in self.merged_data.columns:
                        ic_metrics = self.calculate_factor_ic(factor, target)
                        factor_result[f'{target}_ic'] = ic_metrics['rank_ic']
                        factor_result[f'{target}_pval'] = ic_metrics['p_value']

                        # 存储到总结果
                        key = f"{factor}_{target}"
                        ic_results[key] = ic_metrics

                sentiment_results.append(factor_result)

                # 显示最佳目标
                best_ic = -float('inf')
                best_target = None
                for target in targets.keys():
                    if f'{target}_ic' in factor_result and not pd.isna(factor_result[f'{target}_ic']):
                        ic_val = abs(factor_result[f'{target}_ic'])
                        if ic_val > best_ic:
                            best_ic = ic_val
                            best_target = target

                if best_target:
                    ic_val = factor_result[f'{best_target}_ic']
                    p_val = factor_result[f'{best_target}_pval']
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"   {factor_name}: Best Prediction {targets[best_target]} (IC={ic_val:.4f}{significance})")

        # 分析传统因子
        print("\n2. Traditional Technical Factor Effectiveness Analysis:")
        traditional_results = []
        for factor, factor_name in traditional_factors.items():
            if factor in self.merged_data.columns:
                factor_result = {'factor': factor_name, 'factor_code': factor}
                for target, target_name in targets.items():
                    if target in self.merged_data.columns:
                        ic_metrics = self.calculate_factor_ic(factor, target)
                        factor_result[f'{target}_ic'] = ic_metrics['rank_ic']
                        factor_result[f'{target}_pval'] = ic_metrics['p_value']

                        key = f"{factor}_{target}"
                        ic_results[key] = ic_metrics

                traditional_results.append(factor_result)

                # 显示最佳目标
                best_ic = -float('inf')
                best_target = None
                for target in targets.keys():
                    if f'{target}_ic' in factor_result and not pd.isna(factor_result[f'{target}_ic']):
                        ic_val = abs(factor_result[f'{target}_ic'])
                        if ic_val > best_ic:
                            best_ic = ic_val
                            best_target = target

                if best_target:
                    ic_val = factor_result[f'{best_target}_ic']
                    p_val = factor_result[f'{best_target}_pval']
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"   {factor_name}: Best Prediction {targets[best_target]} (IC={ic_val:.4f}{significance})")

        # 分析新闻-成交量交互因子
        print("\n3. News-Volume Interaction Factor Analysis:")
        news_volume_results = []
        for factor, factor_name in news_volume_factors.items():
            if factor in self.merged_data.columns:
                factor_result = {'factor': factor_name, 'factor_code': factor}
                for target, target_name in targets.items():
                    if target in self.merged_data.columns:
                        ic_metrics = self.calculate_factor_ic(factor, target)
                        factor_result[f'{target}_ic'] = ic_metrics['rank_ic']
                        factor_result[f'{target}_pval'] = ic_metrics['p_value']

                        key = f"{factor}_{target}"
                        ic_results[key] = ic_metrics

                news_volume_results.append(factor_result)

                # 显示最佳目标
                best_ic = -float('inf')
                best_target = None
                for target in targets.keys():
                    if f'{target}_ic' in factor_result and not pd.isna(factor_result[f'{target}_ic']):
                        ic_val = abs(factor_result[f'{target}_ic'])
                        if ic_val > best_ic:
                            best_ic = ic_val
                            best_target = target

                if best_target:
                    ic_val = factor_result[f'{best_target}_ic']
                    p_val = factor_result[f'{best_target}_pval']
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"   {factor_name}: Best Prediction {targets[best_target]} (IC={ic_val:.4f}{significance})")

        # 综合排序
        print("\n4. All Factor Ranking (By Maximum IC):")
        all_results = sentiment_results + traditional_results + news_volume_results

        # 计算每个因子的最强IC
        for result in all_results:
            max_ic = 0
            for target in targets.keys():
                ic_key = f'{target}_ic'
                if ic_key in result and not pd.isna(result[ic_key]):
                    max_ic = max(max_ic, abs(result[ic_key]))
            result['max_ic'] = max_ic

        # 排序并显示前10名
        all_results.sort(key=lambda x: x['max_ic'], reverse=True)

        for i, result in enumerate(all_results[:10], 1):
            print(f"   {i:2d}. {result['factor']:30s}: Max IC = {result['max_ic']:.4f}")

        self.ic_results = ic_results
        self.factor_results = {
            'sentiment': sentiment_results,
            'traditional': traditional_results,
            'news_volume': news_volume_results,
            'all': all_results
        }

        return ic_results, all_results

    def analyze_factor_combinations(self):
        """分析因子组合效果"""
        print("\n" + "=" * 60)
        print("步骤 6: 因子组合分析")
        print("=" * 60)

        # 检查是否有有效的因子结果
        if 'all' not in self.factor_results or not self.factor_results['all']:
            print("⚠️ 没有有效的因子结果，跳过组合分析")
            return self.merged_data

        # 选择最佳因子进行组合
        top_factors = self.factor_results['all'][:5]  # 前5个最佳因子

        print("✓ 分析顶级因子组合效果...")
        print("前5个最佳因子:")
        for i, factor in enumerate(top_factors, 1):
            print(f"   {i}. {factor['factor']} (最强IC: {factor['max_ic']:.4f})")

        # 计算组合因子
        merged_data = self.merged_data.copy()

        # 简单等权重组合
        combination_factors = []
        valid_factors = []
        for factor in top_factors:
            factor_code = factor['factor_code']
            if factor_code in merged_data.columns:
                # 标准化因子值
                factor_values = merged_data[factor_code].fillna(0)
                if factor_values.std() > 1e-8:  # 确保因子有变化
                    factor_std = (factor_values - factor_values.mean()) / factor_values.std()
                    combination_factors.append(factor_std)
                    valid_factors.append(factor)
                    print(f"   ✓ 添加因子: {factor['factor']}")
                else:
                    print(f"   ⚠️ 跳过无变化因子: {factor['factor']}")
            else:
                print(f"   ⚠️ 因子列不存在: {factor_code}")

        if len(combination_factors) == 0:
            print("⚠️ 没有有效的因子用于组合，跳过组合分析")
            return merged_data

        print(f"✓ 成功组合 {len(combination_factors)} 个有效因子")

        if combination_factors:
            # 等权重组合
            merged_data['combined_factor_equal'] = np.mean(combination_factors, axis=0)

            # IC加权组合（按最强IC加权）
            weights = [abs(factor['max_ic']) for factor in valid_factors]  # 使用绝对值权重
            if sum(weights) > 0:
                weights = np.array(weights) / sum(weights)  # 归一化权重
                merged_data['combined_factor_ic_weighted'] = np.average(combination_factors, weights=weights, axis=0)
            else:
                merged_data['combined_factor_ic_weighted'] = merged_data['combined_factor_equal'].copy()

            # 立即更新self.merged_data，以便后续的calculate_factor_ic可以使用新创建的列
            self.merged_data = merged_data

            print("\n✓ 组合因子效果评估:")

            targets = ['future_return_1d', 'future_return_3d', 'open_performance_1d']

            for target in targets:
                if target in merged_data.columns:
                    print(f"\n   预测目标: {target}")

                    # 等权重组合
                    equal_ic = self.calculate_factor_ic('combined_factor_equal', target)
                    print(f"     等权重组合: IC={equal_ic['rank_ic']:.4f} (p={equal_ic['p_value']:.4f}, n={equal_ic['sample_size']})")

                    # IC加权组合
                    weighted_ic = self.calculate_factor_ic('combined_factor_ic_weighted', target)
                    print(f"     IC加权组合: IC={weighted_ic['rank_ic']:.4f} (p={weighted_ic['p_value']:.4f}, n={weighted_ic['sample_size']})")

                    # 对比最佳单因子
                    target_ics = [factor.get(f'{target}_ic', 0) for factor in valid_factors if not pd.isna(factor.get(f'{target}_ic', np.nan))]
                    best_single_ic = max(target_ics) if target_ics else 0
                    print(f"     最佳单因子: IC={best_single_ic:.4f}")

                    # 提升效果
                    equal_improvement = abs(equal_ic['rank_ic']) - abs(best_single_ic)
                    weighted_improvement = abs(weighted_ic['rank_ic']) - abs(best_single_ic)

                    print(f"     等权重提升: {equal_improvement:+.4f}")
                    print(f"     加权组合提升: {weighted_improvement:+.4f}")

        return self.merged_data

    def create_visualizations(self):
        """创建可视化分析"""
        print("\n" + "=" * 60)
        print("Step 7: Generate Visual Analysis")
        print("=" * 60)

        fig = plt.figure(figsize=(20, 16))

        # 1. Factor IC Heatmap
        ax1 = plt.subplot(3, 4, 1)
        self._plot_factor_ic_heatmap(ax1)

        # 2. Sentiment Score vs Future Returns Scatter
        ax2 = plt.subplot(3, 4, 2)
        self._plot_sentiment_return_scatter(ax2)

        # 3. Time Series: Sentiment vs Price
        ax3 = plt.subplot(3, 4, 3)
        self._plot_sentiment_price_timeseries(ax3)

        # 4. Factor Effectiveness Bar Chart
        ax4 = plt.subplot(3, 4, 4)
        self._plot_factor_effectiveness_bar(ax4)

        # 5. News-Volume Interaction Analysis
        ax5 = plt.subplot(3, 4, 5)
        self._plot_news_volume_interaction(ax5)

        # 6. Sentiment Grouped Returns Boxplot
        ax6 = plt.subplot(3, 4, 6)
        self._plot_sentiment_group_returns(ax6)

        # 7. Combined Factor Effect Comparison
        ax7 = plt.subplot(3, 4, 7)
        self._plot_combination_factor_comparison(ax7)

        # 8. Factor Stability Analysis
        ax8 = plt.subplot(3, 4, 8)
        self._plot_factor_stability(ax8)

        # 9. News Intensity vs Returns Relationship
        ax9 = plt.subplot(3, 4, 9)
        self._plot_news_intensity_returns(ax9)

        # 10. Volatility Factor Analysis
        ax10 = plt.subplot(3, 4, 10)
        self._plot_volatility_factor_analysis(ax10)

        # 11. Volume-Price Synchronization Analysis
        ax11 = plt.subplot(3, 4, 11)
        self._plot_volume_price_sync(ax11)

        # 12. Comprehensive Factor Prediction Time Series
        ax12 = plt.subplot(3, 4, 12)
        self._plot_prediction_timeseries(ax12)

        plt.tight_layout()
        plt.suptitle(f'{self.stock_name} ({self.stock_code}) - Comprehensive Factor Analysis Report',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.show()

        print("✓ Visual analysis completed")

    def _plot_factor_ic_heatmap(self, ax):
        """绘制因子IC热力图"""
        try:
            # 准备热力图数据
            factors = [f['factor_code'] for f in self.factor_results['all'][:10]]
            targets = ['future_return_1d', 'future_return_3d', 'open_performance_1d']

            ic_matrix = []
            for factor in factors:
                row = []
                for target in targets:
                    ic_key = f"{factor}_{target}"
                    if ic_key in self.ic_results:
                        ic_val = self.ic_results[ic_key]['rank_ic']
                        row.append(ic_val if not pd.isna(ic_val) else 0)
                    else:
                        row.append(0)
                ic_matrix.append(row)

            ic_matrix = np.array(ic_matrix)

            im = ax.imshow(ic_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)

            ax.set_xticks(range(len(targets)))
            ax.set_xticklabels([t.replace('future_', '').replace('_', ' ').title() for t in targets], rotation=45)
            ax.set_yticks(range(len(factors)))
            ax.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in factors], fontsize=8)

            ax.set_title('Factor IC Heatmap', fontweight='bold')

            # 添加数值标签
            for i in range(len(factors)):
                for j in range(len(targets)):
                    text = ax.text(j, i, f'{ic_matrix[i, j]:.3f}',
                                 ha="center", va="center", color='white' if abs(ic_matrix[i, j]) > 0.15 else 'black',
                                 fontsize=7)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
            ax.set_title('Factor IC Heatmap - Error')

    def _plot_sentiment_return_scatter(self, ax):
        """绘制情绪分数与未来收益散点图"""
        try:
            data = self.merged_data.dropna(subset=['overall_score_mean', 'future_return_1d'])
            if len(data) > 0:
                ax.scatter(data['overall_score_mean'], data['future_return_1d'] * 100, alpha=0.6, s=30)

                # 添加趋势线
                if len(data) > 2:
                    z = np.polyfit(data['overall_score_mean'], data['future_return_1d'] * 100, 1)
                    p = np.poly1d(z)
                    ax.plot(data['overall_score_mean'], p(data['overall_score_mean']), "r--", alpha=0.8)

                corr, p_val = pearsonr(data['overall_score_mean'], data['future_return_1d'])
                ax.set_title(f'Sentiment vs Future 1-Day Return\nr = {corr:.4f}, p = {p_val:.4f}', fontweight='bold')
                ax.set_xlabel('Overall Sentiment Score')
                ax.set_ylabel('Future 1-Day Return (%)')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('Sentiment vs Returns')

    def _plot_sentiment_price_timeseries(self, ax):
        """绘制情绪与价格时间序列"""
        try:
            data = self.merged_data.tail(30)  # 最近30天

            ax.plot(data['date'], data['Close'], 'b-', label='Close Price', alpha=0.7)
            ax.set_ylabel('Price', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            ax2 = ax.twinx()
            ax2.plot(data['date'], data['overall_score_mean'], 'r-', label='Sentiment', alpha=0.7)
            ax2.set_ylabel('Sentiment Score', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            ax.set_title('Price vs Sentiment Time Series', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('Price vs Sentiment')

    def _plot_factor_effectiveness_bar(self, ax):
        """绘制因子有效性条形图"""
        try:
            top_factors = self.factor_results['all'][:8]
            factor_names = [f['factor'][:12] + '...' if len(f['factor']) > 12 else f['factor'] for f in top_factors]
            ic_values = [f['max_ic'] for f in top_factors]

            bars = ax.bar(range(len(factor_names)), ic_values, alpha=0.7)
            ax.set_xticks(range(len(factor_names)))
            ax.set_xticklabels(factor_names, rotation=45, ha='right')
            ax.set_ylabel('Max |Rank IC|')
            ax.set_title('Top Factor Effectiveness', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars, ic_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('Factor Effectiveness')

    def _plot_news_volume_interaction(self, ax):
        """绘制新闻-成交量交互因子分析"""
        try:
            # 选择最佳的新闻-成交量交互因子
            interaction_factors = ['news_vol_interaction_5d', 'filtered_positive_news', 'comprehensive_news_vol']

            # 找到存在且有效的因子
            valid_factor = None
            for factor in interaction_factors:
                if factor in self.merged_data.columns:
                    factor_data = self.merged_data[factor].dropna()
                    if len(factor_data) > 10 and factor_data.std() > 1e-8:
                        valid_factor = factor
                        break

            if valid_factor:
                # 绘制因子值与未来收益的散点图
                data = self.merged_data.dropna(subset=[valid_factor, 'future_return_1d'])
                if len(data) > 10:
                    x = data[valid_factor]
                    y = data['future_return_1d'] * 100  # 转为百分比

                    # 散点图
                    ax.scatter(x, y, alpha=0.6, s=30, color='green')

                    # 添加趋势线
                    if len(data) > 2:
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)

                    # 计算IC
                    ic_result = self.calculate_factor_ic(valid_factor, 'future_return_1d')
                    ic_value = ic_result.get('rank_ic', 0)

                    ax.set_xlabel(f'{valid_factor.replace("_", " ").title()}')
                    ax.set_ylabel('Future 1-day Return (%)')
                    ax.set_title(f'News-Volume Interaction\n(IC: {ic_value:.3f})', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No valid interaction factor found', ha='center', va='center', transform=ax.transAxes)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)

        # 确保标题总是被设置
        if not ax.get_title():
            ax.set_title('News-Volume Interaction Analysis')

    def _plot_sentiment_group_returns(self, ax):
        """绘制情绪分组收益率箱线图"""
        try:
            data = self.merged_data.dropna(subset=['overall_score_mean', 'future_return_1d'])
            if len(data) > 10:
                # 创建情绪分组
                data['sentiment_group'] = pd.cut(data['overall_score_mean'],
                                               bins=[-np.inf, -1, 0, 1, np.inf],
                                               labels=['Negative', 'Neutral-', 'Neutral+', 'Positive'])

                groups = []
                labels = []
                for group in ['Negative', 'Neutral-', 'Neutral+', 'Positive']:
                    group_data = data[data['sentiment_group'] == group]['future_return_1d'] * 100
                    if len(group_data) > 0:
                        groups.append(group_data)
                        labels.append(f'{group}\n(n={len(group_data)})')

                if groups:
                    ax.boxplot(groups, labels=labels)
                    ax.set_ylabel('Future 1-Day Return (%)')
                    ax.set_title('Returns by Sentiment Group', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                else:
                    ax.text(0.5, 0.5, 'Insufficient group data', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('Sentiment Group Returns')

    def _plot_combination_factor_comparison(self, ax):
        """绘制组合因子效果对比"""
        try:
            if 'combined_factor_equal' in self.merged_data.columns and 'combined_factor_ic_weighted' in self.merged_data.columns:
                # 计算组合因子的IC
                equal_ic = self.calculate_factor_ic('combined_factor_equal', 'future_return_1d')
                weighted_ic = self.calculate_factor_ic('combined_factor_ic_weighted', 'future_return_1d')

                # 最佳单因子IC
                best_single = max([f['max_ic'] for f in self.factor_results['all'][:5]])

                categories = ['Best Single\nFactor', 'Equal Weight\nCombination', 'IC Weighted\nCombination']
                ic_values = [best_single, abs(equal_ic['rank_ic']), abs(weighted_ic['rank_ic'])]

                bars = ax.bar(categories, ic_values, alpha=0.7,
                             color=['blue', 'green', 'orange'])
                ax.set_ylabel('Absolute Rank IC')
                ax.set_title('Factor Combination Comparison', fontweight='bold')
                ax.grid(True, alpha=0.3)

                # 添加数值标签
                for bar, value in zip(bars, ic_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'Combination factors not calculated', ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('Factor Combinations')

    def _plot_factor_stability(self, ax):
        """绘制因子稳定性分析"""
        try:
            # 选择一个主要因子进行滚动IC分析
            main_factor = self.factor_results['all'][0]['factor_code']
            target = 'future_return_1d'

            if main_factor in self.merged_data.columns and target in self.merged_data.columns:
                # 滚动窗口计算IC
                window_size = 20
                data = self.merged_data.dropna(subset=[main_factor, target])

                if len(data) > window_size:
                    rolling_ics = []
                    dates = []

                    for i in range(window_size, len(data)):
                        window_data = data.iloc[i-window_size:i]
                        ic_val, _ = spearmanr(window_data[main_factor], window_data[target])
                        rolling_ics.append(ic_val if not pd.isna(ic_val) else 0)
                        dates.append(window_data['date'].iloc[-1])

                    ax.plot(dates, rolling_ics, 'b-', alpha=0.7)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    ax.set_ylabel('Rolling Rank IC')
                    ax.set_title(f'Factor Stability: {main_factor[:20]}', fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for rolling analysis', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'Factor not available', ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('Factor Stability')

    def _plot_news_intensity_returns(self, ax):
        """绘制新闻强度与收益率关系"""
        try:
            data = self.merged_data.dropna(subset=['news_intensity', 'future_return_1d'])
            if len(data) > 0:
                ax.scatter(data['news_intensity'], data['future_return_1d'] * 100, alpha=0.6, s=30)

                corr, p_val = pearsonr(data['news_intensity'], data['future_return_1d'])
                ax.set_title(f'News Intensity vs Returns\nr = {corr:.4f}', fontweight='bold')
                ax.set_xlabel('News Intensity (Count)')
                ax.set_ylabel('Future 1-Day Return (%)')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('News Intensity vs Returns')

    def _plot_volatility_factor_analysis(self, ax):
        """绘制波动率因子分析"""
        try:
            data = self.merged_data.dropna(subset=['volatility_5d', 'future_volatility_1d'])
            if len(data) > 0:
                ax.scatter(data['volatility_5d'], data['future_volatility_1d'], alpha=0.6, s=30)

                # 添加对角线
                min_val = min(data['volatility_5d'].min(), data['future_volatility_1d'].min())
                max_val = max(data['volatility_5d'].max(), data['future_volatility_1d'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

                corr, _ = pearsonr(data['volatility_5d'], data['future_volatility_1d'])
                ax.set_title(f'Volatility Persistence\nr = {corr:.4f}', fontweight='bold')
                ax.set_xlabel('Current 5-Day Volatility')
                ax.set_ylabel('Future 1-Day Volatility')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('Volatility Analysis')

    def _plot_volume_price_sync(self, ax):
        """绘制量价配合度分析"""
        try:
            data = self.merged_data.dropna(subset=['volume_price_sync', 'future_return_1d'])
            if len(data) > 0:
                # 按量价配合度分组
                sync_groups = data.groupby('volume_price_sync')['future_return_1d'].agg(['mean', 'count'])

                categories = []
                returns = []
                counts = []

                for sync_val in [-1, 0, 1]:
                    if sync_val in sync_groups.index:
                        categories.append(['Price-Volume\nConflict', 'No Change', 'Price-Volume\nSync'][sync_val + 1])
                        returns.append(sync_groups.loc[sync_val, 'mean'] * 100)
                        counts.append(sync_groups.loc[sync_val, 'count'])

                if categories:
                    bars = ax.bar(categories, returns, alpha=0.7)
                    ax.set_ylabel('Average Future Return (%)')
                    ax.set_title('Volume-Price Synchronization Effect', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

                    # 添加样本数标签
                    for bar, count in zip(bars, counts):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'n={count}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No sync data', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('Volume-Price Sync')

    def _plot_prediction_timeseries(self, ax):
        """绘制预测效果时间序列"""
        try:
            if 'combined_factor_ic_weighted' in self.merged_data.columns:
                data = self.merged_data.dropna(subset=['combined_factor_ic_weighted', 'future_return_1d']).tail(50)

                if len(data) > 0:
                    # 标准化因子值作为预测信号
                    factor_signal = data['combined_factor_ic_weighted']
                    actual_return = data['future_return_1d'] * 100

                    ax.plot(data['date'], factor_signal, 'b-', label='Prediction Signal', alpha=0.7)
                    ax.set_ylabel('Prediction Signal', color='blue')
                    ax.tick_params(axis='y', labelcolor='blue')

                    ax2 = ax.twinx()
                    ax2.plot(data['date'], actual_return, 'r-', label='Actual Return (%)', alpha=0.7)
                    ax2.set_ylabel('Actual Return (%)', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')

                    ax.set_title('Prediction vs Actual Performance', fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'Combined factor not available', ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')

        ax.set_title('Prediction Performance')

    def generate_summary_report(self):
        """生成分析总结报告"""
        print("\n" + "=" * 80)
        print(f"🎉 {self.stock_name} ({self.stock_code}) Comprehensive Factor Analysis Report")
        print("=" * 80)

        # 数据概览
        print("\n📊 Data Overview:")
        print(f"   Analysis Period: {self.merged_data['date'].min()} to {self.merged_data['date'].max()}")
        print(f"   Effective Trading Days: {len(self.merged_data)} days")
        print(f"   Total Sentiment Records: {self.merged_data['overall_score_count'].sum():.0f} records")
        print(f"   Average Daily Sentiment Strength: {self.merged_data['overall_score_mean'].mean():.3f}")

        # 最佳因子
        print("\n🏆 Top 5 Most Effective Factors:")
        for i, factor in enumerate(self.factor_results['all'][:5], 1):
            # 更新因子类型判断逻辑
            sentiment_factors = ['overall_score_mean', 'sentiment_strength', 'weighted_sentiment', 'news_intensity', 'sentiment_change_1d', 'sentiment_change_3d']
            news_vol_factors = ['news_vol_interaction_5d', 'news_vol_interaction_20d', 'filtered_positive_news', 'filtered_negative_news', 'certainty_vol_factor', 'news_money_flow', 'comprehensive_news_vol']

            if factor['factor_code'] in sentiment_factors:
                factor_type = "Sentiment"
            elif factor['factor_code'] in news_vol_factors:
                factor_type = "News-Volume"
            else:
                factor_type = "Technical"

            print(f"   {i}. {factor['factor']:30s} ({factor_type}) - Max IC: {factor['max_ic']:.4f}")

        # 因子类型效果对比
        print("\n📈 Factor Type Performance Comparison:")
        sentiment_avg_ic = np.mean([f['max_ic'] for f in self.factor_results['sentiment']])
        traditional_avg_ic = np.mean([f['max_ic'] for f in self.factor_results['traditional']])
        news_volume_avg_ic = np.mean([f['max_ic'] for f in self.factor_results['news_volume']])

        print(f"   Sentiment Factors Average IC: {sentiment_avg_ic:.4f}")
        print(f"   Traditional Technical Factors Average IC: {traditional_avg_ic:.4f}")
        print(f"   News-Volume Interaction Factors Average IC: {news_volume_avg_ic:.4f}")

        best_type = max([("Sentiment", sentiment_avg_ic), ("Technical", traditional_avg_ic), ("News-Volume", news_volume_avg_ic)], key=lambda x: x[1])
        print(f"   👑 Best Factor Type: {best_type[0]} Factors")

        # 预测目标分析
        print("\n🎯 Prediction Target Effectiveness Analysis:")
        targets = ['future_return_1d', 'future_return_3d', 'open_performance_1d']
        target_names = ['1-day Return', '3-day Return', '1-day Open Performance']

        for target, name in zip(targets, target_names):
            target_ics = []
            for factor in self.factor_results['all'][:10]:
                ic_key = f"{factor['factor_code']}_{target}"
                if ic_key in self.ic_results and not pd.isna(self.ic_results[ic_key]['rank_ic']):
                    target_ics.append(abs(self.ic_results[ic_key]['rank_ic']))

            if target_ics:
                avg_ic = np.mean(target_ics)
                print(f"   {name:20s}: Average Prediction IC = {avg_ic:.4f}")

        # 组合因子效果
        if 'combined_factor_ic_weighted' in self.merged_data.columns:
            print("\n🔗 Factor Combination Performance:")
            combined_ic = self.calculate_factor_ic('combined_factor_ic_weighted', 'future_return_1d')
            best_single_ic = self.factor_results['all'][0]['max_ic']
            improvement = abs(combined_ic['rank_ic']) - best_single_ic

            print(f"   Best Single Factor IC: {best_single_ic:.4f}")
            print(f"   IC-Weighted Combination IC: {abs(combined_ic['rank_ic']):.4f}")
            print(f"   Combination Improvement: {improvement:+.4f} ({improvement/best_single_ic*100:+.1f}%)")

        # 关键发现
        print("\n💡 Key Findings:")

        # 情绪因子发现
        sentiment_factor_names = [f['factor_code'] for f in self.factor_results['sentiment']]
        if sentiment_factor_names:
            best_sentiment = max(self.factor_results['sentiment'], key=lambda x: x['max_ic'])
            print(f"   • Strongest Sentiment Factor: {best_sentiment['factor']} (IC={best_sentiment['max_ic']:.4f})")

        # 技术因子发现
        if self.factor_results['traditional']:
            best_technical = max(self.factor_results['traditional'], key=lambda x: x['max_ic'])
            print(f"   • Strongest Technical Factor: {best_technical['factor']} (IC={best_technical['max_ic']:.4f})")

        # 新闻-成交量交互因子发现
        if self.factor_results['news_volume']:
            best_news_vol = max(self.factor_results['news_volume'], key=lambda x: x['max_ic'])
            print(f"   • Strongest News-Volume Factor: {best_news_vol['factor']} (IC={best_news_vol['max_ic']:.4f})")

        # 市场特征
        avg_return = self.merged_data['future_return_1d'].mean()
        volatility = self.merged_data['future_return_1d'].std()
        avg_open_perf = self.merged_data['open_performance_1d'].mean()

        print(f"   • Average Daily Return: {avg_return*100:.3f}%")
        print(f"   • Return Volatility: {volatility*100:.3f}%")
        print(f"   • Average Open Performance: {avg_open_perf*100:.3f}%")

        # 交易建议
        print("\n💼 Trading Recommendations Based on Analysis:")

        if sentiment_avg_ic > 0.05:
            print("   • Sentiment factors show strong predictive ability, focus on news sentiment changes")

        if news_volume_avg_ic > 0.08:
            print("   • News-volume interaction factors perform excellently, monitor volume-filtered news signals")

        if best_type[0] == "Technical":
            print("   • Traditional technical indicators work best for this stock, prioritize technical analysis")
        elif best_type[0] == "Sentiment":
            print("   • Sentiment factors have strongest predictive power, combine with news analysis")
        elif best_type[0] == "News-Volume":
            print("   • News-volume interaction factors are most effective, use volume to filter news noise")

        if 'combined_factor_ic_weighted' in self.merged_data.columns and improvement > 0:
            print("   • Factor combination outperforms single factors, adopt multi-factor strategy")

        print(f"\n✅ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def run_analysis(self):
        """运行完整分析流程"""
        print(f"🚀 Starting {self.stock_name} ({self.stock_code}) Single Stock Factor Analysis")
        print("=" * 80)

        try:
            # 步骤1: 加载情绪数据
            self.load_sentiment_data()

            # 步骤2: 获取市场数据
            self.load_market_data()

            # 步骤3: 计算传统因子
            self.calculate_traditional_factors()

            # 步骤4: 合并因子数据
            merged_data = self.merge_factors()

            if merged_data is None or len(merged_data) < 10:
                print("❌ 数据不足，无法进行分析")
                return

            # 步骤5: 因子有效性分析
            self.analyze_factor_effectiveness()

            # 步骤6: 因子组合分析
            self.analyze_factor_combinations()

            # 步骤7: 可视化分析
            self.create_visualizations()

            # 步骤8: 生成总结报告
            self.generate_summary_report()

        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

# 主函数
def main():
    """主函数"""
    # 配置文件路径和股票代码
    sentiment_file = r'D:\projects\q\myQ\scripts\news_scores_result_1y_zijin.csv'
    stock_code = '601899'  # 紫金矿业

    # 检查文件是否存在
    if not os.path.exists(sentiment_file):
        print(f"❌ 情绪数据文件不存在: {sentiment_file}")
        return

    # 创建分析器并运行分析
    analyzer = SingleStockFactorAnalyzer(sentiment_file, stock_code)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()