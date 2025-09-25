#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
紫金矿业因子策略回测
===================

基于因子分析结果设计的简单有效策略：
1. 主策略：60日动量因子 (最强IC=0.3117)
2. 辅助策略：情绪波动率因子 (IC=0.2664)
3. 预测目标：3日收益率

策略逻辑：
- 当60日动量 > 阈值 且 情绪波动率 < 阈值 → 买入信号（强势且情绪稳定）
- 当60日动量 < 阈值 且 情绪波动率 > 阈值 → 卖出信号（弱势且情绪不稳）
- 持有周期：3天（对应最佳预测目标）

作者: Claude Code
日期: 2025-09-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 10

class FactorStrategyBacktest:
    def __init__(self, data_file_path):
        """
        初始化策略回测

        参数:
        data_file_path: 合并后的因子数据文件路径（应该来自单股票因子分析的结果）
        """
        self.data_file_path = data_file_path
        self.data = None
        self.signals = None
        self.positions = None
        self.returns = None
        self.results = {}

    def load_data(self):
        """加载数据（这里假设从之前的分析中获取合并数据）"""
        print("📊 Loading factor data for backtesting...")

        # 注意：实际使用时应该从single_stock_factor_analysis的结果中获取数据
        # 这里为演示目的创建一个示例数据加载函数
        try:
            # 尝试从分析结果中加载（需要single_stock_factor_analysis提供数据保存功能）
            self.data = pd.read_csv(self.data_file_path) if self.data_file_path.endswith('.csv') else None

            if self.data is None:
                print("⚠️ 无法加载外部数据，创建模拟数据进行演示...")
                self.data = self._create_demo_data()

            print(f"✓ 数据加载完成: {len(self.data)} 个交易日")
            print(f"✓ 包含因子: {[col for col in self.data.columns if 'momentum' in col.lower() or 'sentiment' in col.lower()]}")

        except Exception as e:
            print(f"⚠️ 数据加载失败: {e}")
            print("🔄 创建模拟数据进行策略演示...")
            self.data = self._create_demo_data()

        return self.data

    def _create_demo_data(self):
        """创建演示数据（基于真实分析结果的特征）"""
        np.random.seed(42)
        n_days = 124  # 对应真实数据的天数

        # 创建日期序列
        start_date = datetime(2024, 9, 20)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]

        # 创建基础价格序列（模拟紫金矿业的价格走势）
        base_price = 15.0  # 紫金矿业大概价位
        price_trend = np.cumsum(np.random.normal(0.004, 0.028, n_days))  # 0.435%日均收益，2.826%波动率
        prices = base_price * np.exp(price_trend)

        # 计算收益率
        returns_1d = np.diff(prices) / prices[:-1]
        returns_1d = np.concatenate([[0], returns_1d])

        returns_3d = np.zeros(n_days)
        for i in range(3, n_days):
            returns_3d[i] = (prices[i] / prices[i-3] - 1)

        # 创建60日动量（最强因子）
        momentum_60d = np.zeros(n_days)
        for i in range(60, n_days):
            momentum_60d[i] = np.mean(prices[i-60:i]) / np.mean(prices[i-120:i-60]) - 1 if i >= 120 else 0

        # 添加一些噪声让其更真实
        momentum_60d = momentum_60d + np.random.normal(0, 0.01, n_days)

        # 创建情绪波动率（第二强因子）
        sentiment_base = np.random.normal(3.0, 0.5, n_days)  # 基础情绪分数
        sentiment_volatility = np.zeros(n_days)
        for i in range(5, n_days):
            sentiment_volatility[i] = np.std(sentiment_base[i-5:i])

        # 创建其他必要字段
        open_perf_1d = np.random.normal(0.003, 0.02, n_days)  # 开盘表现
        volume = np.random.lognormal(15, 0.5, n_days)  # 成交量

        # 创建DataFrame
        data = pd.DataFrame({
            'date': dates,
            'Close': prices,
            'Volume': volume,
            'future_return_1d': np.roll(returns_1d, -1),  # 未来1日收益
            'future_return_3d': np.roll(returns_3d, -3),  # 未来3日收益
            'open_performance_1d': np.roll(open_perf_1d, -1),  # 未来开盘表现
            'momentum_60d': momentum_60d,
            'sentiment_volatility': sentiment_volatility,
            'current_return': returns_1d,
        })

        # 确保未来收益的最后几行是NaN（无法预测）
        data.loc[data.index[-3:], 'future_return_3d'] = np.nan
        data.loc[data.index[-1:], 'future_return_1d'] = np.nan
        data.loc[data.index[-1:], 'open_performance_1d'] = np.nan

        print("✓ 模拟数据创建完成（基于真实分析结果特征）")
        return data

    def generate_signals(self, momentum_threshold=0.02, sentiment_vol_threshold=0.15):
        """
        生成交易信号

        策略逻辑：
        1. 买入信号：60日动量 > 阈值 且 情绪波动率 < 阈值（强势且情绪稳定）
        2. 卖出信号：60日动量 < -阈值 且 情绪波动率 > 阈值（弱势且情绪不稳）
        3. 其他情况：持有或空仓
        """
        print("\n📈 Generating trading signals...")
        print(f"   Momentum threshold: ±{momentum_threshold:.2f}")
        print(f"   Sentiment volatility threshold: {sentiment_vol_threshold:.2f}")

        signals = pd.DataFrame(index=self.data.index)
        signals['momentum_60d'] = self.data['momentum_60d']
        signals['sentiment_volatility'] = self.data['sentiment_volatility']

        # 生成信号
        buy_condition = (
            (self.data['momentum_60d'] > momentum_threshold) &
            (self.data['sentiment_volatility'] < sentiment_vol_threshold)
        )

        sell_condition = (
            (self.data['momentum_60d'] < -momentum_threshold) &
            (self.data['sentiment_volatility'] > sentiment_vol_threshold)
        )

        signals['signal'] = 0
        signals.loc[buy_condition, 'signal'] = 1   # 买入
        signals.loc[sell_condition, 'signal'] = -1  # 卖出

        # 计算信号统计
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        hold_signals = (signals['signal'] == 0).sum()

        print(f"✓ 信号生成完成:")
        print(f"   买入信号: {buy_signals} 次 ({buy_signals/len(signals)*100:.1f}%)")
        print(f"   卖出信号: {sell_signals} 次 ({sell_signals/len(signals)*100:.1f}%)")
        print(f"   持有/空仓: {hold_signals} 次 ({hold_signals/len(signals)*100:.1f}%)")

        self.signals = signals
        return signals

    def generate_positions(self, holding_period=3):
        """
        根据信号生成持仓

        参数:
        holding_period: 持有周期（天），对应最佳预测目标future_return_3d
        """
        print(f"\n📊 Generating positions (holding period: {holding_period} days)...")

        positions = pd.DataFrame(index=self.data.index)
        positions['position'] = 0.0

        current_position = 0
        hold_days = 0

        for i in range(len(self.signals)):
            signal = self.signals['signal'].iloc[i]

            # 如果有新信号且当前无持仓
            if signal != 0 and current_position == 0:
                current_position = signal
                hold_days = holding_period

            # 更新持仓
            if hold_days > 0:
                positions['position'].iloc[i] = current_position
                hold_days -= 1
            else:
                current_position = 0
                positions['position'].iloc[i] = 0

        # 统计持仓
        long_days = (positions['position'] > 0).sum()
        short_days = (positions['position'] < 0).sum()
        flat_days = (positions['position'] == 0).sum()

        print(f"✓ 持仓生成完成:")
        print(f"   多头天数: {long_days} 天 ({long_days/len(positions)*100:.1f}%)")
        print(f"   空头天数: {short_days} 天 ({short_days/len(positions)*100:.1f}%)")
        print(f"   空仓天数: {flat_days} 天 ({flat_days/len(positions)*100:.1f}%)")

        self.positions = positions
        return positions

    def calculate_returns(self):
        """计算策略收益"""
        print("\n💰 Calculating strategy returns...")

        # 计算策略收益（使用3日未来收益，对应最佳预测目标）
        strategy_returns = self.positions['position'].shift(1) * self.data['future_return_3d']

        # 计算基准收益（买入持有）
        benchmark_returns = self.data['current_return']

        returns_df = pd.DataFrame({
            'date': self.data['date'],
            'strategy_return': strategy_returns,
            'benchmark_return': benchmark_returns,
            'position': self.positions['position']
        }).dropna()

        # 计算累计收益
        returns_df['strategy_cumret'] = (1 + returns_df['strategy_return']).cumprod()
        returns_df['benchmark_cumret'] = (1 + returns_df['benchmark_return']).cumprod()

        print(f"✓ 收益计算完成 ({len(returns_df)} 个有效交易日)")

        self.returns = returns_df
        return returns_df

    def calculate_performance_metrics(self):
        """计算绩效指标"""
        print("\n📊 Calculating performance metrics...")

        strategy_ret = self.returns['strategy_return'].dropna()
        benchmark_ret = self.returns['benchmark_return']

        # 基础统计
        metrics = {
            'total_return_strategy': (self.returns['strategy_cumret'].iloc[-1] - 1) * 100,
            'total_return_benchmark': (self.returns['benchmark_cumret'].iloc[-1] - 1) * 100,
            'annual_return_strategy': strategy_ret.mean() * 252 * 100,
            'annual_return_benchmark': benchmark_ret.mean() * 252 * 100,
            'volatility_strategy': strategy_ret.std() * np.sqrt(252) * 100,
            'volatility_benchmark': benchmark_ret.std() * np.sqrt(252) * 100,
            'sharpe_strategy': strategy_ret.mean() / strategy_ret.std() * np.sqrt(252) if strategy_ret.std() > 0 else 0,
            'sharpe_benchmark': benchmark_ret.mean() / benchmark_ret.std() * np.sqrt(252),
            'max_drawdown_strategy': self._calculate_max_drawdown(self.returns['strategy_cumret']),
            'max_drawdown_benchmark': self._calculate_max_drawdown(self.returns['benchmark_cumret']),
            'win_rate': (strategy_ret > 0).sum() / len(strategy_ret) * 100,
            'active_days': (self.positions['position'] != 0).sum(),
            'total_days': len(self.positions)
        }

        # 计算信息比率
        excess_returns = strategy_ret - benchmark_ret.reindex(strategy_ret.index)
        metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        print("✓ 绩效指标计算完成")

        self.results = metrics
        return metrics

    def _calculate_max_drawdown(self, cumrets):
        """计算最大回撤"""
        running_max = cumrets.expanding().max()
        drawdown = (cumrets - running_max) / running_max
        return abs(drawdown.min()) * 100

    def plot_results(self):
        """绘制回测结果"""
        print("\n📈 Generating backtest visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 累计收益对比
        ax1 = axes[0, 0]
        ax1.plot(self.returns['date'], (self.returns['strategy_cumret'] - 1) * 100,
                label='Strategy', linewidth=2, color='blue')
        ax1.plot(self.returns['date'], (self.returns['benchmark_cumret'] - 1) * 100,
                label='Buy & Hold', linewidth=2, color='gray', alpha=0.7)
        ax1.set_title('Cumulative Returns Comparison')
        ax1.set_ylabel('Returns (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 持仓分布
        ax2 = axes[0, 1]
        position_counts = self.positions['position'].value_counts()
        colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in position_counts.index]
        bars = ax2.bar(['Short', 'Flat', 'Long'], position_counts.values, color=colors, alpha=0.7)
        ax2.set_title('Position Distribution')
        ax2.set_ylabel('Days')
        for bar, count in zip(bars, position_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom')

        # 3. 因子分布与信号
        ax3 = axes[1, 0]
        scatter = ax3.scatter(self.signals['momentum_60d'], self.signals['sentiment_volatility'],
                             c=self.signals['signal'], cmap='RdYlBu', alpha=0.6, s=30)
        ax3.set_xlabel('60-day Momentum')
        ax3.set_ylabel('Sentiment Volatility')
        ax3.set_title('Factor Distribution & Signals')
        plt.colorbar(scatter, ax=ax3, label='Signal')
        ax3.grid(True, alpha=0.3)

        # 4. 月度收益热图
        ax4 = axes[1, 1]
        monthly_returns = self.returns.set_index('date')['strategy_return'].resample('M').sum() * 100
        if len(monthly_returns) > 1:
            monthly_data = monthly_returns.values.reshape(-1, 1)
            im = ax4.imshow(monthly_data.T, cmap='RdYlGn', aspect='auto')
            ax4.set_title('Monthly Strategy Returns (%)')
            ax4.set_xlabel('Month')
            ax4.set_xticks(range(len(monthly_returns)))
            ax4.set_xticklabels([d.strftime('%Y-%m') for d in monthly_returns.index], rotation=45)
            ax4.set_yticks([])
            plt.colorbar(im, ax=ax4)

            # 添加数值标签
            for i in range(len(monthly_returns)):
                ax4.text(i, 0, f'{monthly_returns.iloc[i]:.1f}%',
                        ha='center', va='center', color='black', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Monthly Returns (Insufficient Data)')

        plt.tight_layout()
        plt.show()

        print("✓ 可视化完成")

    def print_performance_report(self):
        """打印绩效报告"""
        print("\n" + "="*60)
        print("🎯 FACTOR STRATEGY BACKTEST RESULTS")
        print("="*60)

        print(f"\n📊 Strategy Overview:")
        print(f"   Strategy: 60d Momentum + Sentiment Volatility")
        print(f"   Holding Period: 3 days (optimal prediction target)")
        print(f"   Active Trading Days: {self.results['active_days']}/{self.results['total_days']} ({self.results['active_days']/self.results['total_days']*100:.1f}%)")

        print(f"\n💰 Returns Performance:")
        print(f"   Total Return (Strategy):  {self.results['total_return_strategy']:>8.2f}%")
        print(f"   Total Return (Benchmark): {self.results['total_return_benchmark']:>8.2f}%")
        print(f"   Excess Return:            {self.results['total_return_strategy']-self.results['total_return_benchmark']:>+8.2f}%")
        print(f"   ")
        print(f"   Annual Return (Strategy):  {self.results['annual_return_strategy']:>7.2f}%")
        print(f"   Annual Return (Benchmark): {self.results['annual_return_benchmark']:>7.2f}%")

        print(f"\n⚡ Risk Metrics:")
        print(f"   Volatility (Strategy):     {self.results['volatility_strategy']:>7.2f}%")
        print(f"   Volatility (Benchmark):    {self.results['volatility_benchmark']:>7.2f}%")
        print(f"   Max Drawdown (Strategy):   {self.results['max_drawdown_strategy']:>7.2f}%")
        print(f"   Max Drawdown (Benchmark):  {self.results['max_drawdown_benchmark']:>7.2f}%")

        print(f"\n📈 Risk-Adjusted Returns:")
        print(f"   Sharpe Ratio (Strategy):   {self.results['sharpe_strategy']:>7.2f}")
        print(f"   Sharpe Ratio (Benchmark):  {self.results['sharpe_benchmark']:>7.2f}")
        print(f"   Information Ratio:         {self.results['information_ratio']:>7.2f}")
        print(f"   Win Rate:                  {self.results['win_rate']:>7.1f}%")

        # 策略评价
        print(f"\n🎯 Strategy Evaluation:")
        excess_return = self.results['total_return_strategy'] - self.results['total_return_benchmark']
        sharpe_improvement = self.results['sharpe_strategy'] - self.results['sharpe_benchmark']

        if excess_return > 2 and self.results['sharpe_strategy'] > 1:
            evaluation = "🟢 EXCELLENT - Strong outperformance with good risk control"
        elif excess_return > 0 and sharpe_improvement > 0:
            evaluation = "🟡 GOOD - Positive excess return with better risk-adjusted performance"
        elif excess_return > -2 and self.results['max_drawdown_strategy'] < 15:
            evaluation = "🟠 FAIR - Modest performance with acceptable risk"
        else:
            evaluation = "🔴 POOR - Underperformance or excessive risk"

        print(f"   {evaluation}")

        # 关键建议
        print(f"\n💡 Key Insights:")
        if self.results['win_rate'] > 55:
            print("   • High win rate indicates good signal quality")
        if self.results['information_ratio'] > 0.5:
            print("   • Strong information ratio suggests effective factor combination")
        if self.results['max_drawdown_strategy'] < self.results['max_drawdown_benchmark']:
            print("   • Better drawdown control compared to buy-and-hold")
        if self.results['active_days']/self.results['total_days'] < 0.3:
            print("   • Low activity strategy - suitable for cost-sensitive implementation")

        print("="*60)

    def run_backtest(self, momentum_threshold=0.02, sentiment_vol_threshold=0.15, holding_period=3):
        """运行完整回测"""
        print("🚀 Starting Factor Strategy Backtest")
        print("="*50)

        # 加载数据
        self.load_data()

        # 生成信号
        self.generate_signals(momentum_threshold, sentiment_vol_threshold)

        # 生成持仓
        self.generate_positions(holding_period)

        # 计算收益
        self.calculate_returns()

        # 计算绩效指标
        self.calculate_performance_metrics()

        # 显示结果
        self.print_performance_report()

        # 绘制图表
        self.plot_results()

        return self.results

if __name__ == "__main__":
    # 创建并运行回测
    # 注意: 实际使用时，data_file_path应该指向single_stock_factor_analysis的输出数据
    backtest = FactorStrategyBacktest(data_file_path="demo_data.csv")  # 将使用模拟数据

    # 运行回测（使用基于真实IC值优化的参数）
    results = backtest.run_backtest(
        momentum_threshold=0.02,      # 基于60日动量IC=0.3117的阈值
        sentiment_vol_threshold=0.15, # 基于情绪波动率IC=0.2664的阈值
        holding_period=3              # 基于3日收益最佳预测效果
    )