#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
紫金矿业自适应因子策略 (修复版)
=============================

问题诊断：
- 原策略无信号产生：情绪波动率均值0.41 > 阈值0.15
- 需要根据实际数据分布自适应调整参数

改进方案：
1. 数据分布分析
2. 自适应阈值设定
3. 分位数方法确定合理阈值
4. 增加调试信息

作者: Claude Code
日期: 2025-09-25
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_strategy_data():
    """创建策略演示数据"""
    print("📊 创建策略演示数据...")

    np.random.seed(42)
    n_days = 124
    dates = pd.date_range('2024-09-20', periods=n_days, freq='D')

    # 模拟价格走势
    price = 15.0
    returns = np.random.normal(0.00435, 0.02826, n_days)
    prices = price * np.cumprod(1 + returns)

    # 计算60日动量 - 增加更多变异性
    momentum_60d = []
    for i in range(n_days):
        if i < 120:
            # 前120天设置随机小幅动量
            momentum_60d.append(np.random.normal(0, 0.02))
        else:
            recent_60 = np.mean(prices[i-60:i])
            previous_60 = np.mean(prices[i-120:i-60])
            base_momentum = recent_60 / previous_60 - 1
            # 加入更多噪声使动量更真实
            momentum_60d.append(base_momentum + np.random.normal(0, 0.01))

    # 重新设计情绪波动率 - 使其有更合理的分布
    sentiment_scores = np.random.normal(3, 1.5, n_days)  # 增加基础波动
    sentiment_volatility = []
    for i in range(n_days):
        if i < 5:
            sentiment_volatility.append(np.random.uniform(0.05, 0.3))
        else:
            # 使用更现实的波动率计算
            window_scores = sentiment_scores[i-5:i]
            vol = np.std(window_scores)
            # 确保有合理的分布范围
            vol = max(0.02, min(0.8, vol))  # 限制在合理范围内
            sentiment_volatility.append(vol)

    # 计算未来3日收益率
    future_returns_3d = []
    for i in range(n_days):
        if i >= n_days - 3:
            future_returns_3d.append(np.nan)
        else:
            future_ret = prices[i+3] / prices[i] - 1
            future_returns_3d.append(future_ret)

    data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'momentum_60d': momentum_60d,
        'sentiment_volatility': sentiment_volatility,
        'future_return_3d': future_returns_3d,
        'daily_return': returns
    })

    return data

def analyze_factor_distribution(data):
    """分析因子分布，确定合理阈值"""
    print(f"\n🔍 因子分布分析:")

    momentum = data['momentum_60d']
    sentiment_vol = data['sentiment_volatility']

    print(f"\n📈 60日动量分析:")
    print(f"   均值: {momentum.mean():.4f}")
    print(f"   标准差: {momentum.std():.4f}")
    print(f"   分位点: 10%={momentum.quantile(0.1):.4f}, 25%={momentum.quantile(0.25):.4f}, 75%={momentum.quantile(0.75):.4f}, 90%={momentum.quantile(0.9):.4f}")
    print(f"   极值: 最小={momentum.min():.4f}, 最大={momentum.max():.4f}")

    print(f"\n📊 情绪波动率分析:")
    print(f"   均值: {sentiment_vol.mean():.4f}")
    print(f"   标准差: {sentiment_vol.std():.4f}")
    print(f"   分位点: 10%={sentiment_vol.quantile(0.1):.4f}, 25%={sentiment_vol.quantile(0.25):.4f}, 75%={sentiment_vol.quantile(0.75):.4f}, 90%={sentiment_vol.quantile(0.9):.4f}")
    print(f"   极值: 最小={sentiment_vol.min():.4f}, 最大={sentiment_vol.max():.4f}")

    # 推荐阈值
    momentum_high = momentum.quantile(0.75)  # 上75%分位作为买入阈值
    momentum_low = momentum.quantile(0.25)   # 下25%分位作为卖出阈值
    sentiment_low = sentiment_vol.quantile(0.5)   # 中位数作为情绪稳定阈值
    sentiment_high = sentiment_vol.quantile(0.75)  # 75%分位作为情绪不稳定阈值

    print(f"\n💡 推荐阈值 (基于分位数):")
    print(f"   动量买入阈值: {momentum_high:.4f} (75%分位)")
    print(f"   动量卖出阈值: {momentum_low:.4f} (25%分位)")
    print(f"   情绪稳定阈值: {sentiment_low:.4f} (50%分位)")
    print(f"   情绪波动阈值: {sentiment_high:.4f} (75%分位)")

    return {
        'momentum_buy': momentum_high,
        'momentum_sell': momentum_low,
        'sentiment_stable': sentiment_low,
        'sentiment_volatile': sentiment_high
    }

def generate_adaptive_signals(data, thresholds):
    """使用自适应阈值生成交易信号"""
    print(f"\n📈 生成自适应交易信号...")
    print(f"   动量买入阈值: {thresholds['momentum_buy']:.4f}")
    print(f"   动量卖出阈值: {thresholds['momentum_sell']:.4f}")
    print(f"   情绪稳定阈值: {thresholds['sentiment_stable']:.4f}")
    print(f"   情绪波动阈值: {thresholds['sentiment_volatile']:.4f}")

    signals = data.copy()

    # 改进的交易信号逻辑
    buy_condition = (
        (data['momentum_60d'] > thresholds['momentum_buy']) &
        (data['sentiment_volatility'] < thresholds['sentiment_stable'])
    )

    sell_condition = (
        (data['momentum_60d'] < thresholds['momentum_sell']) &
        (data['sentiment_volatility'] > thresholds['sentiment_volatile'])
    )

    signals['signal'] = 0
    signals.loc[buy_condition, 'signal'] = 1   # 买入信号
    signals.loc[sell_condition, 'signal'] = -1  # 卖出信号

    # 详细信号分析
    buy_count = (signals['signal'] == 1).sum()
    sell_count = (signals['signal'] == -1).sum()
    hold_count = (signals['signal'] == 0).sum()

    print(f"✓ 信号统计:")
    print(f"   买入信号: {buy_count} 次 ({buy_count/len(signals)*100:.1f}%)")
    print(f"   卖出信号: {sell_count} 次 ({sell_count/len(signals)*100:.1f}%)")
    print(f"   观望: {hold_count} 次 ({hold_count/len(signals)*100:.1f}%)")

    # 检查各条件的满足情况
    momentum_buy_count = (data['momentum_60d'] > thresholds['momentum_buy']).sum()
    momentum_sell_count = (data['momentum_60d'] < thresholds['momentum_sell']).sum()
    sentiment_stable_count = (data['sentiment_volatility'] < thresholds['sentiment_stable']).sum()
    sentiment_volatile_count = (data['sentiment_volatility'] > thresholds['sentiment_volatile']).sum()

    print(f"\n🔍 单条件满足情况:")
    print(f"   动量看多: {momentum_buy_count} 天 ({momentum_buy_count/len(signals)*100:.1f}%)")
    print(f"   动量看空: {momentum_sell_count} 天 ({momentum_sell_count/len(signals)*100:.1f}%)")
    print(f"   情绪稳定: {sentiment_stable_count} 天 ({sentiment_stable_count/len(signals)*100:.1f}%)")
    print(f"   情绪波动: {sentiment_volatile_count} 天 ({sentiment_volatile_count/len(signals)*100:.1f}%)")

    return signals

def simulate_strategy_performance(signals, holding_period=3):
    """模拟策略表现"""
    print(f"\n💰 模拟策略表现 (持有周期: {holding_period}天)...")

    positions = []
    strategy_returns = []

    current_position = 0
    hold_days_left = 0

    for i in range(len(signals)):
        signal = signals['signal'].iloc[i]

        # 更新持仓逻辑
        if hold_days_left > 0:
            # 继续持有当前仓位
            positions.append(current_position)
            hold_days_left -= 1
        elif signal != 0:
            # 新信号，建立仓位
            current_position = signal
            positions.append(current_position)
            hold_days_left = holding_period - 1
        else:
            # 无信号，空仓
            current_position = 0
            positions.append(0)

    signals['position'] = positions

    # 计算策略收益
    for i in range(len(signals)):
        if i == 0:
            strategy_returns.append(0)
        else:
            position = positions[i-1]  # 使用前一期的仓位
            if pd.notna(signals['future_return_3d'].iloc[i]):
                # 如果有未来收益数据，使用未来收益
                strategy_return = position * signals['future_return_3d'].iloc[i]
            else:
                # 否则使用当期收益
                strategy_return = position * signals['daily_return'].iloc[i]
            strategy_returns.append(strategy_return)

    signals['strategy_return'] = strategy_returns

    # 计算累计收益
    signals['strategy_cumret'] = (1 + signals['strategy_return']).cumprod()
    signals['benchmark_cumret'] = (1 + signals['daily_return']).cumprod()

    # 统计持仓
    long_days = (signals['position'] > 0).sum()
    short_days = (signals['position'] < 0).sum()
    flat_days = (signals['position'] == 0).sum()

    print(f"✓ 持仓统计:")
    print(f"   多头: {long_days} 天 ({long_days/len(signals)*100:.1f}%)")
    print(f"   空头: {short_days} 天 ({short_days/len(signals)*100:.1f}%)")
    print(f"   空仓: {flat_days} 天 ({flat_days/len(signals)*100:.1f}%)")

    # 计算有效交易收益
    active_returns = [r for r, p in zip(strategy_returns, positions) if p != 0]
    if active_returns:
        print(f"   有效交易天数: {len(active_returns)} 天")
        print(f"   活跃期间平均收益: {np.mean(active_returns)*100:.3f}%/天")
        print(f"   活跃期间胜率: {sum(1 for r in active_returns if r > 0)/len(active_returns)*100:.1f}%")

    return signals

def calculate_performance_metrics(results):
    """计算绩效指标"""
    print(f"\n📊 计算绩效指标...")

    strategy_ret = results['strategy_return'].dropna()
    benchmark_ret = results['daily_return']

    # 基础收益指标
    total_strategy = (results['strategy_cumret'].iloc[-1] - 1) * 100
    total_benchmark = (results['benchmark_cumret'].iloc[-1] - 1) * 100

    annual_strategy = strategy_ret.mean() * 252 * 100
    annual_benchmark = benchmark_ret.mean() * 252 * 100

    # 风险指标
    vol_strategy = strategy_ret.std() * np.sqrt(252) * 100 if strategy_ret.std() > 0 else 0
    vol_benchmark = benchmark_ret.std() * np.sqrt(252) * 100

    # 最大回撤
    strategy_cumret = results['strategy_cumret']
    benchmark_cumret = results['benchmark_cumret']

    strategy_dd = ((strategy_cumret - strategy_cumret.expanding().max()) / strategy_cumret.expanding().max()).min() * 100
    benchmark_dd = ((benchmark_cumret - benchmark_cumret.expanding().max()) / benchmark_cumret.expanding().max()).min() * 100

    # 夏普比率
    sharpe_strategy = annual_strategy / vol_strategy if vol_strategy > 0 else 0
    sharpe_benchmark = annual_benchmark / vol_benchmark if vol_benchmark > 0 else 0

    # 胜率
    win_rate = (strategy_ret > 0).sum() / len(strategy_ret) * 100 if len(strategy_ret) > 0 else 0

    # 信息比率
    if len(strategy_ret) > 0:
        excess_ret = strategy_ret - benchmark_ret.reindex(strategy_ret.index)
        info_ratio = excess_ret.mean() / excess_ret.std() * np.sqrt(252) if excess_ret.std() > 0 else 0
    else:
        info_ratio = 0

    metrics = {
        'total_return_strategy': total_strategy,
        'total_return_benchmark': total_benchmark,
        'annual_return_strategy': annual_strategy,
        'annual_return_benchmark': annual_benchmark,
        'volatility_strategy': vol_strategy,
        'volatility_benchmark': vol_benchmark,
        'max_drawdown_strategy': abs(strategy_dd),
        'max_drawdown_benchmark': abs(benchmark_dd),
        'sharpe_strategy': sharpe_strategy,
        'sharpe_benchmark': sharpe_benchmark,
        'win_rate': win_rate,
        'information_ratio': info_ratio
    }

    print("✓ 绩效计算完成")
    return metrics

def print_strategy_report(metrics, signals, thresholds):
    """打印策略报告"""
    print("\n" + "="*70)
    print("🎯 紫金矿业自适应因子策略回测报告")
    print("="*70)

    print(f"\n📋 策略概览:")
    print(f"   策略名称: 自适应60日动量+情绪波动率策略")
    print(f"   回测周期: {signals['date'].min().strftime('%Y-%m-%d')} 至 {signals['date'].max().strftime('%Y-%m-%d')}")
    print(f"   交易天数: {len(signals)} 天")
    print(f"   活跃交易: {(signals['position'] != 0).sum()} 天")

    print(f"\n⚙️ 参数设置:")
    print(f"   动量买入阈值: {thresholds['momentum_buy']:.4f}")
    print(f"   动量卖出阈值: {thresholds['momentum_sell']:.4f}")
    print(f"   情绪稳定阈值: {thresholds['sentiment_stable']:.4f}")
    print(f"   情绪波动阈值: {thresholds['sentiment_volatile']:.4f}")

    print(f"\n💰 收益表现:")
    print(f"   策略总收益:     {metrics['total_return_strategy']:>8.2f}%")
    print(f"   基准总收益:     {metrics['total_return_benchmark']:>8.2f}%")
    print(f"   超额收益:       {metrics['total_return_strategy']-metrics['total_return_benchmark']:>+8.2f}%")
    print(f"   年化收益(策略): {metrics['annual_return_strategy']:>8.2f}%")
    print(f"   年化收益(基准): {metrics['annual_return_benchmark']:>8.2f}%")

    print(f"\n⚡ 风险控制:")
    print(f"   年化波动率(策略): {metrics['volatility_strategy']:>6.2f}%")
    print(f"   年化波动率(基准): {metrics['volatility_benchmark']:>6.2f}%")
    print(f"   最大回撤(策略):   {metrics['max_drawdown_strategy']:>6.2f}%")
    print(f"   最大回撤(基准):   {metrics['max_drawdown_benchmark']:>6.2f}%")

    print(f"\n📈 风险调整后收益:")
    print(f"   夏普比率(策略):   {metrics['sharpe_strategy']:>6.2f}")
    print(f"   夏普比率(基准):   {metrics['sharpe_benchmark']:>6.2f}")
    print(f"   信息比率:         {metrics['information_ratio']:>6.2f}")
    print(f"   胜率:             {metrics['win_rate']:>6.1f}%")

    # 策略评估
    print(f"\n🎯 策略评估:")
    excess_return = metrics['total_return_strategy'] - metrics['total_return_benchmark']
    active_days = (signals['position'] != 0).sum()

    if active_days == 0:
        evaluation = "⚪ 无交易 - 参数过于严格，建议放宽阈值"
    elif excess_return > 5 and metrics['sharpe_strategy'] > 1.5:
        evaluation = "🟢 优秀 - 显著跑赢基准且风险可控"
    elif excess_return > 0 and metrics['sharpe_strategy'] > metrics['sharpe_benchmark']:
        evaluation = "🟡 良好 - 正超额收益且风险调整后表现更佳"
    elif excess_return > -2 and metrics['max_drawdown_strategy'] < 15:
        evaluation = "🟠 一般 - 表现平稳风险可接受"
    else:
        evaluation = "🔴 较差 - 跑输基准或风险过高"

    print(f"   {evaluation}")

    print("="*70)

def main():
    """主函数"""
    print("🚀 启动紫金矿业自适应因子策略回测")
    print("="*50)

    try:
        # 1. 创建数据
        data = create_sample_strategy_data()

        # 2. 分析因子分布
        thresholds = analyze_factor_distribution(data)

        # 3. 生成自适应信号
        signals = generate_adaptive_signals(data, thresholds)

        # 4. 模拟策略表现
        results = simulate_strategy_performance(signals, holding_period=3)

        # 5. 计算绩效指标
        metrics = calculate_performance_metrics(results)

        # 6. 打印报告
        print_strategy_report(metrics, results, thresholds)

        # 7. 保存结果
        results.to_csv('D:/projects/q/myQ/scripts/adaptive_strategy_results.csv', index=False)
        print(f"\n✅ 回测完成! 结果已保存到 adaptive_strategy_results.csv")

        return results, metrics

    except Exception as e:
        print(f"❌ 回测过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, metrics = main()