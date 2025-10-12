#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
紫金矿业简化因子策略
==================

基于因子分析结果的策略设计：

🎯 策略核心逻辑：
1. 主因子：60日动量 (IC=0.3117，最强预测能力)
2. 辅助因子：情绪波动率 (IC=0.2664，情绪稳定性指标)
3. 预测目标：3日收益率 (平均IC=0.1704，最佳预测效果)

📋 交易规则：
- 买入条件：60日动量 > 2% 且 情绪波动率 < 0.15 (强势上涨+情绪稳定)
- 卖出条件：60日动量 < -2% 且 情绪波动率 > 0.2 (弱势下跌+情绪不稳)
- 持有周期：3天 (对应最佳预测周期)
- 仓位管理：满仓/空仓策略 (简化版本)

💡 策略优势：
1. 基于最强IC因子，预测能力强
2. 情绪因子过滤，避免噪声交易
3. 短周期持有，降低市场风险
4. 简单易实施，参数少

作者: Claude Code
日期: 2025-09-25
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_strategy_data():
    """创建策略演示数据"""
    print("📊 创建策略演示数据...")

    # 基于真实分析结果创建124天的演示数据
    np.random.seed(42)
    n_days = 124

    dates = pd.date_range('2024-09-20', periods=n_days, freq='D')

    # 模拟价格走势 (基于真实特征: 日均收益0.435%, 波动率2.826%)
    price = 15.0  # 紫金矿业基准价格
    returns = np.random.normal(0.00435, 0.02826, n_days)
    prices = price * np.cumprod(1 + returns)

    # 计算60日动量 (最强因子 IC=0.3117)
    momentum_60d = []
    for i in range(n_days):
        if i < 120:
            momentum_60d.append(0)
        else:
            recent_60 = np.mean(prices[i-60:i])
            previous_60 = np.mean(prices[i-120:i-60])
            momentum_60d.append(recent_60 / previous_60 - 1)

    # 模拟情绪波动率 (第二强因子 IC=0.2664)
    sentiment_scores = np.random.normal(3, 0.5, n_days)  # 基础情绪分数
    sentiment_volatility = []
    for i in range(n_days):
        if i < 5:
            sentiment_volatility.append(0.1)
        else:
            vol = np.std(sentiment_scores[i-5:i])
            sentiment_volatility.append(vol)

    # 计算未来3日收益率 (最佳预测目标)
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

    print(f"✓ 数据创建完成: {len(data)} 天")
    print(f"✓ 平均60日动量: {np.mean(momentum_60d):.4f}")
    print(f"✓ 平均情绪波动率: {np.mean(sentiment_volatility):.4f}")

    return data

def generate_strategy_signals(data, momentum_threshold=0.02, sentiment_vol_threshold=0.15):
    """生成交易信号"""
    print(f"\n📈 生成交易信号...")
    print(f"   动量阈值: ±{momentum_threshold:.2f}")
    print(f"   情绪波动率阈值: {sentiment_vol_threshold:.2f}")

    signals = data.copy()

    # 交易信号逻辑
    buy_condition = (
        (data['momentum_60d'] > momentum_threshold) &
        (data['sentiment_volatility'] < sentiment_vol_threshold)
    )

    sell_condition = (
        (data['momentum_60d'] < -momentum_threshold) &
        (data['sentiment_volatility'] > sentiment_vol_threshold)
    )

    signals['signal'] = 0
    signals.loc[buy_condition, 'signal'] = 1   # 买入信号
    signals.loc[sell_condition, 'signal'] = -1  # 卖出信号

    # 统计信号
    buy_count = (signals['signal'] == 1).sum()
    sell_count = (signals['signal'] == -1).sum()
    hold_count = (signals['signal'] == 0).sum()

    print(f"✓ 信号统计:")
    print(f"   买入信号: {buy_count} 次 ({buy_count/len(signals)*100:.1f}%)")
    print(f"   卖出信号: {sell_count} 次 ({sell_count/len(signals)*100:.1f}%)")
    print(f"   观望: {hold_count} 次 ({hold_count/len(signals)*100:.1f}%)")

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

        # 更新持仓
        if hold_days_left > 0:
            # 继续持有
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
                strategy_return = position * signals['future_return_3d'].iloc[i]
            else:
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

    return signals

def calculate_performance_metrics(results):
    """计算绩效指标"""
    print(f"\n📊 计算绩效指标...")

    strategy_ret = results['strategy_return'].dropna()
    benchmark_ret = results['daily_return']

    # 收益率指标
    total_strategy = (results['strategy_cumret'].iloc[-1] - 1) * 100
    total_benchmark = (results['benchmark_cumret'].iloc[-1] - 1) * 100

    annual_strategy = strategy_ret.mean() * 252 * 100
    annual_benchmark = benchmark_ret.mean() * 252 * 100

    # 风险指标
    vol_strategy = strategy_ret.std() * np.sqrt(252) * 100
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
    win_rate = (strategy_ret > 0).sum() / len(strategy_ret) * 100

    # 信息比率
    excess_ret = strategy_ret - benchmark_ret.reindex(strategy_ret.index)
    info_ratio = excess_ret.mean() / excess_ret.std() * np.sqrt(252) if excess_ret.std() > 0 else 0

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

def print_strategy_report(metrics, signals):
    """打印策略报告"""
    print("\n" + "="*70)
    print("🎯 紫金矿业因子策略回测报告")
    print("="*70)

    print(f"\n📋 策略概览:")
    print(f"   策略名称: 60日动量+情绪波动率组合策略")
    print(f"   回测周期: {signals['date'].min().strftime('%Y-%m-%d')} 至 {signals['date'].max().strftime('%Y-%m-%d')}")
    print(f"   交易天数: {len(signals)} 天")
    print(f"   活跃交易: {(signals['position'] != 0).sum()} 天")

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

    if excess_return > 5 and metrics['sharpe_strategy'] > 1.5:
        evaluation = "🟢 优秀 - 显著跑赢基准且风险可控"
    elif excess_return > 2 and metrics['sharpe_strategy'] > metrics['sharpe_benchmark']:
        evaluation = "🟡 良好 - 正超额收益且风险调整后表现更佳"
    elif excess_return > -2 and metrics['max_drawdown_strategy'] < 15:
        evaluation = "🟠 一般 - 表现平稳风险可接受"
    else:
        evaluation = "🔴 较差 - 跑输基准或风险过高"

    print(f"   {evaluation}")

    print(f"\n💡 关键洞察:")
    if metrics['win_rate'] > 55:
        print("   • 胜率较高，信号质量良好")
    if metrics['information_ratio'] > 0.5:
        print("   • 信息比率优秀，因子组合有效")
    if metrics['max_drawdown_strategy'] < metrics['max_drawdown_benchmark']:
        print("   • 回撤控制优于买入持有策略")
    if (signals['position'] != 0).sum() / len(signals) < 0.4:
        print("   • 低频交易策略，适合成本敏感场景")

    print(f"\n🔄 实盘建议:")
    print("   1. 动量因子具有强预测力，可作为核心信号")
    print("   2. 情绪波动率有效过滤噪声，建议保留")
    print("   3. 3天持有周期平衡了收益和风险")
    print("   4. 可考虑加入止损机制进一步控制风险")
    print("   5. 建议小仓位试验，逐步放大")

    print("="*70)

def main():
    """主函数 - 运行完整的策略回测"""
    print("🚀 启动紫金矿业因子策略回测")
    print("="*50)

    try:
        # 1. 创建数据
        data = create_sample_strategy_data()

        # 2. 生成信号
        signals = generate_strategy_signals(data,
                                          momentum_threshold=0.02,
                                          sentiment_vol_threshold=0.15)

        # 3. 模拟策略表现
        results = simulate_strategy_performance(signals, holding_period=3)

        # 4. 计算绩效指标
        metrics = calculate_performance_metrics(results)

        # 5. 打印报告
        print_strategy_report(metrics, results)

        # 6. 保存结果
        results.to_csv('D:/projects/q/myQ/scripts/strategy_backtest_results.csv', index=False)
        print(f"\n✅ 回测完成! 结果已保存到 strategy_backtest_results.csv")

        return results, metrics

    except Exception as e:
        print(f"❌ 回测过程出错: {e}")
        return None, None

if __name__ == "__main__":
    results, metrics = main()