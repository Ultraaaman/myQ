# 回测模块 (Backtest Module)

专业量化交易回测引擎，提供策略回测、性能分析和风险评估功能。

## 🚀 快速开始

### 基本回测流程

```python
from quantlib.strategy import create_ma_cross_strategy
from quantlib.backtest import create_backtrader_engine, analyze_backtest_results
from quantlib.market_data import get_stock_data

# 1. 获取数据
data = get_stock_data('000001', market='CN', period='1y')

# 2. 创建策略
strategy = create_ma_cross_strategy(['000001'], short_window=20, long_window=60)

# 3. 创建回测引擎
engine = create_backtrader_engine(initial_cash=100000, commission=0.001)

# 4. 运行回测
results = engine.run_backtest(strategy, data)

# 5. 分析结果
print(f"总收益率: {results['total_return_pct']:.2f}%")
print(f"最终资金: ${results['final_value']:,.2f}")

# 6. 详细性能分析
engine.print_performance_summary()
```

## 📋 核心组件

### 1. BacktraderEngine - 主回测引擎

基于 Backtrader 框架的专业回测引擎，提供完整的回测功能。

```python
from quantlib.backtest.backtrader_engine import BacktraderEngine

# 创建回测引擎
engine = BacktraderEngine(
    initial_cash=100000.0,  # 初始资金
    commission=0.001        # 手续费率
)

# 运行回测
results = engine.run_backtest(
    strategy=my_strategy,   # 策略实例
    data=stock_data,       # 股票数据
    plot=False            # 是否显示图表
)
```

### 2. 性能分析模块

提供全面的回测性能分析和风险评估。

```python
from quantlib.backtest.performance import (
    calculate_performance_metrics,
    calculate_risk_metrics,
    analyze_trades
)

# 计算性能指标
performance = calculate_performance_metrics(returns_series, benchmark_returns)
print(f"年化收益率: {performance['annual_return_pct']:.2f}%")
print(f"夏普比率: {performance['sharpe_ratio']:.3f}")

# 计算风险指标
risk_metrics = calculate_risk_metrics(returns_series)
print(f"最大回撤: {risk_metrics['max_drawdown_pct']:.2f}%")
print(f"年化波动率: {risk_metrics['volatility_pct']:.2f}%")

# 交易分析
trade_stats = analyze_trades(trade_records)
print(f"胜率: {trade_stats['win_rate_pct']:.1f}%")
print(f"盈亏比: {trade_stats['profit_loss_ratio']:.2f}")
```

## 🛠️ 回测引擎使用

### 1. 基础回测设置

```python
from quantlib.backtest import BacktraderEngine

# 创建回测引擎
engine = BacktraderEngine(
    initial_cash=100000,    # 初始资金
    commission=0.001       # 手续费率 (0.1%)
)

# 配置回测参数
engine.set_parameters(
    cash=100000,           # 初始现金
    commission=0.001,      # 手续费率
    margin=None,           # 保证金（期货用）
    mult=1,                # 合约乘数（期货用）
    slip=0.0              # 滑点
)
```

### 2. 运行回测

```python
# 单个策略回测
results = engine.run_backtest(strategy, data)

# 查看结果
print("回测结果:")
print(f"初始资金: ${results['initial_value']:,.2f}")
print(f"最终资金: ${results['final_value']:,.2f}")
print(f"总收益: ${results['total_return']:,.2f}")
print(f"收益率: {results['total_return_pct']:.2f}%")
print(f"交易次数: {len(results['trades'])}")

# 详细性能摘要
engine.print_performance_summary()
```

### 3. 批量回测

```python
def batch_backtest(strategies, data, initial_cash=100000):
    """批量回测多个策略"""
    results = {}

    for name, strategy in strategies.items():
        print(f"正在回测策略: {name}")

        engine = BacktraderEngine(initial_cash=initial_cash)
        result = engine.run_backtest(strategy, data)

        results[name] = {
            'total_return_pct': result['total_return_pct'],
            'final_value': result['final_value'],
            'trades': len(result['trades']),
            'strategy': strategy
        }

    return results

# 使用示例
strategies = {
    'MA_Cross_20_60': create_ma_cross_strategy(['000001'], 20, 60),
    'MA_Cross_10_30': create_ma_cross_strategy(['000001'], 10, 30),
    'RSI_Strategy': create_rsi_strategy(['000001'])
}

batch_results = batch_backtest(strategies, data)

# 比较结果
for name, result in batch_results.items():
    print(f"{name}: 收益率 {result['total_return_pct']:.2f}%, "
          f"交易次数 {result['trades']}")
```

## 📊 性能分析

### 1. 基础性能指标

```python
from quantlib.backtest.performance import calculate_performance_metrics

# 计算性能指标
performance = calculate_performance_metrics(
    returns=daily_returns,          # 日收益率序列
    benchmark_returns=benchmark,    # 基准收益率（可选）
    risk_free_rate=0.03            # 无风险利率
)

print("📈 收益率指标:")
print(f"总收益率: {performance['total_return_pct']:.2f}%")
print(f"年化收益率: {performance['annual_return_pct']:.2f}%")
print(f"累计收益率: {performance['cumulative_return_pct']:.2f}%")
print(f"平均日收益率: {performance['avg_daily_return_pct']:.3f}%")

print("\n⚠️ 风险指标:")
print(f"年化波动率: {performance['volatility_pct']:.2f}%")
print(f"夏普比率: {performance['sharpe_ratio']:.3f}")
print(f"最大回撤: {performance['max_drawdown_pct']:.2f}%")
```

### 2. 风险分析

```python
from quantlib.backtest.performance import calculate_risk_metrics

risk_metrics = calculate_risk_metrics(returns)

print("📊 详细风险分析:")
print(f"VaR (95%): {risk_metrics['var_95']:.2f}%")
print(f"CVaR (95%): {risk_metrics['cvar_95']:.2f}%")
print(f"索提诺比率: {risk_metrics['sortino_ratio']:.3f}")
print(f"卡尔马比率: {risk_metrics['calmar_ratio']:.3f}")
print(f"最大回撤持续期: {risk_metrics['max_drawdown_duration']} 天")
print(f"偏度: {risk_metrics['skewness']:.3f}")
print(f"峰度: {risk_metrics['kurtosis']:.3f}")
```

### 3. 基准比较

```python
def benchmark_analysis(portfolio_returns, benchmark_returns):
    """与基准的比较分析"""
    from quantlib.backtest.performance import calculate_performance_metrics

    # 计算相对基准的指标
    excess_returns = portfolio_returns - benchmark_returns

    alpha = excess_returns.mean() * 252  # 年化Alpha
    beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)

    # 信息比率
    tracking_error = excess_returns.std() * np.sqrt(252)
    information_ratio = alpha / tracking_error if tracking_error > 0 else 0

    print("📊 基准比较分析:")
    print(f"Alpha (年化): {alpha:.2f}%")
    print(f"Beta: {beta:.3f}")
    print(f"信息比率: {information_ratio:.3f}")
    print(f"跟踪误差: {tracking_error:.2f}%")

    return {
        'alpha': alpha,
        'beta': beta,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error
    }
```

### 4. 交易分析

```python
from quantlib.backtest.performance import analyze_trades

# 交易记录格式示例
trade_records = [
    {
        'symbol': '000001',
        'action': 'buy',
        'quantity': 1000,
        'price': 10.50,
        'timestamp': '2023-01-15',
        'commission': 10.5
    },
    # ... 更多交易记录
]

trade_stats = analyze_trades(trade_records)

print("💼 交易统计:")
print(f"总交易次数: {trade_stats['total_trades']}")
print(f"买入交易: {trade_stats['buy_trades']}")
print(f"卖出交易: {trade_stats['sell_trades']}")
print(f"胜率: {trade_stats['win_rate_pct']:.1f}%")
print(f"平均盈利: ${trade_stats['avg_profit']:.2f}")
print(f"平均亏损: ${trade_stats['avg_loss']:.2f}")
print(f"盈亏比: {trade_stats['profit_loss_ratio']:.2f}")
print(f"最大单笔盈利: ${trade_stats['max_profit']:.2f}")
print(f"最大单笔亏损: ${trade_stats['max_loss']:.2f}")
```

## 🔧 高级功能

### 1. 自定义性能指标

```python
def custom_performance_analysis(returns, prices):
    """自定义性能分析"""
    import pandas as pd
    import numpy as np

    # 计算自定义指标
    def rolling_sharpe(returns, window=252):
        """滚动夏普比率"""
        return (returns.rolling(window).mean() * 252) / (returns.rolling(window).std() * np.sqrt(252))

    def underwater_curve(prices):
        """水下曲线 (Underwater Curve)"""
        peak = prices.expanding().max()
        underwater = (prices - peak) / peak
        return underwater

    def pain_index(returns):
        """痛苦指数 (Pain Index)"""
        drawdowns = underwater_curve(returns.cumsum())
        return np.sqrt(np.mean(drawdowns ** 2))

    # 计算指标
    rolling_sharpe_ratio = rolling_sharpe(returns)
    underwater = underwater_curve(prices)
    pain_idx = pain_index(returns)

    return {
        'rolling_sharpe': rolling_sharpe_ratio,
        'underwater_curve': underwater,
        'pain_index': pain_idx
    }
```

### 2. 分组回测分析

```python
def sector_analysis(strategies_by_sector, data_by_sector):
    """按行业分组的回测分析"""
    sector_results = {}

    for sector, strategies in strategies_by_sector.items():
        print(f"\n分析行业: {sector}")
        sector_data = data_by_sector[sector]

        sector_results[sector] = {}

        for strategy_name, strategy in strategies.items():
            engine = BacktraderEngine(initial_cash=100000)
            result = engine.run_backtest(strategy, sector_data)

            sector_results[sector][strategy_name] = result

            print(f"  {strategy_name}: 收益率 {result['total_return_pct']:.2f}%")

    return sector_results

# 使用示例
strategies_by_sector = {
    '科技': {
        'MA_Cross': create_ma_cross_strategy(['000001'], 20, 60),
        'RSI': create_rsi_strategy(['000001'])
    },
    '金融': {
        'MA_Cross': create_ma_cross_strategy(['000002'], 20, 60),
        'RSI': create_rsi_strategy(['000002'])
    }
}

sector_results = sector_analysis(strategies_by_sector, data_by_sector)
```

### 3. 动态回测

```python
class DynamicBacktest:
    """动态回测类 - 支持实时更新和重新评估"""

    def __init__(self, strategy, initial_cash=100000):
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.engine = BacktraderEngine(initial_cash)
        self.results_history = []

    def update_and_backtest(self, new_data):
        """更新数据并重新回测"""
        # 更新策略数据
        self.strategy.set_data(new_data)
        self.strategy.initialize()

        # 运行回测
        result = self.engine.run_backtest(self.strategy, new_data)

        # 记录历史
        result['timestamp'] = pd.Timestamp.now()
        self.results_history.append(result)

        return result

    def get_performance_trend(self):
        """获取性能趋势"""
        if not self.results_history:
            return None

        trend_data = pd.DataFrame([
            {
                'timestamp': r['timestamp'],
                'total_return_pct': r['total_return_pct'],
                'final_value': r['final_value'],
                'trades': len(r['trades'])
            }
            for r in self.results_history
        ])

        return trend_data
```

### 4. 组合回测

```python
def portfolio_backtest(strategies, weights, data, rebalance_frequency='monthly'):
    """投资组合回测"""
    from quantlib.portfolio import create_portfolio_manager

    # 创建投资组合管理器
    portfolio = create_portfolio_manager(initial_cash=100000)

    # 设置权重
    portfolio.set_target_weights(weights)

    results = []
    rebalance_dates = pd.date_range(
        data.index[0],
        data.index[-1],
        freq='M' if rebalance_frequency == 'monthly' else 'D'
    )

    for date in rebalance_dates:
        # 获取当日数据
        current_data = {}
        for symbol in weights.keys():
            if symbol in data and date in data[symbol].index:
                current_data[symbol] = data[symbol].loc[date]

        if not current_data:
            continue

        # 生成组合信号
        portfolio_signals = []
        for symbol, strategy in strategies.items():
            if symbol in current_data:
                signals = strategy.generate_signals(date, {symbol: current_data[symbol]})
                portfolio_signals.extend(signals)

        # 执行交易并重新平衡
        for signal in portfolio_signals:
            if signal.signal_type == SignalType.BUY:
                portfolio.buy_stock(signal.symbol, 100, current_data[signal.symbol]['close'])
            elif signal.signal_type == SignalType.SELL:
                portfolio.sell_stock(signal.symbol, 100, current_data[signal.symbol]['close'])

        # 更新价格并重新平衡
        portfolio.update_prices(current_data)
        portfolio.rebalance()

        # 记录结果
        performance = portfolio.get_performance_metrics()
        results.append({
            'date': date,
            'total_value': performance['current_value'],
            'total_return': performance['total_return'],
            'positions': len(portfolio.get_positions_summary())
        })

    return pd.DataFrame(results)
```

## 🔍 结果可视化

### 1. 性能图表

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_backtest_results(results, benchmark=None):
    """绘制回测结果图表"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 累计收益曲线
    ax1 = axes[0, 0]
    cumulative_returns = (1 + results['daily_returns']).cumprod()
    ax1.plot(cumulative_returns.index, cumulative_returns.values, label='策略')

    if benchmark is not None:
        benchmark_cumulative = (1 + benchmark).cumprod()
        ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values,
                label='基准', alpha=0.7)

    ax1.set_title('累计收益曲线')
    ax1.legend()
    ax1.grid(True)

    # 2. 回撤曲线
    ax2 = axes[0, 1]
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                    alpha=0.3, color='red', label='回撤')
    ax2.set_title('回撤曲线')
    ax2.legend()
    ax2.grid(True)

    # 3. 月度收益热力图
    ax3 = axes[1, 0]
    monthly_returns = results['daily_returns'].resample('M').sum()
    monthly_returns_table = monthly_returns.groupby([
        monthly_returns.index.year,
        monthly_returns.index.month
    ]).sum().unstack()

    sns.heatmap(monthly_returns_table * 100, annot=True, fmt='.1f',
                cmap='RdYlGn', center=0, ax=ax3)
    ax3.set_title('月度收益率热力图 (%)')

    # 4. 收益分布直方图
    ax4 = axes[1, 1]
    ax4.hist(results['daily_returns'] * 100, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(results['daily_returns'].mean() * 100, color='red',
               linestyle='--', label=f"均值: {results['daily_returns'].mean()*100:.2f}%")
    ax4.set_title('日收益率分布')
    ax4.set_xlabel('日收益率 (%)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()
```

### 2. 风险分析图表

```python
def plot_risk_analysis(returns):
    """绘制风险分析图表"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 滚动波动率
    ax1 = axes[0, 0]
    rolling_vol = returns.rolling(252).std() * np.sqrt(252) * 100
    ax1.plot(rolling_vol.index, rolling_vol.values)
    ax1.set_title('滚动年化波动率 (252日)')
    ax1.set_ylabel('波动率 (%)')
    ax1.grid(True)

    # 2. 滚动夏普比率
    ax2 = axes[0, 1]
    rolling_mean = returns.rolling(252).mean() * 252
    rolling_sharpe = rolling_mean / (returns.rolling(252).std() * np.sqrt(252))
    ax2.plot(rolling_sharpe.index, rolling_sharpe.values)
    ax2.set_title('滚动夏普比率 (252日)')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True)

    # 3. VaR分析
    ax3 = axes[1, 0]
    var_95 = returns.rolling(252).quantile(0.05) * 100
    var_99 = returns.rolling(252).quantile(0.01) * 100
    ax3.plot(var_95.index, var_95.values, label='95% VaR')
    ax3.plot(var_99.index, var_99.values, label='99% VaR')
    ax3.set_title('风险价值 (VaR)')
    ax3.set_ylabel('VaR (%)')
    ax3.legend()
    ax3.grid(True)

    # 4. 收益率散点图（vs基准）
    ax4 = axes[1, 1]
    if len(returns) > 1:
        # 这里需要基准数据，暂时用自相关
        ax4.scatter(returns[:-1], returns[1:], alpha=0.5)
        ax4.set_xlabel('前一日收益率 (%)')
        ax4.set_ylabel('当日收益率 (%)')
        ax4.set_title('收益率散点图')
        ax4.grid(True)

    plt.tight_layout()
    plt.show()
```

## ⚠️ 注意事项

### 1. 数据质量
- 确保数据的完整性和准确性
- 处理股票分红、拆股等公司行为
- 注意数据的时区和交易日历

### 2. 回测偏差
- **生存偏差**: 只包含存续股票，忽略退市股票
- **前瞻偏差**: 使用未来信息进行决策
- **数据偏差**: 历史数据可能不准确
- **过度拟合**: 策略过度适应历史数据

### 3. 交易成本
- 包含真实的手续费和印花税
- 考虑买卖价差（Bid-Ask Spread）
- 计算市场冲击成本
- 考虑滑点影响

### 4. 风险管理
- 设置合理的止损止盈
- 控制单个头寸的风险敞口
- 考虑流动性风险
- 定期评估和调整策略

### 5. 计算资源
- 大规模回测需要足够的计算资源
- 合理设置并行计算
- 优化数据存储和访问
- 监控内存使用

## 🔧 故障排除

### 常见问题

1. **Backtrader未安装**
   ```bash
   pip install backtrader
   ```

2. **数据格式错误**
   ```python
   # 确保数据包含必要列
   required_columns = ['open', 'high', 'low', 'close', 'volume']
   missing = [col for col in required_columns if col not in data.columns]
   if missing:
       print(f"缺少列: {missing}")
   ```

3. **内存不足**
   ```python
   # 分批处理大数据集
   def chunk_backtest(strategy, data, chunk_size=1000):
       results = []
       for i in range(0, len(data), chunk_size):
           chunk_data = data.iloc[i:i+chunk_size]
           result = engine.run_backtest(strategy, chunk_data)
           results.append(result)
       return results
   ```

4. **策略初始化失败**
   ```python
   # 检查策略状态
   if not strategy.is_initialized:
       print("策略未初始化，正在初始化...")
       strategy.initialize()
   ```

## 📖 API 参考

### BacktraderEngine 方法

| 方法 | 说明 | 参数 |
|------|------|------|
| `__init__(initial_cash, commission)` | 初始化引擎 | initial_cash: 初始资金, commission: 手续费率 |
| `run_backtest(strategy, data, plot)` | 运行回测 | strategy: 策略, data: 数据, plot: 是否画图 |
| `print_performance_summary()` | 打印性能摘要 | 无 |

### 性能分析函数

| 函数 | 说明 | 返回值 |
|------|------|-------|
| `calculate_performance_metrics()` | 计算性能指标 | 性能指标字典 |
| `calculate_risk_metrics()` | 计算风险指标 | 风险指标字典 |
| `analyze_trades()` | 分析交易记录 | 交易统计字典 |
| `analyze_backtest_results()` | 综合分析回测结果 | 完整分析报告 |

## 🤝 贡献

欢迎提交问题和改进建议！请遵循项目的代码风格和测试要求。

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。