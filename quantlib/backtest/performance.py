"""
回测性能分析模块 (Performance Analysis Module)

提供全面的回测结果分析功能，包括：
- 收益率分析
- 风险指标计算
- 绩效归因分析
- 基准比较
- 图表生成
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date, timedelta
import warnings

class PerformanceAnalyzer:
    """
    性能分析器

    提供全面的投资组合和策略性能分析功能
    """

    def __init__(self, returns: Union[pd.Series, pd.DataFrame],
                 benchmark: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.02):
        """
        初始化性能分析器

        Args:
            returns: 收益率序列或DataFrame
            benchmark: 基准收益率序列
            risk_free_rate: 无风险利率（年化）
        """
        self.returns = returns.copy()
        self.benchmark = benchmark.copy() if benchmark is not None else None
        self.risk_free_rate = risk_free_rate

        # 确保索引是datetime类型
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            try:
                self.returns.index = pd.to_datetime(self.returns.index)
            except:
                pass

        if self.benchmark is not None and not isinstance(self.benchmark.index, pd.DatetimeIndex):
            try:
                self.benchmark.index = pd.to_datetime(self.benchmark.index)
            except:
                pass

    def calculate_returns_metrics(self) -> Dict[str, float]:
        """计算收益率相关指标"""
        if isinstance(self.returns, pd.DataFrame):
            # 对于DataFrame，使用第一列或'returns'列
            if 'returns' in self.returns.columns:
                returns = self.returns['returns']
            else:
                returns = self.returns.iloc[:, 0]
        else:
            returns = self.returns

        # 基本统计
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)

        # 累计收益
        cumulative_returns = (1 + returns).cumprod() - 1
        final_cumulative_return = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0

        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'avg_daily_return': returns.mean(),
            'avg_daily_return_pct': returns.mean() * 100,
            'cumulative_return': final_cumulative_return,
            'cumulative_return_pct': final_cumulative_return * 100
        }

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """计算风险指标"""
        if isinstance(self.returns, pd.DataFrame):
            if 'returns' in self.returns.columns:
                returns = self.returns['returns']
            else:
                returns = self.returns.iloc[:, 0]
        else:
            returns = self.returns

        # 夏普比率
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 回撤持续时间
        drawdown_periods = self._calculate_drawdown_periods(drawdown)
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

        # VaR (Value at Risk)
        var_95 = returns.quantile(0.05)  # 5% VaR
        var_99 = returns.quantile(0.01)  # 1% VaR

        # 偏度和峰度
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # 下行风险
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (returns.mean() - self.risk_free_rate / 252) / downside_returns.std() * np.sqrt(252)

        # Calmar比率 (年化收益率 / 最大回撤)
        annualized_return = (1 + returns.mean()) ** 252 - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_duration': max_drawdown_duration,
            'var_95': var_95,
            'var_95_pct': var_95 * 100,
            'var_99': var_99,
            'var_99_pct': var_99 * 100,
            'downside_volatility': downside_volatility,
            'downside_volatility_pct': downside_volatility * 100,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def calculate_benchmark_metrics(self) -> Dict[str, float]:
        """计算相对基准的指标"""
        if self.benchmark is None:
            return {}

        if isinstance(self.returns, pd.DataFrame):
            if 'returns' in self.returns.columns:
                returns = self.returns['returns']
            else:
                returns = self.returns.iloc[:, 0]
        else:
            returns = self.returns

        # 对齐时间序列
        aligned_data = pd.concat([returns, self.benchmark], axis=1, join='inner')
        if aligned_data.empty:
            return {}

        strategy_returns = aligned_data.iloc[:, 0]
        benchmark_returns = aligned_data.iloc[:, 1]

        # Alpha和Beta
        excess_strategy = strategy_returns - self.risk_free_rate / 252
        excess_benchmark = benchmark_returns - self.risk_free_rate / 252

        if excess_benchmark.var() > 0:
            beta = excess_strategy.cov(excess_benchmark) / excess_benchmark.var()
            alpha = excess_strategy.mean() - beta * excess_benchmark.mean()
            alpha_annualized = alpha * 252
        else:
            beta = 0
            alpha = 0
            alpha_annualized = 0

        # 信息比率
        active_returns = strategy_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0

        # 胜率
        win_rate = (strategy_returns > benchmark_returns).sum() / len(strategy_returns)

        # 上行/下行捕获率
        up_periods = benchmark_returns > 0
        down_periods = benchmark_returns < 0

        up_capture = 0
        down_capture = 0

        if up_periods.sum() > 0:
            strategy_up = strategy_returns[up_periods].mean()
            benchmark_up = benchmark_returns[up_periods].mean()
            up_capture = strategy_up / benchmark_up if benchmark_up != 0 else 0

        if down_periods.sum() > 0:
            strategy_down = strategy_returns[down_periods].mean()
            benchmark_down = benchmark_returns[down_periods].mean()
            down_capture = strategy_down / benchmark_down if benchmark_down != 0 else 0

        return {
            'alpha': alpha_annualized,
            'alpha_pct': alpha_annualized * 100,
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'tracking_error_pct': tracking_error * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'up_capture': up_capture,
            'down_capture': down_capture
        }

    def calculate_trading_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """
        计算交易相关指标

        Args:
            trades: 交易记录DataFrame，应包含action, amount, pnl等列

        Returns:
            交易指标字典
        """
        if trades.empty:
            return {}

        # 基本交易统计
        total_trades = len(trades)
        buy_trades = len(trades[trades.get('action', '') == 'buy'])
        sell_trades = len(trades[trades.get('action', '') == 'sell'])

        # 盈利交易分析
        if 'pnl' in trades.columns:
            profit_trades = trades[trades['pnl'] > 0]
            loss_trades = trades[trades['pnl'] < 0]

            win_rate = len(profit_trades) / len(trades[trades['pnl'] != 0]) if len(trades[trades['pnl'] != 0]) > 0 else 0
            avg_win = profit_trades['pnl'].mean() if not profit_trades.empty else 0
            avg_loss = abs(loss_trades['pnl'].mean()) if not loss_trades.empty else 0
            profit_factor = (profit_trades['pnl'].sum() / abs(loss_trades['pnl'].sum())) if not loss_trades.empty and loss_trades['pnl'].sum() != 0 else 0

            largest_win = profit_trades['pnl'].max() if not profit_trades.empty else 0
            largest_loss = loss_trades['pnl'].min() if not loss_trades.empty else 0

            # 连续盈利/亏损
            consecutive_wins = self._calculate_consecutive_trades(trades['pnl'] > 0)
            consecutive_losses = self._calculate_consecutive_trades(trades['pnl'] < 0)

            return {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'max_consecutive_wins': max(consecutive_wins) if consecutive_wins else 0,
                'max_consecutive_losses': max(consecutive_losses) if consecutive_losses else 0
            }
        else:
            return {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades
            }

    def _calculate_drawdown_periods(self, drawdown: pd.Series) -> List[int]:
        """计算回撤持续期间"""
        periods = []
        current_period = 0
        in_drawdown = False

        for dd in drawdown:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_period = 1
                else:
                    current_period += 1
            else:
                if in_drawdown:
                    periods.append(current_period)
                    in_drawdown = False
                    current_period = 0

        if in_drawdown:
            periods.append(current_period)

        return periods

    def _calculate_consecutive_trades(self, condition: pd.Series) -> List[int]:
        """计算连续满足条件的交易数"""
        consecutive = []
        current_count = 0

        for cond in condition:
            if cond:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive.append(current_count)
                current_count = 0

        if current_count > 0:
            consecutive.append(current_count)

        return consecutive

    def generate_report(self, trades: Optional[pd.DataFrame] = None) -> str:
        """生成完整的性能报告"""
        report = []
        report.append("=" * 60)
        report.append("投资组合性能分析报告")
        report.append("=" * 60)

        # 收益率指标
        returns_metrics = self.calculate_returns_metrics()
        report.append("\n📈 收益率指标:")
        report.append(f"  总收益率: {returns_metrics.get('total_return_pct', 0):.2f}%")
        report.append(f"  年化收益率: {returns_metrics.get('annualized_return_pct', 0):.2f}%")
        report.append(f"  累计收益率: {returns_metrics.get('cumulative_return_pct', 0):.2f}%")
        report.append(f"  平均日收益率: {returns_metrics.get('avg_daily_return_pct', 0):.3f}%")

        # 风险指标
        risk_metrics = self.calculate_risk_metrics()
        report.append("\n⚠️ 风险指标:")
        report.append(f"  年化波动率: {risk_metrics.get('volatility_pct', 0):.2f}%")
        report.append(f"  夏普比率: {risk_metrics.get('sharpe_ratio', 0):.3f}")
        report.append(f"  索提诺比率: {risk_metrics.get('sortino_ratio', 0):.3f}")
        report.append(f"  卡尔马比率: {risk_metrics.get('calmar_ratio', 0):.3f}")
        report.append(f"  最大回撤: {risk_metrics.get('max_drawdown_pct', 0):.2f}%")
        report.append(f"  最大回撤持续期: {risk_metrics.get('max_drawdown_duration', 0):.0f} 天")
        report.append(f"  95% VaR: {risk_metrics.get('var_95_pct', 0):.2f}%")
        report.append(f"  偏度: {risk_metrics.get('skewness', 0):.3f}")
        report.append(f"  峰度: {risk_metrics.get('kurtosis', 0):.3f}")

        # 基准比较
        if self.benchmark is not None:
            benchmark_metrics = self.calculate_benchmark_metrics()
            if benchmark_metrics:
                report.append("\n📊 基准比较:")
                report.append(f"  Alpha: {benchmark_metrics.get('alpha_pct', 0):.2f}%")
                report.append(f"  Beta: {benchmark_metrics.get('beta', 0):.3f}")
                report.append(f"  信息比率: {benchmark_metrics.get('information_ratio', 0):.3f}")
                report.append(f"  跟踪误差: {benchmark_metrics.get('tracking_error_pct', 0):.2f}%")
                report.append(f"  胜率: {benchmark_metrics.get('win_rate_pct', 0):.1f}%")
                report.append(f"  上行捕获率: {benchmark_metrics.get('up_capture', 0):.3f}")
                report.append(f"  下行捕获率: {benchmark_metrics.get('down_capture', 0):.3f}")

        # 交易统计
        if trades is not None and not trades.empty:
            trading_metrics = self.calculate_trading_metrics(trades)
            if trading_metrics:
                report.append("\n💼 交易统计:")
                report.append(f"  总交易次数: {trading_metrics.get('total_trades', 0)}")
                report.append(f"  买入交易: {trading_metrics.get('buy_trades', 0)}")
                report.append(f"  卖出交易: {trading_metrics.get('sell_trades', 0)}")

                if 'win_rate_pct' in trading_metrics:
                    report.append(f"  胜率: {trading_metrics.get('win_rate_pct', 0):.1f}%")
                    report.append(f"  平均盈利: ${trading_metrics.get('avg_win', 0):.2f}")
                    report.append(f"  平均亏损: ${trading_metrics.get('avg_loss', 0):.2f}")
                    report.append(f"  盈亏比: {trading_metrics.get('profit_factor', 0):.2f}")
                    report.append(f"  最大单笔盈利: ${trading_metrics.get('largest_win', 0):.2f}")
                    report.append(f"  最大单笔亏损: ${trading_metrics.get('largest_loss', 0):.2f}")
                    report.append(f"  最大连续盈利: {trading_metrics.get('max_consecutive_wins', 0)}")
                    report.append(f"  最大连续亏损: {trading_metrics.get('max_consecutive_losses', 0)}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def plot_performance(self, figsize: Tuple[int, int] = (15, 10)):
        """绘制性能图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if isinstance(self.returns, pd.DataFrame):
                if 'returns' in self.returns.columns:
                    returns = self.returns['returns']
                else:
                    returns = self.returns.iloc[:, 0]
            else:
                returns = self.returns

            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('投资组合性能分析', fontsize=16)

            # 累计收益率曲线
            cumulative_returns = (1 + returns).cumprod() - 1
            axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 'b-', linewidth=2, label='策略')

            if self.benchmark is not None:
                benchmark_cumulative = (1 + self.benchmark).cumprod() - 1
                # 对齐时间序列
                aligned_bench = benchmark_cumulative.reindex(cumulative_returns.index, method='nearest')
                axes[0, 0].plot(aligned_bench.index, aligned_bench.values, 'r--', linewidth=2, label='基准')

            axes[0, 0].set_title('累计收益率')
            axes[0, 0].set_ylabel('收益率')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 回撤曲线
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max * 100

            axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            axes[0, 1].set_title('回撤曲线')
            axes[0, 1].set_ylabel('回撤 (%)')
            axes[0, 1].grid(True, alpha=0.3)

            # 日收益率分布
            axes[1, 0].hist(returns.values * 100, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(returns.mean() * 100, color='red', linestyle='--', label=f'均值: {returns.mean()*100:.3f}%')
            axes[1, 0].set_title('日收益率分布')
            axes[1, 0].set_xlabel('日收益率 (%)')
            axes[1, 0].set_ylabel('频数')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 滚动波动率
            rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
            axes[1, 1].plot(rolling_vol.index, rolling_vol.values, 'g-', linewidth=2)
            axes[1, 1].set_title('30日滚动年化波动率')
            axes[1, 1].set_ylabel('波动率 (%)')
            axes[1, 1].grid(True, alpha=0.3)

            # 格式化x轴日期
            for ax in axes.flatten():
                if hasattr(ax.xaxis, 'set_major_formatter'):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib 未安装，无法绘制图表")
        except Exception as e:
            print(f"绘图时发生错误: {e}")

def analyze_backtest_results(returns: Union[pd.Series, pd.DataFrame],
                           benchmark: Optional[pd.Series] = None,
                           trades: Optional[pd.DataFrame] = None,
                           risk_free_rate: float = 0.02,
                           plot: bool = True) -> Dict[str, Any]:
    """
    分析回测结果的便捷函数

    Args:
        returns: 收益率序列
        benchmark: 基准收益率序列
        trades: 交易记录DataFrame
        risk_free_rate: 无风险利率
        plot: 是否绘图

    Returns:
        分析结果字典
    """
    analyzer = PerformanceAnalyzer(returns, benchmark, risk_free_rate)

    results = {
        'returns_metrics': analyzer.calculate_returns_metrics(),
        'risk_metrics': analyzer.calculate_risk_metrics(),
        'analyzer': analyzer
    }

    if benchmark is not None:
        results['benchmark_metrics'] = analyzer.calculate_benchmark_metrics()

    if trades is not None:
        results['trading_metrics'] = analyzer.calculate_trading_metrics(trades)

    # 生成报告
    results['report'] = analyzer.generate_report(trades)

    # 打印报告
    print(results['report'])

    # 绘图
    if plot:
        analyzer.plot_performance()

    return results