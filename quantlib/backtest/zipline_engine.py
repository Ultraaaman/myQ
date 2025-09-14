"""
Zipline 回测引擎集成

提供基于 Zipline 的回测功能，支持：
- 策略回测
- 风险管理
- 性能分析
- 多资产组合
"""
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, date
import logging

# 尝试导入zipline相关模块，如果没有安装则提供友好提示
try:
    from zipline import run_algorithm
    from zipline.api import (
        order, order_target, order_percent, order_target_percent,
        symbol, symbols, get_datetime, record, schedule_function,
        date_rules, time_rules, set_commission, set_slippage
    )
    from zipline.finance import commission, slippage
    from zipline.data import bundles
    from zipline.utils.calendars import get_calendar
    import zipline.utils.events as events
    ZIPLINE_AVAILABLE = True
except ImportError:
    ZIPLINE_AVAILABLE = False
    run_algorithm = None

from ..strategy.base import BaseStrategy, SignalType, TradingSignal
from ..market_data import MarketDataManager

class ZiplineStrategyAdapter:
    """
    将quantlib策略适配到zipline的适配器
    """

    def __init__(self, quantlib_strategy: BaseStrategy):
        self.quantlib_strategy = quantlib_strategy
        self.symbols_list = quantlib_strategy.symbols
        self.current_data_cache = {}
        self.initialized = False

    def initialize(self, context):
        """zipline初始化函数"""
        # 设置股票池
        context.symbols = symbols(*self.symbols_list)
        context.quantlib_strategy = self.quantlib_strategy

        # 设置手续费和滑点
        set_commission(commission.PerTrade(cost=5.0))  # 每笔交易5元手续费
        set_slippage(slippage.VolumeShareSlippage())

        # 初始化quantlib策略
        if hasattr(self.quantlib_strategy, 'initialize'):
            self.quantlib_strategy.initialize()

        self.initialized = True

        # 设置定时任务
        schedule_function(
            self.handle_data_wrapper,
            date_rules.every_day(),
            time_rules.market_close()
        )

    def handle_data_wrapper(self, context, data):
        """数据处理包装函数"""
        self.handle_data(context, data)

    def handle_data(self, context, data):
        """处理每日数据"""
        if not self.initialized:
            return

        current_time = get_datetime()

        # 构建当前数据字典
        current_data = {}
        for sym in context.symbols:
            if data.can_trade(sym):
                current_price = data.current(sym, 'price')
                current_volume = data.current(sym, 'volume')

                # 获取历史数据构建完整的数据行
                hist = data.history(sym, ['open', 'high', 'low', 'close', 'volume'], 1, '1d')
                if len(hist) > 0:
                    row = hist.iloc[-1]
                    current_data[sym.symbol] = pd.Series({
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': current_price,
                        'volume': current_volume,
                        'datetime': current_time
                    })

        # 更新缓存
        self.current_data_cache = current_data

        # 调用quantlib策略
        if hasattr(self.quantlib_strategy, 'on_bar'):
            self.quantlib_strategy.on_bar(current_time, current_data)

        # 生成信号
        signals = self.quantlib_strategy.generate_signals(current_time, current_data)

        # 执行信号
        for signal in signals:
            self._execute_signal(context, data, signal)

    def _execute_signal(self, context, data, signal: TradingSignal):
        """执行交易信号"""
        try:
            sym = symbol(signal.symbol)
        except:
            return  # 股票代码无效

        if not data.can_trade(sym):
            return

        current_price = data.current(sym, 'price')

        if signal.signal_type == SignalType.BUY:
            # 计算买入数量
            if signal.quantity:
                shares = signal.quantity
                order(sym, shares)
            else:
                # 按照最大仓位比例买入
                max_position_pct = getattr(self.quantlib_strategy, 'max_position_size', 0.1)
                order_target_percent(sym, max_position_pct)

            record(action='BUY', symbol=signal.symbol, price=current_price)

        elif signal.signal_type == SignalType.SELL:
            current_position = context.portfolio.positions[sym].amount
            if current_position > 0:
                if signal.quantity:
                    shares_to_sell = min(signal.quantity, current_position)
                    order(sym, -shares_to_sell)
                else:
                    order_target(sym, 0)  # 全部卖出

                record(action='SELL', symbol=signal.symbol, price=current_price)

        elif signal.signal_type == SignalType.CLOSE:
            order_target(sym, 0)  # 平仓
            record(action='CLOSE', symbol=signal.symbol, price=current_price)

class ZiplineEngine:
    """
    Zipline回测引擎

    提供完整的基于zipline的回测功能
    """

    def __init__(self, initial_capital: float = 100000.0):
        if not ZIPLINE_AVAILABLE:
            raise ImportError(
                "Zipline is not installed. Please install it using: pip install zipline-reloaded"
            )

        self.initial_capital = initial_capital
        self.results = None

    def run_backtest(self, strategy: BaseStrategy,
                    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                    start_date: date,
                    end_date: date,
                    benchmark_symbol: Optional[str] = None,
                    bundle: str = 'custom-data-bundle',
                    **kwargs) -> Dict[str, Any]:
        """
        运行zipline回测

        Args:
            strategy: quantlib策略实例
            data: 回测数据
            start_date: 开始日期
            end_date: 结束日期
            benchmark_symbol: 基准股票代码
            bundle: 数据包名称
            **kwargs: 其他参数

        Returns:
            回测结果字典
        """
        # 准备数据
        self._prepare_data_bundle(data, bundle)

        # 设置策略数据
        if isinstance(data, pd.DataFrame):
            strategy.set_data(data)
        else:
            strategy.set_data(data)

        # 创建策略适配器
        adapter = ZiplineStrategyAdapter(strategy)

        # 运行回测
        try:
            print(f"开始Zipline回测...")
            print(f"开始日期: {start_date}")
            print(f"结束日期: {end_date}")
            print(f"初始资金: ${self.initial_capital:,.2f}")

            self.results = run_algorithm(
                start=pd.Timestamp(start_date),
                end=pd.Timestamp(end_date),
                initialize=adapter.initialize,
                capital_base=self.initial_capital,
                data_frequency='daily',
                bundle=bundle
            )

            print(f"回测完成")
            print(f"最终资产: ${self.results.portfolio_value.iloc[-1]:,.2f}")
            print(f"总收益: {((self.results.portfolio_value.iloc[-1] / self.initial_capital) - 1) * 100:.2f}%")

        except Exception as e:
            print(f"Zipline回测失败: {e}")
            raise

        return self._process_results()

    def _prepare_data_bundle(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], bundle_name: str):
        """准备zipline数据包"""
        # 这里应该实现自定义数据包的注册
        # 由于zipline的数据包机制比较复杂，这里提供一个简化的实现框架
        pass

    def _process_results(self) -> Dict[str, Any]:
        """处理回测结果"""
        if self.results is None:
            return {}

        results = self.results

        # 基本性能指标
        total_return = (results.portfolio_value.iloc[-1] / results.portfolio_value.iloc[0]) - 1
        daily_returns = results.returns
        annual_return = daily_returns.mean() * 252
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0

        # 最大回撤
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 交易统计
        transactions = results.transactions if hasattr(results, 'transactions') else pd.DataFrame()
        total_trades = len(transactions) if not transactions.empty else 0

        # 构建结果字典
        performance_summary = {
            'initial_capital': self.initial_capital,
            'final_value': results.portfolio_value.iloc[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annual_return': annual_return,
            'annual_return_pct': annual_return * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': total_trades,
            'portfolio_value': results.portfolio_value,
            'returns': results.returns,
            'positions': results.positions if hasattr(results, 'positions') else None,
            'transactions': transactions,
            'strategy': results  # 保存完整结果供进一步分析
        }

        return performance_summary

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        if self.results is None:
            return {}

        return self._process_results()

    def plot_results(self, figsize: tuple = (12, 8)):
        """绘制回测结果"""
        if self.results is None:
            print("No results to plot")
            return

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # 资产价值曲线
            axes[0, 0].plot(self.results.portfolio_value)
            axes[0, 0].set_title('Portfolio Value')
            axes[0, 0].set_ylabel('Value ($)')

            # 日收益率
            axes[0, 1].plot(self.results.returns)
            axes[0, 1].set_title('Daily Returns')
            axes[0, 1].set_ylabel('Returns')

            # 累积收益率
            cumulative_returns = (1 + self.results.returns).cumprod() - 1
            axes[1, 0].plot(cumulative_returns)
            axes[1, 0].set_title('Cumulative Returns')
            axes[1, 0].set_ylabel('Cumulative Returns')

            # 回撤
            rolling_max = (1 + self.results.returns).cumprod().expanding().max()
            drawdown = ((1 + self.results.returns).cumprod() - rolling_max) / rolling_max
            axes[1, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            axes[1, 1].set_title('Drawdown')
            axes[1, 1].set_ylabel('Drawdown')

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")

    def print_performance_summary(self):
        """打印性能摘要"""
        metrics = self.get_performance_metrics()

        if not metrics:
            print("No performance data available")
            return

        print("\n" + "="*50)
        print("Zipline回测性能摘要")
        print("="*50)

        print(f"初始资金: ${metrics.get('initial_capital', 0):,.2f}")
        print(f"最终价值: ${metrics.get('final_value', 0):,.2f}")
        print(f"总收益率: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"年化收益率: {metrics.get('annual_return_pct', 0):.2f}%")
        print(f"波动率: {metrics.get('volatility_pct', 0):.2f}%")
        print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"最大回撤: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"总交易次数: {metrics.get('total_trades', 0)}")

def create_zipline_engine(initial_capital: float = 100000.0) -> ZiplineEngine:
    """
    创建Zipline回测引擎的便捷函数

    Args:
        initial_capital: 初始资金

    Returns:
        ZiplineEngine实例
    """
    return ZiplineEngine(initial_capital=initial_capital)

# 简单的数据包注册函数（需要根据实际数据定制）
def register_custom_data_bundle(data: Dict[str, pd.DataFrame], bundle_name: str = 'custom-data-bundle'):
    """
    注册自定义数据包

    这是一个简化的实现，实际使用时需要根据zipline的数据包机制进行完整实现
    """
    if not ZIPLINE_AVAILABLE:
        return

    # 这里应该实现完整的数据包注册逻辑
    # 由于zipline的数据包机制相对复杂，建议参考zipline官方文档
    print(f"数据包 {bundle_name} 注册功能需要根据具体需求实现")

def simple_zipline_backtest(strategy_class, symbols: List[str],
                           start_date: date, end_date: date,
                           period: str = '1y', **strategy_params) -> Dict[str, Any]:
    """
    简单Zipline回测函数

    Args:
        strategy_class: 策略类
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        period: 数据周期
        **strategy_params: 策略参数

    Returns:
        回测结果
    """
    if not ZIPLINE_AVAILABLE:
        raise ImportError("Zipline is not installed. Please install it using: pip install zipline-reloaded")

    print("注意: Zipline回测需要特定的数据包配置，当前提供简化实现")
    print("建议使用BacktraderEngine进行完整的回测功能")

    # 创建策略
    strategy = strategy_class(symbols, **strategy_params)

    # 创建引擎
    engine = create_zipline_engine()

    # 获取数据（这里需要实际的数据获取逻辑）
    data_manager = MarketDataManager()

    if len(symbols) == 1:
        data = data_manager.get_stock_data(symbols[0], period=period)
    else:
        data = {}
        for symbol in symbols:
            data[symbol] = data_manager.get_stock_data(symbol, period=period)

    # 注册数据包
    register_custom_data_bundle(data if isinstance(data, dict) else {symbols[0]: data})

    try:
        # 运行回测
        results = engine.run_backtest(strategy, data, start_date, end_date)

        # 打印摘要
        engine.print_performance_summary()

        return results
    except Exception as e:
        print(f"Zipline回测失败: {e}")
        print("建议使用BacktraderEngine作为替代方案")
        return {}