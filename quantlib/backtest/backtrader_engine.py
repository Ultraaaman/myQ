"""
Backtrader 回测引擎集成

提供基于 Backtrader 的回测功能，支持：
- 策略回测
- 多种数据源
- 性能分析
- 可视化
"""
import sys
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date
import logging

# 尝试导入backtrader，如果没有安装则提供友好提示
try:
    import backtrader as bt
    import backtrader.analyzers as btanalyzers
    import backtrader.feeds as btfeeds
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    bt = None
    btanalyzers = None
    btfeeds = None

from ..strategy.base import BaseStrategy, SignalType, TradingSignal
from ..market_data import get_stock_data, MarketDataManager

class BacktraderDataFeed(btfeeds.PandasData):
    """
    自定义数据源，兼容quantlib的数据格式
    """
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )

class QuantlibStrategyAdapter(bt.Strategy):
    """
    将quantlib策略适配到backtrader的适配器
    """

    params = (
        ('quantlib_strategy', None),  # quantlib策略实例
        ('printlog', False),
    )

    def __init__(self):
        if not self.params.quantlib_strategy:
            raise ValueError("Must provide quantlib_strategy parameter")

        self.quantlib_strategy = self.params.quantlib_strategy
        self.data_feeds = {}
        self.current_bar = 0

        # 设置数据映射
        for i, data_feed in enumerate(self.datas):
            symbol = getattr(data_feed, '_name', f'data_{i}')
            self.data_feeds[symbol] = data_feed

        # 初始化quantlib策略
        if hasattr(self.quantlib_strategy, 'initialize'):
            self.quantlib_strategy.initialize()

    def next(self):
        """每个bar调用的方法"""
        current_time = self.data.datetime.datetime(0)

        # 构建当前数据
        current_data = {}
        for symbol, data_feed in self.data_feeds.items():
            current_data[symbol] = pd.Series({
                'open': data_feed.open[0],
                'high': data_feed.high[0],
                'low': data_feed.low[0],
                'close': data_feed.close[0],
                'volume': data_feed.volume[0],
                'datetime': current_time
            })

        # 调用quantlib策略的on_bar方法
        if hasattr(self.quantlib_strategy, 'on_bar'):
            self.quantlib_strategy.on_bar(current_time, current_data)

        # 生成信号
        signals = self.quantlib_strategy.generate_signals(current_time, current_data)

        # 执行信号
        for signal in signals:
            self._execute_signal(signal, current_data)

        self.current_bar += 1

    def _execute_signal(self, signal: TradingSignal, current_data: Dict[str, pd.Series]):
        """执行交易信号"""
        if signal.symbol not in self.data_feeds:
            return

        data_feed = self.data_feeds[signal.symbol]

        if signal.signal_type == SignalType.BUY:
            if not self.position:  # 如果没有持仓
                size = self._calculate_size(signal, current_data[signal.symbol]['close'])
                if size > 0:
                    self.buy(data=data_feed, size=size)
                    if self.params.printlog:
                        print(f'BUY  {signal.symbol}: Size={size}, Price={current_data[signal.symbol]["close"]:.2f}')

        elif signal.signal_type == SignalType.SELL:
            if self.position:  # 如果有持仓
                size = signal.quantity or self.position.size
                self.sell(data=data_feed, size=size)
                if self.params.printlog:
                    print(f'SELL {signal.symbol}: Size={size}, Price={current_data[signal.symbol]["close"]:.2f}')

        elif signal.signal_type == SignalType.CLOSE:
            if self.position:  # 平仓
                self.close(data=data_feed)
                if self.params.printlog:
                    print(f'CLOSE {signal.symbol}: Price={current_data[signal.symbol]["close"]:.2f}')

    def _calculate_size(self, signal: TradingSignal, price: float) -> int:
        """计算买入数量"""
        if signal.quantity:
            return int(signal.quantity)

        # 根据可用资金和最大仓位比例计算
        available_cash = self.broker.get_cash()
        max_position_value = available_cash * getattr(self.quantlib_strategy, 'max_position_size', 0.1)

        return int(max_position_value / price) if price > 0 else 0

class BacktraderEngine:
    """
    Backtrader回测引擎

    提供完整的回测功能，包括策略执行、性能分析和结果可视化
    """

    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.001):
        if not BACKTRADER_AVAILABLE:
            raise ImportError(
                "Backtrader is not installed. Please install it using: pip install backtrader"
            )

        self.initial_cash = initial_cash
        self.commission = commission
        self.cerebro = None
        self.results = None
        self.analyzers_results = {}

    def prepare_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> List[Tuple[str, bt.feeds.PandasData]]:
        """
        准备回测数据

        Args:
            data: 股票数据，可以是单个DataFrame或股票代码到DataFrame的字典

        Returns:
            数据源列表，每个元素是(symbol, datafeed)元组
        """
        data_feeds = []

        if isinstance(data, pd.DataFrame):
            # 单只股票
            symbol = getattr(data, 'symbol', 'STOCK')
            datafeed = self._prepare_single_data(data, symbol)
            data_feeds.append((symbol, datafeed))

        elif isinstance(data, dict):
            # 多只股票
            for symbol, df in data.items():
                datafeed = self._prepare_single_data(df, symbol)
                data_feeds.append((symbol, datafeed))
        else:
            raise ValueError("Data must be DataFrame or dict of DataFrames")

        return data_feeds

    def _prepare_single_data(self, df: pd.DataFrame, symbol: str) -> bt.feeds.PandasData:
        """准备单只股票的数据"""
        # 确保数据包含必要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # 处理日期索引
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('date')

        # 确保索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                raise ValueError("Unable to convert index to datetime")

        # 排序数据
        df = df.sort_index()

        # 创建backtrader数据源
        datafeed = BacktraderDataFeed(dataname=df, name=symbol)
        datafeed._name = symbol  # 添加名称属性

        return datafeed

    def run_backtest(self, strategy: BaseStrategy,
                    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                    start_date: Optional[date] = None,
                    end_date: Optional[date] = None,
                    analyzers: Optional[List[str]] = None,
                    plot: bool = False,
                    **kwargs) -> Dict[str, Any]:
        """
        运行回测

        Args:
            strategy: quantlib策略实例
            data: 回测数据
            start_date: 开始日期
            end_date: 结束日期
            analyzers: 分析器列表
            plot: 是否绘制结果
            **kwargs: 其他参数

        Returns:
            回测结果字典
        """
        # 初始化cerebro
        self.cerebro = bt.Cerebro()

        # 设置初始资金
        self.cerebro.broker.setcash(self.initial_cash)

        # 设置手续费
        self.cerebro.broker.setcommission(commission=self.commission)

        # 准备数据
        data_feeds = self.prepare_data(data)

        # 设置策略数据
        if isinstance(data, pd.DataFrame):
            strategy.set_data(data)
        else:
            strategy.set_data(data)

        # 添加数据源
        for symbol, datafeed in data_feeds:
            self.cerebro.adddata(datafeed, name=symbol)

        # 添加策略
        self.cerebro.addstrategy(
            QuantlibStrategyAdapter,
            quantlib_strategy=strategy,
            printlog=kwargs.get('printlog', False)
        )

        # 添加分析器
        default_analyzers = ['sharpe', 'returns', 'drawdown', 'trades']
        if analyzers is None:
            analyzers = default_analyzers

        analyzer_mapping = {
            'sharpe': btanalyzers.SharpeRatio,
            'returns': btanalyzers.Returns,
            'drawdown': btanalyzers.DrawDown,
            'trades': btanalyzers.TradeAnalyzer,
            'positions': btanalyzers.PositionsValue,
            'transactions': btanalyzers.Transactions
        }

        for analyzer_name in analyzers:
            if analyzer_name in analyzer_mapping:
                self.cerebro.addanalyzer(analyzer_mapping[analyzer_name], _name=analyzer_name)

        # 运行回测
        print(f"开始回测...")
        print(f"初始资金: ${self.initial_cash:,.2f}")

        start_value = self.cerebro.broker.getvalue()
        self.results = self.cerebro.run()
        end_value = self.cerebro.broker.getvalue()

        print(f"最终资金: ${end_value:,.2f}")
        print(f"收益: ${end_value - start_value:,.2f} ({((end_value - start_value) / start_value) * 100:.2f}%)")

        # 提取分析器结果
        if self.results:
            strat_results = self.results[0]
            for analyzer_name in analyzers:
                if hasattr(strat_results.analyzers, analyzer_name):
                    analyzer = getattr(strat_results.analyzers, analyzer_name)
                    self.analyzers_results[analyzer_name] = analyzer.get_analysis()

        # 绘制结果
        if plot:
            try:
                self.cerebro.plot(style='candlestick', barup='green', bardown='red')
            except Exception as e:
                print(f"绘图失败: {e}")

        # 构建结果
        results = {
            'initial_value': start_value,
            'final_value': end_value,
            'total_return': end_value - start_value,
            'total_return_pct': ((end_value - start_value) / start_value) * 100,
            'analyzers': self.analyzers_results,
            'strategy': strategy
        }

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.analyzers_results:
            return {}

        summary = {}

        # Sharpe比率
        if 'sharpe' in self.analyzers_results:
            sharpe_data = self.analyzers_results['sharpe']
            summary['sharpe_ratio'] = sharpe_data.get('sharperatio', None)

        # 收益分析
        if 'returns' in self.analyzers_results:
            returns_data = self.analyzers_results['returns']
            summary['total_return'] = returns_data.get('rtot', None)
            summary['avg_return'] = returns_data.get('ravg', None)

        # 回撤分析
        if 'drawdown' in self.analyzers_results:
            dd_data = self.analyzers_results['drawdown']
            summary['max_drawdown'] = dd_data.get('max', {}).get('drawdown', None)
            summary['max_drawdown_period'] = dd_data.get('max', {}).get('len', None)

        # 交易分析
        if 'trades' in self.analyzers_results:
            trades_data = self.analyzers_results['trades']
            total = trades_data.get('total', {})
            won = trades_data.get('won', {})

            summary['total_trades'] = total.get('total', 0)
            summary['winning_trades'] = won.get('total', 0)
            summary['losing_trades'] = summary['total_trades'] - summary['winning_trades']
            summary['win_rate'] = (summary['winning_trades'] / summary['total_trades'] * 100) if summary['total_trades'] > 0 else 0

            summary['avg_win'] = won.get('pnl', {}).get('average', 0)
            summary['avg_loss'] = trades_data.get('lost', {}).get('pnl', {}).get('average', 0)
            summary['profit_factor'] = abs(summary['avg_win'] * summary['winning_trades'] / (summary['avg_loss'] * summary['losing_trades'])) if summary['losing_trades'] > 0 and summary['avg_loss'] != 0 else 0

        return summary

    def print_performance_summary(self):
        """打印性能摘要"""
        summary = self.get_performance_summary()

        if not summary:
            print("No performance data available")
            return

        print("\n" + "="*50)
        print("回测性能摘要")
        print("="*50)

        if 'total_return' in summary:
            print(f"总收益率: {summary['total_return']:.2f}%")

        if 'sharpe_ratio' in summary and summary['sharpe_ratio']:
            print(f"夏普比率: {summary['sharpe_ratio']:.3f}")

        if 'max_drawdown' in summary and summary['max_drawdown']:
            print(f"最大回撤: {summary['max_drawdown']:.2f}%")

        if 'max_drawdown_period' in summary and summary['max_drawdown_period']:
            print(f"最大回撤期间: {summary['max_drawdown_period']} 天")

        print(f"\n交易统计:")
        if 'total_trades' in summary:
            print(f"总交易次数: {summary['total_trades']}")

        if 'winning_trades' in summary:
            print(f"盈利交易: {summary['winning_trades']}")

        if 'losing_trades' in summary:
            print(f"亏损交易: {summary['losing_trades']}")

        if 'win_rate' in summary:
            print(f"胜率: {summary['win_rate']:.1f}%")

        if 'profit_factor' in summary:
            print(f"盈亏比: {summary['profit_factor']:.2f}")

def create_backtrader_engine(initial_cash: float = 100000.0,
                           commission: float = 0.001) -> BacktraderEngine:
    """
    创建Backtrader回测引擎的便捷函数

    Args:
        initial_cash: 初始资金
        commission: 手续费率

    Returns:
        BacktraderEngine实例
    """
    return BacktraderEngine(initial_cash=initial_cash, commission=commission)

# 示例：简单回测函数
def simple_backtest(strategy_class, symbols: List[str],
                   period: str = '1y', **strategy_params) -> Dict[str, Any]:
    """
    简单回测函数，用于快速测试策略

    Args:
        strategy_class: 策略类
        symbols: 股票代码列表
        period: 数据周期
        **strategy_params: 策略参数

    Returns:
        回测结果
    """
    if not BACKTRADER_AVAILABLE:
        raise ImportError("Backtrader is not installed. Please install it using: pip install backtrader")

    # 获取数据
    data_manager = MarketDataManager()

    if len(symbols) == 1:
        data = data_manager.get_stock_data(symbols[0], period=period)
    else:
        data = {}
        for symbol in symbols:
            data[symbol] = data_manager.get_stock_data(symbol, period=period)

    # 创建策略
    strategy = strategy_class(symbols, **strategy_params)

    # 创建回测引擎
    engine = create_backtrader_engine()

    # 运行回测
    results = engine.run_backtest(strategy, data, plot=False)

    # 打印摘要
    engine.print_performance_summary()

    return results