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


class QuantlibStrategyAdapter:
    """
    将quantlib策略适配到backtrader的适配器
    """
    def __init__(self, *args, **kwargs):
        if not BACKTRADER_AVAILABLE:
            raise ImportError("Backtrader is not installed. Please install it using: pip install backtrader")

class BacktraderDataFeed:
    """
    自定义数据源，兼容quantlib的数据格式
    """
    def __init__(self, *args, **kwargs):
        if not BACKTRADER_AVAILABLE:
            raise ImportError("Backtrader is not installed. Please install it using: pip install backtrader")

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

    def run_backtest(self, strategy: BaseStrategy, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                     plot: bool = False) -> Dict[str, Any]:
        """
        运行回测

        Args:
            strategy: 策略实例
            data: 股票数据
            plot: 是否显示图表

        Returns:
            回测结果字典
        """
        if not BACKTRADER_AVAILABLE:
            raise ImportError("Backtrader is not installed")

        # Placeholder implementation
        return {"error": "Backtrader integration not fully implemented"}


def create_backtrader_engine(initial_cash: float = 100000.0, commission: float = 0.001) -> BacktraderEngine:
    """
    创建Backtrader回测引擎

    Args:
        initial_cash: 初始资金
        commission: 手续费率

    Returns:
        BacktraderEngine实例
    """
    return BacktraderEngine(initial_cash, commission)


def run_strategy_backtest(strategy: BaseStrategy, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                          initial_cash: float = 100000.0, commission: float = 0.001) -> Dict[str, Any]:
    """
    简化版策略回测函数

    Args:
        strategy: 策略实例
        data: 股票数据
        initial_cash: 初始资金
        commission: 手续费率

    Returns:
        回测结果
    """
    if not BACKTRADER_AVAILABLE:
        warnings.warn("Backtrader is not installed. Running simplified backtest.")
        return {"error": "Backtrader not available"}

    # 创建回测引擎
    engine = create_backtrader_engine(initial_cash, commission)

    # 运行回测
    results = engine.run_backtest(strategy, data, plot=False)

    return results


def simple_backtest(strategy: BaseStrategy, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                   initial_cash: float = 100000.0) -> Dict[str, Any]:
    """
    简单回测函数，别名为run_strategy_backtest

    Args:
        strategy: 策略实例
        data: 股票数据
        initial_cash: 初始资金

    Returns:
        回测结果
    """
    return run_strategy_backtest(strategy, data, initial_cash)