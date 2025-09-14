"""
回测模块 (Backtest Module)

本模块提供量化策略回测功能，包括：
- 回测引擎
- 性能分析
- 风险指标计算
- 结果可视化
"""

from .backtrader_engine import (
    BacktraderEngine,
    QuantlibStrategyAdapter,
    BacktraderDataFeed,
    create_backtrader_engine,
    simple_backtest
)

from .zipline_engine import (
    ZiplineEngine,
    ZiplineStrategyAdapter,
    create_zipline_engine,
    simple_zipline_backtest
)

from .performance import (
    PerformanceAnalyzer,
    analyze_backtest_results
)

__all__ = [
    # Backtrader引擎
    'BacktraderEngine',
    'QuantlibStrategyAdapter',
    'BacktraderDataFeed',
    'create_backtrader_engine',
    'simple_backtest',

    # Zipline引擎
    'ZiplineEngine',
    'ZiplineStrategyAdapter',
    'create_zipline_engine',
    'simple_zipline_backtest',

    # 性能分析
    'PerformanceAnalyzer',
    'analyze_backtest_results'
]

__version__ = '1.0.0'