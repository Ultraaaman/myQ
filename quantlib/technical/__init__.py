"""
技术指标分析模块 (Technical Analysis Module)

本模块提供全面的技术指标分析功能，包括：
- 趋势指标：MA、EMA、MACD、布林带等
- 震荡指标：RSI、KDJ、威廉指标等  
- 成交量指标：OBV、成交量价格趋势等
- 支撑阻力：关键价位分析
- 数据源：支持美股(Yahoo Finance)和A股(Akshare)数据
"""

from .base import TechnicalIndicator
from .trend import TrendIndicators
from .oscillator import OscillatorIndicators
from .volume import VolumeIndicators
from .analyzer import TechnicalAnalyzer
# 使用统一的市场数据接口
from ..market_data import (
    get_stock_data,
    get_multiple_stocks_data,
    get_csi300_index,
    get_realtime_data,
    get_company_info
)
# 从market_data模块导入分钟级数据函数，保持向后兼容
from ..market_data import (
    get_a_share_minute_data,
    get_multiple_a_share_minute_data
)
# 为了向后兼容，保留get_a_share_data别名
from ..market_data import get_stock_data as _get_stock_data

def get_a_share_data(symbol, period="1y"):
    """便捷函数：获取A股数据（向后兼容）"""
    return _get_stock_data(symbol, market='CN', period=period, interval="daily")

# Wrapper functions for backward compatibility with strategy module
def calculate_ma(data, period=20, ma_type='sma', price_column='close'):
    """计算移动平均线"""
    import pandas as pd
    from .trend import MovingAverages

    # 如果输入是Series，转换为DataFrame
    if isinstance(data, pd.Series):
        df = pd.DataFrame({price_column: data})
        if hasattr(data, 'index'):
            df.index = data.index

        # 为了满足基类验证，创建虚拟的OHLC列
        if price_column != 'open' and 'open' not in df.columns:
            df['open'] = data
        if price_column != 'high' and 'high' not in df.columns:
            df['high'] = data
        if price_column != 'low' and 'low' not in df.columns:
            df['low'] = data
        if price_column != 'close' and 'close' not in df.columns:
            df['close'] = data
        if 'volume' not in df.columns:
            df['volume'] = 100000  # 虚拟成交量
    else:
        df = data

    ma = MovingAverages(df, price_column)
    results = ma.calculate([period])
    if ma_type.lower() == 'sma':
        return results[f'SMA_{period}']
    else:
        return results[f'EMA_{period}']

def calculate_rsi(data, period=14, price_column='close'):
    """计算RSI指标"""
    import pandas as pd
    from .oscillator import RSI

    # 如果输入是Series，转换为DataFrame
    if isinstance(data, pd.Series):
        df = pd.DataFrame({price_column: data})
        if hasattr(data, 'index'):
            df.index = data.index

        # 为了满足基类验证，创建虚拟的OHLC列
        if price_column != 'open' and 'open' not in df.columns:
            df['open'] = data
        if price_column != 'high' and 'high' not in df.columns:
            df['high'] = data
        if price_column != 'low' and 'low' not in df.columns:
            df['low'] = data
        if price_column != 'close' and 'close' not in df.columns:
            df['close'] = data
        if 'volume' not in df.columns:
            df['volume'] = 100000  # 虚拟成交量
    else:
        df = data

    rsi = RSI(df, price_column)
    results = rsi.calculate(period)
    return results['RSI']

def calculate_bollinger_bands(data, period=20, std_dev=2.0, price_column='close'):
    """计算布林带"""
    import pandas as pd
    from .trend import BollingerBands

    # 如果输入是Series，转换为DataFrame
    if isinstance(data, pd.Series):
        df = pd.DataFrame({price_column: data})
        if hasattr(data, 'index'):
            df.index = data.index

        # 为了满足基类验证，创建虚拟的OHLC列
        if price_column != 'open' and 'open' not in df.columns:
            df['open'] = data
        if price_column != 'high' and 'high' not in df.columns:
            df['high'] = data
        if price_column != 'low' and 'low' not in df.columns:
            df['low'] = data
        if price_column != 'close' and 'close' not in df.columns:
            df['close'] = data
        if 'volume' not in df.columns:
            df['volume'] = 100000  # 虚拟成交量
    else:
        df = data

    bb = BollingerBands(df, price_column)
    results = bb.calculate(period, std_dev)
    return results['Upper_Band'], results['Middle_Band'], results['Lower_Band']

def calculate_macd(data, fast=12, slow=26, signal=9, price_column='close'):
    """计算MACD指标"""
    import pandas as pd
    from .trend import MACD

    # 如果输入是Series，转换为DataFrame
    if isinstance(data, pd.Series):
        df = pd.DataFrame({price_column: data})
        if hasattr(data, 'index'):
            df.index = data.index

        # 为了满足基类验证，创建虚拟的OHLC列
        if price_column != 'open' and 'open' not in df.columns:
            df['open'] = data
        if price_column != 'high' and 'high' not in df.columns:
            df['high'] = data
        if price_column != 'low' and 'low' not in df.columns:
            df['low'] = data
        if price_column != 'close' and 'close' not in df.columns:
            df['close'] = data
        if 'volume' not in df.columns:
            df['volume'] = 100000  # 虚拟成交量
    else:
        df = data

    macd = MACD(df, price_column)
    results = macd.calculate(fast, slow, signal)
    return results['MACD'], results['Signal'], results['Histogram']

def calculate_stochastic(data, k_period=14, d_period=3, smooth_k=3):
    """计算随机指标(KDJ)"""
    import pandas as pd
    from .oscillator import KDJ

    # KDJ需要OHLC数据，如果输入是Series，我们需要更复杂的处理
    # 这里假设输入的data已经是完整的OHLC DataFrame
    if isinstance(data, pd.Series):
        # 如果只有Series，我们无法计算KDJ，因为它需要high、low、close
        raise ValueError("KDJ/Stochastic indicator requires OHLC data, not just a single price series")

    kdj = KDJ(data)
    results = kdj.calculate(k_period, d_period, smooth_k)
    return results['K'], results['D']

__all__ = [
    'TechnicalIndicator',
    'TrendIndicators',
    'OscillatorIndicators',
    'VolumeIndicators',
    'TechnicalAnalyzer',
    'get_stock_data',
    'get_multiple_stocks_data',
    'get_csi300_index',
    'get_realtime_data',
    'get_company_info',
    'get_a_share_minute_data',
    'get_multiple_a_share_minute_data',
    'get_a_share_data',
    # Backward compatibility functions
    'calculate_ma',
    'calculate_rsi',
    'calculate_bollinger_bands',
    'calculate_macd',
    'calculate_stochastic'
]

__version__ = '1.0.0'