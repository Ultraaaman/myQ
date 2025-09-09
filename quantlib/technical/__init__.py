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
    'get_company_info'
]

__version__ = '1.0.0'