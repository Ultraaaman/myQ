"""
市场数据模块 (Market Data Module)

本模块提供统一的市场数据接口，包括：
- 多数据源支持（Yahoo Finance、Akshare）
- 数据标准化处理
- 缓存机制
- 实时数据获取
- 公司基本信息
- 指数数据支持

这是quantlib的核心数据模块，所有其他分析模块都通过它获取市场数据。
"""

from .base import BaseDataProvider, DataProviderFactory, DataCache
from .providers import YahooFinanceProvider, AkshareProvider
from .manager import (
    MarketDataManager,
    get_data_manager,
    get_stock_data,
    get_multiple_stocks_data,
    get_csi300_index,
    get_realtime_data,
    get_company_info,
    get_a_share_minute_data,
    get_multiple_a_share_minute_data,
    get_order_book,
    get_tick_data,
    get_intraday_data
)
from .order_book_analyzer import (
    OrderBookAnalyzer,
    TickDataAnalyzer,
    analyze_order_book,
    analyze_tick_data
)

__all__ = [
    # 基础类
    'BaseDataProvider',
    'DataProviderFactory', 
    'DataCache',
    
    # 数据提供者
    'YahooFinanceProvider',
    'AkshareProvider',
    
    # 数据管理器
    'MarketDataManager',
    'get_data_manager',
    
    # 便捷函数
    'get_stock_data',
    'get_multiple_stocks_data',
    'get_csi300_index',
    'get_realtime_data',
    'get_company_info',
    'get_a_share_minute_data',
    'get_multiple_a_share_minute_data',
    'get_order_book',
    'get_tick_data',
    'get_intraday_data',

    # 订单簿分析
    'OrderBookAnalyzer',
    'TickDataAnalyzer',
    'analyze_order_book',
    'analyze_tick_data'
]

__version__ = '1.0.0'