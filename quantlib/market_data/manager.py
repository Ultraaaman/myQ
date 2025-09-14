"""
统一的市场数据管理器
"""
import pandas as pd
from typing import Optional, Dict, List, Any, Union
from .base import DataProviderFactory, DataCache
from .providers import AkshareProvider


class MarketDataManager:
    """市场数据管理器 - 提供统一的数据接口"""
    
    def __init__(self, cache_enabled: bool = True, cache_size: int = 500):
        self.cache_enabled = cache_enabled
        self.cache = DataCache(max_size=cache_size) if cache_enabled else None
        
    def get_stock_data(self, symbol: str, market: str = 'US', 
                      period: str = "1y", interval: str = "1d",
                      use_cache: bool = True) -> Optional[pd.DataFrame]:
        """获取股票历史数据
        
        Args:
            symbol: 股票代码
            market: 市场类型 ('US', 'CN', 'A股')
            period: 时间周期 ('1mo', '3mo', '6mo', '1y', '5y', 'max')
            interval: 数据间隔 ('1d', '1h', '1m')
            use_cache: 是否使用缓存
        
        Returns:
            标准格式的OHLCV数据
        """
        # 生成缓存键
        cache_key = f"{market}_{symbol}_{period}_{interval}"
        
        # 检查缓存
        if use_cache and self.cache_enabled and self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                print(f"从缓存加载 {symbol} 数据")
                return cached_data
        
        try:
            # 创建数据提供者
            provider = DataProviderFactory.create_provider(symbol, market)
            
            # 获取数据
            data = provider.get_historical_data(period=period, interval=interval)
            
            # 存储到缓存
            if data is not None and use_cache and self.cache_enabled and self.cache:
                self.cache.put(cache_key, data)
            
            return data
            
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def get_multiple_stocks(self, symbols: List[str], market: str = 'US',
                          period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """批量获取多只股票数据"""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, market, period, interval)
                if data is not None:
                    results[symbol] = data
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
                
        return results
    
    def get_realtime_data(self, symbol: str, market: str = 'US') -> Optional[Dict[str, Any]]:
        """获取实时行情数据"""
        try:
            provider = DataProviderFactory.create_provider(symbol, market)
            return provider.get_realtime_data()
        except Exception as e:
            print(f"获取 {symbol} 实时数据失败: {e}")
            return None
    
    def get_company_info(self, symbol: str, market: str = 'US') -> Optional[Dict[str, Any]]:
        """获取公司基本信息"""
        try:
            provider = DataProviderFactory.create_provider(symbol, market)
            return provider.get_company_info()
        except Exception as e:
            print(f"获取 {symbol} 公司信息失败: {e}")
            return None
    
    def get_index_data(self, index_symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """获取指数数据
        
        Args:
            index_symbol: 指数代码 ('CSI300', 'SPY', etc.)
            period: 时间周期
        """
        # 特殊处理沪深300指数
        if index_symbol.upper() in ['CSI300', '沪深300', '000300']:
            try:
                provider = AkshareProvider("000300")  # 虚拟symbol
                return provider.get_csi300_data(period=period)
            except Exception as e:
                print(f"获取沪深300指数数据失败: {e}")
                return None
        
        # 其他指数通过正常渠道获取
        # 根据指数代码推断市场
        if index_symbol.startswith(('00', '39')):  # A股指数
            market = 'CN'
        else:  # 美股指数
            market = 'US'
            
        return self.get_stock_data(index_symbol, market=market, period=period)
    
    def search_stocks(self, query: str, market: str = 'US', limit: int = 10) -> List[Dict[str, Any]]:
        """搜索股票（基础实现）"""
        # 这里可以实现股票搜索功能
        # 暂时返回空列表，后续可以扩展
        print(f"股票搜索功能待实现: {query}")
        return []
    
    def get_market_summary(self, market: str = 'US') -> Dict[str, Any]:
        """获取市场概况"""
        try:
            if market.upper() in ['CN', 'A股', 'CHINA']:
                # A股市场概况
                csi300_data = self.get_index_data('CSI300', period='1d')
                if csi300_data is not None and not csi300_data.empty:
                    latest = csi300_data.iloc[-1]
                    prev = csi300_data.iloc[-2] if len(csi300_data) > 1 else latest
                    
                    return {
                        'market': 'A股',
                        'index': '沪深300',
                        'current_value': latest['close'],
                        'change': latest['close'] - prev['close'],
                        'change_percent': ((latest['close'] - prev['close']) / prev['close']) * 100,
                        'volume': latest.get('volume', 0),
                        'timestamp': latest['date']
                    }
            else:
                # 美股市场概况（使用SPY作为基准）
                spy_data = self.get_stock_data('SPY', market='US', period='5d')
                if spy_data is not None and not spy_data.empty:
                    latest = spy_data.iloc[-1]
                    prev = spy_data.iloc[-2] if len(spy_data) > 1 else latest
                    
                    return {
                        'market': '美股',
                        'index': 'SPY',
                        'current_value': latest['close'],
                        'change': latest['close'] - prev['close'],
                        'change_percent': ((latest['close'] - prev['close']) / prev['close']) * 100,
                        'volume': latest.get('volume', 0),
                        'timestamp': latest['date']
                    }
        except Exception as e:
            print(f"获取市场概况失败: {e}")
        
        return {}
    
    def clear_cache(self):
        """清空数据缓存"""
        if self.cache_enabled and self.cache:
            self.cache.clear()
            print("数据缓存已清空")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        if not self.cache_enabled or not self.cache:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'size': self.cache.size(),
            'max_size': self.cache.max_size
        }
    
    def get_supported_markets(self) -> List[str]:
        """获取支持的市场列表"""
        return DataProviderFactory.get_supported_markets()

    def get_order_book(self, symbol: str, market: str = 'CN', depth: int = 5) -> Optional[Dict[str, Any]]:
        """获取订单簿数据（买卖盘口）

        Args:
            symbol: 股票代码
            market: 市场类型
            depth: 档位深度，默认5档

        Returns:
            Dict包含买卖盘口数据
        """
        try:
            provider = DataProviderFactory.create_provider(symbol, market)

            # 检查provider是否支持订单簿数据
            if hasattr(provider, 'get_order_book'):
                return provider.get_order_book(depth=depth)
            else:
                print(f"{market}市场暂不支持订单簿数据")
                return None

        except Exception as e:
            print(f"获取 {symbol} 订单簿数据失败: {e}")
            return None

    def get_tick_data(self, symbol: str, market: str = 'CN', trade_date: str = None) -> Optional[pd.DataFrame]:
        """获取逐笔交易数据

        Args:
            symbol: 股票代码
            market: 市场类型
            trade_date: 交易日期，格式YYYYMMDD

        Returns:
            包含逐笔交易数据的DataFrame
        """
        try:
            provider = DataProviderFactory.create_provider(symbol, market)

            if hasattr(provider, 'get_tick_data'):
                return provider.get_tick_data(trade_date=trade_date)
            else:
                print(f"{market}市场暂不支持逐笔数据")
                return None

        except Exception as e:
            print(f"获取 {symbol} 逐笔数据失败: {e}")
            return None

    def get_intraday_data(self, symbol: str, market: str = 'CN') -> Optional[pd.DataFrame]:
        """获取盘中交易明细数据

        Args:
            symbol: 股票代码
            market: 市场类型

        Returns:
            包含盘中交易明细的DataFrame
        """
        try:
            provider = DataProviderFactory.create_provider(symbol, market)

            if hasattr(provider, 'get_intraday_data'):
                return provider.get_intraday_data()
            else:
                print(f"{market}市场暂不支持盘中数据")
                return None

        except Exception as e:
            print(f"获取 {symbol} 盘中数据失败: {e}")
            return None


# 创建全局数据管理器实例
_global_manager = None

def get_data_manager() -> MarketDataManager:
    """获取全局数据管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = MarketDataManager()
    return _global_manager


# 便捷函数
def get_stock_data(symbol: str, market: str = 'US', period: str = "1y", 
                  interval: str = "1d") -> Optional[pd.DataFrame]:
    """便捷函数：获取单只股票数据"""
    manager = get_data_manager()
    return manager.get_stock_data(symbol, market, period, interval)


def get_multiple_stocks_data(symbols: List[str], market: str = 'US', 
                           period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """便捷函数：获取多只股票数据"""
    manager = get_data_manager()
    return manager.get_multiple_stocks(symbols, market, period, interval)


def get_csi300_index(period: str = "1y") -> Optional[pd.DataFrame]:
    """便捷函数：获取沪深300指数数据"""
    manager = get_data_manager()
    return manager.get_index_data('CSI300', period)


def get_realtime_data(symbol: str, market: str = 'US') -> Optional[Dict[str, Any]]:
    """便捷函数：获取实时数据"""
    manager = get_data_manager()
    return manager.get_realtime_data(symbol, market)


def get_company_info(symbol: str, market: str = 'US') -> Optional[Dict[str, Any]]:
    """便捷函数：获取公司信息"""
    manager = get_data_manager()
    return manager.get_company_info(symbol, market)


# 分钟级数据便捷函数
def get_a_share_minute_data(symbol: str, interval: str = "1min") -> Optional[pd.DataFrame]:
    """便捷函数：获取A股分钟级数据

    Args:
        symbol: 股票代码（如 '000001'）
        interval: 分钟周期，支持 '1min', '5min', '15min', '30min', '60min'
                 或简写 '1m', '5m', '15m', '30m', '60m'

    Returns:
        pandas.DataFrame: 分钟级K线数据

    Note:
        分钟级数据只能获取近5个交易日的数据（akshare限制）
    """
    return get_stock_data(symbol, market='CN', interval=interval)


def get_multiple_a_share_minute_data(symbols: List[str], interval: str = "1min") -> Dict[str, pd.DataFrame]:
    """便捷函数：批量获取A股分钟级数据

    Args:
        symbols: 股票代码列表
        interval: 分钟周期

    Returns:
        dict: {股票代码: 数据DataFrame}
    """
    return get_multiple_stocks_data(symbols, market='CN', interval=interval)


# 订单簿数据便捷函数
def get_order_book(symbol: str, market: str = 'CN', depth: int = 5) -> Optional[Dict[str, Any]]:
    """便捷函数：获取订单簿数据（买卖盘口）

    Args:
        symbol: 股票代码（如 '000001'）
        market: 市场类型，默认为A股
        depth: 档位深度，默认5档

    Returns:
        Dict包含买卖盘口数据，包括：
        - asks: 卖盘列表（价格从低到高）
        - bids: 买盘列表（价格从高到低）
        - spread: 买卖价差
        - mid_price: 中间价格

    Example:
        order_book = get_order_book('000001')
        print(f"最佳买价: {order_book['bids'][0]['price']}")
        print(f"最佳卖价: {order_book['asks'][0]['price']}")
        print(f"买卖价差: {order_book['spread']}")
    """
    manager = get_data_manager()
    return manager.get_order_book(symbol, market, depth)


def get_tick_data(symbol: str, market: str = 'CN', trade_date: str = None) -> Optional[pd.DataFrame]:
    """便捷函数：获取逐笔交易数据

    Args:
        symbol: 股票代码（如 '000001'）
        market: 市场类型，默认为A股
        trade_date: 交易日期，格式YYYYMMDD，默认为今天

    Returns:
        DataFrame包含逐笔交易数据，包括：
        - time: 交易时间
        - price: 成交价格
        - volume: 成交量
        - side: 买卖方向 (buy/sell/neutral)
        - amount: 成交金额

    Example:
        tick_data = get_tick_data('000001')
        print(f"今日共{len(tick_data)}笔交易")
        buy_volume = tick_data[tick_data['side']=='buy']['volume'].sum()
        sell_volume = tick_data[tick_data['side']=='sell']['volume'].sum()
    """
    manager = get_data_manager()
    return manager.get_tick_data(symbol, market, trade_date)


def get_intraday_data(symbol: str, market: str = 'CN') -> Optional[pd.DataFrame]:
    """便捷函数：获取盘中交易明细数据

    Args:
        symbol: 股票代码（如 '000001'）
        market: 市场类型，默认为A股

    Returns:
        DataFrame包含盘中交易明细数据

    Example:
        intraday_data = get_intraday_data('000001')
        print(f"盘中交易明细: {len(intraday_data)} 条记录")
    """
    manager = get_data_manager()
    return manager.get_intraday_data(symbol, market)