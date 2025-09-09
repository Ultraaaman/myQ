"""
具体的数据提供者实现
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from .base import BaseDataProvider, DataProviderFactory, YFINANCE_AVAILABLE, AKSHARE_AVAILABLE

try:
    import yfinance as yf
except ImportError:
    pass

try:
    import akshare as ak
except ImportError:
    pass


class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance 数据提供者（美股）"""
    
    def __init__(self, symbol: str):
        super().__init__(symbol, 'US')
        
    def get_historical_data(self, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """获取美股历史数据"""
        if not YFINANCE_AVAILABLE:
            raise ImportError("需要安装yfinance: pip install yfinance")
            
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"警告: {self.symbol} 未获取到数据")
                return None
                
            # 重置索引，将Date从索引移到列
            data = data.reset_index()
            
            # 标准化数据格式
            data = self.standardize_data(data)
            
            self.data = data
            print(f"✓ {self.symbol}: 获取 {len(data)} 条美股记录")
            return data
            
        except Exception as e:
            print(f"× {self.symbol}: 获取美股数据失败 - {e}")
            return None
    
    def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """获取美股实时数据"""
        if not YFINANCE_AVAILABLE:
            return None
            
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            return {
                'symbol': self.symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'volume': info.get('regularMarketVolume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            print(f"获取 {self.symbol} 实时数据失败: {e}")
            return None
    
    def get_company_info(self) -> Optional[Dict[str, Any]]:
        """获取公司基本信息"""
        if not YFINANCE_AVAILABLE:
            return None
            
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            return {
                'symbol': self.symbol,
                'company_name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'employee_count': info.get('fullTimeEmployees'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary')
            }
        except Exception as e:
            print(f"获取 {self.symbol} 公司信息失败: {e}")
            return None


class AkshareProvider(BaseDataProvider):
    """Akshare 数据提供者（A股）"""
    
    def __init__(self, symbol: str):
        super().__init__(symbol, 'CN')
        
    def get_historical_data(self, period: str = "1y", interval: str = "daily") -> Optional[pd.DataFrame]:
        """获取A股历史数据"""
        if not AKSHARE_AVAILABLE:
            raise ImportError("需要安装akshare: pip install akshare")
            
        try:
            # 获取历史数据，使用前复权
            data = ak.stock_zh_a_hist(
                symbol=self.symbol, 
                period=interval, 
                adjust="qfq"  # 前复权
            )
            
            if data.empty:
                print(f"警告: {self.symbol} 未获取到数据")
                return None
            
            # 根据period筛选数据
            if period and period != "max":
                end_date = pd.Timestamp.now()
                if period == "1y":
                    start_date = end_date - pd.DateOffset(years=1)
                elif period == "6mo":
                    start_date = end_date - pd.DateOffset(months=6)
                elif period == "3mo":
                    start_date = end_date - pd.DateOffset(months=3)
                elif period == "1mo":
                    start_date = end_date - pd.DateOffset(months=1)
                elif period == "5y":
                    start_date = end_date - pd.DateOffset(years=5)
                else:
                    start_date = None
                    
                if start_date:
                    data['日期'] = pd.to_datetime(data['日期'])
                    data = data[data['日期'] >= start_date]
            
            # 标准化数据格式
            data = self.standardize_data(data)
            
            self.data = data
            print(f"✓ {self.symbol}: 获取 {len(data)} 条A股记录")
            return data
            
        except Exception as e:
            print(f"× {self.symbol}: 获取A股数据失败 - {e}")
            return None
    
    def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """获取A股实时数据"""
        if not AKSHARE_AVAILABLE:
            return None
            
        try:
            # 获取实时行情数据
            data = ak.stock_zh_a_spot_em()
            
            # 查找指定股票
            stock_data = data[data['代码'] == self.symbol]
            if stock_data.empty:
                return None
            
            stock_info = stock_data.iloc[0]
            
            return {
                'symbol': self.symbol,
                'price': stock_info.get('最新价'),
                'change': stock_info.get('涨跌额'),
                'change_percent': stock_info.get('涨跌幅'),
                'volume': stock_info.get('成交量'),
                'market_cap': stock_info.get('总市值'),
                'pe_ratio': stock_info.get('市盈率-动态'),
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            print(f"获取 {self.symbol} 实时数据失败: {e}")
            return None
    
    def get_company_info(self) -> Optional[Dict[str, Any]]:
        """获取A股公司基本信息"""
        if not AKSHARE_AVAILABLE:
            return None
            
        try:
            # 获取公司概况
            info = ak.stock_individual_info_em(symbol=self.symbol)
            
            if info.empty:
                return None
            
            # 转换为字典格式
            info_dict = {}
            for _, row in info.iterrows():
                info_dict[row['item']] = row['value']
            
            return {
                'symbol': self.symbol,
                'company_name': info_dict.get('公司名称'),
                'industry': info_dict.get('行业'),
                'main_business': info_dict.get('主营业务'),
                'market_cap': info_dict.get('总市值'),
                'employee_count': info_dict.get('员工人数'),
                'website': info_dict.get('公司网址'),
                'listing_date': info_dict.get('上市时间')
            }
        except Exception as e:
            print(f"获取 {self.symbol} 公司信息失败: {e}")
            return None
    
    def get_csi300_data(self, period: str = "1y") -> Optional[pd.DataFrame]:
        """获取沪深300指数数据"""
        try:
            # 获取沪深300指数历史数据
            data = ak.stock_zh_index_daily(symbol="sh000300")
            
            if data.empty:
                print("警告: 沪深300指数未获取到数据")
                return None
            
            # 根据period筛选数据
            if period and period != "max":
                end_date = pd.Timestamp.now()
                if period == "1y":
                    start_date = end_date - pd.DateOffset(years=1)
                elif period == "6mo":
                    start_date = end_date - pd.DateOffset(months=6)
                elif period == "3mo":
                    start_date = end_date - pd.DateOffset(months=3)
                elif period == "1mo":
                    start_date = end_date - pd.DateOffset(months=1)
                elif period == "5y":
                    start_date = end_date - pd.DateOffset(years=5)
                else:
                    start_date = None
                    
                if start_date:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data[data['date'] >= start_date]
            
            # 标准化数据格式
            data = self.standardize_data(data)
            
            print(f"✓ 沪深300指数: 获取 {len(data)} 条记录")
            return data
            
        except Exception as e:
            print(f"× 沪深300指数: 获取数据失败 - {e}")
            return None


# 注册数据提供者
DataProviderFactory.register_provider('US', YahooFinanceProvider)
DataProviderFactory.register_provider('CN', AkshareProvider)
DataProviderFactory.register_provider('A股', AkshareProvider)
DataProviderFactory.register_provider('CHINA', AkshareProvider)