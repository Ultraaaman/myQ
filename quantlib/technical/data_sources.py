"""
技术分析数据源模块 - 负责从不同来源获取股票历史价格和技术指标数据
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 尝试导入数据源库
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("警告: yfinance 未安装，无法获取美股数据")

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告: akshare 未安装，无法获取A股数据")


class BaseDataSource:
    """技术分析数据源基类"""
    
    def __init__(self, symbol, market='US'):
        self.symbol = symbol.upper()
        self.market = market
        self.data = None
    
    def get_historical_data(self, period="1y", interval="1d"):
        """获取历史价格数据 - 子类需要实现此方法"""
        raise NotImplementedError
    
    def standardize_data(self, data):
        """标准化数据格式
        确保返回的数据包含标准的OHLCV列名：
        date, open, high, low, close, volume
        """
        if data is None or data.empty:
            return None
            
        # 创建数据副本避免修改原数据
        data = data.copy()
            
        # 标准化列名
        column_mapping = {
            'Date': 'date', '日期': 'date',
            'Open': 'open', '开盘': 'open', 
            'High': 'high', '最高': 'high',
            'Low': 'low', '最低': 'low',
            'Close': 'close', '收盘': 'close',
            'Volume': 'volume', '成交量': 'volume', '成交手': 'volume',
            'Adj Close': 'adj_close'
        }
        
        # 重命名列
        for old_name, new_name in column_mapping.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})
        
        # 确保有日期列 - 特别处理akshare的情况
        if 'date' not in data.columns:
            if data.index.name in ['Date', '日期', 'date']:
                data = data.reset_index()
                if data.columns[0] in ['Date', '日期']:
                    data = data.rename(columns={data.columns[0]: 'date'})
            elif hasattr(data.index, 'name') and data.index.name is None:
                # 如果索引没有名字，可能是日期索引
                data = data.reset_index()
                data = data.rename(columns={'index': 'date'})
        
        # 转换日期格式 - 增强日期解析
        if 'date' in data.columns:
            try:
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                # 移除无效日期
                data = data.dropna(subset=['date'])
            except Exception as e:
                print(f"日期转换警告: {e}")
        
        # 确保数值列为float类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 按日期排序
        if 'date' in data.columns and not data.empty:
            data = data.sort_values('date')
            data = data.reset_index(drop=True)
        
        return data


class YahooFinanceDataSource(BaseDataSource):
    """Yahoo Finance数据源（美股）"""
    
    def __init__(self, symbol):
        super().__init__(symbol, 'US')
        
    def get_historical_data(self, period="1y", interval="1d"):
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
            print(f"✓ {self.symbol}: 获取 {len(data)} 条记录")
            return data
            
        except Exception as e:
            print(f"× {self.symbol}: 获取美股数据失败 - {e}")
            return None


class AkshareDataSource(BaseDataSource):
    """Akshare数据源（A股）"""
    
    def __init__(self, symbol):
        super().__init__(symbol, 'CN')
        
    def get_historical_data(self, period="1y", interval="daily"):
        """获取A股历史数据"""
        if not AKSHARE_AVAILABLE:
            raise ImportError("需要安装akshare: pip install akshare")
            
        try:
            # akshare期间映射
            period_mapping = {
                "1y": None,  # 默认获取最近一年
                "6mo": None,
                "3mo": None,
                "1mo": None
            }
            
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
                else:
                    start_date = None
                    
                if start_date:
                    data['日期'] = pd.to_datetime(data['日期'])
                    data = data[data['日期'] >= start_date]
            
            # 标准化数据格式
            data = self.standardize_data(data)
            
            self.data = data
            print(f"✓ {self.symbol}: 获取 {len(data)} 条记录")
            return data
            
        except Exception as e:
            print(f"× {self.symbol}: 获取A股数据失败 - {e}")
            return None
    
    def get_stock_list(self, market="主板"):
        """获取A股股票列表"""
        if not AKSHARE_AVAILABLE:
            return None
            
        try:
            if market == "主板":
                return ak.stock_info_a_code_name()
            else:
                return ak.stock_info_a_code_name()
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return None
    
    def get_csi300_data(self, period="1y"):
        """获取沪深300指数数据"""
        if not AKSHARE_AVAILABLE:
            raise ImportError("需要安装akshare: pip install akshare")
            
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


class TechnicalDataSourceFactory:
    """技术分析数据源工厂"""
    
    @staticmethod
    def create_data_source(symbol, market='US'):
        """根据市场类型创建相应的数据源"""
        if market.upper() == 'US':
            return YahooFinanceDataSource(symbol)
        elif market.upper() in ['CN', 'A股', 'CHINA']:
            return AkshareDataSource(symbol)
        else:
            raise ValueError(f"不支持的市场类型: {market}")
    
    @staticmethod
    def get_market_data(symbols, market='US', period="1y", interval="1d"):
        """批量获取多只股票数据"""
        results = {}
        
        for symbol in symbols:
            try:
                data_source = TechnicalDataSourceFactory.create_data_source(symbol, market)
                data = data_source.get_historical_data(period=period, interval=interval)
                if data is not None:
                    results[symbol] = data
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
                
        return results


class TechnicalDataManager:
    """技术分析数据管理器"""
    
    def __init__(self):
        self.data_cache = {}
    
    def load_stock_data(self, symbol, market='US', period="1y", interval="1d", use_cache=True):
        """加载股票数据"""
        cache_key = f"{symbol}_{market}_{period}_{interval}"
        
        # 检查缓存
        if use_cache and cache_key in self.data_cache:
            print(f"从缓存加载 {symbol} 数据")
            return self.data_cache[cache_key]
        
        # 获取新数据
        try:
            data_source = TechnicalDataSourceFactory.create_data_source(symbol, market)
            data = data_source.get_historical_data(period=period, interval=interval)
            
            # 缓存数据
            if data is not None and use_cache:
                self.data_cache[cache_key] = data
                
            return data
            
        except Exception as e:
            print(f"加载 {symbol} 数据失败: {e}")
            return None
    
    def load_multiple_stocks(self, symbols, market='US', period="1y", interval="1d"):
        """批量加载多只股票数据"""
        return TechnicalDataSourceFactory.get_market_data(symbols, market, period, interval)
    
    def clear_cache(self):
        """清空数据缓存"""
        self.data_cache.clear()
        print("数据缓存已清空")
    
    def get_a_share_popular_stocks(self):
        """获取A股热门股票代码"""
        return [
            "000001",  # 平安银行
            "000002",  # 万科A
            "000858",  # 五粮液
            "000876",  # 新希望
            "002415",  # 海康威视
            "002594",  # 比亚迪
            "300059",  # 东方财富
            "300750",  # 宁德时代
            "600000",  # 浦发银行
            "600036",  # 招商银行
            "600519",  # 贵州茅台
            "600887",  # 伊利股份
            "601012",  # 隆基绿能
            "601318",  # 中国平安
            "601888"   # 中国中免
        ]


# 便捷函数
def get_stock_data(symbol, market='US', period="1y", interval="1d"):
    """便捷函数：获取单只股票数据"""
    manager = TechnicalDataManager()
    return manager.load_stock_data(symbol, market, period, interval)


def get_multiple_stocks_data(symbols, market='US', period="1y", interval="1d"):
    """便捷函数：获取多只股票数据"""
    manager = TechnicalDataManager()
    return manager.load_multiple_stocks(symbols, market, period, interval)


def get_a_share_data(symbol, period="1y"):
    """便捷函数：获取A股数据"""
    return get_stock_data(symbol, market='CN', period=period, interval="daily")


def get_csi300_index(period="1y"):
    """便捷函数：获取沪深300指数数据"""
    source = AkshareDataSource("000300")  # 虚拟symbol，实际使用sh000300
    return source.get_csi300_data(period)