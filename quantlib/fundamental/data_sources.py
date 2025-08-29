"""
数据源模块 - 负责从不同来源获取股票数据
"""
import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 尝试导入更多数据源
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


class BaseDataSource:
    """数据源基类"""
    
    def __init__(self, symbol, market='US'):
        self.symbol = symbol.upper()
        self.market = market
    
    def load_company_data(self):
        """加载公司基本数据 - 子类需要实现此方法"""
        raise NotImplementedError
    
    def get_financial_statements(self, start_year="2020"):
        """获取财务报表 - 子类需要实现此方法"""
        raise NotImplementedError


class YahooFinanceDataSource(BaseDataSource):
    """Yahoo Finance数据源"""
    
    def __init__(self, symbol):
        super().__init__(symbol, 'US')
        self.ticker = None
        self.company_info = {}
        
    def load_company_data(self):
        """加载美股公司基本数据"""
        try:
            self.ticker = yf.Ticker(self.symbol)
            self.company_info = self.ticker.info
            print(f"✓ 成功加载 {self.symbol} 的公司信息")
            return True
        except Exception as e:
            print(f"加载公司数据失败: {e}")
            return False
    
    def get_financial_statements(self, start_year="2020"):
        """获取美股财务报表"""
        if not self.ticker:
            return None
            
        try:
            financial_data = {
                'income_statement': self.ticker.financials,
                'balance_sheet': self.ticker.balance_sheet,
                'cash_flow': self.ticker.cashflow
            }
            return financial_data
        except Exception as e:
            print(f"获取财务报表失败: {e}")
            return None
    
    def get_historical_prices(self, period="2y"):
        """获取历史价格数据"""
        if not self.ticker:
            return None
            
        try:
            return self.ticker.history(period=period)
        except Exception as e:
            print(f"获取历史价格失败: {e}")
            return None


class AkshareDataSource(BaseDataSource):
    """Akshare数据源（中国股票）"""
    
    def __init__(self, symbol):
        super().__init__(symbol, 'CN')
        self.company_info = {}
        
    def load_company_data(self):
        """加载中国股票基本数据"""
        if not AKSHARE_AVAILABLE:
            print("需要安装akshare: pip install akshare")
            return False
            
        try:
            stock_info = ak.stock_individual_info_em(symbol=self.symbol)
            info_dict = {}
            for _, row in stock_info.iterrows():
                info_dict[row['item']] = row['value']
            self.company_info = info_dict
            print(f"✓ 成功加载 {self.symbol} 的公司信息")
            return True
        except Exception as e:
            print(f"获取中国股票信息失败: {e}")
            return False
    
    def get_financial_statements(self, start_year="2020"):
        """获取中国股票财务指标"""
        if not AKSHARE_AVAILABLE:
            return None
            
        try:
            # 尝试获取财务指标，添加重试机制
            financial_indicators = None
            max_retries = 2
            
            for attempt in range(max_retries):
                try:
                    financial_indicators = ak.stock_financial_analysis_indicator(
                        symbol=self.symbol, start_year=start_year
                    )
                    if financial_indicators is not None and not financial_indicators.empty:
                        break
                    else:
                        print(f"第 {attempt + 1} 次尝试获取财务指标返回空数据")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(1)  # 短暂延迟后重试
                except Exception as e:
                    print(f"第 {attempt + 1} 次尝试获取财务指标失败: {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)
            
            if financial_indicators is not None and not financial_indicators.empty:
                # 按日期排序，确保最新数据在最后
                financial_indicators['日期'] = pd.to_datetime(financial_indicators['日期'])
                financial_indicators = financial_indicators.sort_values('日期', ascending=True)
                return {'indicators': financial_indicators}
            else:
                print(f"多次尝试后仍无法获取 {self.symbol} 的财务指标")
                return None
        except Exception as e:
            print(f"获取财务指标失败: {e}")
            return None
    
    def get_historical_prices(self, period="daily"):
        """获取中国股票历史价格"""
        if not AKSHARE_AVAILABLE:
            return None
            
        try:
            return ak.stock_zh_a_hist(symbol=self.symbol, period=period, adjust="qfq")
        except Exception as e:
            print(f"获取历史价格失败: {e}")
            return None


class DataSourceFactory:
    """数据源工厂"""
    
    @staticmethod
    def create_data_source(symbol, market='US'):
        """根据市场类型创建相应的数据源"""
        if market == 'US':
            return YahooFinanceDataSource(symbol)
        elif market == 'CN':
            return AkshareDataSource(symbol)
        else:
            raise ValueError(f"不支持的市场类型: {market}")