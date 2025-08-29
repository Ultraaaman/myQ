"""
带缓存和错误处理的数据源模块
"""
import time
import json
import os
from .data_sources import YahooFinanceDataSource, AkshareDataSource, DataSourceFactory


class CachedYahooFinanceDataSource(YahooFinanceDataSource):
    """带缓存的Yahoo Finance数据源"""
    
    def __init__(self, symbol, cache_dir="cache"):
        super().__init__(symbol)
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"{symbol}_cache.json")
        self.retry_delay = 2  # 重试延迟（秒）
        self.max_retries = 3  # 最大重试次数
        
        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def load_company_data(self):
        """带缓存和重试的加载公司数据"""
        # 尝试从缓存加载
        if self._load_from_cache():
            print(f"✓ 从缓存加载 {self.symbol} 的公司信息")
            return True
        
        # 缓存失效，尝试从网络获取
        for attempt in range(self.max_retries):
            try:
                print(f"尝试获取 {self.symbol} 数据... (第 {attempt + 1} 次)")
                
                # 添加延迟避免频率限制
                if attempt > 0:
                    time.sleep(self.retry_delay * attempt)
                
                # 尝试加载数据
                if super().load_company_data():
                    # 成功后保存到缓存
                    self._save_to_cache()
                    return True
                    
            except Exception as e:
                print(f"第 {attempt + 1} 次尝试失败: {e}")
                if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                        print(f"遇到频率限制，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    continue
                else:
                    break
        
        # 如果网络获取失败，尝试使用旧缓存
        if self._load_from_cache(ignore_expiry=True):
            print(f"⚠️ 使用过期缓存数据 {self.symbol}")
            return True
        
        print(f"❌ 无法获取 {self.symbol} 的数据")
        return False
    
    def _load_from_cache(self, ignore_expiry=False):
        """从缓存加载数据"""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查缓存是否过期（默认1天）
            if not ignore_expiry:
                cache_time = cache_data.get('timestamp', 0)
                current_time = time.time()
                if current_time - cache_time > 86400:  # 24小时
                    return False
            
            # 恢复数据
            self.company_info = cache_data.get('company_info', {})
            
            # 创建模拟ticker对象（简化版）
            class MockTicker:
                def __init__(self, info):
                    self.info = info
                    
                def history(self, period='1d'):
                    import pandas as pd
                    from datetime import datetime
                    # 返回模拟价格数据
                    price = info.get('currentPrice', 100)
                    return pd.DataFrame({
                        'Close': [price],
                        'Volume': [1000000]
                    }, index=[datetime.now()])
                    
                @property
                def financials(self):
                    # 返回空的DataFrame以避免错误
                    import pandas as pd
                    return pd.DataFrame()
                
                @property 
                def balance_sheet(self):
                    import pandas as pd
                    return pd.DataFrame()
                    
                @property
                def cashflow(self):
                    import pandas as pd
                    return pd.DataFrame()
            
            self.ticker = MockTicker(self.company_info)
            return True
            
        except Exception as e:
            print(f"缓存加载失败: {e}")
            return False
    
    def _save_to_cache(self):
        """保存数据到缓存"""
        try:
            cache_data = {
                'timestamp': time.time(),
                'symbol': self.symbol,
                'company_info': self.company_info
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            print(f"缓存保存失败: {e}")


class ResilientDataSourceFactory(DataSourceFactory):
    """支持错误恢复的数据源工厂"""
    
    @staticmethod
    def create_data_source(symbol, market='US'):
        """创建具有错误恢复能力的数据源"""
        if market == 'US':
            return CachedYahooFinanceDataSource(symbol)
        elif market == 'CN':
            return AkshareDataSource(symbol)
        else:
            raise ValueError(f"不支持的市场类型: {market}")