"""
市场数据基础模块
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 尝试导入数据源库
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


class BaseDataProvider(ABC):
    """市场数据提供者基类"""
    
    def __init__(self, symbol: str, market: str = 'US'):
        self.symbol = symbol.upper()
        self.market = market.upper()
        self.data = None
        
    @abstractmethod
    def get_historical_data(self, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """获取历史数据 - 子类必须实现"""
        pass
    
    @abstractmethod
    def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """获取实时数据 - 子类必须实现"""  
        pass
    
    def standardize_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """标准化数据格式
        
        统一数据格式为：date, open, high, low, close, volume
        """
        if data is None or data.empty:
            return None
            
        data = data.copy()
        
        # 标准化列名映射
        column_mapping = {
            'Date': 'date', '日期': 'date', 'datetime': 'date',
            'Open': 'open', '开盘': 'open', 'OPEN': 'open',
            'High': 'high', '最高': 'high', 'HIGH': 'high', 
            'Low': 'low', '最低': 'low', 'LOW': 'low',
            'Close': 'close', '收盘': 'close', 'CLOSE': 'close',
            'Volume': 'volume', '成交量': 'volume', '成交手': 'volume', 'VOLUME': 'volume',
            'Adj Close': 'adj_close', 'ADJ_CLOSE': 'adj_close'
        }
        
        # 重命名列
        for old_name, new_name in column_mapping.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})
        
        # 处理日期列
        if 'date' not in data.columns:
            if data.index.name in ['Date', '日期', 'date', 'datetime']:
                data = data.reset_index()
                if data.columns[0] in ['Date', '日期', 'date', 'datetime']:
                    data = data.rename(columns={data.columns[0]: 'date'})
            elif hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype):
                data = data.reset_index()
                data = data.rename(columns={'index': 'date'})
        
        # 转换日期格式
        if 'date' in data.columns:
            try:
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
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
            data = data.sort_values('date').reset_index(drop=True)
        
        # 数据质量检查
        self._validate_data_quality(data)
        
        return data
    
    def _validate_data_quality(self, data: pd.DataFrame) -> None:
        """数据质量检查"""
        if data is None or data.empty:
            return
            
        required_columns = ['date', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"警告: 数据缺少必需列: {missing_columns}")
        
        # 检查价格逻辑性
        if all(col in data.columns for col in ['high', 'low', 'close', 'open']):
            invalid_prices = (
                (data['high'] < data['low']) | 
                (data['high'] < data['close']) | 
                (data['high'] < data['open']) |
                (data['low'] > data['close']) | 
                (data['low'] > data['open'])
            )
            if invalid_prices.any():
                invalid_count = invalid_prices.sum()
                print(f"警告: 发现 {invalid_count} 条价格逻辑异常记录")
        
        # 检查空值
        if data.isnull().any().any():
            null_counts = data.isnull().sum()
            null_columns = null_counts[null_counts > 0]
            if not null_columns.empty:
                print(f"警告: 数据包含空值: {null_columns.to_dict()}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要信息"""
        if self.data is None or self.data.empty:
            return {}
        
        summary = {
            'symbol': self.symbol,
            'market': self.market,
            'total_records': len(self.data),
            'date_range': {
                'start': self.data['date'].min() if 'date' in self.data.columns else None,
                'end': self.data['date'].max() if 'date' in self.data.columns else None
            },
            'columns': list(self.data.columns)
        }
        
        if 'close' in self.data.columns:
            summary['price_range'] = {
                'min': self.data['close'].min(),
                'max': self.data['close'].max(),
                'current': self.data['close'].iloc[-1] if not self.data.empty else None
            }
        
        if 'volume' in self.data.columns:
            summary['volume_summary'] = {
                'avg': self.data['volume'].mean(),
                'max': self.data['volume'].max(),
                'min': self.data['volume'].min()
            }
        
        return summary


class DataProviderFactory:
    """数据提供者工厂类"""
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, market: str, provider_class):
        """注册数据提供者"""
        cls._providers[market.upper()] = provider_class
    
    @classmethod
    def create_provider(cls, symbol: str, market: str = 'US') -> BaseDataProvider:
        """创建数据提供者实例"""
        market = market.upper()
        
        if market not in cls._providers:
            raise ValueError(f"不支持的市场类型: {market}")
        
        provider_class = cls._providers[market]
        return provider_class(symbol)
    
    @classmethod
    def get_supported_markets(cls) -> List[str]:
        """获取支持的市场列表"""
        return list(cls._providers.keys())


class DataCache:
    """简单的数据缓存类"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key].copy()
        return None
    
    def put(self, key: str, data: pd.DataFrame) -> None:
        """存储数据到缓存"""
        if len(self.cache) >= self.max_size:
            # 移除访问次数最少的项
            least_used = min(self.access_count, key=self.access_count.get)
            self.cache.pop(least_used, None)
            self.access_count.pop(least_used, None)
        
        self.cache[key] = data.copy()
        self.access_count[key] = 1
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)