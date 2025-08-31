"""
基础技术指标类 - 所有技术指标的基类
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any


class TechnicalIndicator(ABC):
    """技术指标基类"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化技术指标
        
        Args:
            data: 包含OHLCV数据的DataFrame，列名应包括：
                 'open', 'high', 'low', 'close', 'volume'
        """
        self.data = data.copy()
        self._validate_data()
        self.results = {}
    
    def _validate_data(self):
        """验证输入数据的有效性"""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        if self.data.empty:
            raise ValueError("输入数据不能为空")
        
        # 确保数据按日期排序
        if 'date' in self.data.columns:
            self.data = self.data.sort_values('date').reset_index(drop=True)
        elif self.data.index.name in ['date', 'datetime'] or pd.api.types.is_datetime64_any_dtype(self.data.index):
            self.data = self.data.sort_index().reset_index(drop=True)
    
    @abstractmethod
    def calculate(self, **kwargs) -> Dict[str, pd.Series]:
        """
        计算技术指标
        
        Returns:
            包含计算结果的字典，键为指标名称，值为对应的数据序列
        """
        pass
    
    def _sma(self, series: pd.Series, period: int) -> pd.Series:
        """简单移动平均线"""
        return series.rolling(window=period).mean()
    
    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        """指数移动平均线"""
        return series.ewm(span=period).mean()
    
    def _highest(self, series: pd.Series, period: int) -> pd.Series:
        """指定周期内的最高值"""
        return series.rolling(window=period).max()
    
    def _lowest(self, series: pd.Series, period: int) -> pd.Series:
        """指定周期内的最低值"""
        return series.rolling(window=period).min()
    
    def _typical_price(self) -> pd.Series:
        """典型价格 (HLC/3)"""
        return (self.data['high'] + self.data['low'] + self.data['close']) / 3
    
    def _true_range(self) -> pd.Series:
        """真实波动范围"""
        hl = self.data['high'] - self.data['low']
        hc = np.abs(self.data['high'] - self.data['close'].shift(1))
        lc = np.abs(self.data['low'] - self.data['close'].shift(1))
        
        return pd.DataFrame({'hl': hl, 'hc': hc, 'lc': lc}).max(axis=1)
    
    def get_signals(self, **kwargs) -> pd.DataFrame:
        """
        生成交易信号
        
        Returns:
            包含买卖信号的DataFrame
        """
        if not self.results:
            self.calculate(**kwargs)
        
        return self._generate_signals()
    
    @abstractmethod
    def _generate_signals(self) -> pd.DataFrame:
        """生成具体的交易信号"""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要统计"""
        if not self.results:
            raise ValueError("请先计算指标")
        
        summary = {}
        for name, series in self.results.items():
            if isinstance(series, pd.Series):
                summary[name] = {
                    'current': series.iloc[-1] if not pd.isna(series.iloc[-1]) else None,
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'count': series.count()
                }
        
        return summary


class PriceBasedIndicator(TechnicalIndicator):
    """基于价格的技术指标基类"""
    
    def __init__(self, data: pd.DataFrame, price_type: str = 'close'):
        """
        初始化价格基础指标
        
        Args:
            data: OHLCV数据
            price_type: 价格类型 ('open', 'high', 'low', 'close', 'typical')
        """
        super().__init__(data)
        self.price_type = price_type
        self.price_series = self._get_price_series()
    
    def _get_price_series(self) -> pd.Series:
        """获取指定类型的价格序列"""
        if self.price_type == 'typical':
            return self._typical_price()
        elif self.price_type in self.data.columns:
            return self.data[self.price_type]
        else:
            raise ValueError(f"不支持的价格类型: {self.price_type}")


class VolumeBasedIndicator(TechnicalIndicator):
    """基于成交量的技术指标基类"""
    
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        
        if 'volume' not in self.data.columns:
            raise ValueError("成交量指标需要volume列")
        
        self.volume_series = self.data['volume']