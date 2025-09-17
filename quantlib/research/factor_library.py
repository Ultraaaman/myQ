"""
因子库管理系统 (Factor Library)

提供完整的因子管理、计算和存储功能
支持动态添加新因子，因子分类管理，计算缓存等
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, date
from enum import Enum
import warnings
import json
import pickle
from pathlib import Path
from abc import ABC, abstractmethod


class FactorCategory(Enum):
    """因子分类"""
    # 技术因子
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLATILITY = "volatility"
    
    # 基本面因子
    VALUE = "value"
    GROWTH = "growth"
    PROFITABILITY = "profitability" 
    QUALITY = "quality"
    LEVERAGE = "leverage"
    
    # 另类因子
    ALTERNATIVE = "alternative"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    
    # 自定义因子
    CUSTOM = "custom"


class BaseFactor(ABC):
    """因子基类"""
    
    def __init__(self, name: str, category: FactorCategory, description: str = ""):
        self.name = name
        self.category = category
        self.description = description
        self.created_time = datetime.now()
        self.parameters = {}
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算因子值
        
        Args:
            data: 股票数据，包含OHLCV和基本面数据
            **kwargs: 其他参数
            
        Returns:
            因子值序列
        """
        pass
    
    def set_parameters(self, **params):
        """设置因子参数"""
        self.parameters.update(params)
        return self
    
    def __str__(self):
        return f"Factor({self.name}, {self.category.value})"


class TechnicalFactor(BaseFactor):
    """技术因子基类"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, FactorCategory.TECHNICAL, description)


class FundamentalFactor(BaseFactor):
    """基本面因子基类"""
    
    def __init__(self, name: str, category: FactorCategory, description: str = ""):
        if category not in [FactorCategory.VALUE, FactorCategory.GROWTH, 
                           FactorCategory.PROFITABILITY, FactorCategory.QUALITY, FactorCategory.LEVERAGE]:
            raise ValueError(f"Invalid fundamental factor category: {category}")
        super().__init__(name, category, description)


class CustomFactor(BaseFactor):
    """自定义因子"""
    
    def __init__(self, name: str, calc_func: Callable, description: str = "", 
                 category: FactorCategory = FactorCategory.CUSTOM):
        super().__init__(name, category, description)
        self.calc_func = calc_func
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """使用自定义函数计算因子"""
        return self.calc_func(data, **kwargs, **self.parameters)


# ===== 预定义因子实现 =====

class MomentumFactor(TechnicalFactor):
    """动量因子"""
    
    def __init__(self, name: str = "momentum", period: int = 20):
        super().__init__(name, f"{period}日动量因子")
        self.period = period
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算动量因子：过去N日收益率"""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        returns = data['close'].pct_change(self.period)
        return returns.fillna(0)


class RSIFactor(TechnicalFactor):
    """RSI因子"""
    
    def __init__(self, name: str = "rsi", period: int = 14):
        super().__init__(name, f"{period}日RSI因子")
        self.period = period
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算RSI因子"""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)


class VolatilityFactor(TechnicalFactor):
    """波动率因子"""
    
    def __init__(self, name: str = "volatility", period: int = 20):
        super().__init__(name, f"{period}日波动率因子")
        self.period = period
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算波动率因子"""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.period).std() * np.sqrt(252)
        return volatility.fillna(0)


class PEFactor(FundamentalFactor):
    """市盈率因子"""
    
    def __init__(self, name: str = "pe_ratio"):
        super().__init__(name, FactorCategory.VALUE, "市盈率因子")
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算PE因子（取倒数，值越大越好）"""
        if 'pe_ratio' in data.columns:
            pe = data['pe_ratio']
            # 取倒数，并处理异常值
            factor = 1 / pe.replace(0, np.nan).clip(0, 100)
            return factor.fillna(0)
        else:
            warnings.warn("PE ratio data not available")
            return pd.Series(index=data.index, data=0)


class PBFactor(FundamentalFactor):
    """市净率因子"""
    
    def __init__(self, name: str = "pb_ratio"):
        super().__init__(name, FactorCategory.VALUE, "市净率因子")
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算PB因子（取倒数，值越大越好）"""
        if 'pb_ratio' in data.columns:
            pb = data['pb_ratio']
            factor = 1 / pb.replace(0, np.nan).clip(0, 20)
            return factor.fillna(0)
        else:
            warnings.warn("PB ratio data not available")
            return pd.Series(index=data.index, data=0)


class ROEFactor(FundamentalFactor):
    """净资产收益率因子"""
    
    def __init__(self, name: str = "roe"):
        super().__init__(name, FactorCategory.PROFITABILITY, "ROE因子")
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算ROE因子"""
        if 'roe' in data.columns:
            return data['roe'].fillna(0)
        else:
            warnings.warn("ROE data not available")
            return pd.Series(index=data.index, data=0)


class FactorCalculator:
    """因子计算器"""
    
    def __init__(self):
        self.cache = {}
        self.cache_enabled = True
        
    def calculate_factor(self, factor: BaseFactor, data: pd.DataFrame, 
                        use_cache: bool = True, **kwargs) -> pd.Series:
        """
        计算单个因子
        
        Args:
            factor: 因子实例
            data: 数据
            use_cache: 是否使用缓存
            **kwargs: 其他参数
            
        Returns:
            因子值序列
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(factor, data, kwargs)
        
        # 检查缓存
        if use_cache and self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        try:
            # 计算因子
            result = factor.calculate(data, **kwargs)
            
            # 缓存结果
            if use_cache and self.cache_enabled:
                self.cache[cache_key] = result.copy()
            
            return result
            
        except Exception as e:
            warnings.warn(f"Failed to calculate factor '{factor.name}': {e}")
            return pd.Series(index=data.index, data=np.nan)
    
    def calculate_factors(self, factors: List[BaseFactor], data: pd.DataFrame,
                         use_cache: bool = True, **kwargs) -> pd.DataFrame:
        """
        批量计算多个因子
        
        Args:
            factors: 因子列表
            data: 数据
            use_cache: 是否使用缓存
            **kwargs: 其他参数
            
        Returns:
            因子值DataFrame，列为因子名，行为时间
        """
        results = {}
        
        for factor in factors:
            try:
                factor_values = self.calculate_factor(factor, data, use_cache, **kwargs)
                results[factor.name] = factor_values
            except Exception as e:
                warnings.warn(f"Failed to calculate factor '{factor.name}': {e}")
                results[factor.name] = pd.Series(index=data.index, data=np.nan)
        
        return pd.DataFrame(results)
    
    def _generate_cache_key(self, factor: BaseFactor, data: pd.DataFrame, kwargs: dict) -> str:
        """生成缓存键"""
        # 简化的缓存键生成
        data_hash = hash(str(data.index.tolist() + data.columns.tolist()))
        params_hash = hash(str(sorted(kwargs.items())))
        factor_hash = hash(f"{factor.name}_{factor.parameters}")
        
        return f"{factor.name}_{data_hash}_{params_hash}_{factor_hash}"
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'cache_size': len(self.cache),
            'cache_enabled': self.cache_enabled,
            'cache_keys': list(self.cache.keys())
        }


class FactorLibrary:
    """
    因子库管理系统
    
    负责因子的注册、管理、计算和存储
    """
    
    def __init__(self, storage_path: str = "data/factor_library"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 因子存储
        self.factors: Dict[str, BaseFactor] = {}
        self.factor_categories: Dict[FactorCategory, List[str]] = {}
        
        # 计算器
        self.calculator = FactorCalculator()
        
        # 元数据
        self.metadata = {
            'created_time': datetime.now(),
            'factor_count': 0,
            'last_updated': datetime.now()
        }
        
        # 自动加载预定义因子
        self._load_default_factors()
        
        # 尝试加载已保存的因子
        self._load_saved_factors()
    
    def register_factor(self, factor: BaseFactor, overwrite: bool = False) -> bool:
        """
        注册因子到库中
        
        Args:
            factor: 因子实例
            overwrite: 是否覆盖已存在的因子
            
        Returns:
            是否注册成功
        """
        if factor.name in self.factors and not overwrite:
            warnings.warn(f"Factor '{factor.name}' already exists. Use overwrite=True to replace.")
            return False
        
        self.factors[factor.name] = factor
        
        # 更新分类索引
        if factor.category not in self.factor_categories:
            self.factor_categories[factor.category] = []
        
        if factor.name not in self.factor_categories[factor.category]:
            self.factor_categories[factor.category].append(factor.name)
        
        # 更新元数据
        self.metadata['factor_count'] = len(self.factors)
        self.metadata['last_updated'] = datetime.now()
        
        return True
    
    def create_custom_factor(self, name: str, calc_func: Callable, 
                           description: str = "", category: FactorCategory = FactorCategory.CUSTOM) -> bool:
        """
        创建并注册自定义因子
        
        Args:
            name: 因子名称
            calc_func: 计算函数，应接受(data, **kwargs)参数
            description: 因子描述
            category: 因子分类
            
        Returns:
            是否创建成功
        """
        try:
            factor = CustomFactor(name, calc_func, description, category)
            return self.register_factor(factor)
        except Exception as e:
            warnings.warn(f"Failed to create custom factor '{name}': {e}")
            return False
    
    def get_factor(self, name: str) -> Optional[BaseFactor]:
        """获取因子"""
        return self.factors.get(name)
    
    def get_factors_by_category(self, category: FactorCategory) -> List[BaseFactor]:
        """按分类获取因子"""
        factor_names = self.factor_categories.get(category, [])
        return [self.factors[name] for name in factor_names if name in self.factors]
    
    def list_factors(self) -> Dict[str, Dict[str, Any]]:
        """列出所有因子"""
        result = {}
        for name, factor in self.factors.items():
            result[name] = {
                'category': factor.category.value,
                'description': factor.description,
                'created_time': factor.created_time,
                'parameters': factor.parameters
            }
        return result
    
    def calculate_factor(self, factor_name: str, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算单个因子"""
        factor = self.get_factor(factor_name)
        if factor is None:
            raise ValueError(f"Factor '{factor_name}' not found in library")
        
        return self.calculator.calculate_factor(factor, data, **kwargs)
    
    def calculate_factors(self, factor_names: List[str], data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算多个因子"""
        factors = []
        for name in factor_names:
            factor = self.get_factor(name)
            if factor is not None:
                factors.append(factor)
            else:
                warnings.warn(f"Factor '{name}' not found, skipping")
        
        return self.calculator.calculate_factors(factors, data, **kwargs)
    
    def calculate_all_factors(self, data: pd.DataFrame, 
                             categories: Optional[List[FactorCategory]] = None, **kwargs) -> pd.DataFrame:
        """
        计算所有因子或指定分类的因子
        
        Args:
            data: 数据
            categories: 指定计算的因子分类，None表示计算所有
            **kwargs: 其他参数
            
        Returns:
            因子值DataFrame
        """
        if categories is None:
            factors = list(self.factors.values())
        else:
            factors = []
            for category in categories:
                factors.extend(self.get_factors_by_category(category))
        
        return self.calculator.calculate_factors(factors, data, **kwargs)
    
    def save_factor_library(self, filename: str = "factor_library.pkl"):
        """保存因子库"""
        filepath = self.storage_path / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'factors': self.factors,
                    'factor_categories': self.factor_categories,
                    'metadata': self.metadata
                }, f)
            
            # 同时保存JSON格式的元数据
            metadata_file = self.storage_path / "factor_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'factor_count': len(self.factors),
                    'categories': {cat.value: len(names) for cat, names in self.factor_categories.items()},
                    'factor_list': self.list_factors(),
                    'last_saved': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False, default=str)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to save factor library: {e}")
            return False
    
    def load_factor_library(self, filename: str = "factor_library.pkl"):
        """加载因子库"""
        filepath = self.storage_path / filename
        
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.factors = data.get('factors', {})
            self.factor_categories = data.get('factor_categories', {})
            self.metadata = data.get('metadata', {})
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to load factor library: {e}")
            return False
    
    def _load_default_factors(self):
        """加载默认因子"""
        default_factors = [
            # 技术因子
            MomentumFactor("momentum_5d", 5),
            MomentumFactor("momentum_20d", 20),
            MomentumFactor("momentum_60d", 60),
            RSIFactor("rsi_14d", 14),
            RSIFactor("rsi_30d", 30),
            VolatilityFactor("volatility_20d", 20),
            VolatilityFactor("volatility_60d", 60),
            
            # 基本面因子
            PEFactor("pe_ratio"),
            PBFactor("pb_ratio"),
            ROEFactor("roe"),
        ]
        
        for factor in default_factors:
            self.register_factor(factor, overwrite=False)
    
    def _load_saved_factors(self):
        """加载已保存的因子"""
        self.load_factor_library()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取因子库摘要"""
        return {
            'total_factors': len(self.factors),
            'categories': {cat.value: len(names) for cat, names in self.factor_categories.items()},
            'storage_path': str(self.storage_path),
            'cache_info': self.calculator.get_cache_info(),
            'metadata': self.metadata
        }
    
    def __str__(self):
        return f"FactorLibrary(factors={len(self.factors)}, categories={len(self.factor_categories)})"


def create_factor_library(storage_path: str = "data/factor_library") -> FactorLibrary:
    """
    创建因子库的便捷函数
    
    Args:
        storage_path: 存储路径
        
    Returns:
        FactorLibrary实例
    """
    return FactorLibrary(storage_path)


# ===== 示例：添加自定义因子 =====

def example_custom_factors():
    """示例：如何添加自定义因子"""
    
    # 创建因子库
    factor_lib = create_factor_library()
    
    # 示例1: 简单的价格因子
    def price_to_ma_ratio(data, period=20, **kwargs):
        """价格与移动平均线比率"""
        ma = data['close'].rolling(period).mean()
        return data['close'] / ma - 1
    
    factor_lib.create_custom_factor(
        name="price_ma_ratio_20d",
        calc_func=price_to_ma_ratio,
        description="20日价格与均线比率",
        category=FactorCategory.TECHNICAL
    )
    
    # 示例2: 成交量因子
    def volume_momentum(data, period=10, **kwargs):
        """成交量动量"""
        return data['volume'].rolling(period).mean() / data['volume'].rolling(period*2).mean() - 1
    
    factor_lib.create_custom_factor(
        name="volume_momentum_10d",
        calc_func=volume_momentum,
        description="10日成交量动量",
        category=FactorCategory.TECHNICAL
    )
    
    # 示例3: 波动率调整动量
    def risk_adjusted_momentum(data, period=20, **kwargs):
        """风险调整动量"""
        returns = data['close'].pct_change()
        momentum = returns.rolling(period).mean()
        volatility = returns.rolling(period).std()
        return momentum / volatility
    
    factor_lib.create_custom_factor(
        name="risk_adj_momentum_20d",
        calc_func=risk_adjusted_momentum,
        description="20日风险调整动量",
        category=FactorCategory.MOMENTUM
    )
    
    return factor_lib