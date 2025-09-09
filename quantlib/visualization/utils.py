"""
图表工具函数
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ChartUtils:
    """图表工具类"""
    
    @staticmethod
    def detect_chart_periods(data: pd.DataFrame) -> str:
        """自动检测数据周期"""
        if len(data) < 2:
            return "1d"
        
        time_diff = data['date'].iloc[1] - data['date'].iloc[0]
        
        if time_diff <= pd.Timedelta(minutes=1):
            return "1m"
        elif time_diff <= pd.Timedelta(minutes=5):
            return "5m"
        elif time_diff <= pd.Timedelta(minutes=15):
            return "15m"
        elif time_diff <= pd.Timedelta(hours=1):
            return "1h"
        elif time_diff <= pd.Timedelta(hours=4):
            return "4h"
        elif time_diff <= pd.Timedelta(days=1):
            return "1d"
        elif time_diff <= pd.Timedelta(weeks=1):
            return "1w"
        else:
            return "1M"
    
    @staticmethod
    def calculate_price_change(data: pd.DataFrame) -> Dict[str, float]:
        """计算价格变化统计"""
        if data.empty:
            return {}
        
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2] if len(data) > 1 else current_price
        first_price = data['close'].iloc[0]
        
        change = current_price - previous_price
        change_percent = (change / previous_price * 100) if previous_price != 0 else 0
        
        total_change = current_price - first_price
        total_change_percent = (total_change / first_price * 100) if first_price != 0 else 0
        
        return {
            'current_price': current_price,
            'previous_price': previous_price,
            'change': change,
            'change_percent': change_percent,
            'total_change': total_change,
            'total_change_percent': total_change_percent,
            'high': data['high'].max(),
            'low': data['low'].min()
        }
    
    @staticmethod
    def identify_support_resistance(data: pd.DataFrame, 
                                  window: int = 20, 
                                  min_touches: int = 3) -> Dict[str, List[float]]:
        """识别支撑阻力位"""
        if len(data) < window * 2:
            return {'support_levels': [], 'resistance_levels': []}
        
        # 找局部高点和低点
        highs = []
        lows = []
        
        for i in range(window, len(data) - window):
            # 检查是否为局部高点
            if data['high'].iloc[i] == data['high'].iloc[i-window:i+window+1].max():
                highs.append((i, data['high'].iloc[i]))
            
            # 检查是否为局部低点
            if data['low'].iloc[i] == data['low'].iloc[i-window:i+window+1].min():
                lows.append((i, data['low'].iloc[i]))
        
        # 聚类相近的价格水平
        def cluster_levels(levels, tolerance=0.01):
            if not levels:
                return []
            
            levels = sorted([level[1] for level in levels])
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) <= tolerance:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= min_touches:
                        clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            if len(current_cluster) >= min_touches:
                clusters.append(np.mean(current_cluster))
            
            return clusters
        
        support_levels = cluster_levels(lows)
        resistance_levels = cluster_levels(highs)
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    @staticmethod
    def calculate_volume_profile(data: pd.DataFrame, bins: int = 50) -> Dict[str, np.ndarray]:
        """计算成交量分布"""
        if 'volume' not in data.columns or data.empty:
            return {'prices': np.array([]), 'volumes': np.array([])}
        
        price_range = data['high'].max() - data['low'].min()
        bin_size = price_range / bins
        
        # 创建价格区间
        price_bins = np.linspace(data['low'].min(), data['high'].max(), bins + 1)
        volume_profile = np.zeros(bins)
        
        for _, row in data.iterrows():
            # 计算每个价格区间的成交量
            low_bin = int((row['low'] - data['low'].min()) / bin_size)
            high_bin = int((row['high'] - data['low'].min()) / bin_size)
            
            low_bin = max(0, min(bins - 1, low_bin))
            high_bin = max(0, min(bins - 1, high_bin))
            
            # 将成交量分配到价格区间
            for i in range(low_bin, high_bin + 1):
                volume_profile[i] += row['volume'] / (high_bin - low_bin + 1)
        
        return {
            'prices': (price_bins[:-1] + price_bins[1:]) / 2,
            'volumes': volume_profile
        }
    
    @staticmethod
    def format_price(price: float, precision: int = 2) -> str:
        """格式化价格显示"""
        if price >= 1000000:
            return f"{price/1000000:.{precision}f}M"
        elif price >= 1000:
            return f"{price/1000:.{precision}f}K"
        else:
            return f"{price:.{precision}f}"
    
    @staticmethod
    def format_volume(volume: float) -> str:
        """格式化成交量显示"""
        if volume >= 1000000000:
            return f"{volume/1000000000:.2f}B"
        elif volume >= 1000000:
            return f"{volume/1000000:.2f}M"
        elif volume >= 1000:
            return f"{volume/1000:.2f}K"
        else:
            return f"{volume:.0f}"
    
    @staticmethod
    def get_optimal_chart_size(data_length: int) -> Tuple[int, int]:
        """根据数据长度获取最佳图表尺寸"""
        if data_length <= 50:
            return (10, 6)
        elif data_length <= 200:
            return (12, 7)
        elif data_length <= 500:
            return (14, 8)
        else:
            return (16, 9)
    
    @staticmethod
    def resample_data(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """重新采样数据到指定频率"""
        if 'date' not in data.columns:
            raise ValueError("数据必须包含date列")
        
        # 设置日期为索引
        resampled_data = data.set_index('date')
        
        # 重新采样
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        
        if 'volume' in resampled_data.columns:
            ohlc_dict['volume'] = 'sum'
        
        resampled = resampled_data.resample(frequency).agg(ohlc_dict)
        
        # 移除空值
        resampled = resampled.dropna()
        
        # 重置索引
        resampled = resampled.reset_index()
        
        return resampled
    
    @staticmethod
    def add_chart_annotations(fig, annotations: List[Dict[str, Any]], engine: str = 'plotly'):
        """添加图表注释"""
        if engine == 'plotly':
            for annotation in annotations:
                fig.add_annotation(
                    x=annotation.get('x'),
                    y=annotation.get('y'),
                    text=annotation.get('text', ''),
                    showarrow=annotation.get('showarrow', True),
                    arrowhead=annotation.get('arrowhead', 2),
                    arrowsize=annotation.get('arrowsize', 1),
                    arrowwidth=annotation.get('arrowwidth', 2),
                    arrowcolor=annotation.get('arrowcolor', 'black'),
                    font=dict(
                        size=annotation.get('font_size', 10),
                        color=annotation.get('font_color', 'black')
                    )
                )


class ColorPalette:
    """颜色调色板"""
    
    # 技术指标常用颜色
    TECHNICAL_COLORS = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 橄榄色
        '#17becf'   # 青色
    ]
    
    # 中国股市传统配色
    CHINESE_COLORS = {
        'up': '#ff0000',      # 红涨
        'down': '#00aa00',    # 绿跌
        'neutral': '#888888'  # 中性
    }
    
    # 美国股市传统配色
    US_COLORS = {
        'up': '#00aa00',      # 绿涨
        'down': '#ff0000',    # 红跌
        'neutral': '#888888'  # 中性
    }
    
    @staticmethod
    def get_color(index: int, palette: str = 'technical') -> str:
        """根据索引获取颜色"""
        if palette == 'technical':
            return ColorPalette.TECHNICAL_COLORS[index % len(ColorPalette.TECHNICAL_COLORS)]
        else:
            return '#000000'
    
    @staticmethod
    def get_market_colors(market: str = 'CN') -> Dict[str, str]:
        """获取市场特定的涨跌配色"""
        if market.upper() in ['CN', 'CHINA', 'A股']:
            return ColorPalette.CHINESE_COLORS
        elif market.upper() in ['US', 'USA', 'AMERICA']:
            return ColorPalette.US_COLORS
        else:
            return ColorPalette.CHINESE_COLORS  # 默认使用中国配色