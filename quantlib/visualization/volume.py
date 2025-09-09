"""
成交量图表模块
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from .base import BaseChart
from .utils import ColorPalette
import warnings
warnings.filterwarnings('ignore')

# 导入绘图库
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class VolumeChart(BaseChart):
    """成交量图表类"""
    
    def __init__(self, data: pd.DataFrame, engine: str = 'auto'):
        super().__init__(data, engine)
        
        if 'volume' not in self.data.columns:
            raise ValueError("数据必须包含volume列")
            
        self.title = "成交量分析"
        self.width = 12
        self.height = 8
        self.volume_profile_enabled = False
        self.volume_ma_periods = []
        
    def add_volume_ma(self, periods: List[int] = [5, 10, 20]) -> 'VolumeChart':
        """添加成交量移动平均线"""
        self.volume_ma_periods = periods
        
        for period in periods:
            ma_name = f'Vol_MA_{period}'
            self.data[ma_name] = self.data['volume'].rolling(window=period).mean()
            
        return self
    
    def enable_volume_profile(self, enabled: bool = True) -> 'VolumeChart':
        """启用成交量分布图"""
        self.volume_profile_enabled = enabled
        return self
    
    def plot(self) -> 'VolumeChart':
        """绘制成交量图表"""
        if self.engine == 'plotly':
            self._plot_plotly()
        elif self.engine == 'matplotlib':
            self._plot_matplotlib()
        else:
            raise ValueError(f"不支持的绘图引擎: {self.engine}")
            
        return self
    
    def _plot_plotly(self):
        """使用plotly绘制成交量图表"""
        # 创建子图布局
        if self.volume_profile_enabled:
            # 主图 + 成交量分布
            self.figure = make_subplots(
                rows=2, cols=2,
                column_widths=[0.8, 0.2],
                row_heights=[0.6, 0.4],
                shared_xaxes=True,
                shared_yaxes='columns',
                subplot_titles=['K线图', '成交量分布', '成交量', ''],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
        else:
            # 主图 + 成交量
            self.figure = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                subplot_titles=['K线图', '成交量'],
                vertical_spacing=0.1
            )
        
        # 绘制K线图
        candlestick = go.Candlestick(
            x=self.data['date'],
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name='K线',
            increasing_line_color='red',
            decreasing_line_color='green'
        )
        
        self.figure.add_trace(candlestick, row=1, col=1)
        
        # 绘制成交量柱状图
        volume_colors = ['red' if close >= open else 'green' 
                        for close, open in zip(self.data['close'], self.data['open'])]
        
        volume_bar = go.Bar(
            x=self.data['date'],
            y=self.data['volume'],
            name='成交量',
            marker_color=volume_colors,
            opacity=0.7
        )
        
        if self.volume_profile_enabled:
            self.figure.add_trace(volume_bar, row=2, col=1)
        else:
            self.figure.add_trace(volume_bar, row=2, col=1)
        
        # 添加成交量移动平均线
        colors = ColorPalette.TECHNICAL_COLORS
        for i, period in enumerate(self.volume_ma_periods):
            ma_name = f'Vol_MA_{period}'
            if ma_name in self.data.columns:
                ma_trace = go.Scatter(
                    x=self.data['date'],
                    y=self.data[ma_name],
                    mode='lines',
                    name=f'Vol MA{period}',
                    line=dict(color=colors[i % len(colors)], width=1)
                )
                
                if self.volume_profile_enabled:
                    self.figure.add_trace(ma_trace, row=2, col=1)
                else:
                    self.figure.add_trace(ma_trace, row=2, col=1)
        
        # 添加成交量分布图
        if self.volume_profile_enabled:
            self._add_volume_profile_plotly()
        
        # 更新布局
        self.figure.update_layout(
            title=self.title,
            width=self.width * 100,
            height=self.height * 100,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
    
    def _add_volume_profile_plotly(self):
        """添加成交量分布图"""
        from .utils import ChartUtils
        
        # 计算成交量分布
        volume_profile = ChartUtils.calculate_volume_profile(self.data, bins=50)
        
        if len(volume_profile['volumes']) > 0:
            # 横向柱状图显示成交量分布
            profile_trace = go.Bar(
                x=volume_profile['volumes'],
                y=volume_profile['prices'],
                orientation='h',
                name='成交量分布',
                marker_color='lightblue',
                opacity=0.7
            )
            
            self.figure.add_trace(profile_trace, row=1, col=2)
    
    def _plot_matplotlib(self):
        """使用matplotlib绘制成交量图表"""
        if self.volume_profile_enabled:
            # 创建复杂布局
            fig = plt.figure(figsize=(self.width, self.height))
            
            # 定义网格布局
            gs = fig.add_gridspec(2, 2, 
                                width_ratios=[4, 1], 
                                height_ratios=[3, 1],
                                hspace=0.1, wspace=0.1)
            
            ax_price = fig.add_subplot(gs[0, 0])
            ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_price)
            ax_profile = fig.add_subplot(gs[0, 1], sharey=ax_price)
            
            axes = [ax_price, ax_volume, ax_profile]
        else:
            # 简单的上下布局
            fig, axes = plt.subplots(
                2, 1, 
                figsize=(self.width, self.height),
                sharex=True,
                gridspec_kw={'height_ratios': [3, 1]}
            )
            ax_price, ax_volume = axes
        
        self.figure = fig
        self.axes = axes
        
        # 绘制K线图
        self._draw_candlesticks_matplotlib(ax_price)
        ax_price.set_title(self.title)
        ax_price.set_ylabel('价格')
        ax_price.grid(True, alpha=0.3)
        
        # 绘制成交量柱状图
        self._draw_volume_bars(ax_volume)
        
        # 添加成交量移动平均线
        colors = ColorPalette.TECHNICAL_COLORS
        for i, period in enumerate(self.volume_ma_periods):
            ma_name = f'Vol_MA_{period}'
            if ma_name in self.data.columns:
                ax_volume.plot(
                    self.data['date'],
                    self.data[ma_name],
                    label=f'Vol MA{period}',
                    color=colors[i % len(colors)],
                    linewidth=1
                )
        
        ax_volume.set_ylabel('成交量')
        ax_volume.legend()
        ax_volume.grid(True, alpha=0.3)
        
        # 添加成交量分布图
        if self.volume_profile_enabled:
            self._draw_volume_profile_matplotlib(ax_profile)
        
        # 格式化日期轴
        self._format_date_axis(ax_volume)
        plt.tight_layout()
    
    def _draw_volume_bars(self, ax):
        """绘制成交量柱状图"""
        colors = ['red' if close >= open else 'green' 
                 for close, open in zip(self.data['close'], self.data['open'])]
        
        bars = ax.bar(
            self.data['date'], 
            self.data['volume'], 
            color=colors, 
            alpha=0.7,
            width=0.8
        )
        
        return bars
    
    def _draw_volume_profile_matplotlib(self, ax):
        """绘制成交量分布图"""
        from .utils import ChartUtils
        
        # 计算成交量分布
        volume_profile = ChartUtils.calculate_volume_profile(self.data, bins=50)
        
        if len(volume_profile['volumes']) > 0:
            # 横向柱状图
            ax.barh(
                volume_profile['prices'],
                volume_profile['volumes'],
                height=(volume_profile['prices'][1] - volume_profile['prices'][0]) * 0.8,
                color='lightblue',
                alpha=0.7
            )
            
            ax.set_xlabel('成交量分布')
            ax.grid(True, alpha=0.3)
    
    def _draw_candlesticks_matplotlib(self, ax):
        """绘制K线"""
        from matplotlib.patches import Rectangle
        
        width = 0.6
        
        for i, (_, row) in enumerate(self.data.iterrows()):
            date = row['date']
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # 确定颜色
            color = 'red' if close_price >= open_price else 'green'
            
            # 绘制影线
            ax.plot([date, date], [low_price, high_price], color='black', linewidth=0.5)
            
            # 绘制实体
            if close_price >= open_price:  # 阳线
                rect = Rectangle(
                    (date - pd.Timedelta(hours=width*12), open_price),
                    pd.Timedelta(hours=width*24), 
                    close_price - open_price,
                    facecolor=color, 
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )
            else:  # 阴线
                rect = Rectangle(
                    (date - pd.Timedelta(hours=width*12), close_price),
                    pd.Timedelta(hours=width*24),
                    open_price - close_price,
                    facecolor=color,
                    edgecolor='black', 
                    linewidth=0.5,
                    alpha=0.8
                )
            
            ax.add_patch(rect)
    
    def add_volume_oscillator(self, period: int = 14) -> 'VolumeChart':
        """添加成交量震荡器"""
        volume = self.data['volume']
        volume_sma = volume.rolling(window=period).mean()
        volume_oscillator = ((volume - volume_sma) / volume_sma * 100)
        
        self.data['Volume_Oscillator'] = volume_oscillator
        return self
    
    def add_price_volume_trend(self) -> 'VolumeChart':
        """添加价量趋势指标"""
        close_prices = self.data['close']
        volume = self.data['volume']
        
        price_change = close_prices.pct_change()
        pvt = (price_change * volume).cumsum()
        
        self.data['PVT'] = pvt
        return self
    
    def get_volume_statistics(self) -> Dict[str, Any]:
        """获取成交量统计信息"""
        volume = self.data['volume']
        
        return {
            'average_volume': volume.mean(),
            'max_volume': volume.max(),
            'min_volume': volume.min(),
            'volume_std': volume.std(),
            'volume_median': volume.median(),
            'high_volume_days': len(volume[volume > volume.quantile(0.8)]),
            'low_volume_days': len(volume[volume < volume.quantile(0.2)])
        }