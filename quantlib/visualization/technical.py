"""
技术指标图表模块
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


class TechnicalChart(BaseChart):
    """技术指标图表类"""
    
    def __init__(self, data: pd.DataFrame, engine: str = 'auto'):
        super().__init__(data, engine)
        self.indicators = {}
        self.subplot_indicators = {}  # 独立子图的指标
        self.title = "技术指标分析"
        self.width = 12
        self.height = 10
        
    def add_rsi(self, period: int = 14, subplot: bool = True) -> 'TechnicalChart':
        """添加RSI指标"""
        rsi_data = self._calculate_rsi(period)
        
        if subplot:
            self.subplot_indicators['RSI'] = rsi_data
        else:
            self.indicators['RSI'] = rsi_data
            
        return self
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9, 
                 subplot: bool = True) -> 'TechnicalChart':
        """添加MACD指标"""
        macd_data = self._calculate_macd(fast, slow, signal)
        
        if subplot:
            self.subplot_indicators['MACD'] = macd_data
        else:
            self.indicators.update(macd_data)
            
        return self
    
    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> 'TechnicalChart':
        """添加布林带指标"""
        bb_data = self._calculate_bollinger_bands(period, std_dev)
        self.indicators.update(bb_data)
        return self
    
    def add_kdj(self, k_period: int = 9, d_period: int = 3, 
                subplot: bool = True) -> 'TechnicalChart':
        """添加KDJ指标"""
        kdj_data = self._calculate_kdj(k_period, d_period)
        
        if subplot:
            self.subplot_indicators['KDJ'] = kdj_data
        else:
            self.indicators.update(kdj_data)
            
        return self
    
    def add_volume_indicators(self, subplot: bool = True) -> 'TechnicalChart':
        """添加成交量指标"""
        if 'volume' not in self.data.columns:
            print("警告: 数据中没有成交量信息")
            return self
            
        volume_data = self._calculate_volume_indicators()
        
        if subplot:
            self.subplot_indicators['Volume'] = volume_data
        else:
            self.indicators.update(volume_data)
            
        return self
    
    def plot(self) -> 'TechnicalChart':
        """绘制技术指标图表"""
        if self.engine == 'plotly':
            self._plot_plotly()
        elif self.engine == 'matplotlib':
            self._plot_matplotlib()
        else:
            raise ValueError(f"技术指标图表不支持引擎: {self.engine}")
            
        return self
    
    def _calculate_rsi(self, period: int) -> Dict[str, pd.Series]:
        """计算RSI指标"""
        close_prices = self.data['close']
        delta = close_prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {'RSI': rsi}
    
    def _calculate_macd(self, fast: int, slow: int, signal: int) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        close_prices = self.data['close']
        
        exp1 = close_prices.ewm(span=fast).mean()
        exp2 = close_prices.ewm(span=slow).mean()
        
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'MACD_Histogram': macd_histogram
        }
    
    def _calculate_bollinger_bands(self, period: int, std_dev: float) -> Dict[str, pd.Series]:
        """计算布林带指标"""
        close_prices = self.data['close']
        
        sma = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'BB_Upper': upper_band,
            'BB_Middle': sma,
            'BB_Lower': lower_band
        }
    
    def _calculate_kdj(self, k_period: int, d_period: int) -> Dict[str, pd.Series]:
        """计算KDJ指标"""
        high_prices = self.data['high']
        low_prices = self.data['low']
        close_prices = self.data['close']
        
        lowest_low = low_prices.rolling(window=k_period).min()
        highest_high = high_prices.rolling(window=k_period).max()
        
        rsv = (close_prices - lowest_low) / (highest_high - lowest_low) * 100
        
        k_values = rsv.ewm(com=2).mean()
        d_values = k_values.ewm(com=2).mean()
        j_values = 3 * k_values - 2 * d_values
        
        return {
            'K': k_values,
            'D': d_values,
            'J': j_values
        }
    
    def _calculate_volume_indicators(self) -> Dict[str, pd.Series]:
        """计算成交量指标"""
        volume = self.data['volume']
        close_prices = self.data['close']
        
        # OBV (On Balance Volume)
        obv = (volume * np.where(close_prices.diff() > 0, 1, 
               np.where(close_prices.diff() < 0, -1, 0))).cumsum()
        
        # 成交量移动平均
        volume_ma = volume.rolling(window=20).mean()
        
        return {
            'OBV': obv,
            'Volume_MA': volume_ma
        }
    
    def _plot_plotly(self):
        """使用plotly绘制技术指标图表"""
        # 计算子图数量
        subplot_count = 1 + len(self.subplot_indicators)
        
        # 创建子图
        subplot_titles = [self.title] + list(self.subplot_indicators.keys())
        
        self.figure = make_subplots(
            rows=subplot_count,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            vertical_spacing=0.05,
            row_heights=[0.6] + [0.4 / (subplot_count - 1)] * (subplot_count - 1) if subplot_count > 1 else [1.0]
        )
        
        # 绘制主价格图表
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
        
        # 添加主图指标
        colors = ColorPalette.TECHNICAL_COLORS
        color_index = 0
        
        for name, series in self.indicators.items():
            self.figure.add_trace(
                go.Scatter(
                    x=self.data['date'],
                    y=series,
                    mode='lines',
                    name=name,
                    line=dict(color=colors[color_index % len(colors)]),
                    yaxis='y1'
                ),
                row=1, col=1
            )
            color_index += 1
        
        # 添加子图指标
        current_row = 2
        for indicator_name, indicator_data in self.subplot_indicators.items():
            if indicator_name == 'RSI':
                self._add_rsi_subplot(current_row, indicator_data)
            elif indicator_name == 'MACD':
                self._add_macd_subplot(current_row, indicator_data)
            elif indicator_name == 'KDJ':
                self._add_kdj_subplot(current_row, indicator_data)
            elif indicator_name == 'Volume':
                self._add_volume_subplot(current_row, indicator_data)
            
            current_row += 1
        
        # 更新布局
        self.figure.update_layout(
            title=self.title,
            width=self.width * 100,
            height=self.height * 100,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
    
    def _add_rsi_subplot(self, row: int, rsi_data: Dict[str, pd.Series]):
        """添加RSI子图"""
        rsi = rsi_data['RSI']
        
        # RSI线
        self.figure.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='blue')
            ),
            row=row, col=1
        )
        
        # 添加70和30水平线
        self.figure.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
        self.figure.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
        self.figure.add_hline(y=50, line_dash="dot", line_color="gray", row=row, col=1)
    
    def _add_macd_subplot(self, row: int, macd_data: Dict[str, pd.Series]):
        """添加MACD子图"""
        # MACD线
        self.figure.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=macd_data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ),
            row=row, col=1
        )
        
        # Signal线
        self.figure.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=macd_data['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red')
            ),
            row=row, col=1
        )
        
        # 柱状图
        colors = ['red' if h > 0 else 'green' for h in macd_data['MACD_Histogram']]
        self.figure.add_trace(
            go.Bar(
                x=self.data['date'],
                y=macd_data['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.7
            ),
            row=row, col=1
        )
    
    def _add_kdj_subplot(self, row: int, kdj_data: Dict[str, pd.Series]):
        """添加KDJ子图"""
        self.figure.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=kdj_data['K'],
                mode='lines',
                name='K',
                line=dict(color='blue')
            ),
            row=row, col=1
        )
        
        self.figure.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=kdj_data['D'],
                mode='lines',
                name='D',
                line=dict(color='red')
            ),
            row=row, col=1
        )
        
        self.figure.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=kdj_data['J'],
                mode='lines',
                name='J',
                line=dict(color='green')
            ),
            row=row, col=1
        )
        
        # 添加80和20水平线
        self.figure.add_hline(y=80, line_dash="dash", line_color="red", row=row, col=1)
        self.figure.add_hline(y=20, line_dash="dash", line_color="green", row=row, col=1)
    
    def _add_volume_subplot(self, row: int, volume_data: Dict[str, pd.Series]):
        """添加成交量子图"""
        # 成交量柱状图
        colors = ['red' if close >= open else 'green' 
                 for close, open in zip(self.data['close'], self.data['open'])]
        
        self.figure.add_trace(
            go.Bar(
                x=self.data['date'],
                y=self.data['volume'],
                name='成交量',
                marker_color=colors,
                opacity=0.7
            ),
            row=row, col=1
        )
        
        # OBV线
        if 'OBV' in volume_data:
            self.figure.add_trace(
                go.Scatter(
                    x=self.data['date'],
                    y=volume_data['OBV'],
                    mode='lines',
                    name='OBV',
                    line=dict(color='purple'),
                    yaxis='y2'
                ),
                row=row, col=1, secondary_y=True
            )
    
    def _plot_matplotlib(self):
        """使用matplotlib绘制技术指标图表"""
        # 计算子图数量
        subplot_count = 1 + len(self.subplot_indicators)
        
        # 创建子图
        fig, axes = plt.subplots(
            subplot_count, 1, 
            figsize=(self.width, self.height),
            sharex=True,
            gridspec_kw={'height_ratios': [3] + [1] * (subplot_count - 1)}
        )
        
        if subplot_count == 1:
            axes = [axes]
            
        self.figure = fig
        self.axes = axes
        
        # 绘制主价格图
        ax_main = axes[0]
        self._draw_candlesticks_matplotlib(ax_main)
        
        # 添加主图指标
        colors = ColorPalette.TECHNICAL_COLORS
        color_index = 0
        
        for name, series in self.indicators.items():
            ax_main.plot(
                self.data['date'],
                series,
                label=name,
                color=colors[color_index % len(colors)],
                linewidth=1
            )
            color_index += 1
        
        ax_main.set_title(self.title)
        ax_main.set_ylabel('价格')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # 添加子图指标
        current_subplot = 1
        for indicator_name, indicator_data in self.subplot_indicators.items():
            if current_subplot < len(axes):
                ax = axes[current_subplot]
                
                if indicator_name == 'RSI':
                    self._plot_rsi_matplotlib(ax, indicator_data)
                elif indicator_name == 'MACD':
                    self._plot_macd_matplotlib(ax, indicator_data)
                elif indicator_name == 'KDJ':
                    self._plot_kdj_matplotlib(ax, indicator_data)
                elif indicator_name == 'Volume':
                    self._plot_volume_matplotlib(ax, indicator_data)
                
                current_subplot += 1
        
        # 格式化日期轴
        self._format_date_axis(axes[-1])
        plt.tight_layout()
    
    def _plot_rsi_matplotlib(self, ax, rsi_data):
        """matplotlib绘制RSI"""
        rsi = rsi_data['RSI']
        ax.plot(self.data['date'], rsi, 'b-', label='RSI', linewidth=1)
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买')
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel('RSI')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_macd_matplotlib(self, ax, macd_data):
        """matplotlib绘制MACD"""
        ax.plot(self.data['date'], macd_data['MACD'], 'b-', label='MACD', linewidth=1)
        ax.plot(self.data['date'], macd_data['MACD_Signal'], 'r-', label='Signal', linewidth=1)
        
        # 柱状图
        colors = ['red' if h > 0 else 'green' for h in macd_data['MACD_Histogram']]
        ax.bar(self.data['date'], macd_data['MACD_Histogram'], 
               color=colors, alpha=0.7, label='Histogram')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('MACD')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_kdj_matplotlib(self, ax, kdj_data):
        """matplotlib绘制KDJ"""
        ax.plot(self.data['date'], kdj_data['K'], 'b-', label='K', linewidth=1)
        ax.plot(self.data['date'], kdj_data['D'], 'r-', label='D', linewidth=1)
        ax.plot(self.data['date'], kdj_data['J'], 'g-', label='J', linewidth=1)
        
        ax.axhline(y=80, color='r', linestyle='--', alpha=0.7)
        ax.axhline(y=20, color='g', linestyle='--', alpha=0.7)
        ax.set_ylabel('KDJ')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_volume_matplotlib(self, ax, volume_data):
        """matplotlib绘制成交量"""
        colors = ['red' if close >= open else 'green' 
                 for close, open in zip(self.data['close'], self.data['open'])]
        
        ax.bar(self.data['date'], self.data['volume'], 
               color=colors, alpha=0.7, label='成交量')
        
        if 'OBV' in volume_data:
            ax2 = ax.twinx()
            ax2.plot(self.data['date'], volume_data['OBV'], 
                    color='purple', label='OBV', linewidth=1)
            ax2.set_ylabel('OBV', color='purple')
            ax2.legend(loc='upper right')
        
        ax.set_ylabel('成交量')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _draw_candlesticks_matplotlib(self, ax):
        """matplotlib绘制K线"""
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