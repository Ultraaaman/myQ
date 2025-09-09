"""
K线图绘制模块
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from .base import BaseChart
import warnings
warnings.filterwarnings('ignore')

# 导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False


class CandlestickChart(BaseChart):
    """K线图表类"""
    
    def __init__(self, data: pd.DataFrame, engine: str = 'auto'):
        super().__init__(data, engine)
        self.ma_periods = []
        self.indicators = {}
        self.volume_enabled = False
        self.title = "股票K线图"
        self.width = 12
        self.height = 8
        
    def add_ma(self, periods: List[int] = [5, 10, 20]) -> 'CandlestickChart':
        """添加移动平均线"""
        self.ma_periods = periods
        
        # 计算移动平均线
        for period in periods:
            ma_name = f'MA_{period}'
            self.data[ma_name] = self.data['close'].rolling(window=period).mean()
            
        return self
    
    def add_indicator(self, indicator_data: Dict[str, pd.Series]) -> 'CandlestickChart':
        """添加技术指标"""
        validated_data = self._validate_indicator_data(indicator_data)
        self.indicators.update(validated_data)
        return self
    
    def add_volume(self, enabled: bool = True) -> 'CandlestickChart':
        """添加成交量"""
        if 'volume' not in self.data.columns:
            print("警告: 数据中没有成交量信息")
            return self
            
        self.volume_enabled = enabled
        return self
    
    def plot(self) -> 'CandlestickChart':
        """绘制K线图"""
        if self.engine == 'mplfinance':
            self._plot_mplfinance()
        elif self.engine == 'plotly':
            self._plot_plotly()
        elif self.engine == 'matplotlib':
            self._plot_matplotlib()
        else:
            raise ValueError(f"不支持的绘图引擎: {self.engine}")
            
        return self
    
    def _plot_mplfinance(self):
        """使用mplfinance绘制"""
        plot_data = self._prepare_data_for_plotting()
        
        # 准备附加图形
        addplot_list = []
        
        # 添加移动平均线
        for period in self.ma_periods:
            ma_name = f'MA_{period}'
            if ma_name in self.data.columns:
                addplot_list.append(
                    mpf.make_addplot(
                        self.data[ma_name], 
                        panel=0,
                        secondary_y=False
                    )
                )
        
        # 添加技术指标
        for name, series in self.indicators.items():
            addplot_list.append(
                mpf.make_addplot(
                    series,
                    panel=1,
                    secondary_y=False,
                    ylabel=name
                )
            )
        
        # 设置样式
        style = mpf.make_marketcolors(
            up='red',      # 上涨为红色（中国习惯）
            down='green',  # 下跌为绿色
            edge='inherit',
            wick={'up': 'red', 'down': 'green'},
            volume='in'
        )
        
        mpf_style = mpf.make_mpf_style(
            marketcolors=style,
            gridstyle='-',
            y_on_right=True
        )
        
        # 绘制图表
        self.figure, self.axes = mpf.plot(
            plot_data,
            type='candle',
            style=mpf_style,
            addplot=addplot_list if addplot_list else None,
            volume=self.volume_enabled,
            title=self.title,
            figsize=(self.width, self.height),
            returnfig=True
        )
    
    def _plot_plotly(self):
        """使用plotly绘制"""
        # 创建子图
        subplot_specs = [[{"secondary_y": True}]]
        subplot_titles = [self.title]
        
        # 如果有技术指标，增加子图
        if self.indicators:
            subplot_specs.append([{"secondary_y": False}])
            subplot_titles.append("技术指标")
            
        # 如果有成交量，增加子图
        if self.volume_enabled and 'volume' in self.data.columns:
            subplot_specs.append([{"secondary_y": False}])
            subplot_titles.append("成交量")
        
        self.figure = make_subplots(
            rows=len(subplot_specs),
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            specs=subplot_specs,
            vertical_spacing=0.1,
            row_width=[0.6, 0.2, 0.2] if len(subplot_specs) == 3 else [0.7, 0.3]
        )
        
        # 绘制K线
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
        
        # 添加移动平均线
        colors = ['blue', 'orange', 'purple', 'brown', 'pink']
        for i, period in enumerate(self.ma_periods):
            ma_name = f'MA_{period}'
            if ma_name in self.data.columns:
                ma_trace = go.Scatter(
                    x=self.data['date'],
                    y=self.data[ma_name],
                    mode='lines',
                    name=f'MA{period}',
                    line=dict(color=colors[i % len(colors)], width=1)
                )
                self.figure.add_trace(ma_trace, row=1, col=1)
        
        # 添加基准对比线
        if hasattr(self, 'benchmark_data'):
            for benchmark_name, benchmark_info in self.benchmark_data.items():
                benchmark_trace = go.Scatter(
                    x=benchmark_info['data']['date'],
                    y=benchmark_info['data']['close_normalized'],
                    mode='lines',
                    name=benchmark_name,
                    line=dict(
                        color=benchmark_info['color'], 
                        width=1.5,
                        dash='dash'
                    ),
                    opacity=0.7
                )
                self.figure.add_trace(benchmark_trace, row=1, col=1)
        
        # 添加技术指标
        current_row = 2
        if self.indicators:
            for name, series in self.indicators.items():
                indicator_trace = go.Scatter(
                    x=self.data['date'],
                    y=series,
                    mode='lines',
                    name=name,
                    line=dict(width=1)
                )
                self.figure.add_trace(indicator_trace, row=current_row, col=1)
            current_row += 1
        
        # 添加成交量
        if self.volume_enabled and 'volume' in self.data.columns:
            # 成交量颜色与K线一致
            volume_colors = ['red' if close >= open else 'green' 
                           for close, open in zip(self.data['close'], self.data['open'])]
            
            volume_trace = go.Bar(
                x=self.data['date'],
                y=self.data['volume'],
                name='成交量',
                marker_color=volume_colors,
                opacity=0.7
            )
            self.figure.add_trace(volume_trace, row=current_row, col=1)
        
        # 更新布局
        self.figure.update_layout(
            title=self.title,
            xaxis_title='日期',
            yaxis_title='价格',
            width=self.width * 100,
            height=self.height * 100,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # 更新x轴格式
        self.figure.update_xaxes(type='date')
    
    def _plot_matplotlib(self):
        """使用matplotlib绘制"""
        # 计算子图数量
        subplot_count = 1
        if self.volume_enabled and 'volume' in self.data.columns:
            subplot_count += 1
        if self.indicators:
            subplot_count += 1
            
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
        
        # 绘制K线
        ax_main = axes[0]
        self._draw_candlesticks(ax_main)
        
        # 添加移动平均线
        colors = ['blue', 'orange', 'purple', 'brown', 'pink']
        for i, period in enumerate(self.ma_periods):
            ma_name = f'MA_{period}'
            if ma_name in self.data.columns:
                ax_main.plot(
                    self.data['date'], 
                    self.data[ma_name], 
                    label=f'MA{period}',
                    color=colors[i % len(colors)],
                    linewidth=1
                )
        
        # 添加基准对比线
        if hasattr(self, 'benchmark_data'):
            for benchmark_name, benchmark_info in self.benchmark_data.items():
                ax_main.plot(
                    benchmark_info['data']['date'],
                    benchmark_info['data']['close_normalized'],
                    label=benchmark_name,
                    color=benchmark_info['color'],
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.7
                )
        
        ax_main.set_title(self.title)
        ax_main.set_ylabel('价格')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        current_subplot = 1
        
        # 添加成交量
        if self.volume_enabled and 'volume' in self.data.columns and current_subplot < len(axes):
            ax_vol = axes[current_subplot]
            self._draw_volume(ax_vol)
            current_subplot += 1
        
        # 添加技术指标
        if self.indicators and current_subplot < len(axes):
            ax_ind = axes[current_subplot]
            for name, series in self.indicators.items():
                ax_ind.plot(self.data['date'], series, label=name, linewidth=1)
            ax_ind.set_ylabel('指标值')
            ax_ind.legend()
            ax_ind.grid(True, alpha=0.3)
        
        # 格式化日期轴
        self._format_date_axis(axes[-1])
        
        plt.tight_layout()
    
    def _draw_candlesticks(self, ax):
        """绘制K线蜡烛图"""
        width = 0.6
        width2 = 0.05
        
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
    
    def _draw_volume(self, ax):
        """绘制成交量"""
        colors = ['red' if close >= open else 'green' 
                 for close, open in zip(self.data['close'], self.data['open'])]
        
        bars = ax.bar(
            self.data['date'], 
            self.data['volume'], 
            color=colors, 
            alpha=0.7,
            width=0.8
        )
        
        ax.set_ylabel('成交量')
        ax.grid(True, alpha=0.3)
    
    def add_support_resistance(self, levels: List[float], 
                             colors: List[str] = None) -> 'CandlestickChart':
        """添加支撑阻力位"""
        if colors is None:
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            
        self.support_resistance = []
        for i, level in enumerate(levels):
            color = colors[i % len(colors)]
            self.support_resistance.append({'level': level, 'color': color})
            
        return self
    
    def add_trend_line(self, start_date: str, start_price: float,
                       end_date: str, end_price: float,
                       color: str = 'blue', style: str = '--') -> 'CandlestickChart':
        """添加趋势线"""
        if not hasattr(self, 'trend_lines'):
            self.trend_lines = []
            
        self.trend_lines.append({
            'start_date': pd.to_datetime(start_date),
            'start_price': start_price,
            'end_date': pd.to_datetime(end_date),
            'end_price': end_price,
            'color': color,
            'style': style
        })
        
        return self
    
    def add_benchmark(self, benchmark_data: pd.DataFrame, 
                     name: str = "基准", color: str = 'gray') -> 'CandlestickChart':
        """添加基准对比线（如大盘指数）"""
        if 'close' not in benchmark_data.columns or 'date' not in benchmark_data.columns:
            print(f"警告: {name}数据格式不正确，需要包含date和close列")
            return self
            
        # 标准化处理：将基准数据标准化到与股票数据相同的起始点
        if not self.data.empty and not benchmark_data.empty:
            # 找到时间范围的交集
            stock_start = self.data['date'].min()
            stock_end = self.data['date'].max()
            
            # 筛选基准数据到相同时间范围
            benchmark_filtered = benchmark_data[
                (benchmark_data['date'] >= stock_start) & 
                (benchmark_data['date'] <= stock_end)
            ].copy()
            
            if not benchmark_filtered.empty and not self.data.empty:
                # 标准化：以第一个交集日期的价格为基准
                stock_base = self.data['close'].iloc[0]
                benchmark_base = benchmark_filtered['close'].iloc[0]
                
                if benchmark_base != 0:
                    benchmark_normalized = (benchmark_filtered['close'] / benchmark_base) * stock_base
                    benchmark_filtered['close_normalized'] = benchmark_normalized
                    
                    # 存储基准数据
                    if not hasattr(self, 'benchmark_data'):
                        self.benchmark_data = {}
                    
                    self.benchmark_data[name] = {
                        'data': benchmark_filtered,
                        'color': color
                    }
                    
                    print(f"✓ 已添加{name}基准线")
                else:
                    print(f"警告: {name}基准数据无效")
            else:
                print(f"警告: {name}与股票数据时间范围无交集")
        
        return self
    
    def set_date_range(self, start_date: str, end_date: str) -> 'CandlestickChart':
        """设置日期范围"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
        self.data = self.data.loc[mask].reset_index(drop=True)
        
        return self