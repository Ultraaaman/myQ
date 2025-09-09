"""
市场概览图表模块
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from .base import BaseChart
from .utils import ColorPalette, ChartUtils
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
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class MarketChart(BaseChart):
    """市场概览图表类"""
    
    def __init__(self, data: Dict[str, pd.DataFrame], engine: str = 'auto'):
        """
        初始化市场概览图表
        
        Args:
            data: 多只股票的数据字典 {股票代码: DataFrame}
            engine: 绘图引擎
        """
        # 验证输入数据
        if not isinstance(data, dict) or not data:
            raise ValueError("数据必须是包含多只股票数据的字典")
        
        # 验证每个DataFrame的格式
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError(f"股票 {symbol} 的数据无效")
                
        self.stock_data = data
        self.symbols = list(data.keys())
        
        # 使用第一只股票的数据初始化基类
        first_stock_data = list(data.values())[0]
        super().__init__(first_stock_data, engine)
        
        self.title = "市场概览"
        self.width = 14
        self.height = 10
        
    def plot_correlation_matrix(self) -> 'MarketChart':
        """绘制股票相关性矩阵"""
        # 收集所有股票的收盘价
        price_data = {}
        for symbol, data in self.stock_data.items():
            price_data[symbol] = data.set_index('date')['close']
        
        price_df = pd.DataFrame(price_data)
        correlation_matrix = price_df.corr()
        
        if self.engine == 'plotly':
            self._plot_correlation_plotly(correlation_matrix)
        elif self.engine == 'matplotlib':
            self._plot_correlation_matplotlib(correlation_matrix)
            
        return self
    
    def plot_returns_comparison(self, period: str = '1M') -> 'MarketChart':
        """绘制收益率比较图"""
        returns_data = self._calculate_returns(period)
        
        if self.engine == 'plotly':
            self._plot_returns_plotly(returns_data, period)
        elif self.engine == 'matplotlib':
            self._plot_returns_matplotlib(returns_data, period)
            
        return self
    
    def plot_volatility_comparison(self, window: int = 20) -> 'MarketChart':
        """绘制波动率比较图"""
        volatility_data = self._calculate_volatility(window)
        
        if self.engine == 'plotly':
            self._plot_volatility_plotly(volatility_data, window)
        elif self.engine == 'matplotlib':
            self._plot_volatility_matplotlib(volatility_data, window)
            
        return self
    
    def plot_market_overview(self) -> 'MarketChart':
        """绘制市场概览综合图表"""
        if self.engine == 'plotly':
            self._plot_overview_plotly()
        elif self.engine == 'matplotlib':
            self._plot_overview_matplotlib()
            
        return self
    
    def _calculate_returns(self, period: str) -> Dict[str, float]:
        """计算各股票的收益率"""
        returns = {}
        
        for symbol, data in self.stock_data.items():
            if len(data) < 2:
                returns[symbol] = 0.0
                continue
                
            current_price = data['close'].iloc[-1]
            
            if period == '1D':
                start_price = data['close'].iloc[-2] if len(data) > 1 else current_price
            elif period == '1W':
                start_idx = max(0, len(data) - 7)
                start_price = data['close'].iloc[start_idx]
            elif period == '1M':
                start_idx = max(0, len(data) - 30)
                start_price = data['close'].iloc[start_idx]
            elif period == '3M':
                start_idx = max(0, len(data) - 90)
                start_price = data['close'].iloc[start_idx]
            elif period == '1Y':
                start_idx = max(0, len(data) - 252)
                start_price = data['close'].iloc[start_idx]
            else:
                start_price = data['close'].iloc[0]
            
            if start_price != 0:
                returns[symbol] = ((current_price - start_price) / start_price) * 100
            else:
                returns[symbol] = 0.0
                
        return returns
    
    def _calculate_volatility(self, window: int) -> Dict[str, float]:
        """计算各股票的波动率"""
        volatility = {}
        
        for symbol, data in self.stock_data.items():
            if len(data) < window:
                volatility[symbol] = 0.0
                continue
                
            # 计算日收益率
            daily_returns = data['close'].pct_change().dropna()
            
            if len(daily_returns) >= window:
                # 计算滚动标准差（年化波动率）
                vol = daily_returns.rolling(window=window).std().iloc[-1] * np.sqrt(252) * 100
                volatility[symbol] = vol if not np.isnan(vol) else 0.0
            else:
                volatility[symbol] = 0.0
                
        return volatility
    
    def _plot_correlation_plotly(self, correlation_matrix: pd.DataFrame):
        """使用plotly绘制相关性矩阵"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="股票相关性矩阵",
            width=self.width * 80,
            height=self.height * 80,
            xaxis_title="股票代码",
            yaxis_title="股票代码"
        )
        
        self.figure = fig
    
    def _plot_returns_plotly(self, returns_data: Dict[str, float], period: str):
        """使用plotly绘制收益率比较"""
        symbols = list(returns_data.keys())
        returns = list(returns_data.values())
        colors = ['red' if r >= 0 else 'green' for r in returns]
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=returns,
                marker_color=colors,
                text=[f'{r:.2f}%' for r in returns],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f"股票收益率比较 ({period})",
            xaxis_title="股票代码",
            yaxis_title="收益率 (%)",
            width=self.width * 80,
            height=self.height * 60
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        self.figure = fig
    
    def _plot_volatility_plotly(self, volatility_data: Dict[str, float], window: int):
        """使用plotly绘制波动率比较"""
        symbols = list(volatility_data.keys())
        volatilities = list(volatility_data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=volatilities,
                marker_color='lightblue',
                text=[f'{v:.2f}%' for v in volatilities],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f"股票波动率比较 ({window}日滚动)",
            xaxis_title="股票代码",
            yaxis_title="年化波动率 (%)",
            width=self.width * 80,
            height=self.height * 60
        )
        
        self.figure = fig
    
    def _plot_overview_plotly(self):
        """使用plotly绘制市场概览"""
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['价格走势', '收益率分布', '波动率比较', '成交量对比'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ColorPalette.TECHNICAL_COLORS
        
        # 1. 价格走势
        for i, (symbol, data) in enumerate(self.stock_data.items()):
            # 标准化价格（以第一个价格为基准）
            normalized_prices = (data['close'] / data['close'].iloc[0]) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=normalized_prices,
                    mode='lines',
                    name=symbol,
                    line=dict(color=colors[i % len(colors)])
                ),
                row=1, col=1
            )
        
        # 2. 收益率分布
        returns_1m = self._calculate_returns('1M')
        symbols = list(returns_1m.keys())
        returns = list(returns_1m.values())
        colors_returns = ['red' if r >= 0 else 'green' for r in returns]
        
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=returns,
                marker_color=colors_returns,
                name="月收益率",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. 波动率比较
        volatility = self._calculate_volatility(20)
        vol_symbols = list(volatility.keys())
        vol_values = list(volatility.values())
        
        fig.add_trace(
            go.Bar(
                x=vol_symbols,
                y=vol_values,
                marker_color='lightblue',
                name="波动率",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. 成交量对比
        if all('volume' in data.columns for data in self.stock_data.values()):
            volume_avg = {}
            for symbol, data in self.stock_data.items():
                volume_avg[symbol] = data['volume'].tail(20).mean()  # 最近20日平均成交量
            
            vol_symbols = list(volume_avg.keys())
            vol_values = [ChartUtils.format_volume(v) for v in volume_avg.values()]
            vol_numeric = list(volume_avg.values())
            
            fig.add_trace(
                go.Bar(
                    x=vol_symbols,
                    y=vol_numeric,
                    text=vol_values,
                    textposition='outside',
                    marker_color='orange',
                    name="平均成交量",
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title=self.title,
            width=self.width * 80,
            height=self.height * 80,
            showlegend=True
        )
        
        # 更新子图标签
        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_yaxes(title_text="标准化价格", row=1, col=1)
        
        fig.update_xaxes(title_text="股票代码", row=1, col=2)
        fig.update_yaxes(title_text="收益率 (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="股票代码", row=2, col=1)
        fig.update_yaxes(title_text="波动率 (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="股票代码", row=2, col=2)
        fig.update_yaxes(title_text="成交量", row=2, col=2)
        
        self.figure = fig
    
    def _plot_correlation_matplotlib(self, correlation_matrix: pd.DataFrame):
        """使用matplotlib绘制相关性矩阵"""
        fig, ax = plt.subplots(figsize=(self.width * 0.8, self.height * 0.8))
        
        im = ax.imshow(correlation_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('相关系数')
        
        # 设置刻度标签
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax.set_yticklabels(correlation_matrix.index)
        
        # 添加数值标签
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('股票相关性矩阵')
        plt.tight_layout()
        self.figure = fig
    
    def _plot_returns_matplotlib(self, returns_data: Dict[str, float], period: str):
        """使用matplotlib绘制收益率比较"""
        symbols = list(returns_data.keys())
        returns = list(returns_data.values())
        colors = ['red' if r >= 0 else 'green' for r in returns]
        
        fig, ax = plt.subplots(figsize=(self.width * 0.8, self.height * 0.6))
        
        bars = ax.bar(symbols, returns, color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ret:.2f}%',
                   ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax.set_title(f'股票收益率比较 ({period})')
        ax.set_xlabel('股票代码')
        ax.set_ylabel('收益率 (%)')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.figure = fig
    
    def _plot_volatility_matplotlib(self, volatility_data: Dict[str, float], window: int):
        """使用matplotlib绘制波动率比较"""
        symbols = list(volatility_data.keys())
        volatilities = list(volatility_data.values())
        
        fig, ax = plt.subplots(figsize=(self.width * 0.8, self.height * 0.6))
        
        bars = ax.bar(symbols, volatilities, color='lightblue', alpha=0.7)
        
        # 添加数值标签
        for bar, vol in zip(bars, volatilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{vol:.2f}%',
                   ha='center', va='bottom')
        
        ax.set_title(f'股票波动率比较 ({window}日滚动)')
        ax.set_xlabel('股票代码')
        ax.set_ylabel('年化波动率 (%)')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.figure = fig
    
    def _plot_overview_matplotlib(self):
        """使用matplotlib绘制市场概览"""
        fig, axes = plt.subplots(2, 2, figsize=(self.width, self.height))
        
        colors = ColorPalette.TECHNICAL_COLORS
        
        # 1. 价格走势
        ax1 = axes[0, 0]
        for i, (symbol, data) in enumerate(self.stock_data.items()):
            # 标准化价格
            normalized_prices = (data['close'] / data['close'].iloc[0]) * 100
            ax1.plot(data['date'], normalized_prices, 
                    label=symbol, color=colors[i % len(colors)])
        
        ax1.set_title('价格走势 (标准化)')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('标准化价格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收益率分布
        ax2 = axes[0, 1]
        returns_1m = self._calculate_returns('1M')
        symbols = list(returns_1m.keys())
        returns = list(returns_1m.values())
        colors_returns = ['red' if r >= 0 else 'green' for r in returns]
        
        bars = ax2.bar(symbols, returns, color=colors_returns, alpha=0.7)
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax2.set_title('月收益率')
        ax2.set_xlabel('股票代码')
        ax2.set_ylabel('收益率 (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 波动率比较
        ax3 = axes[1, 0]
        volatility = self._calculate_volatility(20)
        vol_symbols = list(volatility.keys())
        vol_values = list(volatility.values())
        
        bars = ax3.bar(vol_symbols, vol_values, color='lightblue', alpha=0.7)
        for bar, vol in zip(bars, vol_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{vol:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax3.set_title('波动率 (20日)')
        ax3.set_xlabel('股票代码')
        ax3.set_ylabel('年化波动率 (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 成交量对比
        ax4 = axes[1, 1]
        if all('volume' in data.columns for data in self.stock_data.values()):
            volume_avg = {}
            for symbol, data in self.stock_data.items():
                volume_avg[symbol] = data['volume'].tail(20).mean()
            
            vol_symbols = list(volume_avg.keys())
            vol_values = list(volume_avg.values())
            
            bars = ax4.bar(vol_symbols, vol_values, color='orange', alpha=0.7)
            for bar, vol in zip(bars, vol_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        ChartUtils.format_volume(vol), 
                        ha='center', va='bottom', fontsize=8)
            
            ax4.set_title('平均成交量 (20日)')
            ax4.set_xlabel('股票代码')
            ax4.set_ylabel('成交量')
        else:
            ax4.text(0.5, 0.5, '无成交量数据', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('成交量对比')
        
        ax4.grid(True, alpha=0.3)
        
        # 调整所有子图的x轴标签
        for ax in axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.suptitle(self.title, fontsize=16)
        plt.tight_layout()
        self.figure = fig
    
    def get_market_summary(self) -> Dict[str, Any]:
        """获取市场摘要统计"""
        summary = {
            'symbols': self.symbols,
            'total_stocks': len(self.symbols),
            'returns_1d': self._calculate_returns('1D'),
            'returns_1m': self._calculate_returns('1M'),
            'volatility_20d': self._calculate_volatility(20),
            'best_performer': '',
            'worst_performer': '',
            'most_volatile': '',
            'least_volatile': ''
        }
        
        # 找出最佳和最差表现者
        returns_1m = summary['returns_1m']
        if returns_1m:
            best_symbol = max(returns_1m.keys(), key=lambda k: returns_1m[k])
            worst_symbol = min(returns_1m.keys(), key=lambda k: returns_1m[k])
            summary['best_performer'] = f"{best_symbol} (+{returns_1m[best_symbol]:.2f}%)"
            summary['worst_performer'] = f"{worst_symbol} ({returns_1m[worst_symbol]:.2f}%)"
        
        # 找出波动率最高和最低
        volatility = summary['volatility_20d']
        if volatility:
            most_vol_symbol = max(volatility.keys(), key=lambda k: volatility[k])
            least_vol_symbol = min(volatility.keys(), key=lambda k: volatility[k])
            summary['most_volatile'] = f"{most_vol_symbol} ({volatility[most_vol_symbol]:.2f}%)"
            summary['least_volatile'] = f"{least_vol_symbol} ({volatility[least_vol_symbol]:.2f}%)"
        
        return summary