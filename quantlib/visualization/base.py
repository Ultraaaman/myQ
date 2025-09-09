"""
基础图表类
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
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

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False


class BaseChart:
    """基础图表类"""
    
    def __init__(self, data: pd.DataFrame, engine: str = 'auto'):
        """
        初始化图表
        
        Args:
            data: 包含OHLCV数据的DataFrame
            engine: 绘图引擎 ('matplotlib', 'plotly', 'mplfinance', 'auto')
        """
        self.data = self._validate_data(data)
        self.engine = self._select_engine(engine)
        self.figure = None
        self.subplots = []
        
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证数据格式"""
        if data is None or data.empty:
            raise ValueError("数据不能为空")
            
        # 检查必需列
        required_columns = ['date', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"缺少必需列: {missing_columns}")
        
        # 确保数据类型正确
        data = data.copy()
        
        # 转换日期
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        # 转换数值列
        numeric_columns = ['open', 'high', 'low', 'close']
        if 'volume' in data.columns:
            numeric_columns.append('volume')
            
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 移除无效数据
        data = data.dropna(subset=required_columns)
        
        # 按日期排序
        data = data.sort_values('date').reset_index(drop=True)
        
        return data
    
    def _select_engine(self, engine: str) -> str:
        """选择绘图引擎"""
        if engine == 'auto':
            if MPLFINANCE_AVAILABLE:
                return 'mplfinance'
            elif PLOTLY_AVAILABLE:
                return 'plotly'
            elif MATPLOTLIB_AVAILABLE:
                return 'matplotlib'
            else:
                raise ImportError("需要安装绘图库: pip install matplotlib plotly mplfinance")
        
        engine_mapping = {
            'matplotlib': MATPLOTLIB_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE,
            'mplfinance': MPLFINANCE_AVAILABLE
        }
        
        if engine not in engine_mapping:
            raise ValueError(f"不支持的绘图引擎: {engine}")
            
        if not engine_mapping[engine]:
            raise ImportError(f"未安装 {engine} 库")
            
        return engine
    
    def set_title(self, title: str):
        """设置图表标题"""
        self.title = title
        return self
    
    def set_size(self, width: int = 12, height: int = 8):
        """设置图表尺寸"""
        self.width = width
        self.height = height
        return self
    
    def add_subplot(self, height_ratio: float = 1.0):
        """添加子图"""
        self.subplots.append(height_ratio)
        return len(self.subplots)
    
    def _prepare_data_for_plotting(self) -> pd.DataFrame:
        """为绘图准备数据"""
        plot_data = self.data.copy()
        
        # 设置日期为索引（某些库需要）
        if 'date' in plot_data.columns:
            plot_data = plot_data.set_index('date')
            
        # 重命名列以符合标准格式
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in plot_data.columns:
                plot_data = plot_data.rename(columns={old_col: new_col})
                
        return plot_data
    
    def _get_price_range(self) -> Tuple[float, float]:
        """获取价格范围"""
        price_data = self.data[['high', 'low', 'open', 'close']]
        min_price = price_data.min().min()
        max_price = price_data.max().max()
        
        # 添加5%的边距
        padding = (max_price - min_price) * 0.05
        return min_price - padding, max_price + padding
    
    def _format_date_axis(self, ax):
        """格式化日期轴"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # 根据数据范围选择日期格式
        data_range = (self.data['date'].max() - self.data['date'].min()).days
        
        if data_range <= 30:  # 一个月内
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        elif data_range <= 365:  # 一年内
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:  # 超过一年
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def save(self, filename: str, dpi: int = 300):
        """保存图表"""
        if self.figure is None:
            raise ValueError("请先创建图表")
            
        if self.engine == 'matplotlib' or self.engine == 'mplfinance':
            self.figure.savefig(filename, dpi=dpi, bbox_inches='tight')
        elif self.engine == 'plotly':
            if filename.endswith('.html'):
                self.figure.write_html(filename)
            else:
                self.figure.write_image(filename, width=self.width*100, height=self.height*100)
                
        print(f"图表已保存到: {filename}")
    
    def show(self):
        """显示图表"""
        if self.figure is None:
            raise ValueError("请先创建图表")
            
        if self.engine == 'matplotlib' or self.engine == 'mplfinance':
            plt.show()
        elif self.engine == 'plotly':
            self.figure.show()
    
    def _validate_indicator_data(self, indicator_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """验证技术指标数据"""
        validated_data = {}
        
        for name, series in indicator_data.items():
            if not isinstance(series, pd.Series):
                continue
                
            # 确保索引对齐
            if len(series) == len(self.data):
                validated_data[name] = series
            else:
                print(f"警告: 指标 {name} 的数据长度与价格数据不匹配")
                
        return validated_data