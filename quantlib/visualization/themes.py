"""
图表主题和样式配置
"""
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class ChartTheme:
    """图表主题基类"""
    
    def __init__(self):
        self.name = "default"
        self.colors = self._default_colors()
        self.style = self._default_style()
    
    def _default_colors(self) -> Dict[str, str]:
        """默认颜色配置"""
        return {
            'up': '#ff4444',        # 上涨红色
            'down': '#00aa00',      # 下跌绿色
            'volume_up': '#ff6666', # 上涨成交量
            'volume_down': '#66aa66', # 下跌成交量
            'ma_5': '#ff6600',      # MA5橙色
            'ma_10': '#0066ff',     # MA10蓝色
            'ma_20': '#9900ff',     # MA20紫色
            'ma_50': '#ff9900',     # MA50黄色
            'ma_200': '#666666',    # MA200灰色
            'grid': '#e0e0e0',      # 网格线
            'text': '#333333',      # 文字颜色
            'background': '#ffffff', # 背景色
            'border': '#cccccc'     # 边框色
        }
    
    def _default_style(self) -> Dict[str, Any]:
        """默认样式配置"""
        return {
            'figure_size': (12, 8),
            'dpi': 100,
            'grid': True,
            'grid_alpha': 0.3,
            'line_width': 1.0,
            'candle_width': 0.6,
            'volume_alpha': 0.7,
            'font_size': 10,
            'title_font_size': 14,
            'legend_font_size': 9
        }
    
    def get_ma_color(self, period: int) -> str:
        """获取移动平均线颜色"""
        color_map = {
            5: self.colors['ma_5'],
            10: self.colors['ma_10'],
            20: self.colors['ma_20'],
            50: self.colors['ma_50'],
            200: self.colors['ma_200']
        }
        return color_map.get(period, '#000000')
    
    def get_matplotlib_style(self) -> Dict[str, Any]:
        """获取matplotlib样式配置"""
        return {
            'figure.figsize': self.style['figure_size'],
            'figure.dpi': self.style['dpi'],
            'axes.grid': self.style['grid'],
            'grid.alpha': self.style['grid_alpha'],
            'lines.linewidth': self.style['line_width'],
            'font.size': self.style['font_size'],
            'axes.titlesize': self.style['title_font_size'],
            'legend.fontsize': self.style['legend_font_size'],
            'axes.facecolor': self.colors['background'],
            'figure.facecolor': self.colors['background'],
            'text.color': self.colors['text'],
            'axes.edgecolor': self.colors['border'],
            'grid.color': self.colors['grid']
        }
    
    def get_plotly_layout(self) -> Dict[str, Any]:
        """获取plotly布局配置"""
        return {
            'template': 'plotly_white' if self.colors['background'] == '#ffffff' else 'plotly_dark',
            'font': {'color': self.colors['text'], 'size': self.style['font_size']},
            'plot_bgcolor': self.colors['background'],
            'paper_bgcolor': self.colors['background'],
            'xaxis': {
                'gridcolor': self.colors['grid'],
                'gridwidth': 1,
                'showgrid': self.style['grid']
            },
            'yaxis': {
                'gridcolor': self.colors['grid'],
                'gridwidth': 1,
                'showgrid': self.style['grid']
            }
        }


class LightTheme(ChartTheme):
    """亮色主题"""
    
    def __init__(self):
        super().__init__()
        self.name = "light"
        self.colors = self._light_colors()
    
    def _light_colors(self) -> Dict[str, str]:
        """亮色主题配色"""
        return {
            'up': '#e74c3c',        # 上涨红色
            'down': '#27ae60',      # 下跌绿色
            'volume_up': '#f39c12', # 上涨成交量
            'volume_down': '#2ecc71', # 下跌成交量
            'ma_5': '#f39c12',      # MA5橙色
            'ma_10': '#3498db',     # MA10蓝色
            'ma_20': '#9b59b6',     # MA20紫色
            'ma_50': '#e67e22',     # MA50橙色
            'ma_200': '#95a5a6',    # MA200灰色
            'grid': '#ecf0f1',      # 网格线
            'text': '#2c3e50',      # 文字颜色
            'background': '#ffffff', # 背景色
            'border': '#bdc3c7'     # 边框色
        }


class DarkTheme(ChartTheme):
    """暗色主题"""
    
    def __init__(self):
        super().__init__()
        self.name = "dark"
        self.colors = self._dark_colors()
    
    def _dark_colors(self) -> Dict[str, str]:
        """暗色主题配色"""
        return {
            'up': '#ff5252',        # 上涨红色
            'down': '#4caf50',      # 下跌绿色
            'volume_up': '#ff9800', # 上涨成交量
            'volume_down': '#66bb6a', # 下跌成交量
            'ma_5': '#ff9800',      # MA5橙色
            'ma_10': '#2196f3',     # MA10蓝色
            'ma_20': '#9c27b0',     # MA20紫色
            'ma_50': '#ff5722',     # MA50橙色
            'ma_200': '#757575',    # MA200灰色
            'grid': '#424242',      # 网格线
            'text': '#e0e0e0',      # 文字颜色
            'background': '#1e1e1e', # 背景色
            'border': '#616161'     # 边框色
        }


class MinimalTheme(ChartTheme):
    """简约主题"""
    
    def __init__(self):
        super().__init__()
        self.name = "minimal"
        self.colors = self._minimal_colors()
        self.style = self._minimal_style()
    
    def _minimal_colors(self) -> Dict[str, str]:
        """简约主题配色"""
        return {
            'up': '#000000',        # 上涨黑色
            'down': '#666666',      # 下跌灰色
            'volume_up': '#cccccc', # 上涨成交量
            'volume_down': '#999999', # 下跌成交量
            'ma_5': '#000000',      # MA5黑色
            'ma_10': '#333333',     # MA10深灰
            'ma_20': '#666666',     # MA20中灰
            'ma_50': '#999999',     # MA50浅灰
            'ma_200': '#cccccc',    # MA200很浅灰
            'grid': '#f5f5f5',      # 网格线
            'text': '#333333',      # 文字颜色
            'background': '#ffffff', # 背景色
            'border': '#dddddd'     # 边框色
        }
    
    def _minimal_style(self) -> Dict[str, Any]:
        """简约样式配置"""
        style = super()._default_style()
        style.update({
            'grid_alpha': 0.2,
            'line_width': 0.8,
            'volume_alpha': 0.5
        })
        return style


class ColorBlindTheme(ChartTheme):
    """色盲友好主题"""
    
    def __init__(self):
        super().__init__()
        self.name = "colorblind"
        self.colors = self._colorblind_colors()
    
    def _colorblind_colors(self) -> Dict[str, str]:
        """色盲友好配色"""
        return {
            'up': '#0173b2',        # 上涨蓝色
            'down': '#cc78bc',      # 下跌粉色
            'volume_up': '#56b4e9', # 上涨成交量
            'volume_down': '#de8f05', # 下跌成交量
            'ma_5': '#e69f00',      # MA5橙色
            'ma_10': '#0173b2',     # MA10蓝色
            'ma_20': '#cc78bc',     # MA20粉色
            'ma_50': '#029e73',     # MA50绿色
            'ma_200': '#949494',    # MA200灰色
            'grid': '#e0e0e0',      # 网格线
            'text': '#333333',      # 文字颜色
            'background': '#ffffff', # 背景色
            'border': '#cccccc'     # 边框色
        }


# 主题注册表
THEMES = {
    'default': ChartTheme,
    'light': LightTheme,
    'dark': DarkTheme,
    'minimal': MinimalTheme,
    'colorblind': ColorBlindTheme
}


def get_theme(theme_name: str = 'default') -> ChartTheme:
    """获取指定主题"""
    if theme_name not in THEMES:
        print(f"未知主题 '{theme_name}'，使用默认主题")
        theme_name = 'default'
    
    return THEMES[theme_name]()


def list_themes() -> List[str]:
    """列出所有可用主题"""
    return list(THEMES.keys())