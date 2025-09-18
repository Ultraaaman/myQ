# QuantLib Research Module

量化研究模块 - 提供完整的因子研究、分析和回测功能

## 🚀 模块概述

QuantLib Research Module 是一个专业的量化投资研究工具包，集成了因子库管理、因子分析、策略回测和报告生成功能。该模块为量化研究人员提供了从因子开发到策略验证的完整工作流程。

## 📦 核心组件

### 1. 因子库管理 (`factor_library.py`)

**主要功能:**
- 内置丰富的技术因子和基本面因子
- 支持自定义因子开发
- 因子计算缓存机制
- 因子分类管理系统

**核心类:**
- `FactorLibrary`: 因子库管理器
- `BaseFactor`: 因子基类
- `FactorCalculator`: 因子计算引擎
- `FactorCategory`: 因子分类枚举

**预置因子类别:**
```python
# 技术因子
- MomentumFactor (动量因子)
- RSIFactor (RSI因子)
- VolatilityFactor (波动率因子)

# 基本面因子
- PEFactor (市盈率因子)
- PBFactor (市净率因子)
- ROEFactor (净资产收益率因子)
```

**使用示例:**
```python
from quantlib.research import create_factor_library

# 创建因子库
factor_lib = create_factor_library()

# 添加自定义因子
def my_factor(data, period=20, **kwargs):
    return data['close'].rolling(period).mean() / data['close'] - 1

factor_lib.create_custom_factor(
    name="price_ma_ratio",
    calc_func=my_factor,
    description="价格与均线比率",
    category=FactorCategory.TECHNICAL
)

# 计算因子值
factor_values = factor_lib.calculate_factors(['momentum_20d', 'rsi_14d'], stock_data)
```

### 2. 因子分析器 (`factor_analyzer.py`)

**主要功能:**
- 信息系数(IC)分析
- 因子有效性评估
- 分位数收益分析
- 多空组合表现分析
- 因子相关性分析

**核心类:**
- `FactorAnalyzer`: 因子分析器
- `ICAnalysis`: IC分析结果
- `FactorPerformance`: 因子表现结果

**关键分析指标:**
- IC均值和标准差
- IC信息比率(IR)
- 换手率
- 因子自相关性
- 多空收益和夏普比

**使用示例:**
```python
from quantlib.research import create_factor_analyzer

analyzer = create_factor_analyzer()

# 综合因子分析
result = analyzer.comprehensive_factor_analysis(
    factor_data=factor_series,
    returns=return_series,
    factor_name="momentum_20d"
)

print(f"IC均值: {result.ic_analysis.ic_mean:.4f}")
print(f"IC信息比率: {result.ic_analysis.ic_ir:.4f}")
```

### 3. 研究框架 (`research_framework.py`)

**主要功能:**
- 整合因子库和分析器
- 批量因子研究
- 因子策略回测
- 综合研究报告

**核心类:**
- `ResearchFramework`: 研究框架主类
- `FactorBacktester`: 因子回测器
- `BacktestConfig`: 回测配置
- `BacktestResult`: 回测结果

**回测配置示例:**
```python
from quantlib.research import ResearchFramework, BacktestConfig
from datetime import datetime

config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=1000000,
    commission=0.001,
    long_pct=0.2,
    short_pct=0.2,
    rebalance_freq='M'
)

# 创建研究框架
framework = ResearchFramework()

# 综合因子研究
results = framework.comprehensive_factor_study(
    data=stock_data,
    price_data=price_data,
    returns=returns,
    config=config
)
```

### 4. 报告生成器 (`report_generator.py`)

**主要功能:**
- HTML/Markdown 报告生成
- 可视化图表创建
- 因子分析报告
- 回测结果报告

**核心类:**
- `ReportGenerator`: 报告生成器

**使用示例:**
```python
from quantlib.research import create_research_report

# 生成分析报告
report_path = create_research_report(
    analysis_results=analysis_results,
    title="因子分析报告",
    format="html",
    output_path="reports"
)
```

## 🔧 安装和快速开始

### 1. 基础设置
```python
from quantlib.research import (
    create_factor_library,
    create_factor_analyzer,
    create_research_framework
)

# 初始化组件
factor_lib = create_factor_library()
analyzer = create_factor_analyzer()
framework = create_research_framework()
```

### 2. 数据准备
```python
import pandas as pd

# 准备股票数据
stock_data = pd.DataFrame({
    'close': [...],      # 收盘价
    'open': [...],       # 开盘价
    'high': [...],       # 最高价
    'low': [...],        # 最低价
    'volume': [...],     # 成交量
    'pe_ratio': [...],   # 市盈率
    'pb_ratio': [...],   # 市净率
    'roe': [...]         # ROE
})

# 收益率数据
returns = stock_data['close'].pct_change()
```

### 3. 完整工作流程
```python
from datetime import datetime

# 1. 进行因子研究
analysis_results = framework.conduct_factor_research(
    data=stock_data,
    returns=returns
)

# 2. 配置回测参数
config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=1000000
)

# 3. 执行回测
backtest_results = framework.conduct_factor_backtest(
    factor_data=factor_data,
    price_data=price_data,
    config=config
)

# 4. 生成报告
from quantlib.research import create_research_report

report_path = create_research_report(
    analysis_results=analysis_results,
    backtest_results=backtest_results,
    title="综合研究报告"
)
```

## 📊 输出结果

### 因子分析结果
- **IC统计**: IC均值、标准差、信息比率
- **稳定性**: IC胜率、绝对IC均值
- **交易特性**: 换手率、自相关性
- **收益特性**: 多空收益、夏普比率

### 回测结果
- **收益指标**: 总收益、年化收益、波动率
- **风险指标**: 最大回撤、Calmar比率
- **交易指标**: 胜率、盈亏比
- **组合轨迹**: 净值曲线、持仓记录

### 可视化图表
- IC分布图和时序图
- 因子表现对比图
- 净值曲线图
- 相关性热力图

## 🎯 应用场景

### 1. 因子挖掘
- 评估新因子的有效性
- 对比不同因子的表现
- 识别最优因子组合

### 2. 策略开发
- 因子选股策略构建
- 多因子模型开发
- 策略参数优化

### 3. 风险管理
- 因子暴露分析
- 相关性风险评估
- 回撤控制研究

### 4. 投研支持
- 定期因子报告
- 策略表现监控
- 投资决策支持

## ⚙️ 高级功能

### 自定义因子开发
```python
class MyCustomFactor(BaseFactor):
    def __init__(self, name, period=20):
        super().__init__(name, FactorCategory.CUSTOM)
        self.period = period

    def calculate(self, data, **kwargs):
        # 实现自定义计算逻辑
        return data['close'].rolling(self.period).apply(
            lambda x: custom_calculation(x)
        )

# 注册到因子库
factor_lib.register_factor(MyCustomFactor("my_factor"))
```

### 批量因子测试
```python
# 测试多个参数组合
momentum_factors = []
for period in [5, 10, 20, 60]:
    factor = MomentumFactor(f"momentum_{period}d", period)
    momentum_factors.append(factor)

# 批量分析
results = {}
for factor in momentum_factors:
    factor_values = factor_lib.calculate_factor(factor.name, stock_data)
    results[factor.name] = analyzer.comprehensive_factor_analysis(
        factor_values, returns, factor.name
    )
```

### 因子组合优化
```python
# 相关性过滤
correlation_matrix = analyzer.factor_correlation_analysis(factor_data_dict)
low_corr_factors = select_low_correlation_factors(correlation_matrix, threshold=0.5)

# 构建多因子模型
multi_factor_score = combine_factors(low_corr_factors, weights)
```

## 🛠️ 配置和自定义

### 存储配置
```python
# 自定义存储路径
factor_lib = create_factor_library(storage_path="custom/factor/path")
framework = create_research_framework(storage_path="custom/research/path")
```

### 计算参数
```python
# 分析器参数
analyzer = create_factor_analyzer(min_periods=30)

# 回测参数调整
config = BacktestConfig(
    commission=0.002,        # 手续费
    long_pct=0.3,           # 做多比例
    rebalance_freq='W',     # 周度调仓
    min_stocks=10           # 最小持股数
)
```

## 🔍 性能优化

### 缓存机制
```python
# 启用因子计算缓存
factor_lib.calculator.cache_enabled = True

# 清理缓存
factor_lib.calculator.clear_cache()

# 查看缓存信息
cache_info = factor_lib.calculator.get_cache_info()
```

### 并行计算
```python
# 批量因子计算自动并行
factor_data = factor_lib.calculate_factors(
    factor_names=['momentum_20d', 'rsi_14d', 'volatility_20d'],
    data=stock_data,
    use_cache=True
)
```

## 📈 最佳实践

### 1. 数据质量控制
- 确保数据的完整性和准确性
- 处理缺失值和异常值
- 统一数据频率和时间对齐

### 2. 因子有效性验证
- 使用足够长的历史数据
- 进行样本内外验证
- 考虑数据偏差和生存偏差

### 3. 风险管理
- 设置合理的止损机制
- 控制单一因子的权重
- 定期更新因子表现

### 4. 交易成本考虑
- 考虑手续费和冲击成本
- 优化调仓频率
- 评估流动性风险

## 📚 扩展阅读

- [因子投资理论基础](docs/factor_theory.md)
- [多因子模型构建指南](docs/multifactor_model.md)
- [风险控制最佳实践](docs/risk_management.md)
- [API参考文档](docs/api_reference.md)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个模块:

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源。

---

*QuantLib Research Module - 让量化研究更简单、更专业* 🚀