# Portfolio与Strategy模块集成指南

## 🎯 **模块架构重新设计**

根据您的正确建议，我们重新设计了模块架构：

### 📁 **Strategy模块** - 所有策略算法
```python
quantlib/strategy/
├── base.py                 # 策略基类
├── examples.py            # 技术分析策略
├── factor_strategies.py   # 因子策略 (新移入)
└── __init__.py
```

### 📁 **Portfolio模块** - 投资组合管理
```python  
quantlib/portfolio/
├── manager.py             # 组合管理器
├── strategy_executor.py   # 统一执行框架  
└── __init__.py
```

## 🚀 **快速开始**

### 1. 基础策略执行
```python
from quantlib.portfolio import create_strategy_executor, StrategyType
from quantlib.strategy import create_ma_cross_strategy

# 创建执行器
executor = create_strategy_executor(mode="live", initial_capital=100000)

# 创建策略
ma_strategy = create_ma_cross_strategy(['000001'], short_window=20, long_window=60)

# 添加策略  
executor.add_strategy("MA_Cross", ma_strategy, weight=1.0, strategy_type=StrategyType.TECHNICAL)
```

### 2. 因子投资策略
```python
from quantlib.strategy import create_factor_strategy, FactorType
from quantlib.portfolio import create_factor_executor

# 创建因子策略
factor_strategy = create_factor_strategy(
    factor_type=FactorType.VALUE,
    symbols=['000001', '000002'],
    factor_data=value_factor_data
)

# 使用因子执行器
executor = create_factor_executor(initial_capital=200000, mode="live")
executor.add_strategy("Value_Factor", factor_strategy)
```

### 3. 回测集成
```python
# 切换到回测模式
backtest_executor = create_strategy_executor(mode="backtest", initial_capital=100000)

# 添加相同的策略
backtest_executor.add_strategy("MA_Cross", ma_strategy, weight=0.6)
backtest_executor.add_strategy("Factor", factor_strategy, weight=0.4)

# 运行回测
results = backtest_executor.run_backtest(historical_data)
```

## 🔧 **故障排除**

### 语法错误修复
如果遇到语法错误，请检查：

1. **导入路径更新**:
```python
# 正确的导入方式
from quantlib.strategy import create_factor_strategy, FactorType  # ✅
from quantlib.portfolio import create_strategy_executor         # ✅

# 错误的旧导入方式  
from quantlib.portfolio.factor_strategies import ...  # ❌ 已移动
```

2. **函数名称更新**:
```python
# 因子策略现在在strategy模块中
from quantlib.strategy import (
    create_factor_strategy,           # 单因子策略
    create_factor_multi_strategy      # 多因子策略
)
```

### 测试导入
创建测试文件验证导入：
```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from quantlib.portfolio import create_strategy_executor
    from quantlib.strategy import create_factor_strategy, FactorType
    print("✅ 导入成功!")
except Exception as e:
    print(f"❌ 导入失败: {e}")
```

## 📊 **架构优势**

### 重新设计后的优势

1. **职责单一**: 
   - Strategy模块: 专注信号生成算法
   - Portfolio模块: 专注组合管理

2. **逻辑一致**:
   - 所有策略类型都在Strategy模块
   - 统一的BaseStrategy接口

3. **扩展性强**:
   - 新增策略类型只需扩展Strategy模块
   - Portfolio可管理任意类型的策略

### 使用场景对比

| 场景 | Strategy模块职责 | Portfolio模块职责 |
|------|------------------|------------------|
| 技术分析 | 均线交叉算法 | 执行技术信号 |
| 因子投资 | 因子排序算法 | 管理因子组合 |
| 多策略 | 各策略算法 | 信号聚合+风控 |
| 回测 | 生成历史信号 | 模拟交易执行 |

## 🎉 **完整工作流程**

```python
# 1. 策略开发 (Strategy模块)
ma_strategy = create_ma_cross_strategy(['000001'], 20, 60)
factor_strategy = create_factor_strategy(FactorType.VALUE, ['000001'], factor_data)

# 2. 组合管理 (Portfolio模块)  
executor = create_strategy_executor("backtest", 100000)
executor.add_strategy("Technical", ma_strategy, 0.6)
executor.add_strategy("Factor", factor_strategy, 0.4)

# 3. 回测验证
results = executor.run_backtest(historical_data)

# 4. 实盘部署
if results['total_return'] > 0.1:
    live_executor = create_strategy_executor("live", 100000)
    # 部署相同策略配置...
```

## 📝 **升级指南**

如果您之前使用了旧版本，请按以下步骤升级：

1. **更新导入语句**:
```python
# 旧版本 
from quantlib.portfolio.factor_strategies import SingleFactorStrategy

# 新版本
from quantlib.strategy.factor_strategies import SingleFactorStrategy  
```

2. **更新创建函数**:
```python
# 多因子策略创建
from quantlib.strategy import create_factor_multi_strategy

strategy = create_factor_multi_strategy(symbols, factor_data, weights)
```

3. **验证功能**:
```python
# 运行简单测试确保一切正常
executor = create_strategy_executor("live", 10000)
print("✅ 升级成功!")
```

感谢您指出架构问题，重新设计让模块职责更加清晰！