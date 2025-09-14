# 投资组合管理模块 (Portfolio Module)

专业的投资组合管理系统，提供仓位管理、风险控制、资金分配和绩效评估功能。

## 🚀 快速开始

### 基本投资组合管理

```python
from quantlib.portfolio import create_portfolio_manager
from quantlib.market_data import get_stock_data

# 创建投资组合管理器
portfolio = create_portfolio_manager(
    initial_cash=100000,    # 初始资金
    name="我的投资组合"     # 组合名称
)

print(portfolio)  # Portfolio '我的投资组合': Value=$100,000.00, Positions=0, Cash=100.0%

# 买入股票
success = portfolio.buy_stock('000001', 1000, 10.50)  # 买入1000股，价格10.50
if success:
    print("买入成功")

# 查看持仓
positions = portfolio.get_positions_summary()
print("当前持仓:")
print(positions)

# 更新价格
current_prices = {'000001': 11.20}
portfolio.update_prices(current_prices)

# 查看绩效
performance = portfolio.get_performance_metrics()
print(f"总收益: {performance['total_return']:.2f}")
print(f"收益率: {performance['total_return_pct']:.2f}%")
```

## 📋 核心组件

### 1. PortfolioManager - 投资组合管理器

核心的投资组合管理类，提供完整的资产管理功能。

```python
from quantlib.portfolio.manager import PortfolioManager

# 创建投资组合管理器
portfolio = PortfolioManager(
    initial_cash=100000,           # 初始资金
    name="策略组合",               # 组合名称
    commission_rate=0.001,         # 手续费率
    min_commission=5.0             # 最小手续费
)

# 基本信息
print(f"组合名称: {portfolio.name}")
print(f"总资产: ${portfolio.total_value:,.2f}")
print(f"现金: ${portfolio.cash:,.2f}")
print(f"持仓数量: {len(portfolio.positions)}")
```

### 2. PortfolioPosition - 持仓管理

管理单个股票的持仓信息。

```python
from quantlib.portfolio.manager import PortfolioPosition

# 创建持仓（通常由PortfolioManager自动管理）
position = PortfolioPosition(
    symbol='000001',
    quantity=1000,        # 持仓数量
    avg_cost=10.50,      # 平均成本
    current_price=11.20,  # 当前价格
    weight=0.10          # 权重
)

print(f"市值: ${position.market_value:,.2f}")
print(f"成本: ${position.cost_basis:,.2f}")
print(f"未实现收益率: {position.unrealized_return:.2f}%")
```

## 🛠️ 交易操作

### 1. 买入操作

```python
# 基本买入
success = portfolio.buy_stock(
    symbol='000001',      # 股票代码
    quantity=1000,        # 数量
    price=10.50          # 价格
)

if success:
    print("买入成功")
    print(f"剩余现金: ${portfolio.cash:,.2f}")
else:
    print("买入失败：资金不足")

# 按金额买入
def buy_by_amount(portfolio, symbol, amount, price):
    """按指定金额买入股票"""
    quantity = int(amount // price)  # 计算可买股数
    if quantity > 0:
        return portfolio.buy_stock(symbol, quantity, price)
    return False

# 使用示例
success = buy_by_amount(portfolio, '000001', 10000, 10.50)  # 买入1万元的股票
```

### 2. 卖出操作

```python
# 基本卖出
success = portfolio.sell_stock(
    symbol='000001',      # 股票代码
    quantity=500,         # 数量
    price=11.20          # 价格
)

if success:
    print("卖出成功")
    print(f"当前现金: ${portfolio.cash:,.2f}")

# 全部卖出
def sell_all(portfolio, symbol, price):
    """卖出某股票的全部持仓"""
    if symbol in portfolio.positions:
        quantity = portfolio.positions[symbol].quantity
        return portfolio.sell_stock(symbol, quantity, price)
    return False

# 按比例卖出
def sell_percentage(portfolio, symbol, percentage, price):
    """按比例卖出持仓"""
    if symbol in portfolio.positions:
        total_quantity = portfolio.positions[symbol].quantity
        sell_quantity = int(total_quantity * percentage)
        if sell_quantity > 0:
            return portfolio.sell_stock(symbol, sell_quantity, price)
    return False

# 使用示例
success = sell_percentage(portfolio, '000001', 0.3, 11.20)  # 卖出30%的持仓
```

### 3. 批量交易

```python
def batch_trade(portfolio, trades):
    """批量执行交易"""
    results = []

    for trade in trades:
        if trade['action'] == 'buy':
            success = portfolio.buy_stock(
                trade['symbol'],
                trade['quantity'],
                trade['price']
            )
        elif trade['action'] == 'sell':
            success = portfolio.sell_stock(
                trade['symbol'],
                trade['quantity'],
                trade['price']
            )

        results.append({
            'symbol': trade['symbol'],
            'action': trade['action'],
            'success': success
        })

    return results

# 使用示例
trades = [
    {'action': 'buy', 'symbol': '000001', 'quantity': 1000, 'price': 10.50},
    {'action': 'buy', 'symbol': '000002', 'quantity': 500, 'price': 20.30},
    {'action': 'sell', 'symbol': '000003', 'quantity': 300, 'price': 15.80}
]

results = batch_trade(portfolio, trades)
for result in results:
    print(f"{result['symbol']} {result['action']}: {'成功' if result['success'] else '失败'}")
```

## 📊 投资组合分析

### 1. 持仓分析

```python
# 获取持仓摘要
positions = portfolio.get_positions_summary()
print("持仓明细:")
print(positions)

# 按权重排序
if not positions.empty:
    top_holdings = positions.nlargest(5, 'weight')
    print("\n前5大持仓:")
    for _, position in top_holdings.iterrows():
        print(f"{position['symbol']}: 权重 {position['weight']:.1f}%, "
              f"市值 ${position['market_value']:,.0f}")

# 行业分布分析（需要股票行业数据）
def analyze_sector_allocation(portfolio, sector_mapping):
    """分析行业配置"""
    positions = portfolio.get_positions_summary()
    if positions.empty:
        return {}

    sector_allocation = {}
    total_value = portfolio.total_value

    for _, pos in positions.iterrows():
        sector = sector_mapping.get(pos['symbol'], '其他')
        if sector not in sector_allocation:
            sector_allocation[sector] = {'weight': 0, 'symbols': []}

        sector_allocation[sector]['weight'] += pos['market_value'] / total_value
        sector_allocation[sector]['symbols'].append(pos['symbol'])

    return sector_allocation

# 使用示例
sector_mapping = {
    '000001': '银行',
    '000002': '房地产',
    '000858': '白酒'
}

sectors = analyze_sector_allocation(portfolio, sector_mapping)
for sector, data in sectors.items():
    print(f"{sector}: {data['weight']:.1%} ({len(data['symbols'])}只股票)")
```

### 2. 绩效分析

```python
# 基本绩效指标
performance = portfolio.get_performance_metrics()

print("📈 投资组合绩效:")
print(f"总资产价值: ${performance['current_value']:,.2f}")
print(f"总收益: ${performance['total_return']:,.2f}")
print(f"总收益率: {performance['total_return_pct']:.2f}%")
print(f"年化收益率: {performance['annual_return_pct']:.2f}%")
print(f"波动率: {performance['volatility_pct']:.2f}%")
print(f"夏普比率: {performance['sharpe_ratio']:.3f}")
print(f"最大回撤: {performance['max_drawdown_pct']:.2f}%")
print(f"胜率: {performance['win_rate_pct']:.1f}%")
print(f"交易次数: {performance['total_trades']}")

# 详细绩效分析
def detailed_performance_analysis(portfolio):
    """详细绩效分析"""
    performance = portfolio.get_performance_metrics()

    # 计算各种比率
    sortino_ratio = performance['annual_return'] / performance.get('downside_deviation', performance['volatility'])
    calmar_ratio = performance['annual_return'] / abs(performance['max_drawdown']) if performance['max_drawdown'] != 0 else 0

    print("\n📊 风险调整后收益:")
    print(f"索提诺比率: {sortino_ratio:.3f}")
    print(f"卡尔马比率: {calmar_ratio:.3f}")

    # 收益分解
    positions = portfolio.get_positions_summary()
    if not positions.empty:
        print("\n💰 收益贡献分析:")
        for _, pos in positions.iterrows():
            contribution = (pos['market_value'] - pos['cost_basis'])
            contribution_pct = contribution / portfolio.initial_cash * 100
            print(f"{pos['symbol']}: 贡献 ${contribution:,.0f} ({contribution_pct:+.2f}%)")

detailed_performance_analysis(portfolio)
```

### 3. 风险分析

```python
def portfolio_risk_analysis(portfolio, price_history=None):
    """投资组合风险分析"""

    positions = portfolio.get_positions_summary()
    if positions.empty:
        print("暂无持仓，无法进行风险分析")
        return

    print("⚠️ 风险分析:")

    # 1. 集中度风险
    max_weight = positions['weight'].max()
    top3_weight = positions.nlargest(3, 'weight')['weight'].sum()

    print(f"最大单一持仓权重: {max_weight:.1f}%")
    print(f"前三大持仓合计权重: {top3_weight:.1f}%")

    if max_weight > 20:
        print("⚠️ 警告: 单一持仓权重过高，存在集中度风险")
    if top3_weight > 60:
        print("⚠️ 警告: 前三大持仓权重过高，缺乏分散化")

    # 2. 现金比例
    cash_ratio = portfolio.cash / portfolio.total_value
    print(f"现金比例: {cash_ratio:.1%}")

    if cash_ratio > 20:
        print("💡 提示: 现金比例较高，可考虑增加投资")
    elif cash_ratio < 5:
        print("⚠️ 警告: 现金比例过低，流动性不足")

    # 3. 持仓分散度
    position_count = len(positions)
    print(f"持仓股票数量: {position_count}")

    if position_count < 10:
        print("💡 提示: 持仓股票较少，可考虑增加分散化")
    elif position_count > 50:
        print("💡 提示: 持仓股票较多，可能存在过度分散化")

portfolio_risk_analysis(portfolio)
```

## 🔧 高级功能

### 1. 智能仓位管理

```python
class SmartPositionManager:
    """智能仓位管理器"""

    def __init__(self, portfolio, max_position_size=0.10, max_sector_weight=0.30):
        self.portfolio = portfolio
        self.max_position_size = max_position_size  # 单个股票最大权重
        self.max_sector_weight = max_sector_weight   # 单个行业最大权重

    def calculate_optimal_quantity(self, symbol, target_weight, price, sector_mapping=None):
        """计算最优买入数量"""
        target_value = self.portfolio.total_value * target_weight

        # 检查单个持仓限制
        if target_weight > self.max_position_size:
            target_weight = self.max_position_size
            target_value = self.portfolio.total_value * target_weight
            print(f"⚠️ {symbol} 目标权重超限，调整至 {target_weight:.1%}")

        # 检查行业权重限制
        if sector_mapping:
            sector = sector_mapping.get(symbol)
            if sector:
                current_sector_weight = self._get_sector_weight(sector, sector_mapping)
                if current_sector_weight + target_weight > self.max_sector_weight:
                    max_additional = self.max_sector_weight - current_sector_weight
                    if max_additional > 0:
                        target_weight = max_additional
                        target_value = self.portfolio.total_value * target_weight
                        print(f"⚠️ {sector}行业权重超限，{symbol} 调整至 {target_weight:.1%}")
                    else:
                        print(f"❌ {sector}行业权重已满，无法买入 {symbol}")
                        return 0

        # 计算数量
        quantity = int(target_value // price)
        actual_cost = quantity * price

        # 检查资金是否充足
        if actual_cost > self.portfolio.cash:
            quantity = int(self.portfolio.cash // price)
            print(f"⚠️ 资金不足，{symbol} 数量调整至 {quantity}")

        return quantity

    def _get_sector_weight(self, sector, sector_mapping):
        """计算当前行业权重"""
        positions = self.portfolio.get_positions_summary()
        if positions.empty:
            return 0

        sector_weight = 0
        for _, pos in positions.iterrows():
            if sector_mapping.get(pos['symbol']) == sector:
                sector_weight += pos['weight'] / 100  # 转换为小数

        return sector_weight

# 使用智能仓位管理
smart_manager = SmartPositionManager(portfolio)

# 计算最优买入数量
quantity = smart_manager.calculate_optimal_quantity(
    symbol='000001',
    target_weight=0.08,  # 目标权重8%
    price=10.50,
    sector_mapping={'000001': '银行'}
)

if quantity > 0:
    portfolio.buy_stock('000001', quantity, 10.50)
```

### 2. 自动再平衡

```python
def auto_rebalance(portfolio, target_weights, current_prices, tolerance=0.05):
    """自动再平衡投资组合"""

    print("开始自动再平衡...")

    # 更新价格
    portfolio.update_prices(current_prices)

    # 获取当前权重
    positions = portfolio.get_positions_summary()
    current_weights = {}

    if not positions.empty:
        for _, pos in positions.iterrows():
            current_weights[pos['symbol']] = pos['weight'] / 100

    # 计算需要调整的股票
    trades = []
    total_value = portfolio.total_value

    for symbol, target_weight in target_weights.items():
        current_weight = current_weights.get(symbol, 0)
        weight_diff = target_weight - current_weight

        # 如果偏差超过容忍度，则需要调整
        if abs(weight_diff) > tolerance:
            target_value = total_value * target_weight
            current_value = total_value * current_weight
            value_diff = target_value - current_value

            price = current_prices[symbol]

            if value_diff > 0:  # 需要买入
                quantity = int(value_diff // price)
                if quantity > 0 and quantity * price <= portfolio.cash:
                    trades.append({
                        'action': 'buy',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'reason': f'权重偏差 {weight_diff:+.1%}'
                    })
            else:  # 需要卖出
                if symbol in portfolio.positions:
                    quantity = min(
                        int(abs(value_diff) // price),
                        portfolio.positions[symbol].quantity
                    )
                    if quantity > 0:
                        trades.append({
                            'action': 'sell',
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': price,
                            'reason': f'权重偏差 {weight_diff:+.1%}'
                        })

    # 执行交易
    if trades:
        print(f"\n需要执行 {len(trades)} 笔调整交易:")
        for trade in trades:
            print(f"{trade['action'].upper()} {trade['symbol']} {trade['quantity']}股 "
                  f"@ {trade['price']:.2f} - {trade['reason']}")

        results = batch_trade(portfolio, trades)
        successful_trades = sum(1 for r in results if r['success'])
        print(f"\n再平衡完成: {successful_trades}/{len(trades)} 笔交易成功")
    else:
        print("投资组合权重在容忍范围内，无需调整")

# 使用示例
target_weights = {
    '000001': 0.20,  # 目标权重20%
    '000002': 0.15,  # 目标权重15%
    '000858': 0.10   # 目标权重10%
}

current_prices = {
    '000001': 10.50,
    '000002': 20.30,
    '000858': 180.50
}

auto_rebalance(portfolio, target_weights, current_prices)
```

### 3. 风险预算管理

```python
class RiskBudgetManager:
    """风险预算管理器"""

    def __init__(self, portfolio, total_risk_budget=0.15):
        self.portfolio = portfolio
        self.total_risk_budget = total_risk_budget  # 总风险预算（年化波动率）

    def allocate_risk_budget(self, symbols, risk_estimates, target_weights):
        """分配风险预算"""

        # 计算各股票的风险贡献
        risk_contributions = {}
        total_portfolio_risk = 0

        for symbol in symbols:
            weight = target_weights.get(symbol, 0)
            individual_risk = risk_estimates.get(symbol, 0.20)  # 默认20%波动率

            # 简化的风险贡献计算（实际应考虑相关性）
            risk_contribution = weight * individual_risk
            risk_contributions[symbol] = risk_contribution
            total_portfolio_risk += risk_contribution ** 2

        total_portfolio_risk = np.sqrt(total_portfolio_risk)

        print(f"预估组合风险: {total_portfolio_risk:.2%}")
        print(f"风险预算限制: {self.total_risk_budget:.2%}")

        # 检查是否超出风险预算
        if total_portfolio_risk > self.total_risk_budget:
            # 按比例缩放权重
            scale_factor = self.total_risk_budget / total_portfolio_risk
            adjusted_weights = {}

            print(f"\n⚠️ 超出风险预算，按比例调整权重 (缩放因子: {scale_factor:.3f})")

            for symbol, weight in target_weights.items():
                adjusted_weights[symbol] = weight * scale_factor
                print(f"{symbol}: {weight:.1%} -> {adjusted_weights[symbol]:.1%}")

            return adjusted_weights
        else:
            print("✓ 风险预算充足")
            return target_weights

    def monitor_risk_exposure(self):
        """监控风险敞口"""
        positions = self.portfolio.get_positions_summary()

        if positions.empty:
            print("无持仓，无风险敞口")
            return

        print("\n📊 风险敞口监控:")

        # 按权重排序显示风险敞口
        for _, pos in positions.nlargest(10, 'weight').iterrows():
            risk_exposure = pos['weight'] / 100 * 0.20  # 假设20%个股波动率
            print(f"{pos['symbol']}: 权重 {pos['weight']:.1f}% -> "
                  f"风险敞口 ~{risk_exposure:.2%}")

# 使用风险预算管理
risk_manager = RiskBudgetManager(portfolio, total_risk_budget=0.12)

# 风险估计（实际应基于历史数据计算）
risk_estimates = {
    '000001': 0.15,  # 银行股相对稳定
    '000002': 0.25,  # 房地产波动较大
    '000858': 0.30   # 白酒股波动很大
}

target_weights = {
    '000001': 0.30,
    '000002': 0.20,
    '000858': 0.15
}

adjusted_weights = risk_manager.allocate_risk_budget(
    list(target_weights.keys()),
    risk_estimates,
    target_weights
)

# 监控当前风险敞口
risk_manager.monitor_risk_exposure()
```

### 4. 动态对冲策略

```python
class DynamicHedging:
    """动态对冲管理"""

    def __init__(self, portfolio, hedge_ratio=0.5, hedge_symbols=None):
        self.portfolio = portfolio
        self.hedge_ratio = hedge_ratio  # 对冲比例
        self.hedge_symbols = hedge_symbols or ['000300']  # 默认用沪深300对冲

    def calculate_hedge_requirement(self, market_exposure):
        """计算对冲需求"""
        # 计算需要对冲的市场风险敞口
        hedge_value = market_exposure * self.hedge_ratio

        print(f"市场敞口: ${market_exposure:,.0f}")
        print(f"对冲比例: {self.hedge_ratio:.0%}")
        print(f"对冲需求: ${hedge_value:,.0f}")

        return hedge_value

    def execute_hedge(self, hedge_value, hedge_price):
        """执行对冲操作"""
        if not self.hedge_symbols:
            print("未设置对冲工具")
            return False

        hedge_symbol = self.hedge_symbols[0]  # 使用第一个对冲工具

        # 计算做空数量（这里简化处理，实际需要考虑期货、期权等工具）
        hedge_quantity = int(hedge_value // hedge_price)

        print(f"\n执行对冲:")
        print(f"对冲工具: {hedge_symbol}")
        print(f"对冲数量: {hedge_quantity}")
        print(f"对冲价值: ${hedge_quantity * hedge_price:,.0f}")

        # 这里应该是做空操作，但简化为记录对冲头寸
        return True

    def monitor_hedge_effectiveness(self, portfolio_returns, hedge_returns):
        """监控对冲效果"""
        if len(portfolio_returns) != len(hedge_returns):
            print("数据长度不匹配")
            return

        # 计算对冲组合收益
        hedged_returns = portfolio_returns - self.hedge_ratio * hedge_returns

        # 对冲效果分析
        original_vol = portfolio_returns.std() * np.sqrt(252)
        hedged_vol = hedged_returns.std() * np.sqrt(252)

        print(f"\n🛡️ 对冲效果分析:")
        print(f"原始组合波动率: {original_vol:.2%}")
        print(f"对冲后波动率: {hedged_vol:.2%}")
        print(f"风险降低: {(original_vol - hedged_vol):.2%}")
        print(f"对冲效果: {(1 - hedged_vol/original_vol):.1%}")

# 使用动态对冲
hedging_manager = DynamicHedging(portfolio, hedge_ratio=0.6)

# 计算当前市场敞口
stock_positions = portfolio.get_positions_summary()
if not stock_positions.empty:
    total_stock_value = stock_positions['market_value'].sum()
    hedge_value = hedging_manager.calculate_hedge_requirement(total_stock_value)

    # 执行对冲（需要期货价格）
    # hedge_success = hedging_manager.execute_hedge(hedge_value, index_futures_price)
```

## 📈 绩效归因分析

### 1. 收益归因

```python
def performance_attribution(portfolio, benchmark_weights, period_returns):
    """绩效归因分析"""

    positions = portfolio.get_positions_summary()
    if positions.empty:
        return

    print("📊 绩效归因分析")
    print("="*50)

    # 计算总的超额收益
    total_return = sum(pos['weight']/100 * period_returns.get(pos['symbol'], 0)
                      for _, pos in positions.iterrows())
    benchmark_return = sum(benchmark_weights.get(symbol, 0) * return_rate
                          for symbol, return_rate in period_returns.items())

    excess_return = total_return - benchmark_return

    print(f"组合收益: {total_return:.2%}")
    print(f"基准收益: {benchmark_return:.2%}")
    print(f"超额收益: {excess_return:.2%}")
    print()

    # 分解超额收益来源
    asset_allocation_effect = 0  # 资产配置效应
    stock_selection_effect = 0  # 个股选择效应

    for _, pos in positions.iterrows():
        symbol = pos['symbol']
        portfolio_weight = pos['weight'] / 100
        benchmark_weight = benchmark_weights.get(symbol, 0)
        stock_return = period_returns.get(symbol, 0)

        # 资产配置效应 = (组合权重 - 基准权重) × 基准收益
        allocation_contrib = (portfolio_weight - benchmark_weight) * benchmark_return
        asset_allocation_effect += allocation_contrib

        # 个股选择效应 = 基准权重 × (个股收益 - 基准收益)
        selection_contrib = benchmark_weight * (stock_return - benchmark_return)
        stock_selection_effect += selection_contrib

        print(f"{symbol}:")
        print(f"  权重差异: {(portfolio_weight - benchmark_weight):+.1%}")
        print(f"  配置贡献: {allocation_contrib:+.2%}")
        print(f"  选择贡献: {selection_contrib:+.2%}")
        print()

    print("归因总结:")
    print(f"资产配置效应: {asset_allocation_effect:+.2%}")
    print(f"个股选择效应: {stock_selection_effect:+.2%}")
    print(f"交互效应: {excess_return - asset_allocation_effect - stock_selection_effect:+.2%}")

# 使用示例
benchmark_weights = {
    '000001': 0.15,
    '000002': 0.08,
    '000858': 0.05
}

period_returns = {
    '000001': 0.05,
    '000002': -0.02,
    '000858': 0.12
}

performance_attribution(portfolio, benchmark_weights, period_returns)
```

## ⚠️ 注意事项

### 1. 风险管理
- 合理分散投资，避免过度集中
- 定期评估和调整投资组合
- 设置止损和风险限额
- 考虑流动性风险

### 2. 交易成本
- 考虑手续费对收益的影响
- 避免过度频繁交易
- 优化交易时机和方式
- 考虑市场冲击成本

### 3. 数据质量
- 确保价格数据的准确性
- 及时更新持仓信息
- 处理除权除息等公司行为
- 定期核对账户余额

### 4. 合规要求
- 遵守相关法律法规
- 注意持仓比例限制
- 及时披露重要信息
- 保持适当的内控制度

## 📖 API 参考

### PortfolioManager 核心方法

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|-------|
| `__init__()` | 初始化组合管理器 | initial_cash, name等 | - |
| `buy_stock()` | 买入股票 | symbol, quantity, price | bool |
| `sell_stock()` | 卖出股票 | symbol, quantity, price | bool |
| `update_prices()` | 更新股票价格 | price_dict | - |
| `get_positions_summary()` | 获取持仓摘要 | - | DataFrame |
| `get_performance_metrics()` | 获取绩效指标 | - | dict |
| `rebalance()` | 重新平衡组合 | target_weights | - |

### 便捷函数

| 函数 | 说明 | 参数 | 返回值 |
|------|------|------|-------|
| `create_portfolio_manager()` | 创建组合管理器 | initial_cash, name | PortfolioManager |
| `calculate_equal_weights()` | 计算等权重配置 | symbols | dict |

## 🤝 贡献

欢迎提交问题和改进建议！请遵循项目的代码风格和测试要求。

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。