# 计算所有因子
if stock_data is not None:
    print(f"🧮 正在计算所有因子...")

    try:
        # 确保数据格式正确
        print(f"   数据预处理...")

        # 检查必需的列是否存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in stock_data.columns]
        missing_columns = [col for col in required_columns if col not in stock_data.columns]

        print(f"   可用列: {available_columns}")
        if missing_columns:
            print(f"   缺失列: {missing_columns}")

        # 创建结果DataFrame，从原始数据开始
        factor_results = stock_data.copy()

        print(f"   开始因子计算...")

        # === 1. 动量因子 ===
        print(f"   计算动量因子...")

        # 5日动量
        factor_results['factor_momentum_5d'] = stock_data['close'].pct_change(5)

        # 20日动量
        factor_results['factor_momentum_20d'] = stock_data['close'].pct_change(20)

        # 60日动量
        factor_results['factor_momentum_60d'] = stock_data['close'].pct_change(60)

        # === 2. 反转因子 ===
        print(f"   计算反转因子...")

        # 1日反转（负动量）
        factor_results['factor_reversal_1d'] = -stock_data['close'].pct_change(1)

        # 5日反转
        factor_results['factor_reversal_5d'] = -stock_data['close'].pct_change(5)

        # === 3. 波动率因子 ===
        print(f"   计算波动率因子...")

        # 计算日收益率
        returns = stock_data['close'].pct_change()

        # 20日波动率
        factor_results['factor_volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)

        # 60日波动率
        factor_results['factor_volatility_60d'] = returns.rolling(60).std() * np.sqrt(252)

        # === 4. RSI因子 ===
        print(f"   计算RSI因子...")

        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)

        # 14日RSI
        factor_results['factor_rsi_14d'] = calculate_rsi(stock_data['close'], 14)

        # 30日RSI
        factor_results['factor_rsi_30d'] = calculate_rsi(stock_data['close'], 30)

        # === 5. 移动平均因子 ===
        print(f"   计算移动平均因子...")

        # 价格相对于20日均线的位置
        ma20 = stock_data['close'].rolling(20).mean()
        factor_results['factor_ma_ratio_20d'] = stock_data['close'] / ma20 - 1

        # 价格相对于60日均线的位置
        ma60 = stock_data['close'].rolling(60).mean()
        factor_results['factor_ma_ratio_60d'] = stock_data['close'] / ma60 - 1

        # === 6. 成交量因子 ===
        print(f"   计算成交量因子...")

        # 成交量相对比率
        vol_ma20 = stock_data['volume'].rolling(20).mean()
        factor_results['factor_volume_ratio_20d'] = stock_data['volume'] / vol_ma20 - 1

        # 价量关系
        factor_results['factor_price_volume_corr_20d'] = returns.rolling(20).corr(stock_data['volume'].pct_change())

        # === 7. 价格位置因子 ===
        print(f"   计算价格位置因子...")

        # 20日内价格位置（威廉指标思想）
        high_20 = stock_data['high'].rolling(20).max()
        low_20 = stock_data['low'].rolling(20).min()
        factor_results['factor_price_position_20d'] = (stock_data['close'] - low_20) / (high_20 - low_20)

        # === 8. 振幅因子 ===
        print(f"   计算振幅因子...")

        # 日振幅
        factor_results['factor_amplitude_1d'] = (stock_data['high'] - stock_data['low']) / stock_data['close']

        # 20日平均振幅
        factor_results['factor_amplitude_20d'] = factor_results['factor_amplitude_1d'].rolling(20).mean()

        # === 9. 跳空因子 ===
        print(f"   计算跳空因子...")

        # 向上跳空
        factor_results['factor_gap_up'] = (stock_data['open'] / stock_data['close'].shift(1) - 1).clip(lower=0)

        # 向下跳空
        factor_results['factor_gap_down'] = (stock_data['open'] / stock_data['close'].shift(1) - 1).clip(upper=0)

        # === 10. 综合技术指标 ===
        print(f"   计算综合指标...")

        # 简化版MACD
        ema12 = stock_data['close'].ewm(span=12).mean()
        ema26 = stock_data['close'].ewm(span=26).mean()
        factor_results['factor_macd'] = ema12 - ema26

        print(f"✅ 因子计算完成!")
        print(f"   结果形状: {factor_results.shape}")

        # 统计因子列
        factor_columns = [col for col in factor_results.columns if col.startswith('factor_')]
        price_columns = [col for col in factor_results.columns if col in required_columns]

        print(f"   价格列数: {len(price_columns)} ({price_columns})")
        print(f"   因子列数: {len(factor_columns)}")

        # 显示因子列名
        if factor_columns:
            print(f"\n📊 计算的因子 ({len(factor_columns)}个):")
            factor_categories = {
                'momentum': [f for f in factor_columns if 'momentum' in f],
                'reversal': [f for f in factor_columns if 'reversal' in f],
                'volatility': [f for f in factor_columns if 'volatility' in f],
                'rsi': [f for f in factor_columns if 'rsi' in f],
                'ma_ratio': [f for f in factor_columns if 'ma_ratio' in f],
                'volume': [f for f in factor_columns if 'volume' in f or 'price_volume' in f],
                'position': [f for f in factor_columns if 'position' in f],
                'amplitude': [f for f in factor_columns if 'amplitude' in f],
                'gap': [f for f in factor_columns if 'gap' in f],
                'technical': [f for f in factor_columns if 'macd' in f]
            }

            for category, factors in factor_categories.items():
                if factors:
                    print(f"  {category.upper()} ({len(factors)}个): {factors}")

            # 显示基础统计信息
            print(f"\n📈 因子统计信息:")
            display(factor_results[factor_columns].describe())
        else:
            print(f"⚠️ 没有计算出因子，可能是数据格式不匹配")

    except Exception as e:
        print(f"❌ 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        factor_results = stock_data  # 至少保留原始数据

else:
    print("⚠️ 无法进行因子计算：缺少数据")
    factor_results = None