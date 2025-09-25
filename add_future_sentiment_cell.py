# 🔮 添加未来收益率和情绪因子
# 扩展因子数据，加入未来收益率目标变量和情绪因子

# 添加未来收益率和情绪因子
if factor_results is not None:
    print(f"🔮 正在计算未来收益率...")

    try:
        # === 计算未来收益率 ===
        print(f"   计算不同跨度的未来收益率...")

        # 1日未来收益率
        factor_results['future_return_1d'] = factor_results['close'].shift(-1) / factor_results['close'] - 1

        # 3日未来收益率
        factor_results['future_return_3d'] = factor_results['close'].shift(-3) / factor_results['close'] - 1

        # 5日未来收益率
        factor_results['future_return_5d'] = factor_results['close'].shift(-5) / factor_results['close'] - 1

        # 10日未来收益率
        factor_results['future_return_10d'] = factor_results['close'].shift(-10) / factor_results['close'] - 1

        # 20日未来收益率
        factor_results['future_return_20d'] = factor_results['close'].shift(-20) / factor_results['close'] - 1

        # 开盘表现（次日开盘相对当日收盘）
        factor_results['open_performance_1d'] = factor_results['open'].shift(-1) / factor_results['close'] - 1

        print(f"✅ 未来收益率计算完成")

        # === 加载和处理情绪因子 ===
        print(f"\n📰 正在加载情绪因子数据...")

        import os
        sentiment_file = "D:/projects/q/myQ/scripts/news_scores_result_1y_zijin.csv"

        if os.path.exists(sentiment_file):
            # 读取情绪数据
            sentiment_df = pd.read_csv(sentiment_file, encoding='utf-8-sig')
            print(f"✓ 加载情绪数据: {len(sentiment_df)} 条记录")

            # 转换日期格式
            sentiment_df['date'] = pd.to_datetime(sentiment_df['original_date']).dt.date
            factor_results['date'] = factor_results.index.date

            print(f"✓ 情绪数据日期范围: {sentiment_df['date'].min()} 到 {sentiment_df['date'].max()}")

            # 聚合日度情绪数据
            print(f"   聚合日度情绪数据...")
            daily_sentiment = sentiment_df.groupby('date').agg({
                'overall_score': ['mean', 'std', 'count', 'min', 'max', 'sum'],
                'direct_impact_score': ['mean', 'std'],
                'indirect_impact_score': ['mean', 'std'],
                'certainty': ['mean', 'std', 'min', 'max'],
                'sentiment': lambda x: x.mode()[0] if not x.empty else 'neutral'
            }).round(4)

            # 扁平化列名
            daily_sentiment.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0]
                                     for col in daily_sentiment.columns]
            daily_sentiment = daily_sentiment.reset_index()

            print(f"✓ 日度聚合: {len(daily_sentiment)} 天")

            # 计算情绪衍生因子
            print(f"   计算情绪衍生因子...")

            # 按日期排序
            daily_sentiment = daily_sentiment.sort_values('date').reset_index(drop=True)

            # 1. 基础情绪因子
            daily_sentiment['news_intensity'] = daily_sentiment['overall_score_count']  # 新闻强度
            daily_sentiment['sentiment_strength'] = abs(daily_sentiment['overall_score_mean'])  # 情绪强度
            daily_sentiment['weighted_sentiment'] = (daily_sentiment['overall_score_mean'] *
                                                   daily_sentiment['certainty_mean'])  # 确定性加权情绪

            # 2. 情绪变化因子
            daily_sentiment['sentiment_change_1d'] = daily_sentiment['overall_score_mean'].diff()
            daily_sentiment['sentiment_change_3d'] = daily_sentiment['overall_score_mean'].diff(3)
            daily_sentiment['sentiment_momentum'] = daily_sentiment['overall_score_mean'].rolling(3).mean()
            daily_sentiment['sentiment_volatility'] = daily_sentiment['overall_score_mean'].rolling(5).std()

            # 3. 情绪极值因子
            daily_sentiment['sentiment_max_impact'] = daily_sentiment['overall_score_max']
            daily_sentiment['sentiment_min_impact'] = daily_sentiment['overall_score_min']
            daily_sentiment['sentiment_range'] = (daily_sentiment['overall_score_max'] -
                                                daily_sentiment['overall_score_min'])

            # 4. 情绪一致性因子
            daily_sentiment['sentiment_consistency'] = (1 - daily_sentiment['overall_score_std'].fillna(0))
            daily_sentiment['certainty_strength'] = daily_sentiment['certainty_mean']

            # 5. 情绪累积因子
            daily_sentiment['sentiment_cumsum_3d'] = daily_sentiment['overall_score_mean'].rolling(3).sum()
            daily_sentiment['sentiment_cumsum_5d'] = daily_sentiment['overall_score_mean'].rolling(5).sum()
            daily_sentiment['sentiment_cumsum_10d'] = daily_sentiment['overall_score_mean'].rolling(10).sum()

            print(f"✓ 情绪因子计算完成")

            # === 合并情绪因子到主数据 ===
            print(f"   合并情绪因子到主数据...")

            # 合并数据
            factor_results_with_sentiment = pd.merge(
                factor_results.reset_index(),
                daily_sentiment,
                on='date',
                how='left'
            ).set_index('date')

            # 更新factor_results
            factor_results = factor_results_with_sentiment

            # 填充缺失的情绪数据
            sentiment_columns = [col for col in factor_results.columns if any(word in col for word in
                               ['sentiment', 'news', 'certainty', 'impact', 'overall_score'])]

            factor_results[sentiment_columns] = factor_results[sentiment_columns].fillna(0)

            print(f"✓ 情绪因子合并完成: {len(sentiment_columns)} 个情绪因子")

            # === 计算新闻-成交量交互因子 ===
            print(f"   计算新闻-成交量交互因子...")

            # 确保有成交量数据
            if 'volume' in factor_results.columns:
                # 1. 成交量相关基础因子
                factor_results['volume_ratio_5d'] = factor_results['volume'] / factor_results['volume'].rolling(5).mean()
                factor_results['volume_ratio_20d'] = factor_results['volume'] / factor_results['volume'].rolling(20).mean()
                factor_results['volume_change'] = factor_results['volume'].pct_change()

                # 2. 新闻-成交量交互因子
                factor_results['news_vol_interaction_5d'] = (factor_results['overall_score_mean'] *
                                                           factor_results['volume_ratio_5d'])
                factor_results['news_vol_interaction_20d'] = (factor_results['overall_score_mean'] *
                                                            factor_results['volume_ratio_20d'])

                # 3. 情绪强度过滤因子
                positive_sentiment = np.where(factor_results['overall_score_mean'] > 0,
                                            factor_results['overall_score_mean'], 0)
                negative_sentiment = np.where(factor_results['overall_score_mean'] < 0,
                                            factor_results['overall_score_mean'], 0)
                volume_amplification = np.where(factor_results['volume_ratio_5d'] > 1.2,
                                              factor_results['volume_ratio_5d'], 0)

                factor_results['filtered_positive_news'] = positive_sentiment * volume_amplification
                factor_results['filtered_negative_news'] = negative_sentiment * volume_amplification

                # 4. 确定性加权的新闻-成交量因子
                factor_results['certainty_vol_factor'] = (factor_results['certainty_mean'] *
                                                        factor_results['volume_ratio_20d'])

                # 5. 综合新闻-成交量因子
                factor_results['comprehensive_news_vol'] = (
                    factor_results['overall_score_mean'] *
                    factor_results['certainty_mean'] *
                    factor_results['volume_ratio_20d'] *
                    np.sign(factor_results['volume_change'])
                )

                print(f"✓ 新闻-成交量交互因子计算完成")
            else:
                print(f"⚠️ 缺少成交量数据，跳过新闻-成交量交互因子")

        else:
            print(f"⚠️ 情绪数据文件不存在: {sentiment_file}")
            print(f"   跳过情绪因子处理，仅计算未来收益率")

        # === 统计结果 ===
        print(f"\n📊 数据扩展完成!")
        print(f"   最终数据形状: {factor_results.shape}")

        # 统计各类因子
        future_return_columns = [col for col in factor_results.columns if col.startswith('future_return') or col.startswith('open_performance')]
        sentiment_factor_columns = [col for col in factor_results.columns if any(word in col for word in
                                   ['sentiment', 'news', 'certainty', 'impact', 'overall_score'])]
        technical_factor_columns = [col for col in factor_results.columns if col.startswith('factor_')]
        interaction_factor_columns = [col for col in factor_results.columns if any(word in col for word in
                                     ['interaction', 'filtered', 'comprehensive', 'certainty_vol'])]

        print(f"   技术因子: {len(technical_factor_columns)} 个")
        print(f"   未来收益率: {len(future_return_columns)} 个")
        print(f"   情绪因子: {len(sentiment_factor_columns)} 个")
        print(f"   交互因子: {len(interaction_factor_columns)} 个")
        print(f"   总因子数: {len(technical_factor_columns) + len(sentiment_factor_columns) + len(interaction_factor_columns)} 个")

        # 显示因子分类
        if sentiment_factor_columns:
            print(f"\n📰 情绪因子列表:")
            sentiment_categories = {
                '基础情绪': [f for f in sentiment_factor_columns if any(word in f for word in
                            ['overall_score_mean', 'news_intensity', 'sentiment_strength', 'weighted_sentiment'])],
                '情绪变化': [f for f in sentiment_factor_columns if any(word in f for word in
                            ['change', 'momentum', 'volatility'])],
                '情绪极值': [f for f in sentiment_factor_columns if any(word in f for word in
                            ['max', 'min', 'range'])],
                '情绪一致性': [f for f in sentiment_factor_columns if any(word in f for word in
                              ['consistency', 'certainty'])],
                '情绪累积': [f for f in sentiment_factor_columns if 'cumsum' in f]
            }

            for category, factors in sentiment_categories.items():
                if factors:
                    print(f"  {category} ({len(factors)}个): {factors[:3]}{'...' if len(factors) > 3 else ''}")

        print(f"\n📈 未来收益率列表: {future_return_columns}")

        # 数据质量检查
        print(f"\n🔍 数据质量检查:")
        missing_count = factor_results.isnull().sum().sum()
        total_values = factor_results.shape[0] * factor_results.shape[1]
        completeness = (1 - missing_count / total_values) * 100
        print(f"   数据完整度: {completeness:.1f}%")
        print(f"   缺失值: {missing_count} / {total_values}")

        # 显示时间范围
        print(f"   时间范围: {factor_results.index[0]} 至 {factor_results.index[-1]}")
        print(f"   有效交易日: {len(factor_results)} 天")

    except Exception as e:
        print(f"❌ 未来收益率和情绪因子计算失败: {e}")
        import traceback
        traceback.print_exc()

else:
    print("⚠️ 无法计算未来收益率：缺少factor_results数据")