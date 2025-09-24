#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日新闻分析脚本（正式版v2 - 批量LLM处理 + 重试机制）
"""

import tushare as ts
import pandas as pd
import json
import requests
import os
import time
import random
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyNewsAnalyzer:
    def __init__(self, tushare_token, openrouter_api_key):
        """初始化新闻分析器"""
        print(f"🔧 正在初始化分析器...")

        # 初始化API
        print(f"🔧 初始化Tushare API...")
        self.ts_pro = ts.pro_api(tushare_token)
        print(f"✅ Tushare API初始化完成")

        self.api_key = openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }

        # 设置文件路径
        print(f"🔧 设置文件路径...")
        self.base_dir = Path("D:/projects/q/myQ")
        self.output_dir = self.base_dir / "output" / "daily_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 输出目录: {self.output_dir}")

        # 加载股票池
        print(f"🔧 加载股票池...")
        self.stock_pool = self._load_stock_pool()
        print(f"✅ 股票池加载完成: {len(self.stock_pool)} 只股票")

        print(f"🔧 提取关键词...")
        self.stock_keywords = self._extract_keywords()
        print(f"✅ 关键词提取完成: {len(self.stock_keywords)} 个股票")

    def _load_stock_pool(self):
        """加载股票池"""
        stock_pool_path = self.base_dir / "config" / "stock_pool.json"
        print(f"📁 读取股票池文件: {stock_pool_path}")

        with open(stock_pool_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📊 原始JSON数据键: {list(data.keys())}")

        # 支持不同的JSON格式
        if 'stocks' in data:
            stocks = data['stocks']
            print(f"✅ 使用 'stocks' 键")
        elif 'stock_database' in data:
            stocks = data['stock_database']
            print(f"✅ 使用 'stock_database' 键")
        else:
            stocks = data if isinstance(data, list) else []
            print(f"✅ 数据本身就是列表")

        print(f"📊 最终股票数量: {len(stocks)}")
        return stocks

    def _extract_keywords(self):
        """提取关键词"""
        keywords = {}

        for i, stock in enumerate(self.stock_pool):
            stock_keywords = []
            stock_keywords.append(stock['stock_name'])
            stock_keywords.append(stock['stock_code'])

            # 提取行业和主营业务关键词
            if 'industry' in stock:
                stock_keywords.extend(stock['industry'].split())
            if 'main_business' in stock:
                business_words = stock['main_business'].replace('，', ' ').replace('、', ' ').split()
                business_words = [w for w in business_words if len(w) >= 2][:5]
                stock_keywords.extend(business_words)

            # 去重
            keywords[stock['stock_code']] = list(set(stock_keywords))

            # 显示前3个股票的关键词用于调试
            if i < 3:
                print(f"  {stock['stock_name']} ({stock['stock_code']}): {keywords[stock['stock_code']]}")

        return keywords

    def get_daily_news(self, target_date=None):
        """获取当日新闻"""
        if target_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d 09:00:00')
            end_date = datetime.now().strftime('%Y-%m-%d 18:00:00')
        else:
            start_date = f"{target_date} 09:00:00"
            end_date = f"{target_date} 18:00:00"

        print(f"📰 准备获取新闻数据...")
        print(f"   - 开始时间: {start_date}")
        print(f"   - 结束时间: {end_date}")

        news_df = self.ts_pro.news(
            src='sina',
            start_date=start_date,
            end_date=end_date
        )

        print(f"📊 API返回结果:")
        print(f"   - 数据类型: {type(news_df)}")
        print(f"   - 是否为空: {news_df.empty if hasattr(news_df, 'empty') else 'N/A'}")
        print(f"   - 数据形状: {news_df.shape if hasattr(news_df, 'shape') else 'N/A'}")

        if hasattr(news_df, 'columns'):
            print(f"   - 列名: {list(news_df.columns)}")

        return news_df

    def match_news_to_stocks(self, news_df):
        """匹配新闻到股票"""
        if news_df.empty:
            print("⚠️ 新闻数据为空，跳过匹配")
            return news_df

        print(f"🔍 开始匹配新闻到股票...")
        matched_records = []

        for i, (_, news_row) in enumerate(news_df.iterrows()):
            if i < 2:  # 显示前2条新闻的调试信息
                title = news_row.get('title') or news_row.get('Title') or ''
                print(f"  处理第 {i+1} 条新闻: {str(title)[:50] if title else '(无标题)'}...")

            # 安全获取字段值
            title = news_row.get('title') or news_row.get('Title') or ''
            content = news_row.get('content') or news_row.get('Content') or ''

            news_title = str(title).lower() if title else ''
            news_content = str(content).lower() if content else ''
            full_text = f"{news_title} {news_content}"

            # 匹配股票
            for stock in self.stock_pool:
                stock_code = stock['stock_code']
                keywords = self.stock_keywords.get(stock_code, [])

                # 检查是否匹配
                matched = False
                matched_keyword = None
                for keyword in keywords:
                    if len(keyword) >= 2 and keyword.lower() in full_text:
                        matched = True
                        matched_keyword = keyword
                        break

                if matched:
                    record = news_row.to_dict()
                    record.update(stock)
                    matched_records.append(record)

        matched_df = pd.DataFrame(matched_records)
        print(f"🎯 匹配完成，找到 {len(matched_df)} 条股票相关新闻")

        return matched_df

    def score_news_with_llm(self, matched_df, batch_size=4):
        """使用大模型批量对新闻进行评分"""
        if matched_df.empty:
            print("⚠️ 匹配的新闻数据为空，跳过LLM评分")
            return matched_df

        print(f"🤖 开始使用大模型批量评分...")
        print(f"   - 总新闻数量: {len(matched_df)}")
        print(f"   - 批次大小: {batch_size}")

        scored_results = []
        total_batches = (len(matched_df) + batch_size - 1) // batch_size
        print(f"   - 总批次数: {total_batches}")

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(matched_df))
            batch_df = matched_df.iloc[start_idx:end_idx]

            print(f"  📦 处理批次 {batch_idx + 1}/{total_batches} (新闻 {start_idx + 1}-{end_idx})")

            batch_results = self._score_batch_llm(batch_df)
            if batch_results:
                scored_results.extend(batch_results)
                print(f"    ✅ 批次评分完成，获得 {len(batch_results)} 条结果")
            else:
                print(f"    ❌ 批次评分失败")

            # 批次间等待时间（避免API速率限制）
            if batch_idx < total_batches - 1:
                wait_time = 2 + random.uniform(0, 1)  # 2-3秒间隔
                print(f"    ⏰ 等待 {wait_time:.1f} 秒后处理下一批...")
                time.sleep(wait_time)

        scored_df = pd.DataFrame(scored_results)
        print(f"🎯 LLM批量评分完成，共 {len(scored_df)} 条")

        return scored_df

    def _score_batch_llm(self, batch_df):
        """批量评分一批新闻"""
        # 构建批量prompt
        prompt = self._build_batch_prompt(batch_df)

        # 调用API（带重试机制）
        response = self._call_llm_api_with_retry(prompt)

        if response:
            return self._parse_batch_response(response, batch_df)
        else:
            return []

    def _build_batch_prompt(self, batch_df):
        """构建批量评分的prompt"""
        prompt = """你是一位专业的金融分析师，请分析以下多条新闻对对应个股的影响。

请对每条新闻独立分析，从以下维度评估：

1. 直接影响评估（-5到+5分）
   - 新闻是否直接提及该公司或其核心业务？
   - 对公司收入/成本/利润的直接影响？

2. 间接影响评估（-5到+5分）
   - 对行业整体的影响？
   - 对产业链上下游的影响？
   - 对竞争格局的影响？

3. 确定性评估（0-1）
   - 影响发生的可能性有多大？

4. 影响时间窗口
   - 立即、1周内、1个月内、3个月内、6个月以上

5. 综合评分（-10到+10）
   - 综合考虑直接影响、间接影响、确定性得出总分

新闻列表：
"""

        for idx, (_, row) in enumerate(batch_df.iterrows()):
            prompt += f"""
【新闻{idx+1}】
股票：{row['stock_name']}({row['stock_code']})
行业：{row.get('industry', 'N/A')}
主营：{row.get('main_business', 'N/A')}
标题：{str(row.get('title', ''))}
内容：{str(row.get('content', ''))[:600]}
时间：{row.get('datetime', '')}
"""

        prompt += """
请严格按照以下JSON数组格式返回，数组中每个对象对应一条新闻：
[
  {
    "news_index": 1,
    "sentiment": "强烈正面/正面/中性偏正/中性/中性偏负/负面/强烈负面",
    "direct_impact_score": 分数,
    "direct_impact_desc": "描述",
    "indirect_impact_score": 分数,
    "indirect_impact_desc": "描述",
    "certainty": 0.xx,
    "time_to_effect": "时间窗口",
    "overall_score": 综合分数,
    "risk_factors": "主要风险因素",
    "action_suggestion": "建议操作"
  },
  ...
]
"""

        return prompt

    def _call_llm_api_with_retry(self, prompt, max_retries=3):
        """调用LLM API（带指数退避重试机制）"""
        base_delay = 20  # 基础延迟时间（秒）

        for attempt in range(max_retries):
            try:
                print(f"    🔧 调用LLM API (第{attempt+1}次尝试)...")

                payload = {
                    "model": "deepseek/deepseek-chat-v3.1:free",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4000
                }

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )

                print(f"    📊 API响应状态: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    print(f"    📝 返回内容长度: {len(content)}")
                    return content

                elif response.status_code == 429:
                    # 429错误使用更长的等待时间
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"    ⚠️ API速率限制(429)，等待 {wait_time:.2f} 秒后重试 {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    continue

                else:
                    # 其他错误使用较短的等待时间
                    wait_time = base_delay * (1.5 ** attempt)
                    print(f"    ⚠️ API请求失败，状态码: {response.status_code}, 等待 {wait_time:.2f} 秒后重试 {attempt + 1}/{max_retries}")
                    if response.text:
                        print(f"    错误信息: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    continue

            except Exception as e:
                wait_time = base_delay * (2 ** attempt)
                print(f"    ❌ API调用出错: {e}, 等待 {wait_time:.2f} 秒后重试 {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue

        print(f"    ❌ API调用失败，已重试 {max_retries} 次")
        return None

    def _parse_batch_response(self, content, batch_df):
        """解析批量响应"""
        results = []

        try:
            # 清理内容
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # 查找JSON数组
            start = content.find('[')
            end = content.rfind(']') + 1

            if start != -1 and end > start:
                json_str = content[start:end]

                # 清理可能的问题字符
                json_str = json_str.replace('\n', '').replace('\r', '').replace('\t', '')

                # 尝试解析
                parsed_array = json.loads(json_str)

                # 验证是否为数组
                if isinstance(parsed_array, list):
                    for i, analysis_result in enumerate(parsed_array):
                        if i < len(batch_df):
                            # 获取对应的新闻行
                            row = batch_df.iloc[i]

                            # 构建结果
                            result = row.to_dict()
                            result['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            # 验证必要字段
                            required_fields = ['sentiment', 'overall_score', 'certainty']
                            for field in required_fields:
                                if field not in analysis_result:
                                    analysis_result[field] = 0 if field in ['overall_score', 'certainty'] else "中性"

                            result.update(analysis_result)
                            results.append(result)

                print(f"    📝 解析完成，获得 {len(results)} 条分析结果")
                return results
            else:
                print(f"    ❌ 无法找到有效的JSON数组")
                return []

        except json.JSONDecodeError as e:
            print(f"    ❌ JSON解析错误: {e}")
            print(f"    原始响应: {content[:500]}...")
            return []
        except Exception as e:
            print(f"    ❌ 解析出错: {e}")
            return []

    def generate_factor_report(self, scored_df, target_date):
        """生成因子强度报告"""
        if scored_df.empty:
            print("⚠️ 没有评分数据，无法生成因子报告")
            return

        print(f"📊 生成因子强度报告...")

        # 计算每只股票的因子强度
        stock_factors = []

        for stock_code in scored_df['stock_code'].unique():
            stock_news = scored_df[scored_df['stock_code'] == stock_code]

            stock_name = stock_news.iloc[0]['stock_name']

            # 计算统计指标
            news_count = len(stock_news)
            avg_score = stock_news['overall_score'].mean()
            max_score = stock_news['overall_score'].max()
            min_score = stock_news['overall_score'].min()
            avg_certainty = stock_news['certainty'].mean()

            # 计算因子强度 = 平均分 * 0.4 + 最高分 * 0.3 + 确定性*10 * 0.3
            factor_strength = avg_score * 0.4 + max_score * 0.3 + avg_certainty * 10 * 0.3

            stock_factors.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'news_count': news_count,
                'avg_score': round(avg_score, 2),
                'max_score': round(max_score, 2),
                'min_score': round(min_score, 2),
                'avg_certainty': round(avg_certainty, 3),
                'factor_strength': round(factor_strength, 2)
            })

        # 转换为DataFrame并排序
        factor_df = pd.DataFrame(stock_factors)
        factor_df = factor_df.sort_values('factor_strength', ascending=False)

        # 保存详细结果
        detail_filename = self.output_dir / f"news_analysis_detail_{target_date.replace('-', '')}.csv"
        scored_df.to_csv(detail_filename, index=False, encoding='utf-8-sig')
        print(f"💾 详细分析结果已保存: {detail_filename}")

        # 保存因子报告
        factor_filename = self.output_dir / f"factor_strength_{target_date.replace('-', '')}.csv"
        factor_df.to_csv(factor_filename, index=False, encoding='utf-8-sig')
        print(f"💾 因子强度报告已保存: {factor_filename}")

        # 生成Markdown报告
        self._generate_markdown_report(factor_df, scored_df, target_date)

        print(f"🎯 因子报告生成完成！")

    def _generate_markdown_report(self, factor_df, scored_df, target_date):
        """生成Markdown格式的报告"""
        report = f"""# 每日新闻因子分析报告

**分析日期**: {target_date}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析新闻总数**: {len(scored_df)}
**涉及股票数量**: {len(factor_df)}

## 因子强度排行榜

| 排名 | 股票代码 | 股票名称 | 新闻数量 | 平均分 | 最高分 | 确定性 | **因子强度** |
|------|----------|----------|----------|--------|--------|--------|--------------|
"""

        for i, (idx, row) in enumerate(factor_df.head(15).iterrows()):
            report += f"| {i+1} | {row['stock_code']} | {row['stock_name']} | {row['news_count']} | {row['avg_score']} | {row['max_score']} | {row['avg_certainty']} | **{row['factor_strength']}** |\n"

        report += f"""
## 高关注股票详情

"""

        # 显示前5只高因子强度股票的详细新闻
        for i, (_, stock) in enumerate(factor_df.head(5).iterrows()):
            stock_code = stock['stock_code']
            stock_name = stock['stock_name']

            report += f"""### {stock_name} ({stock_code})
**因子强度**: {stock['factor_strength']} | **新闻数量**: {stock['news_count']}

"""

            stock_news = scored_df[scored_df['stock_code'] == stock_code].sort_values('overall_score', ascending=False)

            for _, news in stock_news.iterrows():
                sentiment = news['sentiment']
                score = news['overall_score']
                title = str(news.get('title', ''))[:60]

                report += f"- **[{sentiment}] {score}分** - {title}...\n"

            report += "\n"

        report += f"""
---
*报告由新闻因子分析系统自动生成*
*因子强度计算公式: 平均分 × 0.4 + 最高分 × 0.3 + 确定性×10 × 0.3*
"""

        # 保存Markdown报告
        md_filename = self.output_dir / f"factor_report_{target_date.replace('-', '')}.md"
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📝 Markdown报告已生成: {md_filename}")

def main():
    """正式版主函数"""
    print("🚀 开始正式版新闻分析（带重试机制）...")

    # 导入配置
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "config"))
    from api_config import TUSHARE_TOKEN, OPENROUTER_API_KEY

    print(f"🔑 配置检查:")
    print(f"   - Tushare Token: {'已配置' if TUSHARE_TOKEN != 'your_tushare_token_here' else '未配置'}")
    print(f"   - OpenRouter Key: {'已配置' if OPENROUTER_API_KEY != 'your_openrouter_api_key_here' else '未配置'}")

    # 初始化分析器
    analyzer = DailyNewsAnalyzer(TUSHARE_TOKEN, OPENROUTER_API_KEY)

    # 完整流程
    target_date = "2024-12-20"  # 可以修改为需要的日期

    print(f"\n📰 步骤1: 获取新闻数据...")
    news_df = analyzer.get_daily_news(target_date)

    if hasattr(news_df, 'empty') and not news_df.empty:
        print(f"\n🔍 步骤2: 匹配股票相关新闻...")
        matched_df = analyzer.match_news_to_stocks(news_df)

        if not matched_df.empty:
            print(f"\n🤖 步骤3: LLM批量情感分析...")
            scored_df = analyzer.score_news_with_llm(matched_df, batch_size=4)

            if not scored_df.empty:
                print(f"\n📊 步骤4: 生成因子强度报告...")
                analyzer.generate_factor_report(scored_df, target_date)
            else:
                print("❌ LLM评分失败")
        else:
            print("❌ 没有匹配到股票相关新闻")
    else:
        print("❌ 没有新闻数据")

    print(f"\n🎉 正式版分析完成！")

if __name__ == "__main__":
    main()