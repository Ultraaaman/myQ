#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日新闻分析脚本（正式版 - 批量LLM处理）
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

        with open(stock_pool_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 支持不同的JSON格式
        if 'stocks' in data:
            stocks = data['stocks']
        elif 'stock_database' in data:
            stocks = data['stock_database']
        else:
            stocks = data if isinstance(data, list) else []

        logger.info(f"加载股票池，共 {len(stocks)} 只股票")
        return stocks

    def _extract_keywords(self):
        """提取关键词"""
        keywords = {}

        for stock in self.stock_pool:
            stock_keywords = []
            stock_keywords.append(stock['stock_name'])
            stock_keywords.append(stock['stock_code'])

            # 暂时只用股票名称和代码作为关键词（用户已注释掉行业和业务关键词）
            keywords[stock['stock_code']] = list(set(stock_keywords))

        logger.info(f"提取关键词完成，覆盖 {len(keywords)} 只股票")
        return keywords

    def get_daily_news(self, target_date=None):
        """获取当日新闻"""
        if target_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d 09:00:00')
            end_date = datetime.now().strftime('%Y-%m-%d 18:00:00')
        else:
            start_date = f"{target_date} 09:00:00"
            end_date = f"{target_date} 18:00:00"

        logger.info(f"获取 {target_date or '今日'} 的新闻数据...")

        news_df = self.ts_pro.news(
            src='sina',
            start_date=start_date,
            end_date=end_date
        )

        if news_df.empty:
            logger.warning("未获取到新闻数据")
            return pd.DataFrame()

        logger.info(f"获取到 {len(news_df)} 条新闻")
        return news_df

    def match_news_to_stocks(self, news_df):
        """匹配新闻到股票"""
        if news_df.empty:
            return news_df

        logger.info("开始匹配新闻到股票...")
        matched_records = []

        for _, news_row in news_df.iterrows():
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
                for keyword in keywords:
                    if len(keyword) >= 2 and keyword.lower() in full_text:
                        matched = True
                        break

                if matched:
                    record = news_row.to_dict()
                    record.update(stock)
                    matched_records.append(record)

        matched_df = pd.DataFrame(matched_records)
        logger.info(f"匹配完成，找到 {len(matched_df)} 条股票相关新闻")

        return matched_df

    def score_news_with_llm(self, matched_df, batch_size=5):
        """使用大模型批量对新闻进行评分"""
        if matched_df.empty:
            logger.warning("匹配的新闻数据为空，跳过LLM评分")
            return matched_df

        logger.info(f"开始使用大模型批量评分，共 {len(matched_df)} 条新闻，批量大小: {batch_size}")

        scored_results = []
        total_batches = (len(matched_df) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(matched_df))
            batch_df = matched_df.iloc[start_idx:end_idx]

            logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批次，包含 {len(batch_df)} 条新闻")

            # 批量评分
            batch_results = self._score_batch_llm(batch_df)
            scored_results.extend(batch_results)

            # 控制请求频率
            if batch_idx < total_batches - 1:  # 不是最后一批
                time.sleep(random.uniform(2.0, 3.0))

        scored_df = pd.DataFrame(scored_results)
        logger.info(f"LLM批量评分完成，成功处理 {len(scored_df)} 条")

        return scored_df

    def _score_batch_llm(self, batch_df):
        """批量评分一组新闻"""
        # 构建批量prompt模板
        batch_prompt_template = """
你是一位专业的金融分析师，请分析以下多条新闻对对应个股的影响。

请对以下每条新闻分别进行分析，严格按照JSON数组格式返回，每个对象对应一条新闻：

{news_items}

分析维度：
1. 直接影响评估（-5到+5分）
2. 间接影响评估（-5到+5分）
3. 确定性评估（0-1）
4. 综合评分（-10到+10）

请严格按照以下JSON数组格式返回，数组中每个对象对应上述一条新闻：
[
  {{
    "news_index": 0,
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
  }},
  {{
    "news_index": 1,
    ...
  }}
]
"""

        # 构建新闻项目列表
        news_items = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            news_item = f"""
新闻 {idx}:
- 股票：{row['stock_name']}({row['stock_code']})
- 所属行业：{row.get('industry', 'N/A')}
- 主营业务：{row.get('main_business', 'N/A')}
- 当前市值：{row.get('market_cap', 'N/A')}
- 标题：{str(row.get('title', ''))}
- 内容：{str(row.get('content', ''))[:600]}
- 发布时间：{row.get('datetime', '')}
- 消息来源：{row.get('source', '')}
"""
            news_items.append(news_item)

        # 构建完整提示词
        prompt = batch_prompt_template.format(
            news_items='\n'.join(news_items)
        )

        # 调用API
        responses = self._call_batch_llm_api(prompt, len(batch_df))

        # 处理结果
        batch_results = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            result = row.to_dict()
            result['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 匹配对应的评分结果
            if responses and idx < len(responses):
                response = responses[idx]
                if isinstance(response, dict):
                    result.update(response)
                    logger.info(f"已评分: {row['stock_name']} - 综合分数: {response.get('overall_score', 'N/A')}")
                else:
                    logger.warning(f"评分结果格式异常: {row['stock_name']}")
            else:
                logger.warning(f"未获取到评分结果: {row['stock_name']}")

            batch_results.append(result)

        return batch_results

    def _call_batch_llm_api(self, prompt, expected_count):
        """调用批量LLM API"""
        logger.info(f"调用LLM API，期望返回 {expected_count} 条评分结果...")

        payload = {
            "model": "deepseek/deepseek-chat-v3.1:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4000  # 增加token限制以支持批量响应
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=60  # 增加超时时间
        )

        logger.info(f"API响应状态: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            logger.info(f"返回内容长度: {len(content)}")

            # 解析JSON数组
            try:
                # 清理内容
                content = content.strip()

                # 查找JSON数组
                start = content.find('[')
                end = content.rfind(']') + 1

                if start != -1 and end > start:
                    json_str = content[start:end]

                    # 清理可能的问题字符
                    json_str = json_str.replace('\n', '').replace('\r', '').replace('\t', '')

                    # 尝试解析
                    parsed_json = json.loads(json_str)

                    if isinstance(parsed_json, list):
                        logger.info(f"成功解析 {len(parsed_json)} 条评分结果")

                        # 验证每个结果的必要字段
                        for i, item in enumerate(parsed_json):
                            if isinstance(item, dict):
                                required_fields = ['sentiment', 'overall_score', 'certainty']
                                for field in required_fields:
                                    if field not in item:
                                        logger.warning(f"第{i}条结果缺少必要字段: {field}")
                                        item[field] = 0 if field in ['overall_score', 'certainty'] else "中性"

                        return parsed_json
                    else:
                        logger.error("返回内容不是JSON数组格式")
                        return []

            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {e}")
                logger.debug(f"原始响应: {content[:500]}...")
                return []

        else:
            logger.error(f"API请求失败: {response.status_code}")
            if response.text:
                logger.error(f"错误信息: {response.text[:200]}")

        return []

    def save_results(self, results_df, target_date):
        """保存结果到CSV，支持追加和覆盖"""
        if results_df.empty:
            logger.warning("没有结果可保存")
            return

        filename = self.output_dir / f"news_analysis_{target_date.replace('-', '')}.csv"

        # 直接覆盖文件（实现同日覆盖）
        results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"结果已保存到: {filename}")

    def generate_factor_report(self, results_df, target_date):
        """生成因子强度报告"""
        if results_df.empty:
            logger.warning("没有数据生成报告")
            return pd.DataFrame()

        logger.info("开始生成因子强度报告...")

        # 按股票聚合
        stock_scores = results_df.groupby(['stock_code', 'stock_name']).agg({
            'overall_score': ['mean', 'max', 'count'],
            'certainty': 'mean'
        }).round(4)

        # 重命名列
        stock_scores.columns = ['avg_score', 'max_score', 'news_count', 'avg_certainty']
        stock_scores = stock_scores.reset_index()

        # 计算因子强度
        stock_scores['factor_strength'] = (
            stock_scores['avg_score'] * 0.4 +
            stock_scores['max_score'] * 0.3 +
            stock_scores['avg_certainty'] * 10 * 0.3
        ).round(4)

        # 按因子强度排序
        stock_scores = stock_scores.sort_values('factor_strength', ascending=False)

        # 保存股票评分结果
        scores_filename = self.output_dir / f"stock_scores_{target_date.replace('-', '')}.csv"
        stock_scores.to_csv(scores_filename, index=False, encoding='utf-8-sig')

        # 生成Markdown报告
        report_filename = self.output_dir / f"factor_report_{target_date.replace('-', '')}.md"

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Daily News Factor Analysis Report - {target_date}\n\n")

            # 强因子股票
            strong_stocks = stock_scores[stock_scores['factor_strength'] >= 3.0]

            if not strong_stocks.empty:
                f.write("## Strong Factor Stocks (Factor Strength >= 3.0)\n\n")
                f.write("| Rank | Stock | Code | Factor Strength | Avg Score | Max Score | News Count |\n")
                f.write("|------|-------|------|----------------|-----------|-----------|------------|\n")

                for idx, row in strong_stocks.head(10).iterrows():
                    f.write(f"| {idx+1} | {row['stock_name']} | {row['stock_code']} | {row['factor_strength']:.4f} | {row['avg_score']:.4f} | {row['max_score']:.4f} | {row['news_count']} |\n")
            else:
                f.write("## No Strong Factor Stocks Found\n\n")
                f.write("No stocks with factor strength >= 3.0 found today.\n\n")

            f.write("\n## All Analyzed Stocks\n\n")
            f.write("| Rank | Stock | Code | Factor Strength | Avg Score | Max Score | News Count |\n")
            f.write("|------|-------|------|----------------|-----------|-----------|------------|\n")

            for idx, row in stock_scores.head(20).iterrows():
                f.write(f"| {idx+1} | {row['stock_name']} | {row['stock_code']} | {row['factor_strength']:.4f} | {row['avg_score']:.4f} | {row['max_score']:.4f} | {row['news_count']} |\n")

        logger.info(f"因子强度报告已保存到: {report_filename}")

        # 返回强因子股票
        return strong_stocks

def main():
    """正式版主函数"""
    # 导入配置
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "config"))
    from api_config import TUSHARE_TOKEN, OPENROUTER_API_KEY

    target_date = "2024-12-20"
    print(f"🚀 开始每日新闻因子分析 - {target_date}")

    # 初始化分析器
    analyzer = DailyNewsAnalyzer(TUSHARE_TOKEN, OPENROUTER_API_KEY)

    # 1. 获取当日新闻
    print("\n📰 步骤1: 获取新闻数据...")
    news_df = analyzer.get_daily_news(target_date)
    if news_df.empty:
        print("未获取到新闻数据，程序结束")
        return

    # 2. 匹配新闻到股票
    print("\n🔍 步骤2: 匹配股票相关新闻...")
    matched_df = analyzer.match_news_to_stocks(news_df)
    if matched_df.empty:
        print("未匹配到相关股票新闻，程序结束")
        return

    # 3. 大模型批量评分
    print("\n🤖 步骤3: 批量LLM评分...")
    scored_df = analyzer.score_news_with_llm(matched_df)
    if scored_df.empty:
        print("LLM评分失败，程序结束")
        return

    # 4. 保存结果
    print("\n💾 步骤4: 保存结果...")
    analyzer.save_results(scored_df, target_date)

    # 5. 生成因子强度报告
    print("\n📊 步骤5: 生成因子强度报告...")
    strong_stocks = analyzer.generate_factor_report(scored_df, target_date)

    print(f"\n✅ 分析完成！")
    print(f"📊 分析了 {len(scored_df)} 条股票新闻")

    if not strong_stocks.empty:
        print(f"🔥 发现 {len(strong_stocks)} 只强因子股票：")
        for idx, row in strong_stocks.head(5).iterrows():
            print(f"   {idx+1}. {row['stock_name']} ({row['stock_code']}): 因子强度 {row['factor_strength']:.2f}")
    else:
        print("📉 今日未发现强因子股票")

    print(f"\n📁 结果已保存到: {analyzer.output_dir}")

if __name__ == "__main__":
    main()