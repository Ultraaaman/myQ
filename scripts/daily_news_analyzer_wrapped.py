#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日新闻分析脚本（修复Tushare API版本）
- 获取全部新闻
- 匹配候选股票池
- 大模型评分
- 生成因子强度报告
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
        """
        初始化新闻分析器

        Args:
            tushare_token: Tushare API token
            openrouter_api_key: OpenRouter API key
        """
        # 修正：使用pro_api方式初始化
        self.ts_pro = ts.pro_api(tushare_token)
        self.api_key = openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }

        # 首先设置文件路径
        self.base_dir = Path("D:/projects/q/myQ")
        self.output_dir = self.base_dir / "output" / "daily_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 然后加载股票池
        self.stock_pool = self._load_stock_pool()
        self.stock_keywords = self._extract_keywords()

    def _load_stock_pool(self):
        """加载股票池 - 修复版本"""
        stock_pool_path = self.base_dir / "config" / "stock_pool.json"
        try:
            with open(stock_pool_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 支持不同的JSON格式
            if 'stocks' in data:
                stocks = data['stocks']
            elif 'stock_database' in data:
                stocks = data['stock_database']
            else:
                # 假设数据直接是股票列表
                stocks = data if isinstance(data, list) else []

            logger.info(f"加载股票池，共 {len(stocks)} 只股票")
            return stocks
        except Exception as e:
            logger.error(f"加载股票池失败: {e}")
            return []

    def _extract_keywords(self):
        """提取关键词"""
        keywords = {}
        for stock in self.stock_pool:
            stock_keywords = []
            stock_keywords.append(stock['stock_name'])
            stock_keywords.append(stock['stock_code'])

            # 提取行业和主营业务关键词
            if 'industry' in stock:
                stock_keywords.extend(stock['industry'].split())
            if 'main_business' in stock:
                # 简单提取主营业务中的关键词
                business_words = stock['main_business'].replace('，', ' ').replace('、', ' ').split()
                business_words = [w for w in business_words if len(w) >= 2][:5]  # 限制数量
                stock_keywords.extend(business_words)

            # 去重
            keywords[stock['stock_code']] = list(set(stock_keywords))

        logger.info(f"提取关键词完成，覆盖 {len(keywords)} 只股票")
        return keywords

    def get_daily_news(self, target_date=None):
        """获取当日新闻 - 修正API调用"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y%m%d')
        else:
            # 转换格式 YYYY-MM-DD -> YYYYMMDD
            target_date = target_date.replace('-', '')

        logger.info(f"获取 {target_date} 的新闻数据...")

        try:
            # 修正：使用正确的API调用格式
            news_df = self.ts_pro.news(
                src='sina',
                start_date=target_date,
                end_date=target_date
            )

            if news_df.empty:
                logger.warning("未获取到新闻数据")
                return pd.DataFrame()

            # 重命名列
            news_df = news_df.rename(columns={
                'datetime': 'original_date',
                'title': 'original_title',
                'content': 'original_content',
                'source': 'original_source'
            })

            logger.info(f"获取到 {len(news_df)} 条新闻")
            return news_df

        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return pd.DataFrame()

    def match_news_to_stocks(self, news_df):
        """匹配新闻到股票"""
        if news_df.empty:
            return news_df

        logger.info("开始匹配新闻到股票...")

        matched_records = []

        for _, news_row in news_df.iterrows():
            news_title = str(news_row['original_title']).lower()
            news_content = str(news_row['original_content']).lower()
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

    def score_news_with_llm(self, matched_df):
        """使用大模型对新闻进行评分"""
        if matched_df.empty:
            return matched_df

        logger.info("开始使用大模型评分...")

        # 构建prompt模板
        prompt_template = """
你是一位专业的金融分析师，请分析以下新闻对指定个股的影响。

背景信息：
- 股票：{stock_name}({stock_code})
- 所属行业：{industry}
- 主营业务：{main_business}
- 当前市值：{market_cap}

新闻信息：
标题：{news_title}
内容：{news_content}
发布时间：{publish_time}
消息来源：{source}

请从以下维度分析：

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

请严格按照以下JSON格式返回：
{
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
}
"""

        scored_results = []

        for idx, row in matched_df.iterrows():
            try:
                # 构建提示词
                prompt = prompt_template.format(
                    stock_name=row['stock_name'],
                    stock_code=row['stock_code'],
                    industry=row.get('industry', 'N/A'),
                    main_business=row.get('main_business', 'N/A'),
                    market_cap=row.get('market_cap', 'N/A'),
                    news_title=row['original_title'],
                    news_content=str(row['original_content'])[:1000],  # 限制长度
                    publish_time=row['original_date'],
                    source=row['original_source']
                )

                # 调用API
                response = self._call_llm_api(prompt)

                if response:
                    # 解析结果
                    result = row.to_dict()
                    result['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    result.update(response)
                    scored_results.append(result)

                    logger.info(f"已评分: {row['stock_name']} - 综合分数: {response.get('overall_score', 'N/A')}")

                # 控制请求频率
                time.sleep(random.uniform(0.5, 1.0))

            except Exception as e:
                logger.error(f"评分失败 {row['stock_name']}: {e}")
                continue

        logger.info(f"完成评分，共 {len(scored_results)} 条")
        return pd.DataFrame(scored_results)

    def _call_llm_api(self, prompt):
        """调用大模型API - 修复版本"""
        try:
            payload = {
                "model": "deepseek/deepseek-chat-v3.1:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # 尝试解析JSON - 改进版本
                try:
                    # 清理内容
                    content = content.strip()

                    # 查找JSON块
                    start = content.find('{')
                    end = content.rfind('}') + 1

                    if start != -1 and end > start:
                        json_str = content[start:end]

                        # 清理可能的问题字符
                        json_str = json_str.replace('\n', '').replace('\r', '').replace('\t', '')

                        # 尝试解析
                        parsed_json = json.loads(json_str)

                        # 验证必要字段
                        required_fields = ['sentiment', 'overall_score', 'certainty']
                        for field in required_fields:
                            if field not in parsed_json:
                                logger.warning(f"缺少必要字段: {field}")
                                parsed_json[field] = 0 if field in ['overall_score', 'certainty'] else "中性"

                        return parsed_json

                except json.JSONDecodeError as je:
                    logger.error(f"JSON解析错误: {je}")
                    logger.debug(f"原始响应: {content}")

                    # 尝试提取基本信息
                    fallback_result = {
                        "sentiment": "中性",
                        "overall_score": 0,
                        "certainty": 0.5,
                        "raw_response": content,
                        "error": "JSON解析失败"
                    }
                    return fallback_result

                except Exception as e:
                    logger.error(f"响应处理异常: {e}")
                    logger.debug(f"原始响应: {content}")
                    return None

            else:
                logger.error(f"API请求失败: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"API调用异常: {e}")

        return None

    def save_results(self, results_df, target_date):
        """保存结果到CSV，支持追加和覆盖"""
        if results_df.empty:
            logger.warning("没有结果可保存")
            return

        filename = self.output_dir / f"news_analysis_{target_date.replace('-', '')}.csv"

        try:
            # 直接覆盖文件（实现同日覆盖）
            results_df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"结果已保存到: {filename}")

        except Exception as e:
            logger.error(f"保存结果失败: {e}")

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
    """主函数"""
    # 这个函数在run_daily_analysis.py中调用，这里保持简洁
    pass

if __name__ == "__main__":
    main()