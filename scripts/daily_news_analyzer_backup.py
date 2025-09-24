#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日新闻分析脚本
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
        self.ts_pro = ts.pro_api(tushare_token)
        self.api_key = openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }

        # 加载股票池
        self.stock_pool = self._load_stock_pool()
        self.stock_keywords = self._extract_keywords()

        # 文件路径
        self.base_dir = Path("D:/projects/q/myQ")
        self.output_dir = self.base_dir / "output" / "daily_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_stock_pool(self):
        """加载股票池"""
        with open("D:/projects/q/myQ/config/stock_pool.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['stock_database']

    def _extract_keywords(self):
        """提取股票关键词用于新闻匹配"""
        keywords = {}
        for stock in self.stock_pool:
            stock_name = stock['stock_name']
            stock_code = stock['stock_code']
            industry = stock['industry']

            # 为每只股票创建关键词列表
            stock_keywords = [stock_name, stock_code]

            # 添加行业相关关键词
            if '银行' in industry:
                stock_keywords.extend(['银行业', '金融'])
            elif '电信' in industry:
                stock_keywords.extend(['通信', '5G'])
            elif '汽车' in industry:
                stock_keywords.extend(['新能源车', '汽车行业'])
            # 可以继续添加更多行业关键词

            keywords[stock_code] = {
                'stock_info': stock,
                'keywords': stock_keywords
            }

        return keywords

    def get_daily_news(self, target_date=None):
        """获取指定日期的新闻"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"获取 {target_date} 的新闻数据...")

        start_time = f"{target_date} 00:00:00"
        end_time = f"{target_date} 23:59:59"

        all_news = []

        try:
            # 获取财经新闻
            df = self.ts_pro.news(
                src='sina',  # 新浪财经
                start_date=start_time,
                end_date=end_time
            )

            if not df.empty:
                logger.info(f"获取到 {len(df)} 条新闻")
                all_news.append(df)
            else:
                logger.warning("未获取到新闻数据")

        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return pd.DataFrame()

        if all_news:
            combined_df = pd.concat(all_news, ignore_index=True)
            # 去重
            combined_df = combined_df.drop_duplicates(subset=['title', 'datetime'])
            logger.info(f"去重后共 {len(combined_df)} 条新闻")
            return combined_df
        else:
            return pd.DataFrame()

    def match_news_to_stocks(self, news_df):
        """将新闻匹配到相关股票"""
        matched_news = []

        logger.info("开始匹配新闻到股票...")

        for _, news in news_df.iterrows():
            title = str(news.get('title', ''))
            content = str(news.get('content', ''))
            combined_text = (title + ' ' + content).lower()

            # 检查每只股票的关键词
            for stock_code, stock_data in self.stock_keywords.items():
                keywords = stock_data['keywords']
                stock_info = stock_data['stock_info']

                # 检查是否包含关键词
                matched = False
                matched_keywords = []

                for keyword in keywords:
                    if keyword.lower() in combined_text:
                        matched = True
                        matched_keywords.append(keyword)

                if matched:
                    matched_news.append({
                        'news_id': f"{news.get('datetime', '')}_{hash(title)}",
                        'original_title': title,
                        'original_content': content,
                        'original_date': news.get('datetime', ''),
                        'original_source': news.get('src', '未知来源'),
                        'stock_name': stock_info['stock_name'],
                        'stock_code': stock_info['stock_code'],
                        'industry': stock_info['industry'],
                        'main_business': stock_info['main_business'],
                        'market_cap': stock_info['market_cap'],
                        'matched_keywords': ','.join(matched_keywords)
                    })

        logger.info(f"匹配到 {len(matched_news)} 条相关新闻")
        return pd.DataFrame(matched_news)

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

请按以下JSON格式返回：
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
                    industry=row['industry'],
                    main_business=row['main_business'],
                    market_cap=row['market_cap'],
                    news_title=row['original_title'],
                    news_content=row['original_content'][:1000],  # 限制长度
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
        """调用大模型API"""
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

                # 尝试解析JSON
                try:
                    # 提取JSON部分
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end != -1:
                        json_str = content[start:end]
                        return json.loads(json_str)
                except:
                    logger.warning("JSON解析失败，返回原始内容")
                    return {"raw_response": content}
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

        if filename.exists():
            # 读取现有数据
            existing_df = pd.read_csv(filename, encoding='utf-8-sig')

            # 删除同日期的旧数据（允许覆盖）
            existing_df = existing_df[existing_df['original_date'].str[:10] != target_date]

            # 追加新数据
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df = combined_df.sort_values('original_date').reset_index(drop=True)

            combined_df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"数据已更新到 {filename}")

        else:
            # 创建新文件
            results_df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存到 {filename}")

    def generate_factor_report(self, results_df, target_date):
        """生成因子强度报告"""
        if results_df.empty:
            logger.warning("没有数据可生成报告")
            return

        logger.info("生成因子强度报告...")

        # 计算各股票的因子强度
        stock_scores = results_df.groupby('stock_code').agg({
            'stock_name': 'first',
            'industry': 'first',
            'overall_score': ['mean', 'max', 'count'],
            'certainty': 'mean'
        }).reset_index()

        # 展平列名
        stock_scores.columns = ['stock_code', 'stock_name', 'industry',
                               'avg_score', 'max_score', 'news_count', 'avg_certainty']

        # 计算综合因子强度
        stock_scores['factor_strength'] = (
            stock_scores['avg_score'] * 0.4 +
            stock_scores['max_score'] * 0.3 +
            stock_scores['avg_certainty'] * 10 * 0.3
        )

        # 按因子强度排序
        stock_scores = stock_scores.sort_values('factor_strength', ascending=False)

        # 筛选强因子股票
        strong_factor_stocks = stock_scores[stock_scores['factor_strength'] >= 3.0].copy()

        # 生成报告
        report_lines = []
        report_lines.append(f"# {target_date} 新闻因子强度分析报告\n")
        report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"分析股票数: {len(stock_scores)}\n")
        report_lines.append(f"强因子股票数: {len(strong_factor_stocks)}\n\n")

        if len(strong_factor_stocks) > 0:
            report_lines.append("## 强因子股票列表\n")
            report_lines.append("| 排名 | 股票名称 | 代码 | 行业 | 因子强度 | 平均分 | 最高分 | 新闻数 | 确定性 |\n")
            report_lines.append("|------|----------|------|------|----------|--------|--------|--------|--------|\n")

            for idx, row in strong_factor_stocks.head(20).iterrows():
                report_lines.append(
                    f"| {idx+1} | {row['stock_name']} | {row['stock_code']} | {row['industry']} | "
                    f"{row['factor_strength']:.2f} | {row['avg_score']:.2f} | {row['max_score']:.2f} | "
                    f"{row['news_count']} | {row['avg_certainty']:.2f} |\n"
                )

            report_lines.append("\n## 详细新闻分析\n")

            # 为每只强因子股票显示相关新闻
            for _, stock_row in strong_factor_stocks.head(10).iterrows():
                stock_code = stock_row['stock_code']
                stock_news = results_df[results_df['stock_code'] == stock_code]

                report_lines.append(f"\n### {stock_row['stock_name']} ({stock_code})\n")
                report_lines.append(f"因子强度: {stock_row['factor_strength']:.2f}\n\n")

                for _, news_row in stock_news.iterrows():
                    report_lines.append(f"**{news_row['original_title']}**\n")
                    report_lines.append(f"- 时间: {news_row['original_date']}\n")
                    report_lines.append(f"- 综合评分: {news_row.get('overall_score', 'N/A')}\n")
                    report_lines.append(f"- 情感倾向: {news_row.get('sentiment', 'N/A')}\n")
                    report_lines.append(f"- 确定性: {news_row.get('certainty', 'N/A')}\n")
                    report_lines.append(f"- 建议操作: {news_row.get('action_suggestion', 'N/A')}\n\n")
        else:
            report_lines.append("## 今日无强因子股票\n")
            report_lines.append("所有股票的因子强度均低于3.0阈值\n\n")

        # 保存报告
        report_file = self.output_dir / f"factor_report_{target_date.replace('-', '')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)

        logger.info(f"因子强度报告已保存到 {report_file}")

        # 同时保存CSV格式的股票评分
        score_file = self.output_dir / f"stock_scores_{target_date.replace('-', '')}.csv"
        stock_scores.to_csv(score_file, index=False, encoding='utf-8-sig')
        logger.info(f"股票评分已保存到 {score_file}")

        return strong_factor_stocks

def main():
    """主函数"""
    # 导入配置
    try:
        import sys
        sys.path.append(str(Path(__file__).parent.parent / "config"))
        from api_config import TUSHARE_TOKEN, OPENROUTER_API_KEY
    except ImportError:
        logger.error("请在 config/api_config.py 中配置API密钥")
        return

    # 目标分析日期（默认为今天）
    target_date = datetime.now().strftime('%Y-%m-%d')

    try:
        # 初始化分析器
        analyzer = DailyNewsAnalyzer(TUSHARE_TOKEN, OPENROUTER_API_KEY)

        logger.info(f"开始分析 {target_date} 的新闻...")

        # 1. 获取当日新闻
        news_df = analyzer.get_daily_news(target_date)
        if news_df.empty:
            logger.warning("未获取到新闻数据，程序结束")
            return

        # 2. 匹配新闻到股票
        matched_df = analyzer.match_news_to_stocks(news_df)
        if matched_df.empty:
            logger.warning("未匹配到相关股票新闻，程序结束")
            return

        # 3. 大模型评分
        scored_df = analyzer.score_news_with_llm(matched_df)
        if scored_df.empty:
            logger.warning("评分失败，程序结束")
            return

        # 4. 保存结果
        analyzer.save_results(scored_df, target_date)

        # 5. 生成因子强度报告
        strong_stocks = analyzer.generate_factor_report(scored_df, target_date)

        logger.info("分析完成！")

        if not strong_stocks.empty:
            print(f"\n今日发现 {len(strong_stocks)} 只强因子股票：")
            for _, row in strong_stocks.head(5).iterrows():
                print(f"- {row['stock_name']} ({row['stock_code']}): 因子强度 {row['factor_strength']:.2f}")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()