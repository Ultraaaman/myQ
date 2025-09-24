import requests
import json
import pandas as pd
import os
from datetime import datetime
import logging
import time
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsScorer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 政策分析prompt模板
        self.policy_prompt = """
你是一位专业的金融分析师，请分析以下政策新闻对个股的影响。

背景信息：
- 股票：{stock_name}({stock_code})
- 所属行业：{industry}
- 主营业务：{main_business}
- 当前市值：{market_cap}

政策新闻：
标题：{news_title}
内容：{news_content}
发布时间：{publish_time}
消息来源：{source}

请从以下维度分析：

1. 直接影响评估
   - 政策是否直接提及该公司或其主营业务？
   - 政策对公司收入/成本/利润的潜在影响？

2. 间接影响评估
   - 对公司上下游产业链的影响？
   - 对竞争格局的影响？
   - 对行业整体的影响？

3. 影响时间判断
   - 政策落地的确定性如何？
   - 影响显现需要多长时间？

请严格按照以下JSON格式输出，不要添加任何其他文字：

{{
  "sentiment": "强烈正面|正面|中性偏正|中性|中性偏负|负面|强烈负面",
  "direct_impact": {{
    "score": {direct_score},
    "description": "直接影响说明"
  }},
  "indirect_impact": {{
    "score": {indirect_score},
    "description": "间接影响说明"
  }},
  "certainty": {certainty_score},
  "time_to_effect": "立即|1个月内|1-3个月|3-6个月|6个月以上",
  "overall_score": {overall_score},
  "risk_factors": ["风险点1", "风险点2"],
  "action_suggestion": "强烈关注|关注|观望|谨慎|回避"
}}

注意：
- direct_score和indirect_score范围：-5到+5
- overall_score范围：-10到+10
- certainty范围：0.0到1.0
- 只输出JSON，不要添加解释文字
"""

        # 批量分析prompt模板
        self.batch_policy_prompt = """
你是一位专业的金融分析师，请分析以下多条政策新闻对个股的影响。

背景信息：
- 股票：{stock_name}({stock_code})
- 所属行业：{industry}
- 主营业务：{main_business}
- 当前市值：{market_cap}

重要提示：以下是{batch_count}条独立的新闻，请独立分析每条新闻，不要考虑新闻之间的关联性。每条新闻都需要独立评估其对该股票的影响。

{news_list}

请从以下维度分析每条新闻：

1. 直接影响评估
   - 政策是否直接提及该公司或其主营业务？
   - 政策对公司收入/成本/利润的潜在影响？

2. 间接影响评估
   - 对公司上下游产业链的影响？
   - 对竞争格局的影响？
   - 对行业整体的影响？

3. 影响时间判断
   - 政策落地的确定性如何？
   - 影响显现需要多长时间？

请严格按照以下JSON格式输出，返回一个包含{batch_count}个分析结果的数组，顺序对应输入的新闻顺序：

[
{{
  "news_index": 1,
  "sentiment": "强烈正面|正面|中性偏正|中性|中性偏负|负面|强烈负面",
  "direct_impact": {{
    "score": 数值(-5到+5),
    "description": "直接影响说明"
  }},
  "indirect_impact": {{
    "score": 数值(-5到+5),
    "description": "间接影响说明"
  }},
  "certainty": 数值(0.0到1.0),
  "time_to_effect": "立即|1个月内|1-3个月|3-6个月|6个月以上",
  "overall_score": 数值(-10到+10),
  "risk_factors": ["风险点1", "风险点2"],
  "action_suggestion": "强烈关注|关注|观望|谨慎|回避"
}},
{{
  "news_index": 2,
  ...
}}
]

注意：
- 必须返回{batch_count}个分析结果的JSON数组
- 每个结果的news_index对应输入新闻的序号
- 请独立分析每条新闻，给出客观评分
- 只输出JSON数组，不要添加解释文字
"""

    def call_llm(self, prompt, max_retries=3):
        """调用大模型API（添加指数退避重试）"""
        base_delay = 20  # 基础延迟时间（秒）

        for attempt in range(max_retries):
            try:
                data = {
                    "model": "deepseek/deepseek-chat-v3.1:free",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,  # 降低随机性，提高一致性
                    "max_tokens": 4000  # 增加token数量以支持批量处理
                }

                response = requests.post(
                    url=self.api_url,
                    headers=self.headers,
                    data=json.dumps(data),
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    return content
                elif response.status_code == 429:
                    # 429错误使用更长的等待时间
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"API速率限制(429)，等待 {wait_time:.2f} 秒后重试 {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    # 其他错误使用较短的等待时间
                    wait_time = base_delay * (1.5 ** attempt)
                    logger.warning(f"API请求失败，状态码: {response.status_code}, 等待 {wait_time:.2f} 秒后重试 {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)

            except Exception as e:
                wait_time = base_delay * (2 ** attempt)
                logger.error(f"API调用出错: {e}, 等待 {wait_time:.2f} 秒后重试 {attempt + 1}/{max_retries}")
                time.sleep(wait_time)

        return None

    def parse_llm_response(self, response_text):
        """解析大模型返回的JSON"""
        try:
            # 清理响应文本
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # 解析JSON
            result = json.loads(response_text)

            # 验证必要字段
            required_fields = ['sentiment', 'direct_impact', 'indirect_impact',
                             'certainty', 'time_to_effect', 'overall_score',
                             'risk_factors', 'action_suggestion']

            for field in required_fields:
                if field not in result:
                    logger.warning(f"缺少必要字段: {field}")
                    return None

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始响应: {response_text}")
            return None
        except Exception as e:
            logger.error(f"响应解析出错: {e}")
            return None

    def parse_batch_response(self, response_text, news_list):
        """解析大模型返回的批量JSON响应"""
        try:
            # 清理响应文本
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # 解析JSON数组
            batch_results = json.loads(response_text)

            if not isinstance(batch_results, list):
                logger.error("批量响应不是JSON数组格式")
                return None

            # 验证结果数量
            if len(batch_results) != len(news_list):
                logger.warning(f"返回结果数量({len(batch_results)})与输入新闻数量({len(news_list)})不匹配")

            # 处理每个结果
            processed_results = []
            required_fields = ['sentiment', 'direct_impact', 'indirect_impact',
                             'certainty', 'time_to_effect', 'overall_score',
                             'risk_factors', 'action_suggestion']

            for i, result in enumerate(batch_results):
                try:
                    # 验证必要字段
                    for field in required_fields:
                        if field not in result:
                            logger.warning(f"批量结果 {i+1} 缺少必要字段: {field}")
                            continue

                    # 添加原始新闻信息
                    if i < len(news_list):
                        news_row = news_list[i]
                        result['original_title'] = news_row.get('title', '')
                        result['original_date'] = news_row.get('date', '')
                        result['original_source'] = news_row.get('source', '')
                        result['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    processed_results.append(result)

                except Exception as e:
                    logger.error(f"处理批量结果 {i+1} 时出错: {e}")
                    continue

            logger.info(f"成功解析 {len(processed_results)} 个批量结果")
            return processed_results

        except json.JSONDecodeError as e:
            logger.error(f"批量JSON解析失败: {e}")
            logger.error(f"原始响应: {response_text}")
            return None
        except Exception as e:
            logger.error(f"批量响应解析出错: {e}")
            return None

    def score_news(self, news_row, stock_info):
        """对单条新闻进行评分"""
        try:
            # 构建prompt
            prompt = self.policy_prompt.format(
                stock_name=stock_info.get('stock_name', ''),
                stock_code=stock_info.get('stock_code', ''),
                industry=stock_info.get('industry', ''),
                main_business=stock_info.get('main_business', ''),
                market_cap=stock_info.get('market_cap', ''),
                news_title=news_row.get('title', ''),
                news_content=news_row.get('content', ''),
                publish_time=news_row.get('date', ''),
                source=news_row.get('source', ''),
                direct_score=0,  # 占位符
                indirect_score=0,  # 占位符
                certainty_score=0.5,  # 占位符
                overall_score=0  # 占位符
            )

            # 调用大模型
            logger.info(f"正在分析新闻: {news_row.get('title', '')[:50]}...")
            response = self.call_llm(prompt)

            if response is None:
                logger.error("大模型调用失败")
                return None

            # 解析响应
            parsed_result = self.parse_llm_response(response)

            if parsed_result is None:
                logger.error("响应解析失败")
                return None

            # 添加原始新闻信息
            parsed_result['original_title'] = news_row.get('title', '')
            parsed_result['original_date'] = news_row.get('date', '')
            parsed_result['original_source'] = news_row.get('source', '')
            parsed_result['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            logger.info(f"✓ 分析完成，综合得分: {parsed_result.get('overall_score', 0)}")
            return parsed_result

        except Exception as e:
            logger.error(f"新闻评分出错: {e}")
            return None

    def score_news_batch(self, news_list, stock_info, batch_size=4):
        """批量对多条新闻进行评分"""
        try:
            # 构建新闻列表字符串
            news_items = []
            for i, news_row in enumerate(news_list, 1):
                news_item = f"""
新闻 {i}：
标题：{news_row.get('title', '')}
内容：{news_row.get('content', '')}
发布时间：{news_row.get('date', '')}
消息来源：{news_row.get('source', '')}
"""
                news_items.append(news_item)

            news_list_str = "\n".join(news_items)

            # 构建批量prompt
            prompt = self.batch_policy_prompt.format(
                stock_name=stock_info.get('stock_name', ''),
                stock_code=stock_info.get('stock_code', ''),
                industry=stock_info.get('industry', ''),
                main_business=stock_info.get('main_business', ''),
                market_cap=stock_info.get('market_cap', ''),
                batch_count=len(news_list),
                news_list=news_list_str
            )

            # 调用大模型
            logger.info(f"正在批量分析 {len(news_list)} 条新闻...")
            response = self.call_llm(prompt)

            if response is None:
                logger.error("大模型调用失败")
                return None

            # 解析批量响应
            parsed_results = self.parse_batch_response(response, news_list)

            if parsed_results is None:
                logger.error("批量响应解析失败")
                return None

            logger.info(f"✓ 批量分析完成，共处理 {len(parsed_results)} 条新闻")
            return parsed_results

        except Exception as e:
            logger.error(f"批量新闻评分出错: {e}")
            return None

    def get_analyzed_dates(self, output_file):
        """获取已分析的新闻日期列表"""
        analyzed_dates = set()
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file, encoding='utf-8-sig')
                if 'original_date' in existing_df.columns:
                    # 提取日期部分（忽略时间）
                    existing_df['date_only'] = pd.to_datetime(existing_df['original_date']).dt.date
                    analyzed_dates = set(existing_df['date_only'].astype(str))
                    logger.info(f"发现已分析的日期: {len(analyzed_dates)} 个")
            except Exception as e:
                logger.warning(f"读取已有结果文件失败: {e}")
        return analyzed_dates

    def filter_unanalyzed_news(self, df, analyzed_dates):
        """过滤掉已分析日期的新闻"""
        if not analyzed_dates:
            return df

        # 转换新闻日期格式
        df['date_only'] = pd.to_datetime(df['date']).dt.date.astype(str)

        # 过滤未分析的新闻
        unanalyzed_df = df[~df['date_only'].isin(analyzed_dates)].copy()
        unanalyzed_df = unanalyzed_df.drop('date_only', axis=1)

        filtered_count = len(df) - len(unanalyzed_df)
        if filtered_count > 0:
            logger.info(f"过滤掉已分析的新闻: {filtered_count} 条")
            logger.info(f"待分析新闻: {len(unanalyzed_df)} 条")

        return unanalyzed_df

    def append_results(self, new_results_df, output_file):
        """追加结果到文件"""
        if new_results_df.empty:
            logger.info("没有新结果需要保存")
            return

        if os.path.exists(output_file):
            try:
                # 读取现有数据
                existing_df = pd.read_csv(output_file, encoding='utf-8-sig')

                # 合并数据
                combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)

                # 按时间排序
                combined_df['sort_date'] = pd.to_datetime(combined_df['original_date'])
                combined_df = combined_df.sort_values('sort_date', ascending=False).drop('sort_date', axis=1)
                combined_df = combined_df.reset_index(drop=True)

                # 保存
                combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"结果已追加到 {output_file}")
                logger.info(f"原有数据: {len(existing_df)} 条，新增数据: {len(new_results_df)} 条，总计: {len(combined_df)} 条")

            except Exception as e:
                logger.error(f"追加失败，改为覆盖模式: {e}")
                new_results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"结果已保存到 {output_file}")
        else:
            # 文件不存在，直接保存
            new_results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"结果已保存到 {output_file}")

    def process_news_batch(self, csv_file, stock_info, output_file='news_scores.csv', batch_size=5):
        """批量处理新闻评分（智能去重）"""
        try:
            # 读取新闻数据
            logger.info(f"读取新闻数据: {csv_file}")
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            logger.info(f"共读取到 {len(df)} 条新闻")

            # 检查必要列
            required_columns = ['date', 'content']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件缺少必要列: {missing_columns}")

            # 添加可选列的默认值
            if 'title' not in df.columns:
                df['title'] = df['content'].str[:100] + '...'
            if 'source' not in df.columns:
                df['source'] = '未知来源'

            # 获取已分析的日期
            analyzed_dates = self.get_analyzed_dates(output_file)

            # 过滤未分析的新闻
            df_to_analyze = self.filter_unanalyzed_news(df, analyzed_dates)

            if df_to_analyze.empty:
                logger.info("所有新闻都已分析过，无需重复处理")
                return pd.DataFrame()

            # 存储评分结果
            scored_results = []

            # 处理新闻（使用批量处理逻辑）
            batch_size = 4  # 每批处理4条新闻
            total_batches = (len(df_to_analyze) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(df_to_analyze))

                # 获取当前批次的新闻
                current_batch = df_to_analyze.iloc[start_idx:end_idx]
                batch_news_list = [row.to_dict() for _, row in current_batch.iterrows()]

                logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批，包含 {len(batch_news_list)} 条新闻")

                # 批量评分
                batch_results = self.score_news_batch(batch_news_list, stock_info, batch_size)

                # 添加批次间隔，避免API速率限制
                if batch_idx < total_batches - 1:  # 最后一批不需要等待
                    wait_time = 2 + random.uniform(0, 1)  # 2-3秒间隔
                    logger.info(f"等待 {wait_time:.1f} 秒后处理下一批...")
                    time.sleep(wait_time)

                if batch_results:
                    # 处理批量结果
                    batch_flattened_results = []
                    for i, score_result in enumerate(batch_results):
                        try:
                            # 展平结果用于DataFrame
                            flattened_result = {
                                'news_id': start_idx + i + 1,
                                'original_title': score_result['original_title'],
                                'original_date': score_result['original_date'],
                                'original_source': score_result['original_source'],
                                'analysis_time': score_result['analysis_time'],
                                'sentiment': score_result['sentiment'],
                                'direct_impact_score': score_result['direct_impact']['score'],
                                'direct_impact_desc': score_result['direct_impact']['description'],
                                'indirect_impact_score': score_result['indirect_impact']['score'],
                                'indirect_impact_desc': score_result['indirect_impact']['description'],
                                'certainty': score_result['certainty'],
                                'time_to_effect': score_result['time_to_effect'],
                                'overall_score': score_result['overall_score'],
                                'risk_factors': '|'.join(score_result.get('risk_factors', [])),
                                'action_suggestion': score_result['action_suggestion']
                            }
                            batch_flattened_results.append(flattened_result)
                            scored_results.append(flattened_result)
                        except Exception as e:
                            logger.error(f"处理批量结果 {i+1} 时出错: {e}")
                            continue

                    # 每个批次处理完后立即写入文件
                    if batch_flattened_results:
                        batch_df = pd.DataFrame(batch_flattened_results)
                        self.append_results(batch_df, output_file)
                        logger.info(f"✓ 第 {batch_idx + 1} 批结果已保存，共 {len(batch_flattened_results)} 条")
                else:
                    logger.warning(f"第 {batch_idx + 1} 批新闻评分失败，尝试单条处理")
                    # 如果批量处理失败，回退到单条处理
                    fallback_results = []
                    for i, news_row in enumerate(batch_news_list):
                        score_result = self.score_news(news_row, stock_info)
                        if score_result:
                            flattened_result = {
                                'news_id': start_idx + i + 1,
                                'original_title': score_result['original_title'],
                                'original_date': score_result['original_date'],
                                'original_source': score_result['original_source'],
                                'analysis_time': score_result['analysis_time'],
                                'sentiment': score_result['sentiment'],
                                'direct_impact_score': score_result['direct_impact']['score'],
                                'direct_impact_desc': score_result['direct_impact']['description'],
                                'indirect_impact_score': score_result['indirect_impact']['score'],
                                'indirect_impact_desc': score_result['indirect_impact']['description'],
                                'certainty': score_result['certainty'],
                                'time_to_effect': score_result['time_to_effect'],
                                'overall_score': score_result['overall_score'],
                                'risk_factors': '|'.join(score_result.get('risk_factors', [])),
                                'action_suggestion': score_result['action_suggestion']
                            }
                            fallback_results.append(flattened_result)
                            scored_results.append(flattened_result)

                    # 单条处理的结果也立即写入
                    if fallback_results:
                        fallback_df = pd.DataFrame(fallback_results)
                        self.append_results(fallback_df, output_file)
                        logger.info(f"✓ 第 {batch_idx + 1} 批单条处理结果已保存，共 {len(fallback_results)} 条")

            # 最终统计
            if scored_results:
                results_df = pd.DataFrame(scored_results)
                logger.info(f"✓ 批量处理完成，总共成功处理 {len(results_df)} 条新闻")

                # 显示统计信息
                self.show_statistics(results_df)

                return results_df
            else:
                logger.error("没有成功处理的新闻")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"批量处理出错: {e}")
            return pd.DataFrame()

    def show_statistics(self, df):
        """显示评分统计信息"""
        logger.info("\n=== 本次分析统计 ===")
        logger.info(f"本次分析新闻数: {len(df)}")
        if len(df) > 0:
            logger.info(f"平均综合得分: {df['overall_score'].mean():.2f}")
            logger.info(f"情绪分布:")
            sentiment_counts = df['sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                logger.info(f"  {sentiment}: {count} 条")

            logger.info(f"行动建议分布:")
            action_counts = df['action_suggestion'].value_counts()
            for action, count in action_counts.items():
                logger.info(f"  {action}: {count} 条")

# 使用示例
if __name__ == "__main__":
    # API配置
    openrouter_key = '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'

    # 创建评分器
    scorer = NewsScorer(openrouter_key)

    # 股票信息
    # stock_info = {
    #     'stock_name': '紫金矿业',
    #     'stock_code': '601899',
    #     'industry': '有色金属',
    #     'main_business': '黄金、铜等有色金属的开采、选矿、冶炼及销售',
    #     'market_cap': '3000亿元'
    # }
    stock_info={
      "stock_name": "盐津铺子",
      "stock_code": "002847",
      "industry": "休闲食品",
      "main_business": "烘焙食品、辣条及其他休闲食品的研发、生产、销售",
      "market_cap": "150亿元"
    }

    # 处理新闻评分（智能去重）
    results_df = scorer.process_news_batch(
        csv_file='tushare_news_yanjinpuzi.csv',
        stock_info=stock_info,
        output_file='news_scores_result_1y_yanjin.csv',
        batch_size=5  # 参数保留但不再使用批次逻辑
    )

    if not results_df.empty:
        print("\n=== 处理完成 ===")
        print(f"本次成功分析 {len(results_df)} 条新闻")
        print("结果文件: news_scores_result.csv")

        # 显示前几条结果
        print("\n=== 前3条评分结果 ===")
        for idx, row in results_df.head(3).iterrows():
            print(f"\n{idx+1}. {row['original_title'][:50]}...")
            print(f"   情绪: {row['sentiment']}")
            print(f"   综合得分: {row['overall_score']}")
            print(f"   建议: {row['action_suggestion']}")
    else:
        print("没有新新闻需要分析，或处理失败")