#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量新闻情感分析脚本

功能说明：
- 读取period_news_fetcher.py抓取的新闻CSV文件
- 使用大模型进行批量情感分析和评分
- 支持重复检测，避免重复分析已处理的新闻
- 增量写入分析结果，保证数据安全

使用方法：
    python batch_sentiment_analyzer.py --input news_20250901_20250930.csv

参数说明：
    --input: 输入的新闻CSV文件名（位于output/period_news目录）
    --batch_size: 批次大小，默认4条新闻一批

输出文件：
    news_20250901_20250930_analyzed.csv - 分析结果文件
"""

import pandas as pd
import json
import requests
import time
import random
from datetime import datetime
from pathlib import Path
import argparse


class BatchSentimentAnalyzer:
    def __init__(self, openrouter_api_key):
        """初始化情感分析器"""
        print(f"🔧 正在初始化情感分析器...")

        self.api_key = openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }

        # 设置文件路径
        self.base_dir = Path("D:/projects/q/myQ")
        self.input_dir = self.base_dir / "output" / "period_news"
        self.output_dir = self.base_dir / "output" / "analyzed_news"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ 初始化完成")
        print(f"   输入目录: {self.input_dir}")
        print(f"   输出目录: {self.output_dir}")

    def load_news_csv(self, input_filename):
        """加载新闻CSV文件"""
        input_path = self.input_dir / input_filename
        print(f"\n📁 读取新闻文件: {input_path}")

        if not input_path.exists():
            print(f"❌ 文件不存在: {input_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(input_path, encoding='utf-8-sig')
            print(f"✅ 成功读取 {len(df)} 条新闻")
            return df
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return pd.DataFrame()

    def _load_existing_results(self, output_filename):
        """加载已存在的分析结果"""
        output_path = self.output_dir / output_filename

        if output_path.exists():
            try:
                existing_df = pd.read_csv(output_path, encoding='utf-8-sig')
                print(f"📁 发现已有分析结果: {len(existing_df)} 条记录")
                return existing_df
            except Exception as e:
                print(f"⚠️ 读取已有结果失败: {e}")
                return pd.DataFrame()
        else:
            print(f"📁 未发现已有分析结果，将创建新文件")
            return pd.DataFrame()

    def _detect_duplicates(self, news_df, existing_df):
        """检测并过滤重复新闻（基于内容指纹）"""
        if existing_df.empty:
            print(f"✓ 无历史数据，所有 {len(news_df)} 条新闻都将进行分析")
            return news_df

        print(f"🔍 开始检测重复新闻...")

        # 创建内容指纹
        def create_content_fingerprint(df):
            if 'content' in df.columns:
                content_col = 'content'
            elif 'title' in df.columns:
                content_col = 'title'
            else:
                return pd.Series([''] * len(df))

            cleaned = df[content_col].fillna('').astype(str).str.strip()
            cleaned = cleaned.str.replace('\n', ' ', regex=False).str.replace('\r', '', regex=False).str.replace('\t', ' ', regex=False)
            cleaned = cleaned.str.replace(r'\d{4}-\d{2}-\d{2}', '', regex=True)
            cleaned = cleaned.str.replace(r'\d{1,2}:\d{2}(:\d{2})?', '', regex=True)
            cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
            fingerprint = cleaned.str[:150]
            return fingerprint

        # 为新数据和已有数据创建指纹
        news_df['fingerprint'] = create_content_fingerprint(news_df)
        existing_df['fingerprint'] = create_content_fingerprint(existing_df)

        # 创建检测键（内容指纹 + 股票代码）
        news_df['check_key'] = news_df['fingerprint'] + '|' + news_df['stock_code'].astype(str)
        existing_df['check_key'] = existing_df['fingerprint'] + '|' + existing_df['stock_code'].astype(str)

        # 检测重复
        existing_keys = set(existing_df['check_key'].tolist())
        news_df['is_duplicate'] = news_df['check_key'].isin(existing_keys)

        duplicate_count = news_df['is_duplicate'].sum()
        new_count = (~news_df['is_duplicate']).sum()

        print(f"📊 重复检测结果:")
        print(f"   - 总新闻数: {len(news_df)}")
        print(f"   - 已分析过: {duplicate_count} 条 (跳过)")
        print(f"   - 待分析: {new_count} 条")

        if duplicate_count > 0:
            print(f"💰 预计节省: {duplicate_count} 次LLM调用")

        # 返回未分析的新闻
        new_df = news_df[~news_df['is_duplicate']].copy()
        new_df = new_df.drop(['fingerprint', 'check_key', 'is_duplicate'], axis=1)

        return new_df

    def analyze_news(self, input_filename, output_filename, batch_size=4):
        """分析新闻情感"""
        print(f"\n🚀 开始新闻情感分析")

        # 加载新闻数据
        news_df = self.load_news_csv(input_filename)
        if news_df.empty:
            print("❌ 没有可分析的新闻数据")
            return

        # 加载已有分析结果
        existing_df = self._load_existing_results(output_filename)

        # 检测重复
        new_df = self._detect_duplicates(news_df, existing_df)

        if new_df.empty:
            print(f"\n🎉 所有新闻都已分析过，无需重复API调用！")
            return

        print(f"\n🤖 开始批量LLM分析...")
        print(f"   - 待分析: {len(new_df)} 条")
        print(f"   - 批次大小: {batch_size}")

        output_path = self.output_dir / output_filename
        total_batches = (len(new_df) + batch_size - 1) // batch_size
        print(f"   - 总批次数: {total_batches}\n")

        # 判断是否为第一次写入
        first_write = not output_path.exists() or existing_df.empty

        # 逐批次处理
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(new_df))
            batch_df = new_df.iloc[start_idx:end_idx]

            print(f"📦 批次 {batch_idx + 1}/{total_batches} (新闻 {start_idx + 1}-{end_idx})")

            # 调用LLM分析
            batch_results = self._analyze_batch(batch_df)

            if batch_results:
                results_df = pd.DataFrame(batch_results)

                # 立即写入文件
                if first_write:
                    results_df.to_csv(output_path, mode='w', index=False, encoding='utf-8-sig')
                    first_write = False
                    print(f"   ✅ 已写入 {len(batch_results)} 条结果（创建文件）")
                else:
                    results_df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                    print(f"   ✅ 已追加 {len(batch_results)} 条结果")
            else:
                print(f"   ❌ 批次分析失败")

            # 批次间延迟
            if batch_idx < total_batches - 1:
                wait_time = 2 + random.uniform(0, 1)
                print(f"   ⏰ 等待 {wait_time:.1f} 秒...\n")
                time.sleep(wait_time)

        print(f"\n🎉 分析完成！")
        print(f"💾 结果文件: {output_path}")

        # 统计最终结果
        try:
            final_df = pd.read_csv(output_path, encoding='utf-8-sig')
            print(f"📊 总分析数: {len(final_df)}")
            print(f"📊 涉及股票: {final_df['stock_code'].nunique()}")
        except:
            pass

    def _analyze_batch(self, batch_df):
        """批量分析一批新闻"""
        prompt = self._build_batch_prompt(batch_df)
        response = self._call_llm_api_with_retry(prompt)

        if response:
            return self._parse_batch_response(response, batch_df)
        else:
            return []

    def _build_batch_prompt(self, batch_df):
        """构建批量分析的prompt"""
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
            content = str(row.get('content', ''))[:600]
            prompt += f"""
【新闻{idx+1}】
股票：{row['stock_name']}({row['stock_code']})
行业：{row.get('industry', 'N/A')}
标题：{str(row.get('title', ''))}
内容：{content}
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
        """调用LLM API（带重试机制）"""
        base_delay = 20

        for attempt in range(max_retries):
            try:
                print(f"   🔧 调用LLM API (第{attempt+1}次)...", end=" ")

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

                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    print(f"✓")
                    return content

                elif response.status_code == 429:
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ 速率限制(429)")
                    print(f"   等待 {wait_time:.1f}秒后重试...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    continue

                else:
                    wait_time = base_delay * (1.5 ** attempt)
                    print(f"⚠️ 失败(状态:{response.status_code})")
                    print(f"   等待 {wait_time:.1f}秒后重试...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    continue

            except Exception as e:
                wait_time = base_delay * (2 ** attempt)
                print(f"❌ 异常: {str(e)[:50]}")
                print(f"   等待 {wait_time:.1f}秒后重试...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue

        print(f"   ❌ API调用失败，已重试 {max_retries} 次")
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
                parsed_array = json.loads(json_str)

                if isinstance(parsed_array, list):
                    for i, analysis_result in enumerate(parsed_array):
                        if i < len(batch_df):
                            row = batch_df.iloc[i]
                            result = row.to_dict()
                            result['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            # 验证必要字段
                            required_fields = ['sentiment', 'overall_score', 'certainty']
                            for field in required_fields:
                                if field not in analysis_result:
                                    analysis_result[field] = 0 if field in ['overall_score', 'certainty'] else "中性"

                            result.update(analysis_result)
                            results.append(result)

                    print(f"   📝 解析成功: {len(results)} 条")
                    return results
            else:
                print(f"   ❌ 无法找到有效的JSON数组")
                return []

        except json.JSONDecodeError as e:
            print(f"   ❌ JSON解析错误: {e}")
            return []
        except Exception as e:
            print(f"   ❌ 解析出错: {e}")
            return []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量新闻情感分析工具')
    parser.add_argument('--input', type=str, required=True, help='输入的新闻CSV文件名')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小，默认4')

    args = parser.parse_args()

    # 导入配置
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "config"))
    from api_config import OPENROUTER_API_KEY

    print(f"🔑 配置检查:")
    print(f"   - OpenRouter Key: {'已配置' if OPENROUTER_API_KEY != 'your_openrouter_api_key_here' else '未配置'}")

    # 初始化分析器
    analyzer = BatchSentimentAnalyzer(OPENROUTER_API_KEY)

    # 生成输出文件名
    input_name = args.input.replace('.csv', '')
    output_filename = f"{input_name}_analyzed.csv"

    # 执行分析
    analyzer.analyze_news(args.input, output_filename, args.batch_size)


if __name__ == "__main__":
    main()
