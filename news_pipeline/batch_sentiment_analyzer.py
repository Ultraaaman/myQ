#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量新闻情感分析脚本（改进版）

功能说明：
- 读取period_news_fetcher.py抓取的新闻CSV文件
- 使用大模型进行批量情感分析和评分
- 支持按日期去重，避免重复分析已处理的新闻
- 增量写入分析结果，边分析边写入，保证数据安全
- 单日请求数量控制，避免超出API限额

使用方法：
    python batch_sentiment_analyzer.py --input news_20250901_20250930.csv

参数说明：
    --input: 输入的新闻CSV文件名（位于output/period_news目录）
    --batch_size: 单个请求的并行新闻条数，默认20（建议10-50，最大100）
    --daily_limit: 单日最大请求数量，默认950
    --content_limit: 单条新闻内容长度限制，默认1500字符

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
    def __init__(self, openrouter_api_key, model,daily_limit=950, content_limit=1500):
        """初始化情感分析器"""
        print(f"🔧 正在初始化情感分析器...")
        print(f"   模型: DeepSeek-V3.1 (128K tokens ≈ 20万汉字)")

        self.api_key = openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        }
        self.model=model

        # 设置文件路径
        self.base_dir = Path("D:/projects/q/myQ")
        self.input_dir = self.base_dir / "output" / "period_news"
        self.output_dir = self.base_dir / "output" / "analyzed_news"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 请求统计
        self.daily_limit = daily_limit
        self.request_count = 0
        self.content_limit = content_limit
        self.request_log_file = self.output_dir / f"request_log_{datetime.now().strftime('%Y%m%d')}.txt"
        self._load_request_count()

        print(f"✅ 初始化完成")
        print(f"   输入目录: {self.input_dir}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   单日请求上限: {self.daily_limit}")
        print(f"   内容长度限制: {self.content_limit} 字符")
        print(f"   今日已使用: {self.request_count} 次")

    def _load_request_count(self):
        """加载今日请求计数"""
        if self.request_log_file.exists():
            try:
                with open(self.request_log_file, 'r', encoding='utf-8') as f:
                    self.request_count = int(f.read().strip())
            except:
                self.request_count = 0
        else:
            self.request_count = 0

    def _save_request_count(self):
        """保存请求计数"""
        with open(self.request_log_file, 'w', encoding='utf-8') as f:
            f.write(str(self.request_count))

    def _check_request_limit(self):
        """检查是否达到请求上限"""
        return self.request_count < self.daily_limit

    def _increment_request_count(self):
        """增加请求计数"""
        self.request_count += 1
        self._save_request_count()

    def _suggest_batch_size(self, current_batch_size):
        """根据DeepSeek-V3.1的能力建议batch_size"""
        # 估算单条新闻token数（中文约 1.5-2 字符 = 1 token）
        avg_tokens_per_news = (self.content_limit // 1.5) + 200  # 内容 + 元数据

        # DeepSeek-V3.1: 128K context (约20万汉字)
        max_context_tokens = 128000
        base_prompt_tokens = 800  # 系统prompt
        response_tokens_reserve = 8000  # 预留输出空间

        # 可用于输入的tokens
        available_tokens = max_context_tokens - base_prompt_tokens - response_tokens_reserve

        # 计算理论最大batch_size
        theoretical_max = int(available_tokens // avg_tokens_per_news)

        # 实际推荐上限（考虑稳定性和输出质量）
        recommended_max = min(100, theoretical_max * 0.8)
        optimal_range = (20, 50)

        print(f"📊 Batch Size 分析（基于内容长度 {self.content_limit} 字符）:")
        print(f"   - 理论最大值: {theoretical_max}")
        print(f"   - 推荐上限: {int(recommended_max)}")

        if current_batch_size > recommended_max:
            print(f"   ⚠️ 当前值 {current_batch_size} 过大，建议 ≤ {int(recommended_max)}")
        elif current_batch_size < optimal_range[0]:
            print(f"   💡 当前值 {current_batch_size} 偏小，建议 {optimal_range[0]}-{optimal_range[1]} 以提高效率")
        elif current_batch_size <= optimal_range[1]:
            print(f"   ✅ 当前值 {current_batch_size} 设置合理")
        else:
            print(f"   ⚡ 当前值 {current_batch_size} 较大，充分利用长上下文能力")

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
        """检测并过滤重复新闻（基于日期去重，避免重复请求）"""
        if existing_df.empty:
            print(f"✓ 无历史数据，所有 {len(news_df)} 条新闻都将进行分析")
            return news_df

        print(f"🔍 开始检测重复新闻（按日期去重）...")

        # 提取日期（从 datetime 列中提取日期部分）
        def extract_date(df):
            if 'datetime' not in df.columns:
                return pd.Series([''] * len(df))

            # 尝试解析日期
            dates = pd.to_datetime(df['datetime'], errors='coerce')
            return dates.dt.strftime('%Y-%m-%d').fillna('')

        # 为新数据和已有数据提取日期
        news_df['date'] = extract_date(news_df)
        existing_df['date'] = extract_date(existing_df)

        # 创建检测键（日期 + 股票代码）
        news_df['check_key'] = news_df['date'] + '|' + news_df['stock_code'].astype(str)
        existing_df['check_key'] = existing_df['date'] + '|' + existing_df['stock_code'].astype(str)

        # 检测重复（按日期+股票代码）
        existing_keys = set(existing_df['check_key'].tolist())
        news_df['is_duplicate'] = news_df['check_key'].isin(existing_keys)

        duplicate_count = news_df['is_duplicate'].sum()
        new_count = (~news_df['is_duplicate']).sum()

        # 统计重复的日期
        if duplicate_count > 0:
            duplicate_dates = news_df[news_df['is_duplicate']]['date'].unique()
            print(f"📊 重复检测结果:")
            print(f"   - 总新闻数: {len(news_df)}")
            print(f"   - 已分析的日期: {len(duplicate_dates)} 天 (跳过 {duplicate_count} 条)")
            print(f"   - 待分析: {new_count} 条")
            print(f"💰 预计节省: {duplicate_count} 次LLM调用")
        else:
            print(f"📊 重复检测结果:")
            print(f"   - 总新闻数: {len(news_df)}")
            print(f"   - 待分析: {new_count} 条")

        # 返回未分析的新闻
        new_df = news_df[~news_df['is_duplicate']].copy()
        new_df = new_df.drop(['date', 'check_key', 'is_duplicate'], axis=1)

        return new_df

    def analyze_news(self, input_filename, output_filename, batch_size=20):
        """分析新闻情感"""
        print(f"\n🚀 开始新闻情感分析")

        # 检查请求限额
        remaining_requests = self.daily_limit - self.request_count
        if remaining_requests <= 0:
            print(f"❌ 今日请求已达上限 ({self.daily_limit})，请明天再试")
            return

        print(f"📊 今日剩余请求额度: {remaining_requests}/{self.daily_limit}")

        # 给出batch_size建议
        self._suggest_batch_size(batch_size)

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

        # 计算实际可处理的批次数
        max_batches = min(total_batches, remaining_requests)
        if max_batches < total_batches:
            print(f"   ⚠️ 由于请求限额，本次只能处理 {max_batches}/{total_batches} 批次")
            total_batches = max_batches

        print(f"   - 总批次数: {total_batches}\n")

        # 判断是否为第一次写入
        first_write = not output_path.exists() or existing_df.empty

        # 统计
        success_count = 0
        failed_count = 0

        # 逐批次处理
        for batch_idx in range(total_batches):
            # 再次检查请求限额
            if not self._check_request_limit():
                print(f"\n⚠️ 已达到今日请求上限 ({self.daily_limit})，停止分析")
                break

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(new_df))
            batch_df = new_df.iloc[start_idx:end_idx]

            print(f"📦 批次 {batch_idx + 1}/{total_batches} (新闻 {start_idx + 1}-{end_idx})")
            print(f"   📊 当前进度: {self.request_count}/{self.daily_limit} 请求")

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

                success_count += len(batch_results)
            else:
                print(f"   ❌ 批次分析失败")
                failed_count += len(batch_df)

            # 批次间延迟
            if batch_idx < total_batches - 1:
                wait_time = 2 + random.uniform(0, 1)
                print(f"   ⏰ 等待 {wait_time:.1f} 秒...\n")
                time.sleep(wait_time)

        print(f"\n🎉 分析完成！")
        print(f"💾 结果文件: {output_path}")
        print(f"\n📊 本次统计:")
        print(f"   - 成功分析: {success_count} 条")
        if failed_count > 0:
            print(f"   - 分析失败: {failed_count} 条")
        print(f"   - 使用请求: {self.request_count}/{self.daily_limit}")

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
        response = self._call_llm_api_with_retry(prompt, batch_df)

        if response:
            # 请求成功，增加计数
            self._increment_request_count()
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
            content = str(row.get('content', ''))[:self.content_limit]
            prompt += f"""
【新闻{idx+1}】
股票：{row['stock_name']}({row['stock_code']})
行业：{row.get('industry', 'N/A')}
标题：{str(row.get('title', ''))}
内容：{content}
时间：{row.get('datetime', '')}
"""

        prompt += """

重要要求：
1. 必须只返回JSON数组，不要有任何额外说明文字
2. 不要使用markdown代码块包裹JSON
3. JSON必须是有效格式，确保所有字段都有引号

请严格按照以下JSON数组格式返回（每个对象对应一条新闻）：
[{"news_index":1,"sentiment":"中性","direct_impact_score":0,"direct_impact_desc":"描述","indirect_impact_score":0,"indirect_impact_desc":"描述","certainty":0.5,"time_to_effect":"1周内","overall_score":0,"risk_factors":"风险","action_suggestion":"建议"}]

只返回JSON数组，不要有其他内容。
"""

        return prompt

    def _call_llm_api_with_retry(self, prompt, batch_df, max_retries=3):
        """调用LLM API（带重试机制）"""
        base_delay = 30

        for attempt in range(max_retries):
            try:
                print(f"   🔧 调用LLM API (第{attempt+1}次)...", end=" ")

                # 根据batch_size动态调整max_tokens
                # 每条新闻约需 250-350 tokens 的输出
                estimated_tokens = len(prompt.split()) * 1.5 + 350 * len(batch_df)
                max_tokens = min(16000, max(2000, int(estimated_tokens * 1.2)))

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": max_tokens
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
        """解析批量响应（增强容错）"""
        results = []

        try:
            # 保存原始响应用于调试
            debug_file = self.output_dir / f"debug_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            # 清理内容
            content = content.strip()

            # 移除可能的markdown代码块标记
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # 查找JSON数组的开始和结束
            start = content.find('[')
            end = content.rfind(']') + 1

            if start == -1 or end <= start:
                print(f"   ❌ 无法找到有效的JSON数组（可能被截断）")
                # 检查是否是截断问题
                if start != -1 and len(content) > 10000:
                    print(f"   ⚠️ 响应很长但缺少结束符，可能是batch_size过大导致")
                    print(f"   💡 建议: 减小 --batch_size 参数（当前建议5-10）")
                # 保存失败的响应用于调试
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"原始响应长度: {len(content)}\n")
                    f.write(f"查找结果: start={start}, end={end}\n")
                    f.write(f"响应内容:\n{content}\n")
                return []

            # 提取JSON字符串
            json_str = content[start:end]

            # 尝试解析JSON
            try:
                parsed_array = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON解析错误在位置 {e.pos}: {str(e)[:50]}")
                # 尝试修复常见问题
                json_str_fixed = json_str.replace("'", '"')  # 单引号改双引号
                json_str_fixed = json_str_fixed.replace('：', ':')  # 中文冒号改英文
                try:
                    parsed_array = json.loads(json_str_fixed)
                    print(f"   ✓ JSON修复成功")
                except:
                    # 保存失败的JSON用于调试
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(f"JSON解析失败\n错误: {e}\n\n")
                        f.write(f"JSON内容:\n{json_str[:2000]}\n")
                    return []

            if not isinstance(parsed_array, list):
                print(f"   ❌ 响应不是数组格式")
                return []

            # 处理解析结果
            for i, analysis_result in enumerate(parsed_array):
                if i >= len(batch_df):
                    break

                row = batch_df.iloc[i]
                result = row.to_dict()
                result['analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 验证并填充必要字段
                required_fields = {
                    'sentiment': '中性',
                    'overall_score': 0,
                    'certainty': 0.5,
                    'direct_impact_score': 0,
                    'indirect_impact_score': 0,
                    'direct_impact_desc': '',
                    'indirect_impact_desc': '',
                    'time_to_effect': '未知',
                    'risk_factors': '',
                    'action_suggestion': ''
                }

                for field, default_value in required_fields.items():
                    if field not in analysis_result:
                        analysis_result[field] = default_value

                result.update(analysis_result)
                results.append(result)

            print(f"   📝 解析成功: {len(results)}/{len(batch_df)} 条")
            return results

        except Exception as e:
            print(f"   ❌ 解析出错: {str(e)[:100]}")
            # 保存异常信息
            debug_file = self.output_dir / f"debug_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"异常: {e}\n\n")
                f.write(f"响应内容:\n{content[:2000] if 'content' in locals() else 'N/A'}\n")
            return []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量新闻情感分析工具（改进版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 使用默认参数（推荐配置）
  python batch_sentiment_analyzer.py --input news_20250901_20250930.csv

  # 极速模式：充分利用DeepSeek-V3.1的128K上下文（20万汉字）
  python batch_sentiment_analyzer.py --input news_20250901_20250930.csv --batch_size 50

  # 超大批次模式（适合大量新闻）
  python batch_sentiment_analyzer.py --input news_20250901_20250930.csv --batch_size 80

  # 高质量模式：更长内容 + 中等批次
  python batch_sentiment_analyzer.py --input news_20250901_20250930.csv --batch_size 30 --content_limit 2000

说明：
  - batch_size：单个请求处理的新闻条数，默认20，建议10-50（最大100）
  - daily_limit：单日最大请求数量，默认950，避免超出API限额
  - content_limit：单条新闻内容长度限制，默认1500字符
  - 支持按日期去重，避免重复分析同一天的新闻
  - 边分析边写入，支持增量追加
  - 优化适配DeepSeek-V3.1的128K上下文能力（约20万汉字）
        """
    )
    parser.add_argument('--input', type=str, required=True, help='输入的新闻CSV文件名')
    parser.add_argument('--batch_size', type=int, default=10, help='单个请求的并行新闻条数，默认5（GLM模型建议3-10）')
    parser.add_argument('--daily_limit', type=int, default=950, help='单日最大请求数量，默认950')
    parser.add_argument('--content_limit', type=int, default=1500, help='单条新闻内容长度限制（字符），默认1500')

    args = parser.parse_args()

    # 验证参数
    if args.batch_size < 1:
        print(f"❌ batch_size必须大于0")
        return
    elif args.batch_size > 100:
        print(f"⚠️ batch_size={args.batch_size} 超过最大限制100")
        print(f"   强制设置为100")
        args.batch_size = 100

    if args.daily_limit < 1:
        print(f"❌ daily_limit必须大于0")
        return

    if args.content_limit < 100:
        print(f"⚠️ content_limit={args.content_limit} 过小，建议至少300字符")
    elif args.content_limit > 3000:
        print(f"⚠️ content_limit={args.content_limit} 过大，可能影响效率")

    # 导入配置
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "config"))
    from api_config import OPENROUTER_API_KEY,DEFAULT_MODEL

    print(f"🔑 配置检查:")
    print(f"   - OpenRouter Key: {'已配置' if OPENROUTER_API_KEY != 'your_openrouter_api_key_here' else '未配置'}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Daily Limit: {args.daily_limit}")
    print(f"   - Content Limit: {args.content_limit} 字符")

    # 初始化分析器
    analyzer = BatchSentimentAnalyzer(
        OPENROUTER_API_KEY,
        DEFAULT_MODEL,
        daily_limit=args.daily_limit,
        content_limit=args.content_limit
    )

    # 生成输出文件名
    input_name = args.input.replace('.csv', '')
    output_filename = f"{input_name}_analyzed.csv"

    # 执行分析
    analyzer.analyze_news(args.input, output_filename, args.batch_size)


if __name__ == "__main__":
    main()
