#!/bin/bash
# 新闻分析流水线快捷脚本

echo "========================================"
echo "   新闻数据处理流水线"
echo "========================================"
echo ""

# 检查参数
if [ $# -lt 2 ]; then
    echo "使用方法:"
    echo "  ./run_pipeline.sh START_DATE END_DATE"
    echo ""
    echo "示例:"
    echo "  ./run_pipeline.sh 2025-09-01 2025-09-30"
    echo ""
    exit 1
fi

START_DATE=$1
END_DATE=$2
FILENAME="news_${START_DATE//-/}_${END_DATE//-/}.csv"

echo "配置信息:"
echo "  开始日期: $START_DATE"
echo "  结束日期: $END_DATE"
echo "  输出文件: $FILENAME"
echo ""

echo "----------------------------------------"
echo "步骤 1/2: 抓取新闻数据"
echo "----------------------------------------"
python period_news_fetcher.py --start_date "$START_DATE" --end_date "$END_DATE"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 新闻抓取失败！"
    exit 1
fi

echo ""
echo "----------------------------------------"
echo "步骤 2/2: 情感分析"
echo "----------------------------------------"
read -p "是否继续进行情感分析? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo ""
    echo "⏸️  已跳过情感分析"
    echo "✓ 新闻文件已保存"
    exit 0
fi

python batch_sentiment_analyzer.py --input "$FILENAME"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 情感分析失败！"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ 流水线执行完成！"
echo "========================================"
echo ""
echo "输出文件:"
echo "  新闻数据: output/period_news/$FILENAME"
echo "  分析结果: output/analyzed_news/${FILENAME%.csv}_analyzed.csv"
echo ""
