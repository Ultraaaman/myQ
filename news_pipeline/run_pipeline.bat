@echo off
chcp 65001 >nul
REM 新闻分析流水线快捷脚本

echo ========================================
echo    新闻数据处理流水线
echo ========================================
echo.

REM 检查参数
if "%1"=="" (
    echo 使用方法:
    echo   run_pipeline.bat START_DATE END_DATE
    echo.
    echo 示例:
    echo   run_pipeline.bat 2025-09-01 2025-09-30
    echo.
    exit /b 1
)

if "%2"=="" (
    echo 错误: 请提供结束日期
    echo.
    echo 使用方法:
    echo   run_pipeline.bat START_DATE END_DATE
    echo.
    exit /b 1
)

set START_DATE=%1
set END_DATE=%2
set FILENAME=news_%START_DATE:~0,4%%START_DATE:~5,2%%START_DATE:~8,2%_%END_DATE:~0,4%%END_DATE:~5,2%%END_DATE:~8,2%.csv

echo 配置信息:
echo   开始日期: %START_DATE%
echo   结束日期: %END_DATE%
echo   输出文件: %FILENAME%
echo.

echo ----------------------------------------
echo 步骤 1/2: 抓取新闻数据
echo ----------------------------------------
python period_news_fetcher.py --start_date %START_DATE% --end_date %END_DATE%

if errorlevel 1 (
    echo.
    echo ❌ 新闻抓取失败！
    exit /b 1
)

echo.
echo ----------------------------------------
echo 步骤 2/2: 情感分析
echo ----------------------------------------
set /p CONFIRM="是否继续进行情感分析? (y/n): "

if /i not "%CONFIRM%"=="y" (
    echo.
    echo ⏸️  已跳过情感分析
    echo ✓ 新闻文件已保存
    exit /b 0
)

python batch_sentiment_analyzer.py --input %FILENAME%

if errorlevel 1 (
    echo.
    echo ❌ 情感分析失败！
    exit /b 1
)

echo.
echo ========================================
echo ✓ 流水线执行完成！
echo ========================================
echo.
echo 输出文件:
echo   新闻数据: output\period_news\%FILENAME%
echo   分析结果: output\analyzed_news\%FILENAME:~0,-4%_analyzed.csv
echo.
