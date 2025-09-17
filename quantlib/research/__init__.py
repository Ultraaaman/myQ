"""
量化研究模块 (Research Module)

本模块提供完整的量化投资研究功能，包括：
- 因子库管理和计算
- 因子有效性分析
- 因子回测和评估
- 研究报告生成
- 策略研究框架
"""

from .factor_library import (
    FactorLibrary,
    FactorCalculator,
    FactorCategory,
    create_factor_library
)

from .factor_analyzer import (
    FactorAnalyzer,
    ICAnalysis,
    FactorPerformance,
    create_factor_analyzer
)

from .research_framework import (
    ResearchFramework,
    FactorBacktester,
    create_research_framework
)

from .report_generator import (
    ReportGenerator,
    create_research_report
)

__all__ = [
    # 因子库管理
    'FactorLibrary',
    'FactorCalculator',
    'FactorCategory',
    'create_factor_library',
    
    # 因子分析
    'FactorAnalyzer',
    'ICAnalysis',
    'FactorPerformance',
    'create_factor_analyzer',
    
    # 研究框架
    'ResearchFramework',
    'FactorBacktester',
    'create_research_framework',
    
    # 报告生成
    'ReportGenerator',
    'create_research_report'
]

__version__ = '1.0.0'