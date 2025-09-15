"""
数据存储管理器 - 负责按symbol和月份组织存储分钟级数据
"""
import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import logging
from .resample import get_resampler

logger = logging.getLogger(__name__)


class DataStorage:
    """数据存储管理器"""
    
    def __init__(self, base_path: str = "data/minute_data", 
                 file_format: str = "parquet"):
        """
        初始化数据存储管理器
        
        Args:
            base_path: 数据存储基础路径
            file_format: 文件格式 ('parquet' 或 'csv')
        """
        self.base_path = Path(base_path)
        self.file_format = file_format.lower()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        if self.file_format not in ['parquet', 'csv']:
            raise ValueError("文件格式只支持 'parquet' 或 'csv'")
    
    def get_file_path(self, symbol: str, year: int, month: int) -> Path:
        """
        获取指定symbol和月份的文件路径
        
        Args:
            symbol: 股票代码
            year: 年份
            month: 月份
            
        Returns:
            文件路径
        """
        symbol_dir = self.base_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{year}-{month:02d}.{self.file_format}"
        return symbol_dir / filename
    
    def save_data(self, symbol: str, data: pd.DataFrame, 
                  append_mode: bool = True) -> bool:
        """
        保存数据到对应的文件
        
        Args:
            symbol: 股票代码
            data: 要保存的数据
            append_mode: 是否追加模式（合并到现有数据）
            
        Returns:
            是否保存成功
        """
        try:
            if data.empty:
                logger.warning(f"数据为空，跳过保存: {symbol}")
                return False
            
            # 确保有date列
            if 'date' not in data.columns:
                logger.error(f"数据缺少date列: {symbol}")
                return False
            
            # 按月份分组数据
            data['date'] = pd.to_datetime(data['date'])
            data['year_month'] = data['date'].dt.to_period('M')
            
            success_count = 0
            total_groups = len(data.groupby('year_month'))
            
            for period, group_data in data.groupby('year_month'):
                year = period.year
                month = period.month
                file_path = self.get_file_path(symbol, year, month)
                
                # 移除辅助列
                group_data = group_data.drop('year_month', axis=1)
                
                try:
                    if append_mode and file_path.exists():
                        # 追加模式：合并现有数据
                        existing_data = self._load_file(file_path)
                        if existing_data is not None:
                            # 合并数据并去重
                            combined_data = pd.concat([existing_data, group_data], ignore_index=True)
                            combined_data = self._remove_duplicates(combined_data)
                            self._save_file(combined_data, file_path)
                        else:
                            self._save_file(group_data, file_path)
                    else:
                        # 覆盖模式
                        self._save_file(group_data, file_path)
                    
                    success_count += 1
                    logger.info(f"成功保存 {symbol} {year}-{month:02d} 数据: {len(group_data)} 条记录")
                    
                except Exception as e:
                    logger.error(f"保存 {symbol} {year}-{month:02d} 数据失败: {e}")
            
            logger.info(f"数据保存完成: {symbol}, 成功 {success_count}/{total_groups} 个月份")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"保存数据失败 {symbol}: {e}")
            return False
    
    def load_data(self, symbol: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """
        加载指定symbol和月份的数据
        
        Args:
            symbol: 股票代码
            year: 年份
            month: 月份
            
        Returns:
            数据DataFrame或None
        """
        try:
            file_path = self.get_file_path(symbol, year, month)
            if not file_path.exists():
                return None
            
            return self._load_file(file_path)
            
        except Exception as e:
            logger.error(f"加载数据失败 {symbol} {year}-{month:02d}: {e}")
            return None
    
    def load_symbol_data(self, symbol: str, start_date: Optional[date] = None, 
                        end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """
        加载指定symbol的所有数据或日期范围内的数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            合并后的数据DataFrame或None
        """
        try:
            symbol_dir = self.base_path / symbol
            if not symbol_dir.exists():
                return None
            
            # 获取所有月份文件
            pattern = f"*.{self.file_format}"
            files = list(symbol_dir.glob(pattern))
            
            if not files:
                return None
            
            # 筛选日期范围内的文件
            valid_files = []
            for file_path in files:
                # 解析文件名中的年月 (格式: YYYY-MM.ext)
                stem = file_path.stem  # 去掉扩展名
                try:
                    year, month = map(int, stem.split('-'))
                    file_date = date(year, month, 1)
                    
                    # 检查是否在日期范围内
                    if start_date and file_date < start_date.replace(day=1):
                        continue
                    if end_date and file_date > end_date.replace(day=1):
                        continue
                    
                    valid_files.append(file_path)
                except ValueError:
                    logger.warning(f"文件名格式错误，跳过: {file_path}")
                    continue
            
            if not valid_files:
                return None
            
            # 加载并合并所有文件
            all_data = []
            for file_path in sorted(valid_files):
                month_data = self._load_file(file_path)
                if month_data is not None:
                    all_data.append(month_data)
            
            if not all_data:
                return None
            
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # 按日期范围再次筛选
            if start_date or end_date:
                combined_data['date'] = pd.to_datetime(combined_data['date'])
                if start_date:
                    combined_data = combined_data[combined_data['date'].dt.date >= start_date]
                if end_date:
                    combined_data = combined_data[combined_data['date'].dt.date <= end_date]
            
            # 按时间排序
            if 'date' in combined_data.columns:
                combined_data = combined_data.sort_values('date').reset_index(drop=True)
            
            logger.info(f"加载 {symbol} 数据完成: {len(combined_data)} 条记录")
            return combined_data
            
        except Exception as e:
            logger.error(f"加载symbol数据失败 {symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """获取所有可用的symbol列表"""
        try:
            symbols = []
            for item in self.base_path.iterdir():
                if item.is_dir():
                    symbols.append(item.name)
            return sorted(symbols)
        except Exception as e:
            logger.error(f"获取symbol列表失败: {e}")
            return []
    
    def get_symbol_months(self, symbol: str) -> List[tuple]:
        """
        获取指定symbol的所有可用月份
        
        Args:
            symbol: 股票代码
            
        Returns:
            (year, month) 元组列表
        """
        try:
            symbol_dir = self.base_path / symbol
            if not symbol_dir.exists():
                return []
            
            months = []
            pattern = f"*.{self.file_format}"
            for file_path in symbol_dir.glob(pattern):
                try:
                    stem = file_path.stem
                    year, month = map(int, stem.split('-'))
                    months.append((year, month))
                except ValueError:
                    continue
            
            return sorted(months)
            
        except Exception as e:
            logger.error(f"获取 {symbol} 月份列表失败: {e}")
            return []
    
    def get_data_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取指定symbol的数据统计信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            数据统计信息
        """
        try:
            months = self.get_symbol_months(symbol)
            if not months:
                return {'symbol': symbol, 'months': 0, 'records': 0, 'date_range': None}
            
            total_records = 0
            date_range = None
            
            # 计算总记录数和日期范围
            for year, month in months:
                data = self.load_data(symbol, year, month)
                if data is not None:
                    total_records += len(data)
                    
                    if 'date' in data.columns and not data.empty:
                        data['date'] = pd.to_datetime(data['date'])
                        month_start = data['date'].min()
                        month_end = data['date'].max()
                        
                        if date_range is None:
                            date_range = (month_start, month_end)
                        else:
                            date_range = (
                                min(date_range[0], month_start),
                                max(date_range[1], month_end)
                            )
            
            return {
                'symbol': symbol,
                'months': len(months),
                'records': total_records,
                'date_range': date_range,
                'month_list': months
            }
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据信息失败: {e}")
            return {'symbol': symbol, 'months': 0, 'records': 0, 'date_range': None}
    
    def load_and_resample(self, symbol: str, target_interval: str, 
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None,
                         **resample_kwargs) -> Optional[pd.DataFrame]:
        """
        加载数据并重采样到目标时间间隔
        
        Args:
            symbol: 股票代码
            target_interval: 目标时间间隔 (如 '5min', '1h', '1d')
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            **resample_kwargs: 重采样的其他参数
            
        Returns:
            重采样后的数据DataFrame或None
        """
        try:
            # 加载原始数据
            raw_data = self.load_symbol_data(symbol, start_date, end_date)
            if raw_data is None or raw_data.empty:
                logger.warning(f"没有找到 {symbol} 的数据")
                return None
            
            # 重采样
            resampler = get_resampler()
            resampled_data = resampler.resample(
                data=raw_data,
                target_interval=target_interval,
                datetime_column='date',
                **resample_kwargs
            )
            
            logger.info(f"{symbol} 数据重采样完成: {len(raw_data)} -> {len(resampled_data)} 条记录 ({target_interval})")
            return resampled_data
            
        except Exception as e:
            logger.error(f"加载并重采样 {symbol} 数据失败: {e}")
            return None
    
    def batch_resample_symbol(self, symbol: str, target_intervals: List[str],
                             start_date: Optional[date] = None,
                             end_date: Optional[date] = None,
                             save_results: bool = False,
                             **resample_kwargs) -> Dict[str, pd.DataFrame]:
        """
        批量重采样单个股票到多个时间间隔
        
        Args:
            symbol: 股票代码
            target_intervals: 目标时间间隔列表
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            save_results: 是否保存重采样结果
            **resample_kwargs: 重采样的其他参数
            
        Returns:
            {时间间隔: DataFrame} 字典
        """
        try:
            # 加载原始数据
            raw_data = self.load_symbol_data(symbol, start_date, end_date)
            if raw_data is None or raw_data.empty:
                logger.warning(f"没有找到 {symbol} 的数据")
                return {}
            
            # 批量重采样
            resampler = get_resampler()
            results = resampler.batch_resample(
                data=raw_data,
                target_intervals=target_intervals,
                datetime_column='date',
                **resample_kwargs
            )
            
            # 保存结果（可选）
            if save_results:
                for interval, data in results.items():
                    if not data.empty:
                        # 创建重采样数据的存储路径
                        interval_storage = DataStorage(
                            base_path=f"{self.base_path}_resampled_{interval}",
                            file_format=self.file_format
                        )
                        interval_storage.save_data(symbol, data, append_mode=False)
            
            logger.info(f"{symbol} 批量重采样完成: {len(target_intervals)} 个时间间隔")
            return results
            
        except Exception as e:
            logger.error(f"批量重采样 {symbol} 失败: {e}")
            return {}
    
    def get_resample_summary(self, symbol: str, intervals: List[str]) -> Dict[str, Any]:
        """
        获取重采样数据摘要
        
        Args:
            symbol: 股票代码  
            intervals: 时间间隔列表
            
        Returns:
            重采样摘要信息
        """
        try:
            original_info = self.get_data_info(symbol)
            if original_info['records'] == 0:
                return {'error': f'没有 {symbol} 的原始数据'}
            
            summary = {
                'symbol': symbol,
                'original_records': original_info['records'],
                'original_date_range': original_info['date_range'],
                'intervals': {}
            }
            
            for interval in intervals:
                try:
                    resampled = self.load_and_resample(symbol, interval)
                    if resampled is not None:
                        summary['intervals'][interval] = {
                            'records': len(resampled),
                            'compression_ratio': len(resampled) / original_info['records'],
                            'date_range': (resampled['date'].min(), resampled['date'].max()) if not resampled.empty else None
                        }
                    else:
                        summary['intervals'][interval] = {
                            'records': 0,
                            'compression_ratio': 0,
                            'date_range': None
                        }
                except Exception as e:
                    summary['intervals'][interval] = {
                        'error': str(e)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取重采样摘要失败 {symbol}: {e}")
            return {'error': str(e)}
    
    def clean_old_data(self, symbol: str, keep_months: int = 12):
        """
        清理旧数据，只保留最近几个月的数据
        
        Args:
            symbol: 股票代码
            keep_months: 保留月份数量
        """
        try:
            months = self.get_symbol_months(symbol)
            if len(months) <= keep_months:
                return
            
            # 排序并删除最旧的数据
            months.sort(reverse=True)  # 最新的在前
            months_to_delete = months[keep_months:]
            
            deleted_count = 0
            for year, month in months_to_delete:
                file_path = self.get_file_path(symbol, year, month)
                if file_path.exists():
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"删除旧数据: {symbol} {year}-{month:02d}")
            
            logger.info(f"清理完成: {symbol}, 删除 {deleted_count} 个月份的数据")
            
        except Exception as e:
            logger.error(f"清理旧数据失败 {symbol}: {e}")
    
    def _load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """加载文件"""
        try:
            if self.file_format == 'parquet':
                return pd.read_parquet(file_path)
            else:  # csv
                return pd.read_csv(file_path, parse_dates=['date'])
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            return None
    
    def _save_file(self, data: pd.DataFrame, file_path: Path):
        """保存文件"""
        if self.file_format == 'parquet':
            data.to_parquet(file_path, index=False)
        else:  # csv
            data.to_csv(file_path, index=False)
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        去除重复数据
        
        Args:
            data: 原始数据
            
        Returns:
            去重后的数据
        """
        try:
            # 基于时间去重，保留最新的记录
            if 'date' in data.columns:
                data = data.drop_duplicates(subset=['date'], keep='last')
            else:
                # 如果没有时间列，基于所有列去重
                data = data.drop_duplicates()
            
            return data.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"去重失败: {e}")
            return data