"""
数据重采样工具 - 将DataFrame数据转换到不同的时间间隔
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union, List
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class DataResampler:
    """数据重采样器 - 转换时间间隔"""
    
    def __init__(self):
        """初始化重采样器"""
        # 支持的时间间隔映射
        self.interval_mapping = {
            # 分钟级
            '1min': '1T', '1m': '1T', '1分钟': '1T',
            '2min': '2T', '2m': '2T', '2分钟': '2T',
            '3min': '3T', '3m': '3T', '3分钟': '3T',
            '5min': '5T', '5m': '5T', '5分钟': '5T',
            '10min': '10T', '10m': '10T', '10分钟': '10T',
            '15min': '15T', '15m': '15T', '15分钟': '15T',
            '30min': '30T', '30m': '30T', '30分钟': '30T',
            '60min': '60T', '60m': '60T', '60分钟': '60T',
            
            # 小时级
            '1h': '1H', '1hour': '1H', '1小时': '1H',
            '2h': '2H', '2hour': '2H', '2小时': '2H',
            '4h': '4H', '4hour': '4H', '4小时': '4H',
            '6h': '6H', '6hour': '6H', '6小时': '6H',
            '8h': '8H', '8hour': '8H', '8小时': '8H',
            '12h': '12H', '12hour': '12H', '12小时': '12H',
            
            # 日级
            '1d': '1D', '1day': '1D', '1日': '1D', 'daily': '1D',
            '2d': '2D', '2day': '2D', '2日': '2D',
            '3d': '3D', '3day': '3D', '3日': '3D',
            
            # 周级
            '1w': '1W', '1week': '1W', '1周': '1W', 'weekly': '1W',
            '2w': '2W', '2week': '2W', '2周': '2W',
            
            # 月级
            '1M': '1M', '1month': '1M', '1月': '1M', 'monthly': '1M',
            '3M': '3M', '3month': '3M', '3月': '3M', 'quarterly': '3M',
            '6M': '6M', '6month': '6M', '6月': '6M',
            
            # 年级
            '1Y': '1Y', '1year': '1Y', '1年': '1Y', 'yearly': '1Y'
        }
    
    def resample(self, data: pd.DataFrame, 
                target_interval: str,
                datetime_column: str = 'date',
                price_columns: Optional[List[str]] = None,
                volume_columns: Optional[List[str]] = None,
                custom_agg: Optional[Dict[str, Union[str, callable]]] = None,
                drop_incomplete: bool = True) -> pd.DataFrame:
        """
        将数据重采样到目标时间间隔
        
        Args:
            data: 原始数据DataFrame
            target_interval: 目标时间间隔 (如 '5min', '1h', '1d')
            datetime_column: 时间列名
            price_columns: 价格类列名列表，使用OHLC聚合
            volume_columns: 成交量类列名列表，使用sum聚合
            custom_agg: 自定义聚合规则
            drop_incomplete: 是否删除不完整的周期
            
        Returns:
            重采样后的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空")
            return data.copy()
        
        try:
            # 复制数据避免修改原始数据
            df = data.copy()
            
            # 检查时间列
            if datetime_column not in df.columns:
                raise ValueError(f"未找到时间列: {datetime_column}")
            
            # 转换时间列
            df[datetime_column] = pd.to_datetime(df[datetime_column])
            
            # 设置时间索引
            df.set_index(datetime_column, inplace=True)
            
            # 检查数据是否按时间排序
            if not df.index.is_monotonic_increasing:
                df.sort_index(inplace=True)
                logger.info("数据已按时间排序")
            
            # 转换间隔格式
            pandas_interval = self._convert_interval(target_interval)
            
            # 检测价格和成交量列
            if price_columns is None:
                price_columns = self._detect_price_columns(df.columns)
            if volume_columns is None:
                volume_columns = self._detect_volume_columns(df.columns)
            
            # 构建聚合规则
            agg_rules = self._build_aggregation_rules(
                df.columns, price_columns, volume_columns, custom_agg
            )
            
            logger.info(f"重采样: {len(df)} 条记录 -> {target_interval}")
            logger.debug(f"聚合规则: {agg_rules}")
            
            # 执行重采样
            resampled = df.resample(pandas_interval, label='left', closed='left').agg(agg_rules)
            
            # 删除不完整的周期（可选）
            if drop_incomplete and not resampled.empty:
                # 删除最后一个可能不完整的周期
                last_complete_time = self._get_last_complete_period(
                    df.index[-1], pandas_interval
                )
                if last_complete_time is not None:
                    resampled = resampled[resampled.index <= last_complete_time]
            
            # 删除空值行
            resampled.dropna(how='all', inplace=True)
            
            # 重置索引，将时间从索引变为列
            resampled.reset_index(inplace=True)
            
            logger.info(f"重采样完成: {len(resampled)} 条记录")
            return resampled
            
        except Exception as e:
            logger.error(f"重采样失败: {e}")
            raise
    
    def resample_ohlcv(self, data: pd.DataFrame,
                      target_interval: str,
                      datetime_column: str = 'date',
                      open_col: str = 'open',
                      high_col: str = 'high',
                      low_col: str = 'low',
                      close_col: str = 'close',
                      volume_col: str = 'volume',
                      additional_cols: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        专门用于OHLCV数据的重采样
        
        Args:
            data: 原始OHLCV数据
            target_interval: 目标时间间隔
            datetime_column: 时间列名
            open_col: 开盘价列名
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名
            additional_cols: 其他列的聚合规则 {'列名': '聚合方法'}
            
        Returns:
            重采样后的OHLCV DataFrame
        """
        try:
            # 构建OHLCV聚合规则
            agg_rules = {}
            
            # 检查列是否存在
            available_cols = data.columns.tolist()
            
            if open_col in available_cols:
                agg_rules[open_col] = 'first'
            if high_col in available_cols:
                agg_rules[high_col] = 'max'
            if low_col in available_cols:
                agg_rules[low_col] = 'min'
            if close_col in available_cols:
                agg_rules[close_col] = 'last'
            if volume_col in available_cols:
                agg_rules[volume_col] = 'sum'
            
            # 添加其他列的聚合规则
            if additional_cols:
                for col, agg_method in additional_cols.items():
                    if col in available_cols:
                        agg_rules[col] = agg_method
            
            if not agg_rules:
                raise ValueError("未找到有效的OHLCV列")
            
            return self.resample(
                data=data,
                target_interval=target_interval,
                datetime_column=datetime_column,
                custom_agg=agg_rules
            )
            
        except Exception as e:
            logger.error(f"OHLCV重采样失败: {e}")
            raise
    
    def batch_resample(self, data: pd.DataFrame,
                      target_intervals: List[str],
                      datetime_column: str = 'date',
                      **kwargs) -> Dict[str, pd.DataFrame]:
        """
        批量重采样到多个时间间隔
        
        Args:
            data: 原始数据
            target_intervals: 目标时间间隔列表
            datetime_column: 时间列名
            **kwargs: resample方法的其他参数
            
        Returns:
            {时间间隔: DataFrame} 字典
        """
        results = {}
        
        for interval in target_intervals:
            try:
                resampled_data = self.resample(
                    data=data,
                    target_interval=interval,
                    datetime_column=datetime_column,
                    **kwargs
                )
                results[interval] = resampled_data
                logger.info(f"批量重采样 {interval}: {len(resampled_data)} 条记录")
            except Exception as e:
                logger.error(f"批量重采样 {interval} 失败: {e}")
                results[interval] = pd.DataFrame()
        
        return results
    
    def upsample(self, data: pd.DataFrame,
                target_interval: str,
                method: str = 'ffill',
                datetime_column: str = 'date') -> pd.DataFrame:
        """
        上采样 - 将低频数据转为高频数据
        
        Args:
            data: 原始数据
            target_interval: 目标时间间隔
            method: 填充方法 ('ffill', 'bfill', 'interpolate')
            datetime_column: 时间列名
            
        Returns:
            上采样后的DataFrame
        """
        try:
            df = data.copy()
            df[datetime_column] = pd.to_datetime(df[datetime_column])
            df.set_index(datetime_column, inplace=True)
            
            pandas_interval = self._convert_interval(target_interval)
            
            # 创建完整的时间索引
            full_index = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq=pandas_interval
            )
            
            # 重新索引
            df_reindexed = df.reindex(full_index)
            
            # 填充缺失值
            if method == 'ffill':
                df_reindexed.fillna(method='ffill', inplace=True)
            elif method == 'bfill':
                df_reindexed.fillna(method='bfill', inplace=True)
            elif method == 'interpolate':
                df_reindexed.interpolate(inplace=True)
            
            df_reindexed.reset_index(inplace=True)
            df_reindexed.rename(columns={'index': datetime_column}, inplace=True)
            
            logger.info(f"上采样完成: {len(data)} -> {len(df_reindexed)} 条记录")
            return df_reindexed
            
        except Exception as e:
            logger.error(f"上采样失败: {e}")
            raise
    
    def _convert_interval(self, interval: str) -> str:
        """转换时间间隔格式为pandas格式"""
        if interval in self.interval_mapping:
            return self.interval_mapping[interval]
        
        # 尝试直接使用（可能已经是pandas格式）
        try:
            pd.Timedelta(interval)
            return interval
        except:
            pass
        
        raise ValueError(f"不支持的时间间隔: {interval}")
    
    def _detect_price_columns(self, columns: List[str]) -> List[str]:
        """自动检测价格列"""
        price_patterns = [
            'open', 'high', 'low', 'close', 'adj_close',
            '开盘', '最高', '最低', '收盘', '调整收盘',
            'price', '价格'
        ]
        
        price_cols = []
        for col in columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in price_patterns):
                price_cols.append(col)
        
        return price_cols
    
    def _detect_volume_columns(self, columns: List[str]) -> List[str]:
        """自动检测成交量列"""
        volume_patterns = [
            'volume', 'vol', 'amount', 'turnover',
            '成交量', '成交额', '换手率'
        ]
        
        volume_cols = []
        for col in columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in volume_patterns):
                volume_cols.append(col)
        
        return volume_cols
    
    def _build_aggregation_rules(self, columns: List[str],
                                price_columns: List[str],
                                volume_columns: List[str],
                                custom_agg: Optional[Dict[str, Union[str, callable]]]) -> Dict:
        """构建聚合规则"""
        agg_rules = {}
        
        # 自定义聚合规则优先
        if custom_agg:
            agg_rules.update(custom_agg)
        
        # 为未指定的列添加默认规则
        for col in columns:
            if col not in agg_rules:
                if col in price_columns:
                    # 价格列使用OHLC逻辑
                    if 'open' in col.lower() or '开盘' in col:
                        agg_rules[col] = 'first'
                    elif 'high' in col.lower() or '最高' in col:
                        agg_rules[col] = 'max'
                    elif 'low' in col.lower() or '最低' in col:
                        agg_rules[col] = 'min'
                    elif 'close' in col.lower() or '收盘' in col:
                        agg_rules[col] = 'last'
                    else:
                        agg_rules[col] = 'mean'  # 其他价格列取平均
                elif col in volume_columns:
                    # 成交量列使用求和
                    agg_rules[col] = 'sum'
                else:
                    # 其他列使用最后值
                    agg_rules[col] = 'last'
        
        return agg_rules
    
    def _get_last_complete_period(self, last_time: pd.Timestamp, freq: str) -> Optional[pd.Timestamp]:
        """获取最后一个完整周期的时间"""
        try:
            # 根据频率计算完整周期的结束时间
            if 'T' in freq or 'min' in freq:  # 分钟
                minutes = pd.Timedelta(freq).total_seconds() / 60
                last_complete = last_time.floor(f"{int(minutes)}T")
            elif 'H' in freq:  # 小时
                hours = pd.Timedelta(freq).total_seconds() / 3600
                last_complete = last_time.floor(f"{int(hours)}H")
            elif 'D' in freq:  # 天
                last_complete = last_time.floor('D')
            elif 'W' in freq:  # 周
                last_complete = last_time.floor('W')
            elif 'M' in freq:  # 月
                last_complete = last_time.floor('M')
            else:
                return None
            
            return last_complete
            
        except Exception:
            return None
    
    def get_supported_intervals(self) -> List[str]:
        """获取支持的时间间隔列表"""
        return list(self.interval_mapping.keys())
    
    def get_interval_info(self, interval: str) -> Dict[str, Any]:
        """获取时间间隔信息"""
        try:
            pandas_freq = self._convert_interval(interval)
            timedelta_obj = pd.Timedelta(pandas_freq)
            
            return {
                'original': interval,
                'pandas_format': pandas_freq,
                'seconds': timedelta_obj.total_seconds(),
                'minutes': timedelta_obj.total_seconds() / 60,
                'hours': timedelta_obj.total_seconds() / 3600,
                'days': timedelta_obj.total_seconds() / 86400,
                'description': self._get_interval_description(interval)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_interval_description(self, interval: str) -> str:
        """获取时间间隔描述"""
        descriptions = {
            '1min': '1分钟', '5min': '5分钟', '15min': '15分钟', '30min': '30分钟',
            '1h': '1小时', '4h': '4小时', '1d': '1天', '1w': '1周', '1M': '1月'
        }
        return descriptions.get(interval, interval)


# 全局实例
_global_resampler = None


def get_resampler() -> DataResampler:
    """获取全局重采样器实例"""
    global _global_resampler
    if _global_resampler is None:
        _global_resampler = DataResampler()
    return _global_resampler


# 便捷函数
def resample_data(data: pd.DataFrame, target_interval: str, **kwargs) -> pd.DataFrame:
    """便捷函数：重采样数据"""
    resampler = get_resampler()
    return resampler.resample(data, target_interval, **kwargs)


def resample_ohlcv(data: pd.DataFrame, target_interval: str, **kwargs) -> pd.DataFrame:
    """便捷函数：重采样OHLCV数据"""
    resampler = get_resampler()
    return resampler.resample_ohlcv(data, target_interval, **kwargs)


def batch_resample(data: pd.DataFrame, target_intervals: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
    """便捷函数：批量重采样"""
    resampler = get_resampler()
    return resampler.batch_resample(data, target_intervals, **kwargs)


def upsample_data(data: pd.DataFrame, target_interval: str, method: str = 'ffill', **kwargs) -> pd.DataFrame:
    """便捷函数：上采样数据"""
    resampler = get_resampler()
    return resampler.upsample(data, target_interval, method, **kwargs)