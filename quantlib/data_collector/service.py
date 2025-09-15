"""
数据收集服务 - 主服务类，负责定期获取和存储分钟级数据
"""
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import json

from ..market_data.manager import get_data_manager
from .storage import DataStorage

logger = logging.getLogger(__name__)


class DataCollectionService:
    """数据收集服务"""
    
    def __init__(self, config_path: str = "config/data_collection.json"):
        """
        初始化数据收集服务
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 初始化组件
        self.data_manager = get_data_manager()
        self.storage = DataStorage(
            base_path=self.config.get('storage_path', 'data/minute_data'),
            file_format=self.config.get('file_format', 'parquet')
        )
        
        # 运行状态
        self.is_running = False
        self.last_collection_time = None
        
        logger.info("数据收集服务初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "symbols": ["000001", "000002", "000858", "600519", "600036"],
            "intervals": ["1min", "5min"],
            "storage_path": "data/minute_data",
            "file_format": "parquet",
            "collection_frequency_hours": 168,  # 一周一次
            "market": "CN",
            "max_retries": 3,
            "retry_delay_seconds": 60,
            "cleanup_enabled": True,
            "keep_months": 12,
            "log_level": "INFO"
        }
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    logger.info(f"配置文件加载成功: {self.config_path}")
            else:
                # 创建默认配置文件
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_config(default_config)
                logger.info(f"创建默认配置文件: {self.config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}, 使用默认配置")
        
        return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def collect_data_once(self) -> Dict[str, Any]:
        """
        执行一次数据收集
        
        Returns:
            收集结果统计
        """
        start_time = datetime.now()
        logger.info("开始数据收集...")
        
        symbols = self.config.get('symbols', [])
        intervals = self.config.get('intervals', ['1min'])
        market = self.config.get('market', 'CN')
        max_retries = self.config.get('max_retries', 3)
        retry_delay = self.config.get('retry_delay_seconds', 60)
        
        results = {
            'start_time': start_time,
            'symbols_total': len(symbols),
            'intervals_total': len(intervals),
            'success_count': 0,
            'failed_count': 0,
            'total_records': 0,
            'errors': [],
            'details': {}
        }
        
        for symbol in symbols:
            symbol_results = {'symbol': symbol, 'intervals': {}}
            
            for interval in intervals:
                interval_result = self._collect_symbol_interval_data(
                    symbol, interval, market, max_retries, retry_delay
                )
                symbol_results['intervals'][interval] = interval_result
                
                if interval_result['success']:
                    results['success_count'] += 1
                    results['total_records'] += interval_result['records']
                else:
                    results['failed_count'] += 1
                    results['errors'].append(f"{symbol}({interval}): {interval_result['error']}")
            
            results['details'][symbol] = symbol_results
        
        # 清理旧数据（如果启用）
        if self.config.get('cleanup_enabled', True):
            self._cleanup_old_data()
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - start_time).total_seconds()
        
        self.last_collection_time = datetime.now()
        
        logger.info(f"数据收集完成: 成功 {results['success_count']}, 失败 {results['failed_count']}, "
                   f"总记录 {results['total_records']}, 耗时 {results['duration']:.1f}秒")
        
        return results
    
    def _collect_symbol_interval_data(self, symbol: str, interval: str, market: str,
                                    max_retries: int, retry_delay: int) -> Dict[str, Any]:
        """
        收集单个symbol的单个时间间隔数据
        
        Args:
            symbol: 股票代码
            interval: 时间间隔
            market: 市场
            max_retries: 最大重试次数
            retry_delay: 重试延迟
            
        Returns:
            收集结果
        """
        result = {
            'success': False,
            'records': 0,
            'error': None,
            'retry_count': 0
        }
        
        for attempt in range(max_retries + 1):
            try:
                # 获取分钟级数据（akshare限制为近5个交易日）
                data = self.data_manager.get_stock_data(
                    symbol=symbol,
                    market=market,
                    interval=interval
                )
                
                if data is None or data.empty:
                    result['error'] = f"未获取到数据: {symbol}({interval})"
                    logger.warning(result['error'])
                    return result
                
                # 保存数据
                success = self.storage.save_data(symbol, data, append_mode=True)
                
                if success:
                    result['success'] = True
                    result['records'] = len(data)
                    logger.info(f"✓ {symbol}({interval}): 收集 {len(data)} 条记录")
                    return result
                else:
                    result['error'] = f"数据保存失败: {symbol}({interval})"
                    
            except Exception as e:
                result['error'] = str(e)
                result['retry_count'] = attempt + 1
                
                if attempt < max_retries:
                    logger.warning(f"× {symbol}({interval}) 第{attempt+1}次尝试失败: {e}, "
                                 f"{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"× {symbol}({interval}) 最终失败: {e}")
        
        return result
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            keep_months = self.config.get('keep_months', 12)
            symbols = self.storage.get_available_symbols()
            
            for symbol in symbols:
                self.storage.clean_old_data(symbol, keep_months)
                
            logger.info(f"旧数据清理完成，保留最近 {keep_months} 个月")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
    
    def run_continuous(self):
        """持续运行模式"""
        self.is_running = True
        frequency_hours = self.config.get('collection_frequency_hours', 168)  # 默认一周
        
        logger.info(f"启动持续运行模式，收集频率: {frequency_hours} 小时")
        
        try:
            while self.is_running:
                # 执行数据收集
                results = self.collect_data_once()
                
                # 记录收集结果
                self._log_collection_results(results)
                
                if not self.is_running:
                    break
                
                # 等待下次收集
                logger.info(f"等待 {frequency_hours} 小时后进行下次收集...")
                
                # 分段等待，以便能够响应停止信号
                wait_seconds = frequency_hours * 3600
                waited = 0
                while waited < wait_seconds and self.is_running:
                    time.sleep(min(60, wait_seconds - waited))  # 每分钟检查一次
                    waited += 60
                    
        except KeyboardInterrupt:
            logger.info("接收到停止信号，正在停止服务...")
        except Exception as e:
            logger.error(f"持续运行模式出错: {e}")
        finally:
            self.is_running = False
            logger.info("数据收集服务已停止")
    
    def stop(self):
        """停止服务"""
        self.is_running = False
        logger.info("数据收集服务停止请求已发送")
    
    def _log_collection_results(self, results: Dict[str, Any]):
        """记录收集结果到日志"""
        try:
            # 创建结果摘要
            summary = {
                'timestamp': results['start_time'].isoformat(),
                'duration_seconds': results['duration'],
                'symbols_total': results['symbols_total'],
                'success_count': results['success_count'],
                'failed_count': results['failed_count'],
                'total_records': results['total_records']
            }
            
            # 保存到文件
            log_dir = Path('logs/data_collection')
            log_dir.mkdir(parents=True, exist_ok=True)
            
            date_str = results['start_time'].strftime('%Y-%m-%d')
            log_file = log_dir / f"collection_{date_str}.json"
            
            # 读取现有日志
            logs = []
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            logs.append(summary)
            
            # 保存更新后的日志
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"记录收集结果失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'is_running': self.is_running,
            'last_collection_time': self.last_collection_time,
            'config': self.config,
            'available_symbols': self.storage.get_available_symbols(),
            'storage_path': str(self.storage.base_path),
            'file_format': self.storage.file_format
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据存储摘要"""
        try:
            symbols = self.storage.get_available_symbols()
            summary = {
                'total_symbols': len(symbols),
                'symbols': {},
                'total_records': 0,
                'total_months': 0
            }
            
            for symbol in symbols:
                info = self.storage.get_data_info(symbol)
                summary['symbols'][symbol] = info
                summary['total_records'] += info['records']
                summary['total_months'] += info['months']
            
            return summary
            
        except Exception as e:
            logger.error(f"获取数据摘要失败: {e}")
            return {'error': str(e)}
    
    def add_symbol(self, symbol: str) -> bool:
        """
        添加新的symbol到监控列表
        
        Args:
            symbol: 股票代码
            
        Returns:
            是否添加成功
        """
        try:
            symbols = self.config.get('symbols', [])
            if symbol not in symbols:
                symbols.append(symbol)
                self.config['symbols'] = symbols
                self._save_config(self.config)
                logger.info(f"添加新symbol: {symbol}")
                return True
            else:
                logger.warning(f"Symbol已存在: {symbol}")
                return False
        except Exception as e:
            logger.error(f"添加symbol失败: {e}")
            return False
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        从监控列表移除symbol
        
        Args:
            symbol: 股票代码
            
        Returns:
            是否移除成功
        """
        try:
            symbols = self.config.get('symbols', [])
            if symbol in symbols:
                symbols.remove(symbol)
                self.config['symbols'] = symbols
                self._save_config(self.config)
                logger.info(f"移除symbol: {symbol}")
                return True
            else:
                logger.warning(f"Symbol不存在: {symbol}")
                return False
        except Exception as e:
            logger.error(f"移除symbol失败: {e}")
            return False
    
    def update_config(self, **kwargs) -> bool:
        """
        更新配置
        
        Args:
            **kwargs: 配置项
            
        Returns:
            是否更新成功
        """
        try:
            for key, value in kwargs.items():
                if key in self.config:
                    self.config[key] = value
                    logger.info(f"配置更新: {key} = {value}")
            
            self._save_config(self.config)
            return True
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False