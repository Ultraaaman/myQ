"""
数据收集调度器 - 负责管理服务的启动、停止和定时执行
"""
import threading
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional, Callable
import schedule
import time

from .service import DataCollectionService

logger = logging.getLogger(__name__)


class DataScheduler:
    """数据收集调度器"""
    
    def __init__(self, service: DataCollectionService):
        """
        初始化调度器
        
        Args:
            service: 数据收集服务实例
        """
        self.service = service
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("数据收集调度器初始化完成")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，正在停止调度器...")
        self.stop()
    
    def start_once(self) -> dict:
        """执行一次数据收集"""
        logger.info("手动执行数据收集")
        return self.service.collect_data_once()
    
    def start_scheduled(self, schedule_type: str = "weekly", 
                       schedule_time: str = "02:00",
                       weekday: str = "sunday") -> bool:
        """
        启动定时调度
        
        Args:
            schedule_type: 调度类型 ('daily', 'weekly', 'hourly')
            schedule_time: 执行时间 (格式: "HH:MM")
            weekday: 星期几 (仅weekly模式使用)
            
        Returns:
            是否启动成功
        """
        try:
            if self.is_running:
                logger.warning("调度器已在运行")
                return False
            
            # 清除现有调度
            schedule.clear()
            
            # 设置调度规则
            if schedule_type == "daily":
                schedule.every().day.at(schedule_time).do(self._scheduled_job)
                logger.info(f"设置每日调度: {schedule_time}")
                
            elif schedule_type == "weekly":
                if weekday.lower() == "monday":
                    schedule.every().monday.at(schedule_time).do(self._scheduled_job)
                elif weekday.lower() == "tuesday":
                    schedule.every().tuesday.at(schedule_time).do(self._scheduled_job)
                elif weekday.lower() == "wednesday":
                    schedule.every().wednesday.at(schedule_time).do(self._scheduled_job)
                elif weekday.lower() == "thursday":
                    schedule.every().thursday.at(schedule_time).do(self._scheduled_job)
                elif weekday.lower() == "friday":
                    schedule.every().friday.at(schedule_time).do(self._scheduled_job)
                elif weekday.lower() == "saturday":
                    schedule.every().saturday.at(schedule_time).do(self._scheduled_job)
                elif weekday.lower() == "sunday":
                    schedule.every().sunday.at(schedule_time).do(self._scheduled_job)
                else:
                    raise ValueError(f"无效的星期: {weekday}")
                logger.info(f"设置每周调度: {weekday} {schedule_time}")
                
            elif schedule_type == "hourly":
                # hourly模式下，schedule_time表示小时间隔
                hours = int(schedule_time) if schedule_time.isdigit() else 1
                schedule.every(hours).hours.do(self._scheduled_job)
                logger.info(f"设置每{hours}小时调度")
                
            else:
                raise ValueError(f"不支持的调度类型: {schedule_type}")
            
            # 启动调度线程
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("定时调度启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动定时调度失败: {e}")
            return False
    
    def start_continuous(self, interval_hours: int = 168) -> bool:
        """
        启动连续运行模式
        
        Args:
            interval_hours: 收集间隔（小时）
            
        Returns:
            是否启动成功
        """
        try:
            if self.is_running:
                logger.warning("调度器已在运行")
                return False
            
            self.is_running = True
            
            # 更新服务配置
            self.service.update_config(collection_frequency_hours=interval_hours)
            
            # 启动连续运行线程
            self.scheduler_thread = threading.Thread(
                target=self.service.run_continuous, 
                daemon=True
            )
            self.scheduler_thread.start()
            
            logger.info(f"连续运行模式启动成功，间隔: {interval_hours} 小时")
            return True
            
        except Exception as e:
            logger.error(f"启动连续运行模式失败: {e}")
            return False
    
    def stop(self):
        """停止调度器"""
        if not self.is_running:
            logger.info("调度器未在运行")
            return
        
        logger.info("正在停止调度器...")
        
        self.is_running = False
        schedule.clear()
        
        # 停止服务
        self.service.stop()
        
        # 等待线程结束
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("调度器已停止")
    
    def _run_scheduler(self):
        """运行调度器主循环"""
        logger.info("调度器主循环启动")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
                
        except Exception as e:
            logger.error(f"调度器运行出错: {e}")
        finally:
            logger.info("调度器主循环结束")
    
    def _scheduled_job(self):
        """定时任务执行函数"""
        try:
            logger.info("执行定时数据收集任务")
            results = self.service.collect_data_once()
            
            # 记录执行结果
            logger.info(f"定时任务完成: 成功 {results['success_count']}, "
                       f"失败 {results['failed_count']}, "
                       f"记录数 {results['total_records']}")
            
        except Exception as e:
            logger.error(f"定时任务执行失败: {e}")
    
    def get_next_run_time(self) -> Optional[datetime]:
        """获取下次运行时间"""
        try:
            if not schedule.jobs:
                return None
            
            next_job = schedule.next_run()
            return next_job
            
        except Exception as e:
            logger.error(f"获取下次运行时间失败: {e}")
            return None
    
    def get_status(self) -> dict:
        """获取调度器状态"""
        return {
            'is_running': self.is_running,
            'scheduled_jobs': len(schedule.jobs),
            'next_run_time': self.get_next_run_time(),
            'thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False,
            'service_status': self.service.get_status()
        }
    
    def add_custom_job(self, job_func: Callable, schedule_rule: str) -> bool:
        """
        添加自定义定时任务
        
        Args:
            job_func: 任务函数
            schedule_rule: 调度规则描述
            
        Returns:
            是否添加成功
        """
        try:
            # 这里可以根据schedule_rule解析并添加任务
            # 示例实现：每天特定时间执行
            if schedule_rule.startswith("daily_"):
                time_str = schedule_rule.replace("daily_", "")
                schedule.every().day.at(time_str).do(job_func)
                logger.info(f"添加自定义任务: {job_func.__name__} at {time_str}")
                return True
            else:
                logger.warning(f"不支持的调度规则: {schedule_rule}")
                return False
                
        except Exception as e:
            logger.error(f"添加自定义任务失败: {e}")
            return False
    
    def wait_for_completion(self):
        """等待调度器完成（阻塞模式）"""
        try:
            if self.scheduler_thread:
                self.scheduler_thread.join()
            logger.info("调度器执行完成")
        except KeyboardInterrupt:
            logger.info("接收到中断信号")
            self.stop()


# 便捷函数
def create_and_start_service(config_path: str = "config/data_collection.json",
                           schedule_type: str = "weekly",
                           schedule_time: str = "02:00",
                           weekday: str = "sunday") -> DataScheduler:
    """
    创建并启动数据收集服务
    
    Args:
        config_path: 配置文件路径
        schedule_type: 调度类型
        schedule_time: 执行时间
        weekday: 星期几
        
    Returns:
        调度器实例
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_collection.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 创建服务和调度器
    service = DataCollectionService(config_path)
    scheduler = DataScheduler(service)
    
    # 启动调度
    if schedule_type == "continuous":
        interval_hours = int(schedule_time) if schedule_time.isdigit() else 168
        scheduler.start_continuous(interval_hours)
    else:
        scheduler.start_scheduled(schedule_type, schedule_time, weekday)
    
    return scheduler


def run_once(config_path: str = "config/data_collection.json") -> dict:
    """
    执行一次数据收集
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        收集结果
    """
    logging.basicConfig(level=logging.INFO)
    
    service = DataCollectionService(config_path)
    return service.collect_data_once()