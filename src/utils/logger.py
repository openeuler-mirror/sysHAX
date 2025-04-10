# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import logging.handlers
import os
import logging
import gzip
import shutil
from datetime import datetime, timedelta
import inspect
import threading

def archive_logs(base_log_path, max_days=7):
    """归档旧的日志文件，转换为gzip格式以节省空间"""
    if not os.path.exists(base_log_path):
        return
        
    for filename in os.listdir(base_log_path):
        file_path = os.path.join(base_log_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.log'):
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if datetime.now() - modified_time > timedelta(days=max_days):
                with open(file_path, 'rb') as f_in:
                    with gzip.open(f'{file_path}.gz', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(file_path)

def schedule_archive_logs(interval=86400, base_log_path='logs'):
    """定期调度日志归档任务"""
    def wrapper():
        archive_logs(base_log_path)
        threading.Timer(interval, wrapper).start()
    wrapper()

class CustomLoggerAdapter(logging.LoggerAdapter):
    """自定义日志适配器，添加文件名和行号信息"""
    def process(self, msg, kwargs):
        frame = inspect.currentframe().f_back
        filename = 'unknown'
        lineno = 0
        while frame:
            filename = os.path.basename(frame.f_code.co_filename)
            if filename not in ('logger.py', '__init__.py'):
                lineno = frame.f_lineno
                break
            frame = frame.f_back
        return f"{filename}:{lineno} - {msg}", kwargs

class Logger:
    """日志管理类，提供统一的日志记录接口"""
    
    # 确保日志目录存在
    log_dir = os.path.join('.', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志器
    logger = logging.getLogger('pd_separation')
    logger.setLevel(logging.INFO)
    
    # 检查是否已经有处理器，避免重复添加
    if not logger.handlers:
        # 使用RotatingFileHandler处理日志文件滚动
        file_path = os.path.join(log_dir, 'pd_separation.log')
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,              # 保留5个旧日志文件
            encoding='UTF-8'
        )
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        # 添加控制台处理器 - 设置为WARNING级别，只显示警告和错误
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)  # 只显示WARNING及以上级别
        logger.addHandler(console_handler)
    
    # 自定义日志适配器
    adapter = CustomLoggerAdapter(logger, {})
    
    @classmethod
    def debug(cls, log_msg):
        """记录调试信息"""
        cls.adapter.debug(log_msg)

    @classmethod
    def info(cls, log_msg):
        """记录一般信息，只写入文件而不打印到控制台"""
        cls.adapter.info(log_msg)
        # 不再使用print输出到控制台

    @classmethod
    def warning(cls, log_msg):
        """记录警告信息"""
        cls.adapter.warning(log_msg)
        # 警告级别已经会由控制台处理器输出，无需额外print

    @classmethod
    def error(cls, log_msg, exc_info=False):
        """记录错误信息
        
        Args:
            log_msg: 错误信息
            exc_info: 是否包含异常堆栈，默认为False
        """
        cls.adapter.error(log_msg, exc_info=exc_info)
        # 错误级别已经会由控制台处理器输出，无需额外print

    @classmethod
    def critical(cls, log_msg):
        """记录严重错误信息"""
        cls.adapter.critical(log_msg)
        # 严重级别已经会由控制台处理器输出，无需额外print

    @classmethod
    def set_level(cls, level):
        """设置日志级别"""
        cls.logger.setLevel(level)
        for handler in cls.logger.handlers:
            handler.setLevel(level)

# 启动时安排日志归档任务
# schedule_archive_logs()

# 默认日志级别
if os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes'):
    Logger.set_level(logging.DEBUG) 