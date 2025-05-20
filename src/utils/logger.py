"""
日志工具模块

提供统一的日志记录接口和日志格式化功能。

Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""

import inspect
import logging
import logging.handlers
import os
from collections.abc import MutableMapping
from pathlib import Path


class CustomLoggerAdapter(logging.LoggerAdapter):
    """自定义日志适配器，添加文件名和行号信息"""

    def process(self, msg: str, kwargs: MutableMapping) -> tuple[str, MutableMapping]:
        """格式化日志消息，添加文件名和行号信息"""
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
        filename = "unknown"
        lineno = 0
        while frame:
            filename = Path(frame.f_code.co_filename).name
            if filename not in ("logger.py", "__init__.py"):
                lineno = frame.f_lineno
                break
            frame = frame.f_back
        return f"{filename}:{lineno} - {msg}", kwargs


class Logger:
    """日志管理类，提供统一的日志记录接口"""

    # 确保日志目录存在
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志器
    logger = logging.getLogger("sysHAX")
    logger.setLevel(logging.INFO)

    # 检查是否已经有处理器，避免重复添加
    if not logger.handlers:
        # 使用RotatingFileHandler处理日志文件滚动
        file_path = log_dir / "sysHAX.log"
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,  # 保留5个旧日志文件
            encoding="UTF-8",
        )
        file_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # 添加控制台处理器 - 默认WARNING级别，只显示警告和错误
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)

    # 自定义日志适配器
    adapter = CustomLoggerAdapter(logger, {})

    @classmethod
    def debug(cls, msg: str) -> None:
        """记录调试信息"""
        cls.adapter.debug(msg)

    @classmethod
    def info(cls, msg: str) -> None:
        """记录一般信息"""
        cls.adapter.info(msg)

    @classmethod
    def warning(cls, msg: str) -> None:
        """记录警告信息"""
        cls.adapter.warning(msg)

    @classmethod
    def error(cls, msg: str, *, exc_info: bool = False) -> None:
        """
        记录错误信息

        Args:
            msg: 错误信息
            exc_info: 是否包含异常堆栈，默认为False

        """
        cls.adapter.error(msg, exc_info=exc_info)

    @classmethod
    def critical(cls, msg: str) -> None:
        """记录严重错误信息"""
        cls.adapter.critical(msg)

    @classmethod
    def set_level(cls, level: int) -> None:
        """
        设置日志级别

        Args:
            level: 日志级别，可使用logging.DEBUG、logging.INFO等

        """
        cls.logger.setLevel(level)
        for handler in cls.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                # 只修改控制台处理器的级别
                handler.setLevel(level)


# 根据环境变量设置日志级别
if os.environ.get("DEBUG", "").lower() in ("true", "1", "yes"):
    Logger.set_level(logging.DEBUG)
level = os.environ.get("LOG_LEVEL")
if level is not None:
    level_name = level.upper()
    if hasattr(logging, level_name):
        Logger.set_level(getattr(logging, level_name))
