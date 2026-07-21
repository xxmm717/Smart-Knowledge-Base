"""
导入流程统一日志模块

提供统一的 logger 实例，整个 import_process 包都通过此模块打日志。
用法：
    from processor.import_process.logger import logger

    logger.info("xxx")
    logger.error("xxx")
"""

import logging
import sys


# 日志格式
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """
    配置日志格式（全局生效一次即可）

    Args:
        level: 日志级别，默认 INFO
    """
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
        stream=sys.stdout,
    )


# 包级别的 logger，所有模块统一使用它
logger = logging.getLogger("import_process")
