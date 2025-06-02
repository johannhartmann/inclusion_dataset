"""Logging setup and configuration."""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file handler for all logs
    logger.add(
        Path(log_dir) / "inclusion_dataset.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # Add separate error log
    logger.add(
        Path(log_dir) / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="5 MB",
        retention="60 days"
    )
    
    # Add pipeline progress log
    logger.add(
        Path(log_dir) / "pipeline_progress.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        filter=lambda record: "PROGRESS" in record["extra"],
        rotation="5 MB",
        retention="30 days"
    )
    
    logger.info(f"Logging setup complete. Logs will be saved to: {log_dir}")