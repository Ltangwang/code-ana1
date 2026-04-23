"""Centralized logging configuration for the application."""

import sys
import logging
from pathlib import Path
from typing import Optional

import structlog
from structlog.types import Processor


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "structured",
    output_dir: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Format type ("structured" for JSON or "plain" for human-readable)
        output_dir: Directory for log files (None = console only)
        enable_console: Whether to output to console
    """
    # Create log directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure processors based on format
    if log_format == "structured":
        processors: list[Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Human-readable console format
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer()
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True
    )
    
    # Configure standard logging (for third-party libraries)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if enable_console else None,
        level=getattr(logging, log_level)
    )


def get_logger(name: str = __name__):
    """Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Structured logger
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, **kwargs):
        """Initialize log context.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context = kwargs
        self.tokens = []
    
    def __enter__(self):
        """Enter context."""
        for key, value in self.context.items():
            token = structlog.contextvars.bind_contextvars(**{key: value})
            self.tokens.append(token)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        for token in self.tokens:
            structlog.contextvars.unbind_contextvars(token)


# Convenience functions for common log patterns

def log_local_score(logger, fragment_location: str, score: float, reasoning: str):
    """Log local confidence score."""
    logger.info(
        "local_score",
        fragment=fragment_location,
        confidence=score,
        reasoning=reasoning
    )


def log_upload_decision(
    logger,
    fragment_location: str,
    should_upload: bool,
    reason: str,
    confidence: float,
    budget_remaining: float
):
    """Log upload decision."""
    logger.info(
        "upload_decision",
        fragment=fragment_location,
        upload=should_upload,
        reason=reason,
        confidence=confidence,
        budget_remaining_pct=budget_remaining
    )


def log_cloud_latency(
    logger,
    provider: str,
    model: str,
    latency_ms: float,
    tokens_used: int
):
    """Log cloud API latency."""
    logger.info(
        "cloud_latency",
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        tokens=tokens_used
    )


def log_refinement_delta(
    logger,
    fragment_location: str,
    original_confidence: float,
    final_confidence: float,
    confidence_boost: float
):
    """Log confidence improvement from cloud refinement."""
    logger.info(
        "refinement_delta",
        fragment=fragment_location,
        original_confidence=original_confidence,
        final_confidence=final_confidence,
        boost=confidence_boost
    )


def log_budget_update(
    logger,
    operation: str,
    cost: float,
    remaining: float,
    remaining_pct: float
):
    """Log budget update."""
    logger.info(
        "budget_update",
        operation=operation,
        cost=cost,
        remaining=remaining,
        remaining_pct=remaining_pct
    )


def log_error(logger, context: str, error: Exception, **kwargs):
    """Log error with context."""
    logger.error(
        context,
        error_type=type(error).__name__,
        error_message=str(error),
        **kwargs
    )


# Metrics logging

def log_analysis_start(logger, target: str, target_type: str = "file"):
    """Log start of analysis."""
    logger.info(
        "analysis_start",
        target=target,
        target_type=target_type
    )


def log_analysis_complete(
    logger,
    target: str,
    duration_ms: float,
    fragments_analyzed: int,
    issues_found: int,
    cloud_calls: int
):
    """Log completion of analysis."""
    logger.info(
        "analysis_complete",
        target=target,
        duration_ms=duration_ms,
        fragments=fragments_analyzed,
        issues=issues_found,
        cloud_calls=cloud_calls
    )


def log_hotspot_detection(
    logger,
    file_path: str,
    hotspot_count: int,
    top_scores: list[float]
):
    """Log hotspot detection results."""
    logger.info(
        "hotspot_detection",
        file=file_path,
        hotspot_count=hotspot_count,
        top_scores=top_scores[:5]  # Top 5
    )


def log_batch_processing(
    logger,
    batch_size: int,
    batch_index: int,
    total_batches: int
):
    """Log batch processing progress."""
    logger.info(
        "batch_processing",
        batch_size=batch_size,
        batch_index=batch_index,
        total_batches=total_batches,
        progress_pct=(batch_index / total_batches * 100)
    )

