# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hesam Haddad
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

LOG_FILE = "logs/flexilearn.log"

# -----------------------------------------------------------------------------
# Console (rich only — no logger attached)
# -----------------------------------------------------------------------------
console = Console()


# -----------------------------------------------------------------------------
# Logger (file only — no console handler)
# -----------------------------------------------------------------------------
def _setup_logger(name: str = "flexilearn", log_file: str = LOG_FILE) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return logger

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logger


logger = _setup_logger()


# -----------------------------------------------------------------------------
# Monitoring API
# -----------------------------------------------------------------------------
def debug(msg: str):
    """Silent — file only. Use for internal diagnostic events."""
    logger.debug(msg)


def info(msg: str):
    console.print(f"[bold cyan]INFO[/] {msg}")
    logger.info(msg)


def success(msg: str):
    console.print(f"[bold green]SUCCESS[/] {msg}")
    logger.info(f"SUCCESS | {msg}")


def warn(msg: str):
    console.print(f"[bold yellow]WARNING[/] {msg}")
    logger.warning(msg)


def error(msg: str):
    console.print(f"[bold red]ERROR[/] {msg}")
    logger.error(msg)


def section(title: str):
    console.rule(f"[bold cyan]{title}")
    logger.info(f"--- {title} ---")


# -----------------------------------------------------------------------------
# Rich helpers
# -----------------------------------------------------------------------------
def print_metrics(metrics: dict[str, Any], title: str = "Metrics"):
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    console.print(table)


def create_progress() -> Progress:
    """Progress bar suitable for epochs or steps."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    )
