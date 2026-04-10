"""
Einstieg für die V10-Pipeline: dieselbe öffentliche API wie ``pipeline_runner``.
"""
from __future__ import annotations

from lib.stock_rally_v10.pipeline_runner import (
    bind_step_functions,
    run_pipeline_default,
)

__all__ = ["bind_step_functions", "run_pipeline_default"]
