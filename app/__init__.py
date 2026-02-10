"""
Predictive Pipeline Optimizer - App Package

Streamlit demo application for pipeline failure prediction.
"""

from app.demo_data import (
    generate_pipeline_history,
    get_pipeline_configs,
    get_upcoming_schedules,
    get_pipeline_summary
)

__all__ = [
    "generate_pipeline_history",
    "get_pipeline_configs",
    "get_upcoming_schedules",
    "get_pipeline_summary"
]
__version__ = "1.0.0"
