"""
Demo Data Generator for Predictive Pipeline Optimizer

Provides realistic sample pipeline execution data with
various patterns for demonstrating prediction capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for a simulated pipeline."""
    pipeline_id: str
    name: str
    category: str
    schedule: str
    avg_duration: float
    failure_rate: float
    resource_profile: str  # light, medium, heavy


# Sample pipeline configurations
SAMPLE_PIPELINES = [
    PipelineConfig("etl_sales_daily", "Daily Sales ETL", "Sales", "Daily 2:00 AM", 45, 0.05, "medium"),
    PipelineConfig("etl_inventory", "Inventory Sync", "Operations", "Daily 3:00 AM", 120, 0.15, "heavy"),
    PipelineConfig("etl_customer_360", "Customer 360 Build", "Marketing", "Daily 4:00 AM", 90, 0.08, "heavy"),
    PipelineConfig("etl_finance_report", "Finance Reporting", "Finance", "Daily 6:00 AM", 60, 0.03, "medium"),
    PipelineConfig("etl_web_analytics", "Web Analytics", "Digital", "Hourly", 15, 0.02, "light"),
    PipelineConfig("etl_product_catalog", "Product Catalog Update", "E-commerce", "Daily 1:00 AM", 30, 0.04, "light"),
    PipelineConfig("ml_recommendations", "ML Recommendations", "Data Science", "Daily 5:00 AM", 180, 0.12, "heavy"),
    PipelineConfig("etl_crm_sync", "CRM Data Sync", "Sales", "Every 4 hours", 25, 0.06, "light"),
    PipelineConfig("etl_warehouse_load", "Data Warehouse Load", "BI", "Daily 7:00 AM", 240, 0.10, "heavy"),
    PipelineConfig("etl_marketing_attribution", "Marketing Attribution", "Marketing", "Daily 8:00 AM", 75, 0.07, "medium"),
]


def get_pipeline_configs() -> list[PipelineConfig]:
    """Get list of sample pipeline configurations."""
    return SAMPLE_PIPELINES


def generate_pipeline_history(
    days: int = 90,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic pipeline execution history.

    Args:
        days: Number of days of history to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with pipeline execution history
    """
    np.random.seed(seed)

    records = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    for config in SAMPLE_PIPELINES:
        # Determine number of runs based on schedule
        if "Hourly" in config.schedule:
            runs_per_day = 24
        elif "Every 4 hours" in config.schedule:
            runs_per_day = 6
        else:
            runs_per_day = 1

        total_runs = days * runs_per_day

        for i in range(total_runs):
            # Calculate run time
            run_date = start_date + timedelta(days=i // runs_per_day)

            if runs_per_day == 1:
                # Extract hour from schedule
                hour = int(config.schedule.split()[1].split(":")[0])
            else:
                hour = (i % runs_per_day) * (24 // runs_per_day)

            run_time = run_date.replace(hour=hour, minute=0, second=0)

            # Generate duration with some variance
            base_duration = config.avg_duration
            duration_variance = base_duration * 0.3
            duration = max(5, np.random.normal(base_duration, duration_variance))

            # Determine success/failure
            # Increase failure rate for certain conditions
            failure_prob = config.failure_rate

            # Higher failure rate on weekends
            if run_time.weekday() >= 5:
                failure_prob *= 1.3

            # Higher failure rate at month end
            if run_time.day >= 28:
                failure_prob *= 1.5

            # Higher failure rate during peak hours
            if 2 <= run_time.hour <= 7:
                failure_prob *= 1.2

            success = np.random.random() > failure_prob

            # Generate resource metrics
            if config.resource_profile == "light":
                avg_cpu = np.random.uniform(0.2, 0.4)
                avg_memory = np.random.uniform(0.3, 0.5)
                peak_memory = avg_memory + np.random.uniform(0.1, 0.2)
            elif config.resource_profile == "medium":
                avg_cpu = np.random.uniform(0.4, 0.6)
                avg_memory = np.random.uniform(0.5, 0.7)
                peak_memory = avg_memory + np.random.uniform(0.1, 0.25)
            else:  # heavy
                avg_cpu = np.random.uniform(0.6, 0.85)
                avg_memory = np.random.uniform(0.7, 0.9)
                peak_memory = min(0.99, avg_memory + np.random.uniform(0.05, 0.15))

            # Failed runs often have high resource usage
            if not success:
                peak_memory = min(0.99, peak_memory * 1.2)
                duration *= 1.5  # Failed runs tend to run longer before failing

            io_wait = np.random.uniform(0.05, 0.25)

            # Data volume (varies by pipeline)
            data_volume = np.random.exponential(config.avg_duration / 10)
            row_count = int(data_volume * 1_000_000)

            records.append({
                'pipeline_id': config.pipeline_id,
                'pipeline_name': config.name,
                'category': config.category,
                'start_time': run_time,
                'end_time': run_time + timedelta(minutes=duration),
                'duration_minutes': round(duration, 2),
                'success': success,
                'avg_cpu': round(avg_cpu, 3),
                'avg_memory': round(avg_memory, 3),
                'peak_memory': round(peak_memory, 3),
                'io_wait': round(io_wait, 3),
                'data_volume_gb': round(data_volume, 2),
                'row_count': row_count,
                'error_message': None if success else np.random.choice([
                    "OutOfMemoryError: Java heap space",
                    "Connection timeout to database",
                    "Query execution exceeded time limit",
                    "Upstream dependency failed",
                    "Resource quota exceeded",
                    "Data validation failed",
                    "Schema mismatch detected"
                ])
            })

    df = pd.DataFrame(records)
    return df.sort_values('start_time').reset_index(drop=True)


def generate_current_metrics(pipeline_id: str, seed: Optional[int] = None) -> dict:
    """
    Generate current system metrics for a pipeline.

    Args:
        pipeline_id: Pipeline identifier
        seed: Optional random seed

    Returns:
        Dictionary with current metrics
    """
    if seed:
        np.random.seed(seed)

    config = next((p for p in SAMPLE_PIPELINES if p.pipeline_id == pipeline_id), None)

    if config is None:
        return {}

    if config.resource_profile == "light":
        base_cpu, base_memory = 0.3, 0.4
    elif config.resource_profile == "medium":
        base_cpu, base_memory = 0.5, 0.6
    else:
        base_cpu, base_memory = 0.7, 0.8

    return {
        'avg_cpu': round(base_cpu + np.random.uniform(-0.1, 0.1), 3),
        'avg_memory': round(base_memory + np.random.uniform(-0.1, 0.1), 3),
        'peak_memory': round(min(0.99, base_memory + np.random.uniform(0.05, 0.2)), 3),
        'io_wait': round(np.random.uniform(0.05, 0.2), 3),
        'concurrent_jobs': np.random.randint(3, 12)
    }


def get_upcoming_schedules(hours: int = 24) -> pd.DataFrame:
    """
    Get upcoming pipeline schedules.

    Args:
        hours: Look-ahead window in hours

    Returns:
        DataFrame with upcoming scheduled runs
    """
    now = datetime.now()
    schedules = []

    for config in SAMPLE_PIPELINES:
        if "Hourly" in config.schedule:
            # Next few hours
            for h in range(min(hours, 6)):
                next_run = now.replace(minute=0, second=0) + timedelta(hours=h+1)
                schedules.append({
                    'pipeline_id': config.pipeline_id,
                    'pipeline_name': config.name,
                    'category': config.category,
                    'scheduled_time': next_run,
                    'avg_duration': config.avg_duration,
                    'resource_profile': config.resource_profile
                })
        elif "Every 4 hours" in config.schedule:
            for h in range(0, hours, 4):
                next_run = now.replace(minute=0, second=0) + timedelta(hours=h+1)
                schedules.append({
                    'pipeline_id': config.pipeline_id,
                    'pipeline_name': config.name,
                    'category': config.category,
                    'scheduled_time': next_run,
                    'avg_duration': config.avg_duration,
                    'resource_profile': config.resource_profile
                })
        else:
            # Daily - extract hour
            hour = int(config.schedule.split()[1].split(":")[0])
            next_run = now.replace(hour=hour, minute=0, second=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            schedules.append({
                'pipeline_id': config.pipeline_id,
                'pipeline_name': config.name,
                'category': config.category,
                'scheduled_time': next_run,
                'avg_duration': config.avg_duration,
                'resource_profile': config.resource_profile
            })

    df = pd.DataFrame(schedules)
    return df.sort_values('scheduled_time').reset_index(drop=True)


def get_pipeline_summary(history: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for each pipeline.

    Args:
        history: Pipeline execution history

    Returns:
        DataFrame with summary per pipeline
    """
    summary = history.groupby('pipeline_id').agg({
        'pipeline_name': 'first',
        'category': 'first',
        'duration_minutes': ['mean', 'std', 'min', 'max'],
        'success': ['mean', 'sum', 'count'],
        'avg_cpu': 'mean',
        'avg_memory': 'mean',
        'peak_memory': 'max'
    }).round(2)

    # Flatten column names
    summary.columns = [
        'name', 'category',
        'avg_duration', 'std_duration', 'min_duration', 'max_duration',
        'success_rate', 'successful_runs', 'total_runs',
        'avg_cpu', 'avg_memory', 'peak_memory'
    ]

    summary['failure_count'] = summary['total_runs'] - summary['successful_runs']
    summary['success_rate'] = (summary['success_rate'] * 100).round(1)

    return summary.reset_index()
