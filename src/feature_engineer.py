"""
Feature Engineer Module

Transforms raw pipeline metrics into predictive features
with temporal, resource, and contextual signals.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


@dataclass
class PipelineFeatures:
    """Engineered features for a pipeline."""

    pipeline_id: str
    scheduled_time: datetime

    # Temporal features
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_month_end: bool = False
    minutes_since_last_run: float = 0

    # Historical performance
    avg_duration_7d: float = 0
    avg_duration_30d: float = 0
    success_rate_7d: float = 0
    success_rate_30d: float = 0
    failure_count_7d: int = 0
    duration_trend: float = 0  # Positive = getting slower

    # Resource features
    avg_cpu_last_run: float = 0
    avg_memory_last_run: float = 0
    peak_memory_last_run: float = 0
    io_wait_last_run: float = 0

    # Context features
    concurrent_pipelines: int = 0
    data_volume_gb: float = 0
    row_count_estimate: int = 0
    upstream_failures_24h: int = 0

    # Derived features
    risk_score: float = 0
    feature_vector: list = field(default_factory=list)


class FeatureEngineer:
    """
    Transforms raw metrics into ML-ready features.
    """

    FEATURE_COLUMNS = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_month_end',
        'minutes_since_last_run', 'avg_duration_7d', 'avg_duration_30d',
        'success_rate_7d', 'success_rate_30d', 'failure_count_7d',
        'duration_trend', 'avg_cpu_last_run', 'avg_memory_last_run',
        'peak_memory_last_run', 'io_wait_last_run', 'concurrent_pipelines',
        'data_volume_gb', 'row_count_estimate', 'upstream_failures_24h'
    ]

    def __init__(self, history_df: Optional[pd.DataFrame] = None):
        """
        Initialize feature engineer.

        Args:
            history_df: Historical pipeline execution data
        """
        self.history = history_df
        self._pipeline_stats = {}

    def load_history(self, history_df: pd.DataFrame) -> None:
        """Load historical execution data."""
        self.history = history_df
        self._precompute_stats()

    def _precompute_stats(self) -> None:
        """Precompute pipeline statistics for fast lookup."""
        if self.history is None:
            return

        for pipeline_id in self.history['pipeline_id'].unique():
            pipeline_data = self.history[self.history['pipeline_id'] == pipeline_id]
            self._pipeline_stats[pipeline_id] = {
                'mean_duration': pipeline_data['duration_minutes'].mean(),
                'std_duration': pipeline_data['duration_minutes'].std(),
                'success_rate': pipeline_data['success'].mean(),
                'run_count': len(pipeline_data)
            }

    def engineer_features(
        self,
        pipeline_id: str,
        scheduled_time: datetime,
        current_metrics: Optional[dict] = None
    ) -> PipelineFeatures:
        """
        Engineer features for a pipeline run.

        Args:
            pipeline_id: Pipeline identifier
            scheduled_time: Scheduled execution time
            current_metrics: Current system metrics

        Returns:
            PipelineFeatures with all engineered features
        """
        features = PipelineFeatures(
            pipeline_id=pipeline_id,
            scheduled_time=scheduled_time
        )

        # Temporal features
        features = self._add_temporal_features(features, scheduled_time)

        # Historical features
        if self.history is not None:
            features = self._add_historical_features(features, pipeline_id, scheduled_time)

        # Resource features
        if current_metrics:
            features = self._add_resource_features(features, current_metrics)

        # Context features
        features = self._add_context_features(features, scheduled_time)

        # Build feature vector
        features.feature_vector = self._build_feature_vector(features)

        # Calculate risk score
        features.risk_score = self._calculate_risk_score(features)

        return features

    def _add_temporal_features(
        self,
        features: PipelineFeatures,
        scheduled_time: datetime
    ) -> PipelineFeatures:
        """Add time-based features."""
        features.hour_of_day = scheduled_time.hour
        features.day_of_week = scheduled_time.weekday()
        features.is_weekend = scheduled_time.weekday() >= 5
        features.is_month_end = scheduled_time.day >= 28

        return features

    def _add_historical_features(
        self,
        features: PipelineFeatures,
        pipeline_id: str,
        current_time: datetime
    ) -> PipelineFeatures:
        """Add features from historical data."""
        pipeline_history = self.history[
            self.history['pipeline_id'] == pipeline_id
        ].copy()

        if len(pipeline_history) == 0:
            return features

        # Convert to datetime if needed
        if 'start_time' in pipeline_history.columns:
            pipeline_history['start_time'] = pd.to_datetime(pipeline_history['start_time'])

        # Last 7 days
        cutoff_7d = current_time - timedelta(days=7)
        recent_7d = pipeline_history[pipeline_history['start_time'] >= cutoff_7d]

        if len(recent_7d) > 0:
            features.avg_duration_7d = recent_7d['duration_minutes'].mean()
            features.success_rate_7d = recent_7d['success'].mean() if 'success' in recent_7d else 1.0
            features.failure_count_7d = (~recent_7d['success']).sum() if 'success' in recent_7d else 0

        # Last 30 days
        cutoff_30d = current_time - timedelta(days=30)
        recent_30d = pipeline_history[pipeline_history['start_time'] >= cutoff_30d]

        if len(recent_30d) > 0:
            features.avg_duration_30d = recent_30d['duration_minutes'].mean()
            features.success_rate_30d = recent_30d['success'].mean() if 'success' in recent_30d else 1.0

        # Duration trend (comparing recent to older)
        if features.avg_duration_7d > 0 and features.avg_duration_30d > 0:
            features.duration_trend = (
                (features.avg_duration_7d - features.avg_duration_30d) /
                features.avg_duration_30d
            )

        # Time since last run
        if len(pipeline_history) > 0:
            last_run = pipeline_history['start_time'].max()
            features.minutes_since_last_run = (current_time - last_run).total_seconds() / 60

        return features

    def _add_resource_features(
        self,
        features: PipelineFeatures,
        metrics: dict
    ) -> PipelineFeatures:
        """Add resource utilization features."""
        features.avg_cpu_last_run = metrics.get('avg_cpu', 0)
        features.avg_memory_last_run = metrics.get('avg_memory', 0)
        features.peak_memory_last_run = metrics.get('peak_memory', 0)
        features.io_wait_last_run = metrics.get('io_wait', 0)

        return features

    def _add_context_features(
        self,
        features: PipelineFeatures,
        scheduled_time: datetime
    ) -> PipelineFeatures:
        """Add contextual features."""
        # Count concurrent pipelines at scheduled time
        if self.history is not None:
            # Pipelines running at similar times historically
            similar_hour = self.history[
                self.history['start_time'].dt.hour == scheduled_time.hour
            ]
            features.concurrent_pipelines = similar_hour['pipeline_id'].nunique()

        return features

    def _build_feature_vector(self, features: PipelineFeatures) -> list:
        """Build numeric feature vector for ML model."""
        return [
            features.hour_of_day,
            features.day_of_week,
            int(features.is_weekend),
            int(features.is_month_end),
            features.minutes_since_last_run,
            features.avg_duration_7d,
            features.avg_duration_30d,
            features.success_rate_7d,
            features.success_rate_30d,
            features.failure_count_7d,
            features.duration_trend,
            features.avg_cpu_last_run,
            features.avg_memory_last_run,
            features.peak_memory_last_run,
            features.io_wait_last_run,
            features.concurrent_pipelines,
            features.data_volume_gb,
            features.row_count_estimate,
            features.upstream_failures_24h
        ]

    def _calculate_risk_score(self, features: PipelineFeatures) -> float:
        """Calculate preliminary risk score from features."""
        risk = 0.0

        # Low success rate = high risk
        if features.success_rate_7d < 0.9:
            risk += (1 - features.success_rate_7d) * 0.4

        # Recent failures
        risk += min(features.failure_count_7d * 0.1, 0.3)

        # Duration trend (getting slower = risky)
        if features.duration_trend > 0.2:
            risk += 0.2

        # High resource usage
        if features.peak_memory_last_run > 0.9:
            risk += 0.1

        return min(1.0, risk)

    def batch_engineer(
        self,
        pipeline_ids: list[str],
        scheduled_times: list[datetime]
    ) -> pd.DataFrame:
        """Engineer features for multiple pipelines."""
        records = []
        for pid, stime in zip(pipeline_ids, scheduled_times):
            features = self.engineer_features(pid, stime)
            record = {col: getattr(features, col) for col in self.FEATURE_COLUMNS}
            record['pipeline_id'] = pid
            record['scheduled_time'] = stime
            records.append(record)

        return pd.DataFrame(records)
