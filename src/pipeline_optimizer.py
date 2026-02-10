"""
Pipeline Optimizer - Main Entry Point

Combines feature engineering, failure prediction, and optimization
recommendations into a unified interface.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from .feature_engineer import FeatureEngineer, PipelineFeatures
from .failure_predictor import FailurePredictor, FailurePrediction


@dataclass
class OptimizationRecommendation:
    """Single optimization recommendation."""

    action: str
    priority: str  # high, medium, low
    expected_impact: str
    estimated_improvement: float  # Percentage reduction in failure risk
    implementation_effort: str  # easy, medium, hard


@dataclass
class PipelinePrediction:
    """Complete prediction result for a pipeline."""

    pipeline_id: str
    scheduled_time: datetime

    # Failure prediction
    failure_prob: float
    risk_level: str
    confidence: float

    # Duration prediction
    expected_duration: float
    duration_range: tuple[float, float]  # 95% CI

    # Recommendations
    recommendations: list[OptimizationRecommendation]

    # Features used
    features: PipelineFeatures


@dataclass
class SystemHealth:
    """Overall system health assessment."""

    total_pipelines: int
    healthy_count: int
    at_risk_count: int
    critical_count: int
    overall_health_score: float
    next_24h_risk: list[dict]


class PipelineOptimizer:
    """
    Main interface for pipeline prediction and optimization.

    Example:
        >>> optimizer = PipelineOptimizer()
        >>> optimizer.train("history.parquet")
        >>> pred = optimizer.predict("etl_daily", datetime.now())
        >>> print(f"Failure risk: {pred.failure_prob:.1%}")
    """

    def __init__(self):
        """Initialize optimizer components."""
        self.feature_engineer = FeatureEngineer()
        self.failure_predictor = FailurePredictor()
        self.is_trained = False

    def train(
        self,
        history_data: pd.DataFrame | str,
        target_column: str = "success"
    ) -> dict:
        """
        Train prediction models on historical data.

        Args:
            history_data: DataFrame or path to parquet/csv file
            target_column: Column indicating success/failure

        Returns:
            Dictionary with training metrics
        """
        # Load data if path provided
        if isinstance(history_data, str):
            if history_data.endswith('.parquet'):
                history_data = pd.read_parquet(history_data)
            else:
                history_data = pd.read_csv(history_data)

        # Ensure required columns
        required = ['pipeline_id', 'start_time', 'duration_minutes', target_column]
        missing = [c for c in required if c not in history_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Load history into feature engineer
        self.feature_engineer.load_history(history_data)

        # Engineer features for all historical runs
        features_list = []
        for _, row in history_data.iterrows():
            features = self.feature_engineer.engineer_features(
                pipeline_id=row['pipeline_id'],
                scheduled_time=row['start_time'],
                current_metrics=row.to_dict()
            )
            features_list.append(features.feature_vector)

        X = pd.DataFrame(
            features_list,
            columns=self.feature_engineer.FEATURE_COLUMNS
        )

        # Invert success to get failure (1 = failure)
        y = (~history_data[target_column].astype(bool)).astype(int)

        # Train failure predictor
        metrics = self.failure_predictor.train(X, y)

        self.is_trained = True

        return {
            "auc_roc": metrics.auc_roc,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "cv_scores": metrics.cv_scores,
            "samples_trained": len(history_data)
        }

    def predict(
        self,
        pipeline_id: str,
        scheduled_time: datetime,
        current_metrics: Optional[dict] = None
    ) -> PipelinePrediction:
        """
        Predict pipeline outcome and get recommendations.

        Args:
            pipeline_id: Pipeline identifier
            scheduled_time: Scheduled execution time
            current_metrics: Current system metrics

        Returns:
            PipelinePrediction with risk and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Engineer features
        features = self.feature_engineer.engineer_features(
            pipeline_id=pipeline_id,
            scheduled_time=scheduled_time,
            current_metrics=current_metrics
        )

        # Get failure prediction
        failure_pred = self.failure_predictor.predict(
            features.feature_vector,
            pipeline_id=pipeline_id,
            scheduled_time=scheduled_time
        )

        # Estimate duration
        expected_duration = features.avg_duration_7d or features.avg_duration_30d or 60
        duration_range = (
            expected_duration * 0.7,
            expected_duration * 1.5
        )

        # Generate recommendations
        recommendations = self.recommend(
            pipeline_id=pipeline_id,
            features=features,
            failure_prob=failure_pred.failure_probability
        )

        return PipelinePrediction(
            pipeline_id=pipeline_id,
            scheduled_time=scheduled_time,
            failure_prob=failure_pred.failure_probability,
            risk_level=failure_pred.risk_level,
            confidence=failure_pred.confidence,
            expected_duration=expected_duration,
            duration_range=duration_range,
            recommendations=recommendations,
            features=features
        )

    def recommend(
        self,
        pipeline_id: str,
        features: Optional[PipelineFeatures] = None,
        failure_prob: float = 0.0
    ) -> list[OptimizationRecommendation]:
        """
        Generate optimization recommendations.

        Args:
            pipeline_id: Pipeline identifier
            features: Pre-computed features
            failure_prob: Predicted failure probability

        Returns:
            List of recommendations sorted by priority
        """
        recommendations = []

        if features is None:
            return recommendations

        # Memory recommendation
        if features.peak_memory_last_run > 0.85:
            recommendations.append(OptimizationRecommendation(
                action="Increase memory allocation by 25%",
                priority="high",
                expected_impact="Reduce OOM errors and improve stability",
                estimated_improvement=15.0,
                implementation_effort="easy"
            ))

        # Scheduling recommendation
        if features.concurrent_pipelines > 5:
            recommendations.append(OptimizationRecommendation(
                action="Reschedule to off-peak time (2-4 AM)",
                priority="medium",
                expected_impact="Reduce resource contention",
                estimated_improvement=10.0,
                implementation_effort="easy"
            ))

        # Duration trend recommendation
        if features.duration_trend > 0.2:
            recommendations.append(OptimizationRecommendation(
                action="Optimize slow queries or add partitioning",
                priority="high",
                expected_impact="Reduce execution time and timeout risk",
                estimated_improvement=20.0,
                implementation_effort="medium"
            ))

        # Success rate recommendation
        if features.success_rate_7d < 0.9:
            recommendations.append(OptimizationRecommendation(
                action="Review and fix root cause of recent failures",
                priority="high",
                expected_impact="Improve reliability",
                estimated_improvement=25.0,
                implementation_effort="medium"
            ))

        # IO recommendation
        if features.io_wait_last_run > 0.2:
            recommendations.append(OptimizationRecommendation(
                action="Optimize data access patterns or increase IOPS",
                priority="medium",
                expected_impact="Reduce IO bottlenecks",
                estimated_improvement=12.0,
                implementation_effort="medium"
            ))

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 2))

        return recommendations

    def get_system_health(
        self,
        pipeline_ids: list[str],
        horizon_hours: int = 24
    ) -> SystemHealth:
        """
        Get overall system health assessment.

        Args:
            pipeline_ids: List of pipeline IDs to assess
            horizon_hours: Look-ahead window in hours

        Returns:
            SystemHealth summary
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        now = datetime.now()
        predictions = []

        for pid in pipeline_ids:
            # Predict for next scheduled time
            scheduled = now + timedelta(hours=np.random.randint(1, horizon_hours))
            pred = self.predict(pid, scheduled)
            predictions.append({
                "pipeline_id": pid,
                "scheduled_time": scheduled,
                "failure_prob": pred.failure_prob,
                "risk_level": pred.risk_level
            })

        # Categorize
        healthy = sum(1 for p in predictions if p["risk_level"] == "low")
        at_risk = sum(1 for p in predictions if p["risk_level"] in ("medium", "high"))
        critical = sum(1 for p in predictions if p["risk_level"] == "critical")

        # Overall health score
        health_score = 100 - (
            sum(p["failure_prob"] for p in predictions) / len(predictions) * 100
        ) if predictions else 100

        # Sort by risk for next 24h view
        next_24h = sorted(predictions, key=lambda x: -x["failure_prob"])[:10]

        return SystemHealth(
            total_pipelines=len(pipeline_ids),
            healthy_count=healthy,
            at_risk_count=at_risk,
            critical_count=critical,
            overall_health_score=health_score,
            next_24h_risk=next_24h
        )

    def save_model(self, path: str) -> None:
        """Save trained model."""
        self.failure_predictor.save(path)

    def load_model(self, path: str) -> None:
        """Load trained model."""
        self.failure_predictor.load(path)
        self.is_trained = True


# Convenience functions
def train_optimizer(history_path: str) -> PipelineOptimizer:
    """Quick training of optimizer."""
    optimizer = PipelineOptimizer()
    optimizer.train(history_path)
    return optimizer


def predict_pipeline(
    optimizer: PipelineOptimizer,
    pipeline_id: str,
    scheduled_time: Optional[datetime] = None
) -> PipelinePrediction:
    """Quick prediction for a pipeline."""
    scheduled_time = scheduled_time or datetime.now() + timedelta(hours=1)
    return optimizer.predict(pipeline_id, scheduled_time)
