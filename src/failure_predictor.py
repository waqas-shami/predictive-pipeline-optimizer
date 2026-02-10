"""
Failure Predictor Module

XGBoost-based model for predicting pipeline failure probability.
"""

import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import pickle

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report


@dataclass
class FailurePrediction:
    """Prediction result for pipeline failure."""

    pipeline_id: str
    scheduled_time: datetime
    failure_probability: float
    risk_level: str  # low, medium, high, critical
    confidence: float
    contributing_factors: list[dict]
    recommendation: str


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    cv_scores: list[float]


class FailurePredictor:
    """
    Predicts pipeline failure probability using XGBoost.
    """

    FEATURE_COLUMNS = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_month_end',
        'minutes_since_last_run', 'avg_duration_7d', 'avg_duration_30d',
        'success_rate_7d', 'success_rate_30d', 'failure_count_7d',
        'duration_trend', 'avg_cpu_last_run', 'avg_memory_last_run',
        'peak_memory_last_run', 'io_wait_last_run', 'concurrent_pipelines',
        'data_volume_gb', 'row_count_estimate', 'upstream_failures_24h'
    ]

    RISK_THRESHOLDS = {
        'low': 0.2,
        'medium': 0.5,
        'high': 0.8,
        'critical': 1.0
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize failure predictor.

        Args:
            model_path: Path to saved model file
        """
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42
        )
        self.is_trained = False
        self.metrics: Optional[ModelMetrics] = None
        self.feature_importance: Optional[dict] = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> ModelMetrics:
        """
        Train the failure prediction model.

        Args:
            X: Feature DataFrame
            y: Target series (1 = failure, 0 = success)
            validation_split: Fraction for validation

        Returns:
            ModelMetrics with performance scores
        """
        # Ensure feature columns exist
        X = X[self.FEATURE_COLUMNS].copy()

        # Handle missing values
        X = X.fillna(0)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = self.model.predict(X_val)

        # Calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)

        report = classification_report(y_val, y_pred, output_dict=True)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring='roc_auc'
        ).tolist()

        self.metrics = ModelMetrics(
            auc_roc=auc,
            precision=report['1']['precision'] if '1' in report else 0,
            recall=report['1']['recall'] if '1' in report else 0,
            f1_score=report['1']['f1-score'] if '1' in report else 0,
            cv_scores=cv_scores
        )

        # Feature importance
        self.feature_importance = dict(zip(
            self.FEATURE_COLUMNS,
            self.model.feature_importances_
        ))

        self.is_trained = True
        return self.metrics

    def predict(
        self,
        features: pd.DataFrame | list,
        pipeline_id: str = "unknown",
        scheduled_time: Optional[datetime] = None
    ) -> FailurePrediction:
        """
        Predict failure probability for a pipeline.

        Args:
            features: Feature values (DataFrame or list)
            pipeline_id: Pipeline identifier
            scheduled_time: Scheduled execution time

        Returns:
            FailurePrediction with probability and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to DataFrame if needed
        if isinstance(features, list):
            features = pd.DataFrame([features], columns=self.FEATURE_COLUMNS)

        # Ensure correct columns
        features = features[self.FEATURE_COLUMNS].fillna(0)

        # Predict
        failure_prob = self.model.predict_proba(features)[0, 1]

        # Determine risk level
        risk_level = self._determine_risk_level(failure_prob)

        # Get contributing factors
        contributing_factors = self._get_contributing_factors(features.iloc[0])

        # Generate recommendation
        recommendation = self._generate_recommendation(
            failure_prob, risk_level, contributing_factors
        )

        # Calculate confidence based on historical accuracy
        confidence = self._calculate_confidence(features.iloc[0])

        return FailurePrediction(
            pipeline_id=pipeline_id,
            scheduled_time=scheduled_time or datetime.now(),
            failure_probability=failure_prob,
            risk_level=risk_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            recommendation=recommendation
        )

    def predict_batch(self, features: pd.DataFrame) -> list[float]:
        """Predict failure probabilities for multiple pipelines."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        features = features[self.FEATURE_COLUMNS].fillna(0)
        return self.model.predict_proba(features)[:, 1].tolist()

    def _determine_risk_level(self, probability: float) -> str:
        """Map probability to risk level."""
        for level, threshold in self.RISK_THRESHOLDS.items():
            if probability < threshold:
                return level
        return "critical"

    def _get_contributing_factors(self, features: pd.Series) -> list[dict]:
        """Identify top factors contributing to prediction."""
        if self.feature_importance is None:
            return []

        factors = []

        # Get top contributing features
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for feature, importance in sorted_importance[:5]:
            value = features[feature]
            contribution = self._explain_contribution(feature, value, importance)
            if contribution:
                factors.append(contribution)

        return factors

    def _explain_contribution(
        self,
        feature: str,
        value: float,
        importance: float
    ) -> Optional[dict]:
        """Generate human-readable explanation for feature contribution."""
        explanations = {
            'success_rate_7d': lambda v: f"Recent success rate is {v*100:.0f}%" if v < 0.95 else None,
            'failure_count_7d': lambda v: f"{int(v)} failures in last 7 days" if v > 0 else None,
            'duration_trend': lambda v: f"Duration trending up {v*100:.0f}%" if v > 0.1 else None,
            'peak_memory_last_run': lambda v: f"High memory usage ({v*100:.0f}%)" if v > 0.8 else None,
            'concurrent_pipelines': lambda v: f"High concurrency ({int(v)} pipelines)" if v > 5 else None,
            'io_wait_last_run': lambda v: f"High IO wait ({v*100:.0f}%)" if v > 0.2 else None
        }

        if feature in explanations:
            explanation = explanations[feature](value)
            if explanation:
                return {
                    "feature": feature,
                    "value": value,
                    "importance": importance,
                    "explanation": explanation
                }
        return None

    def _generate_recommendation(
        self,
        probability: float,
        risk_level: str,
        factors: list[dict]
    ) -> str:
        """Generate actionable recommendation."""
        if risk_level == "low":
            return "Pipeline is healthy. No action needed."

        if risk_level == "medium":
            return "Monitor closely. Consider reviewing resource allocation."

        recommendations = []

        for factor in factors:
            feature = factor.get("feature", "")
            if "memory" in feature:
                recommendations.append("Increase memory allocation")
            elif "failure" in feature or "success" in feature:
                recommendations.append("Review recent failure logs")
            elif "duration" in feature:
                recommendations.append("Investigate performance degradation")
            elif "concurrent" in feature:
                recommendations.append("Reschedule to reduce contention")

        if not recommendations:
            recommendations.append("Manual review recommended")

        return "; ".join(set(recommendations[:3]))

    def _calculate_confidence(self, features: pd.Series) -> float:
        """Calculate prediction confidence."""
        # Higher confidence when we have more historical data
        confidence = 0.7  # Base confidence

        if features['success_rate_30d'] > 0:  # Has history
            confidence += 0.15

        if features['avg_duration_7d'] > 0:  # Recent data
            confidence += 0.15

        return min(0.95, confidence)

    def save(self, path: str) -> None:
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained
            }, f)

    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.metrics = data['metrics']
            self.feature_importance = data['feature_importance']
            self.is_trained = data['is_trained']
