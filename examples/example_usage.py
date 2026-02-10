"""
Example Usage - Predictive Pipeline Optimizer

Demonstrates training, prediction, and recommendation
capabilities for pipeline failure prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pipeline_optimizer import PipelineOptimizer
from src.feature_engineer import FeatureEngineer
from src.failure_predictor import FailurePredictor


def create_sample_history() -> pd.DataFrame:
    """Create sample historical execution data."""
    np.random.seed(42)

    pipelines = [
        "etl_sales_daily",
        "etl_inventory_hourly",
        "etl_customer_360",
        "etl_finance_monthly"
    ]

    records = []
    base_date = datetime.now() - timedelta(days=90)

    for day in range(90):
        for pipeline in pipelines:
            # Multiple runs per day for some pipelines
            runs_per_day = 24 if "hourly" in pipeline else 1

            for run in range(runs_per_day):
                start_time = base_date + timedelta(days=day, hours=run if runs_per_day > 1 else 2)

                # Add some randomness to success rate
                base_success_rate = 0.92 if "finance" not in pipeline else 0.85
                success = np.random.random() < base_success_rate

                # Duration varies
                base_duration = {"etl_sales_daily": 45, "etl_inventory_hourly": 15,
                               "etl_customer_360": 90, "etl_finance_monthly": 120}
                duration = base_duration[pipeline] * np.random.uniform(0.8, 1.3)

                records.append({
                    "pipeline_id": pipeline,
                    "start_time": start_time,
                    "duration_minutes": duration,
                    "success": success,
                    "cpu_avg": np.random.uniform(0.3, 0.8),
                    "memory_avg": np.random.uniform(0.4, 0.9),
                    "memory_peak": np.random.uniform(0.6, 0.95),
                    "io_wait": np.random.uniform(0.05, 0.25),
                    "rows_processed": np.random.randint(100000, 10000000)
                })

    return pd.DataFrame(records)


def demo_full_workflow():
    """Demonstrate complete training and prediction workflow."""
    print("=" * 60)
    print("FULL WORKFLOW DEMO")
    print("=" * 60)

    # Create sample data
    print("\n1. Creating sample historical data...")
    history = create_sample_history()
    print(f"   Generated {len(history)} historical records")
    print(f"   Pipelines: {history['pipeline_id'].unique().tolist()}")
    print(f"   Date range: {history['start_time'].min()} to {history['start_time'].max()}")

    # Initialize and train optimizer
    print("\n2. Training prediction models...")
    optimizer = PipelineOptimizer()
    metrics = optimizer.train(history)
    print(f"   AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")

    # Make predictions
    print("\n3. Making predictions...")
    pipelines = history['pipeline_id'].unique()

    for pipeline in pipelines:
        scheduled = datetime.now() + timedelta(hours=2)
        prediction = optimizer.predict(pipeline, scheduled)

        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        print(f"\n   {pipeline}:")
        print(f"   {risk_emoji.get(prediction.risk_level, '⚪')} Risk: {prediction.risk_level} ({prediction.failure_prob*100:.1f}%)")
        print(f"   ⏱️  Expected duration: {prediction.expected_duration:.0f} min")

        if prediction.recommendations:
            print(f"   💡 Top recommendation: {prediction.recommendations[0].action}")

    # System health
    print("\n4. System Health Assessment...")
    health = optimizer.get_system_health(pipelines.tolist())
    print(f"   Overall Health Score: {health.overall_health_score:.1f}%")
    print(f"   Healthy: {health.healthy_count}")
    print(f"   At Risk: {health.at_risk_count}")
    print(f"   Critical: {health.critical_count}")


def demo_feature_engineering():
    """Demonstrate feature engineering."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING DEMO")
    print("=" * 60)

    history = create_sample_history()
    engineer = FeatureEngineer()
    engineer.load_history(history)

    # Engineer features for a pipeline
    pipeline_id = "etl_sales_daily"
    scheduled = datetime.now() + timedelta(hours=2)

    features = engineer.engineer_features(
        pipeline_id=pipeline_id,
        scheduled_time=scheduled,
        current_metrics={
            "avg_cpu": 0.65,
            "avg_memory": 0.72,
            "peak_memory": 0.85,
            "io_wait": 0.12
        }
    )

    print(f"\nFeatures for {pipeline_id}:")
    print(f"  Temporal:")
    print(f"    - Hour: {features.hour_of_day}")
    print(f"    - Day of week: {features.day_of_week}")
    print(f"    - Is weekend: {features.is_weekend}")

    print(f"  Historical:")
    print(f"    - Avg duration (7d): {features.avg_duration_7d:.1f} min")
    print(f"    - Success rate (7d): {features.success_rate_7d*100:.0f}%")
    print(f"    - Failures (7d): {features.failure_count_7d}")
    print(f"    - Duration trend: {features.duration_trend*100:.1f}%")

    print(f"  Resources:")
    print(f"    - CPU: {features.avg_cpu_last_run*100:.0f}%")
    print(f"    - Memory: {features.avg_memory_last_run*100:.0f}%")
    print(f"    - Peak memory: {features.peak_memory_last_run*100:.0f}%")

    print(f"  Risk Score: {features.risk_score:.2f}")


def demo_failure_predictor():
    """Demonstrate failure prediction model."""
    print("\n" + "=" * 60)
    print("FAILURE PREDICTOR DEMO")
    print("=" * 60)

    # Create training data
    np.random.seed(42)
    n_samples = 1000

    # Simulate features
    X = pd.DataFrame({
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'is_month_end': np.random.randint(0, 2, n_samples),
        'minutes_since_last_run': np.random.uniform(0, 1440, n_samples),
        'avg_duration_7d': np.random.uniform(30, 120, n_samples),
        'avg_duration_30d': np.random.uniform(30, 120, n_samples),
        'success_rate_7d': np.random.uniform(0.7, 1.0, n_samples),
        'success_rate_30d': np.random.uniform(0.7, 1.0, n_samples),
        'failure_count_7d': np.random.randint(0, 5, n_samples),
        'duration_trend': np.random.uniform(-0.2, 0.3, n_samples),
        'avg_cpu_last_run': np.random.uniform(0.3, 0.9, n_samples),
        'avg_memory_last_run': np.random.uniform(0.4, 0.9, n_samples),
        'peak_memory_last_run': np.random.uniform(0.5, 0.95, n_samples),
        'io_wait_last_run': np.random.uniform(0.05, 0.3, n_samples),
        'concurrent_pipelines': np.random.randint(1, 10, n_samples),
        'data_volume_gb': np.random.uniform(1, 100, n_samples),
        'row_count_estimate': np.random.randint(10000, 10000000, n_samples),
        'upstream_failures_24h': np.random.randint(0, 3, n_samples)
    })

    # Target: failure probability increases with certain features
    failure_prob = (
        (1 - X['success_rate_7d']) * 0.4 +
        (X['peak_memory_last_run'] > 0.9).astype(float) * 0.3 +
        (X['failure_count_7d'] > 2).astype(float) * 0.2 +
        np.random.uniform(0, 0.1, n_samples)
    )
    y = (failure_prob > 0.3).astype(int)

    # Train model
    predictor = FailurePredictor()
    metrics = predictor.train(X, y)

    print(f"\nModel Performance:")
    print(f"  AUC-ROC: {metrics.auc_roc:.3f}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1 Score: {metrics.f1_score:.3f}")

    print(f"\nTop Feature Importance:")
    sorted_importance = sorted(predictor.feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:5]:
        print(f"  {feature}: {importance:.3f}")

    # Make a prediction
    test_features = X.iloc[[0]]
    prediction = predictor.predict(test_features, pipeline_id="test_pipeline")

    print(f"\nSample Prediction:")
    print(f"  Failure Probability: {prediction.failure_probability*100:.1f}%")
    print(f"  Risk Level: {prediction.risk_level}")
    print(f"  Confidence: {prediction.confidence*100:.0f}%")
    print(f"  Recommendation: {prediction.recommendation}")


if __name__ == "__main__":
    demo_full_workflow()
    demo_feature_engineering()
    demo_failure_predictor()
