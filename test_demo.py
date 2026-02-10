#!/usr/bin/env python
"""
Test script for Predictive Pipeline Optimizer Demo

Validates that all components work correctly before deployment.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_demo_data():
    """Test the demo data generation."""
    print("Testing Demo Data Generation...")

    from app.demo_data import (
        generate_pipeline_history,
        get_pipeline_configs,
        get_upcoming_schedules,
        get_pipeline_summary,
        generate_current_metrics,
        SAMPLE_PIPELINES
    )

    # Test pipeline configs
    configs = get_pipeline_configs()
    assert len(configs) == 10, f"Expected 10 pipelines, got {len(configs)}"
    print(f"  [PASSED] get_pipeline_configs: {len(configs)} pipelines")

    # Test history generation
    history = generate_pipeline_history(days=30)
    assert len(history) > 0, "History should not be empty"
    assert 'pipeline_id' in history.columns, "Missing pipeline_id column"
    assert 'success' in history.columns, "Missing success column"
    print(f"  [PASSED] generate_pipeline_history: {len(history)} records")

    # Test upcoming schedules
    schedules = get_upcoming_schedules(hours=24)
    assert len(schedules) > 0, "Should have upcoming schedules"
    print(f"  [PASSED] get_upcoming_schedules: {len(schedules)} scheduled runs")

    # Test pipeline summary
    summary = get_pipeline_summary(history)
    assert len(summary) == 10, "Should have summary for all pipelines"
    print(f"  [PASSED] get_pipeline_summary: {len(summary)} pipelines")

    # Test current metrics
    metrics = generate_current_metrics("etl_sales_daily")
    assert 'avg_cpu' in metrics, "Missing avg_cpu in metrics"
    print(f"  [PASSED] generate_current_metrics")

    print("Demo Data: All tests passed!\n")
    return history


def test_feature_engineer(history):
    """Test the feature engineer."""
    print("Testing FeatureEngineer...")

    from src.feature_engineer import FeatureEngineer
    from datetime import datetime

    engineer = FeatureEngineer()
    engineer.load_history(history)

    # Test feature engineering
    features = engineer.engineer_features(
        pipeline_id="etl_sales_daily",
        scheduled_time=datetime.now(),
        current_metrics={'avg_cpu': 0.5, 'avg_memory': 0.6, 'peak_memory': 0.7, 'io_wait': 0.1}
    )

    assert features.pipeline_id == "etl_sales_daily", "Pipeline ID mismatch"
    assert len(features.feature_vector) == 19, f"Expected 19 features, got {len(features.feature_vector)}"
    print(f"  [PASSED] engineer_features: {len(features.feature_vector)} features")

    # Test batch engineering
    from datetime import timedelta
    batch_df = engineer.batch_engineer(
        ["etl_sales_daily", "etl_inventory"],
        [datetime.now(), datetime.now() + timedelta(hours=1)]
    )
    assert len(batch_df) == 2, "Batch should have 2 rows"
    print(f"  [PASSED] batch_engineer: {len(batch_df)} pipelines")

    print("FeatureEngineer: All tests passed!\n")
    return engineer


def test_failure_predictor(history, feature_engineer):
    """Test the failure predictor."""
    print("Testing FailurePredictor...")

    from src.failure_predictor import FailurePredictor
    import pandas as pd

    predictor = FailurePredictor()

    # Prepare training data
    features_list = []
    for _, row in history.head(500).iterrows():  # Use subset for faster testing
        features = feature_engineer.engineer_features(
            pipeline_id=row['pipeline_id'],
            scheduled_time=row['start_time'],
            current_metrics={
                'avg_cpu': row.get('avg_cpu', 0),
                'avg_memory': row.get('avg_memory', 0),
                'peak_memory': row.get('peak_memory', 0),
                'io_wait': row.get('io_wait', 0)
            }
        )
        features_list.append(features.feature_vector)

    X = pd.DataFrame(features_list, columns=feature_engineer.FEATURE_COLUMNS)
    y = (~history.head(500)['success']).astype(int)

    # Train
    metrics = predictor.train(X, y)
    assert predictor.is_trained, "Model should be trained"
    assert metrics.auc_roc > 0.5, f"AUC-ROC should be > 0.5, got {metrics.auc_roc}"
    print(f"  [PASSED] train: AUC-ROC = {metrics.auc_roc:.4f}")

    # Predict
    from datetime import datetime
    prediction = predictor.predict(
        features_list[0],
        pipeline_id="test_pipeline",
        scheduled_time=datetime.now()
    )
    assert 0 <= prediction.failure_probability <= 1, "Probability should be 0-1"
    assert prediction.risk_level in ('low', 'medium', 'high', 'critical'), "Invalid risk level"
    print(f"  [PASSED] predict: prob={prediction.failure_probability:.4f}, risk={prediction.risk_level}")

    # Batch predict
    batch_probs = predictor.predict_batch(X.head(10))
    assert len(batch_probs) == 10, "Batch should return 10 probabilities"
    print(f"  [PASSED] predict_batch: {len(batch_probs)} predictions")

    print("FailurePredictor: All tests passed!\n")


def test_streamlit_imports():
    """Test that all Streamlit app imports work."""
    print("Testing Streamlit app imports...")

    try:
        import streamlit
        print(f"  [PASSED] Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("  [SKIPPED] Streamlit not installed")
        return

    try:
        from app.demo_data import generate_pipeline_history
        from src.feature_engineer import FeatureEngineer
        from src.failure_predictor import FailurePredictor
        print("  [PASSED] All module imports successful")
    except ImportError as e:
        print(f"  [FAILED] Import error: {e}")
        return

    print("Streamlit imports: All tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("  Predictive Pipeline Optimizer - Test Suite")
    print("=" * 60)
    print()

    try:
        history = test_demo_data()
        feature_engineer = test_feature_engineer(history)
        test_failure_predictor(history, feature_engineer)
        test_streamlit_imports()

        print("=" * 60)
        print("  ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Ready to run: python run_demo.py")
        print("Or deploy to Streamlit Cloud")
        return 0

    except Exception as e:
        print(f"\n[FAILED] Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
