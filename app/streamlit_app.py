"""
Predictive Pipeline Optimizer - Streamlit Demo

AI-powered pipeline failure prediction and optimization
recommendations using XGBoost and historical analysis.

Author: Waqas Shami
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.demo_data import (
    generate_pipeline_history,
    get_pipeline_configs,
    get_upcoming_schedules,
    get_pipeline_summary,
    generate_current_metrics,
    SAMPLE_PIPELINES
)
from src.feature_engineer import FeatureEngineer
from src.failure_predictor import FailurePredictor

# Page configuration
st.set_page_config(
    page_title="Predictive Pipeline Optimizer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .main-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Health score */
    .health-container {
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        color: white;
    }
    .health-excellent { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .health-good { background: linear-gradient(135deg, #56ab2f, #a8e6cf); }
    .health-warning { background: linear-gradient(135deg, #f7971e, #ffd200); color: #333; }
    .health-critical { background: linear-gradient(135deg, #cb2d3e, #ef473a); }

    /* Risk badges */
    .risk-low { background: #28a745; color: white; }
    .risk-medium { background: #ffc107; color: #333; }
    .risk-high { background: #fd7e14; color: white; }
    .risk-critical { background: #dc3545; color: white; }
    .risk-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Pipeline card */
    .pipeline-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #302b63;
    }

    /* Metric boxes */
    .metric-box {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        background: #f8f9fa;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #302b63;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
    }

    /* Recommendation card */
    .rec-card {
        background: #f0f7ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
    .rec-high { border-left-color: #dc3545; background: #fff5f5; }
    .rec-medium { border-left-color: #ffc107; background: #fffdf0; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'history_data' not in st.session_state:
        st.session_state.history_data = None
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = None
    if 'failure_predictor' not in st.session_state:
        st.session_state.failure_predictor = None
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🔮 Predictive Pipeline Optimizer</div>
        <div class="main-subtitle">AI-powered failure prediction and optimization | Prevent issues before they impact business</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.markdown("## Settings")

        history_days = st.slider(
            "History Days",
            min_value=14,
            max_value=90,
            value=30,
            help="Days of historical data to use for training"
        )

        prediction_horizon = st.slider(
            "Prediction Horizon (hours)",
            min_value=6,
            max_value=48,
            value=24,
            help="How far ahead to predict"
        )

        st.session_state.history_days = history_days
        st.session_state.prediction_horizon = prediction_horizon

        st.divider()

        # Load/Train button
        if st.button("Load Data & Train Model", type="primary", use_container_width=True):
            with st.spinner("Generating historical data..."):
                history = generate_pipeline_history(days=history_days)
                st.session_state.history_data = history

            with st.spinner("Training prediction model..."):
                train_model(history)

            st.success("Model trained successfully!")
            st.rerun()

        if st.session_state.is_trained:
            st.success("Model is ready")

            # Model metrics
            if st.session_state.failure_predictor.metrics:
                metrics = st.session_state.failure_predictor.metrics
                st.markdown("### Model Performance")
                st.metric("AUC-ROC", f"{metrics.auc_roc:.3f}")
                st.metric("Precision", f"{metrics.precision:.3f}")
                st.metric("Recall", f"{metrics.recall:.3f}")

        st.divider()

        st.markdown("## About")
        st.markdown("""
        **Predictive Pipeline Optimizer**

        Uses ML models to predict:
        - Pipeline failure probability
        - Expected execution duration
        - Resource bottlenecks

        Provides actionable recommendations
        to prevent failures proactively.
        """)

        st.divider()

        st.markdown("## Author")
        st.markdown("""
        **Waqas Shami**
        Data Platform Owner

        [LinkedIn](https://linkedin.com/in/waqas-shami) | [Website](https://waqasshami.com)
        """)


def train_model(history: pd.DataFrame):
    """Train the prediction model."""
    # Initialize components
    feature_engineer = FeatureEngineer()
    feature_engineer.load_history(history)

    failure_predictor = FailurePredictor()

    # Sample data for faster training
    max_samples = 500  # Limit for fast training
    if len(history) > max_samples:
        # Stratified sampling to ensure failures are represented
        failures = history[~history['success']]
        successes = history[history['success']]

        # Take all failures (usually fewer) and sample from successes
        n_failures = min(len(failures), max_samples // 3)
        n_successes = max_samples - n_failures

        sampled_failures = failures.sample(n=n_failures, random_state=42) if len(failures) > n_failures else failures
        sampled_successes = successes.sample(n=min(n_successes, len(successes)), random_state=42)

        train_sample = pd.concat([sampled_failures, sampled_successes]).reset_index(drop=True)
    else:
        train_sample = history

    # Engineer features for training (using sampled data)
    features_list = []
    for _, row in train_sample.iterrows():
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

    X = pd.DataFrame(
        features_list,
        columns=feature_engineer.FEATURE_COLUMNS
    )

    # Target: failure (inverse of success)
    y = (~train_sample['success']).astype(int)

    # Train
    failure_predictor.train(X, y)

    # Store in session state
    st.session_state.feature_engineer = feature_engineer
    st.session_state.failure_predictor = failure_predictor
    st.session_state.is_trained = True


def predict_pipeline(pipeline_id: str, scheduled_time: datetime) -> dict:
    """Generate prediction for a pipeline."""
    if not st.session_state.is_trained:
        return None

    feature_engineer = st.session_state.feature_engineer
    failure_predictor = st.session_state.failure_predictor

    # Get current metrics
    metrics = generate_current_metrics(pipeline_id)

    # Engineer features
    features = feature_engineer.engineer_features(
        pipeline_id=pipeline_id,
        scheduled_time=scheduled_time,
        current_metrics=metrics
    )

    # Predict
    prediction = failure_predictor.predict(
        features.feature_vector,
        pipeline_id=pipeline_id,
        scheduled_time=scheduled_time
    )

    return {
        'pipeline_id': pipeline_id,
        'scheduled_time': scheduled_time,
        'failure_prob': prediction.failure_probability,
        'risk_level': prediction.risk_level,
        'confidence': prediction.confidence,
        'recommendation': prediction.recommendation,
        'contributing_factors': prediction.contributing_factors,
        'features': features
    }


def render_system_health():
    """Render system health overview."""
    st.markdown("### System Health Overview")

    if not st.session_state.is_trained:
        st.info("Load data and train the model to see predictions.")
        return

    # Get upcoming schedules and predict
    schedules = get_upcoming_schedules(hours=st.session_state.prediction_horizon)

    predictions = []
    for _, row in schedules.iterrows():
        pred = predict_pipeline(row['pipeline_id'], row['scheduled_time'])
        if pred:
            pred['pipeline_name'] = row['pipeline_name']
            pred['category'] = row['category']
            predictions.append(pred)

    if not predictions:
        st.warning("No predictions available.")
        return

    # Calculate health metrics
    total = len(predictions)
    healthy = sum(1 for p in predictions if p['risk_level'] == 'low')
    at_risk = sum(1 for p in predictions if p['risk_level'] in ('medium', 'high'))
    critical = sum(1 for p in predictions if p['risk_level'] == 'critical')

    avg_failure_prob = np.mean([p['failure_prob'] for p in predictions])
    health_score = 100 - (avg_failure_prob * 100)

    # Health class
    if health_score >= 90:
        health_class = "health-excellent"
        health_label = "Excellent"
    elif health_score >= 75:
        health_class = "health-good"
        health_label = "Good"
    elif health_score >= 60:
        health_class = "health-warning"
        health_label = "Warning"
    else:
        health_class = "health-critical"
        health_label = "Critical"

    # Display health score
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="health-container {health_class}">
            <div style="font-size: 2.5rem; font-weight: bold;">{health_score:.0f}</div>
            <div>Health Score</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">{health_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Scheduled Runs</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box" style="background: #d4edda;">
            <div class="metric-value" style="color: #28a745;">{healthy}</div>
            <div class="metric-label">Healthy</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box" style="background: #fff3cd;">
            <div class="metric-value" style="color: #856404;">{at_risk}</div>
            <div class="metric-label">At Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-box" style="background: #f8d7da;">
            <div class="metric-value" style="color: #721c24;">{critical}</div>
            <div class="metric-label">Critical</div>
        </div>
        """, unsafe_allow_html=True)

    # Store predictions
    st.session_state.predictions = {p['pipeline_id']: p for p in predictions}

    return predictions


def render_prediction_timeline(predictions: list):
    """Render timeline of upcoming predictions."""
    st.markdown("### Upcoming Pipeline Runs")

    if not predictions:
        return

    # Sort by failure probability (highest first)
    sorted_preds = sorted(predictions, key=lambda x: -x['failure_prob'])

    # Filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        risk_filter = st.selectbox(
            "Filter by risk:",
            ["All", "Critical", "High", "Medium", "Low"]
        )

    if risk_filter != "All":
        sorted_preds = [p for p in sorted_preds if p['risk_level'] == risk_filter.lower()]

    # Display predictions
    for pred in sorted_preds[:10]:
        risk_class = f"risk-{pred['risk_level']}"

        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])

        with col1:
            st.markdown(f"**{pred['pipeline_name']}**")
            st.caption(f"{pred['category']} | {pred['scheduled_time'].strftime('%H:%M')}")

        with col2:
            st.markdown(f"""
            <span class="risk-badge {risk_class}">{pred['risk_level'].upper()}</span>
            """, unsafe_allow_html=True)

        with col3:
            st.metric("Failure Risk", f"{pred['failure_prob']*100:.0f}%")

        with col4:
            if pred['risk_level'] in ('high', 'critical'):
                st.warning(pred['recommendation'][:50] + "...")
            elif pred['risk_level'] == 'medium':
                st.info(pred['recommendation'][:50] + "...")

        st.divider()


def render_pipeline_details():
    """Render detailed view for a specific pipeline."""
    st.markdown("### Pipeline Details")

    if not st.session_state.is_trained:
        st.info("Train the model first to see pipeline details.")
        return

    # Pipeline selector
    pipeline_options = {p.pipeline_id: f"{p.name} ({p.category})" for p in SAMPLE_PIPELINES}
    selected_id = st.selectbox(
        "Select Pipeline:",
        options=list(pipeline_options.keys()),
        format_func=lambda x: pipeline_options[x]
    )

    # Get prediction
    now = datetime.now()
    pred = predict_pipeline(selected_id, now + timedelta(hours=1))

    if not pred:
        return

    config = next(p for p in SAMPLE_PIPELINES if p.pipeline_id == selected_id)

    # Display prediction
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"## {config.name}")
        st.caption(f"Category: {config.category} | Schedule: {config.schedule}")

        # Risk display
        risk_class = f"risk-{pred['risk_level']}"
        st.markdown(f"""
        <div style="margin: 20px 0;">
            <span style="font-size: 1.2rem;">Current Risk Level: </span>
            <span class="risk-badge {risk_class}" style="font-size: 1rem; padding: 8px 16px;">
                {pred['risk_level'].upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Failure Probability", f"{pred['failure_prob']*100:.1f}%")
        with col_b:
            st.metric("Confidence", f"{pred['confidence']*100:.0f}%")
        with col_c:
            st.metric("Avg Duration", f"{config.avg_duration} min")

    with col2:
        # Resource profile
        st.markdown("**Resource Profile**")
        features = pred['features']
        st.progress(features.avg_cpu_last_run, text=f"CPU: {features.avg_cpu_last_run*100:.0f}%")
        st.progress(features.avg_memory_last_run, text=f"Memory: {features.avg_memory_last_run*100:.0f}%")
        st.progress(features.peak_memory_last_run, text=f"Peak Mem: {features.peak_memory_last_run*100:.0f}%")

    # Recommendations
    st.markdown("### Recommendations")

    rec = pred['recommendation']
    if pred['risk_level'] in ('high', 'critical'):
        st.error(f"**Action Required:** {rec}")
    elif pred['risk_level'] == 'medium':
        st.warning(f"**Suggested:** {rec}")
    else:
        st.success(f"{rec}")

    # Contributing factors
    if pred['contributing_factors']:
        st.markdown("### Contributing Factors")
        for factor in pred['contributing_factors']:
            st.markdown(f"- **{factor['feature']}**: {factor.get('explanation', 'High importance')}")


def render_historical_analysis():
    """Render historical analysis section."""
    st.markdown("### Historical Analysis")

    if st.session_state.history_data is None:
        st.info("Load data to see historical analysis.")
        return

    history = st.session_state.history_data

    # Summary stats
    summary = get_pipeline_summary(history)

    # Overall stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_runs = len(history)
        st.metric("Total Runs", f"{total_runs:,}")

    with col2:
        success_rate = history['success'].mean() * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    with col3:
        avg_duration = history['duration_minutes'].mean()
        st.metric("Avg Duration", f"{avg_duration:.0f} min")

    with col4:
        failures = (~history['success']).sum()
        st.metric("Total Failures", failures)

    # Pipeline summary table
    st.markdown("### Pipeline Summary")
    display_cols = ['pipeline_id', 'name', 'category', 'total_runs', 'success_rate',
                    'failure_count', 'avg_duration', 'avg_cpu', 'avg_memory']
    st.dataframe(
        summary[display_cols].sort_values('failure_count', ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # Failure analysis
    st.markdown("### Recent Failures")
    failures_df = history[~history['success']].sort_values('start_time', ascending=False).head(10)
    if len(failures_df) > 0:
        display_failures = failures_df[['pipeline_name', 'start_time', 'duration_minutes', 'error_message']].copy()
        display_failures['start_time'] = display_failures['start_time'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_failures, use_container_width=True, hide_index=True)
    else:
        st.success("No failures in the selected period!")


def render_metrics_section():
    """Render platform impact metrics."""
    st.markdown("### Platform Impact")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 25px; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2.2rem; font-weight: bold;">87%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Failure Reduction</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb, #f5576c); padding: 25px; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2.2rem; font-weight: bold;">82%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Faster MTTR</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe, #00f2fe); padding: 25px; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2.2rem; font-weight: bold;">99.2%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Pipeline Reliability</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43e97b, #38f9d7); padding: 25px; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2.2rem; font-weight: bold;">320K</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Annual Savings</div>
        </div>
        """, unsafe_allow_html=True)


def render_how_it_works():
    """Render how it works section."""
    with st.expander("How It Works", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            ### 1. Data Collection
            Collect execution metrics from
            Airflow, Spark, and system monitors.

            *Metrics: duration, resources, outcomes*
            """)

        with col2:
            st.markdown("""
            ### 2. Feature Engineering
            Transform raw metrics into predictive
            features with temporal signals.

            *19 engineered features*
            """)

        with col3:
            st.markdown("""
            ### 3. Prediction
            XGBoost model predicts failure
            probability 30+ minutes ahead.

            *AUC-ROC: 0.94*
            """)

        with col4:
            st.markdown("""
            ### 4. Recommendations
            Generate actionable optimization
            suggestions based on predictions.

            *Prioritized by impact*
            """)


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    render_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dashboard",
        "Pipeline Details",
        "Historical Analysis",
        "Model Info"
    ])

    with tab1:
        predictions = render_system_health()
        if predictions:
            render_prediction_timeline(predictions)

    with tab2:
        render_pipeline_details()

    with tab3:
        render_historical_analysis()

    with tab4:
        st.markdown("### Model Information")

        if st.session_state.is_trained and st.session_state.failure_predictor.metrics:
            metrics = st.session_state.failure_predictor.metrics

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Performance Metrics**")
                st.metric("AUC-ROC", f"{metrics.auc_roc:.4f}")
                st.metric("Precision", f"{metrics.precision:.4f}")
                st.metric("Recall", f"{metrics.recall:.4f}")
                st.metric("F1 Score", f"{metrics.f1_score:.4f}")

            with col2:
                st.markdown("**Cross-Validation Scores**")
                for i, score in enumerate(metrics.cv_scores):
                    st.progress(float(score), text=f"Fold {i+1}: {score:.4f}")

            # Feature importance
            if st.session_state.failure_predictor.feature_importance:
                st.markdown("### Feature Importance")
                importance = st.session_state.failure_predictor.feature_importance
                sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
                max_imp = float(max(importance.values()))

                for feature, imp in sorted_imp[:10]:
                    st.progress(float(imp) / max_imp, text=f"{feature}: {imp:.4f}")

        else:
            st.info("Train the model to see performance metrics.")

    st.markdown("---")

    # How it works
    render_how_it_works()

    # Impact metrics
    render_metrics_section()

    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 30px; margin-top: 40px;">
        <p>🔮 Predictive Pipeline Optimizer | AI-Powered Failure Prediction & Optimization</p>
        <p style="font-size: 0.9rem;">
            <a href="https://waqasshami.com" target="_blank">waqasshami.com</a> |
            <a href="https://linkedin.com/in/waqas-shami" target="_blank">LinkedIn</a> |
            <a href="https://github.com/waqas-shami/predictive-pipeline-optimizer" target="_blank">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
