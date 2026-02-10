"""
Streamlit Dashboard for Pipeline Health Monitoring

Real-time visualization of pipeline predictions and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
sys.path.append('..')

# Page config
st.set_page_config(
    page_title="Pipeline Health Dashboard",
    page_icon="🔧",
    layout="wide"
)

# Generate sample data for demo
@st.cache_data
def generate_sample_data():
    """Generate sample pipeline data for demonstration."""
    np.random.seed(42)

    pipelines = [
        "etl_sales_daily",
        "etl_inventory_hourly",
        "etl_customer_360",
        "etl_finance_monthly",
        "etl_marketing_events",
        "etl_product_catalog",
        "etl_orders_streaming",
        "etl_user_activity"
    ]

    now = datetime.now()
    data = []

    for pipeline in pipelines:
        # Generate risk score with some variation
        base_risk = np.random.uniform(0.05, 0.4)
        risk = min(0.95, base_risk + np.random.normal(0, 0.1))
        risk = max(0.01, risk)

        scheduled = now + timedelta(hours=np.random.randint(1, 12))

        data.append({
            "pipeline_id": pipeline,
            "scheduled_time": scheduled,
            "failure_prob": risk,
            "expected_duration": np.random.randint(30, 180),
            "last_status": np.random.choice(["success", "success", "success", "failed"]),
            "avg_duration_7d": np.random.randint(25, 160),
            "success_rate_7d": np.random.uniform(0.7, 1.0),
            "cpu_usage": np.random.uniform(0.3, 0.9),
            "memory_usage": np.random.uniform(0.4, 0.95)
        })

    return pd.DataFrame(data)


@st.cache_data
def generate_history():
    """Generate historical execution data."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

    history = []
    for date in dates:
        history.append({
            "date": date,
            "total_runs": np.random.randint(20, 40),
            "failures": np.random.randint(0, 5),
            "avg_duration": np.random.randint(45, 90)
        })

    df = pd.DataFrame(history)
    df['success_rate'] = (df['total_runs'] - df['failures']) / df['total_runs'] * 100
    return df


def get_risk_color(prob):
    """Get color based on risk level."""
    if prob < 0.2:
        return "🟢"
    elif prob < 0.5:
        return "🟡"
    elif prob < 0.8:
        return "🟠"
    return "🔴"


def get_risk_level(prob):
    """Get risk level string."""
    if prob < 0.2:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    elif prob < 0.8:
        return "High"
    return "Critical"


# Load data
df = generate_sample_data()
history = generate_history()

# Header
st.title("🔧 Pipeline Health Dashboard")
st.markdown("Real-time predictions and optimization recommendations powered by ML")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    healthy = len(df[df['failure_prob'] < 0.2])
    st.metric(
        "Healthy Pipelines",
        f"{healthy}/{len(df)}",
        delta=f"{healthy/len(df)*100:.0f}%"
    )

with col2:
    at_risk = len(df[(df['failure_prob'] >= 0.2) & (df['failure_prob'] < 0.5)])
    st.metric(
        "At Risk",
        at_risk,
        delta="Medium" if at_risk > 0 else "None",
        delta_color="off"
    )

with col3:
    critical = len(df[df['failure_prob'] >= 0.8])
    st.metric(
        "Critical",
        critical,
        delta="Needs attention" if critical > 0 else "All clear",
        delta_color="inverse" if critical > 0 else "normal"
    )

with col4:
    overall_health = 100 - df['failure_prob'].mean() * 100
    st.metric(
        "Overall Health",
        f"{overall_health:.1f}%",
        delta="Good" if overall_health > 80 else "Degraded"
    )

st.divider()

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Pipeline Overview", "📈 Historical Trends", "💡 Recommendations"])

with tab1:
    st.subheader("Pipeline Risk Assessment")

    # Risk distribution chart
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            df.sort_values('failure_prob', ascending=True),
            x='failure_prob',
            y='pipeline_id',
            orientation='h',
            color='failure_prob',
            color_continuous_scale=['green', 'yellow', 'orange', 'red'],
            labels={'failure_prob': 'Failure Probability', 'pipeline_id': 'Pipeline'},
            title='Pipeline Risk Levels'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk distribution pie
        risk_counts = {
            "Low": len(df[df['failure_prob'] < 0.2]),
            "Medium": len(df[(df['failure_prob'] >= 0.2) & (df['failure_prob'] < 0.5)]),
            "High": len(df[(df['failure_prob'] >= 0.5) & (df['failure_prob'] < 0.8)]),
            "Critical": len(df[df['failure_prob'] >= 0.8])
        }
        fig_pie = px.pie(
            values=list(risk_counts.values()),
            names=list(risk_counts.keys()),
            color_discrete_sequence=['#28a745', '#ffc107', '#fd7e14', '#dc3545'],
            title='Risk Distribution'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed table
    st.subheader("Upcoming Pipeline Runs")

    display_df = df.copy()
    display_df['Risk'] = display_df['failure_prob'].apply(lambda x: f"{get_risk_color(x)} {get_risk_level(x)}")
    display_df['Failure Prob'] = display_df['failure_prob'].apply(lambda x: f"{x*100:.1f}%")
    display_df['Next Run'] = display_df['scheduled_time'].dt.strftime('%H:%M')
    display_df['Duration (est)'] = display_df['expected_duration'].apply(lambda x: f"{x} min")
    display_df['Success Rate (7d)'] = display_df['success_rate_7d'].apply(lambda x: f"{x*100:.0f}%")

    st.dataframe(
        display_df[['pipeline_id', 'Risk', 'Failure Prob', 'Next Run', 'Duration (est)', 'Success Rate (7d)']],
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("Historical Performance")

    col1, col2 = st.columns(2)

    with col1:
        fig_success = px.line(
            history,
            x='date',
            y='success_rate',
            title='Success Rate Trend (30 days)',
            labels={'success_rate': 'Success Rate (%)', 'date': 'Date'}
        )
        fig_success.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target: 95%")
        st.plotly_chart(fig_success, use_container_width=True)

    with col2:
        fig_duration = px.line(
            history,
            x='date',
            y='avg_duration',
            title='Average Duration Trend',
            labels={'avg_duration': 'Duration (min)', 'date': 'Date'}
        )
        st.plotly_chart(fig_duration, use_container_width=True)

    # Failures over time
    fig_failures = px.bar(
        history,
        x='date',
        y='failures',
        title='Daily Failures',
        labels={'failures': 'Failure Count', 'date': 'Date'},
        color='failures',
        color_continuous_scale=['green', 'yellow', 'red']
    )
    st.plotly_chart(fig_failures, use_container_width=True)

with tab3:
    st.subheader("Optimization Recommendations")

    # High-risk pipelines
    high_risk = df[df['failure_prob'] >= 0.3].sort_values('failure_prob', ascending=False)

    if len(high_risk) == 0:
        st.success("✅ All pipelines are healthy! No immediate actions required.")
    else:
        for _, row in high_risk.iterrows():
            with st.expander(f"🔧 {row['pipeline_id']} - {get_risk_color(row['failure_prob'])} {row['failure_prob']*100:.0f}% risk"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Current Metrics:**")
                    st.write(f"- CPU Usage: {row['cpu_usage']*100:.0f}%")
                    st.write(f"- Memory Usage: {row['memory_usage']*100:.0f}%")
                    st.write(f"- Success Rate (7d): {row['success_rate_7d']*100:.0f}%")

                with col2:
                    st.markdown("**Recommendations:**")
                    if row['memory_usage'] > 0.85:
                        st.write("⚡ Increase memory allocation by 25%")
                    if row['success_rate_7d'] < 0.9:
                        st.write("🔍 Review and fix root cause of recent failures")
                    if row['cpu_usage'] > 0.8:
                        st.write("📊 Optimize queries or scale up compute")
                    st.write("📅 Consider rescheduling to off-peak hours")

    # General recommendations
    st.subheader("System-Wide Recommendations")

    recommendations = [
        ("High", "Review memory allocation for 3 pipelines approaching limits"),
        ("Medium", "Consider implementing incremental processing for large pipelines"),
        ("Low", "Update monitoring thresholds based on recent patterns"),
    ]

    for priority, rec in recommendations:
        color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[priority]
        st.markdown(f"{color} **{priority}**: {rec}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Predictive Pipeline Optimizer | Built with ML | Last updated: {}</small>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
