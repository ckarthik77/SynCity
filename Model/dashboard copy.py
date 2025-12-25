import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from inference_realtime import TrafficPredictor

# Page config
st.set_page_config(
    page_title="SynCity - AI Traffic Prediction",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🚗 SynCity - AI Traffic Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Horizon Attention-Based LSTM for Smart Traffic Management</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")
data_source = st.sidebar.radio("Data Source", ["Real-Time Predictions", "Model Evaluation"])

# Load data based on selection
@st.cache_data
def load_realtime_data():
    try:
        df = pd.read_csv('realtime_predictions.csv')
        return df
    except:
        st.error("⚠️ realtime_predictions.csv not found. Run traci_logger_realtime.py first!")
        return None

@st.cache_data
def load_evaluation_metrics():
    try:
        df = pd.read_csv('models/evaluation_metrics.csv')
        return df
    except:
        return None

# Main content
if data_source == "Real-Time Predictions":
    st.header("📊 Real-Time Prediction Analysis")
    
    df = load_realtime_data()
    
    if df is not None:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", f"{len(df):,}")
        with col2:
            st.metric("Unique Vehicles", df['vehicle_id'].nunique())
        with col3:
            st.metric("Simulation Duration", f"{df['time'].max():.0f}s")
        with col4:
            avg_speed = df['current_speed'].mean()
            st.metric("Avg Speed", f"{avg_speed:.1f} m/s")
        
        st.markdown("---")
        
        # Vehicle selection
        st.subheader("🚙 Individual Vehicle Analysis")
        vehicles = df['vehicle_id'].unique()
        selected_vehicle = st.selectbox("Select Vehicle", vehicles[:50])  # Limit to 50 for performance
        
        vehicle_data = df[df['vehicle_id'] == selected_vehicle].sort_values('time')
        
        if len(vehicle_data) > 0:
            # Multi-horizon prediction plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['time'],
                y=vehicle_data['current_speed'],
                mode='lines',
                name='Current Speed',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['time'],
                y=vehicle_data['predicted_30s'],
                mode='lines',
                name='Predicted @ 30s',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['time'],
                y=vehicle_data['predicted_90s'],
                mode='lines',
                name='Predicted @ 90s',
                line=dict(color='orange', width=2, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['time'],
                y=vehicle_data['predicted_150s'],
                mode='lines',
                name='Predicted @ 150s',
                line=dict(color='red', width=2, dash='dashdot')
            ))
            
            fig.update_layout(
                title=f"Speed Predictions for {selected_vehicle}",
                xaxis_title="Time (s)",
                yaxis_title="Speed (m/s)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Delta predictions
            st.subheader("📈 Speed Change Predictions (Δ)")
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=vehicle_data['time'],
                y=vehicle_data['delta_30s'],
                mode='lines+markers',
                name='Δ @ 30s',
                line=dict(color='green', width=2)
            ))
            
            fig2.add_trace(go.Scatter(
                x=vehicle_data['time'],
                y=vehicle_data['delta_90s'],
                mode='lines+markers',
                name='Δ @ 90s',
                line=dict(color='orange', width=2)
            ))
            
            fig2.add_trace(go.Scatter(
                x=vehicle_data['time'],
                y=vehicle_data['delta_150s'],
                mode='lines+markers',
                name='Δ @ 150s',
                line=dict(color='red', width=2)
            ))
            
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig2.update_layout(
                title="Predicted Speed Changes",
                xaxis_title="Time (s)",
                yaxis_title="Δ Speed (m/s)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # Overall statistics
        st.subheader("📊 Overall Prediction Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of current speeds
            fig3 = px.histogram(
                df,
                x='current_speed',
                nbins=50,
                title="Distribution of Current Speeds",
                labels={'current_speed': 'Speed (m/s)', 'count': 'Frequency'}
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Distribution of 30s predictions
            fig4 = px.histogram(
                df,
                x='delta_30s',
                nbins=50,
                title="Distribution of 30s Speed Changes",
                labels={'delta_30s': 'Δ Speed (m/s)', 'count': 'Frequency'}
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Prediction accuracy over time
        st.subheader("⏱️ Predictions Over Time")
        
        # Sample every 10th row for performance
        sampled_df = df[::10]
        
        fig5 = px.scatter(
            sampled_df,
            x='time',
            y='current_speed',
            color='delta_30s',
            size=abs(sampled_df['delta_30s']) + 0.1,
            title="Speed vs Time (colored by predicted 30s change)",
            labels={'time': 'Time (s)', 'current_speed': 'Speed (m/s)', 'delta_30s': 'Δ30s'},
            color_continuous_scale='RdYlGn_r'
        )
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)

else:  # Model Evaluation
    st.header("🎯 Model Performance Evaluation")
    
    metrics_df = load_evaluation_metrics()
    
    if metrics_df is not None:
        # Display metrics table
        st.subheader("📈 Performance Metrics")
        
        # Format the dataframe
        metrics_display = metrics_df.copy()
        metrics_display.index.name = 'Prediction Horizon'
        
        st.dataframe(
            metrics_display.style.format({
                'MAE': '{:.4f}',
                'RMSE': '{:.4f}',
                'R²': '{:.4f}',
                'MAPE': '{:.2f}%'
            }).background_gradient(cmap='RdYlGn', subset=['R²']),
            use_container_width=True
        )
        
        # Metrics comparison
        st.subheader("📊 Metrics Comparison Across Horizons")
        
        fig_metrics = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAE', 'RMSE', 'R² Score', 'MAPE'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        horizons = metrics_df.index.tolist()
        
        # MAE
        fig_metrics.add_trace(
            go.Bar(x=horizons, y=metrics_df['MAE'], name='MAE', marker_color='#2ecc71'),
            row=1, col=1
        )
        
        # RMSE
        fig_metrics.add_trace(
            go.Bar(x=horizons, y=metrics_df['RMSE'], name='RMSE', marker_color='#3498db'),
            row=1, col=2
        )
        
        # R²
        fig_metrics.add_trace(
            go.Bar(x=horizons, y=metrics_df['R²'], name='R²', marker_color='#e74c3c'),
            row=2, col=1
        )
        
        # MAPE
        fig_metrics.add_trace(
            go.Bar(x=horizons, y=metrics_df['MAPE'], name='MAPE', marker_color='#f39c12'),
            row=2, col=2
        )
        
        fig_metrics.update_layout(height=700, showlegend=False)
        fig_metrics.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Key insights
        st.subheader("💡 Key Insights")
        
        best_r2_horizon = metrics_df['R²'].idxmax()
        best_r2_value = metrics_df['R²'].max()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"**Best R² Score:** {best_r2_value:.4f} @ {best_r2_horizon}")
        
        with col2:
            avg_mae = metrics_df['MAE'].mean()
            st.info(f"**Average MAE:** {avg_mae:.4f}")
        
        with col3:
            best_mae_horizon = metrics_df['MAE'].idxmin()
            st.warning(f"**Best MAE:** {metrics_df['MAE'].min():.4f} @ {best_mae_horizon}")
    
    else:
        st.warning("⚠️ Evaluation metrics not found. Run evaluate_multihorizon.py first!")
    
    # Load and display visualizations from models folder
    st.markdown("---")
    st.subheader("📊 Detailed Visualizations")
    
    viz_files = {
        "Actual vs Predicted": "models/actual_vs_predicted.png",
        "Time Series Comparison": "models/timeseries_comparison.png",
        "Error Distribution": "models/error_distribution.png",
        "Attention Heatmap": "models/attention_heatmap.png",
        "Average Attention Weights": "models/avg_attention_weights.png",
        "Metrics Comparison": "models/metrics_comparison.png"
    }
    
    for title, path in viz_files.items():
        try:
            st.image(path, caption=title, use_container_width=True)
        except:
            st.warning(f"⚠️ {title} not found at {path}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>SynCity - AI Traffic Prediction System</strong></p>
        <p>Multi-Horizon Attention-LSTM | SUMO Integration | Real-Time Inference</p>
    </div>
""", unsafe_allow_html=True)