import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="SynCity | Traffic Intelligence Platform",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 0 4px 12px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 400;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-tagline {
        font-size: 1rem;
        font-weight: 300;
        color: rgba(255,255,255,0.7);
        margin-top: 1rem;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .css-1d391kg .stRadio > label, [data-testid="stSidebar"] .stRadio > label {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stRadio label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 1rem;
        padding: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .stRadio label:hover {
        color: white !important;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
    }
    
    /* Dataframe Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Success/Warning/Info Boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.6);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading {
        background: linear-gradient(90deg, rgba(255,255,255,0.05) 25%, rgba(102,126,234,0.1) 50%, rgba(255,255,255,0.05) 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HERO SECTION ====================
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🏙️ SynCity</h1>
        <p class="hero-subtitle">Synchronized Urban Traffic via AV-Infrastructure Synergy</p>
        <p class="hero-tagline">Multi-Horizon Attention-Based LSTM | Real-Time Traffic Intelligence | Smart City Integration</p>
    </div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    view_mode = st.radio(
        "Select Analysis Mode",
        ["🎯 Real-Time Predictions", "📊 Model Performance", "🔬 System Architecture"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📌 Quick Stats")
    st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;'>
            <p style='color: rgba(255,255,255,0.6); margin: 0; font-size: 0.85rem;'>MODEL VERSION</p>
            <p style='color: white; margin: 0.25rem 0 0 0; font-weight: 600;'>Multi-Horizon LSTM v2.0</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔗 Resources")
    st.markdown("""
        - [📖 Documentation](#)
        - [🐙 GitHub Repository](#)
        - [📊 Research Paper](#)
    """)

# ==================== DATA LOADING ====================
@st.cache_data
def load_realtime_data():
    try:
        df = pd.read_csv('realtime_predictions.csv')
        return df
    except:
        return None

@st.cache_data
def load_evaluation_metrics():
    try:
        df = pd.read_csv('evaluation_metrics.csv')
        return df
    except:
        return None

# ==================== MAIN CONTENT ====================

if view_mode == "🎯 Real-Time Predictions":
    st.markdown('<p class="section-header">📡 Real-Time Traffic Prediction Analysis</p>', unsafe_allow_html=True)
    
    df = load_realtime_data()
    
    if df is not None and len(df) > 0:
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Total Predictions</p>
                    <p class="metric-value">{len(df):,}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Unique Vehicles</p>
                    <p class="metric-value">{df['vehicle_id'].nunique()}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Simulation Duration</p>
                    <p class="metric-value">{df['time'].max():.0f}s</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_speed = df['current_speed'].mean()
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Average Speed</p>
                    <p class="metric-value">{avg_speed:.1f}</p>
                    <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.25rem;">m/s</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Vehicle Analysis
        st.markdown('<p class="section-header">🚗 Individual Vehicle Trajectory Analysis</p>', unsafe_allow_html=True)
        
        vehicles = df['vehicle_id'].unique()
        selected_vehicle = st.selectbox("Select Vehicle ID", vehicles[:100], label_visibility="collapsed")
        
        vehicle_data = df[df['vehicle_id'] == selected_vehicle].sort_values('time')
        
        if len(vehicle_data) > 0:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Speed Predictions", "🔄 Delta Analysis", "📊 Statistics"])
            
            with tab1:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=vehicle_data['time'],
                    y=vehicle_data['current_speed'],
                    mode='lines',
                    name='Current Speed',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=vehicle_data['time'],
                    y=vehicle_data['predicted_30s'],
                    mode='lines',
                    name='Predicted @ 30s',
                    line=dict(color='#2ecc71', width=2, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=vehicle_data['time'],
                    y=vehicle_data['predicted_90s'],
                    mode='lines',
                    name='Predicted @ 90s',
                    line=dict(color='#f39c12', width=2, dash='dot')
                ))
                
                fig.add_trace(go.Scatter(
                    x=vehicle_data['time'],
                    y=vehicle_data['predicted_150s'],
                    mode='lines',
                    name='Predicted @ 150s',
                    line=dict(color='#e74c3c', width=2, dash='dashdot')
                ))
                
                fig.update_layout(
                    title=f"Multi-Horizon Speed Predictions | Vehicle: {selected_vehicle}",
                    xaxis_title="Simulation Time (seconds)",
                    yaxis_title="Speed (m/s)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=vehicle_data['time'],
                    y=vehicle_data['delta_30s'],
                    mode='lines+markers',
                    name='Δ @ 30s',
                    line=dict(color='#2ecc71', width=2),
                    marker=dict(size=6)
                ))
                
                fig2.add_trace(go.Scatter(
                    x=vehicle_data['time'],
                    y=vehicle_data['delta_90s'],
                    mode='lines+markers',
                    name='Δ @ 90s',
                    line=dict(color='#f39c12', width=2),
                    marker=dict(size=6)
                ))
                
                fig2.add_trace(go.Scatter(
                    x=vehicle_data['time'],
                    y=vehicle_data['delta_150s'],
                    mode='lines+markers',
                    name='Δ @ 150s',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=6)
                ))
                
                fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                
                fig2.update_layout(
                    title="Predicted Speed Changes (Δ)",
                    xaxis_title="Simulation Time (seconds)",
                    yaxis_title="Δ Speed (m/s)",
                    hovermode='x unified',
                    height=450,
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='white')
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.markdown("#### 📊 Speed Statistics")
                    st.markdown(f"""
                        - **Mean Speed:** {vehicle_data['current_speed'].mean():.2f} m/s
                        - **Max Speed:** {vehicle_data['current_speed'].max():.2f} m/s
                        - **Min Speed:** {vehicle_data['current_speed'].min():.2f} m/s
                        - **Std Dev:** {vehicle_data['current_speed'].std():.2f} m/s
                    """)
                
                with stats_col2:
                    st.markdown("#### 🎯 Prediction Statistics")
                    st.markdown(f"""
                        - **Avg Δ30s:** {vehicle_data['delta_30s'].mean():.3f} m/s
                        - **Avg Δ90s:** {vehicle_data['delta_90s'].mean():.3f} m/s
                        - **Avg Δ150s:** {vehicle_data['delta_150s'].mean():.3f} m/s
                        - **Data Points:** {len(vehicle_data)}
                    """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Overall Statistics
        st.markdown('<p class="section-header">📊 Fleet-Wide Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.histogram(
                df.sample(min(5000, len(df))),
                x='current_speed',
                nbins=50,
                title="Speed Distribution Across All Vehicles",
                labels={'current_speed': 'Speed (m/s)', 'count': 'Frequency'},
                template='plotly_dark'
            )
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='white'),
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = px.histogram(
                df.sample(min(5000, len(df))),
                x='delta_30s',
                nbins=50,
                title="30s Prediction Delta Distribution",
                labels={'delta_30s': 'Δ Speed (m/s)', 'count': 'Frequency'},
                template='plotly_dark',
                color_discrete_sequence=['#667eea']
            )
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='white'),
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    else:
        st.warning("⚠️ No real-time prediction data available. Run `traci_logger_realtime.py` first.")

elif view_mode == "📊 Model Performance":
    st.markdown('<p class="section-header">🎯 Model Performance Evaluation</p>', unsafe_allow_html=True)
    
    metrics_df = load_evaluation_metrics()
    
    if metrics_df is not None:
        # Display metrics
        st.markdown("#### 📈 Performance Metrics Summary")
        
        # Format dataframe
        metrics_display = metrics_df.copy()
        metrics_display.index.name = 'Prediction Horizon'
        
        st.dataframe(
            metrics_display.style.format({
                'MAE': '{:.4f}',
                'RMSE': '{:.4f}',
                'R²': '{:.4f}',
                'MAPE': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics visualization
        fig_metrics = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Absolute Error', 'Root Mean Squared Error', 'R² Score', 'Mean Absolute Percentage Error'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        horizons = metrics_df.index.tolist()
        colors = ['#667eea', '#764ba2', '#f093fb']
        
        fig_metrics.add_trace(
            go.Bar(x=horizons, y=metrics_df['MAE'], marker_color=colors, showlegend=False),
            row=1, col=1
        )
        
        fig_metrics.add_trace(
            go.Bar(x=horizons, y=metrics_df['RMSE'], marker_color=colors, showlegend=False),
            row=1, col=2
        )
        
        fig_metrics.add_trace(
            go.Bar(x=horizons, y=metrics_df['R²'], marker_color=colors, showlegend=False),
            row=2, col=1
        )
        
        fig_metrics.add_trace(
            go.Bar(x=horizons, y=metrics_df['MAPE'], marker_color=colors, showlegend=False),
            row=2, col=2
        )
        
        fig_metrics.update_layout(
            height=700,
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='white')
        )
        fig_metrics.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Key Insights
        st.markdown('<p class="section-header">💡 Key Performance Insights</p>', unsafe_allow_html=True)
        
        best_r2_horizon = metrics_df['R²'].idxmax()
        best_r2_value = metrics_df['R²'].max()
        avg_mae = metrics_df['MAE'].mean()
        best_mae_horizon = metrics_df['MAE'].idxmin()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"**🏆 Best R² Score**\n\n{best_r2_value:.4f} @ {best_r2_horizon}")
        
        with col2:
            st.info(f"**📊 Average MAE**\n\n{avg_mae:.4f} across all horizons")
        
        with col3:
            st.warning(f"**🎯 Best Accuracy**\n\n{best_mae_horizon} with MAE {metrics_df['MAE'].min():.4f}")
        
        # Visualizations
        st.markdown('<p class="section-header">📊 Detailed Performance Visualizations</p>', unsafe_allow_html=True)
        
        viz_files = {
            "Actual vs Predicted": "actual_vs_predicted.png",
            "Time Series Comparison": "timeseries_comparison.png",
            "Error Distribution": "error_distribution.png",
            "Attention Heatmap": "attention_heatmap.png",
            "Average Attention Weights": "avg_attention_weights.png",
            "Metrics Comparison": "metrics_comparison.png"
        }
        
        tabs = st.tabs(list(viz_files.keys()))
        
        for idx, (title, path) in enumerate(viz_files.items()):
            with tabs[idx]:
                try:
                    st.image(path, use_container_width=True)
                except:
                    st.warning(f"⚠️ {title} visualization not found. Run evaluation script first.")
    
    else:
        st.warning("⚠️ Evaluation metrics not found. Download from Colab or run evaluation script.")

else:  # System Architecture
    st.markdown('<p class="section-header">🔬 System Architecture Overview</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🏗️ SynCity: Synchronized Urban Traffic via AV-Infrastructure Synergy
    
    **Core Components:**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🧠 AI/ML Pipeline
        - **Data Collection**: SUMO + TraCI integration
        - **Feature Engineering**: 6 validated traffic features
        - **Model Architecture**: Multi-Horizon Attention-LSTM
        - **Training**: 50 epochs, 74K+ data points
        - **Inference**: Real-time prediction every 5 seconds
        """)
        
        st.markdown("""
        #### 📊 Performance Benchmarks
        - **1-step (30s)**: R² = 0.89, MAE = 0.045
        - **3-step (90s)**: R² = 0.90, MAE = 0.094
        - **5-step (150s)**: R² = 0.87, MAE = 0.169
        """)
    
    with col2:
        st.markdown("""
        #### 🚀 Key Innovations
        1. **Attention Mechanism**: Identifies critical past timesteps
        2. **Multi-Horizon Prediction**: 30s, 90s, 150s ahead
        3. **Real-Time Integration**: Live SUMO simulation
        4. **Scalable Architecture**: Ready for deployment
        """)
        
        st.markdown("""
        #### 🎯 Applications
        - Smart traffic signal control
        - Congestion prediction & mitigation
        - Route optimization for AVs
        - Emergency vehicle prioritization
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔄 Data Flow Architecture
```
    SUMO Simulation → TraCI API → Feature Extraction → 
    Rolling Window Buffer → LSTM Model → Multi-Horizon Predictions →
    Decision Support System → Traffic Management Actions
```
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📚 Technology Stack
    
    | Component | Technology |
    |-----------|-----------|
    | **Simulation** | SUMO (Simulation of Urban Mobility) |
    | **Data Collection** | TraCI (Python API) |
    | **Deep Learning** | PyTorch, LSTM, Attention Mechanism |
    | **Data Processing** | Pandas, NumPy, Scikit-learn |
    | **Visualization** | Streamlit, Plotly, Matplotlib |
    | **Deployment** | Docker, Flask/FastAPI (ready) |
    """)

# ==================== FOOTER ====================
st.markdown("""
    <div class="footer">
        <p><strong>SynCity Traffic Intelligence Platform</strong> | v2.0</p>
        <p>Multi-Horizon Attention-LSTM | Real-Time Inference | Smart City Integration</p>
        <p style="margin-top: 1rem; font-size: 0.85rem;">
            Developed with ❤️ for Smart Urban Mobility | 
            <a href="#" style="color: #667eea; text-decoration: none;">Documentation</a> | 
            <a href="#" style="color: #667eea; text-decoration: none;">GitHub</a> | 
            <a href="#" style="color: #667eea; text-decoration: none;">Research Paper</a>
        </p>
    </div>
""", unsafe_allow_html=True)