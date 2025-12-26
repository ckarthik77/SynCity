import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="SynCity | Traffic Intelligence Platform",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== GITHUB & PROJECT INFO ====================
GITHUB_REPO = "https://github.com/ckarthik77/SynCity"
DOCUMENTATION_URL = f"{GITHUB_REPO}#readme"
RESEARCH_PAPER_URL = f"{GITHUB_REPO}/blob/main/docs/RESEARCH_PAPER.pdf"

PROJECT_INFO = {
    "version": "v2.0",
    "model": "Multi-Horizon Attention-LSTM",
    "authors": ["Ch. Karthikeya (22761A5477)", "Ch. Sai Jyothi (22761A5481)", "O. Vinay Venkatesh (22761A54A8)"],
    "institution": "LBRCE, Mylavaram",
    "supervisor": "Mrs. K. Lakshmi Padmavathi"
}

# ==================== ENHANCED CUSTOM CSS ====================
st.markdown("""
    <style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }
    
    /* Hero Section - Enhanced */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3.5rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow: 0 25px 70px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        animation: pulse 10s ease-in-out infinite;
    }
    
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(118, 75, 162, 0.2) 0%, transparent 70%);
        animation: pulse 15s ease-in-out infinite reverse;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.5; }
        50% { transform: scale(1.1) rotate(5deg); opacity: 0.8; }
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        font-weight: 500;
        color: rgba(255,255,255,0.95);
        margin-top: 0.75rem;
        position: relative;
        z-index: 1;
        line-height: 1.6;
    }
    
    .hero-tagline {
        font-size: 1.05rem;
        font-weight: 400;
        color: rgba(255,255,255,0.8);
        margin-top: 1.25rem;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        color: white;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards - Enhanced */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        transform: translateX(-100%);
        transition: transform 0.6s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    .metric-card:hover::before {
        transform: translateX(0);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.75rem 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .metric-sublabel {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 0.5rem;
    }
    
    /* Section Headers - Enhanced */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2, transparent) 1;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Sidebar - Enhanced */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .sidebar-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar-badge p {
        margin: 0;
        color: white;
        font-weight: 600;
    }
    
    /* Radio Buttons - Enhanced */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stRadio > div:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .stRadio label {
        color: rgba(255, 255, 255, 0.85) !important;
        font-size: 1.05rem;
        padding: 0.75rem;
        transition: all 0.3s ease;
        border-radius: 10px;
        font-weight: 500;
    }
    
    .stRadio label:hover {
        color: white !important;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
    }
    
    /* Selectbox - Enhanced */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        color: white;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Tabs - Enhanced */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: rgba(255, 255, 255, 0.03);
        padding: 0.75rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.6);
        border-radius: 10px;
        padding: 0.85rem 1.75rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.15);
        color: white;
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        border-color: transparent !important;
    }
    
    /* Alert Boxes - Enhanced */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.05) !important;
        border-left: 4px solid !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        padding: 1.25rem !important;
    }
    
    .stSuccess {
        border-color: #2ecc71 !important;
    }
    
    .stInfo {
        border-color: #3498db !important;
    }
    
    .stWarning {
        border-color: #f39c12 !important;
    }
    
    /* Footer - Enhanced */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.95rem;
        margin-top: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(0, 0, 0, 0.2);
        border-radius: 20px 20px 0 0;
    }
    
    .footer-links {
        margin-top: 1.5rem;
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
    }
    
    .footer-link {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    
    .footer-link:hover {
        color: #f093fb;
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    /* Stats Badge */
    .stats-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Improved Dataframe */
    .dataframe {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Loading State */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading {
        background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(102,126,234,0.1) 50%, rgba(255,255,255,0.03) 75%);
        background-size: 1000px 100%;
        animation: shimmer 2.5s infinite;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2, #f093fb);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== ENHANCED HERO SECTION ====================
st.markdown(f"""
    <div class="hero-container">
        <h1 class="hero-title">🏙️ SynCity</h1>
        <p class="hero-subtitle">Synchronized Urban Traffic via AV-Infrastructure Synergy</p>
        <p class="hero-tagline">Multi-Horizon Attention-Based LSTM | Real-Time Traffic Intelligence | Smart City Integration</p>
        <span class="hero-badge">📊 {PROJECT_INFO['model']} {PROJECT_INFO['version']}</span>
    </div>
""", unsafe_allow_html=True)

# ==================== ENHANCED SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    view_mode = st.radio(
        "Select Analysis Mode",
        ["🎯 Real-Time Predictions", "📊 Model Performance", "🔬 System Architecture", "👥 Team Info"],
        label_visibility="collapsed",
        key="view_mode_radio"
    )
    
    st.markdown("---")
    
    st.markdown("### 📌 Quick Stats")
    st.markdown(f"""
        <div class='sidebar-badge'>
            <p style='font-size: 0.75rem; margin-bottom: 0.25rem; opacity: 0.8;'>MODEL VERSION</p>
            <p style='font-size: 1.1rem; font-weight: 700;'>{PROJECT_INFO['model']}</p>
            <p style='font-size: 0.9rem; margin-top: 0.25rem;'>{PROJECT_INFO['version']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔗 Quick Links")
    st.markdown(f"""
        <div style='display: flex; flex-direction: column; gap: 0.75rem;'>
            <a href='{DOCUMENTATION_URL}' target='_blank' class='footer-link' style='display: block; text-align: center;'>
                📖 Documentation
            </a>
            <a href='{GITHUB_REPO}' target='_blank' class='footer-link' style='display: block; text-align: center;'>
                🐙 GitHub Repository
            </a>
            <a href='{RESEARCH_PAPER_URL}' target='_blank' class='footer-link' style='display: block; text-align: center;'>
                📊 Research Paper
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📅 Last Updated")
    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px; text-align: center;'>
            <p style='color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem;'>{datetime.now().strftime('%B %d, %Y')}</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_realtime_data():
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'realtime_predictions.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        else:
            st.warning(f"File not found: {csv_path}")
            return None
    except Exception as e:
        st.warning(f"Error loading realtime predictions: {e}")
        return None

@st.cache_data
def load_evaluation_metrics():
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'evaluation_metrics.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        else:
            st.warning(f"File not found: {csv_path}")
            return None
    except Exception as e:
        st.warning(f"Error loading evaluation metrics: {e}")
        return None

# ==================== MAIN CONTENT ====================

if view_mode == "🎯 Real-Time Predictions":
    st.markdown('<p class="section-header">📡 Real-Time Traffic Prediction Analysis</p>', unsafe_allow_html=True)
    
    df = load_realtime_data()
    
    if df is not None and len(df) > 0:
        # Enhanced Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Total Predictions</p>
                    <p class="metric-value">{len(df):,}</p>
                    <p class="metric-sublabel">Generated in real-time</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            unique_vehicles = df['vehicle_id'].nunique()
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Unique Vehicles</p>
                    <p class="metric-value">{unique_vehicles}</p>
                    <p class="metric-sublabel">Tracked simultaneously</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            duration = df['time'].max()
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Simulation Duration</p>
                    <p class="metric-value">{duration:.0f}s</p>
                    <p class="metric-sublabel">{duration/60:.1f} minutes</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_speed = df['current_speed'].mean()
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Average Speed</p>
                    <p class="metric-value">{avg_speed:.1f}</p>
                    <p class="metric-sublabel">meters per second</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Vehicle Analysis Section
        st.markdown('<p class="section-header">🚗 Individual Vehicle Trajectory Analysis</p>', unsafe_allow_html=True)
        
        vehicles = df['vehicle_id'].unique()
        selected_vehicle = st.selectbox(
            "Select Vehicle ID for Detailed Analysis", 
            vehicles[:100],
            label_visibility="collapsed"
        )
        
        vehicle_data = df[df['vehicle_id'] == selected_vehicle].sort_values('time')
        
        if len(vehicle_data) > 0:
            # Create enhanced tabs
            tab1, tab2, tab3 = st.tabs(["📈 Speed Predictions", "🔄 Delta Analysis", "📊 Statistics"])
            
            with tab1:
                fig = go.Figure()
                
                # Current speed with filled area
                fig.add_trace(go.Scatter(
                    x=vehicle_data['time'],
                    y=vehicle_data['current_speed'],
                    mode='lines',
                    name='Current Speed',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.15)'
                ))
                
                # Prediction horizons
                predictions = [
                    ('predicted_30s', 'Predicted @ 30s', '#2ecc71', 'dash'),
                    ('predicted_90s', 'Predicted @ 90s', '#f39c12', 'dot'),
                    ('predicted_150s', 'Predicted @ 150s', '#e74c3c', 'dashdot')
                ]
                
                for col, name, color, dash in predictions:
                    fig.add_trace(go.Scatter(
                        x=vehicle_data['time'],
                        y=vehicle_data[col],
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=2.5, dash=dash)
                    ))
                
                fig.update_layout(
                    title={
                        'text': f"Multi-Horizon Speed Predictions | Vehicle: {selected_vehicle}",
                        'font': {'size': 20, 'family': 'Inter', 'weight': 700}
                    },
                    xaxis_title="Simulation Time (seconds)",
                    yaxis_title="Speed (m/s)",
                    hovermode='x unified',
                    height=550,
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='white', size=12),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig2 = go.Figure()
                
                deltas = [
                    ('delta_30s', 'Δ @ 30s', '#2ecc71'),
                    ('delta_90s', 'Δ @ 90s', '#f39c12'),
                    ('delta_150s', 'Δ @ 150s', '#e74c3c')
                ]
                
                for col, name, color in deltas:
                    fig2.add_trace(go.Scatter(
                        x=vehicle_data['time'],
                        y=vehicle_data[col],
                        mode='lines+markers',
                        name=name,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7, line=dict(width=1, color='white'))
                    ))
                
                fig2.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="rgba(255,255,255,0.3)",
                    annotation_text="Zero Change",
                    annotation_position="right"
                )
                
                fig2.update_layout(
                    title={
                        'text': "Predicted Speed Changes (Δ)",
                        'font': {'size': 20, 'family': 'Inter', 'weight': 700}
                    },
                    xaxis_title="Simulation Time (seconds)",
                    yaxis_title="Δ Speed (m/s)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='white', size=12),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.markdown("#### 📊 Speed Statistics")
                    st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);'>
                            <p><strong>Mean Speed:</strong> <span class='stats-badge'>{vehicle_data['current_speed'].mean():.2f} m/s</span></p>
                            <p><strong>Max Speed:</strong> <span class='stats-badge'>{vehicle_data['current_speed'].max():.2f} m/s</span></p>
                            <p><strong>Min Speed:</strong> <span class='stats-badge'>{vehicle_data['current_speed'].min():.2f} m/s</span></p>
                            <p><strong>Std Dev:</strong> <span class='stats-badge'>{vehicle_data['current_speed'].std():.2f} m/s</span></p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with stats_col2:
                    st.markdown("#### 🎯 Prediction Statistics")
                    st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);'>
                            <p><strong>Avg Δ30s:</strong> <span class='stats-badge'>{vehicle_data['delta_30s'].mean():.3f} m/s</span></p>
                            <p><strong>Avg Δ90s:</strong> <span class='stats-badge'>{vehicle_data['delta_90s'].mean():.3f} m/s</span></p>
                            <p><strong>Avg Δ150s:</strong> <span class='stats-badge'>{vehicle_data['delta_150s'].mean():.3f} m/s</span></p>
                            <p><strong>Data Points:</strong> <span class='stats-badge'>{len(vehicle_data)}</span></p>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Fleet-Wide Analysis
        st.markdown('<p class="section-header">📊 Fleet-Wide Traffic Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_data = df.sample(min(5000, len(df)))
            fig3 = px.histogram(
                sample_data,
                x='current_speed',
                nbins=50,
                title="Speed Distribution Across All Vehicles",
                labels={'current_speed': 'Speed (m/s)', 'count': 'Frequency'},
                template='plotly_dark',
                color_discrete_sequence=['#667eea']
            )
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='white'),
                height=450
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = px.histogram(
                sample_data,
                x='delta_30s',
                nbins=50,
                title="30s Prediction Delta Distribution",
                labels={'delta_30s': 'Δ Speed (m/s)', 'count': 'Frequency'},
                template='plotly_dark',
                color_discrete_sequence=['#f093fb']
            )
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='white'),
                height=450
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    else:
        st.warning("⚠️ No real-time prediction data available. Run `traci_logger_realtime.py` first.")

elif view_mode == "📊 Model Performance":
    st.markdown('<p class="section-header">🎯 Model Performance Evaluation</p>', unsafe_allow_html=True)
    metrics_df = load_evaluation_metrics()

    if metrics_df is not None:
        # Performance Summary Cards
        st.markdown("#### 💡 Key Performance Insights")
        
        best_r2_horizon = metrics_df['R²'].idxmax()
        best_r2_value = metrics_df['R²'].max()
        avg_mae = metrics_df['MAE'].mean()
        best_mae_horizon = metrics_df['MAE'].idxmin()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"""
🏆 Best R² Score
{best_r2_value:.4f} @ {best_r2_horizon}
Exceptional prediction accuracy
""")
        with col2:
            st.info(f"""
📊 Average MAE
{avg_mae:.4f}
Across all horizons
""")
        with col3:
            st.warning(f"""
🎯 Best Accuracy
MAE {metrics_df['MAE'].min():.4f}
@ {best_mae_horizon}
""")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics Table
        st.markdown("#### 📈 Detailed Performance Metrics")
        
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
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics Visualization
        fig_metrics = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 
                          'R² Score (Coefficient of Determination)', 'Mean Absolute Percentage Error (MAPE)'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        horizons = metrics_df.index.tolist()
        colors = ['#667eea', '#764ba2', '#f093fb']
        
        # MAE
        fig_metrics.add_trace(
            go.Bar(
                x=horizons, 
                y=metrics_df['MAE'], 
                marker_color=colors, 
                showlegend=False,
                text=metrics_df['MAE'].round(4),
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # RMSE
        fig_metrics.add_trace(
            go.Bar(
                x=horizons, 
                y=metrics_df['RMSE'], 
                marker_color=colors, 
                showlegend=False,
                text=metrics_df['RMSE'].round(4),
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # R²
        fig_metrics.add_trace(
            go.Bar(
                x=horizons, 
                y=metrics_df['R²'], 
                marker_color=colors, 
                showlegend=False,
                text=metrics_df['R²'].round(4),
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # MAPE
        fig_metrics.add_trace(
            go.Bar(
                x=horizons, 
                y=metrics_df['MAPE'], 
                marker_color=colors, 
                showlegend=False,
                text=metrics_df['MAPE'].round(2).astype(str) + '%',
                textposition='outside'
            ),
            row=2, col=2
        )
        
        fig_metrics.update_layout(
            height=750,
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='white', size=12),
            showlegend=False
        )
        fig_metrics.update_xaxes(tickangle=15)
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
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
        
        # Get script directory for absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        for idx, (title, path) in enumerate(viz_files.items()):
            with tabs[idx]:
                try:
                    full_path = os.path.join(script_dir, path)
                    if os.path.exists(full_path):
                        st.image(full_path, use_container_width=True, caption=title)
                    else:
                        st.warning(f"⚠️ {title} visualization not found at `{full_path}`. Please ensure evaluation script has been run.")
                except Exception as e:
                    st.warning(f"⚠️ Error loading {title}: {str(e)}")

    else:
        st.warning("⚠️ Evaluation metrics not found. Please run the evaluation script or download files from Colab.")

elif view_mode == "🔬 System Architecture":
    st.markdown('<p class="section-header">🏗️ System Architecture Overview</p>', unsafe_allow_html=True)
    st.markdown("""
### SynCity: Synchronized Urban Traffic via AV-Infrastructure Synergy

A comprehensive AI-powered traffic prediction and management system leveraging multi-horizon attention-based LSTM neural networks.
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
#### 🧠 AI/ML Pipeline

**Data Collection Layer**
- **SUMO Simulation**: Urban traffic microscopic simulation
- **TraCI Integration**: Real-time Python API control
- **Feature Extraction**: 6 validated traffic parameters

**Model Architecture**
- **LSTM Encoder**: 2 layers, 128 hidden units
- **Attention Mechanism**: Temporal attention weights
- **Multi-Horizon Heads**: 30s, 90s, 150s predictions
- **Total Parameters**: ~250,000

**Training Configuration**
- **Dataset**: 74,000+ timesteps
- **Epochs**: 50 with early stopping
- **Optimizer**: Adam (lr=0.001)
- **Hardware**: NVIDIA Tesla P100 GPU
""")

    with col2:
        st.markdown("""
#### 📊 Performance Benchmarks

**Model Metrics**
- **1-step (30s)**: R² = 0.89, MAE = 0.045
- **3-step (90s)**: R² = 0.90, MAE = 0.094
- **5-step (150s)**: R² = 0.87, MAE = 0.169

**Real-Time Performance**
- **Inference Time**: <10ms per prediction
- **Throughput**: 100+ vehicles simultaneously
- **Latency**: 5-second update intervals

**Deployment Ready**
- Docker containerization
- RESTful API endpoints
- Scalable cloud architecture
""")

    st.markdown("---")

    st.markdown("### 🔄 System Data Flow")

    st.code("""
┌─────────────────────────────────────────────────────────────────────┐
│                    SUMO Traffic Simulation                           │
│                   (Urban Mobility Network)                           │
└──────────────────────────┬──────────────────────────────────────────┘
│ TraCI API
▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Real-Time Feature Extraction                        │
│  • Vehicle Speed        • Acceleration      • Front Vehicle Distance │
│  • Leader Speed         • Lane Density      • Average Lane Speed     │
└──────────────────────────┬──────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│              Rolling Window Buffer (SEQ_LEN=30)                      │
│                    Maintains temporal context                        │
└──────────────────────────┬──────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│            Multi-Horizon Attention-LSTM Model                        │
│                                                                       │
│  ┌────────────────────────────────────────────────┐                 │
│  │  LSTM Encoder (2 layers, 128 units each)      │                 │
│  └───────────────────┬────────────────────────────┘                 │
│                      │                                               │
│  ┌───────────────────▼────────────────────────────┐                 │
│  │  Temporal Attention Mechanism                  │                 │
│  │  (Learns critical past timestep importance)    │                 │
│  └───────────────────┬────────────────────────────┘                 │
│                      │                                               │
│         ┌────────────┼────────────┐                                 │
│         ▼            ▼            ▼                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                            │
│  │ Head 30s │ │ Head 90s │ │ Head 150s│                            │
│  └──────────┘ └──────────┘ └──────────┘                            │
└──────────────────────────┬──────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│              Multi-Horizon Speed Predictions                         │
│         Δspeed_30s  |  Δspeed_90s  |  Δspeed_150s                   │
└──────────────────────────┬──────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│           Decision Support & Traffic Management                      │
│  • Signal Optimization  • Route Planning  • Congestion Alert        │
└─────────────────────────────────────────────────────────────────────┘
""", language="text")
    st.markdown("---")

    st.markdown("### 🚀 Key Innovations")

    innov_col1, innov_col2 = st.columns(2)

    with innov_col1:
        st.markdown("""
**1. Attention Mechanism**
- Identifies which past timesteps are most influential
- Provides interpretability for predictions
- Improves accuracy by 27% over standard LSTM

**2. Multi-Horizon Architecture**
- Simultaneous predictions at 3 time horizons
- Enables short-term and long-term traffic planning
- Single model inference reduces computational overhead
""")

    with innov_col2:
        st.markdown("""
**3. Real-Time Integration**
- Live SUMO simulation connectivity
- Sub-10ms inference latency
- Production-ready deployment pipeline

**4. Scalable Design**
- Handles 100+ vehicles concurrently
- Cloud-native architecture
- Docker containerization for easy deployment
""")

    st.markdown("---")

    st.markdown("### 📚 Technology Stack")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("""
**Simulation & Data**
- SUMO v1.14+
- TraCI Python API
- OpenStreetMap
- Pandas & NumPy
""")

    with tech_col2:
        st.markdown("""
**Deep Learning**
- PyTorch 2.0
- LSTM Networks
- Attention Layers
- Scikit-learn
""")

    with tech_col3:
        st.markdown("""
**Deployment & UI**
- Streamlit
- Plotly & Matplotlib
- Docker
- FastAPI/Flask
""")

    st.markdown("---")

    st.markdown("### 🎯 Real-World Applications")

    apps_col1, apps_col2 = st.columns(2)

    with apps_col1:
        st.markdown("""
**Traffic Management**
- Dynamic signal timing optimization
- Congestion prediction and early warning
- Incident detection and response
- Emergency vehicle routing priority
""")

    with apps_col2:
        st.markdown("""
**Smart City Integration**
- Autonomous vehicle coordination
- Public transport optimization
- Environmental impact reduction
- Urban planning data insights
""")

else:  # Team Info
    st.markdown('<p class="section-header">👥 Project Team & Information</p>', unsafe_allow_html=True)
    st.markdown("""
### 🎓 Project Team

This project was developed as part of the Final Year Project at **Lakireddy Bali Reddy College of Engineering (LBRCE), Mylavaram**.
""")

    team_col1, team_col2, team_col3 = st.columns(3)

    with team_col1:
        st.markdown("""
<div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); 
            padding: 2rem; border-radius: 16px; border: 1px solid rgba(102, 126, 234, 0.3); text-align: center;'>
    <h3 style='color: #667eea; margin-bottom: 1rem;'>👨‍💻 Ch. Karthikeya</h3>
    <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem;'><strong>22761A5477</strong></p>
    <p style='color: rgba(255,255,255,0.6); margin-top: 0.5rem;'>AI & Data Science</p>
</div>
""", unsafe_allow_html=True)

    with team_col2:
        st.markdown("""
<div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); 
            padding: 2rem; border-radius: 16px; border: 1px solid rgba(102, 126, 234, 0.3); text-align: center;'>
    <h3 style='color: #667eea; margin-bottom: 1rem;'>👩‍💻 Ch. Sai Jyothi</h3>
    <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem;'><strong>22761A5481</strong></p>
    <p style='color: rgba(255,255,255,0.6); margin-top: 0.5rem;'>AI & Data Science</p>
</div>
""", unsafe_allow_html=True)

    with team_col3:
        st.markdown("""
<div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); 
            padding: 2rem; border-radius: 16px; border: 1px solid rgba(102, 126, 234, 0.3); text-align: center;'>
    <h3 style='color: #667eea; margin-bottom: 1rem;'>👨‍💻 O. Vinay Venkatesh</h3>
    <p style='color: rgba(255,255,255,0.8); font-size: 1.1rem;'><strong>22761A54A8</strong></p>
    <p style='color: rgba(255,255,255,0.6); margin-top: 0.5rem;'>AI & Data Science</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
### 👨‍🏫 Project Supervision

**Mrs. K. Lakshmi Padmavathi**  
Assistant Professor  
Department of AI & Data Science  
LBRCE, Mylavaram
""")

    st.markdown("---")

    st.markdown("### 🏛️ Institution")

    inst_col1, inst_col2 = st.columns([1, 2])

    with inst_col1:
        st.markdown("""
**Lakireddy Bali Reddy College of Engineering**

Mylavaram, Krishna District  
Andhra Pradesh, India
""")

    with inst_col2:
        st.markdown("""
**Department of Artificial Intelligence & Data Science**

Fostering innovation in AI/ML research and development, with focus on real-world applications 
in smart cities, autonomous systems, and data-driven decision making.
""")

    st.markdown("---")

    st.markdown("### 📅 Project Timeline")

    timeline_data = {
        "Phase": ["Research & Planning", "Data Collection", "Model Development", "Real-Time Integration", "Testing & Validation", "Documentation & Deployment"],
        "Duration": ["2 weeks", "3 weeks", "4 weeks", "2 weeks", "2 weeks", "1 week"],
        "Status": ["✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete", "🔄 In Progress"]
    }

    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("### 🔗 Project Resources")

    resource_col1, resource_col2, resource_col3 = st.columns(3)

    with resource_col1:
        st.markdown(f"""
<a href='{GITHUB_REPO}' target='_blank' style='text-decoration: none;'>
    <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; border-radius: 16px; text-align: center; transition: transform 0.3s;'>
        <h3 style='color: white; margin: 0;'>🐙 GitHub</h3>
        <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem;'>Source Code</p>
    </div>
</a>
""", unsafe_allow_html=True)

    with resource_col2:
        st.markdown(f"""
<a href='{DOCUMENTATION_URL}' target='_blank' style='text-decoration: none;'>
    <div style='background: linear-gradient(135deg, #764ba2, #f093fb); padding: 2rem; border-radius: 16px; text-align: center; transition: transform 0.3s;'>
        <h3 style='color: white; margin: 0;'>📖 Docs</h3>
        <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem;'>Documentation</p>
    </div>
</a>
""", unsafe_allow_html=True)

    with resource_col3:
        st.markdown(f"""
<a href='{RESEARCH_PAPER_URL}' target='_blank' style='text-decoration: none;'>
    <div style='background: linear-gradient(135deg, #f093fb, #667eea); padding: 2rem; border-radius: 16px; text-align: center; transition: transform 0.3s;'>
        <h3 style='color: white; margin: 0;'>📊 Paper</h3>
        <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem;'>Research</p>
    </div>
</a>
""", unsafe_allow_html=True)

# ==================== ENHANCED FOOTER ====================
st.markdown(f"""
<div class="footer">
<p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>
<strong>SynCity Traffic Intelligence Platform</strong> {PROJECT_INFO['version']}
</p>
<p style='margin: 0.5rem 0;'>
{PROJECT_INFO['model']} | Real-Time Inference | Smart City Integration
</p>
<p style='margin: 1rem 0 0.5rem 0; color: rgba(255,255,255,0.6);'>
Developed by {', '.join(PROJECT_INFO['authors'][:2])} & {PROJECT_INFO['authors'][2]}
</p>
<p style='color: rgba(255,255,255,0.5);'>
{PROJECT_INFO['institution']} | Under supervision of {PROJECT_INFO['supervisor']}
</p>
<div class='footer-links'>
<a href='{GITHUB_REPO}' target='_blank' class='footer-link'>🐙 GitHub</a>
<a href='{DOCUMENTATION_URL}' target='_blank' class='footer-link'>📖 Documentation</a>
<a href='{RESEARCH_PAPER_URL}' target='_blank' class='footer-link'>📊 Research Paper</a>
</div>
<p style='margin-top: 1.5rem; font-size: 0.85rem; color: rgba(255,255,255,0.4);'>
© 2024-2025 SynCity Project. Built with ❤️ for Smart Urban Mobility.
</p>
</div>
""", unsafe_allow_html=True)