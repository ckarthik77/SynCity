#!/usr/bin/env python3
"""
generate_paper_figures.py
Generate all publication-quality figures for SynCity paper
"""

import pandas as pd
import numpy as np
import torch
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add Model folder to path
sys.path.append(r"E:\Desktop\Jobs\projects\Syncity\Model")
from model_attention_lstm import MultiHorizonAttentionLSTM

# ================= MATPLOTLIB SETTINGS =================
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Set style
sns.set_style("whitegrid")
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange

# ================= CONFIG =================
DATA_FILE = "syncity_clean_dataset.csv"
MODEL_PATH = "Model/multihorizon_lstm_new.pth"
SCALER_PATH = "Model/multihorizon_scaler_new.pkl"
SEQUENCE_LENGTH = 30
HORIZONS = {'30s': 60, '90s': 180, '150s': 300}
OUTPUT_DIR = "Figures"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("🎨 Generating Publication Figures for SynCity Paper")
print("="*80)

# ================= LOAD MODEL & DATA =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultiHorizonAttentionLSTM(
    input_size=6,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with open(SCALER_PATH, 'rb') as f:
    scaler_dict = pickle.load(f)

X_scaler = scaler_dict['X_scaler']
y_scaler_30s = scaler_dict['y_scaler_30s']
y_scaler_90s = scaler_dict['y_scaler_90s']
y_scaler_150s = scaler_dict['y_scaler_150s']

print("✅ Model and scalers loaded")

# ================= PREPARE DATA =================
def create_sequences(df, seq_length=30):
    sequences = []
    targets_30s = []
    targets_90s = []
    targets_150s = []
    current_speeds = []
    lane_densities = []
    
    vehicle_groups = df.groupby('veh_id')
    
    for veh_id, veh_data in vehicle_groups:
        veh_data = veh_data.sort_values('time').reset_index(drop=True)
        
        if len(veh_data) < seq_length + HORIZONS['150s']:
            continue
        
        features = veh_data[['speed', 'accel', 'front_vehicle_dist', 
                             'front_vehicle_speed', 'lane_density', 'avg_lane_speed']].values
        
        speeds = veh_data['speed'].values
        densities = veh_data['lane_density'].values
        
        for i in range(len(features) - seq_length - HORIZONS['150s']):
            seq = features[i:i+seq_length]
            current_speed = speeds[i + seq_length - 1]
            
            target_30s = speeds[i + seq_length + HORIZONS['30s']] - current_speed
            target_90s = speeds[i + seq_length + HORIZONS['90s']] - current_speed
            target_150s = speeds[i + seq_length + HORIZONS['150s']] - current_speed
            
            sequences.append(seq)
            targets_30s.append(target_30s)
            targets_90s.append(target_90s)
            targets_150s.append(target_150s)
            current_speeds.append(current_speed)
            lane_densities.append(densities[i + seq_length - 1])
    
    return (np.array(sequences), 
            np.array(targets_30s), 
            np.array(targets_90s), 
            np.array(targets_150s),
            np.array(current_speeds),
            np.array(lane_densities))

print("\n📂 Loading and processing data...")
df = pd.read_csv(DATA_FILE)
X, y_30s_delta, y_90s_delta, y_150s_delta, current_speeds, lane_densities = create_sequences(df, SEQUENCE_LENGTH)

# Sample subset for faster processing (use 5000 samples)
sample_size = 5000
sample_idx = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
X_sample = X[sample_idx]
y_30s_sample = y_30s_delta[sample_idx]
y_90s_sample = y_90s_delta[sample_idx]
y_150s_sample = y_150s_delta[sample_idx]
current_speeds_sample = current_speeds[sample_idx]
lane_densities_sample = lane_densities[sample_idx]

print(f"✅ Using {len(X_sample)} samples for visualization")

# Normalize and predict
X_reshaped = X_sample.reshape(-1, X_sample.shape[-1])
X_normalized_reshaped = X_scaler.transform(X_reshaped)
X_normalized = X_normalized_reshaped.reshape(X_sample.shape)

X_tensor = torch.FloatTensor(X_normalized).to(device)

with torch.no_grad():
    pred_30s_norm, pred_90s_norm, pred_150s_norm, attention_weights = model(X_tensor)
    pred_30s_norm = pred_30s_norm.cpu().numpy().flatten()
    pred_90s_norm = pred_90s_norm.cpu().numpy().flatten()
    pred_150s_norm = pred_150s_norm.cpu().numpy().flatten()
    attention_weights = attention_weights.cpu().numpy()

# Denormalize
pred_30s_delta = y_scaler_30s.inverse_transform(pred_30s_norm.reshape(-1, 1)).flatten()
pred_90s_delta = y_scaler_90s.inverse_transform(pred_90s_norm.reshape(-1, 1)).flatten()
pred_150s_delta = y_scaler_150s.inverse_transform(pred_150s_norm.reshape(-1, 1)).flatten()

pred_30s_abs = np.clip(current_speeds_sample + pred_30s_delta, 0, 30)
pred_90s_abs = np.clip(current_speeds_sample + pred_90s_delta, 0, 30)
pred_150s_abs = np.clip(current_speeds_sample + pred_150s_delta, 0, 30)

y_30s_abs = current_speeds_sample + y_30s_sample
y_90s_abs = current_speeds_sample + y_90s_sample
y_150s_abs = current_speeds_sample + y_150s_sample

print("✅ Predictions generated")

# ================= FIGURE 1: PREDICTION VS GROUND TRUTH TIME SERIES =================
print("\n📊 Generating Figure 1: Prediction vs Ground Truth...")

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# Select 3 interesting sample trajectories
sample_vehicles = [0, 1000, 2000]

for idx, (ax, sample_idx, color) in enumerate(zip(axes, sample_vehicles, colors)):
    time_points = np.arange(0, 151, 1)  # 150 seconds
    
    # Plot ground truth
    true_vals = [current_speeds_sample[sample_idx], 
                 y_30s_abs[sample_idx], 
                 y_90s_abs[sample_idx], 
                 y_150s_abs[sample_idx]]
    true_times = [0, 30, 90, 150]
    
    # Plot predictions
    pred_vals = [current_speeds_sample[sample_idx], 
                 pred_30s_abs[sample_idx], 
                 pred_90s_abs[sample_idx], 
                 pred_150s_abs[sample_idx]]
    
    # Interpolate for smooth lines
    ax.plot(true_times, true_vals, 'o-', color='black', linewidth=2.5, 
            markersize=8, label='Ground Truth', alpha=0.8)
    ax.plot(true_times, pred_vals, 's--', color=color, linewidth=2.5, 
            markersize=8, label='Predicted', alpha=0.8)
    
    ax.set_ylabel('Speed (m/s)', fontweight='bold')
    ax.set_xlabel('Time Horizon (seconds)', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Sample Vehicle {idx+1} (Initial Speed: {current_speeds_sample[sample_idx]:.1f} m/s)', 
                 fontweight='bold')
    ax.set_xlim(-5, 155)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure1_prediction_vs_truth.png', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/figure1_prediction_vs_truth.pdf', bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/figure1_prediction_vs_truth.png")
plt.close()

# ================= FIGURE 2: ERROR DISTRIBUTION =================
print("\n📊 Generating Figure 2: Error Distributions...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

horizons = [
    ('30s', pred_30s_abs, y_30s_abs),
    ('90s', pred_90s_abs, y_90s_abs),
    ('150s', pred_150s_abs, y_150s_abs)
]

for ax, (horizon_name, pred, true), color in zip(axes, horizons, colors):
    errors = pred - true
    
    # Histogram
    n, bins, patches = ax.hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Fit normal distribution
    mu, sigma = errors.mean(), errors.std()
    x = np.linspace(errors.min(), errors.max(), 100)
    ax.plot(x, len(errors) * (bins[1] - bins[0]) * stats.norm.pdf(x, mu, sigma),
            'r-', linewidth=2.5, label=f'Normal Fit\nμ={mu:.3f}, σ={sigma:.3f}')
    
    # Zero line
    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    ax.set_xlabel('Prediction Error (m/s)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{horizon_name} Horizon\nMAE={mae:.3f}, RMSE={rmse:.3f}', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure2_error_distribution.png', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/figure2_error_distribution.pdf', bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/figure2_error_distribution.png")
plt.close()

# ================= FIGURE 3: MAE VS TRAFFIC DENSITY =================
print("\n📊 Generating Figure 3: MAE vs Traffic Density...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

EDGE_LENGTH_KM = 0.07001
density_veh_km = lane_densities_sample / EDGE_LENGTH_KM

for ax, (horizon_name, pred, true), color in zip(axes, horizons, colors):
    errors = np.abs(pred - true)
    
    # Scatter plot with transparency
    ax.scatter(density_veh_km, errors, alpha=0.3, s=20, color=color, edgecolors='none')
    
    # Binned average
    bins = np.arange(0, density_veh_km.max() + 10, 5)
    bin_centers = []
    bin_means = []
    bin_stds = []
    
    for i in range(len(bins)-1):
        mask = (density_veh_km >= bins[i]) & (density_veh_km < bins[i+1])
        if mask.sum() > 10:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(errors[mask].mean())
            bin_stds.append(errors[mask].std())
    
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    # Plot binned average with confidence interval
    ax.plot(bin_centers, bin_means, 'r-', linewidth=3, label='Binned Mean', zorder=5)
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, 
                     color='red', alpha=0.2, label='±1 Std Dev')
    
    # Regime boundaries
    ax.axvline(15, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Free-flow/Moderate')
    ax.axvline(40, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Moderate/Congested')
    
    ax.set_xlabel('Traffic Density (veh/km)', fontweight='bold')
    ax.set_ylabel('Absolute Error (m/s)', fontweight='bold')
    ax.set_title(f'{horizon_name} Horizon', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(density_veh_km.max(), 80))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure3_mae_vs_density.png', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/figure3_mae_vs_density.pdf', bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/figure3_mae_vs_density.png")
plt.close()

# ================= FIGURE 4: ATTENTION WEIGHTS HEATMAP =================
print("\n📊 Generating Figure 4: Attention Weights Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Select samples with different characteristics
# Sample 1: Low initial speed (likely accelerating)
low_speed_idx = np.argmin(current_speeds_sample[:1000])
# Sample 2: High initial speed (likely decelerating)
high_speed_idx = np.argmax(current_speeds_sample[:1000])

samples = [low_speed_idx, high_speed_idx]
titles = [
    f'Accelerating Vehicle\n(Initial: {current_speeds_sample[low_speed_idx]:.1f} m/s)',
    f'Decelerating Vehicle\n(Initial: {current_speeds_sample[high_speed_idx]:.1f} m/s)'
]

for ax, sample_idx, title in zip(axes, samples, titles):
    # Get attention weights for this sample (shape: [30])
    attn = attention_weights[sample_idx]
    
    # Create time indices (t-29 to t-0)
    time_indices = np.arange(-29, 1)
    
    # Bar plot
    bars = ax.bar(time_indices, attn, color=colors[0], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Highlight most important timesteps
    top_5_idx = np.argsort(attn)[-5:]
    for idx in top_5_idx:
        bars[idx].set_color(colors[2])
        bars[idx].set_alpha(0.9)
    
    ax.set_xlabel('Timestep (relative to current)', fontweight='bold')
    ax.set_ylabel('Attention Weight', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-30, 1)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure4_attention_weights.png', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/figure4_attention_weights.pdf', bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/figure4_attention_weights.png")
plt.close()

# ================= FIGURE 5: SCATTER PLOT - PREDICTED VS TRUE =================
print("\n📊 Generating Figure 5: Predicted vs True Scatter Plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (horizon_name, pred, true), color in zip(axes, horizons, colors):
    # Scatter plot
    ax.scatter(true, pred, alpha=0.4, s=15, color=color, edgecolors='none')
    
    # Perfect prediction line
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, 
            label='Perfect Prediction', alpha=0.8)
    
    # Calculate metrics
    from sklearn.metrics import r2_score
    r2 = r2_score(true, pred)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    
    # Add text box with metrics
    textstr = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('True Speed (m/s)', fontweight='bold')
    ax.set_ylabel('Predicted Speed (m/s)', fontweight='bold')
    ax.set_title(f'{horizon_name} Horizon', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure5_scatter_pred_vs_true.png', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/figure5_scatter_pred_vs_true.pdf', bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/figure5_scatter_pred_vs_true.png")
plt.close()

# ================= FIGURE 6: CROSS-VALIDATION RESULTS =================
print("\n📊 Generating Figure 6: Cross-Validation Results...")

# Load CV results
import json
with open('Model/cv_results.json', 'r') as f:
    cv_results = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['mae', 'rmse', 'r2']
metric_names = ['MAE (m/s)', 'RMSE (m/s)', 'R²']
horizons_cv = ['30s', '90s', '150s']

for ax, metric, metric_name in zip(axes, metrics, metric_names):
    means = []
    stds = []
    
    for horizon in horizons_cv:
        values = cv_results['raw_results'][f'{metric}_{horizon}']
        means.append(np.mean(values))
        stds.append(np.std(values))
    
    x_pos = np.arange(len(horizons_cv))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
                   color=colors, edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(horizons_cv)
    ax.set_xlabel('Prediction Horizon', fontweight='bold')
    ax.set_ylabel(metric_name, fontweight='bold')
    ax.set_title(f'{metric_name} across Horizons\n(5-Fold CV)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure6_cross_validation.png', bbox_inches='tight', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/figure6_cross_validation.pdf', bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/figure6_cross_validation.png")
plt.close()

# ================= SUMMARY =================
print("\n" + "="*80)
print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*80)
print(f"\nFigures saved in: {OUTPUT_DIR}/")
print("\nGenerated figures:")
print("  1. figure1_prediction_vs_truth - Time series predictions")
print("  2. figure2_error_distribution - Error histograms")
print("  3. figure3_mae_vs_density - Traffic density analysis")
print("  4. figure4_attention_weights - Attention mechanism visualization")
print("  5. figure5_scatter_pred_vs_true - Scatter plots with R²")
print("  6. figure6_cross_validation - CV results bar charts")
print("\nFormats: PNG (300 DPI) and PDF (vector)")
print("="*80)