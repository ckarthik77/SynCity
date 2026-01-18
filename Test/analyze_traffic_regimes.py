#!/usr/bin/env python3
"""
analyze_traffic_regimes.py
Analyzes model performance across different traffic density regimes
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sys
import pickle

# Add Model folder to path
sys.path.append(r"E:\Desktop\Jobs\projects\Syncity\Model")
from model_attention_lstm import MultiHorizonAttentionLSTM

# ================= CONFIG =================
TELEMETRY_FILE = r"E:\Desktop\Jobs\projects\Syncity\Test\Olddataset_2.csv"
MODEL_PATH = r"E:\Desktop\Jobs\projects\Syncity\Model\multihorizon_lstm.pth"
SCALER_PATH = r"E:\Desktop\Jobs\projects\Syncity\Model\multihorizon_scaler.pkl"
SEQUENCE_LENGTH = 30
OUTPUT_CSV = "regime_analysis.csv"

# ================= LOAD MODEL =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model architecture
model = MultiHorizonAttentionLSTM(
    input_size=6,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")

# ================= LOAD SCALER =================
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print(f"✅ Loaded StandardScaler from training")
print(f"   Feature mean shape: {scaler.mean_.shape}")
print(f"   Feature std shape: {scaler.scale_.shape}\n")

# ================= LOAD & PREPROCESS DATA =================
df = pd.read_csv(TELEMETRY_FILE)
print(f"✅ Loaded {len(df)} rows from dataset")

# Calculate vehicles per km on each lane
EDGE_LENGTH_KM = 0.07001
df['density_veh_per_km'] = df['lane_density'] / EDGE_LENGTH_KM

# Classify traffic regimes
df['regime'] = pd.cut(
    df['density_veh_per_km'], 
    bins=[0, 15, 40, float('inf')],
    labels=['Free-flow', 'Moderate', 'Congested']
)

print("\nTraffic Regime Distribution:")
print(df['regime'].value_counts())
print()

# ================= PREPARE SEQUENCES =================
def create_sequences(data, seq_length=30):
    """Create sliding window sequences"""
    sequences = []
    targets_30s = []
    targets_90s = []
    targets_150s = []
    current_speeds = []
    regimes = []
    
    vehicle_groups = data.groupby('veh_id')
    
    for veh_id, veh_data in vehicle_groups:
        veh_data = veh_data.sort_values('time')
        
        if len(veh_data) < seq_length + 300:
            continue
        
        features = veh_data[['speed', 'accel', 'front_vehicle_dist', 
                             'front_vehicle_speed', 'lane_density', 'avg_lane_speed']].values
        
        speeds = veh_data['speed'].values
        density_regime = veh_data['regime'].values
        
        for i in range(len(features) - seq_length - 300):
            seq = features[i:i+seq_length]
            
            current_speed = speeds[i + seq_length - 1]
            
            # Target deltas
            target_30s = speeds[i + seq_length + 60] - current_speed
            target_90s = speeds[i + seq_length + 180] - current_speed
            target_150s = speeds[i + seq_length + 300] - current_speed
            
            regime = density_regime[i + seq_length - 1]
            
            sequences.append(seq)
            targets_30s.append(target_30s)
            targets_90s.append(target_90s)
            targets_150s.append(target_150s)
            current_speeds.append(current_speed)
            regimes.append(regime)
    
    return (np.array(sequences), 
            np.array(targets_30s), 
            np.array(targets_90s), 
            np.array(targets_150s),
            np.array(current_speeds),
            np.array(regimes))

print("Creating sequences...")
X, y_30s_delta, y_90s_delta, y_150s_delta, current_speeds, regimes = create_sequences(df, SEQUENCE_LENGTH)
print(f"✅ Created {len(X)} sequences\n")

# ================= NORMALIZE USING TRAINING SCALER =================
# Reshape for StandardScaler (expects 2D: [n_samples, n_features])
X_reshaped = X.reshape(-1, X.shape[-1])
X_normalized_reshaped = scaler.transform(X_reshaped)
X_normalized = X_normalized_reshaped.reshape(X.shape)

print("✅ Normalized features using training StandardScaler\n")

# ================= PREDICT (WITH TARGET DENORMALIZATION) =================
print("Running predictions...")
X_tensor = torch.FloatTensor(X_normalized).to(device)

batch_size = 64
all_preds_30s = []
all_preds_90s = []
all_preds_150s = []

with torch.no_grad():
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i+batch_size]
        pred_1, pred_3, pred_5, attn = model(batch)
        
        all_preds_30s.append(pred_1.cpu().numpy())
        all_preds_90s.append(pred_3.cpu().numpy())
        all_preds_150s.append(pred_5.cpu().numpy())

# Predictions are NORMALIZED deltas
pred_30s_delta_norm = np.concatenate(all_preds_30s).flatten()
pred_90s_delta_norm = np.concatenate(all_preds_90s).flatten()
pred_150s_delta_norm = np.concatenate(all_preds_150s).flatten()

# ⚠️ CRITICAL: Calculate target normalization from TRUE delta distribution
print("\n📊 Calculating target denormalization parameters...")

# Use the true deltas to estimate the normalization used during training
y_delta_mean_30s = y_30s_delta.mean()
y_delta_std_30s = y_30s_delta.std()
y_delta_mean_90s = y_90s_delta.mean()
y_delta_std_90s = y_90s_delta.std()
y_delta_mean_150s = y_150s_delta.mean()
y_delta_std_150s = y_150s_delta.std()

print(f"Target delta statistics from test data:")
print(f"  30s:  mean={y_delta_mean_30s:.4f}, std={y_delta_std_30s:.4f}")
print(f"  90s:  mean={y_delta_mean_90s:.4f}, std={y_delta_std_90s:.4f}")
print(f"  150s: mean={y_delta_mean_150s:.4f}, std={y_delta_std_150s:.4f}")

# Denormalize predictions: denorm = norm * std + mean
pred_30s_delta = pred_30s_delta_norm * y_delta_std_30s + y_delta_mean_30s
pred_90s_delta = pred_90s_delta_norm * y_delta_std_90s + y_delta_mean_90s
pred_150s_delta = pred_150s_delta_norm * y_delta_std_150s + y_delta_mean_150s

# Convert to absolute speeds
pred_30s_absolute = current_speeds + pred_30s_delta
pred_90s_absolute = current_speeds + pred_90s_delta
pred_150s_absolute = current_speeds + pred_150s_delta

y_30s_absolute = current_speeds + y_30s_delta
y_90s_absolute = current_speeds + y_90s_delta
y_150s_absolute = current_speeds + y_150s_delta

# Clip predictions to valid speed range [0, 30] m/s
pred_30s_absolute = np.clip(pred_30s_absolute, 0, 30)
pred_90s_absolute = np.clip(pred_90s_absolute, 0, 30)
pred_150s_absolute = np.clip(pred_150s_absolute, 0, 30)

print(f"\n✅ Predictions completed and denormalized: {len(pred_30s_absolute)} samples\n")

# ================= DIAGNOSTIC CHECK - MULTIPLE METRICS =================
print("="*80)
print("🔍 COMPREHENSIVE DIAGNOSTIC - TESTING ALL POSSIBLE METRICS")
print("="*80)

print(f"\n1️⃣ NORMALIZED DELTA PREDICTIONS (Model raw output):")
print(f"   Range: [{pred_30s_delta_norm.min():.4f}, {pred_30s_delta_norm.max():.4f}]")
print(f"   Mean: {pred_30s_delta_norm.mean():.4f}, Std: {pred_30s_delta_norm.std():.4f}")

print(f"\n2️⃣ DENORMALIZED DELTA PREDICTIONS:")
print(f"   30s:  [{pred_30s_delta.min():.4f}, {pred_30s_delta.max():.4f}]")
print(f"   90s:  [{pred_90s_delta.min():.4f}, {pred_90s_delta.max():.4f}]")
print(f"   150s: [{pred_150s_delta.min():.4f}, {pred_150s_delta.max():.4f}]")

print(f"\n3️⃣ TRUE DELTA RANGE:")
print(f"   30s:  [{y_30s_delta.min():.4f}, {y_30s_delta.max():.4f}]")
print(f"   90s:  [{y_90s_delta.min():.4f}, {y_90s_delta.max():.4f}]")
print(f"   150s: [{y_150s_delta.min():.4f}, {y_150s_delta.max():.4f}]")

print(f"\n4️⃣ ABSOLUTE SPEED PREDICTIONS:")
print(f"   Predicted: [{pred_30s_absolute.min():.4f}, {pred_30s_absolute.max():.4f}]")
print(f"   True:      [{y_30s_absolute.min():.4f}, {y_30s_absolute.max():.4f}]")

# ================= TEST ALL POSSIBLE MAE CALCULATIONS =================
print("\n" + "="*80)
print("🧪 TESTING ALL POSSIBLE MAE FORMULATIONS")
print("="*80)

# Normalize true deltas for comparison
y_30s_delta_norm = (y_30s_delta - y_delta_mean_30s) / y_delta_std_30s
y_90s_delta_norm = (y_90s_delta - y_delta_mean_90s) / y_delta_std_90s
y_150s_delta_norm = (y_150s_delta - y_delta_mean_150s) / y_delta_std_150s

print("\n📊 MAE Calculation 1: NORMALIZED DELTAS (model output scale)")
mae_30s_norm = np.mean(np.abs(pred_30s_delta_norm - y_30s_delta_norm))
mae_90s_norm = np.mean(np.abs(pred_90s_delta_norm - y_90s_delta_norm))
mae_150s_norm = np.mean(np.abs(pred_150s_delta_norm - y_150s_delta_norm))
print(f"   30s:  MAE = {mae_30s_norm:.4f} (normalized units)")
print(f"   90s:  MAE = {mae_90s_norm:.4f} (normalized units)")
print(f"   150s: MAE = {mae_150s_norm:.4f} (normalized units)")

print("\n📊 MAE Calculation 2: DENORMALIZED DELTAS (m/s change)")
mae_30s_delta = np.mean(np.abs(pred_30s_delta - y_30s_delta))
mae_90s_delta = np.mean(np.abs(pred_90s_delta - y_90s_delta))
mae_150s_delta = np.mean(np.abs(pred_150s_delta - y_150s_delta))
print(f"   30s:  MAE = {mae_30s_delta:.4f} m/s")
print(f"   90s:  MAE = {mae_90s_delta:.4f} m/s")
print(f"   150s: MAE = {mae_150s_delta:.4f} m/s")

print("\n📊 MAE Calculation 3: ABSOLUTE SPEEDS (m/s)")
mae_30s_abs = np.mean(np.abs(pred_30s_absolute - y_30s_absolute))
mae_90s_abs = np.mean(np.abs(pred_90s_absolute - y_90s_absolute))
mae_150s_abs = np.mean(np.abs(pred_150s_absolute - y_150s_absolute))
print(f"   30s:  MAE = {mae_30s_abs:.4f} m/s")
print(f"   90s:  MAE = {mae_90s_abs:.4f} m/s")
print(f"   150s: MAE = {mae_150s_abs:.4f} m/s")

print("\n📊 MAE Calculation 4: NORMALIZED ABSOLUTE SPEEDS")
y_30s_abs_norm = (y_30s_absolute - y_30s_absolute.mean()) / y_30s_absolute.std()
pred_30s_abs_norm = (pred_30s_absolute - pred_30s_absolute.mean()) / pred_30s_absolute.std()
mae_30s_abs_norm = np.mean(np.abs(pred_30s_abs_norm - y_30s_abs_norm))
print(f"   30s:  MAE = {mae_30s_abs_norm:.4f} (normalized units)")

print("\n" + "="*80)
print("📝 COMPARISON WITH PAPER")
print("="*80)
print("\nPaper reported (30s horizon): MAE = 0.0454")
print("\nYour results:")
print(f"  If paper used NORMALIZED DELTAS:    MAE = {mae_30s_norm:.4f}  ← {'✅ MATCH!' if abs(mae_30s_norm - 0.0454) < 0.01 else '❌ No match'}")
print(f"  If paper used DENORMALIZED DELTAS:  MAE = {mae_30s_delta:.4f}  ← {'✅ MATCH!' if abs(mae_30s_delta - 0.0454) < 0.01 else '❌ No match'}")
print(f"  If paper used ABSOLUTE SPEEDS:      MAE = {mae_30s_abs:.4f}  ← {'✅ MATCH!' if abs(mae_30s_abs - 0.0454) < 0.01 else '❌ No match'}")
print(f"  If paper used NORMALIZED ABS SPEEDS: MAE = {mae_30s_abs_norm:.4f}  ← {'✅ MATCH!' if abs(mae_30s_abs_norm - 0.0454) < 0.01 else '❌ No match'}")

# ================= DETAILED SAMPLE COMPARISON =================
print("\n" + "="*80)
print("📋 FIRST 10 SAMPLE PREDICTIONS (30s horizon)")
print("="*80)
print(f"{'#':<5} {'Current':<10} {'Δ Pred':<10} {'Δ True':<10} {'Pred':<10} {'True':<10} {'Error':<10}")
print("-" * 70)
for i in range(10):
    print(f"{i:<5} {current_speeds[i]:<10.2f} {pred_30s_delta[i]:<10.4f} {y_30s_delta[i]:<10.4f} {pred_30s_absolute[i]:<10.2f} {y_30s_absolute[i]:<10.2f} {abs(pred_30s_absolute[i]-y_30s_absolute[i]):<10.4f}")

# ================= TRAFFIC REGIME ANALYSIS =================
print("\n" + "="*80)
print("PERFORMANCE BY TRAFFIC REGIME (Using Denormalized Deltas)")
print("="*80)

results = []

for regime_name in ['Free-flow', 'Moderate', 'Congested']:
    mask = (regimes == regime_name)
    
    if mask.sum() == 0:
        print(f"⚠️  No samples for {regime_name} regime\n")
        continue
    
    # Use DELTA MAE (most likely what paper reports)
    mae_30s = np.mean(np.abs(pred_30s_delta[mask] - y_30s_delta[mask]))
    rmse_30s = np.sqrt(np.mean((pred_30s_delta[mask] - y_30s_delta[mask])**2))
    
    mae_90s = np.mean(np.abs(pred_90s_delta[mask] - y_90s_delta[mask]))
    rmse_90s = np.sqrt(np.mean((pred_90s_delta[mask] - y_90s_delta[mask])**2))
    
    mae_150s = np.mean(np.abs(pred_150s_delta[mask] - y_150s_delta[mask]))
    rmse_150s = np.sqrt(np.mean((pred_150s_delta[mask] - y_150s_delta[mask])**2))
    
    # R² on deltas
    from sklearn.metrics import r2_score
    r2_30s = r2_score(y_30s_delta[mask], pred_30s_delta[mask])
    r2_90s = r2_score(y_90s_delta[mask], pred_90s_delta[mask])
    r2_150s = r2_score(y_150s_delta[mask], pred_150s_delta[mask])
    
    results.append({
        'Regime': regime_name,
        'Sample_Count': int(mask.sum()),
        'MAE_30s': mae_30s,
        'RMSE_30s': rmse_30s,
        'R2_30s': r2_30s,
        'MAE_90s': mae_90s,
        'RMSE_90s': rmse_90s,
        'R2_90s': r2_90s,
        'MAE_150s': mae_150s,
        'RMSE_150s': rmse_150s,
        'R2_150s': r2_150s
    })
    
    print(f"\n{regime_name} Traffic:")
    print(f"  Samples: {mask.sum()}")
    print(f"  30s:  MAE={mae_30s:.4f} m/s, RMSE={rmse_30s:.4f} m/s, R²={r2_30s:.4f}")
    print(f"  90s:  MAE={mae_90s:.4f} m/s, RMSE={rmse_90s:.4f} m/s, R²={r2_90s:.4f}")
    print(f"  150s: MAE={mae_150s:.4f} m/s, RMSE={rmse_150s:.4f} m/s, R²={r2_150s:.4f}")

print("\n" + "="*80)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Results saved to {OUTPUT_CSV}")
print("\nResults Table:")
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("🎯 FINAL VERDICT")
print("="*80)
print(f"\nMost likely metric used in paper: DENORMALIZED DELTA MAE")
print(f"Your result: {results[0]['MAE_30s']:.4f} m/s")
print(f"Paper result: 0.0454 m/s")
print(f"Ratio: {results[0]['MAE_30s']/0.0454:.1f}x")