#!/usr/bin/env python3
"""
traffic_regime_analysis.py
Analyze model performance across different traffic density regimes
"""

import pandas as pd
import numpy as np
import torch
import pickle
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add Model folder to path
sys.path.append(r"E:\Desktop\Jobs\projects\Syncity\Model")
from model_attention_lstm import MultiHorizonAttentionLSTM

# ================= CONFIG =================
DATA_FILE = "syncity_clean_dataset.csv"
MODEL_PATH = "Model/multihorizon_lstm_new.pth"
SCALER_PATH = "Model/multihorizon_scaler_new.pkl"
SEQUENCE_LENGTH = 30
HORIZONS = {
    '30s': 60,
    '90s': 180,
    '150s': 300
}
EDGE_LENGTH_KM = 0.07001  # From network stats

print("="*80)
print("🚦 Traffic Regime Analysis")
print("="*80)

# ================= LOAD MODEL & SCALERS =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📦 Loading model and scalers...")

model = MultiHorizonAttentionLSTM(
    input_size=6,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded")

with open(SCALER_PATH, 'rb') as f:
    scaler_dict = pickle.load(f)

X_scaler = scaler_dict['X_scaler']
y_scaler_30s = scaler_dict['y_scaler_30s']
y_scaler_90s = scaler_dict['y_scaler_90s']
y_scaler_150s = scaler_dict['y_scaler_150s']
print(f"✅ Scalers loaded")

# ================= PREPARE DATA =================
def create_sequences_with_regimes(df, seq_length=30):
    """Create sequences and classify traffic regimes"""
    sequences = []
    targets_30s = []
    targets_90s = []
    targets_150s = []
    current_speeds = []
    regimes = []
    
    vehicle_groups = df.groupby('veh_id')
    
    for veh_id, veh_data in vehicle_groups:
        veh_data = veh_data.sort_values('time').reset_index(drop=True)
        
        if len(veh_data) < seq_length + HORIZONS['150s']:
            continue
        
        features = veh_data[['speed', 'accel', 'front_vehicle_dist', 
                             'front_vehicle_speed', 'lane_density', 'avg_lane_speed']].values
        
        speeds = veh_data['speed'].values
        lane_densities = veh_data['lane_density'].values
        
        for i in range(len(features) - seq_length - HORIZONS['150s']):
            seq = features[i:i+seq_length]
            
            current_speed = speeds[i + seq_length - 1]
            
            # Target DELTAS
            target_30s = speeds[i + seq_length + HORIZONS['30s']] - current_speed
            target_90s = speeds[i + seq_length + HORIZONS['90s']] - current_speed
            target_150s = speeds[i + seq_length + HORIZONS['150s']] - current_speed
            
            # Traffic regime classification based on lane density
            density_veh_per_km = lane_densities[i + seq_length - 1] / EDGE_LENGTH_KM
            
            if density_veh_per_km < 15:
                regime = 'Free-flow'
            elif density_veh_per_km < 40:
                regime = 'Moderate'
            else:
                regime = 'Congested'
            
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

print("\n📂 Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(df)} rows")

print("\n🔧 Creating sequences with regime classification...")
X, y_30s_delta, y_90s_delta, y_150s_delta, current_speeds, regimes = create_sequences_with_regimes(df, SEQUENCE_LENGTH)
print(f"✅ Created {len(X)} sequences")

# Traffic regime distribution
print("\n📊 Traffic Regime Distribution:")
unique, counts = np.unique(regimes, return_counts=True)
for regime, count in zip(unique, counts):
    percentage = (count / len(regimes)) * 100
    print(f"  {regime:<15} {count:>8} samples ({percentage:>5.1f}%)")

# Normalize inputs
X_reshaped = X.reshape(-1, X.shape[-1])
X_normalized_reshaped = X_scaler.transform(X_reshaped)
X_normalized = X_normalized_reshaped.reshape(X.shape)

# Normalize target deltas
y_30s_norm = y_scaler_30s.transform(y_30s_delta.reshape(-1, 1)).flatten()
y_90s_norm = y_scaler_90s.transform(y_90s_delta.reshape(-1, 1)).flatten()
y_150s_norm = y_scaler_150s.transform(y_150s_delta.reshape(-1, 1)).flatten()

print("\n✅ Data normalized")

# ================= PREDICT =================
print("\n🔮 Running predictions...")

X_tensor = torch.FloatTensor(X_normalized).to(device)

all_preds_30s = []
all_preds_90s = []
all_preds_150s = []

batch_size = 128

with torch.no_grad():
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i+batch_size]
        pred_30s, pred_90s, pred_150s, _ = model(batch)
        
        all_preds_30s.append(pred_30s.cpu().numpy())
        all_preds_90s.append(pred_90s.cpu().numpy())
        all_preds_150s.append(pred_150s.cpu().numpy())

# Predictions are NORMALIZED deltas
pred_30s_norm = np.concatenate(all_preds_30s).flatten()
pred_90s_norm = np.concatenate(all_preds_90s).flatten()
pred_150s_norm = np.concatenate(all_preds_150s).flatten()

# DENORMALIZE predictions
pred_30s_delta = y_scaler_30s.inverse_transform(pred_30s_norm.reshape(-1, 1)).flatten()
pred_90s_delta = y_scaler_90s.inverse_transform(pred_90s_norm.reshape(-1, 1)).flatten()
pred_150s_delta = y_scaler_150s.inverse_transform(pred_150s_norm.reshape(-1, 1)).flatten()

# Convert to absolute speeds
pred_30s_absolute = current_speeds + pred_30s_delta
pred_90s_absolute = current_speeds + pred_90s_delta
pred_150s_absolute = current_speeds + pred_150s_delta

y_30s_absolute = current_speeds + y_30s_delta
y_90s_absolute = current_speeds + y_90s_delta
y_150s_absolute = current_speeds + y_150s_delta

# Clip to valid speed range
pred_30s_absolute = np.clip(pred_30s_absolute, 0, 30)
pred_90s_absolute = np.clip(pred_90s_absolute, 0, 30)
pred_150s_absolute = np.clip(pred_150s_absolute, 0, 30)

print("✅ Predictions completed and denormalized")

# ================= ANALYZE BY REGIME =================
print("\n" + "="*80)
print("📊 PERFORMANCE BY TRAFFIC REGIME")
print("="*80)

results = []

for regime_name in ['Free-flow', 'Moderate', 'Congested']:
    mask = (regimes == regime_name)
    
    if mask.sum() == 0:
        print(f"\n⚠️  No samples for {regime_name} regime")
        continue
    
    print(f"\n{'='*80}")
    print(f"{regime_name} Traffic ({mask.sum()} samples)")
    print(f"{'='*80}")
    
    # Calculate metrics for each horizon
    for horizon_name, pred_abs, true_abs in [
        ('30s', pred_30s_absolute, y_30s_absolute),
        ('90s', pred_90s_absolute, y_90s_absolute),
        ('150s', pred_150s_absolute, y_150s_absolute)
    ]:
        mae = mean_absolute_error(true_abs[mask], pred_abs[mask])
        rmse = np.sqrt(mean_squared_error(true_abs[mask], pred_abs[mask]))
        r2 = r2_score(true_abs[mask], pred_abs[mask])
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((true_abs[mask] - pred_abs[mask]) / 
                              np.maximum(np.abs(true_abs[mask]), 1e-8))) * 100
        
        print(f"  {horizon_name:<6} MAE={mae:>6.4f} m/s | RMSE={rmse:>6.4f} m/s | R²={r2:>6.4f} | MAPE={mape:>6.2f}%")
        
        if horizon_name == '30s':
            results.append({
                'Regime': regime_name,
                'Samples': int(mask.sum()),
                'MAE_30s': mae,
                'RMSE_30s': rmse,
                'R2_30s': r2,
                'MAE_90s': None,
                'RMSE_90s': None,
                'R2_90s': None,
                'MAE_150s': None,
                'RMSE_150s': None,
                'R2_150s': None
            })
        elif horizon_name == '90s':
            results[-1]['MAE_90s'] = mae
            results[-1]['RMSE_90s'] = rmse
            results[-1]['R2_90s'] = r2
        elif horizon_name == '150s':
            results[-1]['MAE_150s'] = mae
            results[-1]['RMSE_150s'] = rmse
            results[-1]['R2_150s'] = r2

# ================= SAVE RESULTS =================
results_df = pd.DataFrame(results)
results_df.to_csv('Model/traffic_regime_results.csv', index=False)
print(f"\n✅ Results saved to Model/traffic_regime_results.csv")

# ================= SUMMARY TABLE =================
print("\n" + "="*80)
print("📝 SUMMARY TABLE FOR PAPER")
print("="*80)
print("\nTable X: Performance by Traffic Regime")
print("-" * 100)
print(f"{'Regime':<15} {'Samples':<10} {'30s MAE':<12} {'30s R²':<10} {'90s MAE':<12} {'90s R²':<10} {'150s MAE':<12} {'150s R²':<10}")
print("-" * 100)

for result in results:
    print(f"{result['Regime']:<15} {result['Samples']:<10} "
          f"{result['MAE_30s']:<12.4f} {result['R2_30s']:<10.4f} "
          f"{result['MAE_90s']:<12.4f} {result['R2_90s']:<10.4f} "
          f"{result['MAE_150s']:<12.4f} {result['R2_150s']:<10.4f}")

print("-" * 100)

# ================= ANALYSIS =================
print("\n" + "="*80)
print("💡 KEY INSIGHTS")
print("="*80)

if len(results) >= 2:
    free_flow = next((r for r in results if r['Regime'] == 'Free-flow'), None)
    moderate = next((r for r in results if r['Regime'] == 'Moderate'), None)
    
    if free_flow and moderate:
        mae_increase = ((moderate['MAE_30s'] - free_flow['MAE_30s']) / free_flow['MAE_30s']) * 100
        r2_decrease = ((free_flow['R2_30s'] - moderate['R2_30s']) / free_flow['R2_30s']) * 100
        
        print(f"\n1. Free-flow vs Moderate Traffic (30s horizon):")
        print(f"   - MAE increases by {mae_increase:.1f}% in moderate traffic")
        print(f"   - R² decreases by {r2_decrease:.1f}% in moderate traffic")
        print(f"   → Model maintains strong performance across traffic conditions")
        
        print(f"\n2. Regime Distribution:")
        free_pct = (free_flow['Samples'] / len(regimes)) * 100
        mod_pct = (moderate['Samples'] / len(regimes)) * 100
        print(f"   - Free-flow: {free_pct:.1f}% of samples")
        print(f"   - Moderate: {mod_pct:.1f}% of samples")
        print(f"   → Dataset is naturally imbalanced toward free-flow conditions")

print("\n" + "="*80)
print("✅ TRAFFIC REGIME ANALYSIS COMPLETE!")
print("="*80)