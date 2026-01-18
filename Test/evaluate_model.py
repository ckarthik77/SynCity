#!/usr/bin/env python3
"""
evaluate_model.py
Evaluate trained Multi-Horizon Attention-LSTM model
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

print("="*80)
print("🧪 SynCity Model Evaluation")
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
print(f"✅ Model loaded from {MODEL_PATH}")

with open(SCALER_PATH, 'rb') as f:
    scaler_dict = pickle.load(f)

X_scaler = scaler_dict['X_scaler']
y_scaler_30s = scaler_dict['y_scaler_30s']
y_scaler_90s = scaler_dict['y_scaler_90s']
y_scaler_150s = scaler_dict['y_scaler_150s']
print(f"✅ Scalers loaded from {SCALER_PATH}")

# ================= PREPARE TEST DATA =================
def create_sequences(df, seq_length=30):
    """Create sequences from raw data"""
    sequences = []
    targets_30s = []
    targets_90s = []
    targets_150s = []
    current_speeds = []
    
    vehicle_groups = df.groupby('veh_id')
    
    for veh_id, veh_data in vehicle_groups:
        veh_data = veh_data.sort_values('time').reset_index(drop=True)
        
        if len(veh_data) < seq_length + HORIZONS['150s']:
            continue
        
        features = veh_data[['speed', 'accel', 'front_vehicle_dist', 
                             'front_vehicle_speed', 'lane_density', 'avg_lane_speed']].values
        
        speeds = veh_data['speed'].values
        
        for i in range(len(features) - seq_length - HORIZONS['150s']):
            seq = features[i:i+seq_length]
            
            current_speed = speeds[i + seq_length - 1]
            
            # Target DELTAS
            target_30s = speeds[i + seq_length + HORIZONS['30s']] - current_speed
            target_90s = speeds[i + seq_length + HORIZONS['90s']] - current_speed
            target_150s = speeds[i + seq_length + HORIZONS['150s']] - current_speed
            
            sequences.append(seq)
            targets_30s.append(target_30s)
            targets_90s.append(target_90s)
            targets_150s.append(target_150s)
            current_speeds.append(current_speed)
    
    return (np.array(sequences), 
            np.array(targets_30s), 
            np.array(targets_90s), 
            np.array(targets_150s),
            np.array(current_speeds))

print("\n📂 Loading test data...")
df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(df)} rows")

print("\n🔧 Creating test sequences...")
X, y_30s_delta, y_90s_delta, y_150s_delta, current_speeds = create_sequences(df, SEQUENCE_LENGTH)
print(f"✅ Created {len(X)} test sequences")

# Normalize inputs
X_reshaped = X.reshape(-1, X.shape[-1])
X_normalized_reshaped = X_scaler.transform(X_reshaped)
X_normalized = X_normalized_reshaped.reshape(X.shape)

# Normalize target deltas
y_30s_norm = y_scaler_30s.transform(y_30s_delta.reshape(-1, 1)).flatten()
y_90s_norm = y_scaler_90s.transform(y_90s_delta.reshape(-1, 1)).flatten()
y_150s_norm = y_scaler_150s.transform(y_150s_delta.reshape(-1, 1)).flatten()

print("✅ Data normalized")

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

# ================= CALCULATE METRICS =================
print("\n" + "="*80)
print("📊 EVALUATION RESULTS")
print("="*80)

def calculate_metrics(y_true, y_pred, horizon_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    
    print(f"\n{horizon_name} Horizon:")
    print(f"  MAE:  {mae:.4f} m/s")
    print(f"  RMSE: {rmse:.4f} m/s")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# Calculate for all horizons
results_30s = calculate_metrics(y_30s_absolute, pred_30s_absolute, "30s")
results_90s = calculate_metrics(y_90s_absolute, pred_90s_absolute, "90s")
results_150s = calculate_metrics(y_150s_absolute, pred_150s_absolute, "150s")

# ================= COMPARISON WITH PAPER =================
print("\n" + "="*80)
print("📝 COMPARISON WITH PAPER")
print("="*80)
print("\nPaper Results (Table 1):")
print("  30s:  MAE=0.0454, RMSE=0.0699, R²=0.8889")
print("  90s:  MAE=0.0940, RMSE=0.1884, R²=0.9006")
print("  150s: MAE=0.1686, RMSE=0.3416, R²=0.8749")

print("\nYour Results:")
print(f"  30s:  MAE={results_30s['MAE']:.4f}, RMSE={results_30s['RMSE']:.4f}, R²={results_30s['R2']:.4f}")
print(f"  90s:  MAE={results_90s['MAE']:.4f}, RMSE={results_90s['RMSE']:.4f}, R²={results_90s['R2']:.4f}")
print(f"  150s: MAE={results_150s['MAE']:.4f}, RMSE={results_150s['RMSE']:.4f}, R²={results_150s['R2']:.4f}")

print("\n" + "="*80)

# ================= SAMPLE PREDICTIONS =================
print("\n📋 SAMPLE PREDICTIONS (First 10)")
print("="*80)
print(f"{'#':<5} {'Current':<10} {'Δ Pred':<10} {'Δ True':<10} {'Pred':<10} {'True':<10} {'Error':<10}")
print("-" * 80)
for i in range(10):
    print(f"{i:<5} {current_speeds[i]:<10.2f} {pred_30s_delta[i]:<10.4f} {y_30s_delta[i]:<10.4f} "
          f"{pred_30s_absolute[i]:<10.2f} {y_30s_absolute[i]:<10.2f} "
          f"{abs(pred_30s_absolute[i]-y_30s_absolute[i]):<10.4f}")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)

# ================= VERDICT =================
if results_30s['MAE'] < 1.0 and results_30s['R2'] > 0.7:
    print("\n🎉 SUCCESS! Model is performing well!")
    print("   MAE < 1.0 m/s and R² > 0.7 indicate good prediction quality.")
else:
    print(f"\n⚠️  Model performance differs from paper:")
    print(f"   MAE ratio: {results_30s['MAE']/0.0454:.1f}x")
    print(f"   This may be due to:")
    print(f"   - Different dataset characteristics")
    print(f"   - Need for more training epochs")
    print(f"   - Hyperparameter tuning required")
# Quick test - add this to your evaluate_model.py at the end:

print("\n" + "="*80)
print("🔬 TESTING ALTERNATIVE MAE CALCULATIONS")
print("="*80)

# Test 1: MAE on NORMALIZED scale (what model was trained on)
mae_30s_normalized = np.mean(np.abs(pred_30s_norm - y_30s_norm))
mae_90s_normalized = np.mean(np.abs(pred_90s_norm - y_90s_norm))
mae_150s_normalized = np.mean(np.abs(pred_150s_norm - y_150s_norm))

print(f"\nMAE on Normalized Scale (unitless):")
print(f"  30s:  {mae_30s_normalized:.4f}")
print(f"  90s:  {mae_90s_normalized:.4f}")
print(f"  150s: {mae_150s_normalized:.4f}")

# Test 2: MAE on DELTA predictions (m/s change)
mae_30s_delta = np.mean(np.abs(pred_30s_delta - y_30s_delta))
mae_90s_delta = np.mean(np.abs(pred_90s_delta - y_90s_delta))
mae_150s_delta = np.mean(np.abs(pred_150s_delta - y_150s_delta))

print(f"\nMAE on Delta Predictions (m/s change):")
print(f"  30s:  {mae_30s_delta:.4f} m/s")
print(f"  90s:  {mae_90s_delta:.4f} m/s")
print(f"  150s: {mae_150s_delta:.4f} m/s")

# Test 3: Scaled MAE
print(f"\nMAE scaled by std of targets:")
print(f"  30s:  {mae_30s_delta / y_30s_delta.std():.4f}")
print(f"  90s:  {mae_90s_delta / y_90s_delta.std():.4f}")
print(f"  150s: {mae_150s_delta / y_150s_delta.std():.4f}")

print(f"\nComparison:")
if mae_30s_normalized < 0.25:
    print(f"✅ Normalized MAE ({mae_30s_normalized:.4f}) matches expected range!")
    print(f"   Paper likely uses normalized scale, not m/s")