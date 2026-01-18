#!/usr/bin/env python3
"""
cross_validation.py
5-Fold Cross-Validation for SynCity Multi-Horizon Attention-LSTM
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from tqdm import tqdm
import sys

# Add Model folder to path
sys.path.append(r"E:\Desktop\Jobs\projects\Syncity\Model")
from model_attention_lstm import MultiHorizonAttentionLSTM

# ================= CONFIG =================
DATA_FILE = "syncity_clean_dataset.csv"
SEQUENCE_LENGTH = 30
HORIZONS = {
    '30s': 60,
    '90s': 180,
    '150s': 300
}

# Model hyperparameters
INPUT_SIZE = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
N_FOLDS = 5

print("="*80)
print("🔬 SynCity 5-Fold Cross-Validation")
print("="*80)
print(f"Configuration:")
print(f"  Folds:        {N_FOLDS}")
print(f"  Epochs/fold:  {EPOCHS}")
print(f"  Batch size:   {BATCH_SIZE}")
print(f"  Early stop:   {EARLY_STOP_PATIENCE} epochs patience")
print("="*80)

# ================= DATASET CLASS =================
class VehicleSequenceDataset(Dataset):
    def __init__(self, sequences, targets_30s, targets_90s, targets_150s):
        self.sequences = torch.FloatTensor(sequences)
        self.targets_30s = torch.FloatTensor(targets_30s).unsqueeze(1)
        self.targets_90s = torch.FloatTensor(targets_90s).unsqueeze(1)
        self.targets_150s = torch.FloatTensor(targets_150s).unsqueeze(1)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.targets_30s[idx],
            self.targets_90s[idx],
            self.targets_150s[idx]
        )

# ================= DATA PREPARATION =================
def create_sequences(df, seq_length=30):
    """Create sequences from raw data"""
    sequences = []
    targets_30s = []
    targets_90s = []
    targets_150s = []
    
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
            
            target_30s = speeds[i + seq_length + HORIZONS['30s']] - current_speed
            target_90s = speeds[i + seq_length + HORIZONS['90s']] - current_speed
            target_150s = speeds[i + seq_length + HORIZONS['150s']] - current_speed
            
            sequences.append(seq)
            targets_30s.append(target_30s)
            targets_90s.append(target_90s)
            targets_150s.append(target_150s)
    
    return (np.array(sequences), 
            np.array(targets_30s), 
            np.array(targets_90s), 
            np.array(targets_150s))

# ================= LOAD DATA =================
print("\n📂 Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(df)} rows, {df['veh_id'].nunique()} vehicles")

print("\n🔧 Creating sequences...")
X, y_30s, y_90s, y_150s = create_sequences(df, SEQUENCE_LENGTH)
print(f"✅ Created {len(X)} sequences")

# ================= CROSS-VALIDATION =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Using device: {device}")

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

cv_results = {
    'mae_30s': [],
    'rmse_30s': [],
    'r2_30s': [],
    'mae_90s': [],
    'rmse_90s': [],
    'r2_90s': [],
    'mae_150s': [],
    'rmse_150s': [],
    'r2_150s': []
}

print("\n" + "="*80)
print("🚀 STARTING CROSS-VALIDATION")
print("="*80)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y30_train, y30_val = y_30s[train_idx], y_30s[val_idx]
    y90_train, y90_val = y_90s[train_idx], y_90s[val_idx]
    y150_train, y150_val = y_150s[train_idx], y_150s[val_idx]
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    
    # Normalize features
    X_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_scaler = StandardScaler()
    X_train_norm_reshaped = X_scaler.fit_transform(X_reshaped)
    X_train_norm = X_train_norm_reshaped.reshape(X_train.shape)
    
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_val_norm_reshaped = X_scaler.transform(X_val_reshaped)
    X_val_norm = X_val_norm_reshaped.reshape(X_val.shape)
    
    # Normalize targets
    y_scaler_30s = StandardScaler()
    y_scaler_90s = StandardScaler()
    y_scaler_150s = StandardScaler()
    
    y30_train_norm = y_scaler_30s.fit_transform(y30_train.reshape(-1, 1)).flatten()
    y90_train_norm = y_scaler_90s.fit_transform(y90_train.reshape(-1, 1)).flatten()
    y150_train_norm = y_scaler_150s.fit_transform(y150_train.reshape(-1, 1)).flatten()
    
    y30_val_norm = y_scaler_30s.transform(y30_val.reshape(-1, 1)).flatten()
    y90_val_norm = y_scaler_90s.transform(y90_val.reshape(-1, 1)).flatten()
    y150_val_norm = y_scaler_150s.transform(y150_val.reshape(-1, 1)).flatten()
    
    # Create datasets
    train_dataset = VehicleSequenceDataset(X_train_norm, y30_train_norm, y90_train_norm, y150_train_norm)
    val_dataset = VehicleSequenceDataset(X_val_norm, y30_val_norm, y90_val_norm, y150_val_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = MultiHorizonAttentionLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        
        for X_batch, y30_batch, y90_batch, y150_batch in train_loader:
            X_batch = X_batch.to(device)
            y30_batch = y30_batch.to(device)
            y90_batch = y90_batch.to(device)
            y150_batch = y150_batch.to(device)
            
            optimizer.zero_grad()
            
            pred_30s, pred_90s, pred_150s, _ = model(X_batch)
            
            loss = (
                criterion(pred_30s, y30_batch) +
                criterion(pred_90s, y90_batch) +
                0.8 * criterion(pred_150s, y150_batch)
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y30_batch, y90_batch, y150_batch in val_loader:
                X_batch = X_batch.to(device)
                y30_batch = y30_batch.to(device)
                y90_batch = y90_batch.to(device)
                y150_batch = y150_batch.to(device)
                
                pred_30s, pred_90s, pred_150s, _ = model(X_batch)
                
                loss = (
                    criterion(pred_30s, y30_batch) +
                    criterion(pred_90s, y90_batch) +
                    0.8 * criterion(pred_150s, y150_batch)
                )
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation on validation set
    model.eval()
    all_preds_30s = []
    all_preds_90s = []
    all_preds_150s = []
    all_true_30s = []
    all_true_90s = []
    all_true_150s = []
    
    with torch.no_grad():
        for X_batch, y30_batch, y90_batch, y150_batch in val_loader:
            X_batch = X_batch.to(device)
            
            pred_30s, pred_90s, pred_150s, _ = model(X_batch)
            
            all_preds_30s.append(pred_30s.cpu().numpy())
            all_preds_90s.append(pred_90s.cpu().numpy())
            all_preds_150s.append(pred_150s.cpu().numpy())
            all_true_30s.append(y30_batch.cpu().numpy())
            all_true_90s.append(y90_batch.cpu().numpy())
            all_true_150s.append(y150_batch.cpu().numpy())
    
    # Concatenate predictions
    pred_30s_norm = np.concatenate(all_preds_30s).flatten()
    pred_90s_norm = np.concatenate(all_preds_90s).flatten()
    pred_150s_norm = np.concatenate(all_preds_150s).flatten()
    
    true_30s_norm = np.concatenate(all_true_30s).flatten()
    true_90s_norm = np.concatenate(all_true_90s).flatten()
    true_150s_norm = np.concatenate(all_true_150s).flatten()
    
    # Denormalize predictions
    pred_30s_delta = y_scaler_30s.inverse_transform(pred_30s_norm.reshape(-1, 1)).flatten()
    pred_90s_delta = y_scaler_90s.inverse_transform(pred_90s_norm.reshape(-1, 1)).flatten()
    pred_150s_delta = y_scaler_150s.inverse_transform(pred_150s_norm.reshape(-1, 1)).flatten()
    
    true_30s_delta = y30_val
    true_90s_delta = y90_val
    true_150s_delta = y150_val
    
    # Calculate metrics (on denormalized deltas in m/s)
    mae_30s = mean_absolute_error(true_30s_delta, pred_30s_delta)
    rmse_30s = np.sqrt(mean_squared_error(true_30s_delta, pred_30s_delta))
    r2_30s = r2_score(true_30s_delta, pred_30s_delta)
    
    mae_90s = mean_absolute_error(true_90s_delta, pred_90s_delta)
    rmse_90s = np.sqrt(mean_squared_error(true_90s_delta, pred_90s_delta))
    r2_90s = r2_score(true_90s_delta, pred_90s_delta)
    
    mae_150s = mean_absolute_error(true_150s_delta, pred_150s_delta)
    rmse_150s = np.sqrt(mean_squared_error(true_150s_delta, pred_150s_delta))
    r2_150s = r2_score(true_150s_delta, pred_150s_delta)
    
    # Store results
    cv_results['mae_30s'].append(mae_30s)
    cv_results['rmse_30s'].append(rmse_30s)
    cv_results['r2_30s'].append(r2_30s)
    cv_results['mae_90s'].append(mae_90s)
    cv_results['rmse_90s'].append(rmse_90s)
    cv_results['r2_90s'].append(r2_90s)
    cv_results['mae_150s'].append(mae_150s)
    cv_results['rmse_150s'].append(rmse_150s)
    cv_results['r2_150s'].append(r2_150s)
    
    print(f"\n  Fold {fold+1} Results:")
    print(f"    30s:  MAE={mae_30s:.4f}, RMSE={rmse_30s:.4f}, R²={r2_30s:.4f}")
    print(f"    90s:  MAE={mae_90s:.4f}, RMSE={rmse_90s:.4f}, R²={r2_90s:.4f}")
    print(f"    150s: MAE={mae_150s:.4f}, RMSE={rmse_150s:.4f}, R²={r2_150s:.4f}")

# ================= AGGREGATE RESULTS =================
print("\n" + "="*80)
print("📊 CROSS-VALIDATION SUMMARY (Mean ± Std)")
print("="*80)

def print_cv_stats(horizon, mae_list, rmse_list, r2_list):
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    rmse_mean = np.mean(rmse_list)
    rmse_std = np.std(rmse_list)
    r2_mean = np.mean(r2_list)
    r2_std = np.std(r2_list)
    
    print(f"\n{horizon} Horizon:")
    print(f"  MAE:  {mae_mean:.4f} ± {mae_std:.4f} m/s")
    print(f"  RMSE: {rmse_mean:.4f} ± {rmse_std:.4f} m/s")
    print(f"  R²:   {r2_mean:.4f} ± {r2_std:.4f}")
    
    return {
        'mae_mean': mae_mean, 'mae_std': mae_std,
        'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
        'r2_mean': r2_mean, 'r2_std': r2_std
    }

stats_30s = print_cv_stats("30s", cv_results['mae_30s'], cv_results['rmse_30s'], cv_results['r2_30s'])
stats_90s = print_cv_stats("90s", cv_results['mae_90s'], cv_results['rmse_90s'], cv_results['r2_90s'])
stats_150s = print_cv_stats("150s", cv_results['mae_150s'], cv_results['rmse_150s'], cv_results['r2_150s'])

# ================= SAVE RESULTS =================
results_summary = {
    '30s': stats_30s,
    '90s': stats_90s,
    '150s': stats_150s,
    'raw_results': cv_results
}

with open('Model/cv_results.json', 'w') as f:
    json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                   for kk, vv in v.items()} if isinstance(v, dict) else 
               [float(x) for x in v] if isinstance(v, list) else v
               for k, v in results_summary.items()}, f, indent=2)

print("\n✅ Results saved to Model/cv_results.json")

# ================= FOR PAPER =================
print("\n" + "="*80)
print("📝 FOR PAPER - TABLE FORMAT")
print("="*80)
print("\nTable X: Cross-Validation Results (Mean ± Std, n=5 folds)")
print("-" * 80)
print(f"{'Horizon':<10} {'MAE (m/s)':<20} {'RMSE (m/s)':<20} {'R²':<20}")
print("-" * 80)
print(f"{'30s':<10} {stats_30s['mae_mean']:.4f} ± {stats_30s['mae_std']:.4f}     "
      f"{stats_30s['rmse_mean']:.4f} ± {stats_30s['rmse_std']:.4f}     "
      f"{stats_30s['r2_mean']:.4f} ± {stats_30s['r2_std']:.4f}")
print(f"{'90s':<10} {stats_90s['mae_mean']:.4f} ± {stats_90s['mae_std']:.4f}     "
      f"{stats_90s['rmse_mean']:.4f} ± {stats_90s['rmse_std']:.4f}     "
      f"{stats_90s['r2_mean']:.4f} ± {stats_90s['r2_std']:.4f}")
print(f"{'150s':<10} {stats_150s['mae_mean']:.4f} ± {stats_150s['mae_std']:.4f}     "
      f"{stats_150s['rmse_mean']:.4f} ± {stats_150s['rmse_std']:.4f}     "
      f"{stats_150s['r2_mean']:.4f} ± {stats_150s['r2_std']:.4f}")
print("-" * 80)

print("\n" + "="*80)
print("✅ CROSS-VALIDATION COMPLETE!")
print("="*80)

"""   Paper likely uses normalized scale, not m/s
PS E:\Desktop\Jobs\projects\Syncity\Test> py cross_validation.py
================================================================================
🔬 SynCity 5-Fold Cross-Validation
================================================================================
Configuration:
  Folds:        5
  Epochs/fold:  50
  Batch size:   64
  Early stop:   10 epochs patience
================================================================================

📂 Loading data...
✅ Loaded 169887 rows, 402 vehicles

🔧 Creating sequences...
✅ Created 37227 sequences

🖥️  Using device: cpu

================================================================================
🚀 STARTING CROSS-VALIDATION
================================================================================

================================================================================
FOLD 1/5
================================================================================
  Train: 29781 samples
  Val:   7446 samples
  Epoch 10/50 | Train: 1.2224 | Val: 1.2164
  Epoch 20/50 | Train: 0.7656 | Val: 0.7810
  Epoch 30/50 | Train: 0.4811 | Val: 0.4791
  Epoch 40/50 | Train: 0.3319 | Val: 0.3313
  Epoch 50/50 | Train: 0.2476 | Val: 0.2540

  Fold 1 Results:
    30s:  MAE=1.6457, RMSE=2.3010, R²=0.9121
    90s:  MAE=1.7041, RMSE=2.4672, R²=0.9146
    150s: MAE=1.8515, RMSE=2.5713, R²=0.8991

================================================================================
FOLD 2/5
================================================================================
  Train: 29781 samples
  Val:   7446 samples
  Epoch 10/50 | Train: 1.2118 | Val: 1.1896
  Epoch 20/50 | Train: 0.7584 | Val: 0.7520
  Epoch 30/50 | Train: 0.4717 | Val: 0.4630
  Epoch 40/50 | Train: 0.3285 | Val: 0.3363
  Epoch 50/50 | Train: 0.2596 | Val: 0.2592

  Fold 2 Results:
    30s:  MAE=1.5844, RMSE=2.2316, R²=0.9178
    90s:  MAE=1.7906, RMSE=2.5163, R²=0.9114
    150s: MAE=1.9028, RMSE=2.6479, R²=0.8929

================================================================================
FOLD 3/5
================================================================================
  Train: 29782 samples
  Val:   7445 samples
  Epoch 10/50 | Train: 1.2300 | Val: 1.2219
  Epoch 20/50 | Train: 0.7982 | Val: 0.8179
  Epoch 30/50 | Train: 0.5148 | Val: 0.4980
  Epoch 40/50 | Train: 0.3520 | Val: 0.3357
  Epoch 50/50 | Train: 0.2591 | Val: 0.2305

  Fold 3 Results:
    30s:  MAE=1.5501, RMSE=2.1367, R²=0.9235
    90s:  MAE=1.7017, RMSE=2.3363, R²=0.9229
    150s: MAE=1.8474, RMSE=2.5100, R²=0.9065

================================================================================
FOLD 4/5
================================================================================
  Train: 29782 samples
  Val:   7445 samples
  Epoch 10/50 | Train: 1.2191 | Val: 1.1750
  Epoch 20/50 | Train: 0.7490 | Val: 0.7205
  Epoch 30/50 | Train: 0.4658 | Val: 0.4617
  Epoch 40/50 | Train: 0.3148 | Val: 0.2918
  Epoch 50/50 | Train: 0.2485 | Val: 0.2315

  Fold 4 Results:
    30s:  MAE=1.6210, RMSE=2.2246, R²=0.9197
    90s:  MAE=1.6797, RMSE=2.3099, R²=0.9245
    150s: MAE=1.7970, RMSE=2.4657, R²=0.9060

================================================================================
FOLD 5/5
================================================================================
  Train: 29782 samples
  Val:   7445 samples
  Epoch 10/50 | Train: 1.2461 | Val: 1.2273
  Epoch 20/50 | Train: 0.7640 | Val: 0.7763
  Epoch 30/50 | Train: 0.4537 | Val: 0.4597
  Epoch 40/50 | Train: 0.3007 | Val: 0.2956
  Epoch 50/50 | Train: 0.2143 | Val: 0.2030

  Fold 5 Results:
    30s:  MAE=1.5507, RMSE=2.1026, R²=0.9259
    90s:  MAE=1.6055, RMSE=2.1865, R²=0.9323
    150s: MAE=1.6967, RMSE=2.2701, R²=0.9205

================================================================================
📊 CROSS-VALIDATION SUMMARY (Mean ± Std)
================================================================================

30s Horizon:
  MAE:  1.5904 ± 0.0380 m/s
  RMSE: 2.1993 ± 0.0711 m/s
  R²:   0.9198 ± 0.0048

90s Horizon:
  MAE:  1.6963 ± 0.0592 m/s
  RMSE: 2.3633 ± 0.1175 m/s
  R²:   0.9211 ± 0.0074

150s Horizon:
  MAE:  1.8191 ± 0.0698 m/s
  RMSE: 2.4930 ± 0.1272 m/s
  R²:   0.9050 ± 0.0092

✅ Results saved to Model/cv_results.json

================================================================================
📝 FOR PAPER - TABLE FORMAT
================================================================================

Table X: Cross-Validation Results (Mean ± Std, n=5 folds)
--------------------------------------------------------------------------------
Horizon    MAE (m/s)            RMSE (m/s)           R²
--------------------------------------------------------------------------------
30s        1.5904 ± 0.0380     2.1993 ± 0.0711     0.9198 ± 0.0048
90s        1.6963 ± 0.0592     2.3633 ± 0.1175     0.9211 ± 0.0074
150s       1.8191 ± 0.0698     2.4930 ± 0.1272     0.9050 ± 0.0092
--------------------------------------------------------------------------------

================================================================================
✅ CROSS-VALIDATION COMPLETE!
================================================================================"""