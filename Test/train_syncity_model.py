#!/usr/bin/env python3
"""
train_syncity_model.py
Train Multi-Horizon Attention-LSTM from scratch with proper normalization
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
from tqdm import tqdm
import sys
import os

# Add Model folder to path
sys.path.append(r"E:\Desktop\Jobs\projects\Syncity\Model")
from model_attention_lstm import MultiHorizonAttentionLSTM

# ================= CONFIG =================
DATA_FILE = "syncity_clean_dataset.csv"
SEQUENCE_LENGTH = 30  # 15 seconds of history at 0.5s timesteps
HORIZONS = {
    '30s': 60,    # 60 steps ahead (30s at 0.5s timesteps)
    '90s': 180,   # 180 steps ahead (90s)
    '150s': 300   # 300 steps ahead (150s)
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

# Output paths
MODEL_SAVE_PATH = "Model/multihorizon_lstm_new.pth"
SCALER_SAVE_PATH = "Model/multihorizon_scaler_new.pkl"
METRICS_SAVE_PATH = "Model/training_metrics.json"

print("="*80)
print("SynCity Multi-Horizon Attention-LSTM Training")
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
    print("\n📊 Creating sequences from data...")
    
    sequences = []
    targets_30s = []
    targets_90s = []
    targets_150s = []
    
    vehicle_groups = df.groupby('veh_id')
    
    for veh_id, veh_data in tqdm(vehicle_groups, desc="Processing vehicles"):
        veh_data = veh_data.sort_values('time').reset_index(drop=True)
        
        # Need enough data for 150s predictions
        if len(veh_data) < seq_length + HORIZONS['150s']:
            continue
        
        # Features: speed, accel, front_dist, front_speed, lane_density, avg_lane_speed
        features = veh_data[['speed', 'accel', 'front_vehicle_dist', 
                             'front_vehicle_speed', 'lane_density', 'avg_lane_speed']].values
        
        speeds = veh_data['speed'].values
        
        # Create sliding windows
        for i in range(len(features) - seq_length - HORIZONS['150s']):
            seq = features[i:i+seq_length]
            
            current_speed = speeds[i + seq_length - 1]
            
            # Target DELTAS (change from current speed)
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

# Create sequences
X, y_30s, y_90s, y_150s = create_sequences(df, SEQUENCE_LENGTH)
print(f"\n✅ Created {len(X)} sequences")
print(f"   Sequence shape: {X.shape}")
print(f"   Features: speed, accel, front_dist, front_speed, lane_density, avg_lane_speed")

# ================= NORMALIZATION =================
print("\n🔧 Normalizing data...")

# Normalize INPUT features (Z-score)
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaler = StandardScaler()
X_normalized_reshaped = X_scaler.fit_transform(X_reshaped)
X_normalized = X_normalized_reshaped.reshape(X.shape)

print(f"   Feature means: {X_scaler.mean_}")
print(f"   Feature stds:  {X_scaler.scale_}")

# Normalize TARGET deltas (CRITICAL!)
y_scaler_30s = StandardScaler()
y_scaler_90s = StandardScaler()
y_scaler_150s = StandardScaler()

y_30s_norm = y_scaler_30s.fit_transform(y_30s.reshape(-1, 1)).flatten()
y_90s_norm = y_scaler_90s.fit_transform(y_90s.reshape(-1, 1)).flatten()
y_150s_norm = y_scaler_150s.fit_transform(y_150s.reshape(-1, 1)).flatten()

print(f"\n   Target delta statistics (before normalization):")
print(f"   30s:  mean={y_30s.mean():.4f}, std={y_30s.std():.4f}")
print(f"   90s:  mean={y_90s.mean():.4f}, std={y_90s.std():.4f}")
print(f"   150s: mean={y_150s.mean():.4f}, std={y_150s.std():.4f}")

# Create Model directory if it doesn't exist
os.makedirs("Model", exist_ok=True)

# Save scalers
scaler_dict = {
    'X_scaler': X_scaler,
    'y_scaler_30s': y_scaler_30s,
    'y_scaler_90s': y_scaler_90s,
    'y_scaler_150s': y_scaler_150s
}

with open(SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler_dict, f)
print(f"\n✅ Scalers saved to {SCALER_SAVE_PATH}")

# ================= TRAIN/VAL/TEST SPLIT =================
print("\n📊 Splitting data (70% train, 15% val, 15% test)...")

# First split: train + val vs test
X_temp, X_test, y30_temp, y30_test, y90_temp, y90_test, y150_temp, y150_test = train_test_split(
    X_normalized, y_30s_norm, y_90s_norm, y_150s_norm,
    test_size=0.15, random_state=42
)

# Second split: train vs val
X_train, X_val, y30_train, y30_val, y90_train, y90_val, y150_train, y150_val = train_test_split(
    X_temp, y30_temp, y90_temp, y150_temp,
    test_size=0.176, random_state=42  # 0.176 of 85% ≈ 15% of total
)

print(f"   Train: {len(X_train)} sequences")
print(f"   Val:   {len(X_val)} sequences")
print(f"   Test:  {len(X_test)} sequences")

# Create datasets
train_dataset = VehicleSequenceDataset(X_train, y30_train, y90_train, y150_train)
val_dataset = VehicleSequenceDataset(X_val, y30_val, y90_val, y150_val)
test_dataset = VehicleSequenceDataset(X_test, y30_test, y90_test, y150_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================= MODEL SETUP =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Using device: {device}")

model = MultiHorizonAttentionLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

print(f"\n✅ Model initialized:")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================= TRAINING LOOP =================
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)

best_val_loss = float('inf')
patience_counter = 0
training_history = {
    'train_loss': [],
    'val_loss': [],
    'val_mae_30s': [],
    'val_mae_90s': [],
    'val_mae_150s': []
}

for epoch in range(EPOCHS):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    
    for X_batch, y30_batch, y90_batch, y150_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        X_batch = X_batch.to(device)
        y30_batch = y30_batch.to(device)
        y90_batch = y90_batch.to(device)
        y150_batch = y150_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_30s, pred_90s, pred_150s, _ = model(X_batch)
        
        # Multi-task loss (weighted)
        loss = (
            criterion(pred_30s, y30_batch) +
            criterion(pred_90s, y90_batch) +
            0.8 * criterion(pred_150s, y150_batch)
        )
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0.0
    val_preds_30s = []
    val_preds_90s = []
    val_preds_150s = []
    val_true_30s = []
    val_true_90s = []
    val_true_150s = []
    
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
            
            val_preds_30s.append(pred_30s.cpu().numpy())
            val_preds_90s.append(pred_90s.cpu().numpy())
            val_preds_150s.append(pred_150s.cpu().numpy())
            val_true_30s.append(y30_batch.cpu().numpy())
            val_true_90s.append(y90_batch.cpu().numpy())
            val_true_150s.append(y150_batch.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Calculate MAE (on normalized scale)
    val_preds_30s = np.concatenate(val_preds_30s)
    val_preds_90s = np.concatenate(val_preds_90s)
    val_preds_150s = np.concatenate(val_preds_150s)
    val_true_30s = np.concatenate(val_true_30s)
    val_true_90s = np.concatenate(val_true_90s)
    val_true_150s = np.concatenate(val_true_150s)
    
    mae_30s = np.mean(np.abs(val_preds_30s - val_true_30s))
    mae_90s = np.mean(np.abs(val_preds_90s - val_true_90s))
    mae_150s = np.mean(np.abs(val_preds_150s - val_true_150s))
    
    # Save history
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    training_history['val_mae_30s'].append(mae_30s)
    training_history['val_mae_90s'].append(mae_90s)
    training_history['val_mae_150s'].append(mae_150s)
    
    print(f"\n  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    print(f"  Val MAE - 30s: {mae_30s:.4f} | 90s: {mae_90s:.4f} | 150s: {mae_150s:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✅ New best model saved!")
    else:
        patience_counter += 1
        print(f"  ⏳ Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
        break

# ================= SAVE METRICS =================
with open(METRICS_SAVE_PATH, 'w') as f:
    json.dump(training_history, f, indent=2)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"  Best model: {MODEL_SAVE_PATH}")
print(f"  Scalers: {SCALER_SAVE_PATH}")
print(f"  Metrics: {METRICS_SAVE_PATH}")
print("="*80)