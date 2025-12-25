import torch
import numpy as np
import pickle
from collections import deque
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_attention_lstm import MultiHorizonAttentionLSTM

class TrafficPredictor:
    """
    Real-time traffic prediction using trained Multi-Horizon LSTM
    """
    def __init__(self, model_path='multihorizon_lstm.pth', 
                 scaler_path='multihorizon_scaler.pkl',
                 seq_len=30):
        
        self.seq_len = seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.features = ['speed', 'accel', 'front_vehicle_dist', 'front_vehicle_speed', 
                        'lane_density', 'avg_lane_speed']
        self.input_size = len(self.features)
        self.hidden_size = 128
        self.num_layers = 2
        
        # Load scaler
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = MultiHorizonAttentionLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Rolling window for each vehicle
        self.vehicle_buffers = {}
        
        print(f"[OK] TrafficPredictor initialized on {self.device}")
    
    def update_vehicle_data(self, vehicle_id, speed, accel, front_vehicle_dist, 
                           front_vehicle_speed, lane_density, avg_lane_speed):
        """
        Update rolling window for a vehicle with new observation
        """
        # Create feature vector
        features = np.array([speed, accel, front_vehicle_dist, front_vehicle_speed, 
                            lane_density, avg_lane_speed])
        
        # Initialize buffer if new vehicle
        if vehicle_id not in self.vehicle_buffers:
            self.vehicle_buffers[vehicle_id] = deque(maxlen=self.seq_len)
        
        # Add to buffer
        self.vehicle_buffers[vehicle_id].append(features)
    
    def predict(self, vehicle_id):
        """
        Predict multi-horizon delta_speed for a vehicle
        Returns: (pred_30s, pred_90s, pred_150s, attention_weights) or None if insufficient data
        """
        # Check if we have enough data
        if vehicle_id not in self.vehicle_buffers:
            return None
        
        if len(self.vehicle_buffers[vehicle_id]) < self.seq_len:
            return None
        
        # Prepare sequence
        sequence = np.array(self.vehicle_buffers[vehicle_id])  # (seq_len, features)
        
        # Scale
        scaled_sequence = self.scaler.transform(sequence)
        
        # Convert to tensor
        x = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(self.device)  # (1, seq_len, features)
        
        # Predict
        with torch.no_grad():
            pred1, pred3, pred5, attention = self.model(x)
        
        # Convert to numpy
        pred_30s = pred1.cpu().numpy()[0][0]
        pred_90s = pred3.cpu().numpy()[0][0]
        pred_150s = pred5.cpu().numpy()[0][0]
        attention_weights = attention.cpu().numpy()[0]
        
        return pred_30s, pred_90s, pred_150s, attention_weights
    
    def get_predicted_speed(self, vehicle_id):
        """
        Get predicted future speeds (absolute values, not delta)
        """
        result = self.predict(vehicle_id)
        if result is None:
            return None
        
        pred_30s, pred_90s, pred_150s, attention = result
        
        # Get current speed
        current_speed = self.vehicle_buffers[vehicle_id][-1][0]
        
        # Calculate future speeds
        future_speed_30s = current_speed + pred_30s
        future_speed_90s = current_speed + pred_90s
        future_speed_150s = current_speed + pred_150s
        
        return {
            'current_speed': current_speed,
            'predicted_30s': future_speed_30s,
            'predicted_90s': future_speed_90s,
            'predicted_150s': future_speed_150s,
            'delta_30s': pred_30s,
            'delta_90s': pred_90s,
            'delta_150s': pred_150s,
            'attention_weights': attention
        }
    
    def clear_vehicle(self, vehicle_id):
        """Remove vehicle from tracking (when it leaves simulation)"""
        if vehicle_id in self.vehicle_buffers:
            del self.vehicle_buffers[vehicle_id]


# Test function
if __name__ == "__main__":
    print("Testing TrafficPredictor...")
    
    predictor = TrafficPredictor()
    
    # Simulate some data updates
    test_vehicle = "test_001"
    
    # Add 30 timesteps of dummy data
    for i in range(35):
        predictor.update_vehicle_data(
            vehicle_id=test_vehicle,
            speed=10 + i * 0.5,
            accel=0.2,
            front_vehicle_dist=50,
            front_vehicle_speed=12,
            lane_density=0.3,
            avg_lane_speed=11
        )
        
        if i >= 30:  # After 30 steps, we can predict
            result = predictor.get_predicted_speed(test_vehicle)
            if result:
                print(f"\nTimestep {i}:")
                print(f"  Current Speed: {result['current_speed']:.2f}")
                print(f"  Predicted @ 30s: {result['predicted_30s']:.2f} (Δ={result['delta_30s']:.3f})")
                print(f"  Predicted @ 90s: {result['predicted_90s']:.2f} (Δ={result['delta_90s']:.3f})")
                print(f"  Predicted @ 150s: {result['predicted_150s']:.2f} (Δ={result['delta_150s']:.3f})")
    
    print("\n✓ Test complete!")