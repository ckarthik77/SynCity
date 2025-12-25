import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHorizonAttentionLSTM(nn.Module):
    """
    Multi-Horizon Prediction: Predict delta_speed at t+1, t+3, t+5
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(MultiHorizonAttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Shared LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Shared Attention
        self.attention = nn.Linear(hidden_size, 1)
        
        # Separate prediction heads for each horizon
        self.head_1step = self._create_prediction_head(hidden_size, dropout)
        self.head_3step = self._create_prediction_head(hidden_size, dropout)
        self.head_5step = self._create_prediction_head(hidden_size, dropout)
        
    def _create_prediction_head(self, hidden_size, dropout):
        """Create prediction head for one horizon"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        Returns:
            pred_1step, pred_3step, pred_5step: Predictions for each horizon
            attention_weights: Attention scores
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_scores = self.attention(lstm_out)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Context vector
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Multi-horizon predictions
        pred_1step = self.head_1step(context)
        pred_3step = self.head_3step(context)
        pred_5step = self.head_5step(context)
        
        return pred_1step, pred_3step, pred_5step, attention_weights.squeeze(-1)