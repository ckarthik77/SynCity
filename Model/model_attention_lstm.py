import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHorizonAttentionLSTM(nn.Module):
    """
    Multi-Horizon Attention LSTM model for time series prediction.
    Predicts 1-step, 3-step, and 5-step ahead deltas.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(MultiHorizonAttentionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention Mechanism
        # This layer will output attention scores for each timestep in the sequence
        self.attention = nn.Linear(hidden_size, 1)

        # Output layers for each horizon
        # These layers will take the context vector (from attention) and produce predictions
        self.fc_1step = nn.Linear(hidden_size, 1)
        self.fc_3step = nn.Linear(hidden_size, 1)
        self.fc_5step = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            pred_1step: Predictions for 1-step ahead (batch_size, 1)
            pred_3step: Predictions for 3-step ahead (batch_size, 1)
            pred_5step: Predictions for 5-step ahead (batch_size, 1)
            attention_weights: Attention scores (batch_size, seq_len)
        """
        # LSTM encoding
        # lstm_out: (batch_size, seq_len, hidden_size)
        # hidden, cell: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention mechanism
        # attention_scores: (batch_size, seq_len, 1)
        attention_scores = self.attention(lstm_out)
        # attention_weights: (batch_size, seq_len, 1) - softmax over sequence length
        attention_weights = F.softmax(attention_scores, dim=1)

        # Create context vector by weighted sum of lstm_out
        # context: (batch_size, hidden_size)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Apply dropout to the context vector before feeding to prediction heads
        context = self.dropout(context)

        # Predict for each horizon
        pred_1step = self.fc_1step(context)
        pred_3step = self.fc_3step(context)
        pred_5step = self.fc_5step(context)

        return pred_1step, pred_3step, pred_5step, attention_weights.squeeze(-1)
