"""LSTM-based network for Big 2 with move history."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMNetwork(nn.Module):
    """
    LSTM network for Big 2 that processes move history.

    Architecture:
    1. LSTM processes move history sequence (16 moves × 52 dims)
    2. LSTM output concatenated with static state (hand, opponent info)
    3. Dense layers process combined features → Q-value
    """

    def __init__(
        self,
        state_dim: int = 195,
        action_dim: int = 52,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        history_length: int = 16,
        **kwargs  # Accept extra config args
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.history_length = history_length

        # LSTM for processing move history
        self.lstm = nn.LSTM(
            input_size=52,  # Each move is 52-dim one-hot
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )

        # Dense layers for Q-value prediction
        # Input: LSTM output + static state + action
        self.fc1 = nn.Linear(lstm_hidden + state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, move_history, state, action):
        """
        Forward pass.

        Args:
            move_history: (batch, history_length, 52) - sequence of moves
            state: (batch, state_dim) - static state (hand, opponent info)
            action: (batch, action_dim) - action encoding

        Returns:
            Q-value: (batch, 1)
        """
        # Process move history with LSTM
        lstm_out, (h_n, c_n) = self.lstm(move_history)
        # lstm_out: (batch, history_length, lstm_hidden)

        # Use final hidden state
        lstm_final = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        # Concatenate LSTM output with static state and action
        x = torch.cat([lstm_final, state, action], dim=1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def predict_q_values(self, move_histories, states, actions):
        """
        Utility method for batch prediction.

        Args:
            move_histories: (batch, history_length, 52)
            states: (batch, state_dim)
            actions: (batch, action_dim)

        Returns:
            Q-values: (batch,)
        """
        q_values = self.forward(move_histories, states, actions)
        return q_values.squeeze(-1)


if __name__ == "__main__":
    # Test
    batch_size = 32
    history_length = 16
    state_dim = 195
    action_dim = 52

    model = LSTMNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lstm_hidden=128,
        lstm_layers=2,
        history_length=history_length
    )

    # Create dummy inputs
    move_history = torch.randn(batch_size, history_length, 52)
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)

    # Forward pass
    q_values = model(move_history, state, action)

    print(f"Input shapes:")
    print(f"  move_history: {move_history.shape}")
    print(f"  state: {state.shape}")
    print(f"  action: {action.shape}")
    print(f"Output Q-values: {q_values.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
