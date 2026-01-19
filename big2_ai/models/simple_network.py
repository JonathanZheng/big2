"""Simple dense network for Big 2 Deep Monte Carlo."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleNetwork(nn.Module):
    """
    Simple dense network for Q-value estimation.

    Architecture:
        Input: state (143) + action (52) = 195 dims
        ↓
        Linear(195 → 256) + ReLU
        ↓
        Linear(256 → 256) + ReLU
        ↓
        Linear(256 → 256) + ReLU
        ↓
        Linear(256 → 1)  # Q-value
    """

    def __init__(self, state_dim: int = 143, action_dim: int = 52, hidden_dim: int = 256):
        """
        Initialize network.

        Args:
            state_dim: Dimension of state encoding
            action_dim: Dimension of action encoding
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim

        # Network layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, state_dim + action_dim)

        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def predict_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predict Q-values for state-action pairs.

        Args:
            states: State tensor of shape (batch_size, state_dim)
            actions: Action tensor of shape (batch_size, action_dim)

        Returns:
            Q-values of shape (batch_size,)
        """
        x = torch.cat([states, actions], dim=1)
        q_values = self.forward(x)
        return q_values.squeeze(-1)


def test_network():
    """Test network forward pass."""
    print("Testing SimpleNetwork...")

    # Create network
    net = SimpleNetwork()
    print(f"Network: {net}")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
    print()

    # Test forward pass
    batch_size = 32
    state_dim = 143
    action_dim = 52

    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)

    q_values = net.predict_q_values(states, actions)
    print(f"Input shapes: states={states.shape}, actions={actions.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Q-value range: [{q_values.min():.3f}, {q_values.max():.3f}]")
    print()

    # Test single prediction
    single_state = torch.randn(1, state_dim)
    single_action = torch.randn(1, action_dim)
    single_q = net.predict_q_values(single_state, single_action)
    print(f"Single prediction: {single_q.item():.3f}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_network()
