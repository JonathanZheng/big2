"""Critic network for PTIE (Perfect Information Training)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """
    Critic network for PTIE that sees perfect information.

    This network is used ONLY during training to provide stable value estimates.
    It sees all 4 players' hands, which allows it to give accurate value
    predictions that help reduce variance in training the actor.

    At test time, this network is NOT used - only the actor (which sees
    observable information) is used for action selection.

    Architecture:
        Input: perfect_state (321) = all hands + game state
        ↓
        Linear(321 → 256) + ReLU
        ↓
        Linear(256 → 256) + ReLU
        ↓
        Linear(256 → 256) + ReLU
        ↓
        Linear(256 → 1)  # Value estimate V(s)
    """

    def __init__(
        self,
        perfect_state_dim: int = 321,
        hidden_dim: int = 256,
        **kwargs
    ):
        """
        Initialize critic network.

        Args:
            perfect_state_dim: Dimension of perfect state encoding (default: 321)
            hidden_dim: Hidden layer dimension (default: 256)
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        super().__init__()

        self.perfect_state_dim = perfect_state_dim
        self.hidden_dim = hidden_dim

        # Network layers (3 hidden layers for value estimation)
        self.fc1 = nn.Linear(perfect_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, perfect_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - estimate value from perfect state.

        Args:
            perfect_state: Perfect state tensor of shape (batch_size, perfect_state_dim)

        Returns:
            Value tensor of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(perfect_state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc_value(x)
        return value

    def predict_value(self, perfect_states: torch.Tensor) -> torch.Tensor:
        """
        Predict values for a batch of perfect states.

        Args:
            perfect_states: Perfect state tensor of shape (batch_size, perfect_state_dim)

        Returns:
            Values of shape (batch_size,)
        """
        values = self.forward(perfect_states)
        return values.squeeze(-1)


def test_critic_network():
    """Test critic network forward pass."""
    print("Testing CriticNetwork (PTIE)...")
    print("=" * 60)

    # Create network
    net = CriticNetwork()
    print(f"Network: {net}")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
    print()

    # Test forward pass
    batch_size = 32
    perfect_state_dim = 321

    perfect_states = torch.randn(batch_size, perfect_state_dim)

    values = net.predict_value(perfect_states)
    print(f"Input shape: perfect_states={perfect_states.shape}")
    print(f"Output shape: {values.shape}")
    print(f"Value range: [{values.min():.3f}, {values.max():.3f}]")
    print()

    # Test single prediction
    single_state = torch.randn(1, perfect_state_dim)
    single_value = net.predict_value(single_state)
    print(f"Single prediction: {single_value.item():.3f}")

    print()
    print("=" * 60)
    print("CriticNetwork test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_critic_network()
