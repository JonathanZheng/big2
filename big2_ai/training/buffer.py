"""Replay buffer for Deep Monte Carlo training."""

import random
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Transition:
    """A single Monte Carlo transition (Stage 2 with optional PTIE support)."""
    state: np.ndarray  # Observable state encoding (167,)
    action: np.ndarray  # Action encoding (52,)
    move_history: np.ndarray  # Move history sequence (16, 52)
    episode_return: float  # Final reward from this state
    perfect_state: Optional[np.ndarray] = None  # Perfect state for PTIE critic (321,)


class ReplayBuffer:
    """
    Circular replay buffer for Monte Carlo transitions.

    Stores (state, action, episode_return) tuples and supports
    random sampling for training.

    Optionally normalizes returns using running statistics (Welford's algorithm).
    """

    def __init__(self, capacity: int = 50000, normalize_returns: bool = True):
        """
        Initialize buffer.

        Args:
            capacity: Maximum number of transitions to store
            normalize_returns: Whether to normalize returns when sampling
        """
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0
        self.normalize = normalize_returns

        # Running statistics for return normalization (Welford's algorithm)
        self.return_count = 0
        self.return_mean = 0.0
        self.return_m2 = 0.0  # Sum of squared differences from mean

    @property
    def return_std(self) -> float:
        """Compute running standard deviation of returns."""
        if self.return_count < 2:
            return 1.0
        variance = self.return_m2 / (self.return_count - 1)
        return math.sqrt(variance) if variance > 0 else 1.0

    def push(self, state: np.ndarray, action: np.ndarray,
             move_history: np.ndarray, episode_return: float,
             perfect_state: Optional[np.ndarray] = None):
        """
        Add a transition to the buffer.

        Args:
            state: Observable state encoding (167,)
            action: Action encoding (52,)
            move_history: Move history sequence (16, 52)
            episode_return: Episode return from this state
            perfect_state: Optional perfect state for PTIE critic (321,)
        """
        # Update running statistics (Welford's algorithm)
        self.return_count += 1
        delta = episode_return - self.return_mean
        self.return_mean += delta / self.return_count
        delta2 = episode_return - self.return_mean
        self.return_m2 += delta * delta2

        # Add transition to buffer
        transition = Transition(state, action, move_history, episode_return, perfect_state)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of Transition objects
        """
        return random.sample(self.buffer, batch_size)

    def sample_arrays(self, batch_size: int, include_perfect: bool = False):
        """
        Sample a random batch and return as numpy arrays.

        If normalize_returns is True, returns are normalized using running
        mean and std computed from all transitions seen so far.

        Args:
            batch_size: Number of transitions to sample
            include_perfect: Whether to include perfect states (for PTIE)

        Returns:
            If include_perfect=False:
                Tuple of (states, actions, move_histories, returns)
            If include_perfect=True:
                Tuple of (states, actions, move_histories, returns, perfect_states)
        """
        batch = self.sample(batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.float32)
        move_histories = np.array([t.move_history for t in batch], dtype=np.float32)
        returns = np.array([t.episode_return for t in batch], dtype=np.float32)

        # Normalize returns if enabled
        if self.normalize:
            std = max(self.return_std, 1e-6)  # Prevent division by zero
            returns = (returns - self.return_mean) / std

        if include_perfect:
            # Check if perfect states are available
            if batch[0].perfect_state is not None:
                perfect_states = np.array([t.perfect_state for t in batch], dtype=np.float32)
            else:
                # Return None if perfect states not collected
                perfect_states = None
            return states, actions, move_histories, returns, perfect_states

        return states, actions, move_histories, returns

    def get_stats(self) -> dict:
        """Get current return statistics."""
        return {
            "count": self.return_count,
            "mean": self.return_mean,
            "std": self.return_std,
        }

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear the buffer and reset statistics."""
        self.buffer = []
        self.position = 0
        self.return_count = 0
        self.return_mean = 0.0
        self.return_m2 = 0.0


def test_buffer():
    """Test replay buffer (Stage 2)."""
    print("Testing ReplayBuffer (Stage 2)...")

    # Test with normalization enabled
    buffer = ReplayBuffer(capacity=100, normalize_returns=True)
    print(f"Created buffer with capacity {buffer.capacity}, normalize={buffer.normalize}")
    print(f"Initial size: {len(buffer)}")
    print()

    # Add some transitions with varied returns
    for i in range(150):
        state = np.random.randn(195).astype(np.float32)
        action = np.random.randn(52).astype(np.float32)
        move_history = np.random.randn(16, 52).astype(np.float32)
        # Simulate game returns (winners: ~0.9, losers: ~-0.9)
        ret = np.random.choice([0.9, -0.9]) + np.random.randn() * 0.1
        buffer.push(state, action, move_history, ret)

    print(f"After adding 150 transitions, size: {len(buffer)}")
    print(f"(Should be capped at capacity: {buffer.capacity})")
    print()

    # Print return statistics
    stats = buffer.get_stats()
    print(f"Return statistics:")
    print(f"  count: {stats['count']}")
    print(f"  mean:  {stats['mean']:.4f}")
    print(f"  std:   {stats['std']:.4f}")
    print()

    # Sample a batch
    batch = buffer.sample(32)
    print(f"Sampled batch of size {len(batch)}")
    print(f"First transition: state shape={batch[0].state.shape}, "
          f"action shape={batch[0].action.shape}, "
          f"move_history shape={batch[0].move_history.shape}, "
          f"return={batch[0].episode_return:.3f}")
    print()

    # Sample as arrays (with normalization)
    states, actions, move_histories, returns = buffer.sample_arrays(32)
    print(f"Sampled as arrays (normalized):")
    print(f"  states: {states.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  move_histories: {move_histories.shape}")
    print(f"  returns: {returns.shape}")
    print(f"  returns mean: {returns.mean():.4f} (should be near 0)")
    print(f"  returns std:  {returns.std():.4f} (should be near 1)")

    # Test without normalization
    buffer_no_norm = ReplayBuffer(capacity=100, normalize_returns=False)
    for i in range(100):
        buffer_no_norm.push(
            np.random.randn(195).astype(np.float32),
            np.random.randn(52).astype(np.float32),
            np.random.randn(16, 52).astype(np.float32),
            np.random.choice([0.9, -0.9])
        )

    _, _, _, raw_returns = buffer_no_norm.sample_arrays(32)
    print(f"\nWithout normalization:")
    print(f"  returns mean: {raw_returns.mean():.4f}")
    print(f"  returns std:  {raw_returns.std():.4f}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_buffer()
