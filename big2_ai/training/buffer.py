"""Replay buffer for Deep Monte Carlo training."""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Transition:
    """A single Monte Carlo transition (Stage 2)."""
    state: np.ndarray  # State encoding (195,)
    action: np.ndarray  # Action encoding (52,)
    move_history: np.ndarray  # Move history sequence (16, 52)
    episode_return: float  # Final reward from this state


class ReplayBuffer:
    """
    Circular replay buffer for Monte Carlo transitions.

    Stores (state, action, episode_return) tuples and supports
    random sampling for training.
    """

    def __init__(self, capacity: int = 50000):
        """
        Initialize buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, state: np.ndarray, action: np.ndarray,
             move_history: np.ndarray, episode_return: float):
        """
        Add a transition to the buffer.

        Args:
            state: State encoding (195,)
            action: Action encoding (52,)
            move_history: Move history sequence (16, 52)
            episode_return: Episode return from this state
        """
        transition = Transition(state, action, move_history, episode_return)

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

    def sample_arrays(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch and return as numpy arrays.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, move_histories, returns) as numpy arrays
        """
        batch = self.sample(batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.float32)
        move_histories = np.array([t.move_history for t in batch], dtype=np.float32)
        returns = np.array([t.episode_return for t in batch], dtype=np.float32)

        return states, actions, move_histories, returns

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0


def test_buffer():
    """Test replay buffer (Stage 2)."""
    print("Testing ReplayBuffer (Stage 2)...")

    buffer = ReplayBuffer(capacity=100)
    print(f"Created buffer with capacity {buffer.capacity}")
    print(f"Initial size: {len(buffer)}")
    print()

    # Add some transitions
    for i in range(150):
        state = np.random.randn(195).astype(np.float32)
        action = np.random.randn(52).astype(np.float32)
        move_history = np.random.randn(16, 52).astype(np.float32)
        ret = np.random.randn()
        buffer.push(state, action, move_history, ret)

    print(f"After adding 150 transitions, size: {len(buffer)}")
    print(f"(Should be capped at capacity: {buffer.capacity})")
    print()

    # Sample a batch
    batch = buffer.sample(32)
    print(f"Sampled batch of size {len(batch)}")
    print(f"First transition: state shape={batch[0].state.shape}, "
          f"action shape={batch[0].action.shape}, "
          f"move_history shape={batch[0].move_history.shape}, "
          f"return={batch[0].episode_return:.3f}")
    print()

    # Sample as arrays
    states, actions, move_histories, returns = buffer.sample_arrays(32)
    print(f"Sampled as arrays:")
    print(f"  states: {states.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  move_histories: {move_histories.shape}")
    print(f"  returns: {returns.shape}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_buffer()
