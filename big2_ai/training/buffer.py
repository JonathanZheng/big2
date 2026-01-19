"""Replay buffer for Deep Monte Carlo training."""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Transition:
    """A single Monte Carlo transition."""
    state: np.ndarray  # State encoding (143,)
    action: np.ndarray  # Action encoding (52,)
    episode_return: float  # Final reward from this state


class ReplayBuffer:
    """
    Circular replay buffer for Monte Carlo transitions.

    Stores (state, action, episode_return) tuples and supports
    random sampling for training.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, state: np.ndarray, action: np.ndarray, episode_return: float):
        """
        Add a transition to the buffer.

        Args:
            state: State encoding
            action: Action encoding
            episode_return: Episode return from this state
        """
        transition = Transition(state, action, episode_return)

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

    def sample_arrays(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch and return as numpy arrays.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, returns) as numpy arrays
        """
        batch = self.sample(batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.float32)
        returns = np.array([t.episode_return for t in batch], dtype=np.float32)

        return states, actions, returns

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0


def test_buffer():
    """Test replay buffer."""
    print("Testing ReplayBuffer...")

    buffer = ReplayBuffer(capacity=100)
    print(f"Created buffer with capacity {buffer.capacity}")
    print(f"Initial size: {len(buffer)}")
    print()

    # Add some transitions
    for i in range(150):
        state = np.random.randn(143).astype(np.float32)
        action = np.random.randn(52).astype(np.float32)
        ret = np.random.randn()
        buffer.push(state, action, ret)

    print(f"After adding 150 transitions, size: {len(buffer)}")
    print(f"(Should be capped at capacity: {buffer.capacity})")
    print()

    # Sample a batch
    batch = buffer.sample(32)
    print(f"Sampled batch of size {len(batch)}")
    print(f"First transition: state shape={batch[0].state.shape}, "
          f"action shape={batch[0].action.shape}, return={batch[0].episode_return:.3f}")
    print()

    # Sample as arrays
    states, actions, returns = buffer.sample_arrays(32)
    print(f"Sampled as arrays:")
    print(f"  states: {states.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  returns: {returns.shape}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_buffer()
