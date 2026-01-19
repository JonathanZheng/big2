"""Configuration for Big 2 AI training."""

# Stage 1 - M3 Air optimized configuration
TRAINING_CONFIG = {
    # Training episodes
    "num_episodes": 50000,

    # Replay buffer
    "buffer_size": 10000,
    "batch_size": 256,

    # Learning
    "learning_rate": 1e-4,
    "grad_clip": 5.0,

    # Exploration
    "epsilon_start": 0.9,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.9999,  # Per episode

    # Target network
    "target_update_freq": 100,  # Episodes between target network updates
    "tau": 0.01,  # Soft update parameter

    # Logging and evaluation
    "eval_freq": 100,  # Episodes between evaluations
    "eval_games": 50,  # Number of games for evaluation
    "save_freq": 1000,  # Episodes between checkpoint saves
    "log_freq": 100,  # Episodes between logging

    # Device
    "device": "cpu",  # Will be set to "mps" if available on M3 Mac

    # Checkpointing
    "checkpoint_dir": "checkpoints",
    "save_best": True,
}

# Network configuration
NETWORK_CONFIG = {
    "state_dim": 143,
    "action_dim": 52,
    "hidden_dim": 256,
}

# Environment configuration
ENV_CONFIG = {
    "num_players": 4,
    "num_cards": 52,
    "hand_size": 13,
}


def get_config():
    """Get full configuration dictionary."""
    return {
        "training": TRAINING_CONFIG,
        "network": NETWORK_CONFIG,
        "env": ENV_CONFIG,
    }


def print_config():
    """Print configuration."""
    config = get_config()

    print("=" * 60)
    print("Big 2 AI Configuration")
    print("=" * 60)

    for section, values in config.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  {key:25s}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    print_config()
