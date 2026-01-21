"""Configuration for Big 2 AI training."""

# Stage 2 - MLP with enhanced features (hand sizes + high cards)
TRAINING_CONFIG = {
    # Training episodes
    "num_episodes": 50000,

    # Replay buffer
    "buffer_size": 50000,  # Increased from 10K for better sample diversity
    "batch_size": 512,

    # Learning
    "learning_rate": 1e-4,  # Increased from 5e-5
    "grad_clip": 5.0,

    # Exploration - CRITICAL FIX: faster decay (reaches ~0.1 at 20K with 6 workers)
    "epsilon_start": 0.9,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.9997,  # Was 0.9999 - decayed too slowly
    "adaptive_epsilon": False,  # Adjust epsilon based on performance
    "epsilon_adapt_window": 500,  # Check win rate every N episodes
    "epsilon_adapt_threshold": 0.05,  # Increase epsilon if win rate drops >5%

    # Target network
    "target_update_freq": 100,  # Episodes between target network updates
    "tau": 0.01,  # Soft update parameter

    # Logging and evaluation
    "eval_freq": 100,  # Episodes between evaluations
    "eval_games": 100,  # Increased from 50 for tighter confidence intervals
    "save_freq": 1000,  # Episodes between checkpoint saves
    "log_freq": 100,  # Episodes between logging

    # Opponent diversity ratios for training
    "self_play_ratio": 0.7,       # 70% pure self-play (all 4 players use model)
    "greedy_opponent_ratio": 0.2,  # 20% vs 3 greedy opponents (consistent challenge)
    "random_opponent_ratio": 0.1,  # 10% vs 3 random opponents (diversity)

    # Device
    "device": "cpu",  # Will be set to "mps" if available on M3 Mac

    # Checkpointing
    "checkpoint_dir": "checkpoints",
    "save_best": True,
}

# Network configuration (Stage 2 - MLP with enhanced features)
NETWORK_CONFIG = {
    "state_dim": 149,  # Changed from 143 (added hand_sizes + high_cards)
    "action_dim": 52,
    "hidden_dim": 256,
    "num_layers": 4,  # Increase from 3 to 4 layers (deeper MLP)
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
