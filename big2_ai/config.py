"""Configuration for Big 2 AI training."""

import math

# Stage 2 - MLP with enhanced features (hand sizes + high cards)
TRAINING_CONFIG = {
    # Training episodes
    "num_episodes": 50000,

    # Replay buffer
    "buffer_size": 100000,  # Increased for better sample diversity
    "batch_size": 512,

    # Learning
    "learning_rate": 5e-5,  # Slightly lower for stability
    "grad_clip": 5.0,

    # Exploration - Flexible epsilon schedule
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,  # Maintain some exploration
    "epsilon_schedule": "cosine",  # "cosine" or "exponential"
    "epsilon_warmup_episodes": 1000,  # Fill buffer before learning
    "epsilon_decay": 0.9997,  # Only used if epsilon_schedule == "exponential"

    # Target network
    "target_update_freq": 100,  # Episodes between target network updates
    "tau": 0.005,  # Slower target updates for stability

    # Logging and evaluation
    "eval_freq": 100,  # Episodes between evaluations
    "eval_games": 100,  # Increased from 50 for tighter confidence intervals
    "save_freq": 1000,  # Episodes between checkpoint saves
    "log_freq": 100,  # Episodes between logging

    # Reward system
    "use_margin_rewards": True,  # Scale rewards with winning margin

    # Return normalization
    "normalize_returns": True,  # Normalize returns in replay buffer

    # Curriculum learning
    "use_curriculum": True,
    "checkpoint_opponent_path": None,  # Path to frozen opponent model (e.g., "checkpoints/22k.pt")
    "curriculum_phases": [
        {"progress": 0.0, "self_play": 0.6, "greedy": 0.2, "checkpoint": 0.2},
        {"progress": 0.33, "self_play": 0.7, "greedy": 0.15, "checkpoint": 0.15},
        {"progress": 0.66, "self_play": 0.8, "greedy": 0.1, "checkpoint": 0.1},
    ],

    # Legacy opponent diversity ratios (used when use_curriculum=False)
    "self_play_ratio": 0.7,       # 70% pure self-play (all 4 players use model)
    "greedy_opponent_ratio": 0.2,  # 20% vs 3 greedy opponents (consistent challenge)
    "random_opponent_ratio": 0.1,  # 10% vs 3 random opponents (diversity)

    # Device
    "device": "cpu",  # Will be set to "mps" if available on M3 Mac

    # Checkpointing
    "checkpoint_dir": "checkpoints",
    "save_best": True,
}


def compute_epsilon(episode: int, total_episodes: int,
                    epsilon_start: float = 0.9,
                    epsilon_end: float = 0.05,
                    warmup_episodes: int = 1000,
                    schedule: str = "cosine") -> float:
    """
    Compute epsilon using flexible schedule - adapts to any episode count.

    Args:
        episode: Current episode number
        total_episodes: Total number of episodes to train
        epsilon_start: Starting exploration rate
        epsilon_end: Ending exploration rate
        warmup_episodes: Episodes before decay starts
        schedule: "cosine" or "exponential"

    Returns:
        Current epsilon value
    """
    if episode < warmup_episodes:
        return epsilon_start

    progress = (episode - warmup_episodes) / max(1, total_episodes - warmup_episodes)
    progress = min(1.0, progress)

    if schedule == "cosine":
        # Cosine annealing: slow start, fast middle, slow end
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return epsilon_end + (epsilon_start - epsilon_end) * cosine_decay
    else:
        # Linear fallback
        return epsilon_end + (epsilon_start - epsilon_end) * (1 - progress)

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


def get_opponent_mix(episode: int, total_episodes: int, config: dict = None) -> dict:
    """
    Get opponent mix based on training progress (curriculum learning).

    Args:
        episode: Current episode number
        total_episodes: Total number of episodes
        config: Training config (uses TRAINING_CONFIG if None)

    Returns:
        dict with keys: "self_play", "greedy", "checkpoint"
    """
    if config is None:
        config = TRAINING_CONFIG

    if not config.get("use_curriculum", False):
        # Use legacy fixed ratios
        return {
            "self_play": config.get("self_play_ratio", 0.7),
            "greedy": config.get("greedy_opponent_ratio", 0.2),
            "checkpoint": config.get("random_opponent_ratio", 0.1),  # Maps to random if no checkpoint
        }

    phases = config.get("curriculum_phases", [
        {"progress": 0.0, "self_play": 0.6, "greedy": 0.2, "checkpoint": 0.2},
        {"progress": 0.33, "self_play": 0.7, "greedy": 0.15, "checkpoint": 0.15},
        {"progress": 0.66, "self_play": 0.8, "greedy": 0.1, "checkpoint": 0.1},
    ])

    progress = episode / max(1, total_episodes)

    # Find the appropriate phase
    current_phase = phases[0]
    for phase in phases:
        if progress >= phase["progress"]:
            current_phase = phase

    return {
        "self_play": current_phase["self_play"],
        "greedy": current_phase["greedy"],
        "checkpoint": current_phase["checkpoint"],
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
            if isinstance(value, list):
                print(f"  {key:25s}:")
                for item in value:
                    print(f"    {item}")
            else:
                print(f"  {key:25s}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    print_config()
