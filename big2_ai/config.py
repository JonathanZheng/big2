"""Configuration for Big 2 AI training."""

import math

# Stage 2 - MLP with enhanced features (hand sizes + high cards)
TRAINING_CONFIG = {
    # Training episodes
    "num_episodes": 50000,

    # Replay buffer
    "buffer_size": 100000,  # Increased for better sample diversity
    "batch_size": 256,

    # Learning
    "learning_rate": 1e-4,
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
    "eval_freq": 1000,  # Episodes between evaluations
    "eval_games": 200,  # Games per evaluation (~±6.4% CI at 30% WR)
    "save_freq": 1000,  # Episodes between checkpoint saves
    "log_freq": 100,  # Episodes between logging

    # Reward system
    "use_margin_rewards": True,  # Scale rewards with winning margin

    # Return normalization
    "normalize_returns": True,  # Normalize returns in replay buffer

    # Curriculum learning
    # NOTE: Old checkpoints are incompatible with new 167-dim state encoding
    # Set checkpoint_opponent_path to None until you have a v2 checkpoint
    "use_curriculum": True,
    "checkpoint_opponent_path": None,  # Path to frozen opponent model (must use new 167-dim encoding)
    # Fixed ratio for league training: 60% self-play, 20% greedy, 20% league
    "curriculum_phases": [
        {"progress": 0.0, "self_play": 0.6, "greedy": 0.2, "checkpoint": 0.2},
    ],

    # Opponent diversity ratios (used when use_curriculum=False)
    "self_play_ratio": 0.6,        # 60% pure self-play (all 4 players use model)
    "greedy_opponent_ratio": 0.2,  # 20% vs 3 greedy opponents
    "rule_based_opponent_ratio": 0.2,  # 20% vs 3 rule-based opponents (upgraded bot)
    "random_opponent_ratio": 0.0,  # 0% random (not used)

    # Device
    "device": "cpu",  # Will be set to "mps" if available on M3 Mac

    # Checkpointing
    "checkpoint_dir": "checkpoints",
    "save_best": True,
    "top_k_checkpoints": 5,  # Number of top checkpoints to keep (ranked by greedy win rate)

    # PTIE (Perfect Information Training for Imperfect-Information Games)
    "use_ptie": True,  # Enable PTIE with critic network
    "critic_learning_rate": 1e-4,  # Separate LR for critic (can be higher)
    "critic_weight": 0.5,  # Weight for critic loss in total loss
    "advantage_weight": 1.0,  # Weight for advantage-based actor loss

    # League training (opponent pool with historical checkpoints)
    "use_league": False,  # Enable league training (replaces single checkpoint opponent)
    "league_dir": "checkpoints/league",  # Directory for league checkpoints
    "league_max_opponents": 20,  # Max opponents to keep in pool
    "league_snapshot_freq": 2000,  # Episodes between snapshots
    "league_initial_snapshot": True,  # Take snapshot at start of training
}


def compute_epsilon(episode: int, total_episodes: int,
                    epsilon_start: float = 0.9,
                    epsilon_end: float = 0.01,
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

    # NEW: Decay Duration Control
    # Finish decaying at 90% of episodes, hold constant for the last 10%
    decay_cutoff = 0.95
    decay_episodes = int(total_episodes * decay_cutoff)
    
    # Calculate progress relative to the CUTOFF, not the total
    progress = (episode - warmup_episodes) / max(1, decay_episodes - warmup_episodes)
    progress = min(1.0, progress) # Cap at 1.0 to hold steady at the end

    if schedule == "cosine":
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return epsilon_end + (epsilon_start - epsilon_end) * cosine_decay
    else:
        return epsilon_end + (epsilon_start - epsilon_end) * (1 - progress)

# Network configuration (Stage 2 - Legitimate state encoding)
NETWORK_CONFIG = {
    "state_dim": 167,  # Changed from 149 (removed cheating, added graveyard + control)
    "action_dim": 52,
    "hidden_dim": 256,
    "num_layers": 4,  # 4-layer MLP
}

# Critic network configuration (for PTIE)
CRITIC_CONFIG = {
    "perfect_state_dim": 321,  # All hands + game state
    "hidden_dim": 256,
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
        # Use fixed ratios (not using curriculum)
        # Note: When using fixed ratios, we use rule_based_opponent_ratio instead of checkpoint
        return {
            "self_play": config.get("self_play_ratio", 0.6),
            "greedy": config.get("greedy_opponent_ratio", 0.2),
            "checkpoint": config.get("random_opponent_ratio", 0.0),  # Not used in fixed ratio mode
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
        "critic": CRITIC_CONFIG,
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
